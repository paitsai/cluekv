"""ClueKVCompressor: full ClueKV pipeline.

Prefill (capture Q_win) → CPS → head alignment → KV compression.
Returns compressed past_key_values ready for model.generate().

Key trick: after external prefill, ``past_key_values._seen_tokens == L`` (the
full prompt length).  When we compress the cache tensors in-place, ``_seen_tokens``
is NOT updated.  So ``model.generate(input_ids, past_key_values=compressed_kv)``
sees ``past_length == input_ids.shape[1]``, trims input_ids to empty, and starts
generating immediately — no re-processing, no manual loop.
"""

from __future__ import annotations

import os

import torch

from .params import ClueKVParams
from . import utils
from .attention import patch_qwen3_attention


def _dbg(debug: bool, *args, **kw):
    if debug:
        print("[ClueKV]", *args, **kw)


class ClueKVCompressor:
    """Orchestrates ClueKV pipeline."""

    def __init__(self, model, params: ClueKVParams | None = None,
                 stop_token_ids: set[int] | None = None, debug: bool = False):
        self.model = model
        self.params = params or ClueKVParams()
        self._empty_logits: torch.Tensor | None = None
        self.stop_token_ids = stop_token_ids
        self.debug = debug

    @torch.no_grad()
    def compute_or_load_empty_logits(
        self, tokenizer, empty_template: str = "",
        device: torch.device | str = "cuda", save_path: str | None = None,
    ) -> torch.Tensor:
        if save_path is not None:
            try:
                self._empty_logits = torch.load(save_path, map_location=device, weights_only=True)
                return self._empty_logits
            except FileNotFoundError:
                pass
        if empty_template:
            template = empty_template
        elif tokenizer.chat_template is not None:
            template = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True)
        else:
            template = tokenizer.bos_token or ""
        empty_inputs = tokenizer(template, return_tensors="pt").to(device)
        empty_out = self.model(**empty_inputs)
        empty_logits = empty_out.logits[0, -1, :].float().cpu()
        self._empty_logits = empty_logits
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(empty_logits, save_path)
        return empty_logits

    # ------------------------------------------------------------------
    # Full pipeline: prefill + compress → generate()-ready cache
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_prefill_and_compress(
        self, input_ids: torch.Tensor, tokenizer=None,
        empty_logits: torch.Tensor | None = None, past_key_values=None,
        stop_words: set[str] | None = None,
    ) -> tuple[torch.Tensor, object]:
        """Run prefill with Q_win capture + compress KV cache in-place.

        Returns (logits, past_key_values).  The returned past_key_values has
        ``_seen_tokens == input_ids.shape[1]`` (full prompt length), so
        ``model.generate(input_ids, past_key_values=past_key_values, ...)``
        starts decoding immediately without re-processing the prompt.
        """
        patch_qwen3_attention(self.model)
        params = self.params
        base_model = getattr(self.model, 'model', self.model)
        debug = self.debug

        # ---- Step 1: Enable Q_win capture ----
        for layer in base_model.layers:
            layer.self_attn._cluekv_obs_window = params.obs_window_size
            layer.self_attn._cluekv_q_win = None

        _dbg(debug, f"{'='*60}")
        _dbg(debug, f"ClueKV: kappa={params.kappa}, tau={params.tau}, "
             f"sink={params.sink_tokens}, obs_win={params.obs_window_size}, "
             f"budget={params.cache_budget_ratio}")

        # ---- Step 2: Full prefill ----
        outputs = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[0, -1, :].float()
        past_key_values = outputs.past_key_values
        L = input_ids.shape[1]
        _dbg(debug, f"Prefill: seq_len={L}, _seen_tokens={past_key_values._seen_tokens}")

        # ---- Step 3: CPS → clue tokens ----
        if empty_logits is None:
            empty_logits = self._empty_logits
        if empty_logits is None:
            raise ValueError("empty_logits required")

        delta_P = utils.compute_cps(logits, empty_logits, eps=params.eps)
        clue_ids = utils.select_clue_tokens(
            delta_P, kappa=params.kappa, tokenizer=tokenizer,
            stop_words=stop_words, stop_token_ids=self.stop_token_ids, debug=debug,
        )
        _dbg(debug, f"CPS: {len(clue_ids)} clues, ΔP>0: {(delta_P > 0).sum().item()}")

        if debug and tokenizer:
            for tid in sorted(clue_ids, key=lambda t: delta_P[t].item(), reverse=True)[:8]:
                _dbg(debug, f"  clue: id={tid} ΔP={delta_P[tid].item():+.1f} '{tokenizer.decode([tid])}'")

        # ---- Step 4: D_clue ----
        D_clue = utils.build_clue_distribution(input_ids, clue_ids, chunk_size=params.chunk_size)
        _dbg(debug, f"D: {len(D_clue)} segs, range=[{D_clue.min().item():.3f}, {D_clue.max().item():.3f}]")

        # ---- Step 5: Per-head attention ----
        attn_dists = utils.compute_attention_distributions(
            self.model, past_key_values,
            attn_compute_chunk=params.attn_compute_chunk,
            sink_tokens=params.sink_tokens, obs_window_size=params.obs_window_size,
            debug=debug,
        )

        # ---- Step 6: Compress each layer ----
        num_kv_heads = base_model.config.num_key_value_heads
        num_heads = base_model.config.num_attention_heads
        heads_per_group = num_heads // num_kv_heads
        B_target = max(1, int(L * params.cache_budget_ratio))

        for layer_idx, s_bars in attn_dists.items():
            if s_bars is None:
                continue
            L_align = utils.compute_head_alignment(
                s_bars, D_clue, chunk_size=params.chunk_size, eps=params.eps)
            sigma = utils.compute_head_weights(L_align, tau=params.tau)
            S_ref = utils.compute_reference_attention(sigma, s_bars)

            if debug:
                _dbg(debug, f"  --- L{layer_idx} ---")
                # JSD alignment per head (1/JSD, higher = more clue-aligned)
                jsd = 1.0 / (L_align + params.eps)
                _dbg(debug, f"    JSD: min={jsd.min().item():.4f} max={jsd.max().item():.4f} "
                     f"mean={jsd.mean().item():.4f}")
                _dbg(debug, f"    σ: min={sigma.min().item():.4f} max={sigma.max().item():.4f} "
                     f"mean={sigma.mean().item():.4f}")
                # Top-5 head indices with their σ and JSD
                top5 = torch.topk(sigma, k=min(5, len(sigma)))
                for rank, (h, w) in enumerate(zip(top5.indices.tolist(), top5.values.tolist())):
                    _dbg(debug, f"      head {h:>3}: σ={w:.4f}  JSD={jsd[h].item():.4f}")
                _dbg(debug, f"    S_ref: [{S_ref.min().item():.6f}, {S_ref.max().item():.6f}] "
                     f"nonzero={(S_ref > 0).sum().item()}/{S_ref.shape[0]}")

            before = past_key_values.key_cache[layer_idx].shape[2]
            utils.compress_layer_kv(
                S_ref, past_key_values, layer_idx=layer_idx,
                num_kv_heads=num_kv_heads, heads_per_group=heads_per_group,
                cache_budget_ratio=params.cache_budget_ratio,
                sink_tokens=params.sink_tokens, obs_window_size=params.obs_window_size,
                pool_mode=params.pool_mode,
                smooth_kernel=params.smooth_kernel,
                debug=debug,
            )
            after = past_key_values.key_cache[layer_idx].shape[2]
            _dbg(debug, f"    compress: {before}→{after} ({after/before*100:.0f}%)")

        # ---- Cleanup Q_win ----
        for layer in base_model.layers:
            layer.self_attn._cluekv_obs_window = 0
            layer.self_attn._cluekv_q_win = None

        _dbg(debug, f"Done. _seen_tokens={past_key_values._seen_tokens}, "
             f"cache_len={past_key_values.key_cache[0].shape[2]}")
        _dbg(debug, f"{'='*60}")

        return logits, past_key_values
