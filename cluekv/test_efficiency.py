#!/usr/bin/env python3
"""Benchmark ClueKV's TTFT overhead and TPOT acceleration vs. full KV cache.

Measures, on the same prompts:

* **TTFT** (Time To First Token)
  - ``full_kv`` : the prefill forward pass
  - ``cluekv``  : prefill + ClueKV compression pipeline
  -> reports the **TTFT overhead** introduced by ClueKV (and its breakdown:
   prefill / CPS / attention / compression)

* **TPOT** (Time Per Output Token)
  - ``full_kv`` : decode with the full KV cache
  - ``cluekv``  : decode with the compressed KV cache
  -> reports the **TPOT speedup** = TPOT(full) / TPOT(cluekv)

Usage::

    python -m cluekv.test_efficiency --model Qwen/Qwen3-1.7B \\
        --prompt-lengths 1024 2048 4096 \\
        --budget 0.5 --decode-tokens 32 --decode-skip 12 \\
        --num-prompts 3 --warmup-len 256

    # reuse cached empty logits / stop token ids
    python -m cluekv.test_efficiency --model Qwen/Qwen3-8B \\
        --empty-logits-path ./cache/empty_logits_qwen3.pt \\
        --stop-token-ids-path ./cache/stop_token_ids_qwen3.txt
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

# Make project root importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cluekv import ClueKVParams, ClueKVCompressor
from cluekv.attention import patch_qwen3_attention, unpatch_qwen3_attention
from cluekv import utils


# ---------------------------------------------------------------------------
# Synthetic prompt generation (natural-ish English text)
# ---------------------------------------------------------------------------

_CORPUS = [
    "The economic report released this morning showed stronger than expected "
    "growth in the manufacturing sector, driven largely by rising domestic "
    "demand and improved supply chains across the region.",
    "Researchers at the university have developed a new method for detecting "
    "early signs of the condition, which could significantly improve patient "
    "outcomes and reduce the burden on hospitals.",
    "The committee voted to approve the revised budget after a lengthy debate "
    "about how to allocate funds between infrastructure, education, and "
    "public health programs over the next fiscal year.",
    "Climate scientists warned that the current trajectory of emissions is "
    "not consistent with the targets set by the international agreement, "
    "urging governments to adopt more aggressive policies immediately.",
    "The software update introduces a redesigned interface, faster search, "
    "and better support for offline usage, along with a number of security "
    "improvements that address previously reported vulnerabilities.",
    "Local officials announced plans to expand the public transit network, "
    "adding several new routes and upgrading existing stations to improve "
    "accessibility and reduce travel times for commuters.",
    "A team of engineers presented a prototype that converts waste heat from "
    "industrial processes into usable electricity, offering a promising "
    "pathway toward more sustainable manufacturing.",
    "The annual report summarized the company's performance, highlighting "
    "record revenue in the second quarter and outlining strategic priorities "
    "for international expansion in the coming years.",
    "Historians traced the origins of the festival back to a harvest "
    "celebration held centuries ago, noting how its customs evolved as "
    "different communities adopted and adapted the tradition.",
    "The new regulation requires companies to disclose more detailed "
    "information about their data practices, giving consumers greater "
    "control over how their personal information is collected and used.",
]


def _build_prompt(tokenizer, target_tokens: int) -> str:
    """Build a natural English prompt of roughly ``target_tokens`` tokens."""
    text = ""
    while True:
        for sentence in _CORPUS:
            text += " " + sentence
            n = len(tokenizer.encode(text, add_special_tokens=False))
            if n >= target_tokens:
                return text.strip()


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _now():
    torch.cuda.synchronize()
    return time.perf_counter()


def _elapsed(t0):
    torch.cuda.synchronize()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Full-KV measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_full_kv(model, tokenizer, input_ids, decode_steps: int,
                    decode_skip: int) -> dict:
    """Return dict with 'ttft' and 'tpot' for full-KV inference."""
    # TTFT = prefill forward pass
    t0 = _now()
    out = model(input_ids=input_ids, use_cache=True)
    ttft = _elapsed(t0)
    past_key_values = out.past_key_values

    tpot = _decode_tpot(model, input_ids, past_key_values, decode_steps, decode_skip)

    return {"ttft": ttft, "tpot": tpot, "cache_len": past_key_values.key_cache[0].shape[2]}


# ---------------------------------------------------------------------------
# ClueKV measurement (with pipeline breakdown)
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_cluekv(model, tokenizer, compressor, empty_logits, input_ids, params,
                   decode_steps: int, decode_skip: int) -> dict:
    """Return dict with 'ttft', 'tpot' and per-stage timings for ClueKV."""
    patch_qwen3_attention(model)
    base_model = model.model
    times: dict[str, float] = {}

    # Enable observation-window Q capture
    for layer in base_model.layers:
        layer.self_attn._cluekv_obs_window = params.obs_window_size
        layer.self_attn._cluekv_q_win = None

    # ---- Prefill (same cost as full KV, but with capture enabled) ----
    t0 = _now()
    out = model(input_ids=input_ids, use_cache=True)
    times["prefill"] = _elapsed(t0)
    past_key_values = out.past_key_values
    L = input_ids.shape[1]

    # ---- CPS -> clue tokens -> D_clue ----
    t0 = _now()
    delta_P = utils.compute_cps(out.logits[0, -1].float(), empty_logits, eps=params.eps)
    clue_ids = utils.select_clue_tokens(
        delta_P, kappa=params.kappa, tokenizer=tokenizer,
        stop_token_ids=compressor.stop_token_ids,
    )
    D_clue = utils.build_clue_distribution(input_ids, clue_ids, chunk_size=params.chunk_size)
    times["cps"] = _elapsed(t0)

    # ---- Per-head attention distributions ----
    t0 = _now()
    attn_dists = utils.compute_attention_distributions(
        model, past_key_values,
        attn_compute_chunk=params.attn_compute_chunk,
        sink_tokens=params.sink_tokens,
        obs_window_size=params.obs_window_size,
    )
    times["attn"] = _elapsed(t0)

    # ---- Head weighting + KV compression ----
    t0 = _now()
    num_kv = base_model.config.num_key_value_heads
    num_h = base_model.config.num_attention_heads
    heads_per_group = num_h // num_kv
    for layer_idx, s_bars in attn_dists.items():
        if s_bars is None:
            continue
        L_align = utils.compute_head_alignment(
            s_bars, D_clue, chunk_size=params.chunk_size, eps=params.eps)
        sigma = utils.compute_head_weights(L_align, tau=params.tau)
        S_ref = utils.compute_reference_attention(sigma, s_bars)
        utils.compress_layer_kv(
            S_ref, past_key_values, layer_idx=layer_idx,
            num_kv_heads=num_kv, heads_per_group=heads_per_group,
            cache_budget_ratio=params.cache_budget_ratio,
            sink_tokens=params.sink_tokens, obs_window_size=params.obs_window_size,
            pool_mode=params.pool_mode, smooth_kernel=params.smooth_kernel,
        )
    times["compress"] = _elapsed(t0)
    times["total"] = sum(times.values())

    # Cleanup capture state
    for layer in base_model.layers:
        layer.self_attn._cluekv_obs_window = 0
        layer.self_attn._cluekv_q_win = None

    tpot = _decode_tpot(model, input_ids, past_key_values, decode_steps, decode_skip)
    cache_len = past_key_values.key_cache[0].shape[2]

    return {"ttft": times["total"], "tpot": tpot, "cache_len": cache_len,
            "times": times}


# ---------------------------------------------------------------------------
# Shared decode timing (fixed number of steps, manual loop)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _decode_tpot(model, input_ids, past_key_values, num_steps: int,
                 skip_steps: int = 0) -> float:
    """Decode ``num_steps`` single tokens; return mean time/token over the
    last ``num_steps - skip_steps`` steps (the first ``skip_steps`` are warmup)."""
    cur = input_ids[:, -1:]
    per_step: list[float] = []
    for i in range(num_steps):
        t0 = _now()
        out = model(input_ids=cur, past_key_values=past_key_values, use_cache=True)
        dt = _elapsed(t0)
        if i >= skip_steps:
            per_step.append(dt)
        nxt = out.logits[0, -1].argmax(-1).view(1, 1)
        cur = nxt
    if not per_step:
        return float("nan")
    return statistics.mean(per_step)


# ---------------------------------------------------------------------------
# CLI / driver
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark ClueKV TTFT/TPOT vs full KV")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--prompt-lengths", type=int, nargs="+", default=[1024, 2048],
                   help="Prompt (prefill) token lengths to benchmark")
    p.add_argument("--num-prompts", type=int, default=3,
                   help="Prompts to average per length (for stability)")
    p.add_argument("--decode-tokens", type=int, default=32,
                   help="Total decode steps for TPOT measurement")
    p.add_argument("--decode-skip", type=int, default=12,
                   help="Skip first N decode steps when averaging TPOT "
                        "(TPOT is the mean over the last decode-tokens-N steps)")
    p.add_argument("--warmup-len", type=int, default=256,
                   help="Run one warmup pass of both methods with a prompt of "
                        "this token length before measuring (0 disables)")
    p.add_argument("--budget", type=float, default=0.5,
                   help="ClueKV KV retention ratio")
    p.add_argument("--obs-window", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--kappa", type=int, default=128)
    p.add_argument("--empty-logits-path", type=str, default=None)
    p.add_argument("--stop-token-ids-path", type=str, default=None)
    p.add_argument("--output", type=str, default=None,
                   help="Optional JSON file to save results")
    return p.parse_args()


def load_model(model_name: str, device: torch.device, dtype_str: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dtype_map = {"auto": "auto", "fp16": torch.float16,
                 "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(dtype_str, "auto")
    print(f"Loading model: {model_name} (dtype={torch_dtype})")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model, device, args.dtype)

    params = ClueKVParams(
        kappa=args.kappa, chunk_size=args.chunk_size,
        obs_window_size=args.obs_window, cache_budget_ratio=args.budget,
    )

    stop_token_ids = None
    if args.stop_token_ids_path:
        from cluekv import load_stop_token_ids
        stop_token_ids = load_stop_token_ids(args.stop_token_ids_path, tokenizer=tokenizer)
        print(f"Loaded {len(stop_token_ids)} stop token IDs")

    compressor = ClueKVCompressor(model, params=params, stop_token_ids=stop_token_ids)

    save_path = args.empty_logits_path or f"./cache/empty_logits_{args.model.replace('/', '_')}.pt"
    empty_logits = compressor.compute_or_load_empty_logits(
        tokenizer, device=device, save_path=save_path)

    if args.decode_skip >= args.decode_tokens:
        raise SystemExit("--decode-skip must be < --decode-tokens")

    # ---- Warmup: run one small pass of both methods to warm CUDA kernels ----
    if args.warmup_len > 0:
        print(f"Warming up with a {args.warmup_len}-token prompt...")
        warm_prompt = _build_prompt(tokenizer, args.warmup_len)
        warm_ids = tokenizer(warm_prompt, return_tensors="pt").to(device).input_ids
        measure_full_kv(model, tokenizer, warm_ids, args.decode_tokens, args.decode_skip)
        measure_cluekv(model, tokenizer, compressor, empty_logits, warm_ids,
                       params, args.decode_tokens, args.decode_skip)
        torch.cuda.synchronize()
        print("  warmup done\n")

    print(f"\nBenchmarking {len(args.prompt_lengths)} lengths × "
          f"{args.num_prompts} prompts...\n")

    rows = []
    for length in args.prompt_lengths:
        for _ in range(args.num_prompts):
            prompt = _build_prompt(tokenizer, length)
            input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
            L = input_ids.shape[1]

            full = measure_full_kv(model, tokenizer, input_ids,
                                   args.decode_tokens, args.decode_skip)
            clue = measure_cluekv(model, tokenizer, compressor, empty_logits,
                                  input_ids, params,
                                  args.decode_tokens, args.decode_skip)
            rows.append({
                "target_length": length,
                "length": L,
                "full_ttft": full["ttft"], "full_tpot": full["tpot"],
                "full_cache_len": full["cache_len"],
                "clue_ttft": clue["ttft"], "clue_tpot": clue["tpot"],
                "clue_cache_len": clue["cache_len"],
                "breakdown": clue["times"],
            })
            print(f"  L≈{length:>6} (actual {L:>6}): "
                  f"TTFT full={full['ttft']*1000:.1f}ms clue={clue['ttft']*1000:.1f}ms | "
                  f"TPOT full={full['tpot']*1000:.2f}ms clue={clue['tpot']*1000:.2f}ms | "
                  f"cache {full['cache_len']}→{clue['cache_len']}")

    # Aggregate per length
    print(f"\n{'='*80}")
    print("Aggregated results (mean over prompts):")
    print(f"{'='*80}")
    header = (f"{'Length':>7} {'TTFT full':>10} {'TTFT clue':>10} {'TTFT ovh%':>9} "
              f"{'TPOT full':>10} {'TPOT clue':>10} {'Speedup':>8}")
    print(header)
    print("-" * len(header))
    summaries = []
    for length in args.prompt_lengths:
        sub = [r for r in rows if r["target_length"] == length]
        f_ttft = statistics.mean(r["full_ttft"] for r in sub)
        c_ttft = statistics.mean(r["clue_ttft"] for r in sub)
        f_tpot = statistics.mean(r["full_tpot"] for r in sub)
        c_tpot = statistics.mean(r["clue_tpot"] for r in sub)
        ovh = (c_ttft - f_ttft) / max(f_ttft, 1e-9) * 100.0
        speedup = f_tpot / max(c_tpot, 1e-9)
        summaries.append({
            "length": length, "full_ttft": f_ttft, "clue_ttft": c_ttft,
            "ttft_overhead_pct": ovh, "full_tpot": f_tpot, "clue_tpot": c_tpot,
            "tpot_speedup": speedup,
        })
        print(f"{length:>7} {f_ttft*1000:>9.1f}ms {c_ttft*1000:>9.1f}ms "
              f"{ovh:>8.1f}% {f_tpot*1000:>9.2f}ms {c_tpot*1000:>9.2f}ms "
              f"{speedup:>7.2f}x")

    # Pipeline breakdown (first prompt of each length)
    print(f"\n{'='*80}")
    print("ClueKV TTFT breakdown (prefill / CPS / attn / compress):")
    print(f"{'='*80}")
    for length in args.prompt_lengths:
        sub = [r for r in rows if r["target_length"] == length]
        if not sub:
            continue
        t = sub[0]["breakdown"]
        parts = "  ".join(f"{k}={v*1000:.1f}ms" for k, v in t.items())
        print(f"  L={length}: {parts}")

    if args.output:
        import json
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"config": vars(args), "results": summaries,
                       "rows": rows}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")

    unpatch_qwen3_attention(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
