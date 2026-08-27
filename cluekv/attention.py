"""Monkey-patch Qwen3Attention to capture observation-window query states during prefill.

Matches the exact Qwen3Attention.forward signature from transformers 4.51+.
Only adds Q_win capture; no other logic is changed.
"""

from __future__ import annotations

import torch

_ORIGINAL_FORWARDS: dict[int, object] = {}


def _make_capture_forward(original_forward):
    """Wrap Qwen3Attention.forward to save Q_win during prefill."""

    def capture_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ---- ClueKV: capture observation-window Q during prefill ----
        obs_window = getattr(self, '_cluekv_obs_window', 0)
        if obs_window > 0:
            Lq = min(obs_window, query_states.shape[2])
            self._cluekv_q_win = query_states[:, :, -Lq:, :].detach().clone()

        # ---- Update KV cache ----
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # ---- Attention computation (identical to original Qwen3) ----
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        else:
            attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return capture_forward


def patch_qwen3_attention(model) -> None:
    """Replace Qwen3Attention.forward with capture_forward (idempotent)."""
    base_model = getattr(model, 'model', model)
    for layer in base_model.layers:
        attn = layer.self_attn
        attn_id = id(attn)
        if attn_id in _ORIGINAL_FORWARDS:
            continue
        _ORIGINAL_FORWARDS[attn_id] = attn.forward
        attn._cluekv_obs_window = 0
        attn._cluekv_q_win = None
        import types
        attn.forward = types.MethodType(_make_capture_forward(attn.forward), attn)


def unpatch_qwen3_attention(model) -> None:
    """Restore original forwards."""
    base_model = getattr(model, 'model', model)
    for layer in base_model.layers:
        attn = layer.self_attn
        attn_id = id(attn)
        if attn_id in _ORIGINAL_FORWARDS:
            attn.forward = _ORIGINAL_FORWARDS[attn_id]
            del _ORIGINAL_FORWARDS[attn_id]
        attn._cluekv_obs_window = 0
        attn._cluekv_q_win = None
