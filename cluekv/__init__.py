"""ClueKV: Self-Guided KV Cache Compression via Adaptive Head Prioritization.

Usage::

    from cluekv import ClueKVParams, ClueKVCompressor, patch_qwen3_attention

    model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
    patch_qwen3_attention(model)  # enable Q_win capture on attention layers

    compressor = ClueKVCompressor(model, params=ClueKVParams(
        kappa=32, tau=1.0, chunk_size=128, obs_window_size=32, cache_budget_ratio=0.5,
    ))
    empty_logits = compressor.compute_or_load_empty_logits(tokenizer, save_path="empty_logits.pt")

    logits, past_key_values = compressor.run_prefill_and_compress(
        input_ids=input_ids, tokenizer=tokenizer, empty_logits=empty_logits,
    )
    # ... continue generation with compressed past_key_values ...
"""

from .params import ClueKVParams
from .attention import patch_qwen3_attention, unpatch_qwen3_attention
from .compressor import ClueKVCompressor
from .utils import load_stop_token_ids

__all__ = [
    "ClueKVParams",
    "ClueKVCompressor",
    "patch_qwen3_attention",
    "unpatch_qwen3_attention",
    "load_stop_token_ids",
]
