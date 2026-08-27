"""ClueKV hyperparameters."""

from dataclasses import dataclass


@dataclass
class ClueKVParams:
    """Hyperparameters for ClueKV KV cache compression.

    Attributes:
        kappa: Number of top-scoring clue tokens from CPS (default 128).
        tau: Temperature for softmax head weight normalization (default 1.0).
        chunk_size: Segment size for clue distribution D and attention aggregation (default 128).
        eps: Numerical stability constant (default 1e-8).
        sink_tokens: Number of initial tokens always retained in KV cache (default 4).
        obs_window_size: Number of most recent tokens always retained in KV cache and
            used as observation window Lq for attention capture (default 16).
            For LongReason: ~64 (question length).
        cache_budget_ratio: Fraction of KV cache to retain, e.g. 0.5 = 50%.
        attn_compute_chunk: Heads per chunk when computing attention distributions
            from saved Q_win and K_cache, to limit peak memory (default 4).
        stop_token_ids_path: Path to pre-computed stop-token-ids .txt file
            (from ``scripts/build_stop_words.py``). If provided, loaded by
            ``ClueKVCompressor`` for fast token-ID-based filtering.
    """
    kappa: int = 128
    tau: float = 1.0
    chunk_size: int = 128
    eps: float = 1e-8
    sink_tokens: int = 4
    obs_window_size: int = 16
    cache_budget_ratio: float = 0.5
    attn_compute_chunk: int = 4
    pool_mode: str = "max"
    smooth_kernel: int = 9
    stop_token_ids_path: str | None = None
