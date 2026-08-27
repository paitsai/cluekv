"""ClueKV utility functions: CPS, clue distribution, head weighting, KV compression."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Default English stop words used to filter CPS-detected clue tokens
_DEFAULT_STOP_WORDS: set[str] = {
    # Standard English stop words
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither", "each", "every",
    "all", "any", "few", "more", "most", "other", "some", "such", "no",
    "only", "own", "same", "than", "too", "very", "just", "because",
    "about", "up", "out", "if", "then", "now", "here", "there",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "him", "her", "them", "his", "their", "who", "whom", "which", "what",
    "when", "where", "why", "how",
    # Common BPE subword prefixes/suffixes (semantically empty)
    "un", "re", "ly", "ing", "ed", "er", "es", "est", "or", "al",
    "ic", "ive", "ous", "ful", "less", "ness", "ment", "tion", "sion",
    "pre", "dis", "mis", "over", "sub", "inter", "anti", "non", "ance",
    "ence", "ize", "ify", "ary", "ate", "ity", "ure", "ile", "ine",
    "ory", "ent", "ial", "ian", "ist", "ism", "ish",
    # Punctuation
    "\n", ",", ".", "?", "!", ";", ":",
    "'", '"', "(", ")", "[", "]", "{", "}", "-", "_", "/", "\\", "|",
    "*", "&", "%", "$", "#", "@", "+", "=", "<", ">", "`", "~", "^",
    # Common answer-start patterns (often get high ΔP but add no content)
    "yes", "no", "true", "false", "answer",
}


def get_stop_words() -> set[str]:
    """Return the default English stop-word set used for clue filtering."""
    return _DEFAULT_STOP_WORDS.copy()


def load_stop_token_ids(path: str, tokenizer=None) -> set[int]:
    """Load stop token IDs from a text file.

    Supports two file formats:

    * **Token-ID format** (legacy): one integer token ID per line.
    * **Word format**: one decoded token string per line.  When ``tokenizer``
      is provided, each word is encoded to its single-token ID via
      ``tokenizer.encode(word, add_special_tokens=False)``.  Words that encode
      to more than one token are skipped (they cannot be matched by a single
      ID lookup in the fast path).

    Args:
        path: Path to a ``.txt`` file (one token ID or one word per line).
        tokenizer: Optional HF tokenizer.  If provided, the file is treated as
            words; if ``None``, the file is treated as integer token IDs.

    Returns:
        A set of token ids to exclude from clue selection.
    """
    stop_ids: set[int] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if tokenizer is not None:
                # Encode original + case variants to catch "Yes"/"yes"/"YES"/"un"
                forms = {line, line.lower(), line.upper(), line.capitalize()}
                for form in forms:
                    ids = tokenizer.encode(form, add_special_tokens=False)
                    if len(ids) == 1:
                        stop_ids.add(ids[0])
            else:
                stop_ids.add(int(line))
    return stop_ids


# ---------------------------------------------------------------------------
# Stage 1: Contrastive Probability Shift → Clue tokens (paper §4.1)
# ---------------------------------------------------------------------------

def compute_cps(
    logits_S: torch.Tensor,       # [vocab_size] logits from first decode position → P(t|C(S))
    logits_empty: torch.Tensor,   # [vocab_size] logits from empty template → P(t|C(∅))
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute contrastive probability shift for every vocabulary token.

    ΔP_cond(t|S) = [P(t|C(S)) − P(t|C(∅))] / [P(t|C(∅)) + ε]

    Returns:
        delta_P: [vocab_size] shift values (higher = more clue-like).
    """
    device = logits_S.device
    logits_empty = logits_empty.to(device)

    probs_S = F.softmax(logits_S.float(), dim=-1)
    probs_empty = F.softmax(logits_empty.float(), dim=-1)

    delta_P = (probs_S - probs_empty) / (probs_empty + eps)
    return delta_P


def select_clue_tokens(
    delta_P: torch.Tensor,       # [vocab_size]
    kappa: int = 256,
    tokenizer=None,
    stop_words: set[str] | None = None,
    stop_token_ids: set[int] | None = None,
    debug: bool = False,
) -> set[int]:
    """Select top-κ clue token ids from CPS, with optional stop-word filtering.

    When ``stop_token_ids`` is provided, filtering is done directly by token ID
    (O(1) set lookup, no decoding). This is the fast path — use pre-computed
    stop-token IDs from ``scripts/build_stop_words.py``.

    Args:
        delta_P: CPS values per vocabulary token.
        kappa: Number of top-scoring candidates to consider.
        tokenizer: Optional HF tokenizer for stop-word filtering (only used
            when ``stop_token_ids`` is None).
        stop_words: Optional set of lower-case strings to exclude (fallback
            when ``stop_token_ids`` is None).
        stop_token_ids: Optional set of token ids to exclude (fast path).

    Returns:
        Set of vocabulary indices identified as clue tokens.
    """
    # Build a mask of valid candidates: ΔP > 0 and not a stop word
    valid_mask = delta_P > 0

    # ---- Layer 1: token-ID fast path (O(|stop_token_ids|)) ----
    if stop_token_ids is not None:
        for tid in stop_token_ids:
            if tid < len(valid_mask):
                valid_mask[tid] = False

    # ---- Layer 2: string-based fallback for case variants, subwords, etc. ----
    if tokenizer is not None:
        _stop_words = stop_words if stop_words is not None else _DEFAULT_STOP_WORDS
        vocab_size = len(delta_P)
        check_count = min(kappa * 8, vocab_size)  # wider net to catch stray variants
        _, check_indices = torch.topk(delta_P, k=check_count)
        for tid in check_indices.tolist():
            token_str = tokenizer.decode([tid]).strip().lower()
            if token_str in _stop_words or len(token_str) == 0:
                valid_mask[tid] = False

    # Set invalid tokens to -inf so topk skips them
    masked_delta = delta_P.clone()
    masked_delta[~valid_mask] = float('-inf')

    k_actual = min(kappa, valid_mask.sum().item())
    if k_actual == 0:
        if debug:
            print("[ClueKV]   select_clue_tokens: no valid candidates after filtering")
        return set()

    _, top_indices = torch.topk(masked_delta, k=k_actual)

    if debug:
        n_stop_filtered = kappa - valid_mask.sum().item() if valid_mask.sum().item() < kappa else 0
        print(f"[ClueKV]   select_clue_tokens: kappa={kappa}, "
              f"valid_before_stop={(delta_P > 0).sum().item()}, "
              f"valid_after_filter={valid_mask.sum().item()}, "
              f"k_actual={k_actual}")

    return set(top_indices.tolist())


# ---------------------------------------------------------------------------
# Stage 2: Clue distribution D (Algorithm 1)
# ---------------------------------------------------------------------------

def _resolve_chunk_size(L: int, chunk_size: int, min_chunks: int = 8) -> int:
    """Clamp chunk_size so that K = ceil(L / chunk_size) >= min_chunks."""
    return min(chunk_size, max(1, L // min_chunks))


def build_clue_distribution(
    input_ids: torch.Tensor,       # [1, L]
    clue_token_ids: set[int],
    chunk_size: int = 128,
    min_chunks: int = 8,
) -> torch.Tensor:
    """Algorithm 1: Build the spatial clue distribution D over context segments.

    Args:
        input_ids: Token ids of the full prefill sequence.
        clue_token_ids: Set of vocabulary ids identified as clues.
        chunk_size: Segment size (paper default 128).
        min_chunks: Minimum number of segments (default 16).

    Returns:
        D: [K] float32 segment-wise clue frequency, K >= min_chunks.
    """
    L = input_ids.shape[1]
    chunk_size = _resolve_chunk_size(L, chunk_size, min_chunks)
    K = (L + chunk_size - 1) // chunk_size

    ids = input_ids[0].tolist()
    N = torch.zeros(K, dtype=torch.float32)

    for k in range(K):
        seg = ids[k * chunk_size : (k + 1) * chunk_size]
        N[k] = sum(1 for t in seg if t in clue_token_ids)

    total = N.sum().item()
    if total > 0:
        D = N / total
    else:
        D = torch.ones(K) / K  # uniform fallback

    return D


# ---------------------------------------------------------------------------
# Stage 3: Head prioritization via JSD (paper §4.2)
# ---------------------------------------------------------------------------

def compute_head_alignment(
    s_bars: torch.Tensor,   # [num_heads, L] per-token attention distributions
    D_clue: torch.Tensor,   # [K] clue distribution
    chunk_size: int = 128,
    eps: float = 1e-8,
    min_chunks: int = 8,
) -> torch.Tensor:
    """Compute per-head alignment scores L_align(h) = 1 / (JSD(ŝ(h), D_clue) + ε).

    Vectorized over heads: segments are formed by reshaping the padded token
    axis into ``[K, chunk_size]`` chunks, so there is no per-head / per-chunk
    Python loop.  Mathematically identical to the naive per-head loop.

    Args:
        s_bars: Per-head attention distributions over all L context positions.
        D_clue: Clue spatial distribution over K segments.
        chunk_size: Segment size matching D_clue's granularity.
        eps: Numerical stability.
        min_chunks: Minimum number of segments (default 16).

    Returns:
        L_align: [num_heads] alignment scores (higher = more clue-aligned).
    """
    num_heads, L = s_bars.shape
    chunk_size = _resolve_chunk_size(L, chunk_size, min_chunks)
    K = (L + chunk_size - 1) // chunk_size
    device = s_bars.device

    D_clue = D_clue.to(device=device, dtype=torch.float32)
    s_bars = s_bars.to(device=device, dtype=torch.float32)

    # ---- Aggregate token-level → segment-level for all heads at once ----
    # Pad the token axis with zeros to a multiple of chunk_size; a partial
    # last chunk then contributes nothing extra, matching the scalar loop.
    pad_len = K * chunk_size - L
    if pad_len > 0:
        s_padded = F.pad(s_bars, (0, pad_len))
    else:
        s_padded = s_bars
    s_hat = s_padded.view(num_heads, K, chunk_size).sum(dim=2)  # [num_heads, K]

    # Normalize per head (skip normalization when the head's total is 0)
    s_sum = s_hat.sum(dim=1, keepdim=True)
    s_hat = s_hat / s_sum.clamp_min(1e-12)
    s_hat = torch.where(s_sum > 0, s_hat, torch.zeros_like(s_hat))

    # ---- Align D_clue to K segments (same for all heads) ----
    if D_clue.shape[0] < K:
        d_seg = F.pad(D_clue, (0, K - D_clue.shape[0]))
    elif D_clue.shape[0] > K:
        d_seg = D_clue[:K]
    else:
        d_seg = D_clue
    d_seg = d_seg.to(device)

    # ---- JSD = ½ KL(P‖M) + ½ KL(Q‖M),  M = ½(P+Q); broadcast over heads ----
    M = 0.5 * (s_hat + d_seg)                     # [num_heads, K]
    kl1 = (s_hat * (torch.log(s_hat + eps) - torch.log(M + eps))).sum(dim=1)
    kl2 = (d_seg * (torch.log(d_seg + eps) - torch.log(M + eps))).sum(dim=1)
    js_div = 0.5 * kl1 + 0.5 * kl2

    return 1.0 / (js_div + eps)


def compute_keep_indices(
    S_ref: torch.Tensor,       # [L]
    L: int,
    B_target: int,
    sink_tokens: int = 4,
    obs_window_size: int = 16,
    smooth_kernel: int = 5,
    pool_mode: str = "max",
) -> torch.Tensor:
    """Compute which token indices to keep (without modifying past_key_values).

    First ``sink_tokens`` and last ``obs_window_size`` positions are always kept.
    Remaining budget is allocated from middle positions via pool-smoothed S_ref.

    Returns a 1D LongTensor of sorted indices to retain.
    """
    device = S_ref.device

    protected = set()
    if sink_tokens > 0:
        protected.update(range(min(sink_tokens, L)))
    if obs_window_size > 0:
        protected.update(range(max(0, L - obs_window_size), L))

    if B_target <= len(protected):
        return torch.tensor(sorted(protected), device=device)

    B_middle = B_target - len(protected)

    S_smooth = _pool1d_smooth(S_ref, kernel_size=smooth_kernel, pool_mode=pool_mode)

    # Mask protected
    S_masked = S_smooth.clone()
    for idx in protected:
        S_masked[idx] = float('-inf')

    _, top_indices = torch.topk(S_masked, k=B_middle)
    all_indices = sorted(protected | set(top_indices.tolist()))
    return torch.tensor(all_indices, device=device)


def compute_head_weights(
    L_align: torch.Tensor,  # [num_heads]
    tau: float = 1.0,
) -> torch.Tensor:
    """Softmax-normalize alignment scores into head importance weights σ(h).

    σ(h) = exp(L_align(h) / τ) / Σ_j exp(L_align(j) / τ)
    """
    return F.softmax(L_align / tau, dim=0)


# ---------------------------------------------------------------------------
# Stage 4: KV cache compression (paper §4.3)
# ---------------------------------------------------------------------------

def compute_reference_attention(
    sigma: torch.Tensor,   # [num_heads] head weights
    s_bars: torch.Tensor,  # [num_heads, L] per-head attention distributions
) -> torch.Tensor:
    """S_ref = Σ σ(h) · s̄(h) — the head-weighted reference attention over tokens."""
    return (sigma.unsqueeze(1) * s_bars).sum(dim=0)  # [L]


def compress_layer_kv(
    S_ref: torch.Tensor,      # [L] or [num_heads, L] reference attention scores
    past_key_values,          # DynamicCache
    layer_idx: int,
    num_kv_heads: int,
    heads_per_group: int,
    cache_budget_ratio: float = 0.5,
    sink_tokens: int = 4,
    obs_window_size: int = 16,
    smooth_kernel: int = 9,
    pool_mode: str = "max",
    sigma: torch.Tensor | None = None,  # [num_heads] head weights for per-head mode
    debug: bool = False,
) -> None:
    """Compress the KV cache for one layer.

    Two modes:
    - **Per-head mode** (sigma is not None, S_ref is [num_heads, L]):
      Each head independently scores positions. Within each GQA group, the
      element-wise weighted-max across heads determines which tokens to keep
      (matching SnapKV's per-head independent selection).
    - **Single-S_ref mode** (sigma is None, S_ref is [L]):
      All heads share the same S_ref (baseline / aggregated mode).

    The first ``sink_tokens`` and last ``obs_window_size`` positions are
    always retained.
    """
    L = S_ref.shape[-1]  # works for both [L] and [num_heads, L]
    B = max(1, int(L * cache_budget_ratio))
    device = S_ref.device

    # Protected positions (clamped to valid range)
    protected = set()
    if sink_tokens > 0:
        protected.update(range(min(sink_tokens, L)))
    if obs_window_size > 0:
        protected.update(range(max(0, L - obs_window_size), L))

    if B <= len(protected):
        keep = torch.tensor(sorted(protected), device=device)
        K_cache = past_key_values.key_cache[layer_idx]
        V_cache = past_key_values.value_cache[layer_idx]
        past_key_values.key_cache[layer_idx] = K_cache[:, :, keep, :]
        past_key_values.value_cache[layer_idx] = V_cache[:, :, keep, :]
        if debug:
            print(f"[ClueKV]     compress: B <= protected, keeping only protected={len(protected)}")
        return

    B_middle = B - len(protected)

    if sigma is not None:
        # ---- Per-head mode: GQA-group-level selection ----
        num_heads = S_ref.shape[0]
        per_head_scores = S_ref  # [num_heads, L]

        # Smooth each head's scores with pool
        if smooth_kernel > 1:
            per_head_smoothed = _pool1d_smooth(per_head_scores, kernel_size=smooth_kernel, pool_mode=pool_mode)
        else:
            per_head_smoothed = per_head_scores

        # Weight by sigma (head importance from JSD alignment)
        weighted = sigma.unsqueeze(1) * per_head_smoothed  # [num_heads, L]

        # Within each GQA group, take element-wise max across heads
        # (any head in the group can flag a token as important — matching SnapKV)
        num_groups = num_kv_heads
        group_scores = torch.zeros(num_groups, L, device=device, dtype=weighted.dtype)
        for g in range(num_groups):
            head_start = g * heads_per_group
            head_end = head_start + heads_per_group
            group_scores[g] = weighted[head_start:head_end].max(dim=0).values

        # Mask protected positions
        for idx in protected:
            group_scores[:, idx] = float('-inf')

        # Per-group topk selection
        all_indices = set(protected)
        B_per_group = B_middle // num_groups
        remainder = B_middle % num_groups
        for g in range(num_groups):
            n_select = B_per_group + (1 if g < remainder else 0)
            _, indices = torch.topk(group_scores[g], k=n_select)
            all_indices.update(indices.tolist())

        keep = torch.tensor(sorted(all_indices), device=device)

        if debug:
            print(f"[ClueKV]     compress (per-head): total_L={L}, B_target={B}, "
                  f"protected={len(protected)}, B_middle={B_middle}, "
                  f"num_heads={num_heads}, num_groups={num_groups}, "
                  f"B_per_group~{B_per_group}, total_kept={len(keep)}")

    else:
        # ---- Single S_ref mode (original) ----
        if smooth_kernel > 1 and B_middle < L - len(protected):
            S_smooth = _pool1d_smooth(S_ref, kernel_size=smooth_kernel, pool_mode=pool_mode)
        else:
            S_smooth = S_ref

        S_masked = S_smooth.clone()
        for idx in protected:
            S_masked[idx] = float('-inf')

        _, top_indices = torch.topk(S_masked, k=B_middle)
        top_indices = top_indices.sort().values
        all_indices = sorted(protected | set(top_indices.tolist()))
        keep = torch.tensor(all_indices, device=device)

        if debug:
            mid_selected = [idx for idx in top_indices.tolist() if idx not in protected]
            print(f"[ClueKV]     compress (single): total_L={L}, B_target={B}, "
                  f"protected={len(protected)}, B_middle={B_middle}, "
                  f"actually_selected_middle={len(mid_selected)}, "
                  f"total_kept={len(keep)}")

    # Apply compression
    K_cache = past_key_values.key_cache[layer_idx]
    V_cache = past_key_values.value_cache[layer_idx]
    past_key_values.key_cache[layer_idx] = K_cache[:, :, keep, :]
    past_key_values.value_cache[layer_idx] = V_cache[:, :, keep, :]


def _pool1d_smooth(x: torch.Tensor, kernel_size: int = 5, pool_mode: str = "max") -> torch.Tensor:
    """Apply 1D pool with same-length output (stride=1, reflect padding).

    Args:
        x: [L] or [N, L] tensor.
        kernel_size: Pooling window size.
        pool_mode: "max" for maxpool, "avg" for avgpool.
    """
    pad = kernel_size // 2
    pool_fn = F.max_pool1d if pool_mode == "max" else F.avg_pool1d
    if x.dim() == 1:
        x_padded = F.pad(x.view(1, 1, -1), (pad, pad), mode='reflect')
    else:
        x_padded = F.pad(x.unsqueeze(1), (pad, pad), mode='reflect')  # [N, 1, L+2p]
    pooled = pool_fn(x_padded, kernel_size=kernel_size, stride=1)
    if x.dim() == 1:
        return pooled.view(-1)
    return pooled.squeeze(1)  # [N, L]


# ---------------------------------------------------------------------------
# Attention distribution computation from saved Q_win + KV cache
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_attention_distributions(
    model,
    past_key_values,
    attn_compute_chunk: int = 4,
    sink_tokens: int = 4,
    obs_window_size: int = 16,
    debug: bool = False,
) -> dict[int, torch.Tensor | None]:
    """Compute per-head attention distributions s̄(h) for every layer.

    Uses saved _cluekv_q_win from the attention modules and K from the KV
    cache (already populated during prefill).

    s̄(h) = mean_{i=1..Lq} softmax(Q_win[h,i] · K_rep[h].T / sqrt(dk))

    The first ``sink_tokens`` and last ``obs_window_size`` positions are
    excluded from the softmax normalisation (their attention is set to 0),
    since these positions are always retained regardless of s̄(h).

    Returns:
        dict mapping layer_idx → s_bars tensor [num_heads, L], or None if
        no Q_win was captured for that layer.
    """
    base_model = getattr(model, 'model', model)
    result: dict[int, torch.Tensor | None] = {}

    for layer_idx, decoder_layer in enumerate(base_model.layers):
        attn = decoder_layer.self_attn
        q_win = getattr(attn, '_cluekv_q_win', None)
        if q_win is None:
            result[layer_idx] = None
            continue

        n_groups = attn.num_key_value_groups
        K_cache = past_key_values.key_cache[layer_idx]  # [1, num_kv_heads, L, head_dim]
        dtype = K_cache.dtype

        num_heads = q_win.shape[1]
        L = K_cache.shape[2]
        scaling = attn.scaling
        device = q_win.device
        Lq = q_win.shape[2]

        # Build attention mask [Lq, L] in native dtype
        minf = torch.finfo(dtype).min
        attn_mask = torch.zeros(Lq, L, device=device, dtype=dtype)
        if sink_tokens > 0:
            attn_mask[:, :sink_tokens] = minf
        if obs_window_size > 0 and obs_window_size <= Lq:
            causal = torch.full((Lq, obs_window_size), minf, device=device, dtype=dtype)
            for qi in range(Lq):
                causal_start = max(0, qi - (Lq - obs_window_size))
                if causal_start < obs_window_size:
                    causal[qi, causal_start:] = 0.0
            attn_mask[:, -obs_window_size:] = causal

        s_bars_list = []
        chunk = attn_compute_chunk
        for i in range(0, num_heads, chunk):
            end = min(i + chunk, num_heads)
            chunk_h = end - i
            q_c = q_win[:, i:end, :, :].reshape(-1, Lq, q_win.shape[3])  # [chunk_h, Lq, D]

            # Map query heads to their KV heads without full expand+copy
            kv_start = i // n_groups
            kv_end = (end - 1) // n_groups + 1
            kv_indices = [h // n_groups - kv_start for h in range(i, end)]
            k_c = K_cache[:, kv_start:kv_end, :, :][:, kv_indices, :, :].reshape(-1, L, q_win.shape[3])  # [chunk_h, L, D]

            attn_scores = torch.bmm(q_c, k_c.transpose(1, 2)) * scaling
            attn_scores = attn_scores + attn_mask  # broadcast
            attn_probs = F.softmax(attn_scores, dim=-1)
            s_bar_c = attn_probs.mean(dim=1)  # [chunk_h, L]
            s_bars_list.append(s_bar_c)

        s_bars = torch.cat(s_bars_list, dim=0)  # [num_heads, L]
        result[layer_idx] = s_bars.to(device=device, dtype=dtype)

        if debug:
            eligible_positions = L - sink_tokens - min(obs_window_size, Lq)
            print(f"[ClueKV]   Layer {layer_idx}: s_bars shape={list(s_bars.shape)}, "
                  f"q_win_Lq={q_win.shape[2]}, eligible_positions~{eligible_positions}/{L}, "
                  f"s_bars nonzero_heads={(s_bars.sum(dim=1) > 0).sum().item()}/{num_heads}, "
                  f"sum_range=[{s_bars.sum(dim=1).min().item():.6f}, "
                  f"{s_bars.sum(dim=1).max().item():.6f}]")

        # Free captured Q_win to reclaim memory
        attn._cluekv_q_win = None

    return result
