#!/usr/bin/env python3
"""LongBench evaluation with token-level cache budgets (paper Table 3 format).

This script evaluates ClueKV on LongBench using fixed token budgets (e.g.,
1024, 512) instead of ratios, matching the paper's reporting convention.

Usage::

    # Single budget
    python longbench/eval_token_budget.py \
        --model Qwen/Qwen3-8B --budgets 1024 512 \
        --methods full_kv cluekv ada_snapkv pyramidkv

    # Quick test
    python longbench/eval_token_budget.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --budgets 1024 --max-examples 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Any

# Make project root importable (needed before importing longbench or cluekv)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from cluekv import ClueKVParams, ClueKVCompressor, patch_qwen3_attention
from longbench import config
from longbench.runner import (
    load_longbench, evaluate, print_results_table, save_results,
    run_example_full_kv, post_process, metrics,
)


# ===========================================================================
# Token-budget-based ClueKV wrapper
# ===========================================================================

def budget_ratio_from_tokens(input_ids: torch.Tensor, token_budget: int) -> float:
    """Convert a token-budget to a cache retention ratio."""
    L = input_ids.shape[1]
    return min(1.0, token_budget / L)


class TokenBudgetCompressor:
    """Wrap ClueKVCompressor to support per-example token-budget adaptation."""

    def __init__(self, model, base_params: ClueKVParams, token_budget: int,
                 stop_token_ids: set[int] | None = None):
        self.model = model
        self.base_params = base_params
        self.token_budget = token_budget
        # Reuse the main compressor with a placeholder ratio; ratio gets
        # updated per-example before compression.
        self._compressor = ClueKVCompressor(model, params=ClueKVParams(
            kappa=base_params.kappa,
            tau=base_params.tau,
            chunk_size=base_params.chunk_size,
            obs_window_size=base_params.obs_window_size,
            attn_compute_chunk=base_params.attn_compute_chunk,
            cache_budget_ratio=0.5,  # dummy, overwritten per-example
            eps=base_params.eps,
        ), stop_token_ids=stop_token_ids)

    def run_prefill_and_compress(self, input_ids, tokenizer, empty_logits, **kw):
        """Adjust ratio per example, then run ClueKV compress."""
        ratio = budget_ratio_from_tokens(input_ids, self.token_budget)
        self._compressor.params.cache_budget_ratio = ratio
        return self._compressor.run_prefill_and_compress(
            input_ids=input_ids, tokenizer=tokenizer,
            empty_logits=empty_logits, **kw,
        )

    def compute_or_load_empty_logits(self, tokenizer, device="cuda", save_path=None):
        return self._compressor.compute_or_load_empty_logits(
            tokenizer, device=device, save_path=save_path,
        )


# ===========================================================================
# Token-budget evaluation
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="LongBench token-budget evaluation")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--budgets", type=int, nargs="+", default=[1024, 512],
                   help="Token budgets (e.g. 1024 512)")
    p.add_argument("--methods", type=str, nargs="+",
                   default=["full_kv", "cluekv", "attn_baseline"],
                   choices=["full_kv", "cluekv", "attn_baseline"])
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--subtasks", type=str, nargs="*", default=None)
    p.add_argument("--data-dir", type=str,
                   default="/home/users/xzr/benchmark/longbench-v1/data",
                   help="Directory containing {subtask}.jsonl files (local-first, HF fallback)")
    p.add_argument("--empty-logits-path", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--kappa", type=int, default=32)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--obs-window", type=int, default=32)
    p.add_argument("--stop-token-ids-path", type=str, default=None,
                   help="Path to pre-computed stop token IDs (.txt)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dtype_map = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, "auto")

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type == "cuda" and torch_dtype == "auto":
        model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = config.MODEL_CONFIGS.get(args.model, config.MODEL_CONFIGS["Qwen/Qwen3-8B"])

    # ---- Patch ----
    if "cluekv" in args.methods or "attn_baseline" in args.methods:
        patch_qwen3_attention(model)

    # ---- Stop token IDs ----
    stop_token_ids = None
    if args.stop_token_ids_path:
        from cluekv import load_stop_token_ids
        print(f"Loading stop token IDs from {args.stop_token_ids_path}")
        stop_token_ids = load_stop_token_ids(args.stop_token_ids_path, tokenizer=tokenizer)
        print(f"  {len(stop_token_ids)} stop tokens loaded")

    # ---- Data ----
    print("Loading LongBench...")
    data = load_longbench(subtasks=args.subtasks, data_dir=args.data_dir)
    print(f"  {len(data)} examples")

    # ---- Evaluate per budget ----
    all_budget_results: dict[int, list[dict]] = {}

    for budget in args.budgets:
        print(f"\n{'#'*60}")
        print(f"#  Token Budget = {budget}")
        print(f"{'#'*60}")

        # ClueKV params
        base_params = ClueKVParams(
            kappa=args.kappa, tau=args.tau,
            chunk_size=args.chunk_size, obs_window_size=args.obs_window,
            cache_budget_ratio=0.5,  # dummy
        )
        token_compressor = TokenBudgetCompressor(model, base_params, budget,
                                                  stop_token_ids=stop_token_ids)

        # Empty logits
        empty_logits = None
        if "cluekv" in args.methods:
            save_path = args.empty_logits_path or f"./cache/empty_logits_{args.model.replace('/', '_')}.pt"
            empty_logits = token_compressor.compute_or_load_empty_logits(
                tokenizer, device=device, save_path=save_path,
            )

        budget_results = []
        for method in args.methods:
            print(f"\n  --- method={method} ---")
            # Adapt: for cluekv, use the token-budget compressor
            compressor_for_runner = token_compressor._compressor if method == "cluekv" else None
            empty_for_runner = empty_logits if method == "cluekv" else None

            result = evaluate(
                model=model, tokenizer=tokenizer,
                compressor=compressor_for_runner,
                empty_logits=empty_for_runner,
                model_config=model_config,
                device=device,
                method=method,
                max_examples=args.max_examples,
                subtasks=args.subtasks,
                data=data,
                data_dir=args.data_dir,
            )
            budget_results.append(result)
            print(f"  {method:>20}: avg={result['average']:.2f}")

        all_budget_results[budget] = budget_results

    # ---- Print full table (like paper Table 3) ----
    print("\n" + "=" * 90)
    print("  FINAL RESULTS TABLE (LongBench, token budgets)")
    print("=" * 90)

    subtask_order = sorted(set(
        s for results in all_budget_results.values()
        for r in results for s in r["scores"].keys()
    ))

    # Header
    print(f"\n{'Budget':>8}  {'Method':>20}", end="")
    for s in subtask_order:
        alias = config.SUBTASK_ALIASES.get(s, s[:6])
        print(f"  {alias:>7}", end="")
    print(f"  {'Avg':>7}")
    print("-" * 90)

    for budget in args.budgets:
        for r in all_budget_results[budget]:
            print(f"{budget:>8}  {r['method']:>20}", end="")
            for s in subtask_order:
                v = r["scores"].get(s)
                print(f"  {v:7.2f}" if v is not None else "       —", end="")
            print(f"  {r['average']:7.2f}")

    print("=" * 90)

    # ---- Save ----
    os.makedirs(args.output_dir, exist_ok=True)
    for budget, results in all_budget_results.items():
        output_path = os.path.join(args.output_dir, f"longbench_budget{budget}.json")
        save_results(results, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
