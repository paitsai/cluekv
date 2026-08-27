#!/usr/bin/env python3
"""LongBench evaluation script for ClueKV KV cache compression.

Usage::

    # Full evaluation (all subtasks, all methods)
    CUDA_VISIBLE_DEVICES=1 python longbench/eval_cluekv.py \
        --model Qwen/Qwen3-8B \
        --methods full_kv cluekv attn_baseline \
        --budget 0.2

    # Quick test with limited examples
    CUDA_VISIBLE_DEVICES=0 python longbench/eval_cluekv.py \
        --model Qwen/Qwen3-1.7B \
        --methods cluekv \
        --max-examples 20 \
        --subtasks qasper

    # With custom empty logits path
    python longbench/eval_cluekv.py \
        --model Qwen/Qwen3-8B \
        --empty-logits-path ./cache/empty_logits_qwen3.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Make project root importable (needed before importing longbench or cluekv)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from cluekv import ClueKVParams, ClueKVCompressor, patch_qwen3_attention
from longbench import config
from longbench.runner import load_longbench, evaluate, print_results_table, save_results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ClueKV on LongBench")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                   help="HF model name or path")
    p.add_argument("--methods", type=str, nargs="+",
                   default=["full_kv", "cluekv", "attn_baseline"],
                   choices=["full_kv", "cluekv", "attn_baseline"],
                   help="Which KV compression methods to evaluate")
    p.add_argument("--budget", type=float, default=None,
                   help="KV cache retention ratio (0.0–1.0). "
                        "Mutually exclusive with --token-budget")
    p.add_argument("--token-budget", type=int, default=None,
                   help="KV cache budget in tokens (e.g. 1024). "
                        "If set, ratio is computed per-example as token_budget/L")
    p.add_argument("--data-dir", type=str,
                   default="/home/users/xzr/benchmark/longbench-v1/data",
                   help="Directory containing {subtask}.jsonl files (local-first, HF fallback)")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Limit to first N examples (for quick smoke tests)")
    p.add_argument("--subtasks", type=str, nargs="*", default=None,
                   help="Only evaluate these subtasks (e.g. narrativeqa qasper)")
    p.add_argument("--empty-logits-path", type=str, default=None,
                   help="Path to cached empty-template logits (.pt)")
    p.add_argument("--output-dir", type=str, default="./results",
                   help="Directory for result JSON files")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for inference")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "fp16", "bf16", "fp32"],
                   help="Model dtype")
    p.add_argument("--attn-impl", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"],
                   help="Attention implementation (default: flash_attention_2)")
    p.add_argument("--kappa", type=int, default=128,
                   help="ClueKV: number of clue tokens")
    p.add_argument("--tau", type=float, default=1.0,
                   help="ClueKV: temperature for head weights")
    p.add_argument("--chunk-size", type=int, default=128,
                   help="ClueKV: segment size for clue distribution")
    p.add_argument("--obs-window", type=int, default=32,
                   help="ClueKV: observation window size Lq")
    p.add_argument("--stop-token-ids-path", type=str, default=None,
                   help="Path to pre-computed stop token IDs (.txt)")
    p.add_argument("--debug", action="store_true",
                   help="Print verbose intermediate results from ClueKV pipeline")
    return p.parse_args()


def load_model(model_name: str, device: torch.device, dtype_str: str,
               attn_impl: str = "flash_attention_2"):
    """Load the model and tokenizer with appropriate dtype."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype_map = {
        "auto": "auto",
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_str, "auto")

    print(f"Loading model: {model_name}  (dtype={torch_dtype}, attn={attn_impl})")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, tokenizer


class TokenBudgetAdapter:
    """Wrap ClueKVCompressor to support per-example token-budget adaptation."""

    def __init__(self, compressor: ClueKVCompressor, token_budget: int):
        self._compressor = compressor
        self.token_budget = token_budget

    def run_prefill_and_compress(self, input_ids, tokenizer, empty_logits, **kw):
        L = input_ids.shape[1]
        ratio = min(1.0, self.token_budget / L)
        self._compressor.params.cache_budget_ratio = ratio
        return self._compressor.run_prefill_and_compress(
            input_ids=input_ids, tokenizer=tokenizer,
            empty_logits=empty_logits, **kw)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    model, tokenizer = load_model(args.model, device, args.dtype, args.attn_impl)

    # ---- Model config ----
    model_config = config.MODEL_CONFIGS.get(
        args.model, config.MODEL_CONFIGS["Qwen/Qwen3-8B"]
    )

    # ---- Patch attention for ClueKV capture ----
    if "cluekv" in args.methods or "attn_baseline" in args.methods:
        patch_qwen3_attention(model)

    # ---- ClueKV params ----
    if args.token_budget is not None:
        default_ratio = 0.5  # placeholder, overwritten per-example
        budget_label = f"t{args.token_budget}"
    else:
        default_ratio = args.budget or 0.5
        budget_label = f"b{default_ratio}"

    cluekv_params = ClueKVParams(
        kappa=args.kappa,
        tau=args.tau,
        chunk_size=args.chunk_size,
        obs_window_size=args.obs_window,
        cache_budget_ratio=default_ratio,
    )
    stop_token_ids = None
    if args.stop_token_ids_path:
        from cluekv import load_stop_token_ids
        print(f"Loading stop token IDs from {args.stop_token_ids_path}")
        stop_token_ids = load_stop_token_ids(args.stop_token_ids_path, tokenizer=tokenizer)
        print(f"  {len(stop_token_ids)} stop tokens loaded")

    if args.token_budget is not None and "cluekv" in args.methods:
        # Wrap compressor to adjust ratio per example from token budget
        _base_comp = ClueKVCompressor(model, params=cluekv_params,
                                       stop_token_ids=stop_token_ids,
                                       debug=args.debug)
        compressor = TokenBudgetAdapter(_base_comp, args.token_budget)
        _base_comp_for_empty = _base_comp
    else:
        compressor = ClueKVCompressor(model, params=cluekv_params,
                                       stop_token_ids=stop_token_ids,
                                       debug=args.debug)
        _base_comp_for_empty = compressor

    # ---- Empty-template logits ----
    empty_logits = None
    if "cluekv" in args.methods:
        save_path = args.empty_logits_path or f"./cache/empty_logits_{args.model.replace('/', '_')}.pt"
        print(f"Computing/loading empty-template logits → {save_path}")
        empty_logits = _base_comp_for_empty.compute_or_load_empty_logits(
            tokenizer, device=device, save_path=save_path,
        )

    # ---- Load data ----
    print("Loading LongBench dataset...")
    data = load_longbench(subtasks=args.subtasks, data_dir=args.data_dir)
    print(f"  {len(data)} examples across {len(set(ex['dataset'] for ex in data))} subtasks")

    # ---- Run evaluation ----
    all_results = []
    os.makedirs(args.output_dir, exist_ok=True)
    for method in args.methods:
        budget_display = f"token_budget={args.token_budget}" if args.token_budget is not None else f"ratio={default_ratio}"
        print(f"\n{'='*60}")
        print(f"Evaluating: {method}  ({budget_display})")
        print(f"{'='*60}")

        if method == "cluekv":
            method_output_dir = os.path.join(args.output_dir, f"cluekv_{budget_label}")
        else:
            method_output_dir = os.path.join(args.output_dir, method)

        result = evaluate(
            model=model,
            tokenizer=tokenizer,
            compressor=compressor if method == "cluekv" else None,
            empty_logits=empty_logits if method == "cluekv" else None,
            model_config=model_config,
            device=device,
            method=method,
            max_examples=args.max_examples,
            subtasks=args.subtasks,
            data=data,
            data_dir=args.data_dir,
            output_dir=method_output_dir,
            token_budget=args.token_budget,
        )
        all_results.append(result)

        print(f"  {method:>20}: Average = {result['average']:.2f}  "
              f"({result['n_examples']} examples, {result['time_elapsed']:.1f}s)")
    subtask_order = sorted(set(
        s for r in all_results for s in r["scores"].keys()
    ))
    print_results_table(all_results, subtask_order=subtask_order)

    # ---- Save combined results ----
    combined_path = os.path.join(args.output_dir, "combined.json")
    save_results(all_results, combined_path)

    # ---- Print summary ----
    full_kv_avg = None
    for r in all_results:
        if r["method"] == "full_kv":
            full_kv_avg = r["average"]
            break

    if full_kv_avg:
        print("\nRetention vs. Full KV:")
        for r in all_results:
            ratio = r["average"] / max(full_kv_avg, 1e-6) * 100.0
            print(f"  {r['method']:>20}: {r['average']:.2f}  ({ratio:.1f}% of Full KV)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
