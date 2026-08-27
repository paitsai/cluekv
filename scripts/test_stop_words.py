#!/usr/bin/env python3
"""Quick smoke test: verify the stop-words file is loaded and filters clue tokens."""

# python3 scripts/test_stop_words.py --model Qwen/Qwen3-1.7B --stop-words-path cache/stop_token_ids_qwen3.txt --subtask narrativeqa --num-examples 5

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from cluekv import ClueKVParams, ClueKVCompressor, load_stop_token_ids, patch_qwen3_attention
from cluekv.utils import compute_cps, select_clue_tokens
from longbench.runner import load_longbench
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser(description="Test stop-words filtering on clue selection")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HF model name")
    p.add_argument("--stop-words-path", default="cache/stop_token_ids_qwen3.txt",
                   help="Path to stop-words .txt file")
    p.add_argument("--data-dir", default="/home/users/xzr/benchmark/longbench-v1/data",
                   help="LongBench data directory")
    p.add_argument("--subtask", default="narrativeqa",
                   help="LongBench subtask to draw text from")
    p.add_argument("--num-examples", type=int, default=5,
                   help="Number of LongBench examples to test on")
    p.add_argument("--kappa", type=int, default=32)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Load tokenizer & stop words
    print("=" * 60)
    print("Step 1: Loading tokenizer and stop words...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    stop_ids = load_stop_token_ids(args.stop_words_path, tokenizer=tokenizer)
    print(f"  Loaded {len(stop_ids)} stop token IDs")
    # Show a few stop words for sanity check
    print("  Sample stop words:")
    for tid in list(stop_ids)[:10]:
        print(f"    id={tid:>6}  →  '{tokenizer.decode([tid])}'")

    # 2. Load model (small 1.7B)
    print("\n" + "=" * 60)
    print(f"Step 2: Loading model {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(args.device)
    model.eval()
    patch_qwen3_attention(model)

    # 3. Compute P(t|C(∅))
    print("\n" + "=" * 60)
    print("Step 3: Computing P(t|C(empty)) ...")
    compressor = ClueKVCompressor(model, params=ClueKVParams(
        kappa=args.kappa, obs_window_size=32,
    ))
    empty_logits = compressor.compute_or_load_empty_logits(tokenizer, device=args.device)

    # 4. Load LongBench examples
    print("\n" + "=" * 60)
    print(f"Step 4: Loading LongBench examples (subtask={args.subtask})")
    examples = load_longbench(subtasks=[args.subtask], data_dir=args.data_dir)
    examples = examples[:args.num_examples]
    print(f"  {len(examples)} examples loaded")

    # 5. CPS → clue tokens on each example, aggregate stats
    print("\n" + "=" * 60)
    print("Step 5: Comparing clue selection with/without stop words")

    total_no_filter = 0
    total_with_filter = 0
    total_filtered = 0
    all_filtered_tokens: list[tuple[int, float, str]] = []

    for i, example in enumerate(examples):
        # Build prompt the same way as runner.py
        from longbench.config import PROMPT_TEMPLATES
        template = PROMPT_TEMPLATES.get(example["dataset"],
            "Answer the question based on the given passage.\n\n{context}\n\nQuestion: {input}\nAnswer:")
        raw_prompt = template.format(**example)

        messages = [{"role": "user", "content": raw_prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Middle-truncation (same as runner.py)
        tokenized = tokenizer.encode(prompt_text, return_tensors="pt", truncation=False)[0]
        max_len = 8192
        if len(tokenized) > max_len:
            half = max_len // 2
            tokenized = torch.cat([tokenized[:half], tokenized[-half:]])
        input_ids = tokenized.unsqueeze(0).to(args.device)
        text_snippet = raw_prompt[:80]

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            logits = outputs.logits[0, -1, :].float()

        delta_P = compute_cps(logits, empty_logits, eps=1e-8)
        clues_no = select_clue_tokens(delta_P, kappa=args.kappa, stop_token_ids=None)
        clues_yes = select_clue_tokens(delta_P, kappa=args.kappa,
                                       stop_token_ids=stop_ids)
        filtered = clues_no - clues_yes

        nf = len(clues_no)
        f_len = len(clues_yes)
        fd = len(filtered)
        total_no_filter += nf
        total_with_filter += f_len
        total_filtered += fd
        print(f"    clues: {nf} → {f_len}  (filtered {fd})")

        # Print selected clue tokens
        print("    --- Selected clue tokens ---")
        for tid in sorted(clues_yes, key=lambda x: delta_P[x].item(), reverse=True):
            word = tokenizer.decode([tid]).replace("\n", "\\n")
            score = delta_P[tid].item()
            print(f"      id={tid:>6}  score={score:+.6f}  →  '{word}'")

        # Print filtered-out tokens
        if filtered:
            print("    --- Filtered by stop words ---")
            for tid in sorted(filtered, key=lambda x: delta_P[x].item(), reverse=True):
                word = tokenizer.decode([tid]).replace("\n", "\\n")
                score = delta_P[tid].item()
                print(f"      id={tid:>6}  score={score:+.6f}  →  '{word}'")
        else:
            print("    (none filtered)")

        for tid in filtered:
            all_filtered_tokens.append((tid, delta_P[tid].item(),
                                        tokenizer.decode([tid]).replace("\n", "\\n")))

    # 6. Summary
    print("\n" + "=" * 60)
    print("Step 6: Summary over all examples")
    print(f"  Total clues (no filter):  {total_no_filter}")
    print(f"  Total clues (with filter): {total_with_filter}")
    print(f"  Total filtered out:       {total_filtered}")
    print(f"  Filter rate: {total_filtered/total_no_filter*100:.1f}%" if total_no_filter else "  N/A")

    if all_filtered_tokens:
        # Aggregate by token and show top filtered
        from collections import Counter
        filtered_counter: Counter = Counter()
        for tid, score, word in all_filtered_tokens:
            filtered_counter[(tid, word)] += 1
        print("\n  Most frequently filtered tokens (across examples):")
        for (tid, word), count in filtered_counter.most_common(20):
            print(f"    id={tid:>6}  ×{count}  →  '{word}'")

    # Final verdict
    print("\n" + "=" * 60)
    if total_filtered > 0:
        print("PASS: stop-words file is working, successfully filtered clue tokens.")
    else:
        print("WARNING: No filtering effect — check stop words file or kappa value.")
    print("=" * 60)


if __name__ == "__main__":
    main()
