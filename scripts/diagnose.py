#!/usr/bin/env python3
"""Diagnostic: compare full_kv vs cluekv on a single LongBench example."""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cluekv import ClueKVParams, ClueKVCompressor, patch_qwen3_attention
from longbench.config import PROMPT_TEMPLATES, DATASET2MAXLEN, MODEL_CONFIGS
from longbench.runner import (
    _build_raw_prompt, _build_chat_prompt,
    _tokenize_with_middle_truncation,
    run_example_full_kv,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--data-dir", default="/home/users/xzr/benchmark/longbench-v1/data")
    p.add_argument("--subtask", default="narrativeqa")
    p.add_argument("--example-idx", type=int, default=0)
    p.add_argument("--budget", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load data
    from longbench.runner import load_longbench
    examples = load_longbench(subtasks=[args.subtask], data_dir=args.data_dir)
    example = examples[args.example_idx]
    print(f"Dataset: {example['dataset']}")
    print(f"Question: {example['input'][:100]}...")
    print(f"Context length: {len(example['context'])} chars")
    print(f"Answers: {example['answers']}")

    # Load model
    print(f"\nLoading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = MODEL_CONFIGS.get(args.model, MODEL_CONFIGS["Qwen/Qwen3-8B"])

    # ============================================================
    # 1. full_kv
    # ============================================================
    print("\n" + "=" * 60)
    print("1. FULL KV")
    print("=" * 60)
    full_output = run_example_full_kv(model, tokenizer, example, model_config, device)
    print(f"  Output: {full_output[:200]}...")

    # ============================================================
    # 2. cluekv with manual decode loop
    # ============================================================
    print("\n" + "=" * 60)
    print("2. CLUEKV (manual decode loop)")
    print("=" * 60)

    patch_qwen3_attention(model)
    compressor = ClueKVCompressor(model, params=ClueKVParams(
        kappa=32, tau=1.0, chunk_size=128,
        sink_tokens=4, obs_window_size=16,
        cache_budget_ratio=args.budget, attn_compute_chunk=4,
    ))

    empty_logits = compressor.compute_or_load_empty_logits(
        tokenizer, device=device,
        save_path=f"./cache/empty_logits_{args.model.replace('/', '_')}.pt",
    )

    # Build prompt same as runner.py
    raw_prompt = _build_raw_prompt(example)
    dataset = example["dataset"]
    _RAW_PROMPT_DATASETS = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}
    if dataset not in _RAW_PROMPT_DATASETS:
        prompt = _build_chat_prompt(raw_prompt, tokenizer)
    else:
        prompt = raw_prompt

    max_length = model_config.get("max_length", 31500)
    input_ids = _tokenize_with_middle_truncation(tokenizer, prompt, max_length, device)
    max_new = DATASET2MAXLEN.get(dataset, 128)

    print(f"  Input length: {input_ids.shape[1]} tokens")
    print(f"  Max new tokens: {max_new}")

    # Prefill + compress
    logits, past_key_values = compressor.run_prefill_and_compress(
        input_ids=input_ids, tokenizer=tokenizer, empty_logits=empty_logits,
    )

    # Check: does the first token match full_kv?
    full_input_ids = _tokenize_with_middle_truncation(
        tokenizer,
        _build_chat_prompt(raw_prompt, tokenizer) if dataset not in _RAW_PROMPT_DATASETS else raw_prompt,
        max_length, device,
    )
    with torch.no_grad():
        full_out = model(full_input_ids, use_cache=True)
    full_first_logits = full_out.logits[0, -1, :].float()
    full_first_token = torch.argmax(full_first_logits, dim=-1).item()

    clue_first_token = torch.argmax(logits, dim=-1).item()
    print(f"\n  Full KV first token:  {tokenizer.decode([full_first_token])!r} (id={full_first_token})")
    print(f"  ClueKV first token:   {tokenizer.decode([clue_first_token])!r} (id={clue_first_token})")
    print(f"  First token match: {full_first_token == clue_first_token}")

    # Check CPS logits diff
    diff = (logits - full_first_logits.to(device)).abs().max().item()
    print(f"  Max logits diff (prefill): {diff:.6f}")

    # Check compressed cache sizes
    print(f"\n  KV cache sizes after compression:")
    for layer_idx in range(min(3, len(past_key_values.key_cache))):
        k_shape = past_key_values.key_cache[layer_idx].shape
        print(f"    Layer {layer_idx}: K={list(k_shape)} (retained {k_shape[2]}/{input_ids.shape[1]})")

    # Manual decode
    first_token = clue_first_token
    first_tensor = torch.tensor([[first_token]], device=device)
    generated_ids = [first_token]
    current_input = first_tensor
    for step in range(max_new - 1):
        with torch.no_grad():
            outputs = model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
        next_token = torch.argmax(outputs.logits[0, -1, :], dim=-1).item()
        generated_ids.append(next_token)
        past_key_values = outputs.past_key_values
        if next_token == tokenizer.eos_token_id:
            break
        current_input = torch.tensor([[next_token]], device=device)

    clue_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\n  ClueKV output: {clue_output[:200]}...")
    print(f"  Full KV output: {full_output[:200]}...")

    # ============================================================
    # 3. Try using model.generate() with compressed cache
    # ============================================================
    print("\n" + "=" * 60)
    print("3. CLUEKV (model.generate with compressed cache)")
    print("=" * 60)

    # Re-do prefill + compress to get fresh cache
    logits2, past_kv2 = compressor.run_prefill_and_compress(
        input_ids=input_ids, tokenizer=tokenizer, empty_logits=empty_logits,
    )
    first_token2 = torch.argmax(logits2, dim=-1).unsqueeze(0).unsqueeze(0)

    gen_outputs = model.generate(
        first_token2,
        past_key_values=past_kv2,
        max_new_tokens=max_new - 1,
        do_sample=False,
        temperature=1.0,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen_text = tokenizer.decode(
        gen_outputs[0][1:], skip_special_tokens=True  # skip the first_token we fed
    )
    # Actually gen_outputs includes the input, so:
    gen_text_full = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
    print(f"  generate() output: {gen_text_full[:200]}...")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Full KV:     {full_output[:150]}")
    print(f"  ClueKV loop: {clue_output[:150]}")
    print(f"  ClueKV gen:  {gen_text_full[:150]}")


if __name__ == "__main__":
    main()
