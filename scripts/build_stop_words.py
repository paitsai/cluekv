#!/usr/bin/env python3
"""Build pre-tokenized stop-word IDs from natural text frequency.

Counts token occurrences across arbitrary natural text (one or more plain
text files), selects the top-N most frequent tokens, and saves their decoded
strings as a .txt file (one per line).

Usage::

    # From arbitrary plain text files
    python scripts/build_stop_words.py \\
        --model Qwen/Qwen3-8B \\
        --text-file corpus1.txt corpus2.txt \\
        --top-n 200 \\
        --output cache/stop_token_ids_qwen3.txt

    # From a directory of text files
    python scripts/build_stop_words.py \\
        --model Qwen/Qwen3-8B \\
        --text-file dir/ \\
        --top-n 200 \\
        --output cache/stop_token_ids_qwen3.txt
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Build stop-token IDs from natural text frequency")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B",
                   help="HF model name or path (for tokenizer)")
    p.add_argument("--text-file", type=str, nargs="+", required=True,
                   help="One or more plain text files or directories of text "
                        "files to mine stop words from")
    p.add_argument("--top-n", type=int, default=2048,
                   help="Select top-N most frequent tokens as stop words")
    p.add_argument("--output", type=str,
                   default="./cache/stop_token_ids_qwen3.txt",
                   help="Output .txt file path")
    p.add_argument("--min-count", type=int, default=32,
                   help="Minimum occurrence count to be considered (pre-filter)")
    return p.parse_args()


def _iter_text_files(paths: list[str]) -> list[str]:
    """Expand file/directory args into a flat list of readable text files."""
    files: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                for name in sorted(names):
                    fp = os.path.join(root, name)
                    if os.path.isfile(fp):
                        files.append(fp)
        elif os.path.isfile(path):
            files.append(path)
        else:
            print(f"  Warning: skipping missing path: {path}")
    return files


def _read_text_files(files: list[str]) -> list[str]:
    """Read each file and return a list of text chunks (one per file)."""
    texts: list[str] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as exc:
            print(f"  Warning: could not read {fp}: {exc}")
            continue
        if content.strip():
            texts.append(content)
    return texts


def main():
    args = parse_args()

    # Load tokenizer
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Read source text files
    files = _iter_text_files(args.text_file)
    print(f"Loading {len(files)} text file(s)...")
    texts = _read_text_files(files)
    print(f"  {len(texts)} non-empty text chunks loaded")
    if not texts:
        raise SystemExit("No readable text found. Check --text-file paths.")

    # Count token frequencies across all texts
    counter: Counter = Counter()
    for i, text in enumerate(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        counter.update(token_ids)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(texts)} files")

    total_tokens = sum(counter.values())
    unique_tokens = len(counter)
    print(f"\nTotal tokens: {total_tokens:,}")
    print(f"Unique tokens: {unique_tokens:,}")

    # Pre-filter by min_count, then select top-N
    frequent = [(tid, cnt) for tid, cnt in counter.items() if cnt >= args.min_count]
    frequent.sort(key=lambda x: x[1], reverse=True)
    top_n = frequent[:args.top_n]

    # Deduplicate by decoded string (different IDs may decode to same token,
    # e.g. due to byte-fallback tokens in some tokenizers)
    seen: set[str] = set()
    stop_words: list[str] = []
    for tid, cnt in top_n:
        word = tokenizer.decode([tid]).strip()
        if word and word not in seen:
            seen.add(word)
            stop_words.append(word)
        if len(stop_words) >= args.top_n:
            break

    print(f"\nStop words selected: {len(stop_words)}")
    print("Top 30 tokens:")
    for word in stop_words[:30]:
        print(f"  '{word}'")

    # Save — one stopword per line
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for word in stop_words:
            f.write(f"{word}\n")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()