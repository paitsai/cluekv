"""Evaluation loop: load data, run inference, compute metrics, report results."""

from __future__ import annotations

import time
import json
import os
import sys
from collections import defaultdict
from typing import Any

import torch

# Make cluekv importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cluekv import ClueKVParams, ClueKVCompressor, patch_qwen3_attention
from . import config  # noqa: E402
from . import metrics  # noqa: E402


# ===========================================================================
# Data loading
# ===========================================================================

def load_longbench(
    subtasks: list[str] | None = None,
    split: str = "test",
    data_dir: str | None = None,
) -> list[dict]:
    """Load LongBench — local JSONL first, HuggingFace datasets as fallback.

    Args:
        subtasks: Optional list of dataset names to filter.
        split: HF dataset split (ignored for local JSONL).
        data_dir: Directory containing ``{subtask}.jsonl`` files. If provided
            and the file exists, loaded locally; otherwise falls back to HF.

    Returns:
        List of dicts, each containing at least:
        input, context, answers, dataset, length, _id, all_labels
    """
    # Determine which subtasks to load
    if subtasks:
        tasks = list(subtasks)
    else:
        tasks = sorted(config.SUBTASKS.keys())

    # Separate: tasks available locally vs need HF fallback
    local_tasks = []
    hf_tasks = []
    for task_name in tasks:
        filepath = os.path.join(data_dir, f"{task_name}.jsonl") if data_dir else None
        if filepath and os.path.exists(filepath):
            local_tasks.append((task_name, filepath))
        else:
            hf_tasks.append(task_name)

    all_data: list[dict] = []

    # Load local JSONL
    for task_name, filepath in local_tasks:
        print(f"  Loading {task_name} from local: {filepath}")
        all_data.extend(_load_jsonl(filepath, task_name))

    # Load from HF in one batch
    if hf_tasks:
        if data_dir:
            print(f"  Loading from HF (local not found): {hf_tasks}")
        else:
            print(f"  Loading from HuggingFace datasets...")
        try:
            from datasets import load_dataset as _load_ds
            ds = _load_ds("THUDM/LongBench", split=split)
        except Exception:
            raise RuntimeError(
                "Failed to load LongBench. Make sure `datasets` is installed and "
                "you have access to 'THUDM/LongBench' on HuggingFace."
            )
        hf_task_set = set(hf_tasks)
        for example in ds:
            subtask = example.get("dataset", "")
            if subtask in hf_task_set:
                all_data.append({
                    "input": example.get("input", ""),
                    "context": example.get("context", ""),
                    "answers": example.get("answers", []),
                    "all_labels": example.get("all_labels", []),
                    "dataset": subtask,
                    "length": example.get("length", -1),
                    "_id": example.get("_id", ""),
                })

    return all_data


def _load_jsonl(filepath: str, task_name: str) -> list[dict]:
    """Load LongBench subtask from a local JSONL file."""
    import json as _json
    samples = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _json.loads(line)
            samples.append({
                "input": obj.get("input", ""),
                "context": obj.get("context", ""),
                "answers": obj.get("answers", []),
                "all_labels": obj.get("all_classes", []),
                "dataset": obj.get("dataset", task_name),
                "length": obj.get("length", -1),
                "_id": obj.get("_id", obj.get("id", str(len(samples)))),
            })
    return samples


# ===========================================================================
# Output post-processing
# ===========================================================================

# Datasets where the official implementation does NOT apply a chat template
# (few-shot and code tasks — the prompt already contains the expected format)
_RAW_PROMPT_DATASETS = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}

# Datasets that need first-line-only truncation in post-processing (official scorer)
_FIRST_LINE_DATASETS = {"trec", "triviaqa", "samsum"}


def post_process(output: str, example: dict, task_type: str, tokenizer) -> str:
    """Extract the model answer and clean up formatting markers."""
    # Remove chat template markers
    for tag in ["<|im_start|>", "<|im_end|>", "<|eot_id|>", "</s>", "<s>",
                 "[INST]", "[/INST]", "assistant", "user", "<|begin_of_text|>",
                 "<|start_header_id|>", "<|end_header_id|>"]:
        output = output.replace(tag, "")

    output = output.strip()

    # Strip leading "Answer:" / "answer:" prefix (chat template artifact)
    for prefix in ["Answer:", "answer:", "Answer：", "answer："]:
        if output.startswith(prefix):
            output = output[len(prefix):].strip()

    # Official scorer: for trec / triviaqa / samsum, take only the first line
    if example["dataset"] in _FIRST_LINE_DATASETS:
        lines = [l.strip() for l in output.split("\n") if l.strip()]
        if lines:
            output = lines[0]

    return output


# ===========================================================================
# Prompt helpers
# ===========================================================================

def _build_raw_prompt(example: dict) -> str:
    """Build the task-specific prompt from official templates (no chat wrapper)."""
    dataset = example["dataset"]
    template = config.PROMPT_TEMPLATES.get(dataset)
    if template is None:
        # Fallback: generic QA-style
        template = (
            "Answer the question based on the given passage.\n\n"
            "{context}\n\n"
            "Question: {input}\n"
            "Answer:"
        )
    return template.format(**example)


def _build_chat_prompt(input_text: str, tokenizer) -> str:
    """Build a chat-formatted prompt with Qwen3 thinking mode disabled."""
    messages = [{"role": "user", "content": input_text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return input_text


def _tokenize_with_middle_truncation(
    tokenizer, text: str, max_length: int, device: torch.device,
) -> torch.Tensor:
    """Tokenize, truncating from the middle (keep both ends) to preserve
    instructions at the start and the question/query at the end."""
    tokenized = tokenizer(text, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) <= max_length:
        return tokenized.unsqueeze(0).to(device)

    half = max_length // 2
    first = tokenized[:half]
    last = tokenized[-half:]
    truncated = torch.cat([first, last])
    return truncated.unsqueeze(0).to(device)


# ===========================================================================
# Single-example inference
# ===========================================================================

@torch.no_grad()
def run_example_full_kv(
    model,
    tokenizer,
    example: dict,
    model_config: dict,
    device: torch.device,
) -> str:
    """Standard full-KV inference on one example."""
    dataset = example["dataset"]
    raw_prompt = _build_raw_prompt(example)

    # Apply chat template only for non-fewshot/non-code tasks (matches official)
    if dataset not in _RAW_PROMPT_DATASETS:
        prompt = _build_chat_prompt(raw_prompt, tokenizer)
    else:
        prompt = raw_prompt

    max_length = model_config.get("max_length", 3500)
    input_ids = _tokenize_with_middle_truncation(tokenizer, prompt, max_length, device)
    inputs = {"input_ids": input_ids}

    max_new = config.DATASET2MAXLEN.get(dataset, 128)
    gen_kwargs = model_config.get("gen_kwargs", dict(do_sample=False, temperature=1.0))

    # Samsum special handling: prevent endless repetition (matches official)
    if dataset == "samsum":
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            min_length=input_ids.shape[-1] + 1,
            eos_token_id=[tokenizer.eos_token_id,
                          tokenizer.encode("\n", add_special_tokens=False)[-1]],
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )[0]
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            **gen_kwargs,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )[0]

    output_text = tokenizer.decode(
        outputs[input_ids.shape[1]:], skip_special_tokens=True
    )
    return output_text


@torch.no_grad()
def run_example_cluekv(
    model,
    tokenizer,
    example: dict,
    compressor: ClueKVCompressor,
    model_config: dict,
    device: torch.device,
    empty_logits: torch.Tensor,
) -> str:
    """ClueKV inference: prefill + compress + decode on one example."""
    dataset = example["dataset"]
    raw_prompt = _build_raw_prompt(example)

    if dataset not in _RAW_PROMPT_DATASETS:
        prompt = _build_chat_prompt(raw_prompt, tokenizer)
    else:
        prompt = raw_prompt

    max_length = model_config.get("max_length", 3500)
    input_ids = _tokenize_with_middle_truncation(tokenizer, prompt, max_length, device)

    max_new = config.DATASET2MAXLEN.get(dataset, 128)

    # ---- Prefill + compress (returns cache with _seen_tokens == L) ----
    _, past_key_values = compressor.run_prefill_and_compress(
        input_ids=input_ids,
        tokenizer=tokenizer,
        empty_logits=empty_logits,
    )

    # ---- Decode with model.generate() ----
    # _seen_tokens == L tricks generate() into thinking prefill is done.
    gen_outputs = model.generate(
        input_ids,
        past_key_values=past_key_values,
        max_new_tokens=max_new,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    output_text = tokenizer.decode(
        gen_outputs[0][input_ids.shape[1]:], skip_special_tokens=True
    )
    return output_text


@torch.no_grad()
def run_example_attn_baseline(
    model,
    tokenizer,
    example: dict,
    model_config: dict,
    device: torch.device,
    budget_ratio: float = 0.5,
    token_budget: int | None = None,
) -> str:
    """Simple attention-based baseline: keep top-attention-scored tokens."""
    dataset = example["dataset"]
    raw_prompt = _build_raw_prompt(example)

    if dataset not in _RAW_PROMPT_DATASETS:
        prompt = _build_chat_prompt(raw_prompt, tokenizer)
    else:
        prompt = raw_prompt

    max_length = model_config.get("max_length", 3500)
    input_ids = _tokenize_with_middle_truncation(tokenizer, prompt, max_length, device)
    L = input_ids.shape[1]

    if token_budget is not None:
        budget_ratio = min(1.0, token_budget / L)

    max_new = config.DATASET2MAXLEN.get(dataset, 128)

    # Prefill with attention capture (reuse ClueKV's capture infrastructure)
    for layer in model.model.layers:
        layer.self_attn._cluekv_obs_window = 32

    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = outputs.past_key_values

    # Compute mean attention across all heads (using same util)
    from cluekv.utils import compute_attention_distributions
    attn_dists = compute_attention_distributions(model, past_key_values, attn_compute_chunk=4)

    # Aggregate all layers' attention by averaging
    all_s_bars = []
    for s_bars in attn_dists.values():
        if s_bars is not None:
            all_s_bars.append(s_bars.mean(dim=0))  # mean over heads
    if all_s_bars:
        S_avg = torch.stack(all_s_bars).mean(dim=0)  # mean over layers
    else:
        S_avg = torch.ones(L, device=device)

    # Protected positions (sink + obs window) — same policy as ClueKV
    sink_tokens = 4
    obs_win = 16
    protected = set()
    if sink_tokens > 0:
        protected.update(range(min(sink_tokens, L)))
    if obs_win > 0:
        protected.update(range(max(0, L - obs_win), L))

    # Budget for middle (eligible) positions
    B = max(len(protected) + 1, int(L * budget_ratio))
    B_middle = B - len(protected)

    # Mask protected positions and select top-B_middle from the rest
    S_masked = S_avg.clone()
    for idx in protected:
        S_masked[idx] = float('-inf')
    _, top_indices = torch.topk(S_masked, k=B_middle)
    top_indices = sorted(protected | set(top_indices.tolist()))
    top_indices = torch.tensor(top_indices, device=device)

    # Compress each layer's KV cache
    for layer_idx in range(len(model.model.layers)):
        k = past_key_values.key_cache[layer_idx]
        v = past_key_values.value_cache[layer_idx]
        past_key_values.key_cache[layer_idx] = k[:, :, top_indices, :]
        past_key_values.value_cache[layer_idx] = v[:, :, top_indices, :]

    # Cleanup capture state
    for layer in model.model.layers:
        layer.self_attn._cluekv_obs_window = 0
        layer.self_attn._cluekv_q_win = None

    # Decode with model.generate() — _seen_tokens == L tricks it into skipping prefill
    gen_outputs = model.generate(
        input_ids,
        past_key_values=past_key_values,
        max_new_tokens=max_new,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    output_text = tokenizer.decode(
        gen_outputs[0][input_ids.shape[1]:], skip_special_tokens=True
    )
    return output_text


# ===========================================================================
# Evaluation loop
# ===========================================================================

def _ensure_output_dir(output_dir: str) -> str:
    """Create and return the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _append_result(method_dir: str, subtask: str, entry: dict) -> None:
    """Append one example result to the subtask JSONL file."""
    filepath = os.path.join(method_dir, f"{subtask}.jsonl")
    with open(filepath, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _save_summary(method_dir: str, method: str) -> None:
    """Rebuild _summary.json from all .jsonl files in the directory."""
    per_subtask: dict[str, list[float]] = defaultdict(list)
    all_scores = []
    for fname in sorted(os.listdir(method_dir)):
        if not fname.endswith(".jsonl"):
            continue
        subtask = fname[:-6]  # strip .jsonl
        fpath = os.path.join(method_dir, fname)
        with open(fpath, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                score = entry.get("score", 0.0)
                per_subtask[subtask].append(score)
                all_scores.append(score)

    scores: dict[str, float] = {}
    for subtask, score_list in sorted(per_subtask.items()):
        avg = sum(score_list) / max(len(score_list), 1)
        scores[subtask] = avg * 100.0
    avg_all = sum(all_scores) / max(len(all_scores), 1) * 100.0

    with open(os.path.join(method_dir, "_summary.json"), "w") as f:
        json.dump({
            "method": method,
            "scores": scores,
            "average": avg_all,
            "n_examples": len(all_scores),
        }, f, indent=2, ensure_ascii=False)


def _build_result_from_jsonl(method_dir: str, method: str) -> dict[str, Any]:
    """Reconstruct evaluation result from existing .jsonl files."""
    per_subtask: dict[str, list[float]] = defaultdict(list)
    per_example = []
    for fname in sorted(os.listdir(method_dir)):
        if not fname.endswith(".jsonl"):
            continue
        subtask = fname[:-6]
        fpath = os.path.join(method_dir, fname)
        with open(fpath, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                score = entry.get("score", 0.0)
                per_subtask[subtask].append(score)
                per_example.append(entry)

    scores = {}
    all_scores = []
    for subtask, score_list in sorted(per_subtask.items()):
        avg = sum(score_list) / max(len(score_list), 1)
        scores[subtask] = avg * 100.0
        all_scores.extend(score_list)
    avg_all = sum(all_scores) / max(len(all_scores), 1) * 100.0

    return {
        "method": method,
        "scores": scores,
        "average": avg_all,
        "n_examples": len(all_scores),
        "time_elapsed": 0.0,
        "per_example": per_example,
    }


# ===========================================================================
# Evaluation loop
# ===========================================================================

def evaluate(
    model,
    tokenizer,
    compressor: ClueKVCompressor | None,
    empty_logits: torch.Tensor | None,
    model_config: dict,
    device: torch.device,
    method: str = "cluekv",       # "full_kv", "cluekv", "attn_baseline"
    max_examples: int | None = None,
    subtasks: list[str] | None = None,
    data: list[dict] | None = None,
    data_dir: str | None = None,
    output_dir: str | None = None,
    token_budget: int | None = None,
) -> dict[str, Any]:
    """Run evaluation over LongBench examples.

    Args:
        method: One of "full_kv", "cluekv", "attn_baseline".
        max_examples: If set, only evaluate the first N examples (for quick tests).
        subtasks: If set, only evaluate these subtasks.
        data_dir: Directory with ``{subtask}.jsonl`` files (local-first, HF fallback).

    Returns:
        dict with 'scores' (per-subtask dict), 'average', 'per_example' results.
    """
    if data is None:
        data = load_longbench(subtasks=subtasks, data_dir=data_dir)

    per_subtask: dict[str, list[float]] = defaultdict(list)
    results = []

    # Set up incremental output directory
    method_dir = None
    if output_dir is not None:
        method_dir = _ensure_output_dir(output_dir)

    # Incremental: skip examples already saved in existing .jsonl files
    if method_dir is not None:
        finished_ids: dict[str, set[str]] = defaultdict(set)
        for fname in os.listdir(method_dir):
            if not fname.endswith(".jsonl"):
                continue
            subtask_name = fname[:-6]
            fpath = os.path.join(method_dir, fname)
            with open(fpath, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        eid = entry.get("_id", "")
                        if eid:
                            finished_ids[subtask_name].add(eid)
                    except json.JSONDecodeError:
                        continue

        pending = []
        skipped = 0
        for ex in data:
            eid = ex.get("_id", "")
            if eid and eid in finished_ids.get(ex["dataset"], set()):
                skipped += 1
            else:
                pending.append(ex)
        if skipped > 0:
            print(f"  [{method}] Skipping {skipped} already-finished examples, "
                  f"{len(pending)} remaining")
        data = pending

    total = len(data)
    if max_examples:
        data = data[:max_examples]
        total = len(data)

    if total == 0:
        print(f"  [{method}] All examples already finished, skipping.")
        return _build_result_from_jsonl(method_dir, method)

    t0 = time.time()

    for i, example in enumerate(data):
        subtask = example["dataset"]
        task_info = config.SUBTASKS.get(subtask, {})
        metric_name = task_info.get("metric", "f1")
        task_type = task_info.get("type", "qa")

        try:
            # ---- Inference ----
            if method == "full_kv":
                output = run_example_full_kv(model, tokenizer, example, model_config, device)
            elif method == "cluekv":
                if compressor is None or empty_logits is None:
                    raise ValueError("compressor and empty_logits required for cluekv")
                output = run_example_cluekv(model, tokenizer, example, compressor, model_config, device, empty_logits)
            elif method == "attn_baseline":
                output = run_example_attn_baseline(model, tokenizer, example, model_config, device,
                                                    token_budget=token_budget)
            else:
                raise ValueError(f"Unknown method: {method}")

            # ---- Post-process ----
            answer = post_process(output, example, task_type, tokenizer)

            # ---- Score ----
            all_labels = example.get("all_labels", [])
            score = metrics.compute_metric(
                answer, example["answers"], metric_name, all_classes=all_labels,
            )
            per_subtask[subtask].append(score)
            results.append({
                "_id": example["_id"],
                "dataset": subtask,
                "score": score,
                "output": answer,
                "answers": example.get("answers", []),
            })

        except Exception as e:
            # Log error but continue
            per_subtask[subtask].append(0.0)
            results.append({
                "_id": example.get("_id", ""),
                "dataset": subtask,
                "score": 0.0,
                "output": f"ERROR: {str(e)[:100]}",
            })

        # Incremental save: append this example to subtask JSONL + update summary
        if method_dir is not None:
            _append_result(method_dir, subtask, results[-1])
            _save_summary(method_dir, method)

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            pct = 100.0 * (i + 1) / total
            print(f"  [{method}] {i+1}/{total} ({pct:.1f}%) — {elapsed:.1f}s elapsed")

    # ---- Aggregate ----
    scores: dict[str, float] = {}
    all_scores = []
    for subtask, score_list in sorted(per_subtask.items()):
        avg = sum(score_list) / max(len(score_list), 1)
        scores[subtask] = avg * 100.0  # percentage
        all_scores.extend(score_list)

    avg_all = sum(all_scores) / max(len(all_scores), 1) * 100.0
    elapsed = time.time() - t0

    return {
        "method": method,
        "scores": scores,
        "average": avg_all,
        "n_examples": len(all_scores),
        "time_elapsed": elapsed,
        "per_example": results,
    }


# ===========================================================================
# Pretty-print results table (mimics paper Table 3 format)
# ===========================================================================

def print_results_table(
    results_list: list[dict],
    subtask_order: list[str] | None = None,
) -> None:
    """Print a table comparing multiple methods across subtasks."""
    if subtask_order is None:
        subtask_order = sorted(config.SUBTASKS.keys())

    # Column headers
    aliases = [config.SUBTASK_ALIASES.get(s, s) for s in subtask_order]
    aliases.append("Avg")

    print("\n" + "=" * 80)
    print(f"{'Method':>20}", end="")
    for alias in aliases:
        print(f"  {alias:>8}", end="")
    print()

    print("-" * 80)

    for result in results_list:
        method = result["method"]
        scores = result["scores"]
        avg = result["average"]

        print(f"{method:>20}", end="")
        for s in subtask_order:
            v = scores.get(s, None)
            if v is not None:
                print(f"  {v:8.2f}", end="")
            else:
                print(f"  {'—':>8}", end="")
        print(f"  {avg:8.2f}")

    print("=" * 80 + "\n")


# ===========================================================================
# Save results to JSON
# ===========================================================================

def save_results(results_list: list[dict], output_path: str) -> None:
    """Save evaluation results (summary + per-example) to a JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    serializable = []
    for r in results_list:
        serializable.append({
            "method": r["method"],
            "scores": r["scores"],
            "average": r["average"],
            "n_examples": r["n_examples"],
            "time_elapsed": r["time_elapsed"],
            "per_example": r.get("per_example", []),
        })
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")
