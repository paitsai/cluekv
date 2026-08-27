#!/bin/bash
# =============================================================================
# LongBench evaluation for ClueKV — convenience launcher
# =============================================================================
#
# Usage:
#   bash longbench/run.sh                                                    # defaults
#   bash longbench/run.sh --model Qwen/Qwen3-1.7B --subtasks qasper
#   bash longbench/run.sh --budget 0.5 --methods full_kv cluekv
#   bash longbench/run.sh --max-examples 50 --subtasks qasper --debug        # debug
#   DATA_DIR=/path/to/jsonl bash longbench/run.sh                            # env override
#
# Paper replication (Table 3):
#   bash longbench/run.sh --budget 0.5 --methods full_kv cluekv attn_baseline
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Defaults (overridable via environment variables)
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-8B}"
METHODS="${METHODS:-full_kv cluekv attn_baseline}"
BUDGET="${BUDGET:-}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1024}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
SUBTASKS="${SUBTASKS:-}"
DEVICE="${DEVICE:-cuda}"
DATA_DIR="${DATA_DIR:-/home/users/xzr/benchmark/longbench-v1/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
STOP_TOKEN_IDS_PATH="${STOP_TOKEN_IDS_PATH:-}"
DEBUG_FLAG=""

# ---------------------------------------------------------------------------
# Parse command-line args (override env defaults)
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2";          shift 2 ;;
        --methods)        shift; METHODS=""; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do METHODS+="$1 "; shift; done ;;
        --budget)         BUDGET="$2"; TOKEN_BUDGET=""; shift 2 ;;
        --token-budget)   TOKEN_BUDGET="$2"; BUDGET="";  shift 2 ;;
        --max-examples)   MAX_EXAMPLES="$2";   shift 2 ;;
        --subtasks)       shift; SUBTASKS="";  while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do SUBTASKS+="$1 "; shift; done ;;
        --device)         DEVICE="$2";         shift 2 ;;
        --data-dir)       DATA_DIR="$2";       shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --stop-token-ids-path) STOP_TOKEN_IDS_PATH="$2"; shift 2 ;;
        --debug)          DEBUG_FLAG="--debug"; shift ;;
        *) shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Build base command (shared args across all subtask iterations)
# ---------------------------------------------------------------------------
BASE_CMD="python3 -m longbench.eval_cluekv"
BASE_CMD+=" --model $MODEL"
if [ -n "$TOKEN_BUDGET" ]; then
    BASE_CMD+=" --token-budget $TOKEN_BUDGET"
else
    BASE_CMD+=" --budget ${BUDGET:-0.5}"
fi
BASE_CMD+=" --device $DEVICE"
BASE_CMD+=" --data-dir $DATA_DIR"
BASE_CMD+=" --output-dir $OUTPUT_DIR"

BASE_CMD+=" --methods"
for m in $METHODS; do BASE_CMD+=" $m"; done

if [ -n "$MAX_EXAMPLES" ]; then
    BASE_CMD+=" --max-examples $MAX_EXAMPLES"
fi
if [ -n "$STOP_TOKEN_IDS_PATH" ]; then
    BASE_CMD+=" --stop-token-ids-path $STOP_TOKEN_IDS_PATH"
fi
if [ -n "$DEBUG_FLAG" ]; then
    BASE_CMD+=" $DEBUG_FLAG"
fi

# ---------------------------------------------------------------------------
# If no subtasks specified, list all from config
# ---------------------------------------------------------------------------
if [ -z "$SUBTASKS" ]; then
    SUBTASKS="2wikimqa qasper hotpotqa musique multifieldqa_en gov_report qmsum multi_news trec triviaqa samsum lcc repobench-p passage_retrieval_en passage_retrieval_zh multifieldqa_zh"
fi

echo "============================================"
echo "  ClueKV LongBench Evaluation"
echo "============================================"
echo "  Model:       $MODEL"
echo "  Methods:     $METHODS"
if [ -n "$TOKEN_BUDGET" ]; then
    echo "  Token budget: $TOKEN_BUDGET"
else
    echo "  Budget:       ${BUDGET:-0.5}"
fi
echo "  Device:      $DEVICE"
echo "  Subtasks:    $SUBTASKS"
echo "  Debug:       ${DEBUG_FLAG:-off}"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Run: one subtask at a time, all methods together per subtask
# ---------------------------------------------------------------------------
TOTAL=$(echo $SUBTASKS | wc -w)
IDX=0
for subtask in $SUBTASKS; do
    IDX=$((IDX + 1))
    echo ""
    echo "##############################################"
    echo "  [$IDX/$TOTAL] Subtask: $subtask"
    echo "##############################################"
    CMD="$BASE_CMD --subtasks $subtask"
    echo "Running: $CMD"
    echo ""
    eval "$CMD"
    echo ""
    echo "  [$IDX/$TOTAL] $subtask — DONE"
done

echo ""
echo "============================================"
echo "  All $TOTAL subtasks complete."
echo "============================================"
