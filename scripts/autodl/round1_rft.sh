#!/usr/bin/env bash
# Round-1 RFT experiment (docs/EXPERIMENT_SWESMITH_RFT_ROUND1.md), run on the GPU host as root.
# Rollouts run as `nobody` through run_pool.sh; training uses the root-only ML env.
# Refuses to start if the GPU already runs a process this script did not start.
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl
AGENT=/opt/coding-agent
CODE=$AGENT/budget-agent
TASKS=$AGENT/swesmith
ROLL=$AGENT/work/round1          # rollout outputs (written by nobody)
OUT=$ML/outputs/round1           # training outputs (root)
CONFIG=$CODE/configs/swesmith-vllm.toml
PY=$ML/env/bin/python
WORKERS=${WORKERS:-32}
TEACHER=$ML/models/Qwen2.5-Coder-7B-Instruct
STUDENT=$ML/models/Qwen2.5-Coder-1.5B-Instruct
mkdir -p "$ROLL" "$OUT"
chown nobody:nogroup "$ROLL"
STEPS=$OUT/steps.jsonl

"$CODE/scripts/autodl/stop_vllm.sh"   # our own server from an earlier, interrupted run
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "GPU already runs another process; not starting experiments." >&2
  exit 1
fi

step() {  # step <name> <command...>: run, time and log one stage
  local name=$1; shift
  local started=$(date +%s)
  echo "== $name $(date -u +%FT%TZ)"
  "$@"
  printf '{"step": "%s", "seconds": %d, "finished": "%s"}\n' "$name" $(( $(date +%s) - started )) "$(date -u +%FT%TZ)" >> "$STEPS"
}
pool() {  # pool <name> <manifest> <split> <samples> <served-model>
  [ -f "$ROLL/$1/summary.json" ] && { echo "skip $1 (done)"; return; }
  "$CODE/scripts/autodl/run_pool.sh" --config "$CONFIG" --tasks "$2" --split "$3" --samples "$4" \
    --workers "$WORKERS" --output "$ROLL/$1" --resume --set "model=$5"
}

step serve-teacher "$CODE/scripts/autodl/serve_vllm.sh" "$TEACHER" teacher
step teacher-train pool teacher-train "$TASKS/train.jsonl" train 4 teacher
step teacher-eval pool teacher-eval "$TASKS/eval.jsonl" eval 2 teacher

step serve-student "$CODE/scripts/autodl/serve_vllm.sh" "$STUDENT" student
step student-base-eval pool student-base-eval "$TASKS/eval.jsonl" eval 4 student
step student-base-train pool student-base-train "$TASKS/train.jsonl" train 1 student

step sft-data "$PY" "$CODE/scripts/sft_data_from_runs.py" --attempts "$ROLL/teacher-train/attempts.jsonl" \
  --split train --per-task 2 --output "$OUT/sft.jsonl"
step stop-vllm "$CODE/scripts/autodl/stop_vllm.sh"
[ -f "$OUT/sft/result.json" ] || step train-sft "$PY" "$CODE/scripts/train_sft_lora.py" --model "$STUDENT" \
  --data "$OUT/sft.jsonl" --output "$OUT/sft" --merge

step serve-student-sft "$CODE/scripts/autodl/serve_vllm.sh" "$OUT/sft/merged" student-sft
step student-sft-eval pool student-sft-eval "$TASKS/eval.jsonl" eval 4 student-sft
step student-sft-train pool student-sft-train "$TASKS/train.jsonl" train 1 student-sft
step stop-vllm-final "$CODE/scripts/autodl/stop_vllm.sh"

PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/compare_pools.py" --baseline "$ROLL/student-base-eval/attempts.jsonl" \
  --candidate "$ROLL/student-sft-eval/attempts.jsonl" --split eval > "$OUT/compare_eval.json"
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/compare_pools.py" --baseline "$ROLL/student-base-train/attempts.jsonl" \
  --candidate "$ROLL/student-sft-train/attempts.jsonl" --split train > "$OUT/compare_train.json"
echo "== round 1 done $(date -u +%FT%TZ)"
