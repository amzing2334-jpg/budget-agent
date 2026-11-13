#!/usr/bin/env bash
# Round 6 (docs/EXPERIMENT_LABELS_SFT.md), run on the GPU host as root: rebuild the
# round-1 SFT data with fixed revision labels (same teacher trajectories, one sample per
# request), retrain the SFT student with round 1's settings, and evaluate it under fixed
# labels with 8 samples per task. The baseline is round 4's relabel pool (the round-1
# student under relabelling); round 4's fixed pool gives the distribution-shift contrast.
# Refuses to start if the GPU already runs a process this script did not start.
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl
AGENT=/opt/coding-agent
CODE=$AGENT/budget-agent
STUDENT=$ML/models/Qwen2.5-Coder-1.5B-Instruct
ROLL=$AGENT/work/round6                # rollout outputs (written by nobody)
OUT=$ML/outputs/round6                 # SFT data, adapter, comparisons (root)
CONFIG=$CODE/configs/swesmith-vllm-tokens.toml
PY=$ML/env/bin/python
BASELINE=$AGENT/work/round4/relabel/attempts.jsonl
SHIFT=$AGENT/work/round4/fixed/attempts.jsonl
for pool in "$BASELINE" "$SHIFT"; do [ -s "$pool" ] || { echo "round-4 pool missing: $pool" >&2; exit 1; }; done
mkdir -p "$ROLL" "$OUT"
chown nobody:nogroup "$ROLL"
STEPS_LOG=$OUT/steps.jsonl

if pgrep -f "^bash $CODE/scripts/autodl/(round1_rft|round2_grpo|round3_grpo|round4_labels_ab|round5_grpo|round6_labels_sft|round7_grpo|round8_grpo|round9_eval|round10_fresh_tasks|round11_seed|round12_3b|grpo_pilot|bench|replicate_eval)\.sh" | grep -qvx "$$"; then
  echo "another experiment script of this project is running; not starting." >&2   # it would lose its vLLM
  exit 1
fi
grep -q test_timeout_seconds "$AGENT/swesmith/eval.jsonl" || { echo "manifests not migrated (add_test_timeouts.py)" >&2; exit 1; }
"$CODE/scripts/autodl/stop_vllm.sh"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "GPU already runs another process; not starting." >&2
  exit 1
fi
trap '"$CODE/scripts/autodl/stop_vllm.sh"' EXIT

step() {  # step <name> <command...>: run, time and log one stage
  local name=$1; shift
  local started=$(date +%s)
  echo "== $name $(date -u +%FT%TZ)"
  "$@"
  printf '{"step": "%s", "seconds": %d, "finished": "%s"}\n' "$name" $(( $(date +%s) - started )) "$(date -u +%FT%TZ)" >> "$STEPS_LOG"
}

# Same winners and order as round 1; the rebuild first reproduces every recorded request.
[ -s "$OUT/sft.jsonl" ] || step sft-data env PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/sft_data_from_runs.py" \
  --attempts "$AGENT/work/round1/teacher-train/attempts.jsonl" --split train --per-task 2 \
  --output "$OUT/sft.jsonl" --revision-labels fixed --sample-mode last
[ -f "$OUT/sft/result.json" ] || step train-sft "$PY" "$CODE/scripts/train_sft_lora.py" --model "$STUDENT" \
  --data "$OUT/sft.jsonl" --output "$OUT/sft" --merge

step serve-student "$CODE/scripts/autodl/serve_vllm.sh" "$OUT/sft/merged" student-fixed
if [ ! -f "$ROLL/fixed-trained/summary.json" ]; then
  "$PY" "$CODE/scripts/vllm_metrics.py" > "$OUT/vllm_before.json"
  step eval "$CODE/scripts/autodl/run_pool.sh" --config "$CONFIG" --tasks "$AGENT/swesmith/eval.jsonl" --split eval \
    --samples 8 --workers 32 --output "$ROLL/fixed-trained" --resume \
    --set model=student-fixed --set revision_labels=fixed --set api_logprobs=true
  "$PY" "$CODE/scripts/vllm_metrics.py" > "$OUT/vllm_after.json"
fi
step stop-vllm "$CODE/scripts/autodl/stop_vllm.sh"

compare() {  # compare <name> <baseline attempts> [flags]
  local name=$1 base=$2; shift 2
  PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/compare_pools.py" --baseline "$base" \
    --candidate "$ROLL/fixed-trained/attempts.jsonl" --split eval "$@" > "$OUT/compare_$name.json"
}
compare eval "$BASELINE"
compare eval_drop_flagged "$BASELINE" --drop-flagged
compare shift "$SHIFT"
# Pre-registered descriptive checks (CPU only).
"$PY" "$CODE/scripts/vllm_metrics.py" --diff "$OUT/vllm_before.json" "$OUT/vllm_after.json" > "$OUT/vllm.json"
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/bench_turn_packing.py" --tokenizer "$OUT/sft/merged" "$ROLL/fixed-trained" \
  2>/dev/null > "$OUT/packing.json"
"$PY" "$CODE/scripts/review_patches.py" "$ROLL/fixed-trained" --original "$AGENT/swesmith-src" \
  --tasks "$AGENT/swesmith/tasks" > "$OUT/original_check.txt"
"$PY" "$CODE/scripts/limit_events.py" "$ROLL/fixed-trained" > "$OUT/limit_events.jsonl"
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/summarize_round.py" --rollouts "$ROLL" --outputs "$OUT" > /dev/null
echo "== round 6 done $(date -u +%FT%TZ)"
