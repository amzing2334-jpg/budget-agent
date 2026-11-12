#!/usr/bin/env bash
# Round-10 GRPO (docs/EXPERIMENT_GRPO_ROUND10.md), run on the GPU host as root: continue round 8's
# final adapter on fresh training tasks. Candidates are the train tasks the round-1 teacher never
# solved; the starting policy samples each 4 times at the rollout temperature, and the tasks it
# solves at least once become the training set (N tasks; min(29, ceil(7N/8)) steps; pipelined;
# rollout-health guard on). Evaluates the adapter final.json names on the held-out split with 8
# samples per task against round 8's evaluation pool.
# Refuses to start if the GPU already runs a process this script did not start.
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl
AGENT=/opt/coding-agent
CODE=$AGENT/budget-agent
POLICY=$ML/outputs/round1/sft/merged
START=$ML/outputs/round8/grpo/adapters/v29
ROLL=$AGENT/work/round10              # rollout outputs (written by nobody)
OUT=$ML/outputs/round10               # tasks, adapters, metrics, comparisons (root)
CONFIG=$CODE/configs/swesmith-vllm-tokens.toml
PY=$ML/env/bin/python
BASELINE=$AGENT/work/round8/eval-grpo/attempts.jsonl     # round 8's v29, 8 samples per task
ROUND5=$AGENT/work/round5/eval-grpo/attempts.jsonl       # round 5's v29
SFT=$AGENT/work/round4/relabel/attempts.jsonl            # the SFT student
TEACHER=$AGENT/work/round1/teacher-train/attempts.jsonl  # what the round-1 teacher solved
for path in "$START/adapter_config.json" "$BASELINE" "$ROUND5" "$SFT" "$TEACHER"; do
  [ -s "$path" ] || { echo "missing: $path" >&2; exit 1; }
done
mkdir -p "$ROLL/grpo" "$OUT"
chown nobody:nogroup "$ROLL" "$ROLL/grpo"
STEPS_LOG=$OUT/steps.jsonl

if pgrep -f "^bash $CODE/scripts/autodl/(round1_rft|round2_grpo|round3_grpo|round4_labels_ab|round5_grpo|round6_labels_sft|round7_grpo|round8_grpo|round9_eval|round10_fresh_tasks|grpo_pilot|bench|replicate_eval)\.sh" | grep -qvx "$$"; then
  echo "another experiment script of this project is running; not starting." >&2   # it would lose its vLLM
  exit 1
fi
grep -q test_timeout_seconds "$AGENT/swesmith/train.jsonl" || { echo "manifests not migrated (add_test_timeouts.py)" >&2; exit 1; }
"$CODE/scripts/autodl/stop_vllm.sh"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "GPU already runs another process; not starting." >&2
  exit 1
fi
trap '"$CODE/scripts/autodl/stop_vllm.sh"' EXIT
echo "start=$START baseline=$BASELINE"

step() {  # step <name> <command...>: run, time and log one stage
  local name=$1; shift
  local started=$(date +%s)
  echo "== $name $(date -u +%FT%TZ)"
  "$@"
  printf '{"step": "%s", "seconds": %d, "finished": "%s"}\n' "$name" $(( $(date +%s) - started )) "$(date -u +%FT%TZ)" >> "$STEPS_LOG"
}

# 1. Candidates: the train tasks the round-1 teacher never solved.
"$PY" "$CODE/scripts/select_tasks.py" "$AGENT/swesmith/train.jsonl" --not-solved-in "$TEACHER" > "$OUT/candidates.jsonl"
# 2. Selection pool: the starting policy, 4 samples per candidate at the rollout temperature.
if [ ! -f "$ROLL/select/summary.json" ]; then
  step serve-start "$CODE/scripts/autodl/serve_vllm.sh" "$POLICY" policy-base \
    --enable-lora --max-lora-rank 32 --lora-modules "policy-start=$START"
  step select "$CODE/scripts/autodl/run_pool.sh" --config "$CONFIG" --tasks "$OUT/candidates.jsonl" \
    --split train --samples 4 --workers 32 --output "$ROLL/select" --resume \
    --set model=policy-start --set temperature=1.0 --set revision_labels=relabel
fi
# 3. Training set: the candidates solved at least once; steps keep about 7 visits per task.
"$PY" "$CODE/scripts/select_tasks.py" "$OUT/candidates.jsonl" --solved-in "$ROLL/select/attempts.jsonl" > "$OUT/tasks.jsonl"
N=$(wc -l < "$OUT/tasks.jsonl")
STEPS=$(( (7 * N + 7) / 8 ))
if [ "$STEPS" -gt 29 ]; then STEPS=29; fi
echo "selected $N of $(wc -l < "$OUT/candidates.jsonl") candidates; $STEPS steps"
if [ "$N" -lt 8 ]; then
  echo "fewer than 8 tasks selected: no training (protocol)"
  "$PY" "$CODE/scripts/pool_behaviour.py" "$ROLL/select" > "$OUT/behaviour.json"
  PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/summarize_round.py" --rollouts "$ROLL" --outputs "$OUT" > /dev/null
  echo "== round 10 done $(date -u +%FT%TZ)"
  exit 0
fi

# 4. GRPO from the starting adapter on the new tasks.
if [ ! -f "$OUT/grpo/final.json" ]; then
  step serve-policy env GPU_UTIL=0.45 "$CODE/scripts/autodl/serve_vllm.sh" "$POLICY" policy-base \
    --enable-lora --max-lora-rank 32 --max-loras 2
  step grpo env PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/grpo_harness.py" --base-model "$POLICY" \
    --tasks "$OUT/tasks.jsonl" --config "$CONFIG" --output "$OUT/grpo" --rollouts "$ROLL/grpo" \
    --steps "$STEPS" --max-minutes 240 --tasks-per-step 8 --group 8 --workers 32 \
    --turns-per-attempt 4 --minibatches 4 --lr 2e-5 --max-len 32768 --seed 3 --pack-turns --pipeline \
    --init-adapter "$START" --halt-invalid-share 0.04 --halt-patience 2 --rollout-set "revision_labels=relabel"
fi
FINAL=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["adapter"])' "$OUT/grpo/final.json")
echo "final adapter: $FINAL"

step serve-final "$CODE/scripts/autodl/serve_vllm.sh" "$POLICY" policy-base \
  --enable-lora --max-lora-rank 32 --lora-modules "policy-final=$FINAL"
if [ ! -f "$ROLL/eval-grpo/summary.json" ]; then
  step eval-grpo "$CODE/scripts/autodl/run_pool.sh" --config "$CONFIG" --tasks "$AGENT/swesmith/eval.jsonl" \
    --split eval --samples 8 --workers 32 --output "$ROLL/eval-grpo" --resume \
    --set model=policy-final --set revision_labels=relabel
fi
step stop-vllm "$CODE/scripts/autodl/stop_vllm.sh"

compare() {  # compare <name> <baseline attempts> [flags]
  local name=$1 base=$2; shift 2
  PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/compare_pools.py" --baseline "$base" \
    --candidate "$ROLL/eval-grpo/attempts.jsonl" --split eval "$@" > "$OUT/compare_$name.json"
}
compare eval "$BASELINE"
compare eval_drop_flagged "$BASELINE" --drop-flagged
compare round5 "$ROUND5"
compare sft "$SFT"
# Pre-registered descriptive checks (CPU only): the pre-bug source, hidden behaviour checks, limit
# events, and behaviour of the selection pool, every training step and the evaluation.
"$PY" "$CODE/scripts/review_patches.py" "$ROLL/eval-grpo" --original "$AGENT/swesmith-src" \
  --tasks "$AGENT/swesmith/tasks" > "$OUT/original_check.txt"
"$PY" "$CODE/scripts/regrade_hidden.py" "$ROLL/eval-grpo" --manifest "$AGENT/swesmith/eval.jsonl" \
  --original "$AGENT/swesmith-src" --as-nobody > "$OUT/hidden_regrade.json"
"$PY" "$CODE/scripts/limit_events.py" "$ROLL/eval-grpo" "$ROLL"/grpo/step-*[0-9] > "$OUT/limit_events.jsonl"
"$PY" "$CODE/scripts/pool_behaviour.py" "$ROLL/eval-grpo" "$ROLL/select" "$ROLL"/grpo/step-[0-9][0-9][0-9][0-9] \
  > "$OUT/behaviour.json"
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/summarize_round.py" --rollouts "$ROLL" --outputs "$OUT" > /dev/null
echo "== round 10 done $(date -u +%FT%TZ)"
