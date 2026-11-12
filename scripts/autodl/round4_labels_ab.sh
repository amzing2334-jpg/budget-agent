#!/usr/bin/env bash
# Round 4 (docs/EXPERIMENT_LABELS_AB.md), run on the GPU host as root: the round-1
# SFT student on the held-out split under the hardened harness, once with
# revision labels relabelled (the default) and once fixed at recording time,
# 8 samples per task each. Each pool gets a fresh vLLM (cold prefix cache) and
# vLLM counters around it. Rollouts run as `nobody`.
# Refuses to start if the GPU already runs a process this script did not start.
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl
AGENT=/opt/coding-agent
CODE=$AGENT/budget-agent
POLICY=$ML/outputs/round1/sft/merged
ROLL=$AGENT/work/round4                # rollout outputs (written by nobody)
OUT=$ML/outputs/round4                 # counters, comparisons, packing (root)
CONFIG=$CODE/configs/swesmith-vllm-tokens.toml
PY=$ML/env/bin/python
mkdir -p "$ROLL" "$OUT"
chown nobody:nogroup "$ROLL"
STEPS_LOG=$OUT/steps.jsonl

if pgrep -f "^bash $CODE/scripts/autodl/(round1_rft|round2_grpo|round3_grpo|round4_labels_ab|round5_grpo|round6_labels_sft|round7_grpo|grpo_pilot|bench|replicate_eval)\.sh" | grep -qvx "$$"; then
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
pool() {  # pool <name> [--set key=value ...]: 8 samples per held-out task with counters around the pool
  local name=$1; shift
  [ -f "$ROLL/$name/summary.json" ] && { echo "skip $name (done)"; return; }
  "$PY" "$CODE/scripts/vllm_metrics.py" > "$OUT/vllm_before_$name.json"
  "$CODE/scripts/autodl/run_pool.sh" --config "$CONFIG" --tasks "$AGENT/swesmith/eval.jsonl" --split eval \
    --samples 8 --workers 32 --output "$ROLL/$name" --resume --set model=student-sft --set api_logprobs=true "$@"
  "$PY" "$CODE/scripts/vllm_metrics.py" > "$OUT/vllm_after_$name.json"
}

step serve-relabel "$CODE/scripts/autodl/serve_vllm.sh" "$POLICY" student-sft
step relabel pool relabel
step serve-fixed "$CODE/scripts/autodl/serve_vllm.sh" "$POLICY" student-sft   # a fresh server: cold prefix cache
step fixed pool fixed --set revision_labels=fixed
step stop-vllm "$CODE/scripts/autodl/stop_vllm.sh"

compare() {  # compare <name> [flags]
  local name=$1; shift
  PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/compare_pools.py" --baseline "$ROLL/relabel/attempts.jsonl" \
    --candidate "$ROLL/fixed/attempts.jsonl" --split eval "$@" > "$OUT/compare_$name.json"
}
compare eval
compare eval_drop_flagged --drop-flagged
for name in relabel fixed; do
  "$PY" "$CODE/scripts/vllm_metrics.py" --diff "$OUT/vllm_before_$name.json" "$OUT/vllm_after_$name.json" > "$OUT/vllm_$name.json"
  PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/bench_turn_packing.py" --tokenizer "$POLICY" "$ROLL/$name" \
    2>/dev/null > "$OUT/packing_$name.json"
done
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/summarize_round.py" --rollouts "$ROLL" --outputs "$OUT" > /dev/null
echo "== round 4 done $(date -u +%FT%TZ)"
