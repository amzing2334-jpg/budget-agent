#!/usr/bin/env bash
# Replication evaluation (docs/EXPERIMENT_REPLICATION.md), run on the GPU host
# as root: rerun the held-out evaluation of the round-1 SFT student and of the
# round-2 GRPO adapter under the exact round-1/round-2 configuration, then
# compare them (primary), each with its original pool (evaluation noise), and
# with the original and replicated samples pooled. Rollouts run as `nobody`.
# Refuses to start if the GPU already runs a process this script did not start.
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl
AGENT=/opt/coding-agent
CODE=$AGENT/budget-agent
POLICY=$ML/outputs/round1/sft/merged
ADAPTER=$ML/outputs/round2/grpo/adapters/v12
EVAL=$AGENT/swesmith/eval.round1.jsonl   # the manifest rounds 1 and 2 used (no per-task limits)
ROLL=$AGENT/work/replication             # rollout outputs (written by nobody)
OUT=$ML/outputs/replication              # comparisons, summary (root)
CONFIG=$CODE/configs/swesmith-vllm.toml
PY=$ML/env/bin/python
mkdir -p "$ROLL" "$OUT"
chown nobody:nogroup "$ROLL"
STEPS=$OUT/steps.jsonl

if pgrep -f "^bash $CODE/scripts/autodl/(round1_rft|round2_grpo|round3_grpo|round4_labels_ab|grpo_pilot|bench|replicate_eval)\.sh" | grep -qvx "$$"; then
  echo "another experiment script of this project is running; not starting." >&2   # it would lose its vLLM
  exit 1
fi
test -f "$EVAL" || { echo "missing $EVAL (copy of the unmigrated eval manifest)" >&2; exit 1; }
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
  printf '{"step": "%s", "seconds": %d, "finished": "%s"}\n' "$name" $(( $(date +%s) - started )) "$(date -u +%FT%TZ)" >> "$STEPS"
}
pool() {  # pool <name> <served-model>
  [ -f "$ROLL/$1/summary.json" ] && { echo "skip $1 (done)"; return; }
  "$CODE/scripts/autodl/run_pool.sh" --config "$CONFIG" --tasks "$EVAL" --split eval --samples 4 \
    --workers 32 --output "$ROLL/$1" --resume --set "model=$2"
}

step serve-sft "$CODE/scripts/autodl/serve_vllm.sh" "$POLICY" student-sft
step rep-sft pool rep-sft student-sft
step serve-grpo "$CODE/scripts/autodl/serve_vllm.sh" "$POLICY" policy-base \
  --enable-lora --max-lora-rank 32 --lora-modules "policy-final=$ADAPTER"
step rep-grpo pool rep-grpo policy-final
step stop-vllm "$CODE/scripts/autodl/stop_vllm.sh"

compare() {  # compare <name> <baseline attempts> <candidate attempts> [flags]
  local name=$1 base=$2 cand=$3; shift 3
  PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/compare_pools.py" --baseline "$base" --candidate "$cand" \
    --split eval "$@" > "$OUT/compare_$name.json"
}
SFT1=$AGENT/work/round1/student-sft-eval/attempts.jsonl
GRPO1=$AGENT/work/round2/grpo-eval/attempts.jsonl
compare eval "$ROLL/rep-sft/attempts.jsonl" "$ROLL/rep-grpo/attempts.jsonl"
compare eval_drop_flagged "$ROLL/rep-sft/attempts.jsonl" "$ROLL/rep-grpo/attempts.jsonl" --drop-flagged
compare noise_sft "$SFT1" "$ROLL/rep-sft/attempts.jsonl"
compare noise_grpo "$GRPO1" "$ROLL/rep-grpo/attempts.jsonl"
cat "$SFT1" "$ROLL/rep-sft/attempts.jsonl" > "$OUT/pooled_sft.jsonl"
cat "$GRPO1" "$ROLL/rep-grpo/attempts.jsonl" > "$OUT/pooled_grpo.jsonl"
compare pooled "$OUT/pooled_sft.jsonl" "$OUT/pooled_grpo.jsonl"
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/summarize_round.py" --rollouts "$ROLL" --outputs "$OUT" > /dev/null
echo "== replication done $(date -u +%FT%TZ)"
