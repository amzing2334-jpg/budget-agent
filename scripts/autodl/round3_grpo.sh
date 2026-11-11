#!/usr/bin/env bash
# Round-3 GRPO (docs/EXPERIMENT_GRPO_ROUND3.md), run on the GPU host as root,
# on the hardened harness: per-task test limits, a per-command memory limit
# and a context budget counted in the served model's own tokens
# (configs/swesmith-vllm-tokens.toml, migrated manifests). Trains a fresh LoRA
# on the round-1 SFT student, then evaluates the SFT student and the last
# adapter on the held-out split under this same harness and compares them.
# Hyperparameters come from the frozen protocol through the environment.
# Refuses to start if the GPU already runs a process this script did not start.
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl
AGENT=/opt/coding-agent
CODE=$AGENT/budget-agent
POLICY=$ML/outputs/round1/sft/merged
ROLL=$AGENT/work/round3               # rollout outputs (written by nobody)
OUT=$ML/outputs/round3                # adapters, metrics, comparisons (root)
CONFIG=$CODE/configs/swesmith-vllm-tokens.toml
PY=$ML/env/bin/python
: "${STEPS:?set by the protocol}" "${MAX_MINUTES:?}" "${LR:?}" "${TURNS_PER_ATTEMPT:?}" "${MINIBATCHES:?}"
mkdir -p "$ROLL/grpo" "$OUT"
chown nobody:nogroup "$ROLL" "$ROLL/grpo"
STEPS_LOG=$OUT/steps.jsonl

if pgrep -f "^bash $CODE/scripts/autodl/(round1_rft|round2_grpo|round3_grpo|round4_labels_ab|round5_grpo|round6_labels_sft|grpo_pilot|bench|replicate_eval)\.sh" | grep -qvx "$$"; then
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
pool() {  # pool <name> <served-model>
  [ -f "$ROLL/$1/summary.json" ] && { echo "skip $1 (done)"; return; }
  "$CODE/scripts/autodl/run_pool.sh" --config "$CONFIG" --tasks "$AGENT/swesmith/eval.jsonl" --split eval \
    --samples 4 --workers 32 --output "$ROLL/$1" --resume --set "model=$2"
}

# Training tasks: train-split tasks the teacher solved at least once in round 1 (as in round 2).
"$PY" - "$AGENT/work/round1/teacher-train/attempts.jsonl" "$AGENT/swesmith/train.jsonl" "$OUT/tasks.jsonl" <<'EOF'
import json, os, sys
attempts, manifest, output = sys.argv[1:]
solved = {row["task_id"] for row in map(json.loads, open(attempts)) if row["resolved"]}
base = os.path.dirname(manifest)
rows = [{**row, "repo": os.path.join(base, row["repo"]), "verifier": os.path.join(base, row["verifier"])}
        for row in map(json.loads, open(manifest)) if row["id"] in solved]
with open(output, "w") as stream:
    stream.writelines(json.dumps(row) + "\n" for row in rows)
print(f"{len(rows)} train tasks the teacher solved at least once")
EOF

if [ ! -f "$OUT/final_adapter.txt" ]; then
  step serve-policy env GPU_UTIL=0.45 "$CODE/scripts/autodl/serve_vllm.sh" "$POLICY" policy-base \
    --enable-lora --max-lora-rank 32 --max-loras 2
  step grpo env PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/grpo_harness.py" --base-model "$POLICY" \
    --tasks "$OUT/tasks.jsonl" --config "$CONFIG" --output "$OUT/grpo" --rollouts "$ROLL/grpo" \
    --steps "$STEPS" --max-minutes "$MAX_MINUTES" --tasks-per-step 8 --group 8 --workers 32 \
    --turns-per-attempt "$TURNS_PER_ATTEMPT" --minibatches "$MINIBATCHES" --lr "$LR" --max-len 32768
  ls -d "$OUT/grpo/adapters"/v* | sort -V | tail -1 > "$OUT/final_adapter.txt"
fi
FINAL=$(cat "$OUT/final_adapter.txt")
echo "final adapter: $FINAL"

step serve-sft "$CODE/scripts/autodl/serve_vllm.sh" "$POLICY" student-sft
step eval-sft pool eval-sft student-sft
step serve-final "$CODE/scripts/autodl/serve_vllm.sh" "$POLICY" policy-base \
  --enable-lora --max-lora-rank 32 --lora-modules "policy-final=$FINAL"
step eval-grpo pool eval-grpo policy-final
step stop-vllm "$CODE/scripts/autodl/stop_vllm.sh"

compare() {  # compare <name> <baseline attempts> <candidate attempts> [flags]
  local name=$1 base=$2 cand=$3; shift 3
  PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/compare_pools.py" --baseline "$base" --candidate "$cand" \
    --split eval "$@" > "$OUT/compare_$name.json"
}
compare eval "$ROLL/eval-sft/attempts.jsonl" "$ROLL/eval-grpo/attempts.jsonl"
compare eval_drop_flagged "$ROLL/eval-sft/attempts.jsonl" "$ROLL/eval-grpo/attempts.jsonl" --drop-flagged
compare harness_sft "$AGENT/work/round1/student-sft-eval/attempts.jsonl" "$ROLL/eval-sft/attempts.jsonl"
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/summarize_round.py" --rollouts "$ROLL" --outputs "$OUT" > /dev/null
echo "== round 3 done $(date -u +%FT%TZ)"
