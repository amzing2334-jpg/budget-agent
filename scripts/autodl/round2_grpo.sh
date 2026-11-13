#!/usr/bin/env bash
# Round-2 GRPO (docs/EXPERIMENT_GRPO_ROUND2.md), run on the GPU host as root:
# GRPO on the round-1 SFT student for at most 30 steps or 120 minutes, then
# the last adapter on the held-out split exactly like round 1, compared with
# the round-1 SFT student. Rollouts run as `nobody` through run_pool.sh.
# Refuses to start if the GPU already runs a process this script did not start.
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl
AGENT=/opt/coding-agent
CODE=$AGENT/budget-agent
POLICY=$ML/outputs/round1/sft/merged
ROLL=$AGENT/work/round2              # rollout outputs (written by nobody)
OUT=$ML/outputs/round2               # adapters, metrics, comparisons (root)
CONFIG=$CODE/configs/swesmith-vllm.toml
PY=$ML/env/bin/python
mkdir -p "$ROLL/grpo" "$OUT"
chown nobody:nogroup "$ROLL" "$ROLL/grpo"
STEPS=$OUT/steps.jsonl

if pgrep -f "^bash $CODE/scripts/autodl/(round1_rft|round2_grpo|round3_grpo|round4_labels_ab|round5_grpo|round6_labels_sft|round7_grpo|round8_grpo|round9_eval|round10_fresh_tasks|round11_seed|round12_3b|grpo_pilot|bench|replicate_eval)\.sh" | grep -qvx "$$"; then
  echo "another experiment script of this project is running; not starting." >&2   # it would lose its vLLM
  exit 1
fi
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

# Training tasks: train-split tasks the teacher solved at least once in round 1 (as in the pilot).
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
    --steps 30 --max-minutes 120 --tasks-per-step 8 --group 8 --workers 32 --turns-per-attempt 6 --minibatches 4
  ls -d "$OUT/grpo/adapters"/v* | sort -V | tail -1 > "$OUT/final_adapter.txt"
fi
FINAL=$(cat "$OUT/final_adapter.txt")
echo "final adapter: $FINAL"

step serve-final "$CODE/scripts/autodl/serve_vllm.sh" "$POLICY" policy-base \
  --enable-lora --max-lora-rank 32 --lora-modules "policy-final=$FINAL"
step grpo-eval "$CODE/scripts/autodl/run_pool.sh" --config "$CONFIG" --tasks "$AGENT/swesmith/eval.jsonl" \
  --split eval --samples 4 --workers 32 --output "$ROLL/grpo-eval" --resume --set model=policy-final
step stop-vllm "$CODE/scripts/autodl/stop_vllm.sh"

BASE=$AGENT/work/round1/student-sft-eval/attempts.jsonl
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/compare_pools.py" --baseline "$BASE" \
  --candidate "$ROLL/grpo-eval/attempts.jsonl" --split eval > "$OUT/compare_eval.json"
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/compare_pools.py" --baseline "$BASE" \
  --candidate "$ROLL/grpo-eval/attempts.jsonl" --split eval --drop-flagged > "$OUT/compare_eval_drop_flagged.json"
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/summarize_round.py" --rollouts "$ROLL" --outputs "$OUT" > /dev/null
echo "== round 2 done $(date -u +%FT%TZ)"
