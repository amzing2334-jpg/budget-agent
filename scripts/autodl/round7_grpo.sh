#!/usr/bin/env bash
# Round-7 GRPO (docs/EXPERIMENT_GRPO_ROUND7.md), run on the GPU host as root: continue
# round 5's run from its last adapter (v29) for another 180 minutes with round 5's
# settings (prefix packing, relabelling) and a new seed, then evaluate the last adapter on
# the held-out split with 8 samples per task and compare it with round 5's evaluation pool
# of v29 (same harness, config and sample count). Hyperparameters come from the frozen
# protocol through the environment.
# Refuses to start if the GPU already runs a process this script did not start.
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl
AGENT=/opt/coding-agent
CODE=$AGENT/budget-agent
POLICY=$ML/outputs/round1/sft/merged
INIT=$ML/outputs/round5/grpo/adapters/v29     # round 5's last adapter
ROLL=$AGENT/work/round7               # rollout outputs (written by nobody)
OUT=$ML/outputs/round7                # adapters, metrics, comparisons (root)
CONFIG=$CODE/configs/swesmith-vllm-tokens.toml
PY=$ML/env/bin/python
: "${STEPS:?set by the protocol}" "${MAX_MINUTES:?}" "${LR:?}" "${TURNS_PER_ATTEMPT:?}" "${MINIBATCHES:?}" "${SEED:?}"
BASELINE=$AGENT/work/round5/eval-grpo/attempts.jsonl    # v29, 8 samples per task
SFT=$AGENT/work/round4/relabel/attempts.jsonl            # the SFT student, 8 samples per task
for path in "$INIT/adapter_config.json" "$BASELINE" "$SFT"; do [ -s "$path" ] || { echo "missing: $path" >&2; exit 1; }; done
mkdir -p "$ROLL/grpo" "$OUT"
chown nobody:nogroup "$ROLL" "$ROLL/grpo"
STEPS_LOG=$OUT/steps.jsonl

if pgrep -f "^bash $CODE/scripts/autodl/(round1_rft|round2_grpo|round3_grpo|round4_labels_ab|round5_grpo|round6_labels_sft|round7_grpo|round8_grpo|round9_eval|round10_fresh_tasks|grpo_pilot|bench|replicate_eval)\.sh" | grep -qvx "$$"; then
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
echo "init=$INIT seed=$SEED steps<=$STEPS minutes<=$MAX_MINUTES baseline=$BASELINE"

step() {  # step <name> <command...>: run, time and log one stage
  local name=$1; shift
  local started=$(date +%s)
  echo "== $name $(date -u +%FT%TZ)"
  "$@"
  printf '{"step": "%s", "seconds": %d, "finished": "%s"}\n' "$name" $(( $(date +%s) - started )) "$(date -u +%FT%TZ)" >> "$STEPS_LOG"
}

# Training tasks: train-split tasks the teacher solved at least once in round 1 (as in rounds 2 to 5).
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
    --turns-per-attempt "$TURNS_PER_ATTEMPT" --minibatches "$MINIBATCHES" --lr "$LR" --max-len 32768 \
    --seed "$SEED" --pack-turns --init-adapter "$INIT" --rollout-set "revision_labels=relabel"
  ls -d "$OUT/grpo/adapters"/v* | sort -V | tail -1 > "$OUT/final_adapter.txt"
fi
FINAL=$(cat "$OUT/final_adapter.txt")
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
compare sft "$SFT"
# Pre-registered descriptive checks (CPU only): the pre-bug source, hidden behaviour checks, limit events.
"$PY" "$CODE/scripts/review_patches.py" "$ROLL/eval-grpo" --original "$AGENT/swesmith-src" \
  --tasks "$AGENT/swesmith/tasks" > "$OUT/original_check.txt"
"$PY" "$CODE/scripts/regrade_hidden.py" "$ROLL/eval-grpo" --manifest "$AGENT/swesmith/eval.jsonl" \
  --original "$AGENT/swesmith-src" --as-nobody > "$OUT/hidden_regrade.json"
"$PY" "$CODE/scripts/limit_events.py" "$ROLL/eval-grpo" "$ROLL"/grpo/step-*[0-9] > "$OUT/limit_events.jsonl"
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/summarize_round.py" --rollouts "$ROLL" --outputs "$OUT" > /dev/null
echo "== round 7 done $(date -u +%FT%TZ)"
