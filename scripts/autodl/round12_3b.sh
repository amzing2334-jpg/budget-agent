#!/usr/bin/env bash
# Round-12 (docs/EXPERIMENT_GRPO_ROUND12.md), run on the GPU host as root: the round-1 SFT and
# round-8 GRPO recipes with a 3B student. SFT on round 1's SFT data with round 1's settings;
# evaluation of the 3B SFT student; GRPO from it (pipelined, 29 steps, rollout-health guard,
# adapter retention); evaluation of the adapter final.json names; comparisons against the 1.5B
# runs. Hyperparameters come from the frozen protocol through the environment.
# Refuses to start if the GPU already runs a process this script did not start.
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl
AGENT=/opt/coding-agent
CODE=$AGENT/budget-agent
BASE=$ML/models/Qwen2.5-Coder-3B-Instruct
DATA=$ML/outputs/round1/sft.jsonl     # round 1's SFT data (the 7B teacher's passing turns)
ROLL=$AGENT/work/round12              # rollout outputs (written by nobody)
OUT=$ML/outputs/round12               # SFT student, adapters, metrics, comparisons (root)
CONFIG=$CODE/configs/swesmith-vllm-tokens.toml
PY=$ML/env/bin/python
: "${STEPS:?set by the protocol}" "${MAX_MINUTES:?}" "${LR:?}" "${TURNS_PER_ATTEMPT:?}" "${MINIBATCHES:?}" "${SEED:?}"
GRPO15=$AGENT/work/round8/eval-grpo/attempts.jsonl      # 1.5B, the same GRPO recipe (primary)
SFT15=$AGENT/work/round4/relabel/attempts.jsonl         # 1.5B SFT student
for path in "$BASE/config.json" "$DATA" "$GRPO15" "$SFT15"; do [ -s "$path" ] || { echo "missing: $path" >&2; exit 1; }; done
mkdir -p "$ROLL/grpo" "$OUT"
chown nobody:nogroup "$ROLL" "$ROLL/grpo"
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
echo "student=$BASE seed=$SEED steps<=$STEPS minutes<=$MAX_MINUTES"

step() {  # step <name> <command...>: run, time and log one stage
  local name=$1; shift
  local started=$(date +%s)
  echo "== $name $(date -u +%FT%TZ)"
  "$@"
  printf '{"step": "%s", "seconds": %d, "finished": "%s"}\n' "$name" $(( $(date +%s) - started )) "$(date -u +%FT%TZ)" >> "$STEPS_LOG"
}

# 1. SFT with round 1's settings (the trainer's defaults: 2 epochs, lr 1e-4, rank 64, 16,384 tokens).
[ -f "$OUT/sft/result.json" ] || step train-sft "$PY" "$CODE/scripts/train_sft_lora.py" --model "$BASE" \
  --data "$DATA" --output "$OUT/sft" --merge
SFT=$OUT/sft/merged

# 2. The 3B SFT student on the held-out split, 8 samples per task.
if [ ! -f "$ROLL/eval-sft/summary.json" ]; then
  step serve-sft "$CODE/scripts/autodl/serve_vllm.sh" "$SFT" student-sft
  step eval-sft "$CODE/scripts/autodl/run_pool.sh" --config "$CONFIG" --tasks "$AGENT/swesmith/eval.jsonl" \
    --split eval --samples 8 --workers 32 --output "$ROLL/eval-sft" --resume \
    --set model=student-sft --set revision_labels=relabel
fi

# 3. Training tasks: train-split tasks the teacher solved at least once in round 1 (as in rounds 2 to 11).
"$PY" - "$AGENT/work/round1/teacher-train/attempts.jsonl" "$AGENT/swesmith/train.jsonl" "$OUT/tasks.jsonl" <<'PYEOF'
import json, os, sys
attempts, manifest, output = sys.argv[1:]
solved = {row["task_id"] for row in map(json.loads, open(attempts)) if row["resolved"]}
base = os.path.dirname(manifest)
rows = [{**row, "repo": os.path.join(base, row["repo"]), "verifier": os.path.join(base, row["verifier"])}
        for row in map(json.loads, open(manifest)) if row["id"] in solved]
with open(output, "w") as stream:
    stream.writelines(json.dumps(row) + "\n" for row in rows)
print(f"{len(rows)} train tasks the teacher solved at least once")
PYEOF

# 4. GRPO from the 3B SFT student with round 8's recipe.
if [ ! -f "$OUT/final_adapter.txt" ]; then
  # vLLM takes 0.43 of the card (0.45 for the 1.5B runs): the memory probe left the 3B trainer about
  # 1.5 GiB of headroom at 0.45 on the longest allowed sequence; expandable segments avoid the
  # fragmentation that ran it out of memory without them.
  step serve-policy env GPU_UTIL=0.43 "$CODE/scripts/autodl/serve_vllm.sh" "$SFT" policy-base \
    --enable-lora --max-lora-rank 32 --max-loras 2
  step grpo env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$CODE/src" "$PY" \
    "$CODE/scripts/grpo_harness.py" --base-model "$SFT" \
    --tasks "$OUT/tasks.jsonl" --config "$CONFIG" --output "$OUT/grpo" --rollouts "$ROLL/grpo" \
    --steps "$STEPS" --max-minutes "$MAX_MINUTES" --tasks-per-step 8 --group 8 --workers 32 \
    --turns-per-attempt "$TURNS_PER_ATTEMPT" --minibatches "$MINIBATCHES" --lr "$LR" --max-len 32768 \
    --seed "$SEED" --pack-turns --pipeline --halt-invalid-share 0.04 --halt-patience 2 --keep-adapters 3 \
    --rollout-set "revision_labels=relabel"
  "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["adapter"])' "$OUT/grpo/final.json" \
    > "$OUT/final_adapter.txt"
fi
FINAL=$(cat "$OUT/final_adapter.txt")
echo "final adapter: $FINAL"

# 5. The 3B GRPO policy on the held-out split, 8 samples per task.
step serve-final "$CODE/scripts/autodl/serve_vllm.sh" "$SFT" policy-base \
  --enable-lora --max-lora-rank 32 --lora-modules "policy-final=$FINAL"
if [ ! -f "$ROLL/eval-grpo/summary.json" ]; then
  step eval-grpo "$CODE/scripts/autodl/run_pool.sh" --config "$CONFIG" --tasks "$AGENT/swesmith/eval.jsonl" \
    --split eval --samples 8 --workers 32 --output "$ROLL/eval-grpo" --resume \
    --set model=policy-final --set revision_labels=relabel
fi
step stop-vllm "$CODE/scripts/autodl/stop_vllm.sh"

compare() {  # compare <name> <baseline attempts> <candidate attempts> [flags]
  local name=$1 base=$2 candidate=$3; shift 3
  PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/compare_pools.py" --baseline "$base" \
    --candidate "$candidate" --split eval "$@" > "$OUT/compare_$name.json"
}
compare eval "$GRPO15" "$ROLL/eval-grpo/attempts.jsonl"
compare eval_drop_flagged "$GRPO15" "$ROLL/eval-grpo/attempts.jsonl" --drop-flagged
compare sft_scale "$SFT15" "$ROLL/eval-sft/attempts.jsonl"
compare grpo_gain "$ROLL/eval-sft/attempts.jsonl" "$ROLL/eval-grpo/attempts.jsonl"
# Pre-registered descriptive checks (CPU only) for both 3B evaluations: the pre-bug source, hidden
# behaviour checks, limit events, and behaviour of the evaluations and of every training step.
for pool in eval-grpo eval-sft; do
  "$PY" "$CODE/scripts/review_patches.py" "$ROLL/$pool" --original "$AGENT/swesmith-src" \
    --tasks "$AGENT/swesmith/tasks" > "$OUT/original_check_$pool.txt"
  "$PY" "$CODE/scripts/regrade_hidden.py" "$ROLL/$pool" --manifest "$AGENT/swesmith/eval.jsonl" \
    --original "$AGENT/swesmith-src" --as-nobody > "$OUT/hidden_regrade_$pool.json"
done
"$PY" "$CODE/scripts/limit_events.py" "$ROLL/eval-grpo" "$ROLL/eval-sft" "$ROLL"/grpo/step-*[0-9] > "$OUT/limit_events.jsonl"
"$PY" "$CODE/scripts/pool_behaviour.py" "$ROLL/eval-grpo" "$ROLL/eval-sft" "$ROLL"/grpo/step-[0-9][0-9][0-9][0-9] \
  > "$OUT/behaviour.json"
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/summarize_round.py" --rollouts "$ROLL" --outputs "$OUT" > /dev/null
echo "== round 12 done $(date -u +%FT%TZ)"
