#!/usr/bin/env bash
# Round-9 evaluation (docs/EXPERIMENT_GRPO_ROUND9.md), run on the GPU host as root: the round-7
# adapter the rollout-health guard would have kept (invalid-action share above 4% for two
# consecutive batches stops the run; v26 is the last policy at or below it), on the held-out
# split with 8 samples per task against round 5's evaluation pool. No training.
# Refuses to start if the GPU already runs a process this script did not start.
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl
AGENT=/opt/coding-agent
CODE=$AGENT/budget-agent
POLICY=$ML/outputs/round1/sft/merged
ADAPTER=$ML/outputs/round7/grpo/adapters/v26
ROLL=$AGENT/work/round9               # rollout outputs (written by nobody)
OUT=$ML/outputs/round9                # comparisons and reports (root)
CONFIG=$CODE/configs/swesmith-vllm-tokens.toml
PY=$ML/env/bin/python
BASELINE=$AGENT/work/round5/eval-grpo/attempts.jsonl    # round 5's v29, 8 samples per task
CONTINUED=$AGENT/work/round7/eval-grpo/attempts.jsonl   # round 7's last adapter, v46
SFT=$AGENT/work/round4/relabel/attempts.jsonl            # the SFT student, 8 samples per task
for path in "$ADAPTER/adapter_config.json" "$BASELINE" "$CONTINUED" "$SFT"; do
  [ -s "$path" ] || { echo "missing: $path" >&2; exit 1; }
done
mkdir -p "$ROLL" "$OUT"
chown nobody:nogroup "$ROLL"
STEPS_LOG=$OUT/steps.jsonl

if pgrep -f "^bash $CODE/scripts/autodl/(round1_rft|round2_grpo|round3_grpo|round4_labels_ab|round5_grpo|round6_labels_sft|round7_grpo|round8_grpo|round9_eval|round10_fresh_tasks|round11_seed|round12_3b|grpo_pilot|bench|replicate_eval)\.sh" | grep -qvx "$$"; then
  echo "another experiment script of this project is running; not starting." >&2   # it would lose its vLLM
  exit 1
fi
grep -q test_timeout_seconds "$AGENT/swesmith/eval.jsonl" || { echo "manifests not migrated (add_test_timeouts.py)" >&2; exit 1; }

# The guard's choice, recomputed from round 7's per-step rollouts before anything runs.
PYTHONPATH="$CODE/src" "$PY" - "$AGENT/work/round7/grpo" <<'PYEOF'
import json, sys
from pathlib import Path
from budget_agent.learning.grpo_core import HealthGuard, rollout_health
steps = sorted(p for p in Path(sys.argv[1]).glob("step-[0-9][0-9][0-9][0-9]") if p.is_dir())
guard = HealthGuard(0.04, patience=2)
stopped = None
for step, pool in enumerate(steps):
    attempts = [json.loads(line) for line in (pool / "attempts.jsonl").read_text().splitlines() if line.strip()]
    if guard.observe(step, rollout_health(attempts)["invalid_action_share"]):
        stopped = step
        break
print(f"guard over {len(steps)} round-7 steps: stops at step {stopped}, keeps v{guard.healthy_policy}")
sys.exit(0 if (stopped, guard.healthy_policy) == (28, 26) else 1)
PYEOF

"$CODE/scripts/autodl/stop_vllm.sh"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "GPU already runs another process; not starting." >&2
  exit 1
fi
trap '"$CODE/scripts/autodl/stop_vllm.sh"' EXIT
echo "adapter=$ADAPTER baseline=$BASELINE"

step() {  # step <name> <command...>: run, time and log one stage
  local name=$1; shift
  local started=$(date +%s)
  echo "== $name $(date -u +%FT%TZ)"
  "$@"
  printf '{"step": "%s", "seconds": %d, "finished": "%s"}\n' "$name" $(( $(date +%s) - started )) "$(date -u +%FT%TZ)" >> "$STEPS_LOG"
}

step serve-final "$CODE/scripts/autodl/serve_vllm.sh" "$POLICY" policy-base \
  --enable-lora --max-lora-rank 32 --lora-modules "policy-final=$ADAPTER"
if [ ! -f "$ROLL/eval-v26/summary.json" ]; then
  step eval-v26 "$CODE/scripts/autodl/run_pool.sh" --config "$CONFIG" --tasks "$AGENT/swesmith/eval.jsonl" \
    --split eval --samples 8 --workers 32 --output "$ROLL/eval-v26" --resume \
    --set model=policy-final --set revision_labels=relabel
fi
step stop-vllm "$CODE/scripts/autodl/stop_vllm.sh"

compare() {  # compare <name> <baseline attempts> [flags]
  local name=$1 base=$2; shift 2
  PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/compare_pools.py" --baseline "$base" \
    --candidate "$ROLL/eval-v26/attempts.jsonl" --split eval "$@" > "$OUT/compare_$name.json"
}
compare eval "$BASELINE"
compare eval_drop_flagged "$BASELINE" --drop-flagged
compare continued "$CONTINUED"
compare sft "$SFT"
# Pre-registered descriptive checks (CPU only): the pre-bug source, hidden behaviour checks, limit
# events and behaviour (invalid actions, output tokens, statuses, tools).
"$PY" "$CODE/scripts/review_patches.py" "$ROLL/eval-v26" --original "$AGENT/swesmith-src" \
  --tasks "$AGENT/swesmith/tasks" > "$OUT/original_check.txt"
"$PY" "$CODE/scripts/regrade_hidden.py" "$ROLL/eval-v26" --manifest "$AGENT/swesmith/eval.jsonl" \
  --original "$AGENT/swesmith-src" --as-nobody > "$OUT/hidden_regrade.json"
"$PY" "$CODE/scripts/limit_events.py" "$ROLL/eval-v26" > "$OUT/limit_events.jsonl"
"$PY" "$CODE/scripts/pool_behaviour.py" "$ROLL/eval-v26" > "$OUT/behaviour.json"
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/summarize_round.py" --rollouts "$ROLL" --outputs "$OUT" > /dev/null
echo "== round 9 done $(date -u +%FT%TZ)"
