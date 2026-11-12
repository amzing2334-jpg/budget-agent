#!/usr/bin/env bash
# GRPO pilot (docs/EXPERIMENT_GRPO_PILOT.md), run on the GPU host as root.
# vLLM serves the round-1 SFT student with runtime LoRA loading on 45% of the
# GPU while grpo_harness.py trains the adapter on the rest; rollouts run as
# `nobody` through run_pool.sh. Refuses to start if the GPU already runs a
# process this script did not start.
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl
AGENT=/opt/coding-agent
CODE=$AGENT/budget-agent
POLICY=${POLICY:-$ML/outputs/round1/sft/merged}
ROLL=$AGENT/work/grpo-pilot          # rollout outputs (written by nobody)
OUT=$ML/outputs/grpo-pilot           # adapters, metrics (root)
PY=$ML/env/bin/python
STEPS=${STEPS:-3}
mkdir -p "$ROLL" "$OUT"
chown nobody:nogroup "$ROLL"

if pgrep -f "^bash $CODE/scripts/autodl/(round1_rft|round2_grpo|round3_grpo|round4_labels_ab|round5_grpo|round6_labels_sft|round7_grpo|round8_grpo|round9_eval|round10_fresh_tasks|grpo_pilot|bench|replicate_eval)\.sh" | grep -qvx "$$"; then
  echo "another experiment script of this project is running; not starting." >&2   # it would lose its vLLM
  exit 1
fi
"$CODE/scripts/autodl/stop_vllm.sh"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "GPU already runs another process; not starting." >&2
  exit 1
fi

# Training tasks: train-split tasks the teacher solved at least once in round 1,
# with repo and verifier paths made absolute (the manifest moves directories).
"$PY" - "$AGENT/work/round1/teacher-train/attempts.jsonl" "$AGENT/swesmith/train.jsonl" "$OUT/tasks.jsonl" <<'EOF'
import json, os, sys
attempts, manifest, output = sys.argv[1:]
solved = {row["task_id"] for row in map(json.loads, open(attempts)) if row["resolved"]}
base = os.path.dirname(manifest)
rows = []
for row in map(json.loads, open(manifest)):
    if row["id"] in solved:
        rows.append({**row, "repo": os.path.join(base, row["repo"]), "verifier": os.path.join(base, row["verifier"])})
with open(output, "w") as stream:
    stream.writelines(json.dumps(row) + "\n" for row in rows)
print(f"{len(rows)} train tasks the teacher solved at least once")
EOF

GPU_UTIL=0.45 "$CODE/scripts/autodl/serve_vllm.sh" "$POLICY" policy-base --enable-lora --max-lora-rank 32 --max-loras 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$OUT/gpu_memory_mib.log" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true; "$CODE/scripts/autodl/stop_vllm.sh"' EXIT  # free the GPU on failure too
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/grpo_harness.py" --base-model "$POLICY" --tasks "$OUT/tasks.jsonl" \
  --config "$CODE/configs/swesmith-vllm.toml" --output "$OUT" --rollouts "$ROLL" \
  --steps "$STEPS" --tasks-per-step 8 --group 8 --workers 32
echo "== grpo pilot done $(date -u +%FT%TZ); peak GPU memory $(sort -n "$OUT/gpu_memory_mib.log" | tail -1) MiB"
