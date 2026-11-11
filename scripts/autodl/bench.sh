#!/usr/bin/env bash
# Rollout throughput versus concurrency (scripts/bench_rollout.py) with the 7B
# teacher on the full GPU, run on the GPU host as root. The same 64 train tasks
# run once per worker count, in increasing order, so later settings can reuse
# prefix-cache entries from earlier ones (the shared system prompt dominates).
# Refuses to start if the GPU already runs a process this script did not start.
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl
AGENT=/opt/coding-agent
CODE=$AGENT/budget-agent
OUT=$AGENT/work/bench                # attempts written by nobody; bench.json/md by root
mkdir -p "$OUT"
chown nobody:nogroup "$OUT"

if pgrep -f "^bash $CODE/scripts/autodl/(round1_rft|round2_grpo|round3_grpo|round4_labels_ab|round5_grpo|round6_labels_sft|grpo_pilot|bench|replicate_eval)\.sh" | grep -qvx "$$"; then
  echo "another experiment script of this project is running; not starting." >&2   # it would lose its vLLM
  exit 1
fi
"$CODE/scripts/autodl/stop_vllm.sh"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "GPU already runs another process; not starting." >&2
  exit 1
fi
trap '"$CODE/scripts/autodl/stop_vllm.sh"' EXIT
"$CODE/scripts/autodl/serve_vllm.sh" "$ML/models/Qwen2.5-Coder-7B-Instruct" teacher
PYTHONPATH="$CODE/src" "$ML/env/bin/python" "$CODE/scripts/bench_rollout.py" --config "$CODE/configs/swesmith-vllm.toml" \
  --tasks "$AGENT/swesmith/train.jsonl" --limit 64 --workers 8 16 32 64 --output "$OUT"
echo "== bench done $(date -u +%FT%TZ)"
