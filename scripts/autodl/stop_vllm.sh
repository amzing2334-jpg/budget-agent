#!/usr/bin/env bash
# Stop only the vLLM server started by serve_vllm.sh (pidfile), freeing the GPU for training.
set -euo pipefail
PIDFILE=/root/autodl-tmp/coding-agent-rl/logs/vllm.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")"
  while kill -0 "$(cat "$PIDFILE")" 2>/dev/null; do sleep 2; done
  echo "vllm stopped"
fi
rm -f "$PIDFILE"
