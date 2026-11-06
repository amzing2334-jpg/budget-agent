#!/usr/bin/env bash
# Serve a model on 127.0.0.1:18017 (never exposed) and wait until it answers.
# Usage: serve_vllm.sh <model-dir> <served-name> [extra vllm args...]
# Only the server started by this script (pidfile) is ever stopped; other
# projects on the shared instance are left alone. GPU_UTIL lowers vLLM's
# memory share when a trainer shares the GPU (GRPO: GPU_UTIL=0.45 plus
# --enable-lora --max-lora-rank 32).
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl
PIDFILE=$ML/logs/vllm.pid
model=$1; name=$2; shift 2
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")"; while kill -0 "$(cat "$PIDFILE")" 2>/dev/null; do sleep 2; done
fi
if curl -sf http://127.0.0.1:18017/v1/models > /dev/null; then
  echo "port 18017 is served by a process this script did not start; refusing"; exit 1
fi
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True nohup "$ML/env/bin/vllm" serve "$model" --served-model-name "$name" --host 127.0.0.1 --port 18017 \
  --max-model-len 32768 --gpu-memory-utilization "${GPU_UTIL:-0.88}" --enable-prefix-caching --max-num-seqs 64 \
  "$@" > "$ML/logs/vllm-$name.log" 2>&1 &
echo $! > "$PIDFILE"
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:18017/v1/models > /dev/null && { echo "vllm ready: $name after $((i * 5))s"; exit 0; }
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break
  sleep 5
done
echo "vllm failed to start; tail:"; tail -30 "$ML/logs/vllm-$name.log"; exit 1
