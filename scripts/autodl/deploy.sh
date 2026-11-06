#!/usr/bin/env bash
# Deploy the current `main` and the SWE-smith data to a shared AutoDL instance.
# Run from the controller machine. Remote changes stay inside /opt/coding-agent
# and /root/autodl-tmp/coding-agent-rl; other projects, the base conda env and
# SSH settings are never touched. Safe to re-run.
#
#   scripts/autodl/deploy.sh <ssh-port> <ssh-host> <swesmith-data.tar.gz>
#
# PIP_INDEX_URL is passed to the remote setup. Measure mirrors first: on the
# Foshan A100 (2026-11-06) the image default (Aliyun) gave 0.35 MB/s, Huawei
# Cloud 10.7 MB/s, files.pythonhosted.org 7.4 MB/s, academic acceleration 0.28 MB/s.
set -euo pipefail
PORT=$1; HOST=$2; DATA=$3
KEY=${KEY:-$HOME/.ssh/autodl_budget_agent_ed25519}
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -p "$PORT" "$HOST")
SCP=(scp -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -P "$PORT")
HERE=$(cd "$(dirname "$0")/../.." && pwd)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGS=${LOGS:-$HERE/runs/deploy-$STAMP}
mkdir -p "$LOGS"

echo "== recon (read-only)"
"${SSH[@]}" 'bash -s' < "$HERE/scripts/autodl/recon.sh" > "$LOGS/recon.txt" 2>&1 || true
if "${SSH[@]}" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader' | grep -q .; then
  echo "GPU already has running processes; deploying files only, no GPU work started." | tee "$LOGS/gpu-busy.txt"
fi

REF=${REF:-main}
echo "== ship code ($REF @ $(git -C "$HERE" rev-parse --short "$REF")) and data"
git -C "$HERE" archive --format=tar.gz -o "$LOGS/budget-agent.tar.gz" "$REF"
"${SSH[@]}" 'mkdir -p /opt/coding-agent/incoming /root/autodl-tmp/coding-agent-rl/logs'
"${SCP[@]}" "$LOGS/budget-agent.tar.gz" "$DATA" "$HOST:/opt/coding-agent/incoming/"
"${SSH[@]}" "set -e; cd /opt/coding-agent
  rm -rf budget-agent.new && mkdir budget-agent.new && tar -xzf incoming/budget-agent.tar.gz -C budget-agent.new
  rm -rf budget-agent && mv budget-agent.new budget-agent
  mkdir -p swesmith-src && tar -xzf incoming/$(basename "$DATA") -C swesmith-src
  chmod -R a+rX /opt/coding-agent"

echo "== isolated environments (background on the instance)"
"${SSH[@]}" "PIP_INDEX_URL='${PIP_INDEX_URL:-}' setsid nohup bash /opt/coding-agent/budget-agent/scripts/autodl/setup_isolated.sh \
  > /root/autodl-tmp/coding-agent-rl/logs/setup-$STAMP.log 2>&1 < /dev/null &"
echo "setup started; follow /root/autodl-tmp/coding-agent-rl/logs/setup-$STAMP.log"
echo "recon saved to $LOGS/recon.txt"
