#!/usr/bin/env bash
# chain_round.sh: start the next experiment round only after the previous one finished cleanly.
#
#   chain_round.sh --after round7_grpo --marker "== round 7 done" \
#     --archive /opt/coding-agent/work/round7 \
#     --deploy /opt/coding-agent/incoming/budget-agent-<commit>.tar.gz --tree <16-hex tree hash> \
#     --start round8_grpo --env STEPS=29 --env SEED=1 ...
#
# Waits while scripts/autodl/<after>.sh runs; requires <marker> in logs/<after-log>.log (the
# round number before the underscore, e.g. logs/round7.log); refuses if any experiment
# script or GPU process is still running; unpacks the tarball and checks its code-tree hash
# (src, scripts, configs); only then archives the given finished rollout directories and
# swaps the new tree in (the old one is kept as budget-agent.<after>); then starts scripts/autodl/<start>.sh
# with the given environment, logging to logs/<start-log>.log. Every step is logged, and it
# stops at the first surprise.
set -euo pipefail
AGENT=${AGENT:-/opt/coding-agent}
ML=${ML:-/root/autodl-tmp/coding-agent-rl}
POLL=${POLL:-60}
AFTER= MARKER= DEPLOY= TREE= START=
ARCHIVE=() ENVS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --after) AFTER=$2; shift 2;;
    --marker) MARKER=$2; shift 2;;
    --archive) ARCHIVE+=("$2"); shift 2;;
    --deploy) DEPLOY=$2; shift 2;;
    --tree) TREE=$2; shift 2;;
    --start) START=$2; shift 2;;
    --env) ENVS+=("$2"); shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
for name in AFTER MARKER DEPLOY TREE START; do
  [ -n "${!name}" ] || { echo "missing --$(echo "$name" | tr '[:upper:]' '[:lower:]')" >&2; exit 2; }
done
logname() { echo "${1%%_*}"; }        # round7_grpo -> round7
tree_hash() { (cd "$1" && find src scripts configs -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16); }
PREVIOUS_LOG=$ML/logs/$(logname "$AFTER").log
echo "== waiting for $AFTER $(date -u +%FT%TZ)"
while pgrep -f "^bash $AGENT/budget-agent/scripts/autodl/$AFTER.sh" > /dev/null; do sleep "$POLL"; done
grep -qF -- "$MARKER" "$PREVIOUS_LOG" || { echo "$AFTER did not finish cleanly (no \"$MARKER\" in $PREVIOUS_LOG); not starting $START"; exit 1; }
echo "== $AFTER finished $(date -u +%FT%TZ)"
if pgrep -f "^bash $AGENT/budget-agent/scripts/autodl/" > /dev/null; then echo "another experiment script runs; stop"; exit 1; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then echo "GPU busy; stop"; exit 1; fi
cd "$AGENT"
rm -rf budget-agent.new && mkdir budget-agent.new && tar -xzf "$DEPLOY" -C budget-agent.new
GOT=$(tree_hash budget-agent.new)
if [ "$GOT" != "$TREE" ]; then echo "tree hash $GOT != $TREE; stop"; rm -rf budget-agent.new; exit 1; fi
for dir in "${ARCHIVE[@]}"; do      # nothing is touched until the new tree has verified
  bash "$AGENT/budget-agent/scripts/autodl/archive_rollouts.sh" "$dir"
done
rm -rf "budget-agent.$AFTER" && mv budget-agent "budget-agent.$AFTER" && mv budget-agent.new budget-agent
chmod -R a+rX budget-agent
echo "== deployed $(basename "$DEPLOY") tree $GOT $(date -u +%FT%TZ)"
cd /tmp
env "${ENVS[@]}" setsid nohup bash "$AGENT/budget-agent/scripts/autodl/$START.sh" \
  > "$ML/logs/$(logname "$START").log" 2>&1 < /dev/null &
sleep "${START_CHECK:-5}"
pgrep -fa "^bash $AGENT/budget-agent/scripts/autodl/$START.sh" > /dev/null || { echo "$START did not start"; exit 1; }
echo "== $START started $(date -u +%FT%TZ)"
