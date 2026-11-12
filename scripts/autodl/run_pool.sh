#!/usr/bin/env bash
# Run scripts/rollout_pool.py as `nobody`: model-written shell commands cannot
# write outside /opt/coding-agent/work or touch other projects under /root.
# Refuses to start when the rollout disk has less than MIN_FREE_GB free (default 5): rollouts
# must live on the system disk (/root is mode 700), and a pool that fills it dies mid-run.
# Move finished rounds to the data disk with scripts/autodl/archive_rollouts.sh.
set -euo pipefail
AGENT=/opt/coding-agent
MIN_FREE_GB=${MIN_FREE_GB:-5}
mkdir -p "$AGENT/work/home"; chown nobody:nogroup "$AGENT/work" "$AGENT/work/home"
FREE_GB=$(df --output=avail -BG "$AGENT/work" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "only ${FREE_GB} GB free for rollouts under $AGENT/work (need ${MIN_FREE_GB}); archive finished rounds first" >&2
  exit 1
fi
exec setpriv --reuid=65534 --regid=65534 --clear-groups \
  env -i HOME="$AGENT/work/home" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 \
  PYTHONPATH="$AGENT/budget-agent/src" PYTHONDONTWRITEBYTECODE=1 \
  "$AGENT/interp/py/bin/python" "$AGENT/budget-agent/scripts/rollout_pool.py" "$@"
