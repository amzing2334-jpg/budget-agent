#!/usr/bin/env bash
# Run scripts/rollout_pool.py as `nobody`: model-written shell commands cannot
# write outside /opt/coding-agent/work or touch other projects under /root.
set -euo pipefail
AGENT=/opt/coding-agent
mkdir -p "$AGENT/work/home"; chown nobody:nogroup "$AGENT/work" "$AGENT/work/home"
exec setpriv --reuid=65534 --regid=65534 --clear-groups \
  env -i HOME="$AGENT/work/home" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 \
  PYTHONPATH="$AGENT/budget-agent/src" PYTHONDONTWRITEBYTECODE=1 \
  "$AGENT/interp/py/bin/python" "$AGENT/budget-agent/scripts/rollout_pool.py" "$@"
