#!/usr/bin/env bash
# archive_rollouts.sh <dir under /opt/coding-agent/work> [...]: move finished rollout
# directories from the system disk to the data-disk archive and leave symlinks, so every
# recorded result_path stays valid. Copies, checks that the file count and byte total
# match, and only then deletes the source. Run as root on the GPU host, never on a
# directory a running pool still writes to.
set -euo pipefail
WORK=${WORK:-/opt/coding-agent/work}
ARCHIVE=${ARCHIVE:-/root/autodl-tmp/coding-agent-rl/archive/work}
[ "$#" -ge 1 ] || { echo "usage: $0 <dir under $WORK> [...]" >&2; exit 2; }
for src in "$@"; do
  src=$(realpath -s "$src")
  case "$src" in "$WORK"/*) ;; *) echo "not under $WORK: $src" >&2; exit 1;; esac
  if [ -L "$src" ]; then echo "already archived: $src"; continue; fi
  dst=$ARCHIVE/$(echo "${src#"$WORK"/}" | tr / -)
  [ -e "$dst" ] && { echo "archive target exists: $dst" >&2; exit 1; }
  mkdir -p "$ARCHIVE"
  cp -a "$src" "$dst"
  before="$(find "$src" -type f | wc -l) $(du -sb --apparent-size "$src" | cut -f1)"
  after="$(find "$dst" -type f | wc -l) $(du -sb --apparent-size "$dst" | cut -f1)"
  if [ "$before" != "$after" ]; then echo "copy differs for $src: $before vs $after; source kept" >&2; exit 1; fi
  rm -rf "$src"
  ln -s "$dst" "$src"
  echo "archived $src -> $dst ($before)"
done
df -h "$WORK" "$ARCHIVE" | tail -n +2
