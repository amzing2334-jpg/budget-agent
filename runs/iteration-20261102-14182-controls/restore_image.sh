#!/usr/bin/env bash
# Restore the 14182 image from its completed, read-only cache on the grading host.
set -euo pipefail
layout=${1:?path to the offline OCI layout}
python=${2:-python3}
here=$(cd "$(dirname "$0")" && pwd -P)
spec="$here/14182-spec.json"
export PYTHONPATH="$here/../../src"

members=$("$python" -m budget_agent swe-cache-files --spec "$spec" \
    --manifest "$here/platform-manifest.json" --layout "$layout")
# Bind the names containerd will install as well as the content digests.
cmp "$here/cache-index.json" "$layout/index.json" >&2
ref=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["solver"]["pinned_image"], end="")' "$spec")
ctr=(sudo -n ctr -a /run/docker/containerd/containerd.sock -n moby)
printf '%s\n' "$members" | tar -C "$layout" --no-recursion -cf - -T - \
    | "${ctr[@]}" images import --local --platform linux/amd64 -
if ! sudo -n docker image inspect "$ref" >/dev/null 2>&1; then
    tag="docker.io/${ref%@*}:14182-verified"
    "${ctr[@]}" images ls | awk -v n="$tag" -v d="${ref#*@}" '$1==n && $3==d {ok=1} END {exit !ok}'
    "${ctr[@]}" images tag --local "$tag" "docker.io/$ref"
fi
sudo -n docker image inspect "$ref" --format '{{.Os}}/{{.Architecture}} {{json .RepoDigests}}'
