#!/usr/bin/env bash
# Cloud-only asset restoration. No inference, training, rental or release actions.
# Establish the provider release timer separately before starting these stages.
set -euo pipefail

stage=${1:-help}
if [[ "$stage" == help ]]; then
    printf '%s\n' \
      'Usage: restore_v5_cloud.sh install|model|image' \
      'All stages: BA_PROJECT BA_RUN_LABEL BA_STOP_UTC (UTC workload cutoff).' \
      'Receipts: BA_RECEIPTS or BA_OUTPUT_ROOT/BA_RUN_LABEL-receipts (output root defaults to PROJECT/runs).' \
      'install: BA_INFER_ENV BA_GRADER_ENV; optional BA_PYTHON (default python3).' \
      'model: BA_INFER_ENV BA_MODEL_METADATA BA_MODEL_HF_HOME;' \
      '       optional BA_CACHED_HF_HOME (try complete existing cache first).' \
      'image: BA_DOCKER_ARCHIVE BA_DOCKER_IMAGE BA_DOCKER_ID.' \
      'Model hashes are checked against the saved official Hub metadata.' \
      'Select HF_ENDPOINT externally if a mirror is required.'
    exit 0
fi
[[ "$(uname -s)" == Linux ]] || { printf 'Run on the authorized Linux cloud host.\n' >&2; exit 2; }
project=${BA_PROJECT:?Set BA_PROJECT}
label=${BA_RUN_LABEL:?Set a new BA_RUN_LABEL}
receipts=${BA_RECEIPTS:-${BA_OUTPUT_ROOT:-$project/runs}/$label-receipts}
deadline=$(date -u -d "${BA_STOP_UTC:?Set the UTC workload cutoff}" +%s)
mkdir -p "$receipts"
cd "$project"

bounded() {
    local seconds=$((deadline - $(date -u +%s)))
    (( seconds > 0 )) || { printf 'Workload cutoff reached.\n' >&2; exit 3; }
    timeout --signal=INT --kill-after=20s "$seconds" "$@"
}

case "$stage" in
install)
    bootstrap=${BA_PYTHON:-python3}
    infer_env=${BA_INFER_ENV:?Set BA_INFER_ENV}
    grader_env=${BA_GRADER_ENV:?Set BA_GRADER_ENV}
    "$bootstrap" -c 'import sys; assert sys.version_info >= (3, 11), sys.version'
    bounded "$bootstrap" -m venv "$infer_env"
    bounded "$infer_env/bin/python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    bounded "$infer_env/bin/python" -m pip install transformers==5.4.0 socksio==1.0.0
    bounded "$infer_env/bin/python" -m pip install -e "$project"
    bounded "$bootstrap" -m venv "$grader_env"
    bounded "$grader_env/bin/python" -m pip install swebench==5.0.2
    "$infer_env/bin/python" -m pip check
    "$grader_env/bin/python" -m pip check
    "$infer_env/bin/python" -m pip freeze > "$receipts/inference.freeze.txt"
    "$grader_env/bin/python" -m pip freeze > "$receipts/grader.freeze.txt"
    ;;
model)
    python=${BA_INFER_ENV:?Set BA_INFER_ENV}/bin/python
    metadata=${BA_MODEL_METADATA:?Set BA_MODEL_METADATA}
    download_home=${BA_MODEL_HF_HOME:?Set BA_MODEL_HF_HOME}
    # The cache check reads metadata and files only; it never loads a model.
    # Missing files permit a download. A complete but corrupt cache fails hashing.
    bounded "$python" - "$metadata" "$download_home" "${BA_CACHED_HF_HOME:-}" "$receipts" <<'PY'
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

metadata_path, download_home, cached_home, receipts = sys.argv[1:]
metadata = json.loads(Path(metadata_path).read_text())
required = [item for item in metadata['siblings']
            if item['rfilename'].endswith(('.json', '.safetensors')) or item['rfilename'] == 'merges.txt']
relative = Path('hub') / ('models--' + metadata['id'].replace('/', '--')) / 'snapshots' / metadata['sha']

def complete(home):
    snapshot = Path(home) / relative
    return all((snapshot / item['rfilename']).is_file() and
               (snapshot / item['rfilename']).stat().st_size == item['size'] for item in required)

home = next((candidate for candidate in [cached_home, download_home] if candidate and complete(candidate)), download_home)
os.environ['HF_HOME'] = home
snapshot = Path(home) / relative
if not complete(home):
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(metadata['id'], revision=metadata['sha'],
                                     allow_patterns=[item['rfilename'] for item in required], max_workers=2))
subprocess.run([sys.executable, 'scripts/verify_hf_snapshot.py', metadata_path, str(snapshot),
                str(Path(receipts) / 'model-integrity.json')], check=True)
Path(receipts, 'model-cache.env').write_text('export HF_HOME=' + shlex.quote(home) + '\n')
print(json.dumps({'model': metadata['id'], 'revision': metadata['sha'], 'hf_home': home,
                  'snapshot': str(snapshot)}))
PY
    ;;
image)
    archive=${BA_DOCKER_ARCHIVE:?Set BA_DOCKER_ARCHIVE}
    image=${BA_DOCKER_IMAGE:?Set BA_DOCKER_IMAGE}
    expected=${BA_DOCKER_ID:?Set BA_DOCKER_ID}
    docker info > "$receipts/docker-info.txt"
    bounded sha256sum -c "$archive.sha256"
    bounded docker load -i "$archive"
    actual=$(docker image inspect --format '{{.Id}}' "$image")
    [[ "$actual" == "$expected" ]] || { printf 'Restored Docker image ID differs: %s\n' "$actual" >&2; exit 4; }
    docker image inspect "$image" > "$receipts/image.json"
    ;;
*) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac

printf 'Completed %s. Receipts: %s\n' "$stage" "$receipts"
