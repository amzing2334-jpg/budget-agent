#!/usr/bin/env bash
# Isolated setup for the coding-agent RL project on a shared AutoDL instance.
# Touches only $ML and $AGENT. Never modifies the base conda env, other
# projects, system packages or SSH settings. Safe to re-run.
set -euo pipefail
ML=/root/autodl-tmp/coding-agent-rl        # data disk: models, training outputs, logs
ENV_ROOT=${ENV_ROOT:-/opt/coding-agent-ml}  # system disk: the ~10 GB vLLM/torch env (data disk is shared)
AGENT=/opt/coding-agent                     # agent-facing: code, task repos, rollouts (run as nobody)
LOG=$ML/logs/setup.log
mkdir -p "$ML"/{models,hf,logs,outputs} "$AGENT"/{work,interp} "$ENV_ROOT"/cache
chmod 700 "$ENV_ROOT"
exec > >(tee -a "$LOG") 2>&1
echo "== setup $(date -u +%FT%TZ)"
export PIP_CACHE_DIR=$ENV_ROOT/cache/pip HF_HOME=$ML/hf MODELSCOPE_CACHE=$ENV_ROOT/cache/modelscope
# Virtualenvs from the image's base interpreter: no conda channels needed (the
# image's mirror config may be stale) and the base env is never modified.
BASE_PY=${BASE_PY:-/root/miniconda3/bin/python3}
[ -x "$ENV_ROOT/env/bin/python" ] || "$BASE_PY" -m venv "$ENV_ROOT/env"
ln -sfn "$ENV_ROOT/env" "$ML/env"   # scripts refer to $ML/env
[ -x "$AGENT/interp/py/bin/python" ] || "$BASE_PY" -m venv "$AGENT/interp/py"

# ML stack: vLLM pins torch/transformers; training extras must accept those pins.
"$ML/env/bin/pip" install -q vllm
"$ML/env/bin/pip" install -q peft accelerate modelscope
"$ML/env/bin/python" - <<'EOF'
import torch, vllm, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("vllm", vllm.__version__, "transformers", transformers.__version__, "peft", peft.__version__)
EOF

# Code is unpacked by the caller into $AGENT/budget-agent and used via PYTHONPATH=src
# (the harness needs only the standard library).
chmod -R a+rX "$AGENT"
chown -R nobody:nogroup "$AGENT/work"
echo "== setup done"

# SWE-smith tasks: per-repo venvs from the base interpreter (the profile asks
# for 3.10; admission re-verifies every task on the interpreter actually used),
# task repos and verifiers readable by `nobody`.
if [ -f /opt/coding-agent/swesmith-src/instances.jsonl ] && [ ! -f /opt/coding-agent/swesmith/summary.json ]; then
  "$AGENT/interp/py/bin/python" "$AGENT/budget-agent/scripts/build_swesmith_local.py" \
    --instances /opt/coding-agent/swesmith-src/instances.jsonl --repos /opt/coding-agent/swesmith-src \
    --venvs /opt/coding-agent/venvs --output /opt/coding-agent/swesmith \
    --python "$BASE_PY" --plugin "$AGENT/budget-agent/src/budget_agent/pytest_outcomes.py" \
    --extras "$AGENT/budget-agent/configs/swesmith-extras.json" --workers 6
  chmod -R a+rX /opt/coding-agent/swesmith /opt/coding-agent/venvs
fi

# Models from ModelScope into the root-only project directory.
for model in Qwen/Qwen2.5-Coder-7B-Instruct Qwen/Qwen2.5-Coder-1.5B-Instruct; do
  target="$ML/models/${model#*/}"
  [ -f "$target/config.json" ] || "$ML/env/bin/modelscope" download --model "$model" --local_dir "$target"
done
echo "== models ready"
