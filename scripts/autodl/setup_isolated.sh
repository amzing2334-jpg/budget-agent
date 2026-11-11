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
[ -n "${PIP_INDEX_URL:-}" ] || unset PIP_INDEX_URL   # empty means: keep the image's pip config
export PIP_CACHE_DIR=$ENV_ROOT/cache/pip HF_HOME=$ML/hf MODELSCOPE_CACHE=$ENV_ROOT/cache/modelscope
# The root-only ML env comes from the image's base interpreter: no conda channels
# needed (the image's mirror config may be stale) and the base env is never modified.
BASE_PY=${BASE_PY:-/root/miniconda3/bin/python3}
[ -x "$ENV_ROOT/env/bin/python" ] || "$BASE_PY" -m venv "$ENV_ROOT/env"
ln -sfn "$ENV_ROOT/env" "$ML/env"   # scripts refer to $ML/env

# ML stack: the exact environment every round ran on. ML_LOCK= (empty) resolves the newest
# vLLM instead; vLLM then pins torch/transformers and the training extras must accept those pins.
CODE=$AGENT/budget-agent
ML_LOCK=${ML_LOCK-$CODE/requirements-gpu-host.lock.txt}
if [ -n "$ML_LOCK" ]; then
  "$ML/env/bin/pip" install -q -r "$ML_LOCK"
else
  "$ML/env/bin/pip" install -q vllm
  "$ML/env/bin/pip" install -q peft accelerate modelscope uv
fi

# Everything that runs as `nobody` (the rollout interpreter, per-repo task venvs)
# needs an interpreter outside /root, which is mode 700: a standalone CPython
# under $AGENT, installed by uv without shims in ~/.local/bin.
AGENT_PY=$AGENT/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12
[ -x "$AGENT_PY" ] || "$ML/env/bin/uv" python install 3.12.16 --install-dir "$AGENT/python" --no-bin
chmod -R a+rX "$AGENT/python"
[ -x "$AGENT/interp/py/bin/python" ] || "$AGENT_PY" -m venv "$AGENT/interp/py"
"$AGENT/interp/py/bin/python" -m pip install -q -r "$CODE/requirements-rollout.lock.txt"   # tokenizers: token-counted budgets
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

# SWE-smith tasks: per-repo venvs from the standalone interpreter (the profile
# asks for 3.10; admission re-verifies every task on the interpreter actually
# used), task repos and verifiers readable by `nobody`.
if [ -f /opt/coding-agent/swesmith-src/instances.jsonl ] && [ ! -f /opt/coding-agent/swesmith/summary.json ]; then
  "$AGENT/interp/py/bin/python" "$AGENT/budget-agent/scripts/build_swesmith_local.py" \
    --instances /opt/coding-agent/swesmith-src/instances.jsonl --repos /opt/coding-agent/swesmith-src \
    --venvs /opt/coding-agent/venvs --output /opt/coding-agent/swesmith \
    --python "$AGENT_PY" --plugin "$AGENT/budget-agent/src/budget_agent/pytest_outcomes.py" \
    --extras "$AGENT/budget-agent/configs/swesmith-extras.json" --workers 6
  chmod -R a+rX /opt/coding-agent/swesmith /opt/coding-agent/venvs
fi

# Models from ModelScope into the root-only project directory.
for model in Qwen/Qwen2.5-Coder-7B-Instruct Qwen/Qwen2.5-Coder-1.5B-Instruct; do
  target="$ML/models/${model#*/}"
  [ -f "$target/config.json" ] || "$ML/env/bin/modelscope" download --model "$model" --local_dir "$target"
done
# Token-counted budgets (configs/swesmith-vllm-tokens.toml) read the served model's
# tokenizer as `nobody`; the 1.5B and 7B Qwen2.5-Coder tokenizers are identical.
mkdir -p "$AGENT/tokenizers/qwen2.5-coder"
cp "$ML/models/Qwen2.5-Coder-1.5B-Instruct/tokenizer.json" "$AGENT/tokenizers/qwen2.5-coder/tokenizer.json"
chmod -R a+rX "$AGENT/tokenizers"
echo "== models ready"
