# Project working agreements

- GitHub repository: `https://github.com/Frank081202/budget-agent` (private, initialized under this account on 2026-10-24). Verify the active account with `gh api user --jq .login` before repository administration. Use the repository-local Frank081202 commit identity; do not change global Git identity.

- The user explicitly prohibits training on their local computer. Do not start local SFT, RL, GPU training, or automatic training smoke jobs. Future model training must use cloud GPU hardware provided by the user; first state the required GPU and resource estimate.
- Local work may edit code, inspect saved artifacts, and run CPU-only software checks. The lightweight `.venv` test environment does not include the optional model training stack.
- Keep agent code direct: typed internal contracts, validation at external boundaries, and specific exceptions. Avoid broad exception swallowing, repeated internal validation, silent fallbacks, or speculative compatibility layers.
- Preserve original experiment artifacts. The `runs/learning-v2/grpo` attempt is invalidated because a grader parsing error produced a false negative. Its update is not evidence of valid RL training. The fixed grader requires fresh cloud rollouts from a valid checkpoint.
