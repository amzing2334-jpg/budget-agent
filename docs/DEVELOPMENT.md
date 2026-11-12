# 开发与实验流程

本文说明本仓库怎样改代码、怎样检查，以及每一轮 GPU 实验怎样从协议走到结果。命令细节见 [REPRODUCE.md](REPRODUCE.md)。

## 本地检查

```bash
python -m pip install -e ".[dev,tokens]"
python -m pytest -q                      # 没装 torch 时，依赖 torch 的测试会跳过
ruff check src scripts tests             # pyflakes、语法错误和 bugbear，规则在 pyproject.toml
mypy                                     # 库和流水线脚本
```

装上 GPU 主机的训练栈（CPU 版 torch，见 REPRODUCE.md）后再跑 `python -m coverage run -m pytest -q && python -m coverage report`，与 CI 的全量任务相同。

## CI

| 任务 | 检查内容 |
|---|---|
| `unit` | Linux 和 Windows、Python 3.11 和 3.13 上的测试，演示运行；Linux 3.13 上另跑 mini-SWE-agent 的上游契约测试 |
| `lint` | ruff 和 mypy，版本固定 |
| `full-suite` | 装有 GPU 主机训练栈（torch 2.14 CPU、transformers 5.20.2、peft 0.21.2）的全量测试：GRPO 和 SFT 的 loss、前缀打包与参考计算一致；库的行加分支覆盖率不低于 88% |
| `openjudge-contract` | 固定版本的 OpenJudge 上游与评测适配器的契约测试 |

pull request 和合并后的 `main` 各跑一次；同一个 pull request 的新提交会取消旧的运行。

## 代码约定

- **库**（`src/budget_agent`）只依赖标准库；tiktoken、tokenizers、torch 等是可选依赖，按需导入。
- **类型**：库和脚本都在 mypy 下无错误（`check_untyped_defs`）。
- **长度契约**：`zip` 一律写 `strict=True`，长度不一致时报错，而不是悄悄截断。
- **测试**：每个行为改动都带测试。训练相关的数值改动要有与参考计算的对照（小模型，CPU）。
- **依赖**：GPU 主机按 `requirements-gpu-host.lock.txt` 安装，`tests/test_dependency_locks.py` 保证 CI、可选依赖范围和锁文件一致。

## 实验流程（预先登记）

1. **写协议**：在 `docs/EXPERIMENT_<名称>.md` 写明问题、设置、冻结的判据（主指标、决策规则、次要指标）、预算和已知局限，注明写定的时间和当时还不知道的结果。驱动脚本放在 `scripts/autodl/`。协议和脚本先合入 `main`，再运行。
2. **部署**：用 `git archive` 打包合入后的 commit，在服务器上核对代码树哈希；运行结束后去掉 `__pycache__` 再核对一次。
3. **运行时的约束**：
   - 驱动脚本在 GPU 已被占用、或本项目已有别的实验在运行时拒绝启动（`tests/test_autodl_guards.py` 保证各脚本的检查一致）；
   - 模型生成的命令一律以 `nobody` 执行，vLLM 只监听本机；
   - 新的 GRPO 驱动脚本默认带 `--pipeline`（第八轮的非劣性规则）和失稳检测 `--halt-invalid-share 0.04 --halt-patience 2`，评测 `final.json` 指定的 adapter；
   - 轮次之间用 `scripts/autodl/chain_round.sh` 衔接：上一轮正常结束（日志里有完成标记）、GPU 空闲、新代码树的哈希核对通过后，才归档上一轮的 rollout、部署并启动下一轮，每一步都记日志；
   - 系统盘剩余空间不足 5 GB 时，`run_pool.sh` 拒绝启动评测池；已完成的 rollout 用 `archive_rollouts.sh` 移到数据盘，分析做完后用 `pack_rollouts.py` 打包（没有实验运行时才做，避免影响测试时限和计时）。
4. **写结果**：
   - 结果只追加在协议后面，判据不改；
   - 原始输出复制到 `docs/data/<轮次>/`；
   - 逐条读通过的补丁，并与注入 bug 之前的源码对照（`review_patches.py --original`）；
   - 发现之前的结论有误时，追加更正并说明依据，不改写原文。

## 合并

每个改动一个 pull request，CI 通过后以 merge commit 合入 `main`（`--no-ff`），保留每个 pull request 的边界。
