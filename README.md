# Budget Agent

**面向 coding agent 后训练的可验证环境和训练闭环。** 评测、拒绝采样微调（RFT）和 GRPO 共用同一个 agent harness 与同一个隔离评分器。全部实验在一张 A100-40GB 上完成，每个结论都有运行前冻结的判据。

## 结果

每一行都是一次运行前冻结判据的比较：留出的 4 个仓库共 96 题，按题配对 bootstrap，区间为 95% 置信区间（百分点）。

| 实验 | 留出集 pass@1 | 差值 [95% CI] | 结论 | 记录 |
|---|---|---|---|---|
| RFT：7B 教师（7.29%）的通过轨迹蒸馏到 1.5B 学生 | 0.26% → 6.25% | +5.99 [+2.86, +9.64] | 提升 | [第一轮](docs/EXPERIMENT_SWESMITH_RFT_ROUND1.md) |
| GRPO 12 步 | 6.25% → 9.64% | +3.39 [+0.26, +7.03] | 提升（边缘） | [第二轮](docs/EXPERIMENT_GRPO_ROUND2.md) |
| 同一对策略独立复现评测 | 7.03% → 9.90% | +2.86 [0.00, +5.73] | 未复现 | [复现](docs/EXPERIMENT_REPLICATION.md) |
| GRPO 25 步，加固后的 harness | 6.51% → **11.98%** | **+5.47 [+2.08, +9.11]** | 提升 | [第三轮](docs/EXPERIMENT_GRPO_ROUND3.md) |
| 固定标注 vs 重新标注（非劣性，界限 −2） | 6.90% → 5.86% | −1.04 [−2.99, +0.91] | 非劣性未证明，不采用 | [第四轮](docs/EXPERIMENT_LABELS_AB.md) |

| | 要点 | 证据 |
|---|---|---|
| **奖励覆盖** | 把每个通过的补丁与注入 bug 之前的源码对照：各策略约 10% 到 15% 的通过补丁通过了测试却没有恢复原有行为，集中在三道测试不全的题上。把第三轮的这类补丁算作失败，GRPO 的增益仍为 +2.6 到 +3.4，区间在 0 以上。GRPO 与 SFT 的比例差异不显著（p = 0.26） | [技术复盘 §3](docs/TECH_REPORT.md) |
| **一致性** | 训练端与采样端的 prompt 在 token 层面完全一致：8,311 个训练 turn 的摘要全部吻合，logprob 平均差 ≤ 0.0028 nat，TIS 从未截断 | [技术复盘 §4](docs/TECH_REPORT.md) |
| **系统** | 单卡 7B agent rollout 受 prefill 限制（prompt token 是生成的 108 到 141 倍）。前缀打包把训练编码的 token 减少 2.9 到 3.1 倍，loss 和梯度等价；改成固定标注可到 4.48 倍（事先估计 4.43 倍），但见上表第四轮。按题测试时限把 rollout 的 p90 从 631 秒降到 235 秒 | [基准](docs/BENCH_ROLLOUT_2026-11-06.md)、[打包](docs/PREFIX_PACKING.md) |

三轮训练、一次复现评测和一次标注 A/B 共 12.7 GPU 小时，约 42 元。留出集上通过的 295 个补丁都逐条人工检查过，没有发现篡改，并且都与原始源码对照过。原始输出在 [`docs/data/`](docs/data/)。

```mermaid
flowchart LR
    T[SWE-smith 实例<br/>三条件准入] --> H[run_task harness<br/>工具 · 上下文策略 · 预算账本]
    V[vLLM<br/>运行时 LoRA] <-->|请求 / token ID · logprob · prompt 摘要| H
    H -->|只应用可编辑文件| G[隔离评分器<br/>python -I · 唯一判定行 · 超时即失败]
    H --> A[(calls · trajectory · result)]
    G --> A
    A --> S[RFT：只用通过的轨迹<br/>只在训练 token 处算 logits]
    A --> R[GRPO：组内优势 · TIS · clip-higher<br/>训练前核对 prompt 摘要]
    S --> M[LoRA]
    R --> M
    M -->|热加载，约 0.7 秒| V
```

## 技术要点

- **奖励可信。**
  - 发现并修复了 5 类评分失真：标准库遮蔽、伪造判定行、git index 被改写、超时被当成"未评分"、统一超时导致的长尾。
  - 原则：每一条"排除"规则，都要问策略能不能主动触发它。
  - 详见 [技术复盘 §3](docs/TECH_REPORT.md)。
- **不依赖 Docker 的环境。** 每个仓库一个 venv，每次运行都断言被测包从当前树导入，准入有三个条件。288/288 题准入，37,237 个 P2P 测试 0 丢弃。
- **token 级数据通路。**
  - 记录采样 token、behavior logprob 和 prompt 摘要；训练前比对摘要，不一致的 turn 不训练。
  - SFT 只在需要预测训练 token 的位置算 logits，16k 样本的显存从约 34 GiB 降到 11.8 GiB，数学上等价。
- **单卡共置的 GRPO。** vLLM 占 45% 显存，训练进程用其余显存。每步保存 LoRA 并热加载，服务不重启。支持 token 级聚合、clip-higher、TIS 和 minibatch PPO。
- **为训练效率设计上下文。** 同一段前缀既决定推理时的前缀缓存命中，也决定训练时能否打包。前缀打包已经实现，并在 CPU 上验证了与逐 turn 前向等价；`revision_labels = fixed` 让历史在窗口滑动之前只追加不改写。数据见 [PREFIX_PACKING.md](docs/PREFIX_PACKING.md)。
- **可复核的结论。**
  - 协议先提交再运行，按题做配对 bootstrap，另外报告剔除可疑样本后的结果。
  - 偏差和事故都写进实验记录，例如试跑里一个变量覆盖的 bug、一次以为已停止的后台任务覆盖了服务器文件。

## 快速开始

只需 CPU：

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m budget_agent demo --output runs/demo   # 脚本化修复，9 次运行，附 HTML 报告
```

构建 SWE-smith 本地任务（只需 CPU，需要访问 HF 和 GitHub）：

```bash
python -m pip install -e ".[swesmith]"
python scripts/select_swesmith_instances.py <SWE-smith 分片目录> instances.jsonl selection.json
python scripts/build_swesmith_local.py --instances instances.jsonl --repos <镜像克隆目录> \
  --venvs <venv 目录> --output swesmith --plugin src/budget_agent/pytest_outcomes.py \
  --extras configs/swesmith-extras.json
```

在 GPU 主机上跑 rollout、RFT 和 GRPO：

```bash
scripts/autodl/serve_vllm.sh <模型目录> teacher           # 只监听 127.0.0.1:18017
scripts/autodl/run_pool.sh --config configs/swesmith-vllm.toml --tasks swesmith/train.jsonl \
  --samples 4 --output /opt/coding-agent/work/teacher        # 以 nobody 身份运行 agent 命令
python scripts/sft_data_from_runs.py --attempts <attempts.jsonl> --output sft.jsonl
python scripts/train_sft_lora.py --model <学生模型> --data sft.jsonl --output out/sft --merge
python scripts/compare_pools.py --baseline <训练前> --candidate <训练后> --split eval
python scripts/grpo_harness.py --base-model <模型> --tasks swesmith/train.jsonl \
  --config configs/swesmith-vllm.toml --output out/grpo --rollouts /opt/coding-agent/work/grpo
```

## 模块

| 路径 | 作用 |
|---|---|
| `runner.py` | 单个任务的执行循环：模型 → 动作 → 环境 → 独立评分 |
| `context.py`、`budget.py` | 上下文策略（full / recent / mask / mask-read / summary / structured）。预算在调用前预留，调用后按实际用量对账 |
| `environment.py` | 文件工具、命令执行、Docker、隔离评分器、git 快照、任务级解释器和网络策略 |
| `models.py` | OpenAI 兼容客户端：拒绝重定向，可选记录 token ID 和 logprob |
| `pytest_outcomes.py` | 记录每个测试节点的精确结果，不依赖解析日志 |
| `learning/grpo_core.py` | 分组、优势、token 级损失、TIS（纯函数，有单元测试） |
| `scripts/` | SWE-smith 构建、rollout 池、RFT、GRPO、对比分析、基准测试 |
| `scripts/autodl/` | 共享 GPU 实例上的隔离运行脚本 |

## 文档

- **[技术复盘：从评测 harness 到 RL 环境](docs/TECH_REPORT.md)**（先读这篇）
- [复现指南](docs/REPRODUCE.md)：硬件、费用、部署和每一轮的命令
- 实验记录：
  - [第一轮 RFT](docs/EXPERIMENT_SWESMITH_RFT_ROUND1.md)
  - [GRPO 试跑](docs/EXPERIMENT_GRPO_PILOT.md)
  - [第二轮 GRPO](docs/EXPERIMENT_GRPO_ROUND2.md)
  - [复现评测](docs/EXPERIMENT_REPLICATION.md)
  - [第三轮 GRPO（加固后的 harness）](docs/EXPERIMENT_GRPO_ROUND3.md)
  - [第四轮：固定标注的非劣性 A/B](docs/EXPERIMENT_LABELS_AB.md)
  - [第五轮：前缀打包下的第二次 GRPO 训练](docs/EXPERIMENT_GRPO_ROUND5.md)（已冻结）
  - [rollout 吞吐基准](docs/BENCH_ROLLOUT_2026-11-06.md)、[前缀打包](docs/PREFIX_PACKING.md)
- [代码审查与修复（2026-11-05）](docs/CODE_REVIEW_2026-11-05.md)、[项目历史](docs/HISTORY.md)、[SWE-bench 协议](docs/EXPERIMENT_PROTOCOL.md)、[训练说明](docs/TRAINING.md)
- 工程：1,000 多个测试；CI 覆盖 Linux 和 Windows、Python 3.11 和 3.13，另有 ruff（含 bugbear）、mypy（库和脚本），以及在 CPU 上用 GPU 主机的版本校验训练数学（GRPO loss、SFT loss、前缀打包）。GPU 主机的环境按锁文件安装（`requirements-gpu-host.lock.txt`），与各轮实验所用的版本完全一致
