# 复现指南

本文写于 2026-11-11。按本文可以从零复现第一轮 RFT、第二轮 GRPO、复现评测、第三轮 GRPO 和 rollout 基准。

每一轮的问题、判据和结果都写在各自的实验记录里，在运行前冻结，运行后只追加结果；原始输出在 [`docs/data/`](data/)。

## 硬件与费用

- **硬件**：一张 A100-PCIE-40GB（AutoDL 佛山区）。容器内存上限 72 GiB，CPU 配额 10 核，系统盘 30 GB，数据盘至少 50 GB。
- **价格**：按 3.28 元/小时计。

| 实验 | GPU 时间 | 费用 | 记录 |
|---|---|---|---|
| 第一轮 RFT | 2.79 小时 | 9.2 元 | [EXPERIMENT_SWESMITH_RFT_ROUND1.md](EXPERIMENT_SWESMITH_RFT_ROUND1.md) |
| GRPO 试跑 | 约 0.3 小时 | 约 1 元 | [EXPERIMENT_GRPO_PILOT.md](EXPERIMENT_GRPO_PILOT.md) |
| rollout 基准 | 约 0.25 小时 | 约 0.8 元 | [BENCH_ROLLOUT_2026-11-06.md](BENCH_ROLLOUT_2026-11-06.md) |
| 第二轮 GRPO | 2.62 小时 | 8.6 元 | [EXPERIMENT_GRPO_ROUND2.md](EXPERIMENT_GRPO_ROUND2.md) |
| 复现评测 | 0.92 小时 | 3.0 元 | [EXPERIMENT_REPLICATION.md](EXPERIMENT_REPLICATION.md) |
| 第三轮 GRPO | 4.27 小时 | 14.0 元 | [EXPERIMENT_GRPO_ROUND3.md](EXPERIMENT_GRPO_ROUND3.md) |
| 第四轮标注 A/B | 约 1.8 小时（计划） | 约 6 元 | [EXPERIMENT_LABELS_AB.md](EXPERIMENT_LABELS_AB.md) |
| 第五轮 GRPO（前缀打包） | 约 4.4 小时（计划） | 约 14.5 元 | [EXPERIMENT_GRPO_ROUND5.md](EXPERIMENT_GRPO_ROUND5.md) |

## 1. 本地检查（只需 CPU）

```bash
python -m pip install -e ".[dev,tokens]"
python -m pytest -q                     # 依赖 torch 的测试会跳过，它们在 GPU 主机的 CPU 上运行
ruff check src scripts tests            # pyflakes、语法错误和 bugbear，规则在 pyproject.toml
mypy                                    # 核心库的类型检查，配置在 pyproject.toml
```

## 2. 构建 SWE-smith 本地任务

- 用 `scripts/select_swesmith_instances.py` 从 SWE-smith 的 HF 分片中选实例。
  - 种子固定为 20261105，12 个仓库中 8 个作训练、4 个作留出（funcy、tomli、parse、tinydb）。
  - 每个仓库最多 24 个实例，筛选条件是补丁只改 1 个文件、1 ≤ F2P ≤ 20、P2P ≤ 2000。
- 把 `instances.jsonl` 和 12 个仓库的镜像克隆（目录名即 SWE-smith 的 repo 名）打包成数据包，作为部署的输入。
- 准入在 GPU 主机上进行（见第 3 步），预期 288/288 题通过、37,237 个 P2P 测试 0 丢弃。准入记录写在 `admission.jsonl`，每题包含干净代码和带 bug 代码上的测试耗时，按题时限就由它推出。

## 3. 部署到 GPU 主机

```bash
PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple \
  scripts/autodl/deploy.sh <ssh-port> <ssh-host> swesmith-data.tar.gz
```

- **`deploy.sh`** 先做只读的环境勘察，然后把 `main` 和数据包传到 `/opt/coding-agent`，再在后台运行 `setup_isolated.sh`。
- **`setup_isolated.sh`** 可以重复运行，依次完成：
  - 在系统盘建 vLLM/torch 环境（只有 root 使用），按 `requirements-gpu-host.lock.txt` 安装。这份锁文件冻结自 GPU 主机，环境从 2026-11-06 装好后没有变过，所有轮次都跑在这套版本上（vLLM 0.33.0、torch 2.14.0 + CUDA 13.0、transformers 5.20.2、peft 0.21.2）。设 `ML_LOCK=` 为空则改装最新的 vLLM；
  - 用 uv 在 `/opt/coding-agent/python` 安装独立的 CPython 3.12.16。`/root` 的权限是 700，所有以 `nobody` 运行的东西都要用这个解释器。
  - 建立 rollout 解释器（按 `requirements-rollout.lock.txt` 安装，含 `tokenizers`）和每个仓库的 venv，完成准入；
  - 从 ModelScope 下载 7B 和 1.5B 模型；
  - 把分词器文件放到 `nobody` 可读的位置。
- **镜像源**：佛山实例上镜像默认的阿里云源只有 0.35 MB/s，华为云有 10.7 MB/s，部署前先测速。
- **核对部署**：在服务器上计算代码树哈希，与本地 `git archive` 解出的结果比对：

```bash
cd /opt/coding-agent/budget-agent && find src scripts configs -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16
```

## 4. 各轮实验

所有驱动脚本都以 root 运行，并遵守以下规则：
- 只停自己启动的 vLLM（按 pidfile）；
- GPU 上已有其他进程时拒绝启动；
- 本项目已有别的实验脚本在运行时也拒绝启动；
- 模型生成的命令一律以 `nobody` 执行；
- vLLM 只监听 `127.0.0.1:18017`。

```bash
setsid nohup bash scripts/autodl/round1_rft.sh > logs/round1.log 2>&1 &     # 教师采样 → 学生基线 → SFT → 评测
setsid nohup bash scripts/autodl/grpo_pilot.sh > logs/grpo-pilot.log 2>&1 &
setsid nohup bash scripts/autodl/bench.sh > logs/bench.log 2>&1 &
setsid nohup bash scripts/autodl/round2_grpo.sh > logs/round2.log 2>&1 &
# 复现评测要用前两轮的原始任务清单：cp eval.jsonl eval.round1.jsonl（在迁移任务清单之前）
setsid nohup bash scripts/autodl/replicate_eval.sh > logs/replication.log 2>&1 &
```

第三轮使用加固后的 harness：`configs/swesmith-vllm-tokens.toml`，按题时限、命令内存上限、按真实 token 计算的上下文预算。先迁移任务清单，再按冻结的参数启动：

```bash
cd /opt/coding-agent/swesmith && python /opt/coding-agent/budget-agent/scripts/add_test_timeouts.py \
  --admission admission.jsonl train.jsonl eval.jsonl smoke32.jsonl
STEPS=40 MAX_MINUTES=180 LR=2e-5 TURNS_PER_ATTEMPT=4 MINIBATCHES=4 \
  setsid nohup bash /opt/coding-agent/budget-agent/scripts/autodl/round3_grpo.sh > logs/round3.log 2>&1 &
```

第四轮（标注 A/B）和第五轮（前缀打包下的第二次 GRPO，标注方式由第四轮的规则决定）：

```bash
setsid nohup bash /opt/coding-agent/budget-agent/scripts/autodl/round4_labels_ab.sh > logs/round4.log 2>&1 &
STEPS=60 MAX_MINUTES=180 LR=2e-5 TURNS_PER_ATTEMPT=4 MINIBATCHES=4 SEED=1 LABELS=fixed \
  setsid nohup bash /opt/coding-agent/budget-agent/scripts/autodl/round5_grpo.sh > logs/round5.log 2>&1 &
```

每一轮都会在 `outputs/<轮次>/` 下写出：
- `steps.jsonl`：各阶段耗时；
- `compare_*.json`：按题配对的 bootstrap 比较；
- `results.md`：`summarize_round.py` 的汇总。

## 5. 核对结论

- **主结论**：读 `compare_eval.json` 的 `ci95` 和 `verdict`，规则在各轮协议里写定：下界大于 0 为提升，区间包含 0 为未检测到提升，上界小于 0 为变差。
- **防作弊**：`python scripts/review_patches.py <评测池目录>` 列出每个通过补丁的改动，逐条阅读；剔除可疑样本后的结果在 `compare_eval_drop_flagged.json`。
- **对照原始源码**：加上 `--original /opt/coding-agent/swesmith-src --tasks /opt/coding-agent/swesmith/tasks`，脚本会把每个补丁应用到带 bug 的文件上，与注入 bug 之前的源码比较，并打印剩余差异。第三轮的判断结果在 `docs/data/round3/original_check.json`。
- **敏感性分析**：把判定为没有恢复原有行为的补丁算作失败，再做配对比较。第三轮的两个数字可以这样复现：

```bash
python scripts/mark_failures.py <SFT 池>/attempts.jsonl docs/data/round3/original_check.json --pool eval-sft --kinds strong > sft.jsonl
python scripts/mark_failures.py <GRPO 池>/attempts.jsonl docs/data/round3/original_check.json --pool eval-grpo --kinds strong > grpo.jsonl
python scripts/compare_pools.py --baseline sft.jsonl --candidate grpo.jsonl --split eval   # +3.39，区间 [+0.78, +6.25]
```

  `--kinds strong,mild` 给出 +2.60，区间 [+0.26, +5.21]。
- **被限制终止的命令**：`python scripts/limit_events.py <评测池或 GRPO 步目录...>`。
- **训练过程**：`grpo/metrics.jsonl` 每一步记录以下指标：
  - `prompt_mismatch`，应为 0；
  - 采样端与训练端 logprob 的差（`sampler_trainer_abs_logprob_gap`）；
  - TIS 权重；
  - clip 比例；
  - 熵；
  - 每阶段耗时。
- **前缀打包的节省**：`scripts/bench_turn_packing.py --tokenizer <模型目录> <GRPO 步目录...>`，方法见 [PREFIX_PACKING.md](PREFIX_PACKING.md)。
