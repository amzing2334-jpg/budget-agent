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
| 第四轮标注 A/B | 2.09 小时 | 6.9 元 | [EXPERIMENT_LABELS_AB.md](EXPERIMENT_LABELS_AB.md) |
| 第五轮 GRPO（前缀打包） | 5.0 小时（含一次中断） | 约 16.4 元 | [EXPERIMENT_GRPO_ROUND5.md](EXPERIMENT_GRPO_ROUND5.md) |
| 第六轮固定标注 SFT | 1.39 小时 | 约 4.6 元 | [EXPERIMENT_LABELS_SFT.md](EXPERIMENT_LABELS_SFT.md) |
| 第七轮继续训练 GRPO | 10.4 小时（评测 5.7 小时，含汇总报错后 1.45 小时的空闲） | 约 34.1 元 | [EXPERIMENT_GRPO_ROUND7.md](EXPERIMENT_GRPO_ROUND7.md) |
| 第八轮流水线 GRPO | 3.85 小时 | 约 12.6 元 | [EXPERIMENT_GRPO_ROUND8.md](EXPERIMENT_GRPO_ROUND8.md) |
| 第九轮评测失稳检测保留的 adapter | 1.28 小时 | 约 4.2 元 | [EXPERIMENT_GRPO_ROUND9.md](EXPERIMENT_GRPO_ROUND9.md) |
| 第十轮在新题上继续训练 | 3.61 小时 | 约 11.8 元 | [EXPERIMENT_GRPO_ROUND10.md](EXPERIMENT_GRPO_ROUND10.md) |
| 第十一轮换种子重复第八轮 | 未运行（约 4 小时） | — | [EXPERIMENT_GRPO_ROUND11.md](EXPERIMENT_GRPO_ROUND11.md) |
| 第十二轮 3B 学生 | 未运行（约 8 到 10 小时） | — | [EXPERIMENT_GRPO_ROUND12.md](EXPERIMENT_GRPO_ROUND12.md) |

## 1. 本地检查（只需 CPU）

```bash
python -m pip install -e ".[dev,tokens]"
python -m pytest -q                     # 没装 torch 时，依赖 torch 的测试会跳过
# 与 CI 的全量任务相同：装上 GPU 主机的训练栈（CPU 版），并检查覆盖率
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers==5.20.2 peft==0.21.2 coverage==7.13.1
python -m coverage run -m pytest -q && python -m coverage report
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
- **磁盘**：rollout 必须写在系统盘的 `/opt/coding-agent/work` 下（`/root` 的权限是 700，`nobody` 进不去），而系统盘只有 30 GB，其中 vLLM/torch 环境约占 14 GB；一个每题 8 次的评测池约 2.5 GB。每轮结束后用 `scripts/autodl/archive_rollouts.sh <目录>` 把已完成的 rollout 移到数据盘并留下符号链接，记录里的路径保持有效。`run_pool.sh` 在剩余空间不足 5 GB 时拒绝启动（`MIN_FREE_GB` 可调），第五轮就曾因系统盘写满而在评测中途中断。数据盘与其他项目共用：一个池有几十万个小文件，每个至少占一个块，分析做完后用 `python scripts/pack_rollouts.py <池或上级目录>` 打包（只在没有实验运行时做），恢复用 `tar -xzf <池>/runs.tar.gz -C <池>`。
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
setsid nohup bash /opt/coding-agent/budget-agent/scripts/autodl/round6_labels_sft.sh > logs/round6.log 2>&1 &
STEPS=60 MAX_MINUTES=180 LR=2e-5 TURNS_PER_ATTEMPT=4 MINIBATCHES=4 SEED=2 \
  setsid nohup bash /opt/coding-agent/budget-agent/scripts/autodl/round7_grpo.sh > logs/round7.log 2>&1 &
STEPS=29 MAX_MINUTES=240 LR=2e-5 TURNS_PER_ATTEMPT=4 MINIBATCHES=4 SEED=1 \
  setsid nohup bash /opt/coding-agent/budget-agent/scripts/autodl/round8_grpo.sh > logs/round8.log 2>&1 &
setsid nohup bash /opt/coding-agent/budget-agent/scripts/autodl/round9_eval.sh > logs/round9.log 2>&1 &   # 只评测，不训练
setsid nohup bash /opt/coding-agent/budget-agent/scripts/autodl/round10_fresh_tasks.sh > logs/round10.log 2>&1 &
STEPS=29 MAX_MINUTES=240 LR=2e-5 TURNS_PER_ATTEMPT=4 MINIBATCHES=4 SEED=2 \
  setsid nohup bash /opt/coding-agent/budget-agent/scripts/autodl/round11_seed.sh > logs/round11.log 2>&1 &
STEPS=29 MAX_MINUTES=360 LR=2e-5 TURNS_PER_ATTEMPT=4 MINIBATCHES=4 SEED=1 \
  setsid nohup bash /opt/coding-agent/budget-agent/scripts/autodl/round12_3b.sh > logs/round12.log 2>&1 &   # 需要先下载 Qwen2.5-Coder-3B-Instruct
```

轮次之间用 `chain_round.sh` 衔接：等上一轮的驱动脚本正常结束（日志里有完成标记）、GPU 空闲、新代码树的哈希核对通过，再部署并启动下一轮。第八轮到第九轮就是这样排队的：

```bash
setsid nohup bash chain_round.sh --after round8_grpo --marker "== round 8 done" \
  --deploy /opt/coding-agent/incoming/budget-agent-<commit>.tar.gz --tree <代码树哈希> \
  --start round9_eval > logs/chain_round9.log 2>&1 &
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
