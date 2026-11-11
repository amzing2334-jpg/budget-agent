# scripts

按流水线阶段分组。每个脚本的文件头都有完整的说明和用法。端到端的命令见 [docs/REPRODUCE.md](../docs/REPRODUCE.md)。

## 任务构建（SWE-smith，不依赖 Docker）

| 脚本 | 作用 |
|---|---|
| `select_swesmith_instances.py` | 预注册的实例选择：固定种子，划分训练和留出仓库，并按补丁和测试规模筛选 |
| `build_swesmith_local.py` | 每个仓库一个 venv。准入有三个条件，每次运行都断言导入来源；生成 verifier 和任务清单 |
| `add_test_timeouts.py` | 用准入记录给已有的任务清单补上按题的测试时限 |

## rollout、评测与结果

| 脚本 | 作用 |
|---|---|
| `rollout_pool.py` | 每题并发运行 K 次 `run_task`，输出 `attempts.jsonl` 和 `summary.json`（无偏 pass@k） |
| `compare_pools.py` | 两个池按题配对做 bootstrap，可以剔除可疑样本（`--drop-flagged`） |
| `summarize_round.py` | 一轮的汇总：各池 pass@k、状态、编辑成功率、篡改审计、SFT 数据、loss 曲线、阶段耗时和费用 |
| `review_patches.py` | 列出一个池里每个通过补丁的改动，用于逐条的防作弊检查；加 `--original` 时与注入 bug 之前的源码对照，打印剩余差异 |
| `mark_failures.py` | 把对照原始源码后判定为没有恢复原有行为的补丁算作失败，生成用于敏感性比较的 attempts 文件 |
| `limit_events.py` | 统计被 harness 限制终止的命令：按题测试时限、命令内存上限，以及 verifier 的超时和内存超限 |

## 训练

| 脚本 | 作用 |
|---|---|
| `sft_data_from_runs.py` | 拒绝采样：只保留通过的轨迹，每题有上限，有篡改信号的排除；`--revision-labels` 把同一批轨迹按另一种标注方式重新渲染（先逐条复现记录的请求） |
| `train_sft_lora.py` | 多 turn 轨迹的 LoRA SFT，只在训练 token 处计算 logits |
| `grpo_harness.py` | 单卡 GRPO：harness rollout 加 LoRA 热加载、TIS、clip-higher、minibatch，以及前缀打包（`--pack-turns`） |

## 系统测量

| 脚本 | 作用 |
|---|---|
| `bench_rollout.py` | rollout 吞吐随并发数的变化，以及 vLLM 的前缀缓存命中率 |
| `bench_turn_packing.py` | 在真实 rollout 上统计前缀打包能省多少 token，并给链断开的原因分类 |
| `bench_context_build.py` | 上下文构造的耗时（CPU） |
| `vllm_metrics.py` | 读取 vLLM 计数的快照，或对两个快照求差 |

## GPU 主机（`autodl/`）

| 脚本 | 作用 |
|---|---|
| `deploy.sh`、`setup_isolated.sh` | 部署代码与数据；建立隔离的环境（独立 CPython、vLLM/torch），完成准入，下载模型 |
| `serve_vllm.sh`、`stop_vllm.sh`、`run_pool.sh` | 只监听本机的 vLLM，只停自己启动的服务；rollout 以 `nobody` 执行 |
| `round1_rft.sh`、`grpo_pilot.sh`、`round2_grpo.sh`、`replicate_eval.sh`、`round3_grpo.sh`、`round4_labels_ab.sh`、`round5_grpo.sh`、`round6_labels_sft.sh`、`bench.sh` | 各轮实验的驱动脚本，对应 `docs/EXPERIMENT_*.md`。GPU 被占用、或本项目已有别的实验脚本在运行时，都拒绝启动 |
| `recon.sh` | 只读的环境勘察 |

## 早期阶段（2026 年 10 月：上下文策略研究与 SWE-bench 试验）

`demo_feedback_control.py`、`demo_governance.py`、`validate_read_loop.py`、`replay_read_memory.py`、`probe_action_reserve.py`、`probe_history_format.py`、`grade_protocol_probe.py`、`prepare_swebench_pilot.py`、`run_mini_swe_baseline.py`、`verify_hf_snapshot.py`、`restore_v5_cloud.sh`

这些脚本服务于早期的上下文策略研究和 SWE-bench 试验，记录见 [docs/HISTORY.md](../docs/HISTORY.md)。
