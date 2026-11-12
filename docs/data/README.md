# 原始输出

各轮实验在 GPU 主机上写出的原始结果，原样复制，没有改动。每一轮的问题、判据和解读在对应的实验记录里。

| 目录或文件 | 内容 | 实验记录 |
|---|---|---|
| `round1/` | 第一轮 RFT：教师采样、学生基线、SFT 和评测 | [EXPERIMENT_SWESMITH_RFT_ROUND1.md](../EXPERIMENT_SWESMITH_RFT_ROUND1.md) |
| `round2/` | 第二轮 GRPO（12 步） | [EXPERIMENT_GRPO_ROUND2.md](../EXPERIMENT_GRPO_ROUND2.md) |
| `replication/` | 两个策略的独立复现评测 | [EXPERIMENT_REPLICATION.md](../EXPERIMENT_REPLICATION.md) |
| `round3/` | 第三轮 GRPO（加固后的 harness，25 步），以及对照原始源码的复核 | [EXPERIMENT_GRPO_ROUND3.md](../EXPERIMENT_GRPO_ROUND3.md) |
| `round4/` | 第四轮：固定标注与重新标注的非劣性 A/B（SFT 学生，每题 8 次），含 vLLM 计数和打包统计 | [EXPERIMENT_LABELS_AB.md](../EXPERIMENT_LABELS_AB.md) |
| `round5/` | 第五轮：前缀打包下的第二次 GRPO 训练（每题 8 次评测），含对照原始源码和隐藏检查的结果 | [EXPERIMENT_GRPO_ROUND5.md](../EXPERIMENT_GRPO_ROUND5.md) |
| `hidden_checks/` | 隐藏行为检查的重新评分：每个检查在原始源码和带 bug 源码上的结果，以及 9 个评测池中每个相关通过补丁的结果 | [tasks/hidden_checks/README.md](../../tasks/hidden_checks/README.md) |
| `bench_rollout_2026-11-06.json` | rollout 吞吐随并发数的变化 | [BENCH_ROLLOUT_2026-11-06.md](../BENCH_ROLLOUT_2026-11-06.md) |
| `packing_round3_steps0-3.json` | 第三轮前 4 步上前缀打包的 token 节省和断点分类 | [PREFIX_PACKING.md](../PREFIX_PACKING.md) |
| `commit_map_20261111.json` | 2026-11-11 历史改写前后的 commit ID 对照，代码树不变 | |

## 每轮目录里的文件

| 文件 | 来源 | 内容 |
|---|---|---|
| `compare_*.json` | `scripts/compare_pools.py` | 按题配对的 bootstrap：两边的 pass@1、差值、95% 置信区间、结论、变好和变差的题数。`_drop_flagged` 剔除了有补丁信号的尝试 |
| `results.json`、`results.md` | `scripts/summarize_round.py` | 各评测池的 pass@k、状态分布、`edit_file` 成功率、篡改审计、阶段耗时和费用 |
| `steps.jsonl` | 驱动脚本 | 每个阶段的耗时 |
| `grpo_metrics.jsonl` | `scripts/grpo_harness.py` | GRPO 每一步：通过率、保留的组、prompt 摘要不一致数、logprob 差、TIS、clip 比例、熵、rollout 和训练用时 |
| `limit_events.jsonl` | `scripts/limit_events.py` | 被按题测试时限、命令内存上限终止的命令，以及 verifier 的超时和内存超限 |
| `original_check.json` | `scripts/review_patches.py --original` 加人工判断 | 每个评测池中与注入 bug 之前的源码一致、行为等价、没有恢复原有行为（轻微或严重）的补丁数，以及每个没有恢复的补丁和理由 |
| `vllm_*.json`、`packing_*.json` | `scripts/vllm_metrics.py`、`scripts/bench_turn_packing.py` | 第四轮每个评测池前后的 vLLM 计数快照及其差值（prompt token、前缀缓存命中），以及打包训练的 token 节省和断点分类 |
| `compare_eval_weak_*.json` | `scripts/mark_failures.py` 加 `compare_pools.py` | 把没有恢复原有行为的补丁算作失败后的配对比较（事后的敏感性分析） |

评测池本身（每次尝试的轨迹、请求记录和补丁）体积较大，留在 GPU 主机上。
