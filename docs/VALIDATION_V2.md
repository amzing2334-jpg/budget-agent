# v2 验证记录：2026-10-21

本轮完成可训练策略、采样、独立奖励、SFT、精简 GRPO 及检查点保存代码。用户随后明确要求不要使用本机训练，已终止所有本项目训练进程并清理残留的工具容器。后续训练只在云端进行。

| 检查 | 观测结果 |
|---|---|
| 评分修复后核心环境自动化测试 | 115 passed，2 个训练依赖测试模块 skipped |
| 停止前包含训练依赖的完整测试 | 126 passed；评分修复后未在本机重跑模型优化测试 |
| 真实 0.5B 模型 base | 6 个合成评测任务通过 1 个，11093 输入加输出 token |
| 60 步 SFT | LoRA 参数哈希变化，保存后重载成功 |
| SFT 后评测 | 相同 6 个任务通过 6 个，6669 token |
| RL 第 1 组 | 奖励 [1,1,1,1]，无相对信号，跳过更新 |
| RL 第 2 组 | 错误记录奖励 [1,0,1,1] 并更新；事后查明评分解析 bug，整组更新无效 |
| RL 第 3 组 | 用户要求停止，未完成该组更新 |
| RL 最终 adapter 与评测 | 尚未生成；不能报告 RL 提升或可恢复的最终 RL 权重 |

第 2 组的 sample-01 私有测试全部通过，但 Docker 合并 stdout/stderr 时把 verdict 拼到测试进度同一行，旧解析器要求 marker 在行首，从而误记为失败。这一错误制造了虚假的组内奖励差异。该更新及后续受影响轨迹不能作为有效 RL 训练证据；原始日志保留，不改写为成功运行。

已修复：unittest 与 verdict 统一输出到 stdout；解析器识别被拼到行中的完整随机 marker。新增回归测试重放实际触发格式。没有重新启动训练，云端需要从有效 SFT adapter 重新采样。现存 base 与 SFT eval 的评分日志未发现同类不一致。

这些任务只有 threshold/range 两种模板，训练与评测仅分离数值参数。不能把 1/6→6/6 写成 RL 增益，不能外推到真实仓库、长程规划或 SWE-bench。

证据位于 `runs/learning-v2/`：`base-eval/result.json`、`sft/result.json`、`sft-eval/result.json`、`grpo/metrics.jsonl` 和各 episode 的轨迹、补丁、私有评分结果。SFT adapter 位于 `runs/learning-v2/sft/adapter/`。

本轮冻结协议为 `docs/TRAINING_PROTOCOL_V2.json`，哈希在 `project_manifest.json`。协议计划的 3 组 RL 及 RL 后评测未全部完成；用户停止是中止原因。

执行说明：`run-train` 包装器不识别本项目 JSON 日志的 step 字段，并把权重加载进度识别成 metric，因此 `train_outputs/sft-execution.json` 的通用解析字段不能作为结果；本记录依据项目自己的 `metrics.jsonl`、`result.json` 和检查点。技能附带的通用输出 writer 依赖文件缺失，标准摘要由项目证据直接整理。
