# 批量尝试中的用量异常停止

观察日期：2026-10-27（第三轮）。目标是准备真实 API 参考实验时，避免一条账本异常后继续自动启动其他任务、策略或重复次数。

## 发现与修复

单次 runner 已在 ModelError 后保存未知用量并停止，但 CLI 的三层批量循环原先只把最终退出码设为失败。在 2 个任务 × 2 个策略 × 2 次重复的 CPU 场景中，第一条返回 usage_unknown 后，旧 CLI 仍启动全部 8 次。它没有重试同一个 HTTP 请求，却会创建后续新的尝试，因此单请求无重试不足以约束整轮费用。

现在当 `usage_unknown`、`budget.unreconciled_request` 或 `budget.overrun` 任一成立时，CLI 生成已有部分报告，立即返回 1，不再调度后续尝试。已完成结果、失败请求账本和原始轨迹保留。正常已知用量下的 budget_exhausted、有效 unresolved 继续参与完整对照；interrupted 仍返回 130。没有加入静默重试、自动恢复或额外调度框架。

## 验收

- 独立回归修复前为 **8 failed / 3 passed**，修复后 **11 passed**。修复前仅保留工具输出，不事后制造 red 日志。
- [修复前复现](../runs/iteration-20261027-02/batch-stop-before.json)：8 次计划，实际启动 8 次；首条异常后继续 7 次。
- [修复后复现](../runs/iteration-20261027-02/batch-stop-after.json)：同一计划只启动 1 次，生成 1 次报告，随后停止。
- [批量回归日志](../runs/iteration-20261027-02/batch-tests-green.log)及 [JUnit](../runs/iteration-20261027-02/batch-tests-green.xml)覆盖首次/中途异常、多任务 run、完整正常矩阵及中断。真实 runner 的预设 CPU 模型集成另外验证了先前结果和未知账本写盘、HTML 保存、后续调用未发生、任务源文件未变。
- 完整 CPU 套件：**516 passed / 8 skipped**，见 [回执](../runs/iteration-20261027-02/cpu-validation.json)。8 个跳过仍为可选训练栈的两个模块、未指定固定上游 checkout 的六项 mini-SWE 集成。
- 新参考 API 的显式思考模式开关经过本地假 HTTP 服务验证：未指定时省略、false/true 顶层传递、max_completion_tokens 保持预算预留值；CLI 覆盖和非法类型也已检查。

这些结果只验证软件行为，不能用“少启动 7 次预设调用”推算真实费用节省。未知请求可能已计费，停止不会撤销它；正常已知预算耗尽会继续后续独立尝试，`max_cost_usd` 仍是每次任务上限，不是全批共享钱包。没有新增模型修复成绩。

后续真实试跑的配置、官方来源、任务选择和预算提案见 [REFERENCE_PILOT_2026-10-27.md](REFERENCE_PILOT_2026-10-27.md)。
