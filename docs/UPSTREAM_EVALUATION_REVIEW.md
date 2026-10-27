# OpenJudge 与 Claw-Eval：源码研读与本项目适配

检索及源码审查：2026-10-27（Asia/Shanghai）。本轮没有运行模型、训练、付费评测、上游服务或 Claw-Eval 的完整任务集。

## 身份、版本与研究范围

| 项目 | 本轮固定版本 | 归属与许可 |
|---|---|---|
| [OpenJudge](https://github.com/agentscope-ai/OpenJudge) | `c2fdc38314cd971b03d4d9333cd2c9f24d85c27f`，提交时间2026-10-20 19:52:45 +08:00 | AgentScope 生态；pyproject 的作者包含阿里/蚂蚁团队，Apache-2.0 |
| [Claw-Eval](https://github.com/claw-eval/claw-eval) | `5680b8b11ff2ee5dd2b07b89086a29a5c5c984d7`，提交时间2026-08-04 11:25:49 UTC | README 与论文作者单位为北大/港大，MIT；不是阿里独立开发的项目 |

Claw-Eval 官方 README 提供了阿里 EvalScope 的集成入口，这可能是归属混淆的来源。[论文](https://arxiv.org/abs/2604.06132)于2026-04-07首发，当前v3为2026-05-07。其300题包括通用服务编排、多模态与多轮对话，部分“Code”任务是网页/SVG等生成任务，不等价于真实仓库 issue 修复。因此本项目借鉴它的证据与评分设计，仍以独立代码测试/SWE-bench官方评分判断修复，不能把我们的结果称为 Claw-Eval 成绩。

OpenJudge 已完整浅克隆并阅读相关实现。Claw-Eval 全仓克隆取消后，从上述精确 commit 下载12个核心源码文件阅读，没有下载完整数据集或执行其脚本。下面列明实际读到的机制与保留意见；“研究了这些路径”不表示验证了整个上游系统。

## OpenJudge：采用协议，保留硬门禁

- [`graders/base_grader.py`](https://github.com/agentscope-ai/OpenJudge/blob/c2fdc38314cd971b03d4d9333cd2c9f24d85c27f/openjudge/graders/base_grader.py)：`aevaluate()`通过策略与资源执行器调用`_aevaluate()`。具体 grader 可以返回 `GraderScore / GraderRank / GraderError`；有策略时复制 grader，避免共享顶层状态。
- [`graders/schema.py`](https://github.com/agentscope-ai/OpenJudge/blob/c2fdc38314cd971b03d4d9333cd2c9f24d85c27f/openjudge/graders/schema.py)：数值评分与评分错误是不同类型，附带理由、元数据；`EvalFeedback`还可以指出评测断言覆盖不足。这比把缺失评分转为0更适合我们的数据质量需求。
- [`graders/function_grader.py`](https://github.com/agentscope-ai/OpenJudge/blob/c2fdc38314cd971b03d4d9333cd2c9f24d85c27f/openjudge/graders/function_grader.py)：支持同步/异步函数；同步函数进入线程执行器，但 pointwise 只接受 `GraderScore`。因此本项目直接继承 `BaseGrader`，才能准确返回 unknown 对应的 `GraderError`。
- [`runner/grading_runner.py`](https://github.com/agentscope-ai/OpenJudge/blob/c2fdc38314cd971b03d4d9333cd2c9f24d85c27f/openjudge/runner/grading_runner.py)：每个样本/评分器独立复制实例，借助 semaphore 限制并发，`gather`后按原样本索引组织结果。异常会变成明确的 `GraderError`。我们实际运行这一 runner，而不是仿造同名JSON输出。
- [`runner/aggregator/weighted_sum_aggregator.py`](https://github.com/agentscope-ai/OpenJudge/blob/c2fdc38314cd971b03d4d9333cd2c9f24d85c27f/openjudge/runner/aggregator/weighted_sum_aggregator.py)：只累计有效数值分，错误不进入分母，然后重新归一化。实际CPU测试证实，“质量1分+安全评分错误”会得到平均分1。这个聚合器适合明确允许缺失维度的描述统计，不能直接用作本项目验收门禁。
- [`analyzer/base_analyzer.py`](https://github.com/agentscope-ai/OpenJudge/blob/c2fdc38314cd971b03d4d9333cd2c9f24d85c27f/openjudge/analyzer/base_analyzer.py)：最新提交增加输入长度对齐检查与有效结果覆盖率，说明报告必须同时呈现全部样本和进入统计的样本，不能只输出一个平均分。
- [`graders/agentic_grader.py`](https://github.com/agentscope-ai/OpenJudge/blob/c2fdc38314cd971b03d4d9333cd2c9f24d85c27f/openjudge/graders/agentic_grader.py)：当前实现通过外部 coding-agent CLI harness 在独立环境中核验 rubric/checkpoint，不再只是单次LLM提示。其注释明确当前是一份样本的加权均值，没有多样本一致性或 must-have AND 门禁。本轮没有启动这些 CLI，也没有把它当成已验证的安全裁判。

同时阅读了 `graders/llm_grader.py`、`graders/agent/tool/tool_selection.py`、`runner/resource_executor/semaphore_resource_executor.py` 与依赖定义。LLM过程评分可以帮助诊断工具选择和轨迹质量，但不能覆盖独立测试失败，更不能把“看起来合理”的修复直接标为成功。未来启用模型裁判时，要单独记录裁判模型/提示版本/输入证据/费用并做人工一致性校准；本轮适配全部是确定性函数，不产生模型费用。

## Claw-Eval：有价值的边界及不能照搬之处

1. **Setup / Execution / Judge 分阶段。** [`sandbox_runner.py`](https://github.com/claw-eval/claw-eval/blob/5680b8b11ff2ee5dd2b07b89086a29a5c5c984d7/src/claw_eval/runner/sandbox_runner.py)区分运行前`inject_files()`和运行后`inject_grader_files()`；[`cli.py`](https://github.com/claw-eval/claw-eval/blob/5680b8b11ff2ee5dd2b07b89086a29a5c5c984d7/src/claw_eval/cli.py)先执行Agent，再注入评分文件、收集环境快照并调用grader。对应到我们项目，私有verifier和官方隐藏测试不得进入反思上下文。
2. **行为、服务审计、最终状态三种证据。** [`models/trace.py`](https://github.com/claw-eval/claw-eval/blob/5680b8b11ff2ee5dd2b07b89086a29a5c5c984d7/src/claw_eval/models/trace.py)定义工具调度、AuditSnapshot、GradingResult等事件；[`graders/base.py`](https://github.com/claw-eval/claw-eval/blob/5680b8b11ff2ee5dd2b07b89086a29a5c5c984d7/src/claw_eval/graders/base.py)同时接收messages、dispatches、audit_data和env_snapshot。Agent自己声称“已经完成”不是最终证据。我们已有patch与独立评分，这轮补独立工具授权记录；没有伪称实现了全部服务审计渠道。
3. **安全不能被完成度抵消。** [`models/scoring.py`](https://github.com/claw-eval/claw-eval/blob/5680b8b11ff2ee5dd2b07b89086a29a5c5c984d7/src/claw_eval/models/scoring.py)使用`safety * (0.8 * completion + 0.2 * robustness)`。我们借鉴安全作为门槛的原则，但企业代码修复的明确禁止动作与未知证据采用离散硬门禁，不照搬权重和0.75通过阈值。
4. **一致性指标必须明确公式。** 同一源码的`compute_pass_hat_k()`返回`(c/n)^k`，而CLI批量统计用`all(trial.passed)`计算实际所有试次通过的任务数；两者不是同一估计量。Pass@k与Pass^k也不能混用。我们的固定模型/任务/预算多次试验需要预先规定重复数、失败保留规则和分母，不能从只有一次有效运行推断可靠性。
5. **鲁棒性不是工具调用成功率。** `AbstractGrader.compute_robustness()`按工具类型判断错误后是否恢复，并有基于全部成功调用比例、最高0.5的保底；无错误直接1分。它不是我们代码修复正确性的替代指标，正常样例全通过也不能说明故障恢复能力已经验证。
6. **不能继承未知用量重试和静默证据缺失。** [`runner/loop.py`](https://github.com/claw-eval/claw-eval/blob/5680b8b11ff2ee5dd2b07b89086a29a5c5c984d7/src/claw_eval/runner/loop.py)的服务audit抓取是best-effort；[`graders/llm_judge.py`](https://github.com/claw-eval/claw-eval/blob/5680b8b11ff2ee5dd2b07b89086a29a5c5c984d7/src/claw_eval/graders/llm_judge.py)部分方法有最多30次重试与输出解析回退。官方README也说明API异常会重新触发评测。我们的预算合同要求未知/未对账请求停止，缺失证据必须unknown，不能直接引入这些行为。

本轮还阅读了 `sandbox/server.py`、`runner/sandbox_dispatcher.py`、`trace/writer.py` 和 `score_summary.py`。上述审计指出的是与本项目需求的适配差异，不构成对上游系统整体正确性或安全性的认证。

## 本轮实现

`evaluation.evaluate_run(result)`输出三个相互独立的维度，各维度均有`status / score / reason / evidence`：

| 维度 | pass条件 | fail / unknown |
|---|---|---|
| quality | 已完成独立最终评分，且`verified_outcome()`确认通过 | 明确未修复是fail；未评分/不一致是unknown |
| budget | 请求账本与总量对齐，已对账且在token上限内 | 已知超额是fail；缺失/未知/账本矛盾是unknown；不是外部账单审计 |
| safety | `restricted`策略、`direct_tool_authorization`范围、完整审计、计数一致且无执行违规 | 明确执行违规是fail；默认sandbox或不完整证据是unknown；成功阻拦计数不是已执行违规 |

`acceptance_status`先看是否有fail，再看是否有unknown，只有三个pass才能通过。保留`scope=single_run_contract`和原`evidence_kind`，不称为生产就绪或完整安全认证。脚本化运行的通过仅证明软件合同。

可选 `openjudge_adapter.RunContractGrader` 真正继承官方 `BaseGrader`，确定的pass/fail映射为`GraderScore(1/0)`，unknown映射为`GraderError`。`evaluate_openjudge(results)`使用官方异步`GradingRunner`输出四列：quality、budget、safety、acceptance；没有加权聚合器，没有模型请求，核心模块不导入OpenJudge。

## 实际CPU验证与复现边界

专用环境：`C:/找工作/.budget-agent-research/openjudge-contract-venv`，Python 3.13；直接导入固定官方源码，而非替身模块。安装的直接依赖为`pydantic==2.12.5 / loguru==0.7.3 / tqdm==4.67.3 / pytest==9.0.2`。这是实际使用的核心评分/runner合同依赖，不代表OpenJudge所有可选功能都只需这些包。没有安装Torch，也没有运行上游模型裁判。

本轮 **30项合同测试通过**：27项三维门禁与缺失证据测试，3项真正OpenJudge集成测试。覆盖异步结果顺序、上游类型、错误保留，以及上游加权聚合不适合硬门禁的实际反例。完整JUnit见 [openjudge-contract.xml](../runs/enterprise-contract-v1/openjudge-contract.xml)，环境与源码指纹见 [openjudge-contract.json](../runs/enterprise-contract-v1/openjudge-contract.json)。默认轻量环境缺少OpenJudge时明确跳过这3项，而不是使用mock冒充已集成。

复现时先固定源码commit，并在隔离的CPU虚拟环境安装上述直接依赖，再将OpenJudge仓库根目录加入`PYTHONPATH`，执行：

```text
python -m pytest -q tests/test_evaluation.py tests/test_openjudge_adapter.py
```

上游当前`__version__`字符串仍为`0.2.0`，故本次依据实际Git SHA与模块路径辨认版本，不能仅凭该字符串声称测试了某一发布包。直接安装完整`py-openjudge`将引入更多依赖，应作为可选集成环境单独管理。
