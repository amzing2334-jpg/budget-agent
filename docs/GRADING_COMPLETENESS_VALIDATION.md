# 评分完成性与失败记录：第五轮验收

检查日期：2026-10-27，Asia/Shanghai。本轮只运行 CPU 软件检查、临时 Python fixture 和固定假策略；没有加载真实模型、训练、调用付费 API 或租用主机。

## 为什么优先修这个问题

项目下一步仍是取得有独立评分的真实修复参考。参考运行不能只看命令是否退出，也不能把没有完成的评分变成训练奖励。审计发现了两条可复现的断点：

1. 私有 verifier 在导入阶段缺依赖，尚未执行测试，`_verified()` 却返回 `False, None`。`grade()` 将它保存成 `unresolved`，`sample_episode()` 因此生成 `reward=0` 的 `episode.json`。这会把没有奖励依据的轨迹交给后续组内归一化。
2. API 首请求失败时，runner 正确保留 `actual_models=[] / usage_sources=[]`。但 paired 要求每条记录的这两个集合与第一条完全一致，于是整个分析直接报错，其他有效配对也无法输出。

这些是软件反例，不是新模型失败率实测。现有真实 API/SWE 运行没有自动导入 learning 的入口，不能声称其已经污染训练。历史无效 GRPO 更新继续无效。

## 修改后的合同

| 情况 | 评分结果 | 训练和报告行为 |
|---|---|---|
| 独立测试完成且通过 | `True / resolved` | 有效奖励 1 |
| 独立测试完成，断言失败或测试函数内异常 | `False / unresolved` | 有效奖励 0 |
| 修改受保护文件 | `False / invalid_patch` | 明确的非法补丁拒绝，仍可作为负奖励 |
| verifier 缺依赖、提前退出、没有完成标记、损坏标记、进程未正常完成 | `None / grading_error` | 保存诊断、轨迹和补丁；不写可训练 Episode |
| 没有发现测试，或全部跳过 | `None / grading_error` | 无足够测试依据，不生成奖励 |
| Docker 本身无法启动 | 既有 `None / infrastructure_error` | 保持原行为 |
| SWE-bench 尚未接入官方评分 | 既有 `pending_official_evaluation` | 保持原行为 |

Local 与 Docker fixture 共用 `_verified()`；学习入口已有的 `resolved is None` 检查直接生效，没有新增重复过滤器或异常吞掉逻辑。`grading_error` 表示没有可用评分，不是根因判断：模型写出的源码语法错误也可能阻止 verifier 导入。当前会保守地终止该训练组；正式开展 RL 前，需要用基线校验与分阶段执行进一步区分模型错误和环境装配错误。不能把这一轮当成完整 reward attribution 的实现。

旧格式报告也已对齐：当原记录明确为 fixture 的 `resolved/unresolved` 状态且含 `test_results=None`、零测试或全部跳过，`verified_outcome()` 将它解释为未评分。聚合、HTML、配对共享这个判据。原始 `result.json` 字节不修改；没有该字段的旧记录不被凭空判无效，官方 SWE-bench 结果与非法补丁拒绝保持原语义。兼容问题由构造的旧格式反例证实，本轮未发现可据此报告的历史真实污染数量。

配对分析现在保留每侧原始 `actual_models / usage_sources`，增加明确的 `ineligible_reasons`。任一侧缺失模型或计量来源，该 pair 不进入胜负和均值；其他有效 pair 仍可分析。所有非空已知模型与计量来源继续执行全批一致性校验，冲突仍拒绝，不能用缺失记录掩盖不同模型比较。逆序输入的报告保持相同。

示例 [paired-contract.json](../runs/iteration-20261027-04/paired-contract.json) 是明确标注的 CPU 合成合同演示：两对匹配记录中一对可评分，另一对首请求失败，四条记录均保留，仅一对进入统计。这里的胜负和 token 差是测试数据，不能作为模型效果或费用收益。

## 验收证据

| 验收 | 修复前 | 修复后 |
|---|---|---|
| fixture grader 与真实 Local→假策略 Episode 链路 | 12 failed / 46 passed | environment + learning_rollout 58 passed |
| 首请求失败、缺字段、顺序与已知模型冲突 | 10 failed / 33 passed | paired 43 passed |
| 旧格式未完成评分与现有报告 | 3 failed / 36 passed | report 39 passed |
| report + paired + 两类 probe 消费方 | — | 120 passed |
| 最终完整 CPU 套件 | — | **561 passed / 8 skipped** |

修复前日志、JUnit 和文件 SHA-256 见 [cpu-validation.json](../runs/iteration-20261027-04/cpu-validation.json)。第一次完整回归 555/8 也保留在 `pytest-pre-legacy.xml`；发现旧格式兼容缺口并修改源码后才重新跑最终完整回归。8 项跳过为 2 个可选 Torch 模块和 6 个未配置固定 mini-SWE 上游路径的集成项。Docker 行为用 CPU 进程替身验证，不能冒充这轮已经启动真实 Docker。

独立审查还执行了真实临时 verifier，确认测试内异常仍是有效负奖励。为了避免新增 metadata 资格让原测试虚假通过，报告测试的正常 fixture 补充了明确模型身份，确保其配对断言仍在检验评分语义。

源码指纹：`ef11d510f9ad840b30c9cd1143d249c0c1d2445d66aff49b0545a316ffd7d67a`。

## 官方岗位与工程资料复核

检索时间：2026-10-27 **12:05 北京时间**。未发现相较前轮的实质要求变化，未确认新增或剩余 HC。

| 官方来源 | 发布日期与时效边界 | 对本项目的要求 |
|---|---|---|
| [百度策略实习 J96208](https://talent.baidu.com/jobs/detail/INTERN/a77dfe3a-0f3a-45b5-95dc-c7478f9d43e4) | 2026-07-21；直达页只有导航/登录，内容通过约两周前的官方搜索索引读取 | 行为分析、上下文与工具策略、成本、指标和消融 |
| [百度架构实习 J101404](https://talent.baidu.com/jobs/detail/INTERN/397771ff-709c-42f5-9fd7-2e4112e0768d) | 2026-07-21；正文可读，“2人”不是剩余 HC 确认 | 可复现沙箱、轨迹质量、SFT/OPD/RL 基础设施、性能 |
| [Seed 校招](https://seed.bytedance.com/zh/seedearlycareer) | 无可见发布日期；不是某个团队实时 HC 清单 | Code Agent RL、长程任务、Memory 与机器学习系统 |
| [Anthropic Agent 评测说明](https://www.anthropic.com/research/demystifying-evals-for-ai-agents) | 2026-01-09；本轮重读不是新发布 | 区分 task、trial、轨迹、grader 与最终环境结果 |

由这些资料及项目失败证据得出的取舍是：先让评测和奖励有真实依据，再讨论上下文策略或 RL 收益。本轮提高的是软件结果的可信度，尚不证明真实修复率、上下文效率或训练效果提高。

## 云端状态与下一步

本轮 Chrome 复核没有运行实例，库存页全部显示售罄，没有发现之前 504 后迟到分配的主机；旧请求的账单仍未核实。未提交新租用，真实 API 请求为 0。

既有这次参考试跑的授权仍保持：API≤5元，主机≤5元/小时、最多1小时，总额≤10元；用户批准的6核/60.9GB备选有效，不能扩展为新实验。模型、两阶段任务、token额度和尝试数均不变。[更新的待执行计划](../runs/iteration-20261027-04/reference-plan.json)冻结本轮源码，历史方案不覆盖。

资源恢复后先核实实例及账单、配置自动释放、验证独立正负评分对照，再跑一次 `slugify-digits` 协议修复；通过后才运行一次 `astropy__astropy-12907`。若未完成评分，立即保留失败并停止，不用本轮新代码继续生成零奖励样本。获得有效真实参考后才开展固定任务和预算的上下文单因素对照。
