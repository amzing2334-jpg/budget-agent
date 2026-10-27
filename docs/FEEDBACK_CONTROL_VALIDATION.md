# 失败证据重呈：补齐反馈策略的简单对照

日期：2026-10-27。基于 `65e089611763ee4e6f95c28b690d8f8eda5d4051`，本轮只补上一轮明确留下的实验条件：`reflection_policy=evidence`。没有真实模型请求、训练、租机或新增费用。

## 为什么先补这一项

上轮实现了关闭/生成反思两种策略，但二者的差异同时包括额外调用、额外输入、信息重呈和生成式诊断。仅比较这两种策略，无法判断更简单的原始证据保留是否同样有用。

本轮加入独立的证据重呈条件，不改变默认设置。三种策略都收到正常的工具反馈；`off` 不是没有反馈的 Agent。

| 策略 | 失败后的额外动作 | 后续输入 | 费用与解释 |
|---|---|---|---|
| `off` | 无 | 原上下文策略选择的正常历史 | 基线，不增加固定反馈记忆 |
| `evidence` | 确定性截取当前失败观察的末尾 | 同一记忆位置额外保留带来源的原文摘录 | 无生成调用；每次再输入正常计费 |
| `failure` | 基于当前可见历史调用模型生成诊断 | 同一记忆位置保留生成的假设 | 反思输入、输出和之后的记忆输入均计费 |

这是**同一个总预算上限下的策略比较**，不是同实际消耗、同信息量或纯推理质量的实验。反思会读取更广的可见历史并选择/解释信息，证据条件只取当前错误尾部；完整反思提示过长时可能跳过，而证据摘录仍可能可用。这些差异必须保留，不能凭空扣除未发生的 token 来制造表面公平。

## 实现与信息边界

- `reflection_policy` 新增 `evidence`；沿用相同失败触发器、`max_reflections`、最后一步不干预和权限拒绝不干预规则。默认仍是 `off`。
- `EvidenceNote` 只从当前失败 Event 构造。保存 event ID、来源 revision、工具、原始输出字符数、摘录起点和原始后缀；没有检索私有评分、Task metadata 或新资料。
- `reflection_max_output_tokens` 在 evidence 条件下是**原文摘录正文**的计数上限，与生成条件的文本额度对应。元数据、JSON 转义和消息包装额外占用实际输入额度。极小额度可能只能保存空摘录，此时 offset/truncated 仍明确，不能声称错误文字可见。
- 摘录作为 `user` 数据，kind 为 `failure_evidence`，包含不可信声明和 stale 标记。修改代码不更新来源 revision；里面的命令或 JSON 不自动执行。全文保留在原轨迹，可用现有 `recall` 查询。
- 记录为 `status=replayed`，生成调用数为 0；没有虚构请求或 usage。后续所有模型输入继续走同一 Budget。若记忆挤占上下文/剩余预算，会明确终止，不静默删除记忆。
- HTML / Markdown 增加 Feedback policy 和 Memory notes，避免把 evidence 的0次反思调用误读成关闭干预。
- `paired --factor reflection_policy` 已支持 off/evidence、evidence/failure；其他预算、次数、输出上限、执行策略或代码指纹不同仍拒绝配对。

这仍是推理时反馈优化，不是参数更新或 Agentic RL。上述数据边界测试也不等于模型抗提示注入能力验证。

## 官方岗位与研究复核

访问窗口为 **2026-10-27 06:57:04–06:57:26 UTC / 14:57:04–14:57:26 北京时间**。这是有限来源复核，不是全市场扫描或岗位剩余名额核验。

| 来源 | 发布日期与本轮可见性 | 结论 |
|---|---|---|
| [百度 Coding Agent 策略算法实习生 J96208](https://talent.baidu.com/jobs/detail/INTERN/a77dfe3a-0f3a-45b5-95dc-c7478f9d43e4) | 仍为2026-07-21；直接页为导航壳，官方域名搜索索引可读，但抓取标记两周前 | 仍强调错误恢复、反馈优化、成本和策略消融，未确认新HC |
| [百度代码智能体大模型架构实习生 J101404](https://talent.baidu.com/jobs/detail/INTERN/397771ff-709c-42f5-9fd7-2e4112e0768d) | 官方正文可读，仍为2026-07-21；页面人数不是实时剩余HC | 可复现沙箱、轨迹质量、后训练与rollout要求未观察到实质变化 |
| [字节 Seed 招聘](https://seed.bytedance.com/zh/seedearlycareer) | 官方正文可读，未见发布日期 | 仍列Code Agent RL、Test-time Scaling、Long Horizon、Memory；2028年毕业符合页面实习毕业范围，不等于具体团队有名额 |

[Reflexion v4](https://arxiv.org/abs/2303.11366v4)首发2023-03-20、v4更新2023-10-10，讨论语言反馈、记忆与反馈消融。本项目的原始证据重呈控制组是据此作出的实验设计推论，不是论文已经证明其有效。

[Anthropic：Quantifying infrastructure noise in agentic coding evals](https://www.anthropic.com/research/infrastructure-noise)，页面发布日期2026-02-05，指出基础设施资源会影响代码Agent评测。后续三条件真实实验除模型、任务、预算外，还应冻结CPU/RAM、超时、并发与镜像。该资料没有验证本项目的反馈机制。

未观察到核心招聘要求的实质变化，也未确认新增HC。补控制组对应的是岗位要求的实验归因能力，不能把本轮软件实现描述成模型效果进步。

## 已完成验证

- 完整 CPU：**684 passed / 11 skipped**。跳过范围与上轮相同：可选Torch模块2、未配置的固定mini-SWE6、可选OpenJudge跳过记录3。
- 专用真实 OpenJudge：**43 passed / 0 skipped**，使用上一轮固定官方源码与合同依赖；本轮增加evidence配置参数化后再验证。
- 新增覆盖Unicode/多字节/极小摘录额度、原事件不变、来源和版本、私有评分不泄漏、权限与最后步边界、计数与配对条件。
- 实际临时仓库反例：在full上下文下，设置恰可供off执行第二次请求的总额度，evidence因额外记忆明确在第一步后`budget_exhausted`。没有第二次免费调用，也没有假造超额。
- 首请求off/evidence消息完全相同；第二次请求移除新增记忆后相同。公开失败之后独立grader只在终局执行一次。
- 只读审查发现演示读取UTF-8轨迹未显式指定编码，已修复；无生产代码兼容回退。没有修复前失败回执，本轮测试是新增功能实现后的验证。

[固定动作三条件演示](../runs/iteration-20261027-05/demo-final/report/report.html)中三次均执行预定修复动作；不存在模型根据反馈选择不同动作。计数器是项目的`byte`估算，不是服务商token或费用实测。逐臂账本、记忆、三组配对、完整原始记录与限制见 [summary.json](../runs/iteration-20261027-05/demo-final/summary.json)。最终计数和SHA-256见 [cpu-validation.json](../runs/iteration-20261027-05/cpu-validation.json)。

复现入口（已有CPU轻量环境）：

```powershell
.venv/Scripts/python.exe scripts/demo_feedback_control.py --output runs/my-feedback-control
.venv/Scripts/python.exe -m budget_agent.paired --runs runs/my-feedback-control --factor reflection_policy --baseline evidence --candidate failure --output runs/my-feedback-control-pair.json
```

真实运行只需在同一审核过的配置上分别设置 `--reflection-policy off/evidence/failure`；这不是新模型费用授权，也不代表现有SWE清单已经补齐可信的公开测试命令。

## 后续优先级

先完成尚未执行的真实修复参考，取得有效补丁、独立评分和完整用量。原≤10元参考授权仅用于既定两阶段协议/开发题试跑，保持reflection=off，不扩展到三臂新实验。本轮没有核验实时云库存或账单，不能沿用先前售罄截图声称当前状态。

有真实参考后，再冻结开发任务、模型、预算与环境，运行三条件对照；报告所有干预/跳过、真实消耗、修复率和退化案例，保留未评分记录。不继续用同一道旧开发题选策略或调摘录上限；最终测试集不用于调参。本轮把下一步实验准备完整，仍没有反思效果、费用节省或泛化能力的新证据。
