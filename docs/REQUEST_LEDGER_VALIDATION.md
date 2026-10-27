# 请求级预算审计：非法明细与已知超额不能被汇总掩盖

日期：2026-10-27；基线提交 `6b3ab66f3bcf23ad3738cf72eaa5057a0a4bb505`。本轮仅修复外部已保存结果的请求账本审计，不改变Agent策略、模型请求或在线Budget计量。

## 问题与证据

首先只读扫描现有59份运行结果：42 unknown、8 pass、9 fail，无异常。这些记录混有脚本化演示与历史模型尝试，计数不是模型成功率；没有发现它们实际出现本轮合成的损坏。

随后以第六轮off演示的有效result为来源，在内存副本和新目录中构造损坏请求账本。旧审计仅检查汇总类型与跨请求相加结果，存在以下反例：

| 输入 | 修复前 | 修复后 |
|---|---|---|
| 原始有效账本 | pass | pass |
| 请求缺input_tokens | KeyError，批次无报告中断 | unknown，保留该样本 |
| requests为null | TypeError，批次无报告中断 | unknown，保留该样本 |
| 一条负输入量由另一条正量抵消，总计仍对齐 | pass | unknown |
| 请求overrun_reasons明确非空，但汇总overrun=false | pass | fail |
| 实际输出超过请求预留输出，但汇总overrun=false | pass | fail |
| 请求实际输入+输出超过预估输入+预留输出 | pass | fail |

[corruption-comparison.json](../runs/iteration-20261027-06/corruption-comparison.json)记录来源路径/哈希、旧版本结果或异常类型、新版本结果和原因。`cases/`内的记录明确标为`synthetic_ledger_mutation`，不是新Agent运行、真实供应商故障或训练数据。原始来源没有被覆盖。

## 最小修复

只在`evaluation._budget()`的外部记录入口新增请求级检查：

- requests必须是列表，每行必须是对象。
- input_tokens、output_tokens、estimated_input_tokens、reserved_output_tokens必须是非负整数，布尔值不能冒充整数。
- overrun_reasons必须是字符串列表；缺失或非法字段属于证据不完整，返回unknown。
- 记录明确的超额原因，或者根据实际计数可计算出超出预留时，返回fail；不能只相信汇总overrun标志。
- 完整遍历请求后再判断，已知超额优先于另一条损坏记录，且与记录顺序无关。报告提供非法/超额请求的零基索引。

沿用原有三维验收及OpenJudge协议。没有宽泛捕获异常后默认成功，没有把unknown转成0分，也没有建立另一套通用schema框架。有效记录的调用、运行和预算行为完全不变。

## 验收事实

- 新增回归在修复前：**33 failed / 33 passed**；修复后同一文件：**66 passed**。原始JUnit分别保留为request-ledger-before.xml和request-ledger-after.xml。
- 完整CPU回归：**724 passed / 12 skipped**。跳过为可选Torch模块2、未配置固定mini-SWE6、可选OpenJudge跳过记录4；后者已在专用环境实际运行。
- 固定官方OpenJudge源码的专用合同环境：**84 passed / 0 skipped**。
- 内置与真实OpenJudge CLI均对7条合成记录完整输出报告，整体退出1符合预期；其中1 pass、3 unknown、3 fail。unknown在真实OpenJudge中为GraderError且不带score。
- [59份历史产物对照](../runs/iteration-20261027-06/historical-audit.json)：整体验收和预算维度均无变化，原文件字节/哈希未改。此兼容性结果不重新背书历史模型效果。
- 最终源码指纹、计数及证据SHA-256见[cpu-validation.json](../runs/iteration-20261027-06/cpu-validation.json)。

可在已有轻量CPU环境审计保存的合成输入，必须使用新输出路径以保留原证据：

```powershell
.venv/Scripts/python.exe -m budget_agent audit --input runs/iteration-20261027-06/cases --output runs/my-ledger-audit.json
```

退出1表示所选批次没有全部通过，不是命令崩溃。该命令不运行模型、不训练、不修改输入。

## 招聘与工程资料复核

本轮访问为 **2026-10-27 11:02:21–11:03:21 UTC / 北京时间19:02:21–19:03:21**。

| 官方来源 | 本轮实际证据与时效边界 |
|---|---|
| [百度 J96208](https://talent.baidu.com/jobs/detail/INTERN/a77dfe3a-0f3a-45b5-95dc-c7478f9d43e4) | 直接页只有导航壳，两轮官方域名搜索未重现之前正文。本轮未重新读到日期或岗位内容；此前7月21日属于历史记录。无法读取不能证明关闭，也不能确认HC仍在。 |
| [百度代码智能体大模型架构实习生 J101404](https://talent.baidu.com/jobs/detail/INTERN/397771ff-709c-42f5-9fd7-2e4112e0768d) | 官方正文可读，日期仍2026-07-21；可复现沙箱、轨迹清洗/质量、后训练与rollout要求未观察到实质变化。页面“2人”不是实时剩余HC。 |
| [Seed招聘](https://seed.bytedance.com/zh/seedearlycareer) | 正文可读，无发布日期；仍列Code Agent RL、Test-time Scaling、Long Horizon、Memory。2028年毕业符合页面范围，不代表具体团队有名额。 |
| [Anthropic：基础设施噪声与代码Agent评测](https://www.anthropic.com/research/infrastructure-noise) | 发布于2026-02-05；支持固定/披露资源、超时和环境。该文章不直接证明本轮账本算法，其作用是约束实验归因。 |

在可读来源中未发现需要改变项目主线的实质要求变化，也未确认新HC。请求级证据检查对应可复查轨迹与可信评测能力，不据此宣称招聘认可或模型效果提升。

## 能力边界与后续

本轮检查仍是记录内部一致性，不核对供应商账单、不认证日志真实性，也不保证任意损坏JSON的全部字段都能被审计。没有运行模型、训练、租机或核验实时云库存/账单；具身项目的100元授权不用于本项目。

只读审查另发现：人为构造`test_results.successful/failures`与顶层resolved互相矛盾时，现有质量解释未完整对账。未在59份历史记录中确认出现；本轮没有扩进共享verifier/报表语义修订，下一轮需要单独建立判据和兼容性回归，不能把此次请求账本修复称为所有评测字段已完备。

真实修复参考仍是优先项；原≤10元待执行授权仅适用已冻结的两阶段试跑。更新参考计划只冻结新源码，保留模型/任务/次数/额度及reflection=off、execution=sandbox，不能扩用于反思或安全新实验。没有值得改变的策略时不再凭旧开发题增加机制。
