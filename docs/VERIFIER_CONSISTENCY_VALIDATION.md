# 保存的评分证据一致性：第八轮验收

检查日期：2026-10-27，Asia/Shanghai。本轮仅进行本地 CPU 软件检查和保存结果审计，没有模型调用、训练、租机或云端库存/账单复核。

## 问题与验收目标

上一轮留下的质量判定缺口可以复现：有效结果的副本仍保留顶层 `resolved=true`，但测试明细写为 `successful=false`、非零失败数或负数计数时，旧 `verified_outcome()` 仍返回成功。删除 `tests_run` 则会抛出 KeyError，使混合批次不能完整输出报告。

这些是外部保存记录的合成损坏，不是已确认的真实模型或供应商事故。先冻结并只读检查上一轮相同的 59 份历史结果，再构造新副本；没有覆盖任何原始实验。验收目标是让相互矛盾或不完整的评分证据保持未评分，同时保留一致的成功、真实测试失败、补丁拒绝和官方评分语义。

| 保存记录的情况 | 修复前 | 修复后 |
|---|---|---|
| 原始一致正例 | pass | pass |
| 顶层成功，但 successful=false | pass | unknown |
| 声称成功，但 failures=1 | pass | unknown |
| tests_run=-1、skipped=-2 | pass | unknown |
| 缺少 tests_run | KeyError | unknown，批次继续输出 |
| test_results=null | unknown | unknown |
| resolved=true，但状态为 unresolved | pass | unknown |
| 一致的正常测试失败 | fail | fail |

[comparison.json](../runs/iteration-20261027-07/comparison.json) 保存来源、哈希、前后结果；`cases/` 中的八条副本明确标为 `synthetic_verifier_mutation`，不能用于宣称模型成功率或当作新训练轨迹。

## 修改与边界

生产修改集中在 `report.verified_outcome()`：

- 明确的 resolved/unresolved/invalid_patch 状态必须与顶层布尔结论相符；沿用 evaluation.resolved 与顶层结果的一致性检查。
- 对 resolved/unresolved 且明确提供 test_results 的记录，要求 verdict 为对象，tests_run/skipped 为非负整数，至少有原判据认可的非跳过测试，successful 为布尔值并与总结一致。
- 若提供 failures/errors，它们必须是非负整数；非零计数不能与成功声明共存。缺少这两个可选字段的旧记录继续兼容。
- 聚合、HTML、配对和内置/OpenJudge 审计继续共用这个入口。unknown 不进入有效评分或配对胜负；真实 OpenJudge 保留为不带 score 的 GraderError。

生产代码净增 14 行，无宽泛异常捕获、字符串/布尔计数强制转换或静默成功。CI 的真实 OpenJudge 任务纳入新合同文件。

校验不能擅自创造 unittest 的规则：失败子测试数量可能超过 tests_run；unexpectedSuccess 可以使 successful=false，同时 failures/errors 都为零。已实际执行这两类 CPU 测试，原始输出见 [unittest-semantics.json](../runs/iteration-20261027-07/unittest-semantics.json)。没有使用“失败数为零便成功”或“失败数不能超过测试数”的错误推断。

本轮只修复离线结果解释；在线 `environment._verified()`、学习奖励生成、模型策略和执行环境未改。没有详细 verdict 的旧汇总仍保留原有兼容行为；官方 SWE-bench 使用自己的评分记录，不套 fixture 字段。此审计不重跑原 verifier、不证明日志真实，也不覆盖任意损坏 JSON 的所有字段。

## 验收结果

| 验证 | 结果 |
|---|---|
| 同一新增测试文件，修复前 | 28 failed / 9 passed / 1 skipped |
| 同一新增测试文件，修复后 | 37 passed / 1 skipped |
| 完整 CPU 套件 | **761 passed / 13 skipped** |
| 固定官方 OpenJudge 源码的专用环境 | **122 passed / 0 skipped** |
| 八条合成记录，内置及 OpenJudge 实际 CLI | 均完整落盘：1 pass、6 unknown、1 fail；退出 1 符合预期 |
| 固定 59 份历史产物 | 原文件哈希不变；质量、整体验收与 verified_outcome 均不变 |

完整套件的跳过记录是可选 Torch 模块 2、未配置固定 mini-SWE 集成 6、可选 OpenJudge 5；后者已在专用环境实际运行。59 份历史结果混有演示和模型尝试，其计数不能解释为模型成功率。全部 JUnit、源码指纹和 19 份证据文件的 SHA-256 见 [cpu-validation.json](../runs/iteration-20261027-07/cpu-validation.json)。

可在已有轻量环境复核保存的合成样本，用新路径输出：

```powershell
.venv/Scripts/python.exe -m budget_agent audit --input runs/iteration-20261027-07/cases --output runs/my-verifier-audit.json
.venv/Scripts/python.exe -m pytest -q tests/test_verifier_consistency.py
```

首个命令退出 1 表示批次存在 fail/unknown，不是程序崩溃。不会调用模型或修改输入。

## 官方需求复核

招聘与工程资料访问时间：**2026-10-27 15:08:21–15:09:02 UTC / 北京时间 23:08:21–23:09:02**。无可确认的新 HC，也未发现足以改变项目主线的招聘要求变化。

| 官方来源 | 发布日期与本轮事实 | 对本轮的启发与限制 |
|---|---|---|
| [百度代码智能体架构实习 J101404](https://talent.baidu.com/jobs/detail/INTERN/397771ff-709c-42f5-9fd7-2e4112e0768d) | 正文可读，日期仍 2026-07-21；要求可复现沙箱、轨迹质量、后训练/rollout，每周 4 天以上、6 个月以上 | 可复查的结果链路有求职相关性；“2人”不等于当前剩余 HC，也不证明本项目已获招聘认可 |
| [百度 J96208](https://talent.baidu.com/jobs/detail/INTERN/a77dfe3a-0f3a-45b5-95dc-c7478f9d43e4) | 直接页为导航壳，两次官方域名查询未重现正文 | 无法确认其内容、关闭状态或剩余 HC |
| [Seed 人才招聘](https://seed.bytedance.com/zh/seedearlycareer) | 无可见发布日期；仍列 Code Agent RL、Test-time Scaling、Long Horizon、Memory；实习毕业时间为 2027 年 9 月及以后 | 用户 2028 年毕业符合列出的范围；具体团队名额未确认 |
| [SWE-bench 官方评测指南](https://www.swebench.com/SWE-bench/guides/evaluation/) | 无可见发布日期；明确区分生成报告、通过要求测试、正常失败、空补丁及无法评分 | 运行命令成功不等于修复成功；矛盾证据→unknown 是本项目的验收设计，不是该指南直接规定的算法 |
| [OpenJudge 固定源码](https://github.com/agentscope-ai/OpenJudge/blob/c2fdc38314cd971b03d4d9333cd2c9f24d85c27f/openjudge/graders/schema.py) | 使用既有 2026-10-20 提交，未追踪最新 HEAD | 保留数值评分与评分错误的类型区别；不通过加权分数掩盖缺证据 |

[CPython 3.11 unittest 源码](https://github.com/python/cpython/blob/3.11/Lib/unittest/result.py)也在本轮读取，用于核对 subtest 与 unexpectedSuccess 的计数含义；分支链接可能变化，最终行为以本轮实际 CPU 输出和测试为证据。

## 未完成问题与下一步

独立审查发现一个原有完成性限制：一个方法包含一个跳过子测试和一个正常通过子测试时，unittest 可返回 tests_run=1/skipped=1/successful=true；现有在线与离线完成性判据会保守解释为未评分。本轮未改变该判据，复现见 [known-subtest-limitation.json](../runs/iteration-20261027-07/known-subtest-limitation.json)。因此不能宣称已完整支持所有 unittest 组合。是否修复应与真实任务相关性一起判断，不能只为增加测试数量继续扩展。

项目优先缺口仍是有效的真实模型修复与独立评分，其次才是固定预算的三条件反馈实验。此前已批准但未执行的 ≤10 元参考试跑保持原范围；[本轮参考计划](../runs/iteration-20261027-07/reference-plan.json)仅更新源码指纹，不改变模型、题目、次数、token 上限、off/sandbox 策略或主机时限。云端库存和账单本轮没有重新核验，不能把此前售罄记录当作当前状态。具身智能项目预算不适用于本项目。

本轮提升的是已保存评测证据的可靠解释；不是新的真实修复成绩、Agentic RL 效果或企业生产认证。
