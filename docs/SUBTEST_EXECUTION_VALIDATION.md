# 子测试执行证据与奖励入口：第九轮验收

日期：2026-10-28，Asia/Shanghai。本轮运行标准 Python 子进程、CPU 软件测试及固定 TokenPolicy 的数据链路检查；没有真实模型、权重更新、训练、实际 Docker 启动或付费租用。

## 从已知边界到实际影响

上一轮记录了一个可复现的限制：unittest 的 tests_run 统计方法，skipped 也会包含子测试，因此 `tests_run > skipped` 不能可靠判断是否已有非跳过结果。

本轮先在原代码上执行 12 个真实临时 verifier，再检查 Local grader → 固定策略 → Episode 链路。混合通过/跳过、失败/跳过、异常/跳过，以及嵌套通过/跳过都被旧判据解释为未评分；应为 1 或 0 的有效奖励均被拒收。这是可控制的 CPU 反例，不是发现了新的真实模型失败或历史训练污染。

| verifier 内容 | 原判据 | 本轮结果 |
|---|---|---|
| 通过子测试与跳过子测试共存 | 未评分 | 有效成功 |
| 失败子测试与跳过子测试共存 | 未评分 | 有效失败 |
| 异常子测试与跳过子测试共存 | 未评分 | 有效失败 |
| 嵌套通过子测试与跳过子测试共存 | 未评分 | 有效成功 |
| 所有子测试或整个方法跳过 | 未评分 | 保持未评分 |
| 空测试集、导入时直接零退出 | 未评分 | 保持未评分 |
| 普通通过、普通失败、expectedFailure、unexpectedSuccess | 原 unittest 语义 | 保持原语义 |

[before.json](../runs/iteration-20261028-01/before.json) 保存原输出；[comparison.json](../runs/iteration-20261028-01/comparison.json) 保存同一 verifier 文件哈希、前后结果和原始子进程输出。源文件位于该目录的 `fixtures/`，不是模型生成的修复。

## 实现选择

新增小型 `verification.py`，包含两个直接职责：

- `VerificationResult` 使用公开的 addSuccess/addSubTest 回调记录成功或完成的子测试结果，结合 unittest 原有 failures/errors/expectedFailures/unexpectedSuccesses，生成明确的 `test_results.tests_executed` 布尔字段。零方法执行仍不提供评分。
- `fixture_outcome()` 统一解释新字段、旧计数和明细一致性。在线 `_verified()` 与离线 `verified_outcome()` 共用它，删除重复的解析规则；非法类型或矛盾声明通过具体 ValueError 拒绝。合法但未执行的原 verdict 保留，损坏标记仍返回 `(None, None)`。

新字段表示“观察到非跳过的 unittest 结果”，不表示所有断言完成、覆盖率充分或测试可信。显式 false 不会被数量推断覆盖；true 不能使零测试通过。旧记录没有该字段时继续使用旧判据，不猜测过去是否执行过子测试。

生成的评分命令嵌入该结果类，只依赖标准库。已在新临时目录用 Python `-I -S` 成功执行通过/跳过案例，见 [isolated-stdlib.json](../runs/iteration-20261028-01/isolated-stdlib.json)。Local 与 Docker 调用同一个命令生成器；这次验证没有启动 Docker，也不构成容器安全认证。

## 奖励链路与验收

实际 Local grader 配合预先写好的 TokenPolicy，修复前混合正例和负例都不产生 Episode；修复后分别产生 reward=1、reward=0，全部跳过仍不生成 Episode。见 [reward-before.json](../runs/iteration-20261028-01/reward-before.json) / [reward-after.json](../runs/iteration-20261028-01/reward-after.json)。这些仅验证数据准入，不更新参数，不说明 RL 效果。历史无效 GRPO 尝试仍无效。

| 检查 | 结果 |
|---|---|
| 同一新增测试文件，修复前 | 17 failed / 3 passed |
| 同一新增测试文件，修复后 | 20 passed |
| verifier、上一轮一致性、环境与奖励入口定向回归 | 115 passed / 1 skipped |
| 完整 CPU 套件 | **781 passed / 13 skipped** |
| 固定官方 OpenJudge 源码的专用环境 | **142 passed / 0 skipped** |
| 固定 59 份历史结果 | 字节、哈希、质量、整体验收和解释结果均不变 |

13 个跳过记录为可选 Torch 模块 2、未配置的固定 mini-SWE 集成 6、可选 OpenJudge 5；后者已在专用环境运行。CI 的 OpenJudge 任务已纳入新文件。

内置与真实 OpenJudge CLI 都实际处理了 12 个仅有 verifier 证据的结果，质量维度为 4 pass / 4 fail / 4 unknown。它们没有完整 Agent 预算和权限证据，因此整体不能标为全部通过，退出 1 是预期结果。没有为这些 CPU 测试伪造完整 Agent 账本。

JUnit、源码指纹及 38 份证据文件哈希见 [cpu-validation.json](../runs/iteration-20261028-01/cpu-validation.json)。复核命令：

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_verifier_execution.py
.venv/Scripts/python.exe -m budget_agent audit --input runs/iteration-20261028-01/cases --output runs/my-subtest-audit.json
```

使用新输出路径，保留原始证据。以上命令不运行模型或训练。

## 官方需求与实现依据

访问时间：**2026-10-27 19:06:20–19:06:41 UTC / 北京时间 2026-10-28 03:06:20–03:06:41**。没有确认新增或剩余 HC，也未发现实质要求变化。

| 来源 | 日期与时效边界 | 本轮关联 |
|---|---|---|
| [百度代码智能体架构实习 J101404](https://talent.baidu.com/jobs/detail/INTERN/397771ff-709c-42f5-9fd7-2e4112e0768d) | 正文可读，仍标 2026-07-21；页面“2人”不代表剩余 HC | 可复现沙箱、轨迹质量、后训练、rollout 与可验证反馈；每周4天以上、6个月以上 |
| [Seed 人才招聘](https://seed.bytedance.com/zh/seedearlycareer) | 无可见发布日期；2027年9月及以后毕业符合实习范围 | Code Agent RL、长程任务、Memory；用户毕业时间符合范围，不代表具体团队有名额 |
| [百度 J96208](https://talent.baidu.com/jobs/detail/INTERN/a77dfe3a-0f3a-45b5-95dc-c7478f9d43e4) | 直接页仅“登录”，本轮未重新检索全文 | 无法确认内容、关闭状态或 HC |
| [SWE-bench 官方评测指南](https://www.swebench.com/SWE-bench/guides/evaluation/) | 无可见发布日期 | 区分完成报告、测试通过、正常失败和无法评分；不直接规定本项目的实现 |
| [Python 3.11 unittest 接口](https://docs.python.org/3.11/library/unittest.html#unittest.TestResult.addSubTest) | 无页面发布日期，本轮显示3.11.18 | resultclass 与结果回调是公开扩展接口，默认成功子测试不会被保存 |
| [CPython v3.11.13 case.py](https://github.com/python/cpython/blob/v3.11.13/Lib/unittest/case.py) | 固定版本源码，本轮读取，不声称最新 | 核对 skip 与 subtest 回调路径；最终实现以实际 CPU 验证为准 |

由此选择修复的是可信反馈链路。不能把招聘关键词、更多测试数量或本轮小范围修复等价于公司认可、真实代码修复能力或训练收益。

## 仍保留的限制与下一步

如果跳过子测试之后只执行一个普通通过断言，且没有其他成功结果回调，unittest 可能不调用 addSuccess/addSubTest。该情况仍保守未评分；原始复现见 [known-callback-limitation.json](../runs/iteration-20261028-01/known-callback-limitation.json)。本轮不扩展为拦截断言或跟踪所有执行行，也未解决 verifier 与恶意被测代码同进程的信任问题。

expectedFailure 保留 unittest 的成功语义，单方法 setup 异常保留既有负结果语义；更细的奖励归因和测试覆盖质量仍需独立设计。不能把新执行标志当成测试内容足够强的证明。

优先级仍是获得真实有效补丁及独立评分，然后进行固定模型、预算和任务的反馈/上下文对照。原待执行的 ≤10 元参考授权没有扩展；[本轮计划](../runs/iteration-20261028-01/reference-plan.json)只更新源码，保留原模型、题目、次数、token 上限、off/sandbox 和费用时限。执行前仍需核验实时库存、旧账单与报价，本轮没有进行这些检查；不使用具身智能项目的预算。
