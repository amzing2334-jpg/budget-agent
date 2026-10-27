# API 端点重定向：凭证边界与用量可信性

检查时间：2026-10-27 10:55–11:03（Asia/Shanghai）。本轮仅运行本地 CPU 假 HTTP 服务与软件测试，没有调用真实模型、读取真实密钥或新租用主机。

## 实际复现的问题

`OpenAICompatibleModel.complete()` 原先使用 `urllib.request.urlopen()`。它的默认重定向处理会在收到 301/302/303 后，把 POST 改成 GET 继续访问 Location，并保留 Authorization。

用两个 loopback 假服务和虚构 Key 独立复现：初始服务收到 1 次 POST；另一个 hostname/端口的目标服务收到 1 次 GET 以及虚构 Authorization。目标服务返回自造的 `model=redirected-fixture`、输入9/输出3 token，适配器将其接受为 `usage_source=provider`。因此问题同时影响凭证发送边界和“用量来自配置端点”的假设。307/308 在本机原实现下没有转发，但也需明确固定其拒绝合同。

这是可复现的软件漏洞，**不表示用户真实密钥已经泄漏**。本轮实际 API 请求为0，所有用于复现的 Key、提示和回复均为测试数据。Python 官方文档说明默认 opener 包含重定向处理器，并区分全局 `install_opener()` 与直接调用实例 `OpenerDirector.open()`；公开文档没有提供项目的安全保证，本项目以实际请求观测验证行为。[Python urllib.request](https://docs.python.org/3/library/urllib.request.html#urllib.request.HTTPRedirectHandler)

另一个边界问题是非法 Location：原处理器先解析跳转地址，`https://[...` 会抛出 `ValueError`，绕过既有 ModelError/未知用量保存路径。新增回归在修复前也复现了这一点。

## 最小修改与行为

- 每个模型客户端创建独立 opener，不修改进程的全局 opener。
- 同源和跨源的301/302/303/307/308均在解析或跟随 Location 前抛出 HTTPError，不发第二个请求，不重试，不新增白名单和兼容开关。
- 关闭 HTTP 错误响应，沿用已有脱敏 ModelError、usage_unknown、未对账预算及批量停止逻辑。不保存错误正文或 Location。
- 正常200响应的 Authorization、消息、模型名称和用量行为保持原合同。

需要迁移服务地址时，应由执行者核对供应商地址并显式修改 `base_url`。本修复没有判断服务是否计费，收到3xx后仍保守记用量未知，不能将其当作免费调用。

## 验收与失败证据

新增13项回归在修复前 **8 failed / 5 passed**，原始输出保存在 [redirect-before.txt](../runs/iteration-20261027-03/redirect-before.txt)。修复后这些回归与既有模型边界测试合计 **47 passed**。完整CPU套件 **529 passed / 8 skipped**，结果见 [cpu-validation.json](../runs/iteration-20261027-03/cpu-validation.json)。

| 检查 | 修复前 | 修复后 |
|---|---|---|
| 同源/跨源301、302、303 | 跟随跳转，接受目标结果 | 仅初始POST，抛受控ModelError |
| 同源/跨源307、308 | 拒绝 | 继续明确拒绝，目标0请求 |
| 非法Location | URL解析ValueError逃逸 | 不解析目标，受控HTTP302错误 |
| 正常200响应 | 正常 | 正常，授权头与9/3假用量完整 |
| CLI两策略×两重复 | 四次计划均继续，错误接受目标用量 | 仅第一条请求；用量未知、费用unknown；保存部分报告并退出1 |
| 结果/轨迹/报告 | 原实现没有该边界回归 | 不包含虚构Key或Location标记 |

8个跳过项：2个可选Torch测试模块未安装依赖；6个固定mini-SWE上游集成未设置上游路径。没有启动训练栈。完整测试已经通过，未因文档修改反复运行套件。

本轮源码指纹为 `d8baa90ea60c07af538a0ac41c5e2f80f2abba9a8c7073d921bf8777c4c52266`。原模型、任务和token预算不变；部署前应采用本轮新源码与 [更新的待执行参考记录](../runs/iteration-20261027-03/reference-plan.json)，不能继续使用修复前的归档。原提案和旧运行证据不覆盖。

## 招聘复核与项目取舍

2026-10-27 **10:55 北京时间**重新核验以下来源，未发现足以改变项目主线的实质变化，也没有确认新增或剩余HC：

| 官方来源 | 发布日期/访问边界 | 相关要求 |
|---|---|---|
| [百度策略实习J96208](https://talent.baidu.com/jobs/detail/INTERN/a77dfe3a-0f3a-45b5-95dc-c7478f9d43e4) | 2026-07-21；直达页仅导航/登录，通过官方域名搜索索引读取，索引约两周前抓取 | 真实行为分析、工具/上下文策略、成本、指标、消融 |
| [百度架构实习J101404](https://talent.baidu.com/jobs/detail/INTERN/397771ff-709c-42f5-9fd7-2e4112e0768d) | 2026-07-21；官方正文可读，页面“2人”不等于剩余2个HC | 可复现沙箱、轨迹质量、SFT/OPD/RL基础设施、推理性能 |
| [Seed校招](https://seed.bytedance.com/zh/seedearlycareer) | 无可见发布日期；官方正文可读，实习范围含2027年9月及以后毕业 | Code Agent RL、长程任务、Memory、机器学习系统；没有团队实时HC |
| [Anthropic评测说明](https://www.anthropic.com/research/demystifying-evals-for-ai-agents) | 2026-01-09；全文可读，重读不等于新发布 | 区分task、trial、轨迹、grader和最终环境结果 |

结合项目目前仍缺乏有效真实补丁，下一步仍是既定参考试跑，再开展固定任务与单因素对照。本轮修复使端点和账本合同更可靠，但不证明修复率提升、上下文策略收益或Agentic RL训练效果。新百炼Key前缀本身没有发现适配缺陷，不增加按Key前缀改写地址的逻辑。
