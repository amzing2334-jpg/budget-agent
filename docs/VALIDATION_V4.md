# v4 验证记录：协议对照完成，四次尝试均未交付修复

> 历史记录（2026 年 10 月）：文中链接的 `runs/` 产物没有收进仓库，保存在作者的本地归档中。当前的结果和方法见 [技术复盘](TECH_REPORT.md)。

日期：2026-10-22。本轮完成招聘/行业系统调研、历史协议与评分状态修订、CPU 软件验证，以及真实云端两模型协议对照。**云端流程运行完成，但四次尝试均为空补丁，没有真实任务修复成功的证据。** 没有进行训练或本地模型推理。Featurize 实例已经归还并通过控制台核验。

## 代码与 CPU 验证

在用户电脑的轻量 `.venv` 中执行 `python -m pytest -q`，结果为 **262 passed, 2 skipped in 20.92s**，日志在 [pytest.txt](../runs/v4-cpu-checks/pytest.txt)。两个训练测试模块因为未安装可选训练依赖而跳过；本轮不在本机加载真实模型。

新增验证覆盖原始非法回复、assistant/观察角色、完整事件组与观察裁剪、chat-template 计数、摘要增量、transcript 兼容、首轮输入一致；协议对照条件与预算核验；官方基础设施异常、空补丁、manifest/hash 对齐，以及独立评分导出、回填和报告链路。软件测试中的评分进程是模拟对象，下面的云端对照才是实际官方评分。

已实施范围：

- 默认 `history_format=chat`，保留显式 `transcript` 对照。原始 assistant 回复与 user 工具观察成对保存；仍是文本动作接口，未声称原生 tool_calls。
- 本次比较保持 SYSTEM、工具行为、采样参数和预算不变，未增加自动重试、强制修改或多次取最好结果。
- `scripts/probe_history_format.py` 保存预先确定的计划，执行两种格式各一次，审计首轮输入和实验条件。
- `scripts/grade_protocol_probe.py` 在 Agent 结束后独立导出、调用官方 grader、回填并生成报告。空补丁记为 `resolved=null / official_empty_patch / tests_executed=false`；基础设施异常单列，不能充当模型测试失败。

## 云端环境和完整性

实际使用单张平台标注 RTX 4090 48G，驱动报告 49,140 MiB；按量单价 3 元/小时。宿主为 Ubuntu 22.04.4、Python 3.11.8、Docker 26.1.0、驱动 610.57.04。独立推理环境使用 PyTorch 2.8.0+cu128、Transformers 5.4.0；独立评分环境使用 swebench 5.0.2。实际依赖冻结文件随回执保存。

- Agent 源码指纹：`83cc596ad244cf761e532a560ef7f05cb6b9d911460b04c1779ab08de9a2976c`。
- 原上传包 SHA256：`13bc946d1f31909331c06bb3f60a66f83ed737bebbd93deaf139701299c3b268`。包不包含密钥、旧运行、权重或隐藏评分材料。
- A 模型固定 `Qwen/Qwen2.5-Coder-7B-Instruct@c03e6d358207e414f1eca0bb1891e29f1db0e242`，11 个所需文件核验通过。
- B 模型固定 `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`，10 个所需文件核验通过。
- 评分数据集使用已固定缓存 revision `b0dde1093fe417d83b7184254edf8199c1f0dff5`。Agent 与官方评分材料分别位于不同目录；gold patch 与隐藏测试不进入 Agent 输入。
- 原 Docker 镜像归档的 SHA256 未变。Docker 存储后端改变导致 `.Id` 表示从 OCI index 变为 config digest；核验 index→manifest→config 三段摘要和 11 个层 DiffID 后，恢复严格身份检查，详情见 [镜像身份回执](../runs/cloud-v4-20261022-01-receipts/image-identity-mapping.json)。这不代表 v3 与 v4 宿主运行时完全相同。

基础设施的失败尝试完整保留：镜像身份预检初次停止；B 下载初次缺 socksio，安装 socksio 1.0.0 后继续；向持久目录下载遇到 `OSError: [Errno 5] Input/output error`，随后改用实例本地 SSD 成功。未确认该 I/O 错误的根因，不能断言为配额不足。这些失败都发生在对应模型尝试开始之前，不是丢弃的 Agent 失败样本。

B 的完整权重只保存在已归还实例的临时 SSD，**本轮没有完成 B 权重的持久化缓存**；以后需按固定 revision 重新下载并核验。持久目录仍有约 101 MB 的不完整 B 缓存，不能作为完整模型使用。代码、配置、模型 metadata、完整性回执、轨迹和评分产物均已保存。没有购买磁盘扩容。

## 先验证评分环境

在独立评分目录执行官方对照：

| 补丁 | 官方 resolved | FAIL_TO_PASS | PASS_TO_PASS |
|---|---|---:|---:|
| 官方正确补丁 | true | 2/2 | 13/13 |
| 无关文本文件的错误补丁 | false | 0/2 | 13/13 |

没有基础设施失败，`controls-verified.json` 已生成。该对照说明当前环境能区分这两个已知补丁，不能据此推导 Agent 会修复问题，也不能证明所有任务的评分环境都可靠。

## 两模型、两历史格式的实际结果

任务只有 `astropy__astropy-12907`，属于已反复调试的开发任务。每模型的 transcript/chat 各一次；recent 上下文策略、8,192 上下文上限、32,000 累计输入加输出上限、768 单次最大输出、20 步、greedy、seed 42、BF16。两种格式各自使用独立容器。模型间相同 token 上限不等于相同 FLOPs 或费用。

| 指标 | A：7B transcript | A：7B chat | B：4B transcript | B：4B chat |
|---|---:|---:|---:|---:|
| 模型调用 / 步数 | 10 | 12 | 10 | 11 |
| 输入 token | 24,204 | 31,286 | 30,360 | 31,869 |
| 输出 token | 609 | 714 | 1,640 | 131 |
| 总 token | 24,813 | 32,000 | 32,000 | 32,000 |
| invalid_action | 2 | 1 | 4 | 2 |
| 同一失败动作在首次之后的重复次数 | 3 | 7 | 0 | 0 |
| 成功编辑 / 写文件 | 0 | 0 | 0 | 0 |
| 执行 run / diff | 0 | 0 | 0 | 0 |
| 补丁字节数 | 0 | 0 | 0 | 0 |
| 结束状态 | finished | budget_exhausted | budget_exhausted | budget_exhausted |

全部预算审计通过；没有超额、未知用量或未结算请求。**四条结果均为 `official_empty_patch`，`tests_executed=false`、`resolved=null`。** 官方没有对这些空补丁执行修复测试，不把它们写成测试通过或有已执行测试依据的 false，也不从这四条记录生成官方已评分成功率。交付层面则可以明确说四次均未产生修改。

两模型内部各自的首轮输入和首轮原始回复完全相同，支持历史格式对照的起点一致；不能将不同模型的两组当作同一实验的统计重复。A、B 的 pipeline 都以 0 退出，只表示驱动、审计与评分导入顺利完成。

## 失败定位与下一步

完整证据和逐步轨迹见 [V4_FAILURE_ANALYSIS.md](V4_FAILURE_ANALYSIS.md)。本轮可支持的判断：

1. **首轮协议和路径遵循还不足。** A 在没有历史时就复制 `relative/file.py` 占位路径；B 在没有历史时一次输出三个 JSON 动作。不能把这些错误归因于历史格式。
2. **A 的主要循环是歧义编辑未恢复。** old 文本命中两处，模型在已看到错误的情况下重复同一编辑。主要循环中代码和错误反馈仍可见，不能归因于它们已被裁掉。
3. **B 的主要循环是重复浏览而没有读到源码。** search 已给出正确路径和行号，但仍重复 list/search 或读取带有 `relative/` 的错误路径；没有成功读取源码正文。工具成功的重复浏览不计入“重复失败动作”，因此该指标为 0 不能表示没有循环。
4. **目录观察带来明确的预算浪费。** list_files 一次输出约 7.1 万字符；B 每条轨迹有三次后续请求各输入 7,424 token，仅这些输入占总预算 69.6%。B chat 的输入占总量 99.59%。这是应优先处理的成本问题，尚无修改后的修复收益对照。
5. **尾部预算耗尽产生了不同性质的格式错误。** A chat 最后只有 43 个输出 token，B 两格式最后只有 1 个；这些截断应与模型主动续写日志或多动作回复区分。

下一轮优先建立真正可修复的基线：单动作及仓库相对路径契约、可控范围的文件查询、可操作的编辑错误反馈，以及完整的定位—修改—测试轨迹。每次只改变一个主要因素，并保留简单对照。尽早在固定的多个仓库开发任务上与固定版本的参考 harness 比较，再用能力更强的模型排查瓶颈；本轮没有运行 mini-swe-agent，因此不能声称已优于或复现它。

待基本能力有效之后，比较 recent、observation masking 与保留定位/编辑/测试证据的上下文策略。只有在有可靠奖励、有效 rollout 和隔离的确认任务时，再进入 SFT 或 Agentic RL；旧 v2 无效 GRPO 更新继续保持无效。本轮没有新训练，也没有证明 chat、结构化上下文或某个模型更好。

## 产物、费用与实例归还

- [A 原始与评分后报告](../runs/cloud-v4-20261022-01-protocol/report/report.html)、[A 本地审计](../runs/cloud-v4-20261022-01-protocol/protocol-local-audit.json)。
- [B 原始与评分后报告](../runs/cloud-v4-qwen3-20261022-01-protocol/report/report.html)、[B 本地审计](../runs/cloud-v4-qwen3-20261022-01-protocol/protocol-local-audit.json)。
- A/B 项目侧归档保存在 `train_outputs/cloud-v4/`；独立评分原始材料保存在项目外的 `C:/找工作/budget-agent-evaluation/cloud-v4-20261022-01` 与 `cloud-v4-qwen3-20261022-01`，分别归档到 `cloud-v4-downloads/`。四个回传包均比对云端 SHA256 后解包，未把评分答案放入 Agent 项目。
- [租用与归还回执](../runs/cloud-v4-20261022-01-receipts/reservation.json)；[归还截图](../runs/cloud-v4-20261022-01-receipts/instance-released.jpg)。

用户授权每小时不超过 5 元、最多 2 小时、总额不超过 10 元。控制台开始时间为北京时间 09:53（精度为分钟）；已预设 11:43 自动归还，在完成回传后于 **10:46:25** 提前调用官方释放命令。随后刷新 Chrome 控制台，显示“未找到实例”；本地回执记录核验时间。约 54 分钟、3 元/小时，估算约 **2.7 元**；实际结算以平台账单为准，本轮没有读取最终账单。定时器成功、SSH 断开与实际释放是不同证据，这里采用了刷新后的控制台状态。
