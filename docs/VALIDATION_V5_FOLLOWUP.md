# v5 后续验收：七次云端尝试完成，尚无有效补丁

日期：2026-10-22。本记录已核对预定的三组真实模型尝试、随后设计的 recent/mask 与 direct/budget-aware 两对适应性开发尝试，以及独立官方评分产物。**七次尝试均导出空补丁，官方记录为 `official_empty_patch`、`tests_executed=false`、`resolved=null`。** 这表示没有可供执行测试的模型补丁，不能表述为“官方测试运行后七次失败”，也没有获得成功修复证据。

mask 在同一 32,000-token 上限下把调用次数从 8 增加到 13，但没有带来有效编辑、测试或修复收益。轻量 CPU 完整回归最新为 **376 passed、8 skipped，25.78 s**；固定 mini-SWE-agent 的六项真实上游代码 CPU 集成检查也已通过。本机没有执行模型推理或训练。

七次云端尝试及官方评分截至 **2026-10-22 07:49:20 UTC（北京时间 15:49:20）** 已完成并回传。最后一次 CPU 回归于 07:50:44 UTC 完成。07:53:39 UTC 提前执行释放，随后刷新控制台核验“未找到实例”，截图已保存；最终结算账单未读取。后续结果应保留各自计划、源码和对照，不能回填到先前实验中。这份记录承接 [v5 验证记录](VALIDATION_V5.md)，保留所有先前原始产物。

## 任务、模型与实际云环境

七次尝试均使用 `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`，任务均为已反复用于开发的 `astropy__astropy-12907`。模型的 10 个必要文件重新通过固定官方 Hub metadata 的哈希校验，见 [model-integrity.json](../runs/cloud-v5-a4000-20261022-01-receipts/model-integrity.json)。以前的 7B 结果保留在各自记录中，不并入本次同模型比较。

| 项目 | 已核验信息 |
|---|---|
| GPU | 单张 NVIDIA RTX A4000，16,376 MiB；驱动 610.57.04；BF16 可用 |
| 推理 | PyTorch 2.8.0+cu128、Transformers 5.4.0、huggingface_hub 1.33.0 |
| 独立评分 | 独立 Python 环境中的 SWE-bench 5.0.2 |
| Docker | Server 26.1.0；实际任务 image ID 为 `sha256:a7cc2f990459911b958bd75b84fb89205fa6a6b33e0ee1cf9b33b175b8fd0df0` |
| 模型与 tokenizer | 同一固定 revision；实际 tokenizer/chat template 精确计量输入输出，无 API 价格估算 |
| 生成 | BF16、greedy（配置 temperature=0）、seed=42，每次最多输出 768 token |
| 预算 | 每次请求上下文上限 8,192，累计输入加输出上限 32,000，最大步数 20 |
| 任务基础提交 | `d16bfe05a744909de4b27f5875fe0d4ed41ce607` |
| 云端项目 | `/home/featurize/work/budget-agent-v5-followup`，与旧项目目录分开 |
| 独立评分目录 | `/home/featurize/work/budget-agent-evaluation/cloud-v5-a4000-20261022-01` |

环境来源：[GPU 回执](../runs/cloud-v5-a4000-20261022-01-receipts/gpu-before.txt)、[实际依赖及 CUDA 预检查](../runs/cloud-v5-a4000-20261022-01-receipts/preflight.txt)、[最终推理依赖冻结](../runs/cloud-v5-a4000-20261022-01-receipts/inference.final.freeze.txt)、[评分环境冻结](../runs/cloud-v5-a4000-20261022-01-receipts/grader.freeze.txt)、[Docker 状态](../runs/cloud-v5-a4000-20261022-01-receipts/docker-info.txt)。

## 预定三组：真实模型尝试与独立评分已完成

| 条件 | 机制 | 解释边界 |
|---|---|---|
| 内部 reserve-1 | recent、chat、内部 JSON 工具；最小动作输出预留 1 | 与 reserve-128 的单参数对照 |
| 内部 reserve-128 | 同上；最小动作输出预留 128 | 两组共用一次模型加载，分别重置种子和任务环境 |
| mini-SWE 外部参考 | 固定 mini-SWE-agent 2.4.6 的真实上游循环，文本 Bash 动作、完整历史与显式提交，经预算/环境适配器连接 | 整系统外部参考；工具、提示、上下文和提交机制均有差异，不是内部上下文消融 |

mini-SWE 固定上游提交为 `a83fcae82d2a08f0ee0c688f9d137b3566c097f8`。本轮已在云端运行实际模型和 Docker 环境中的真实上游循环，不能再写为“仅实现适配器、尚未运行真实上游”。同时，预算和隔离环境使用本项目适配器，结果也不能冒充官方排行榜配置。

`min_action_output_tokens` 是准入和上下文分配的最低输出额度，不强制模型生成相应长度，不保证动作完整。所有尝试在运行前写入计划；保留原始调用、轨迹、补丁、配置和官方导出 manifest。预定三组于 07:16:29 UTC 完成报告。

| 条件 | 后端调用 / 环境动作或步骤 | 输入 token | 输出 token | 合计 token | 运行状态 | 补丁字节 | 官方状态 |
|---|---:|---:|---:|---:|---|---:|---|
| reserve-1 | 8 / 8 | 31,175 | 252 | 31,427 | budget_exhausted | 0 | official_empty_patch |
| reserve-128 | 8 / 8 | 31,175 | 252 | 31,427 | budget_exhausted | 0 | official_empty_patch |
| mini-SWE | 6 / 6 | 25,570 | 1,230 | 26,800 | budget_exhausted | 0 | official_empty_patch |

三组均无预算超限、未知使用量或未结算请求。官方空补丁状态均由明确指定的 aggregate summary、导出 predictions 和 manifest 回填，保留 `tests_executed=false` 与 `resolved=null`。相同 token 上限不等于相同延迟；本轮没有把顺序运行的墙钟差异当作速度提升。

内部两组的 8 次 `messages`、`response`（含用量）及 `requested_max_output_tokens` **逐项完全一致**。最后一次调用的输出额度都是 604，模型实际仅生成 31；剩余总预算 573，已经不能容纳固定提示和完整 issue 的 756-token 输入。因此，本轨迹没有触及“1 与 128 预留导致不同动作”的情形，不能宣称 128 改善了结果，也不能由单轨迹推断它在其他场景始终无效。

内部两组均成功读取源码 4 次、执行命令 1 次；另一次命令因缺少 import 出现 `NameError`。成功命令只验证了正常的非嵌套示例，没有复现 issue 的嵌套失败。没有 `edit_file`、`write_file` 或 `diff` 成功记录。输入占总 token 的 99.20%，但高输入占比本身不是模型失败的充分根因。

mini 的 6 条 Bash 动作全部解析合法、返回码为 0；前两条搜索和读取源码，后四条 `sed` 将匹配字符串替换为自身，没有实际改变。没有测试、diff 或显式提交。官方上游 `api_calls=7` 包含进入下一次 `query()` 后被预算拒绝的一次；实际后端调用与账本均为 6。保存文本上的 tokenizer-only 重建测得下一次输入为 5,982 token，大于剩余的 5,200，说明预算准入阻止了第 7 次生成；它不是一次额外模型推理。证据见 [重建计数](../runs/cloud-v5-a4000-20261022-01-receipts/mini-stop-token-count.json)。

预定结果入口：[reserve 配对计划](../runs/cloud-v5-a4000-20261022-01/reserve-probe/plan.json)、[逐次结果路径](../runs/cloud-v5-a4000-20261022-01/reserve-probe/attempts.json)、[mini 计划](../runs/cloud-v5-a4000-20261022-01/mini-baseline/plan.json)、[预定三组报告](../runs/cloud-v5-a4000-20261022-01/report/report.html)。逐调用预算与行为分析见 [CLOUD_FAILURE_ANALYSIS.md](../runs/v5-followup-20261022-01/CLOUD_FAILURE_ANALYSIS.md)。

## 适应性 recent/mask：增加了可执行轮次，没有修复收益

该对照是在看到预定三组无补丁及历史输入开销后设计的，属于**适应性开发实验**。07:24:43 UTC 写入新计划后才开始执行；不是预定三组的追加重试，也不是独立确认集验证。新源码下重新执行 recent 与 mask 各一次，共用一次模型加载，模型、任务、预算及生成参数一致；两者最小动作输出预留均为 128。

mask 保留完整动作历史及最近两条 observation，较旧 observation 的正文用引用标记替代，仍可经 recall 恢复。本次固定 `mask_keep_recent=2`，不声称这是最优超参数。计划见 [adaptive plan](../runs/cloud-v5-a4000-20261022-01-receipts/adaptive-mask/plan.json)，配置见 [cloud-v5-qwen3-mask.toml](../configs/cloud-v5-qwen3-mask.toml)。

| 新源码下的条件 | 调用 | 输入 / 输出 / 合计 token | 成功 read_file | 成功 edit / write / diff | 成功 run | 官方状态 |
|---|---:|---|---:|---|---:|---|
| recent | 8 | 31,175 / 252 / 31,427 | 4 | 0 / 0 / 0 | 1（正常示例） | official_empty_patch |
| mask | 13 | 31,339 / 347 / 31,686 | 9 | 0 / 0 / 0 | 0 | official_empty_patch |

mask 的调用从 8 增加到 13（+62.5%），平均每次输入从 3,896.9 降至 2,410.7 token；但**累计输入并未下降**，从 31,175 增至 31,339，累计总 token 也从 31,427 增至 31,686。它把预算用于更多搜索和分段阅读，仍没有产生有效修改、验证命令或补丁。两组都在累计预算准入处停止，未发生未知用量或预算超支。

该结果支持的窄结论是：在这条开发轨迹上，屏蔽旧观察降低了每次调用的平均输入占用，并允许更多动作。它不支持“提高修复率”“降低完成任务成本”“提升测试覆盖”或“Agent 已会自我改进”。mask 改变可见证据后，模型选择的动作也改变了；不能把更多阅读直接视为更有效的推理。需要先获得可复现的有效修改与独立通过，再在新的任务集检验泛化。

两组报告与原始产物见 [adaptive-mask/report](../runs/cloud-v5-a4000-20261022-01/adaptive-mask/report/report.html)。附加结果各为 attached=1、unmatched=0、infrastructure_errors=0、official_empty_patches=1，见 [recent 回执](../runs/cloud-v5-a4000-20261022-01-receipts/adaptive-mask/recent-attach.log) 和 [mask 回执](../runs/cloud-v5-a4000-20261022-01-receipts/adaptive-mask/mask-attach.log)。07:28:19 UTC 的 [完成回执](../runs/cloud-v5-a4000-20261022-01-receipts/adaptive-mask/finished) 只证明这两组流程结束，不代表租用结束。

## 适应性 direct/budget-aware：预算提示尚未改变无补丁结果

看到 mask 仍停留在阅读后，又设计了一对最小决策提示干预。07:45:00 UTC 先保存[配对计划](../runs/cloud-v5-a4000-20261022-01/adaptive-guidance/plan.json)，再在同一新源码、同一加载模型和相同任务环境上分别运行 `decision_policy=direct` 与 `budget-aware` 各一次。两者均使用 mask 上下文策略，其余固定参数相同。这是同一道已知开发题上的第二次适应性实验，不是保留集或预注册的确认结果。

budget-aware 在系统提示中增加真实的剩余累计 token、步数、工作区是否变化及连续阅读动作数，并给出通用工作要求：复现报告中的失败、理解相关代码后作最小修改、检查 diff、执行复现与相关测试。它没有指定该题的答案、修改位置或隐藏测试；所有工具仍可选择。它是提示干预，没有参数训练，也不是 Agentic RL。

| 条件 | 调用 | 输入 / 输出 / 合计 token | 成功 read_file | 成功 edit / write / diff / run | 官方状态 |
|---|---:|---|---:|---|---|
| mask + direct | 13 | 31,339 / 347 / 31,686 | 9 | 0 / 0 / 0 / 0 | official_empty_patch |
| mask + budget-aware | 13 | 31,440 / 361 / 31,801 | 10 | 0 / 0 / 0 / 0 | official_empty_patch |

两组均以 `budget_exhausted` 结束，补丁都是 0 字节，没有无效 JSON 动作、预算超支、未知用量或未结算请求。budget-aware 改变了部分阅读选择，但没有让这条轨迹进入编辑或验证阶段；不能把提示更明确写成模型能力提升。两组调用数相同，budget-aware 的合计用量多 115 token，也没有观察到成本或修复收益。

两组独立官方评分回填各为 attached=1、unmatched=0、infrastructure_errors=0、official_empty_patches=1，仍为 `tests_executed=false`、`resolved=null`。证据：[报告](../runs/cloud-v5-a4000-20261022-01/adaptive-guidance/report/report.html)、[direct 附加回执](../runs/cloud-v5-a4000-20261022-01-receipts/adaptive-guidance/direct-attach.log)、[budget-aware 附加回执](../runs/cloud-v5-a4000-20261022-01-receipts/adaptive-guidance/budget-aware-attach.log)、[07:49:20 UTC 完成回执](../runs/cloud-v5-a4000-20261022-01-receipts/adaptive-guidance/finished)。

这对云端尝试的源码为 `d3ca73f7db6c7ecd49832e1c04d9d87fa08b73092b7dd1189a19d83fe98a4528`。之后审查发现：如果使用付费摘要策略，摘要调用消耗预算后，下一次动作请求中的预算提示应在准入前刷新。该修复已进入最终本地源码 `1faa1c77c69d824ec94d072bc3a3e2f0959dc18879ea544ca13b3d68bfa70af1` 并通过 CPU 回归。**本轮两组使用 mask，没有摘要调用；没有用修复后的源码重跑云端模型。** 因此该修复只有软件检查证据，不能把云端结果标记为最终源码的模型验证。原始配对及源码归档保持不变。

## 独立评分控制与证据隔离

在 Agent 运行前，本轮实际重新运行 gold/negative controls，评分发生在项目外的独立目录。gold patch、隐藏测试和完整官方评分日志不进入 Agent 输入。两组控制的实际结果为：

| 控制 | resolved | FAIL_TO_PASS 成功 / 失败 | PASS_TO_PASS 成功 / 失败 |
|---|---|---|---|
| 官方 gold patch | true | 2 / 0 | 13 / 0 |
| 不修改源码的 negative patch | false | 0 / 2 | 13 / 0 |

两者补丁均成功应用，无基础设施失败。控制回执的本地副本位于 [controls-verified.json](C:/找工作/budget-agent-evaluation/cloud-v5-a4000-20261022-01/controls/controls-verified.json)。这为当前任务的评分链路提供正负控制证据，不证明所有任务的环境都可靠，也不意味着模型产出了 gold patch。

模型七次导出的补丁都是 0 字节；官方不会对这些空补丁执行任务测试。保留 `resolved=null` 是为了区分“没有提交可评分补丁”和“执行测试后修复失败”。不能把控制组的 15 项测试算到模型的测试数或成功率中。

## CPU 回归与真实上游集成

| 软件检查 | 实际结果 | 证据与范围 |
|---|---|---|
| 预定三组源码的轻量完整回归 | 342 passed、8 skipped，25.97 s | [回执](../runs/v5-followup-20261022-01/cpu-tests-receipt.json)、[日志](../runs/v5-followup-20261022-01/cpu-tests.log) |
| 增加 observation masking 后的轻量完整回归 | **360 passed、8 skipped，26.22 s** | [回执](../runs/v5-followup-20261022-01/cpu-mask-tests-receipt.json)、[日志](../runs/v5-followup-20261022-01/cpu-mask-tests.log) |
| 增加决策提示并修复摘要后预算提示刷新后的最终轻量完整回归 | **376 passed、8 skipped，25.78 s** | [回执](../runs/v5-followup-20261022-01/cpu-final-tests-receipt.json)、[日志](../runs/v5-followup-20261022-01/cpu-final-tests.log) |
| 固定 mini-SWE-agent 真实上游代码的独立 CPU 集成 | **6 passed、13 deselected，5.08 s** | [回执](../runs/v5-followup-20261022-01/upstream-integration/receipt.json)、[日志](../runs/v5-followup-20261022-01/upstream-integration/integration-validated.log) |

轻量 `.venv` 未安装 torch、transformers、peft、accelerate 或 datasets。八项跳过包括两个需要可选训练栈的测试模块，以及六个需要独立上游依赖环境的集成用例；后六个已在独立 CPU 环境执行，不能把它们写为轻量环境内通过。三次完整回归各自的前后源码指纹一致。最终回归的包装进程耗时为 26.23 s，pytest 报告 25.78 s；两者计时边界不同。最终 CPU 修复与实际云端源码的差异在上一节明确列出。

六项上游 CPU 集成使用真实 `DefaultAgent`、模板、解析、异常流程、提交识别及轨迹保存；模型文本和环境返回是明确替身。场景包括格式错误恢复后提交、连续格式错误、步数上限、调用前预算不足、调用异常和调用中断。原始合成结果用 `cpu-result.json` 保存，避免真实模型报告递归误收。细节见 [集成重放说明](../runs/v5-followup-20261022-01/upstream-integration/REPLAY.md)。软件契约测试与本轮云端模型运行是不同证据，两者不能互相替代。

## 源码、配置与回传追踪

| 范围 | 指纹 / SHA256 |
|---|---|
| 预定内部配对的 `src/budget_agent` | `e9642259b3f1e5d18db9cd75a3780b1dfdaa8012eb3baa6e1c4e8f4ee597f0a5` |
| 适应性 recent/mask 的 `src/budget_agent` | `209a8cff1285870eb9456fdb1db5f38c1e02ef73a987b82ab53ffe1961a4ca4f` |
| 适应性 direct/budget-aware 的云端 `src/budget_agent` | `d3ca73f7db6c7ecd49832e1c04d9d87fa08b73092b7dd1189a19d83fe98a4528` |
| 摘要后预算提示刷新修复后的最终本地 `src/budget_agent`，仅 CPU 检查 | `1faa1c77c69d824ec94d072bc3a3e2f0959dc18879ea544ca13b3d68bfa70af1` |
| 预定 4B 配置 | `a9ccc1828f54d584e0a400ff276b5d02e75765ee8e7bf03feafd20f064f6c883` |
| mask 配置 | `a7e158201600bd04fbbc7cdcc4b4992fd47062011b087bf8dc799799fa210e6e` |
| 任务清单 | `0e7104adbdf85bfe4f3c81c5e56a7ff8ff2884480804e5e73245b70573dc0e0c` |
| mini 适配器文件 | `60e779948e8e57f5bd7d947ccdd40fcf5a73ba4fdcbe47d35ca21451ece2ac92` |

原始 [deployment-files.json](../runs/v5-followup-20261022-01/cloud-receipts/deployment-files.json)、[mask-deployment-files.json](../runs/v5-followup-20261022-01/cloud-receipts/mask-deployment-files.json) 和 [guidance-deployment-files.json](../runs/v5-followup-20261022-01/cloud-receipts/guidance-deployment-files.json) 保留逐文件哈希；`source.tar.gz`、`mask-source.tar.gz`、`guidance-source.tar.gz` 保存对应云端代码。mini 的 result source fingerprint 另包含适配器及上游 SHA，不能直接与内部 `src` 指纹作相等比较。

预定三组与两对适应性实验的 result、calls、trajectory、patch、报告和独立评分产物均已有本地副本；本记录核对了七个 result、七个空补丁、两组 controls、附加回执以及预定两组请求逐项一致性。回传归档见 [预定结果](../runs/v5-followup-20261022-01/cloud-receipts/v5-predefined-results.tar.gz)、[预定评分](../runs/v5-followup-20261022-01/cloud-receipts/v5-predefined-grading.tar.gz)、[适应性结果](../runs/v5-followup-20261022-01/cloud-receipts/v5-adaptive-results.tar.gz)、[适应性评分](../runs/v5-followup-20261022-01/cloud-receipts/v5-adaptive-grading.tar.gz)。决策提示对照另有 [结果归档](../runs/v5-followup-20261022-01/cloud-receipts/v5-guidance-results.tar.gz) 与 [评分归档](../runs/v5-followup-20261022-01/cloud-receipts/v5-guidance-grading.tar.gz)。这里不声称对所有实验归档做了逐字节远端比对。

模型缓存已有独立的持久化往返核验：[model-cache-roundtrip.json](../runs/v5-followup-20261022-01/cloud-receipts/model-cache-roundtrip.json) 记录从持久目录 `/home/featurize/work/budget-agent-assets/qwen3-4b-chunks-v5` 的 **16 个分片**重建到新的 SSD 目录 `/home/featurize/model-roundtrip-v5`；07:48:15 UTC 完成后，固定 revision 的 **10 个必要模型文件全部匹配原始官方 metadata 的哈希**。这证明本轮保存的分片可完整重建模型文件，区别于上轮仅保存在临时 SSD 的状态。下一台实例仍需按同一过程重建核验，不把此次往返核验写成下次实例已经恢复成功。

## 实例释放和费用：已核验无运行实例

本轮按量 A4000 单价为 0.94 元/小时，控制台开始时间为北京时间 14:20（06:20 UTC）。已设置 **16:10（08:10 UTC）官方 CLI 自动释放**，回执为 `ActiveState=active`，见 [release-timer.txt](../runs/cloud-v5-a4000-20261022-01-receipts/release-timer.txt)。实验工作负载截止为 16:00（08:00 UTC）。

**07:53:39 UTC（北京时间 15:53:39）提前启动已配置的官方释放服务；随后刷新 Chrome 的“我租用的实例”，页面明确显示“未找到实例”。** 释放截图在 07:55:46 UTC 前已保存，见 [实例已归还截图](../runs/v5-followup-20261022-01/cloud-receipts/instance-released.png) 和[租用回执](../runs/v5-followup-20261022-01/cloud-receipts/reservation.json)。这次核验没有搜索筛选条件；不是仅根据 SSH 断开或 timer 推断释放。远程 SSH/SFTP 会话随后关闭。

按控制台 14:20 的启动时间和约 95 分钟使用时间、0.94 元/小时估算约 **1.49 元**；最终结算账单未读取，因此不称为实际扣款。实例已在 2 小时授权范围内归还，没有追加租卡、套餐或训练。原始产物、七次附加评分摘要及模型持久化回执均在释放前回传。回传归档已逐一核对本地 SHA256 与云端打包后的 SHA256，记录见 [归档校验清单](../runs/v5-followup-20261022-01/cloud-receipts/archives-verified.json)。

## 目前可以据此做出的判断

可靠的工程进展是：云端真实小模型与固定上游循环能够受预算约束地运行，账目可核查，评分有正负控制，空补丁能按原事实记录；mask 的单位调用输入占用变化可以通过轨迹复核，持久化模型分片已通过独立重建与哈希验证。尚未获得成功修复、修复率提升、SFT 收益或有效 Agentic RL 提升。

当前更紧迫的问题是从阅读进入正确复现、真实修改、可验证的提交，而不只是增加调用次数；本轮增加预算提示的最小干预也未解决这个问题。下一步应检验更具体的行为障碍及模型能力，保留未干预对照和新源码，再记录是否实际产生 diff、是否执行有效验证以及独立评分是否通过。不得硬编码该题修复答案或把 gold/隐藏测试反馈交给 Agent。

所有七次尝试来自同一道已知开发题，适应性对照还利用了此前观察；它们不是七个独立任务，不能给出总体 SWE-bench 成绩，也不能代表大模型、长程任务或招聘岗位上的通用能力。
