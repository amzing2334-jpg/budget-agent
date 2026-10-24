# 预设三组云端轨迹：预算消耗与无补丁机制

本分析读取 2026-10-22 回传产物，先分析预设三组，再在末尾独立分析后续适应性配对。任务均为开发集 `astropy__astropy-12907`；模型均为 `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`；总预算 32,000、上下文 8,192、单次输出上限 768。预设内部两组只改变最小动作输出预留 1/128；mini-SWE 是另一 harness，采用完整历史和显式提交，不能当作内部上下文消融。不向 agent 输入评分私有测试或修复答案。

**直接观察：三组均在生成有效补丁之前耗尽可用请求预算。内部主要消耗是重复输入；mini 还存在连续四次无实际改变的编辑命令。** 三组补丁均为 0 字节，官方状态均为 `official_empty_patch`，`resolved=null`、`tests_executed=false`。这不是“官方测试执行后失败”。

| 组 | 已完成调用 | 输入 / 输出 / 合计 token | 输入占比 | 输入:输出 | 剩余预算 |
| --- | ---: | ---: | ---: | ---: | ---: |
| recent，reserve=1 | 8 | 31,175 / 252 / 31,427 | 99.198% | 123.71:1 | 573 |
| recent，reserve=128 | 8 | 31,175 / 252 / 31,427 | 99.198% | 123.71:1 | 573 |
| mini-SWE | 6 | 25,570 / 1,230 / 26,800 | 95.410% | 20.79:1 | 5,200 |

## 内部 recent：裁剪发生得晚，动作仍停留在阅读和正常示例验证

两组 8 次调用的 **messages、response（含用量）、requested_max_output_tokens 逐项完全一致**。因此本次轨迹没有观察到 reserve=128 改变动作；不能据此断言该参数对其他轨迹无效。以下调用编号均从 1 开始，也对应 `calls.jsonl` 行号。

| 调用 | 输入 / 输出 | 动作与实际反馈 | 输入包含的事件 |
| --- | ---: | --- | --- |
| 1 | 756 / 17 | 搜索 `separability_matrix`，返回 22 处匹配 | 无 |
| 2 | 1,510 / 17 | 读 `separable.py` 第 1–100 行 | e1 |
| 3 | 2,842 / 31 | 读第 101–200 行 | e1–e2 |
| 4 | 4,096 / 31 | 读第 201–317 行，已看到相关实现 | e1–e3 |
| 5 | 5,573 / 15 | recall e1；此前的同一搜索结果仍完整在当前输入中 | e1–e4 |
| 6 | 6,440 / 49 | 运行非嵌套 `cm` 示例；缺少函数 import，发生 `NameError` | e1–e5 |
| 7 | 6,584 / 61 | 补全 import，输出正常的 2×2 对角矩阵；尚未复现 issue 中的嵌套失败 | e1–e6 |
| 8 | 3,374 / 31 | 再读第 304–317 行；该区间已包含在 e4 中，e4 仍在输入内 | e4–e7 |

证据：[reserve-1 调用](C:/找工作/budget-agent/runs/cloud-v5-a4000-20261022-01/reserve-probe/reserve-1/astropy__astropy-12907__recent__20261022T071042Z-2c771740/calls.jsonl:1)、[事件及输出](C:/找工作/budget-agent/runs/cloud-v5-a4000-20261022-01/reserve-probe/reserve-1/astropy__astropy-12907__recent__20261022T071042Z-2c771740/trajectory.jsonl:1)、[reserve-128 调用](C:/找工作/budget-agent/runs/cloud-v5-a4000-20261022-01/reserve-probe/reserve-128/astropy__astropy-12907__recent__20261022T071302Z-adc44a6d/calls.jsonl:1)。

**首次裁剪发生在 call 8，而不是 call 2 或 call 3。** 前 7 次输入都保留此前全部事件；call 8 前累计已用 28,022，剩余 3,978。此时总预算比单次上下文更先约束输入，输入上限分别为 3,977 / 3,850，recent 才丢弃 e1–e3、保留连续后缀 e4–e7。请求输入为 3,374，两组实际输出 allowance 同为 604；模型只生成 31，因此没有碰到输出截断。call 8 后剩 573，连固定系统提示与完整 issue 的 756-token 输入都放不下，第 9 次 context build 在模型调用前停止。

结果里的 `context_stats` 为 `builds=9, compactions=1, excerpted_events=0, last_visible_events=4, last_omitted_events=3`。`last_input_tokens=3374` 是最后成功构建的 call 8 输入，不是被拒绝第 9 次输入的测量值。冻结源码见 [source.tar.gz](C:/找工作/budget-agent/runs/v5-followup-20261022-01/cloud-receipts/source.tar.gz) 内 `src/budget_agent/context.py` 和 `runner.py`；工作区目前已开发新策略，不将其冒充本次运行源码。

### 重复输入的可核对分解

下面的“事件组”包含完整 action、provenance/header、observation 与消息框架，**不是单独观察文本的 token 成本**。call 1–7 的输入是相同 base 加逐次追加事件；各事件组成本由相邻精确输入计数相减得到。e7 的 130 由 call 8 的已知保留组重建。所有 revision 相同，组内容未变化，按实际出现次数加总严格闭合至 31,175。

| 事件组 | 每次边际 token | 被输入调用索引 | 累计 token |
| --- | ---: | --- | ---: |
| 固定 base | 756 | 1–8 | 6,048 |
| e1：搜索 | 754 | 2–7 | 4,524 |
| e2：文件第 1–100 行 | 1,332 | 3–7 | 6,660 |
| e3：第 101–200 行 | 1,254 | 4–7 | 5,016 |
| e4：第 201–317 行 | 1,477 | 5–8 | 5,908 |
| e5：recall 搜索 | 867 | 6–8 | 2,601 |
| e6：NameError | 144 | 7–8 | 288 |
| e7：正常示例输出 | 130 | 8 | 130 |

历史事件组累计 25,127；每组首次输入后的再次传输为 **19,169 token，占全部输入 61.488%**。三个 read_file 组累计 17,584；搜索与其 recall 组累计 7,125。另从原始 observation 字符长度计数，观察体共曝光 73,968 字符，不能将字符数直接当 token。

这里的“再次传输”是账本描述，不等于所有重复证据都没价值。`diagnostics.repeated_observations=0` 也不矛盾：该指标检查同一动作、输出和 revision 的重复事件；本次 call 5 是 recall，call 8 是区间重叠阅读，它们不是同一事件动作的逐字重复。`associated_request_input_tokens` 覆盖整个请求且彼此重叠，本文没有将其相加作为观察成本。

## mini-SWE：合法 Bash 动作没有产生有效修改

| 调用 | 输入 / 输出 | 实际 action 与观察 |
| --- | ---: | --- |
| 1 | 1,667 / 152 | `grep -r "separability_matrix" /testbed/astropy/modeling/`；返回码 0，输出 2,010 字符 |
| 2 | 2,373 / 170 | `cat /testbed/astropy/modeling/separable.py`；返回码 0，输出 9,878 字符 |
| 3 | 4,978 / 258 | `sed` 将 `def _cstack(left, right):` 替换为同一字符串；返回码 0、空输出 |
| 4 | 5,260 / 250 | 再次执行 call 3 的相同 no-op；返回码 0、空输出 |
| 5 | 5,534 / 200 | `sed` 将 `    if isinstance(left, Model):` 替换为同一字符串；返回码 0、空输出 |
| 6 | 5,758 / 200 | 再次执行 call 5 的相同 no-op；返回码 0、空输出 |

证据：[mini calls 1–6](C:/找工作/budget-agent/runs/cloud-v5-a4000-20261022-01/mini-baseline/astropy__astropy-12907__mini-swe-textbased/calls.jsonl:1)、[官方 mini trajectory](C:/找工作/budget-agent/runs/cloud-v5-a4000-20261022-01/mini-baseline/astropy__astropy-12907__mini-swe-textbased/mini.traj.json)。有 6 个合法解析 action、0 个格式错误、6 个返回码 0 的观察；没有验证命令、diff 检查或显式 submission。call 3–6 合计消耗 22,438 token。返回码 0 仅说明这四条替换命令正常结束，不能当作有效编辑证据。

调用输入严格追加历史：每次后续 prompt 的前缀等于上一次 prompt。call 6 输入已有 5,758 token，输出后全局只剩 5,200；下一次请求仍包含这段历史并追加新的 action/observation，预算门返回“允许 0 输出、至少需要 128”，所以第 7 次没有进入模型。原生 `api_calls=7` 是上游在预算检查前递增；真实后端调用、预算账本与 calls 文件都为 6。

补充测量使用同一固定 tokenizer 和 chat template，对已保存官方 trajectory 的 messages 去掉末尾 exit 消息后重建计数：**下一请求输入为 5,982 token**。输入本身已比剩余 5,200 多 782；连同最小输出预留 128，需要 6,110，比剩余额度多 910。因此总预算明确是本次入场障碍，而 6,110 仍小于 8,192 的单次上下文上限。这是云端读取已保存文本后执行的 **tokenizer-only 离线重建测量，没有发生第 7 次模型推理**；已核对回传的 [mini-stop-token-count.json](C:/找工作/budget-agent/runs/cloud-v5-a4000-20261022-01-receipts/mini-stop-token-count.json)。

在 6 个实际请求中，固定 base 累计 1,667×6=10,002 输入 token；含源码读取的 call 2 事件组边际 2,605，被后续 4 次完整重放，累计 10,420。该分解仍包含 action 和消息框架，不能全部归因于源码正文。mini 的空补丁来自缺少显式 submission；本轨迹同时显示四个 no-op，没有实际修改证据，故不能将问题只归因于提交方式。

## 对下一轮 mask 对照的可验证判断

1. **有明确输入开销靶点。** recent 在 32K 累计预算下很晚才裁剪；提前屏蔽旧 observation 可以检验重复输入是否减少、可用动作次数是否增加。实际节约比例必须由新轨迹计量，不能把上述 19,169 全部视为 mask 可省 token，因为 mask 仍保留 action/header 等内容。
2. **动作质量仍是独立瓶颈。** 内部轨迹没有复现嵌套失败，mini 出现连续 no-op；增加可用轮次并不逻辑保证正确复现、有效修改或提交。下一轮应分别观察首次触发错误示例、首次真实 edit/diff、验证和提交，最终仍以独立评分为准。
3. **“最近两条”可能屏蔽仍需使用的源码。** 以本次内部序列为例，后续 recall 搜索和错误 traceback 会使相关源码观察不再属于最近两条。是否通过保留动作定位并 recall 恢复、是否产生新的恢复成本，需要实测，不能预判净收益。
4. **mask 仍有上下文边界。** 完整 action 历史会增长，最近两条观察也可能很长；该策略不自动保证任何长度轨迹都可容纳。应保留预算拒绝/上下文拒绝的真实停止原因。

新源码 recent/mask 属于看到上述轨迹后设计的适应性实验，应独立标记；不能合并进这三组的预设结果，也不能把单任务变化推广为 SWE-bench 成功率或大模型通用结论。

## 适应性 recent/mask 配对：预算换来了更多浏览，尚未换来修复进展

以下是随后回传的独立适应性结果。[执行前声明](C:/找工作/budget-agent/runs/v5-followup-20261022-01/cloud-receipts/adaptive-plan.json) 明确是在预设结果之后提出的开发诊断。两组共同源码指纹为 `209a8cff1285870eb9456fdb1db5f38c1e02ef73a987b82ab53ffe1961a4ca4f`；逐字段核对 source、task、initial_revision、model、model_metadata、environment_provenance 完全相同，config 仅 strategy 不同。它是一次可比较的同源码单任务配对，不是多任务统计证据。

| 指标 | 新源码 recent | 新源码 mask |
| --- | ---: | ---: |
| 调用数 | 8 | 13 |
| 输入 / 输出 / 总 token | 31,175 / 252 / 31,427 | 31,339 / 347 / 31,686 |
| 每调用平均输入 | 3,896.88 | 2,410.69 |
| search / read_file / recall / run | 1 / 4 / 1 / 2 | 4 / 9 / 0 / 0 |
| edit / write / diff | 0 / 0 / 0 | 0 / 0 / 0 |
| 最终补丁 | 0 字节 | 0 字节 |
| 官方评价 | 空补丁、未执行测试 | 空补丁、未执行测试 |

mask 的调用数增加 62.5%，总用量增加 0.824%，平均每次输入减少 38.14%。后续动作已经分叉，平均差异不能全部当作固定轨迹压缩率。一个更直接的局部证据是：前三次 messages/response 完全一致，call 4 只有 e1 observation 被 marker 替换，输入从 4,096 降到 3,429，减少 667（16.28%），输出动作仍相同。call 5 起动作分叉。

mask 前四个动作与 recent 相同，call 4 结束时已读完整 `separable.py` 第 1–317 行。后续轨迹如下；所有事件 revision 相同，逐行验证重复区间文本也完全相同。

| mask call | 输入 / 输出 | 动作 | 实际新增源码行 |
| --- | ---: | --- | ---: |
| 5 | 3,661 / 22 | 在同文件搜索 `separable_matrix` | 0，匹配都在已读行 |
| 6 | 2,759 / 30 | 读 50–110 | 0 / 61 行 |
| 7 | 2,348 / 22 | 搜索 `_separable` | 0，匹配都在已读行 |
| 8 | 2,765 / 31 | 请求 290–400，实际只返回 290–317 | 0 / 28 行 |
| 9 | 2,410 / 22 | 搜索 `_cstack` | 0，匹配都在已读行 |
| 10 | 2,104 / 31 | 读 200–220 | 0 / 21 行 |
| 11 | 2,101 / 31 | 读 219–230 | 0 / 12 行 |
| 12 | 2,280 / 31 | 读 231–250 | 0 / 20 行 |
| 13 | 2,374 / 31 | 读 251–270 | 0 / 20 行 |

9 次 read_file 实际累计返回 479 行，唯一行 317，重复覆盖 162 行；后六次 read_file 均完全重读已见区间。对照 recent 实际返回 331 行、唯一行同为 317、重复 14 行。额外的 call 9–13 消耗 11,415 token，call 5–13 共消耗 23,053；没有 run、编辑、diff、失败复现或补丁。搜索逐渐聚焦到具体函数，但这是浏览范围的收窄，不能作为已完成修复步骤。最终剩 314，已不足固定 base；第 14 次模型调用没有发生。

已用新增的离线覆盖诊断交叉核对上述数字，两组所有 read_file 输出格式均可解析；诊断不改写 result。原 mask 结果里的 48,195 个“观察字符”含 55 条遮蔽 marker、共 2,915 字符；修正后实际 observation/excerpt 字符为 **45,280**，marker 单列，recent 仍为 73,968。原始结果保留旧值，不能把 marker 当作恢复的原始证据。

证据：[mask calls](C:/找工作/budget-agent/runs/cloud-v5-a4000-20261022-01/adaptive-mask/astropy__astropy-12907__mask__20261022T072704Z-bbb60603/calls.jsonl:5)、[mask 原始事件](C:/找工作/budget-agent/runs/cloud-v5-a4000-20261022-01/adaptive-mask/astropy__astropy-12907__mask__20261022T072704Z-bbb60603/trajectory.jsonl:5)、[同源码 recent](C:/找工作/budget-agent/runs/cloud-v5-a4000-20261022-01/adaptive-mask/astropy__astropy-12907__recent__20261022T072506Z-86c192b2/result.json)。

这更符合“反复重新获取已见源码”的行为，不能只因 `repeated_observations=0` 就说没有循环：原指标只识别完全相同动作，而本次改变了搜索词和读取区间。另一方面，旧观察正文确实被遮蔽，重新读取可能是恢复当前可用证据，不能直接认定每次恢复都没必要，更不能仅凭这一个配对确定 mask 导致了全部重复。call 7 开始 e4 的源码正文被屏蔽；call 9 后模型转向该区间的小段重读，轨迹支持继续检验这一机制。

下轮优先实现：

1. **先补可审计的覆盖指标。** 以实际 read_file 输出 header 和行号前缀计算同 revision/path 的新增行、重复行、累计区间；不能用请求的 290–400 冒充实际读了 111 行。遮蔽 marker 单独计数，不算原始观察证据。这是计量修正，无需模型调用。
2. **把一个最小“停滞反馈”作为独立干预。** 基线保留当前 mask；实验组在连续若干浏览动作无新增源码覆盖、也无代码变化/执行验证时，提供一次短且计入预算的反馈：哪些 revision/区间已读、已有可引用 event ID，并要求下一步选择可检验的进展或说明还缺的证据。保留合法重新读取通道，不硬性强迫编辑，不写入任务答案。只改变这一因素，冻结预算与模型。
3. **验证输出价值而不是调用数量。** 分别记录新的源码覆盖、真实 issue 复现、非空有效 diff、独立评分。若反馈仍只延长浏览，则再独立检验保留关键证据或模型能力；不要同时扩预算、换模型、加提示，再把变化归给 mask。

本次支持的结论止于：mask 降低了部分历史传输成本，允许更多调用；在这一任务与模型上，新增调用没有产出可执行验证或补丁。

## 最后一次适应性 guidance 配对：提示改变了浏览顺序，仍没有真实修复进步

最后一组共同使用 mask；两组源码指纹同为 `d3ca73f7db6c7ecd49832e1c04d9d87fa08b73092b7dd1189a19d83fe98a4528`。只读逐字段核对 source、task、initial_revision、model、model_metadata、environment_provenance 均一致，config **仅 `decision_policy=direct/budget-aware` 不同**。因此下表比较的是这次同源码配对，不将此前旧源码结果混入。

| 指标 | direct | budget-aware |
| --- | ---: | ---: |
| 模型调用数 | 13 | 13 |
| 输入 / 输出 / 总 token | 31,339 / 347 / 31,686 | 31,440 / 361 / 31,801 |
| 输入占比 | 98.905% | 98.865% |
| search / read_file | 4 / 9 | 3 / 10 |
| 实际读取行 / 唯一行 / 重复行 | 479 / 317 / 162 | 457 / 317 / 140 |
| run / edit / write / diff | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| patch / 官方状态 | 0 字节 / official_empty_patch | 0 字节 / official_empty_patch |
| resolved / tests_executed | null / false | null / false |

已检查实际 calls：budget-aware 从 call 1 就在系统提示中加入剩余总预算、步骤、工作区是否变化、连续浏览次数以及复现失败情形和适时编辑的提醒；它不是只在某个覆盖阈值之后触发的一次干预。call 1 输入从 direct 的 756 增为 917，并把搜索范围收窄至目标文件；后续整体用量差额 115 不能单独归因为提示成本，因为动作和观察同时变化。

budget-aware 的 call 2–4 仍完整读取 1–317 行。之后所有读取仍是恢复已见源码，没有新增行：call 6 读 58–70，call 7 请求 290–320 而实际返回 290–317，call 9–13 依次读 219–235、236–250、251–270、271–290、291–317。call 5、8 是同一文件内的符号搜索。call 5–13 合计消耗 23,332 token。相比 direct，重读行减少 22、搜索少一次，但读取动作多一次，没有转入执行验证或编辑。

到 call 13，实际提示已明确记录“剩余 2,817、工作区未变、连续 12 次浏览”；模型仍选择再次读取 291–317。输入 2,587、输出 allowance 230、实际输出 31，结束剩 199。下一次构建因固定系统提示和完整 issue 放不下而停止；这里没有输出截断、非法动作或工具执行错误作为替代解释。

证据：[budget-aware calls](C:/找工作/budget-agent/runs/cloud-v5-a4000-20261022-01/adaptive-guidance/budget-aware/astropy__astropy-12907__mask__20261022T074654Z-43348b7b/calls.jsonl:1)、[budget-aware call 13](C:/找工作/budget-agent/runs/cloud-v5-a4000-20261022-01/adaptive-guidance/budget-aware/astropy__astropy-12907__mask__20261022T074654Z-43348b7b/calls.jsonl:13)、[budget-aware 事件](C:/找工作/budget-agent/runs/cloud-v5-a4000-20261022-01/adaptive-guidance/budget-aware/astropy__astropy-12907__mask__20261022T074654Z-43348b7b/trajectory.jsonl:6)、[同源码 direct 结果](C:/找工作/budget-agent/runs/cloud-v5-a4000-20261022-01/adaptive-guidance/direct/astropy__astropy-12907__mask__20261022T074532Z-2df7bd07/result.json)。

**本轮没有获得真实修复能力的显著进步，也没有观察到任何成功修复。** 工程上的增量是可运行的外部基线、可信预算与独立评分、覆盖诊断和两次可审计的适应性配对；这些让失败机制更清楚，但不等价于修复收益。调用更多、重读少 22 行、提示状态更新均不足以支撑能力提升结论。

这次负结果也不证明所有预算感知策略无效：它只检验了当前 4B 模型、当前单任务、固定预算下这一条逐步提示。下一轮应先验证模型在清晰任务输入下能否完成“失败复现→有效编辑→验证”的最小能力链，再单独检验证据恢复或控制策略；不要继续叠加提醒并把 token 或调用数变化包装成效果。保留全部开发结果，使用事先冻结的额外任务检验任何新方案，避免在同一例题上反复调到成功后宣称泛化。
