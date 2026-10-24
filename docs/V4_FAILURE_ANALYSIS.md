# V4 两模型失败分析：历史协议修复没有产生有效补丁

本文件先分析 `cloud-v4-20261022-01-protocol` 中的 A 组 Qwen2.5-Coder-7B-Instruct，再分析 `cloud-v4-qwen3-20261022-01-protocol` 中的 B 组 Qwen3-4B-Instruct-2507。每组在同一模型、同一开发任务、同一预算上限下比较 `transcript` 与 `chat` 两种历史表示。结论来自本地已完整下载的 `calls.jsonl`、`trajectory.jsonl`、`result.json` 和官方回填后的汇总；没有读取 gold patch、独立 grader 答案或隐藏测试，也没有为本分析运行模型。

两组、四次尝试均为空补丁，`resolved=null`，官方没有执行独立测试。A 组主要停在重复失败编辑，B 组甚至没有成功读到源码正文，主要停在动作协议、错误路径和重复浏览。局部格式行为的变化不能替代有效修改与验证。

**A 组没有显示 chat 提高了任务完成效果。** 两种格式都没有成功编辑、没有运行命令、没有检查 diff，最终补丁均为 0 字节。chat 在这一次轨迹中没有再模仿 `Action/Observation` 日志格式，但遇到编辑歧义后重复相同错误更多，最终耗尽预算。协议表达正确是工程条件，不能单凭非法动作减少就声称修复能力改善。

## A 组：实验条件与结果口径

- 模型与 tokenizer：`Qwen/Qwen2.5-Coder-7B-Instruct@c03e6d358207e414f1eca0bb1891e29f1db0e242`，bfloat16，`do_sample=false`，seed 42。
- 任务：`astropy__astropy-12907`，dev；每格式仅一次尝试。
- 策略与预算：`recent`，context 8,192，累计输入加输出 32,000，每次输出最多 768，最多 20 步。
- 两格式共用 source fingerprint `83cc596ad244cf761e532a560ef7f05cb6b9d911460b04c1779ab08de9a2976c` 和镜像 ID `sha256:a7cc2f990459911b958bd75b84fb89205fa6a6b33e0ee1cf9b33b175b8fd0df0`。
- 两次运行预算记录均有效，无 overrun、未知用量或未结算请求。此处“同预算”指相同上限，并非强制花完相同 token 数。

| 指标 | transcript | chat |
|---|---:|---:|
| 结束状态 | finished | budget_exhausted |
| 模型调用 / 步数 | 10 / 10 | 12 / 12 |
| 输入 token | 24,204 | 31,286 |
| 输出 token | 609 | 714 |
| 合计 token | 24,813 | 32,000 |
| 剩余 token | 7,187 | 0 |
| invalid_action | 2 | 1 |
| 对目标源码的同一歧义编辑尝试 | 4 | 8 |
| 同一失败编辑在第一次之后的重复次数 | 3 | 7 |
| 成功 edit_file / write_file | 0 / 0 | 0 / 0 |
| run / diff / search / recall | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| patch.diff 字节数 | 0 | 0 |
| 官方回填状态 | official_empty_patch | official_empty_patch |
| 独立测试是否执行 | false | false |
| resolved | null | null |

`pipeline.exit=0` 只能说明流程执行完成。这里官方将空补丁记录为未执行测试的 `official_empty_patch`，不能写成“通过了官方评测”，也不能把 `resolved=null` 改写成有隐藏测试依据的 `false`。工程上已经确定两次运行都没有交付修改，但没有可据此计算的已评分修复成功率。

## 首轮完全相同：错误发生在历史格式生效之前

逐字比较两边第一次调用，`messages` 相同，`response.content` 也相同。输入均为 705 token、输出均为 138 token；原始输出的 SHA256 均为：

```text
a3849f5d9b1683dfe2e1c0b4b26aa7a08105981201cb86ab33c0afb0076b5eb6
```

两者都没有先检查仓库，而是向 `relative/file.py` 发出 `edit_file`，把 issue 中展示的两个数组字符串作为 old/new。`relative/file.py` 正好是 SYSTEM 的工具示例占位路径；工具反馈 `FileNotFoundError`。数组也只是问题描述中的运行输出，尚无证据表明它们是仓库中可编辑的源码。

这支持“当前模型和提示组合会把示例占位符当实际操作对象”的诊断，但尚未做提示消融，不能断言只替换示例就能解决。更不能把这个共同的首轮错误归因于历史角色：第一次请求根本没有执行历史。

问题描述将右下角对角块作为预期的独立关系，并展示嵌套时出现的非对角相关块；首轮 old/new 的方向还把前者替换为后者。这里能够确认的是模型直接套用了问题中的输出文本，而非基于已检查的实现构造补丁；不能把这个动作视为有效修复尝试。

## 两条失败链

| 阶段 | transcript | chat |
|---|---|---|
| 第一次操作 | 第 1 步：编辑占位路径，文件不存在 | 第 1 步：完全相同 |
| 从文件错误恢复 | 第 2 步输出伪造的事件日志与 `Observation: ['relative/file.py']`，被解析器拒绝；第 3 步才合法 list_files | 第 2 步合法 list_files |
| 获取代码 | 第 4 步读取 `astropy/modeling/separable.py` 第 1–100 行 | 第 3 步读取相同文件和行范围，工具输出相同 |
| 第一次目标源码编辑 | 第 5 步：将 `_separable(transform)` 调用改为增加 `include_nested=True`，匹配到 2 处，拒绝编辑 | 第 4 步：增加 `nested=True`，同样匹配到 2 处，拒绝编辑 |
| 面对歧义错误 | 第 6 步再次生成 Action/Observation 日志；第 7–9 步重复完全相同编辑 | 第 5–11 步连续重复完全相同编辑 |
| 终止 | 第 10 步 finish，无修改、无验证，尚余 7,187 token | 第 12 步生成未完成的编辑 JSON，输出额度用尽；随后累计预算耗尽 |

对目标源码的所有编辑都使用相同 old：

```python
separable_matrix = _separable(transform)
```

读到的 100 行代码中，第 60 行和第 100 行就同时出现该字符串；工具返回：

```text
ValueError: edit_file old must match exactly once; found 2
```

这个反馈有明确的操作含义：单行 old 不是唯一锚点。两条轨迹都没有扩大 old 的上下文、没有搜索调用或定义、没有继续读后续代码，也没有改变编辑目标。新增的两个关键字参数均没有经过函数定义检查或运行验证；在现有证据范围内不能将它们称为正确修复方案。

所有事件的 revision 均为 `e3b0c44298fc1c14`，与空 diff 状态一致；所有 edit_file 都返回失败。没有“有效修改被最终导出遗漏”的迹象。`finish` 的工具成功只代表提交动作被接收，不代表存在补丁或通过测试。

## 反馈确实仍在输入中，主要循环不能解释为裁剪掉了错误

逐次检查实际传给模型的消息，而非只看 canonical trajectory：

| 请求位置 | 可见内容 | 由证据支持的判断 |
|---|---|---|
| transcript 第 6–10 次 | 完整的源码第 1–100 行；此前 `found 2` 错误逐次累积，最多 4 条 | 在错误反馈仍可见时，模型继续重复编辑或直接 finish |
| chat 第 5–11 次 | 完整的源码第 1–100 行；此前 `found 2` 错误逐次累积到 7 条；assistant 原动作与 user 观察成对保留 | 角色修复和完整反馈都没有使这次轨迹摆脱重复 |
| chat 第 12 次 | 剩余预算缩小上下文，保留 e0006–e0011 六个完整失败事件；源码读取事件已退出窗口 | 最后一步确实丢失了源码，但此前已有七次重复；不能用这一晚出现的裁剪解释循环起因 |

这里“反馈未被利用”描述的是可观察行为：错误仍在提示里，而下一次动作没有响应它进行调整。不声称能够从日志判断模型内部是否注意到或理解了这些 token。

## list_files 确实浪费预算并触发裁剪，但不能证明它导致了这次修复失败

两条轨迹的根目录 list_files 都产生 71,052 个字符。紧接着的模型请求都占满 7,424 个输入 token（8,192 减预留输出 768），recent 只能保留目录列表的开头片段，并丢弃更早事件。

transcript 第 4 次的目录观察消息为 23,282 字符；chat 第 3 次为 23,211 字符。两者都在 `astropy/io/misc/asdf/...` 附近结束。完整目录中目标 `astropy/modeling/separable.py` 出现在字符偏移 28,046，**不在任何一个列表节选里**。模型仍选中了正确的路径；issue 已包含相应模块 import，因此不能仅凭路径正确推断模型使用了被裁掉的目录信息，也不能据此断言训练数据记忆。

成功 read_file 后，recent 后缀中不再包含那个过大的目录列表。此后直到末尾预算收紧之前，读到的 100 行源码和工具错误都完整保留。因而：

- 无界目录输出造成一次明显的 token 浪费，是有直接记录支持的 harness 问题。
- 这次没有证据证明“丢了关键错误反馈”触发重复编辑；实际记录反驳了这个具体解释。
- 两边都只读了前 100 行，既有 read_file 输出没有总行数提示，但模型也没有尝试翻页。工具导航信息不足和模型缺少进一步检查，都可能有影响；当前实验无法分离它们。
- `context_stats.compactions` 为 7 与 10，不等于有 7 与 10 次输出被截断。canonical 历史包含大列表，后续 build 仍进入压缩路径；真正记录的 excerpted_events 两边都只有 1。

## invalid_action 的类型必须拆开看

transcript 的第 2、6 次原始输出都生成了 `[event_id=...]`、`Action:`、`Observation:` 这样的日志文本，甚至包含模型自写的观察结果；这与输入历史的序列化形式一致，解析器正确拒绝了它们。这是本次运行中具体可见的协议混淆。

chat 在第 1–11 次没有这种伪造日志回复。唯一的 invalid_action 出现在第 12 次：

- 调用前累计预算只剩 1,686 token。
- 实际输入消耗 1,643 token，获准输出只有 43 token。
- 实际输出也恰为 43 token，文本停止在 `"new":`，没有完成 JSON。

这支持将最后一次记作**预算限制下的截断格式错误**，与日志模仿错误分开归类。只比较 invalid 的 2→1 会掩盖两个事实：chat 的自发日志格式混淆在这一条轨迹中未出现，但有效编辑仍为零，而且它把更多预算消耗在重复失败上。

累计 token 中输入占比分别为 97.55% 与 97.77%。这不是大规模输出推理耗尽预算的案例；主要消耗来自目录大输出和重复发送历史。第 12 次的小额输出调用是总预算约束的结果，不应被包装成模型独立的新能力退化证据。

## 目前可以与不可以得出的结论

可以确认：原始 assistant/工具观察的角色分离已在云端请求中生效；首轮严格相同；这一个任务上 chat 没有生成伪造 Action/Observation 日志，但没有交付补丁。两边共同暴露了占位示例误用、代码检查不足、不能根据唯一匹配错误调整编辑，以及没有通过实际测试验证修复的问题。

不能确认：chat 提高了总体修复率、7B 模型普遍没有 agent 能力、增加预算可以修好、工具反馈本身足够理想、换更大或更新模型即可解决。每条件只有一个开发任务的一次贪心运行；改变历史格式同时改变角色、原文保留方式、序列化 token 开销和后续历史窗口，不能将差异进一步归因于其中某一个组成部分。

模型比较应固定 chat harness、任务和预算口径。下文 B 组提供的是另一模型在当前开发任务上的诊断，不能充当 A 组协议对照的第二次重复。A 组证据更支持先检查“模型能否在看到明确失败反馈后改变操作、继续读实现并产生可验证补丁”，而不是把少一个 invalid 或更多合法工具调用当作项目进展。

## 可复核证据

- [计划与固定条件](../runs/cloud-v4-20261022-01-protocol/plan.json)
- [官方回填后的协议汇总](../runs/cloud-v4-20261022-01-protocol/protocol-after-grading.json)
- [transcript calls](../runs/cloud-v4-20261022-01-protocol/transcript/astropy__astropy-12907__recent__20261022T022547Z-f55e1eb3/calls.jsonl)
- [transcript trajectory](../runs/cloud-v4-20261022-01-protocol/transcript/astropy__astropy-12907__recent__20261022T022547Z-f55e1eb3/trajectory.jsonl)
- [transcript result](../runs/cloud-v4-20261022-01-protocol/transcript/astropy__astropy-12907__recent__20261022T022547Z-f55e1eb3/result.json)
- [chat calls](../runs/cloud-v4-20261022-01-protocol/chat/astropy__astropy-12907__recent__20261022T022630Z-22853d0c/calls.jsonl)
- [chat trajectory](../runs/cloud-v4-20261022-01-protocol/chat/astropy__astropy-12907__recent__20261022T022630Z-22853d0c/trajectory.jsonl)
- [chat result](../runs/cloud-v4-20261022-01-protocol/chat/astropy__astropy-12907__recent__20261022T022630Z-22853d0c/result.json)

## B 组：更少非法动作，仍未成功读取源码

B 组使用 `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`，模型和 tokenizer revision 固定；与 A 组共用任务、镜像 ID、source fingerprint、recent 策略和 8,192 / 32,000 / 768 / 20 的预算限制。两种格式都记录为预算有效，最终累计 token 恰好为 32,000，无超支、未知用量或未结算请求。

| 指标 | B-transcript | B-chat |
|---|---:|---:|
| 结束状态 | budget_exhausted | budget_exhausted |
| 模型调用 / 步数 | 10 / 10 | 11 / 11 |
| 输入 token | 30,360 | 31,869 |
| 输出 token | 1,640 | 131 |
| invalid_action | 4 | 2 |
| 执行 list_files | 3 | 4 |
| 执行 search | 3 | 3 |
| 执行 read_file | 0 | 2，均文件不存在 |
| 成功读取源码正文 | 0 | 0 |
| 执行 edit_file / write_file / run / diff | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| repeated_identical_failed_actions | 0 | 0 |
| patch.diff 字节数 | 0 | 0 |
| 官方回填 / tests_executed / resolved | official_empty_patch / false / null | official_empty_patch / false / null |

两边都没有空字符串回复，没有 `{}` 空动作。末尾的 `{"` 是模型输出的两个字符组成的未完成 JSON 前缀，该 tokenizer 计为 1 个 token；必须与真正的空回复区分。

### B 组首轮一致，均一次提出三个动作

逐字核对两边首轮 `messages` 和 `response.content`，均完全相同，输入 705 token、输出 31 token，原文为：

```json
{"tool":"list_files"}
{"tool":"read_file","path":"relative/file.py"}
{"tool":"search","query":"separability_matrix"}
```

首轮原始回复 SHA256 为：

```text
9623dcc04787c3921a9c9227b01ff87f585744965ee5c20f0a30b801bb2141ca
```

这是三个相邻 JSON 对象，不符合一次一个动作的协议。解析器整体拒绝，三个工具均未执行；这里不能计入一次成功 list 或 read。此时输出距离 768 上限很远，因此是完整回复违反动作基数约束，不是预算截断。`relative/file.py` 的示例占位符也再次出现。该共同错误发生在历史介入之前，历史格式不能解释它的起因。

### B-transcript：检索后输出整段虚构操作过程，再退回重复浏览

第 2 步成功列目录，第 3 步成功搜索 `separability_matrix`。第 4 步没有发出一个可以执行的后续读文件请求，而是连续输出 list、read、多个 search、edit、run、diff、remember、recall 等动作，直到 768 输出 token 截断。

这段文本中的 read、edit、run、diff 都没有被执行。尤其是模型尚未读取任何源码正文，就给出大段函数 old/new，并写出类似“Fixed separability matrix...”的 remember 文本；这些是未执行的模型计划或陈述，不能当作补丁或修复证据。路径也带有不存在的 `relative/` 前缀。即便给它更多输出空间使尾部 JSON 完整，前面的多动作回复仍违反当前协议，因此不能把这次错误全部归因于输出上限。

第 5–6 步重新 list/search，第 7 步再次输出多动作及未经读取验证的函数替换，恰好生成 768 token 后停止。第 8–9 步第三次 list/search，第 10 步只剩 1 个输出 token，生成 `{"` 后累计预算耗尽。

| 非法动作所在步 | 可确认的错误类型 | 是否受输出额度影响 |
|---|---|---|
| 1 | 一次输出三个 JSON 对象 | 否：实际 31 / 获准 768 |
| 4 | 多动作回复；还包含未经执行的修复陈述 | 是：实际 768 / 获准 768，尾部不完整；多动作错误在截断前已存在 |
| 7 | 多动作回复；未检查源码就生成长函数编辑文本 | 是：实际 768 / 获准 768，尾部不完整；多动作错误在截断前已存在 |
| 10 | 总预算尾部的未完成 JSON 前缀 | 是：实际 1 / 获准 1 |

### B-chat：动作数量恢复到单个，但路径语义与浏览循环未解决

第 2 步输出单个 read_file，但路径是 `relative/file.py`，工具返回文件不存在。第 3 步列目录，第 4 步搜索得到正确结果。第 5 步选择了正确模块文件名，却请求：

```text
relative/astropy/modeling/separable.py
```

真实搜索结果给出的是 `astropy/modeling/separable.py`。模型额外添加了 `relative/` 前缀，工具明确报错 `/testbed/relative/astropy/modeling/separable.py` 不存在。这是合法工具调用中的错误路径值，不是缺少工具必填参数，也不是 JSON 解析失败。

第 6–7 步 list/search；第 8–9 步再次 list/search；第 10 步第四次 list；第 11 步只生成 `{"`，随后预算耗尽。chat 在初次多动作失败后确实改为单动作，但没有成功打开任何源码，合法调用数量增加没有转化为修复进展。

### 搜索结果有效，关键路径在做错下一步时完整可见

两边每次 search 都成功返回相同的 2,320 字符结果，包含正确的仓库路径和行号，例如：

```text
astropy/modeling/separable.py:66: def separability_matrix(transform):
astropy/modeling/separable.py:304:     if (transform_matrix := transform._calculate_separability_matrix()) is not NotImplemented:
astropy/modeling/core.py:808:     def _calculate_separability_matrix(self):
```

逐字检查实际模型输入：transcript 第 4、7 次请求中的 search 输出完整保留，仍发生多动作长回复；chat 第 5 次中完整保留，仍加上错误路径前缀；chat 第 8、10 次中也完整保留，但下一步再次 list。因而，这些具体的错误不能归因于“搜索没有找到代码”或“正确文件路径在那次请求里被截掉了”。

与此同时，recent 与重复 list_files 形成了有记录支持的上下文丢失链：每次列目录返回相同的 71,052 字符，下一次请求为了容纳最新目录节选，只保留该 list 事件，先前的格式错误或错误路径事件离开窗口。随后 search 返回时，大目录又装不进窗口，下一次提示通常只剩该 search 事件。模型没有重新看见完整的浏览历史，行为在 list/search 之间循环。

这个机制具有可疑的因果联系，但本实验没有“保留历史但控制同样输入成本”的对照，因此只能说它可能维持循环，不能认定是唯一原因。即使只看当前完整的搜索结果，模型已经拥有做一次正确 read_file 的路径信息，仍未做到。

`repeated_identical_failed_actions=0` 不表示没有循环：这个指标只统计失败工具动作的重复，并排除 invalid_action。B-transcript 将相同成功 list 和 search 分别做了 3 次；B-chat 分别做了 4 次和 3 次，重复调用返回的内容也逐字相同。它们是“工具成功、没有带来新证据”的循环，现有失败重复计数没有覆盖。

### 两个末尾非法动作都是 1-token 预算尾截断

| 格式 | 最后请求前剩余总 token | 最后输入 | 获准输出 / 实际输出 | 最后可见工具证据 |
|---|---:|---:|---:|---|
| B-transcript | 742 | 741 | 1 / 1 | search 事件被裁到只剩不完整头部和 excerpt 标记，已没有搜索正文 |
| B-chat | 1,075 | 1,074 | 1 / 1 | 保留完整 assistant 动作及 provenance，但目录观察只剩开头片段 |

B-transcript 的最后观察只有 86 字符，是旧 transcript 裁剪行为；B-chat 保留了事件组契约，最后工具观察为 1,204 字符。两个调用都不具备足够空间完成通常的动作 JSON，不能拿最终 invalid 来支持模型独立发生了新的格式退化。

B-transcript 输入占总量 94.88%，B-chat 为 99.59%。chat 全程只生成 131 token，却花完 32,000 token 预算，说明此次代价主要来自无效浏览及反复发送大观察，而不是长篇模型输出。每个分支都有三次 7,424-input-token 的目录后续请求；仅这三次输入就各消耗 22,272 token，占总预算 69.6%。

## A、B 合并后的实施优先级与证据边界

1. **先保证单步动作和路径契约可用，再扩大任务规模。** A 的两种格式第一轮都使用占位路径；B 两种格式第一轮同时输出三个动作，chat 后续又把 `relative/` 当作目录。下一次可独立比较更明确的动作模式和仓库相对路径说明，评估能否根据真实搜索结果发出一个正确读取。应保持这些变更可区分，不要为当前错路径写静默自动纠正，也不要把多 JSON 中的第一个动作擅自执行来掩盖协议错误。
2. **优先限制目录与检索观察的浪费，验证是否保留了有效反馈。** 四次尝试都出现无界目录输出；B 每分支至少 69.6% 的总预算用于三次大目录后的输入。范围化或有界文件浏览、明确的分页信息以及可审计的观察压缩，有比新增复杂规划组件更直接的失败依据。仍需受控重跑，不能预先宣布这些修改会提升修复率。
3. **诊断“反馈后能否改变动作”，同时覆盖失败编辑和成功浏览两类循环。** A 的错误编辑反馈始终可见但操作不改；B 的成功搜索结果完整可见却没有正确读文件。应同时记录源码读取是否推进、重复调用是否产生新证据，以及从报错到有效下一步的转换，不能只看 invalid 数或 repeated_failed 数。不要用硬编码重试或强行禁止 finish 替代模型决策能力。
4. **把预算尾截断独立报告，下一轮单独审查输出预留策略。** A-chat 末次仅 43 输出 token，B 两边末次仅 1 token。避免将这些尾部错误与完整预算下的协议违约混合；若修改预留或终止规则，应提前声明并作为另一变量记录，而非把减少末尾 invalid 算成修复能力增益。

这些优先级针对目前可复现的失败路径，不构成 Agentic RL 的训练信号设计结论。四次尝试都没有成功补丁或独立测试结果，目前不能声称已具备可用的代码修复 rollout，更不能据此报告训练后提升。

此次更换为 Qwen3-4B 没有带来有效修复：它产生了不同的失败模式。不同模型的架构、预训练、指令训练、tokenizer 和消息模板都会一起变化；这些单任务诊断不能用于模型排名，也不能把四次空补丁统计成有四个独立任务支撑的失败率。有效模型比较和协议比较需要分别说明条件与分析范围。

### B 组可复核证据

- [B 组计划](../runs/cloud-v4-qwen3-20261022-01-protocol/plan.json)
- [B 组官方回填后的协议汇总](../runs/cloud-v4-qwen3-20261022-01-protocol/protocol-after-grading.json)
- [B-transcript calls](../runs/cloud-v4-qwen3-20261022-01-protocol/transcript/astropy__astropy-12907__recent__20261022T023954Z-2fd23776/calls.jsonl)
- [B-transcript trajectory](../runs/cloud-v4-qwen3-20261022-01-protocol/transcript/astropy__astropy-12907__recent__20261022T023954Z-2fd23776/trajectory.jsonl)
- [B-transcript result](../runs/cloud-v4-qwen3-20261022-01-protocol/transcript/astropy__astropy-12907__recent__20261022T023954Z-2fd23776/result.json)
- [B-chat calls](../runs/cloud-v4-qwen3-20261022-01-protocol/chat/astropy__astropy-12907__recent__20261022T024159Z-1bd2abed/calls.jsonl)
- [B-chat trajectory](../runs/cloud-v4-qwen3-20261022-01-protocol/chat/astropy__astropy-12907__recent__20261022T024159Z-1bd2abed/trajectory.jsonl)
- [B-chat result](../runs/cloud-v4-qwen3-20261022-01-protocol/chat/astropy__astropy-12907__recent__20261022T024159Z-1bd2abed/result.json)
