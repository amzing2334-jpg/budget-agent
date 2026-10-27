# Reflexion 源码研读与受预算约束的适配

研读日期：2026-10-27（Asia/Shanghai）。官方论文为 [arXiv:2303.11366v4](https://arxiv.org/abs/2303.11366v4)，初版发布于2023-03-20，v4更新于2023-10-10。官方代码固定为 [noahshinn/reflexion@218cf0e](https://github.com/noahshinn/reflexion/tree/218cf0ef1df84b05ce379dd4a8e47f17766733a0)，该提交时间为2025-01-13T23:53:43-08:00；不是2026年刚发布的方法。仓库许可证为 [MIT](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/LICENSE)，版权标注 Noah Shinn, 2023。

本轮在项目目录之外浅克隆官方仓库，读取指定源码，没有安装或执行上游代码，也没有调用模型。以下是实现审查，不是论文结果复现。

## 实际读到的实现

| 文件/入口 | 实现事实与启发 |
|---|---|
| [programming_runs/reflexion.py / run_reflexion](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/programming_runs/reflexion.py) | 生成内部测试或读取LeetCode可见测试；先生成函数，执行后得到反馈；失败后调用 self_reflection，再把上一实现、反馈和反思交给 func_impl。内部通过或迭代结束时调用另一条 evaluate 路径。 |
| [generators/generator_utils.py](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/programming_runs/generators/generator_utils.py) | generic_generate_self_reflection 只生成说明；generic_generate_func_impl 将旧实现、反馈与反思编排成多条消息；generic_generate_internal_tests 做生成、语法筛选和抽样。角色和数据流可直接检查。 |
| [generators/py_generate.py / PyGenerator](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/programming_runs/generators/py_generate.py) | 反思被要求解释失败原因，下一步仍由代码生成器改代码；测试筛选检查AST语法，不保证测试语义正确。 |
| [executors/py_executor.py / execute、evaluate](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/programming_runs/executors/py_executor.py) | execute 返回内部测试的通过项、失败项与输出；evaluate 使用正式测试。实现直接调用宿主 exec/globals，并吞掉宽泛异常，不能当作企业沙箱复用。空内部测试列表也会保留初始 is_passing=True。 |
| [generators/model.py / gpt_chat、gpt_completion](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/programming_runs/generators/model.py) | 旧OpenAI接口带最多6次隐式重试；主循环按 max_iters/pass_at_k 限制，没有本项目要求的统一输入、输出与所有辅助调用总账。直接移植会破坏费用和失败合同。 |
| [immediate_refinement.py](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/programming_runs/immediate_refinement.py)、[immediate_reflexion.py](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/programming_runs/immediate_reflexion.py) | 分别提供没有生成式反思、没有测试反馈的消融路径。它们提示我们必须区分“多花计算”和“反思文本”的贡献。 |
| [hotpotqa_runs/agents.py](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/hotpotqa_runs/agents.py)、[alfworld_runs/generate_reflections.py](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/alfworld_runs/generate_reflections.py) | 推理任务区分无记忆、上一轨迹、反思、两者组合；ALFWorld把失败环境的反思写入后续trial记忆，并选最近3条构造反思提示。与单次修复中的工具失败反馈不是同一实验设置。 |

一个容易漏掉的版本差异：论文编程章节描述最多6条生成测试；本次固定源码的 run_reflexion 调用 internal_tests(..., 1)。不能照搬论文设置来描述这个快照。正式测试结果在编程主循环中影响停止/后续pass选择；如果扩大pass_at_k，必须明确它是多次尝试，不能在本项目中把重复查看隐藏评分包装为一次独立测试。

## 对项目的技术判断

Reflexion更新的是上下文中的经验文本，不更新模型参数。称它为语言反馈式优化可以；它不等于SFT、PPO或GRPO，也不能证明本项目已做参数级Agentic RL。论文自述的限制包含依赖自评能力和反馈质量；MBPP Python未显示与其他任务相同的收益，因此反思不是普遍有效的调优按钮。[论文方法、编程分析与局限](https://arxiv.org/html/2303.11366v4)

本项目适合先研究一个窄变体：**同一次修复尝试内，选定工具失败后生成一次短反思，让下一次actor决策使用它。** 它保留失败事实、假设与下一项可验证动作，不重置工作区、不增加免费重试，不跨任务搬运记忆。该实现应叫 Reflexion-inspired bounded feedback，而非原论文完整复现。

反思只使用actor实际收到的上下文与刚完成的失败事件。公开测试或一般命令失败不等于隐藏测试失败，也不自动证明源码有bug；缺依赖、错误路径、协议解析和编辑锚点错误需要不同诊断。独立私有verifier只能在最终提交后执行，其输出不得送进反思、记忆或下一次修复。

`reflection.py` 是无I/O、无模型、无工具执行的纯模块：

- `reflection_trigger(event, blocked=False)` 只选择失败的 run、public_test、edit_file、invalid_action；安全策略拒绝必须明确排除，不能让反思学习绕过策略。
- `reflection_messages(actor_messages, event)` 更换诊断专用system指令，保留已可见的user/assistant内容与当前事件；不读归档、Task.metadata或隐藏评分，不暗中截断输入。
- `ReflectionNote` 保存 schema_version、source_event_id、source_revision、text；作为user数据返回，并标记当前revision与stale。文字仍是模型假设，不能升级为系统规则或独立证据。

次数、输出上限和费用由runner控制。反思必须经过现有 `call(..., purpose="reflection")`，与actor/summary共享总账；后续输入中的反思也要再次按输入计费。插入记忆前，context构建应预留其实际token开销；无法容纳或用量不明时，不另开隐形账本。`off` 必须保持既有调用序列与行为。

上述角色与标签只规定数据流，**不证明模型能够抵御提示注入**。真正执行权限仍由独立工具策略和隔离环境约束。反思产生的shell文本或JSON工具动作也只作为字符串，不执行。

## 验收与后续消融

纯模块CPU测试覆盖20种情况：选定失败触发、成功及不相关动作不触发、安全拒绝不触发、实际可见上下文不变、来源保留、原始输入不被修改、外部指令不提升至system、revision变化标stale，以及大观察不被静默裁剪。这些测试证明软件合同，不证明反思质量或修复收益。

runner集成还须验证：默认关闭无额外调用；反思次数封顶；反思与actor用同一预算；反思费用未知则停止；隐藏verifier始终只终局执行且未泄露；反思提示或记忆太大时留下明确跳过/预算停止记录；反思响应永不作为工具动作执行。

真实收益至少比较三个固定条件：同模型和快照、同任务清单、同总token/时限、同工具和环境下的现有actor、仅显式重呈失败证据、生成式短反思。冻结其他上下文策略和loop_policy；报告全部尝试、独立修复率、有效评分数、总输入/输出、反思占比、修复前后退化、超时与缺依赖情况。除总预算对齐外还需公开各条件实际消耗，不能把更多调用的收益直接归因反思。

本轮不生成新的付费调用；既有已授权参考试跑的默认配置继续关闭反思。先取得有效真实修复参考，再另行制定这个变体的云端对照方案。跨trial记忆、自动生成测试与参数训练均不在这份模块实现范围内。
