# 可见源码重读控制与独立评测

日期：2026-10-26。此轮为软件机制与历史轨迹审计，未启动真实模型、训练或新租用实例。

## 行为契约

`loop_policy` 默认为 `off`，保留原有请求和执行路径。设置为 `recover` 后：

1. 一次成功 `read_file` 的真实输出必须带有可解析的 canonical path、连续行号和完整行内容；截断、空读、未知旧格式和失败读取不参与判定。
2. 控制器检查产生该动作的**实际模型请求**。只有同一连续工作区 revision 内、同一路径的所有返回行，以相同文本完整出现在先前事件的可见观察中，才记一次重复。多个来源事件可以覆盖同一读取范围；部分新行不算完全重复。
3. 第一次重复后，下一次动作调用附带一条反馈，给出事件/路径/行范围/来源，不包含任务答案。没有额外模型调用。模型仍可使用全部工具。
4. 收到该带反馈调用的响应才记录 `warning_sent`。若预算不足、模型请求失败或结束前没有机会发送反馈，不把待发送反馈算作已送达。
5. 若反馈后的下一个动作仍属于连续完全重复读取，状态变为 `stalled`，保存当前补丁并执行原有独立评分。其他动作或 revision 变化结束这个重复段；这不表示已经取得修复进展。每个重复段只发送一次反馈。

代码 A→B→A 回退会清除先前段的计数。被 mask 隐藏、被裁剪掉或只存在摘要中的源码不视为当前完整可见。`run`、`search`、`recall` 和有效轮询不在这个窄判据里。控制器既不是通用死循环检测器，也不能判断模型是否理解了代码。

原始 `trajectory.jsonl` 仍记录原始工具输出，控制事件单独保存于 `controls.jsonl` 和 `result.json.loop_control`。`calls.jsonl` 保留实际输入、反馈和用量。`stalled` 只是停止原因，独立评分仍决定补丁是否正确。

## 运行方式

下列真实模型命令只作为云端执行方案，不在用户电脑上执行：

```bash
python -m budget_agent run --config configs/cloud-v5-qwen3-mask.toml --tasks tasks/swebench-pilot.jsonl --task astropy__astropy-12907 --loop-policy off --output runs/new-loop-pair
python -m budget_agent run --config configs/cloud-v5-qwen3-mask.toml --tasks tasks/swebench-pilot.jsonl --task astropy__astropy-12907 --loop-policy recover --output runs/new-loop-pair
# 分别导出/独立评分两种设置后再比较；若混合在同一根，导出须指向各自运行目录。
python -m budget_agent.paired --runs runs/new-loop-pair --factor loop_policy --baseline off --candidate recover --output runs/new-loop-pair/paired.json
```

该模型仅为保留的旧诊断配置，不代表推荐继续消耗云预算重复旧题。两种设置必须用**相同的新源码**运行，不能把旧版本 `off` 结果与新代码混成消融。比较器拒绝任务缺失、重复尝试、源码/模型/预算/其他策略差异；未独立评分的记录留在清单，不构成胜负。

本轮可直接复现的 CPU 软件场景：

```bash
python scripts/validate_read_loop.py --output runs/new-read-loop-software-check
python -m budget_agent diagnose --input runs/cloud-v5-a4000-20261022-01 --output runs/new-historical-read-audit.json
python -m pytest -q
```

使用新的输出路径，避免把复跑混入原样本。软件场景在同一个小型 fixture 上预先写好动作：持续重读，以及重读后执行已知修复。**后者不证明模型收到反馈就会修复。** 每个场景比较 off/recover，均使用独立评分并保留请求和控制记录。

本轮完整 CPU 回归 **436 项通过、8 项跳过**，无失败或错误；跳过的 2 个可选训练模块和 6 项未配置固定 mini-SWE 上游的集成检查不能算作已经验证，具体原因及源码指纹见 [CPU 回执](../runs/iteration-20261026-01/cpu-validation.json)。[软件场景回执](../runs/iteration-20261026-01/scripted-final/receipt.json)、[单因素配对](../runs/iteration-20261026-01/scripted-final/paired.json)、[离线报告](../runs/iteration-20261026-01/scripted-final/report/report.html) 同步保存。

| 预设动作场景 | off | recover | 可得出的软件结论 |
|---|---|---|---|
| 持续读取相同源码 | 7 次调用，未修复 | 3 次调用，反馈 1 次后 stalled，未修复 | 控制器可在该预设序列中停止继续调用，不证明真实模型节省率 |
| 重读后执行预设修复 | 6 次调用，独立评分通过 | 6 次调用，反馈 1 次，独立评分通过 | 机制没有阻止后续合法编辑与验证；两边都成功，不是候选修复率提升 |

后一场景中 recover 比 off 多消耗 550 个 byte-estimate token，说明反馈有实际计账开销，不能把干预描述为免费。byte-estimate 不是模型 tokenizer 或真实 API 收费 token。

## 历史轨迹审计

原始 7 条 v5 尝试均保留，新的诊断另写新目录。

[历史审计回执](../runs/iteration-20261026-01/historical-read-audit.json) 记录原始结果、请求和轨迹 SHA256，以及实际命中的事件和来源。

| 历史设置 | 轨迹数 | 实际请求内完全可见的重复读取 | 解释 |
|---|---:|---:|---|
| `recent`：动作预留 1/128 及 recent/mask 对照的 recent | 3 | 各 1 次，共 3 次 | 均为 e0008，读取 304–317 行，来源 e0004；都在最后一个工具步骤，没有后续真实反馈响应 |
| `mask`：上下文对照及 direct/budget-aware | 3 | 0 | 历史出现重复覆盖，但不能据此断言读取内容仍在当次输入里 |
| 外部 mini-SWE | 1 | 0 | 其 shell 轨迹不属于本控制器的 `read_file` 输出协议，0 不表示不存在重复 |

没有把回放写成新策略的真实实验，没有计算虚构的节省率。这个结果将下一轮重点导向上下文可见性和参考模型能力，而不是放宽检测条件以制造更多触发。

## 能支持与不能支持的结论

- 支持：判据可复现；反馈计入预算；正常恢复、版本变更和截断不被猜作完整重复；停止后保留独立评分；单因素比较拒绝混杂条件。
- 支持：报告和配对结果不把中断、未评分、明确未执行测试或评分互相矛盾的旧成功标记变成可信成绩。
- 不支持：真实模型更容易修复、SWE-bench 成绩提高、真实 API 或 GPU 成本下降、长程任务能力或 RL 收益。

只有在固定多个开发任务、模型、资源预算和环境的新真实对照中，才能检验收益、误报与代价。软件调用次数减少本身不构成求职简历上的模型能力提升。
