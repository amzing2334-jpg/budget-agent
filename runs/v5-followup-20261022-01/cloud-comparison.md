# 本轮云端产物对照

读取目录：`C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01`

这是已返回产物的描述性清单。缺失字段保留为未知；单任务开发尝试不构成基准成功率。

| 组 | 注册标记 | Task | 模型 / revision | Harness |
| --- | --- | --- | --- | --- |
| adaptive-guidance | additional_not_preregistered | astropy__astropy-12907 | Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 / cdbee75f17c01a7cc42f958dc650907174af0554 | internal_budget_agent |
| adaptive-guidance | additional_not_preregistered | astropy__astropy-12907 | Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 / cdbee75f17c01a7cc42f958dc650907174af0554 | internal_budget_agent |
| adaptive-mask | additional_not_preregistered | astropy__astropy-12907 | Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 / cdbee75f17c01a7cc42f958dc650907174af0554 | internal_budget_agent |
| adaptive-mask | additional_not_preregistered | astropy__astropy-12907 | Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 / cdbee75f17c01a7cc42f958dc650907174af0554 | internal_budget_agent |
| mini-baseline | planned_layout | astropy__astropy-12907 | Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 / cdbee75f17c01a7cc42f958dc650907174af0554 | mini-swe-agent-textbased-budget-adapter |
| reserve-1 | planned_layout | astropy__astropy-12907 | Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 / cdbee75f17c01a7cc42f958dc650907174af0554 | internal_budget_agent |
| reserve-128 | planned_layout | astropy__astropy-12907 | Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 / cdbee75f17c01a7cc42f958dc650907174af0554 | internal_budget_agent |

| 组 | 输入 / 输出 / 合计 tokens | calls | 运行状态 | 评分状态 | eval.resolved | 测试执行 | Patch bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive-guidance | 31440 / 361 / 31801 | 13 | budget_exhausted | official_empty_patch | 未知 | false | 0 |
| adaptive-guidance | 31339 / 347 / 31686 | 13 | budget_exhausted | official_empty_patch | 未知 | false | 0 |
| adaptive-mask | 31339 / 347 / 31686 | 13 | budget_exhausted | official_empty_patch | 未知 | false | 0 |
| adaptive-mask | 31175 / 252 / 31427 | 8 | budget_exhausted | official_empty_patch | 未知 | false | 0 |
| mini-baseline | 25570 / 1230 / 26800 | 6 | budget_exhausted | official_empty_patch | 未知 | false | 0 |
| reserve-1 | 31175 / 252 / 31427 | 8 | budget_exhausted | official_empty_patch | 未知 | false | 0 |
| reserve-128 | 31175 / 252 / 31427 | 8 | budget_exhausted | official_empty_patch | 未知 | false | 0 |

内部诊断仅统计记录到的 JSON 工具动作；重复失败是在相同 revision 下重复同一失败动作。

| 组 | Invalid | 到输出上限的 invalid | 重复失败 | edit / write 成功 | run 成功 | diff 成功 |
| --- | --- | --- | --- | --- | --- | --- |
| adaptive-guidance | 0 | 0 | 0 | 0 / 0 | 0 | 0 |
| adaptive-guidance | 0 | 0 | 0 | 0 / 0 | 0 | 0 |
| adaptive-mask | 0 | 0 | 0 | 0 / 0 | 0 | 0 |
| adaptive-mask | 0 | 0 | 0 | 0 / 0 | 1 | 0 |
| reserve-1 | 0 | 0 | 0 | 0 / 0 | 1 | 0 |
| reserve-128 | 0 | 0 | 0 | 0 / 0 | 1 | 0 |

mini-SWE 单独读取官方 trajectory 与调用账本。它的 Bash 命令数不能与内部 edit/run/diff 次数直接比较；原生 api_calls 也不能替代预算账本。

- mini-baseline：原生退出 budget_exhausted；解析动作 6；格式错误消息 0；有返回码观察 6；非零返回码 0。

mini-SWE 的补丁来自显式 submission；内部 agent 导出工作区 diff。空补丁与未执行测试不等于已执行测试失败。additional_not_preregistered 表示额外/适应性尝试，不能并入原配对；同一预设组存在多份结果时全部标为 ambiguous_multiple_attempts。

证据：

- adaptive-guidance / 20261022T074654Z-43348b7b：[result.json](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-guidance\budget-aware\astropy__astropy-12907__mask__20261022T074654Z-43348b7b\result.json>) · [patch.diff](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-guidance\budget-aware\astropy__astropy-12907__mask__20261022T074654Z-43348b7b\patch.diff>) · [calls.jsonl](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-guidance\budget-aware\astropy__astropy-12907__mask__20261022T074654Z-43348b7b\calls.jsonl>) · [trajectory.jsonl](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-guidance\budget-aware\astropy__astropy-12907__mask__20261022T074654Z-43348b7b\trajectory.jsonl>)
- adaptive-guidance / 20261022T074532Z-2df7bd07：[result.json](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-guidance\direct\astropy__astropy-12907__mask__20261022T074532Z-2df7bd07\result.json>) · [patch.diff](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-guidance\direct\astropy__astropy-12907__mask__20261022T074532Z-2df7bd07\patch.diff>) · [calls.jsonl](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-guidance\direct\astropy__astropy-12907__mask__20261022T074532Z-2df7bd07\calls.jsonl>) · [trajectory.jsonl](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-guidance\direct\astropy__astropy-12907__mask__20261022T074532Z-2df7bd07\trajectory.jsonl>)
- adaptive-mask / 20261022T072704Z-bbb60603：[result.json](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-mask\astropy__astropy-12907__mask__20261022T072704Z-bbb60603\result.json>) · [patch.diff](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-mask\astropy__astropy-12907__mask__20261022T072704Z-bbb60603\patch.diff>) · [calls.jsonl](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-mask\astropy__astropy-12907__mask__20261022T072704Z-bbb60603\calls.jsonl>) · [trajectory.jsonl](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-mask\astropy__astropy-12907__mask__20261022T072704Z-bbb60603\trajectory.jsonl>)
- adaptive-mask / 20261022T072506Z-86c192b2：[result.json](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-mask\astropy__astropy-12907__recent__20261022T072506Z-86c192b2\result.json>) · [patch.diff](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-mask\astropy__astropy-12907__recent__20261022T072506Z-86c192b2\patch.diff>) · [calls.jsonl](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-mask\astropy__astropy-12907__recent__20261022T072506Z-86c192b2\calls.jsonl>) · [trajectory.jsonl](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\adaptive-mask\astropy__astropy-12907__recent__20261022T072506Z-86c192b2\trajectory.jsonl>)
- mini-baseline / astropy__astropy-12907__mini-swe-textbased：[result.json](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\mini-baseline\astropy__astropy-12907__mini-swe-textbased\result.json>) · [patch.diff](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\mini-baseline\astropy__astropy-12907__mini-swe-textbased\patch.diff>) · [calls.jsonl](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\mini-baseline\astropy__astropy-12907__mini-swe-textbased\calls.jsonl>) · [mini.traj.json](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\mini-baseline\astropy__astropy-12907__mini-swe-textbased\mini.traj.json>)
- reserve-1 / 20261022T071042Z-2c771740：[result.json](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\reserve-probe\reserve-1\astropy__astropy-12907__recent__20261022T071042Z-2c771740\result.json>) · [patch.diff](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\reserve-probe\reserve-1\astropy__astropy-12907__recent__20261022T071042Z-2c771740\patch.diff>) · [calls.jsonl](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\reserve-probe\reserve-1\astropy__astropy-12907__recent__20261022T071042Z-2c771740\calls.jsonl>) · [trajectory.jsonl](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\reserve-probe\reserve-1\astropy__astropy-12907__recent__20261022T071042Z-2c771740\trajectory.jsonl>)
- reserve-128 / 20261022T071302Z-adc44a6d：[result.json](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\reserve-probe\reserve-128\astropy__astropy-12907__recent__20261022T071302Z-adc44a6d\result.json>) · [patch.diff](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\reserve-probe\reserve-128\astropy__astropy-12907__recent__20261022T071302Z-adc44a6d\patch.diff>) · [calls.jsonl](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\reserve-probe\reserve-128\astropy__astropy-12907__recent__20261022T071302Z-adc44a6d\calls.jsonl>) · [trajectory.jsonl](<C:\找工作\budget-agent\runs\cloud-v5-a4000-20261022-01\reserve-probe\reserve-128\astropy__astropy-12907__recent__20261022T071302Z-adc44a6d\trajectory.jsonl>)
