# v5 验证记录：工具、预算和诊断已落实，真实模型效果待云端确认

> 历史记录（2026 年 10 月）：文中链接的 `runs/` 产物没有收进仓库，保存在作者的本地归档中。当前的结果和方法见 [技术复盘](TECH_REPORT.md)。

日期：2026-10-22。完成范围见 [v5 实施方案](IMPLEMENTATION_V5.md)。本轮在轻量本地 Python 环境中修改代码、执行 CPU 软件检查、回放历史记录。没有模型推理、训练、API 调用或新租卡费用。

## 完整 CPU 回归

执行命令：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

实际终端结果：`327 passed, 2 skipped in 135.30s (0:02:15)`。

两个跳过的模块是 `test_learning_policy.py` 和 `test_learning_trainer.py`，本机没有可选训练栈。HF 后端和外部 mini-SWE 适配测试使用显式替身，没有加载模型。Docker 文件工具的相关检查是在 CPU 子进程中执行实际内联函数源码，不是 Docker 容器启动测试。

新增检查覆盖：有范围与分页的目录/搜索、16,000 字符页边界、长行命中位置、317 行文件的连续读取、local/容器内联代码一致性、唯一编辑及歧义错误定位、三种预算约束下的动作额度下限、低于下限时没有请求/挂起账目、CLI 只读和拒绝覆盖、历史诊断、报告缺失指标与 harness 分组，以及 mini-SWE 固定版本与提交/计量契约。

初次 focused 检查发现一条新测试错误地假定旧 smoke 用 `edit_file`，实际旧 fixture 动作使用 `write_file`；已改为检查实际契约。环境测试造极长文件名时也曾触发 Windows 路径限制，改用更多较短名称验证同一分页边界。最终完整检查全部通过。

## 保存的执行产物

全部新产物位于 [runs/v5-polish-20261022-01](../runs/v5-polish-20261022-01)。

- [验证回执](../runs/v5-polish-20261022-01/validation-summary.json)：源码指纹、执行边界与产物位置。
- [常规 smoke 报告](../runs/v5-polish-20261022-01/smoke/report/report.html)：3 个 fixture × 3 个策略，9 次预设修复全部通过独立 fixture 评分。这些轨迹未触发策略间的有效差异，不说明模型能力或上下文收益。
- [编辑恢复契约报告](../runs/v5-polish-20261022-01/scripted-recovery/report/report.html)：预设 8 步，保留一次失败后完成正确编辑与独立评分。
- [恢复场景的全部预设动作](../runs/v5-polish-20261022-01/scripted-recovery-actions.json)：可检查并重放，不伪装为模型生成。

恢复场景先尝试替换出现两次的 `"-"`，观察到 `found 2` 和候选行列后读取代码，再唯一替换 `[^a-z]+`。失败时源码 revision 未变，实际成功编辑后 revision 改变。诊断记录成功 read/edit/run/diff 各一次、非法动作零次。独立评分通过，`evidence_kind=scripted_smoke`。

这证明新工具、循环、失败保存、补丁和评分能够连通，**不证明真实模型能自行选择这些动作**。

## v4 轨迹只读回放

对四条原始轨迹运行新的 `diagnose` 命令，生成两个新文件：

- [Qwen2.5-Coder-7B 诊断](../runs/v5-polish-20261022-01/v4-qwen25-diagnostics.json)
- [Qwen3-4B 诊断](../runs/v5-polish-20261022-01/v4-qwen3-diagnostics.json)

| 历史轨迹 | 非法动作 | 达到输出额度的非法动作 | 重复失败动作 | 相同成功观察重复 | 成功读取 | 成功编辑/命令/diff |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5 transcript | 2 | 0 | 3 | 0 | 1 | 均为 0 |
| Qwen2.5 chat | 1 | 1 | 7 | 0 | 1 | 均为 0 |
| Qwen3 transcript | 4 | 3 | 0 | 4 | 0 | 均为 0 |
| Qwen3 chat | 2 | 1 | 0 | 5 | 0 | 均为 0 |

四条调用与事件原文均对齐。Qwen3 transcript 的第 4、7 步同时出现多 JSON 值与输出额度耗尽，二者分开标记，没有强行选一个“根因”。所有记录仍是同一道 Astropy 开发题，不能代表四个独立任务。

回放前后检查四条运行的 `result.json / trajectory.jsonl / calls.jsonl / patch.diff`，共 16 个文件 SHA-256 完全一致。详细哈希见[历史文件完整性记录](../runs/v5-polish-20261022-01/historical-artifact-sha256.json)。没有把诊断字段回写进旧结果，也没有改写空补丁状态。

## 外部基线验证的边界

mini-SWE-agent 的 CLI `--help` 与默认计划入口正常，计划模式没有建立输出运行目录。适配器的 13 项 CPU 契约检查通过，固定 commit 的真实源码接口另经核对。

尚未在本轮安装 mini-SWE 上游依赖、启动真实上游循环、加载真实模型或在 Docker 中执行这条基线。因此目前交付的是可运行入口和已检验的接口契约，**外部基线分数仍为空**。执行时固定上游 SHA、检查导入来源，失败应保留明确日志，不能用替身测试补写成绩。

## 当前可以和不可以声称什么

可以声称：针对既有失败证据，完成了有界工具、编辑定位反馈、动作预算门槛、离线行为诊断及外部基线接入，并通过 CPU 回归与脚本化契约验证。

不能声称：真实 SWE-bench 修复率提高、结构化记忆胜过简单基线、128-token 下限是最优策略、mini-SWE 已在本环境成功修复、已经完成有效 Agentic RL。128 下限也可能拦住本可完成的短动作，必须用真实对照评估收益和损失。

下一轮计划为同一开发题的 v5 下限 1 / 128 与 mini-SWE 外部参考三次预定尝试；模型、预算与隔离条件见 [IMPLEMENTATION_V5.md](IMPLEMENTATION_V5.md)。此前云实例已归还，新增云端运行需新的租用预算授权。
