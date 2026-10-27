# 腾讯朱雀 A.I.G：源码研读与本项目安全执行边界

研读时间：2026-10-27 12:43（Asia/Shanghai）。官方仓库为 [Tencent/AI-Infra-Guard](https://github.com/Tencent/AI-Infra-Guard)，本轮固定源码提交 `33768c58942dbef693dae5e929d633daa0568b34`，提交时间 `2026-10-27T10:39:03+08:00`。以下判断基于源码，不以 README 的能力描述替代实际扫描结果。

上游为 Apache-2.0，并含 [NOTICE](https://github.com/Tencent/AI-Infra-Guard/blob/33768c58942dbef693dae5e929d633daa0568b34/NOTICE) 的归属说明要求。本项目本轮参考 Tencent Zhuque Lab AI-Infra-Guard 的威胁分类、攻击面分级和证据要求，独立实现工具权限边界；没有复制其扫描引擎、没有安装或运行其扫描服务。未来真正分发或集成上游组件时，应保留 LICENSE/NOTICE 并明确归属。

## 它具体做什么

| 源码/文档 | 读到的实际机制 | 对我们的启发与限制 |
|---|---|---|
| [agent-scan/core/agent.py](https://github.com/Tencent/AI-Infra-Guard/blob/33768c58942dbef693dae5e929d633daa0568b34/agent-scan/agent_scan/core/agent.py) 的 `Agent.scan` / `ScanPipeline.run_parallel_detection` | 信息收集、按检测技能并行运行、漏洞审阅，最后生成结构化报告 | 适合上线前红队评测；不是每个工具调用前自动执行的授权器 |
| 同文件 `_merge_worker_outcomes` | 失败 worker 被日志记录并跳过，无提取结果时返回 `No vulnerabilities confirmed.` | 不能把无发现字符串当“安全通过”；完整性、失败 worker、未覆盖场景必须单独记账 |
| [tools/dialogue/dialogue.py](https://github.com/Tencent/AI-Infra-Guard/blob/33768c58942dbef693dae5e929d633daa0568b34/agent-scan/agent_scan/tools/dialogue/dialogue.py) | 通过 provider 向被测 Agent 发真实对话 | 运行它需要可用目标 Agent 与模型调用，不能用本轮 CPU 假策略测试冒充 AIG 扫描 |
| [tools/dispatcher.py](https://github.com/Tencent/AI-Infra-Guard/blob/33768c58942dbef693dae5e929d633daa0568b34/agent-scan/agent_scan/tools/dispatcher.py) 的 `ToolDispatcher.call_tool` | 按已注册函数名分派检测工具并格式化结果 | 它是扫描 Agent 的调度，不是我们 Coding Agent 的操作系统沙箱 |
| [tool-abuse-detection](https://github.com/Tencent/AI-Infra-Guard/blob/33768c58942dbef693dae5e929d633daa0568b34/agent-scan/agent_scan/prompt/skills/tool-abuse-detection/SKILL.md)、[indirect-injection-detection](https://github.com/Tencent/AI-Infra-Guard/blob/33768c58942dbef693dae5e929d633daa0568b34/agent-scan/agent_scan/prompt/skills/indirect-injection-detection/SKILL.md) | 根据文件、执行、网络能力选择攻击面；用含恶意指令的外部内容检查目标行为 | 本项目应测真实工具副作用与访问边界，而不仅检查模型是否输出某个字符串 |
| [agent_security_reviewer.md](https://github.com/Tencent/AI-Infra-Guard/blob/33768c58942dbef693dae5e929d633daa0568b34/agent-scan/agent_scan/prompt/system/agent_security_reviewer.md) | 要求漏洞具备实际 dialogue 证据，去重、归类，拒绝无证据发现 | 这些是审阅模型的提示要求，不是形式化安全保证；我们的验收仍需可重复的执行断言 |
| [Memory Heist 示例](https://github.com/Tencent/AI-Infra-Guard/blob/33768c58942dbef693dae5e929d633daa0568b34/agent-scan/testcase/case2/memory_heist_agent.py) | 外部页面指令与用户记忆混合，随后经任意 URL 导航外传信息 | 反思记忆不能提升原始工具输出的信任等级；密钥不得放进模型上下文或工具环境 |

`skill-scan` 是面向 Skill 源码的 LLM 审计与可疑判决复核，`mcp-scan` 针对 MCP 工具及服务，均不能替代本项目运行时授权。上游主 README 还说明其 Web 平台当前缺认证，不应暴露公网；本轮没有部署该服务。

## 本项目实际补上的边界

现有受控文件工具禁止越界路径、`.git` 与受保护文件修改，但自由 `run` 能执行任意 shell；即使终局 grader 拒绝修改测试的补丁，副作用也已经发生。CPU 临时 fixture 复现了：同一条只写临时 `tests/test_public.py` 的命令，在原有执行能力下生效并于终局被拒绝，在 restricted 模式下根本不启动进程。

新增 `execution_policy=restricted`，按工具能力授权，不用命令关键字黑名单：

- 自由 `run` 明确拒绝；未知外部工具也拒绝。
- `public_test` 不接收模型命令或参数，执行可信任务清单固定的 `Task.test_command`。清单为空时明确不可用，模型不能覆盖 argv。
- Local 假策略软件测试以 argv 执行，不经过 shell；Docker 的 SWE 环境仍经登录 shell 激活 testbed，但使用 `shlex.join` 引用固定 argv，模型不能注入附加参数。
- 文件读写继续使用已有路径与编辑权限检查，避免复制两套校验。越界、`.git`、符号链接和受保护编辑使用结构化 `PermissionError`；Local 返回拒绝标记，Docker 文件工具通过专用退出码 77 传递。统一授权入口将其记为 `allowed=false/rule=repository_boundary`，不会触发反思。错误文本保留，普通不存在文件或编辑锚点不匹配仍是一般工具失败，可触发反思；不按错误文本关键词推断权限拒绝。
- 每次外部工具执行附 `policy_decision={policy,allowed,rule}`；工具类别拒绝发生在执行入口，不启动工具进程。文件权限拒绝在受控文件工具中判定，Docker 会启动检查路径的 Python 进程，但不执行被拒绝的文件操作。runner 将其接入审计与验收。
- 默认 `sandbox` 保留已有实验协议；是否采用 restricted 是显式实验条件，需写入配置和对照条件。真实模型 restricted 必须在 Docker 环境中运行。

## 不应夸大的能力

`public_test` 会执行模型修改过的源码，源码仍可能调用系统命令、修改容器中的文件或试图影响测试框架。因此 restricted **只限制直接工具权限**，不是“禁止一切代码执行”，也不是防注入万能开关。真正的隔离仍来自 Docker 的无网络、资源限制、无新增权限、能力移除与不挂载密钥。容器内行为和生成补丁还需要独立审查；不能把通过本轮 CPU 测试写成已证明抵御所有越狱、供应链攻击或跨租户攻击。

当前公开数据和依赖必须预先准备，禁网执行会影响需要联网下载的项目。未配置公开测试的任务必须显式补充可信命令，不能让模型为自己定义评分规则。公开测试通过也不等于独立最终评分通过。

本轮不把上游攻击示例原样运行到真实账号或外部主机，不调用付费模型。测试使用临时文件、固定动作与 Docker 进程替身；验证的是授权和参数边界，不是模型面对提示注入的防御成功率。后续有独立预算时，可用固定的干净/含注入仓库对照，在同模型与总预算下测量攻击成功率、误拒率、修复成功率和额外开销，并报告每个未完成测试场景。
