# 真实修复参考试跑：待授权方案

检索时间：2026-10-27 06:53–07:04（Asia/Shanghai）。本文件是准备完成的方案，不是已运行实验。当前没有新费用授权、已确认可用服务器或已验证的模型服务连接。

## 为什么下一步需要真实参考

截至上一轮，已有小模型真实开发尝试均为空补丁。源码保留的 CPU 回放只能证明可见性变化，不能区分“模型/协议还不能完成修复”和“上下文策略不好”。本轮先准备一个当前官方可查、固定快照的通用模型参考；不把换模型或增加总预算后的结果归因于 mask-read。

本轮官方岗位复核未发现足以改变主线的实质变化，也没有确认新增 HC：

| 来源 | 发布日期与本轮访问边界 | 对项目的要求 |
|---|---|---|
| [百度策略实习 J96208](https://talent.baidu.com/jobs/detail/INTERN/a77dfe3a-0f3a-45b5-95dc-c7478f9d43e4) | 2026-07-21；直达页需登录，官方域搜索索引正文可读 | 真实行为分析、错误恢复、上下文、成本和消融 |
| [百度架构实习 J101404](https://talent.baidu.com/jobs/detail/INTERN/397771ff-709c-42f5-9fd7-2e4112e0768d) | 2026-07-21；官方正文可读 | 可复现沙箱、轨迹质量、后训练和推理性能 |
| [Seed 官方招聘](https://seed.bytedance.com/zh/seedearlycareer) | 未见发布日期；官方正文可读 | Code Agent RL、长程任务、Memory、机器学习系统；页面存在不等于具体 HC |

[Anthropic 的评测说明](https://www.anthropic.com/research/demystifying-evals-for-ai-agents)（2026-01-09）强调尝试、轨迹与最终环境结果的区分。[基础设施噪声分析](https://www.anthropic.com/research/infrastructure-noise)（2026-02-05）说明资源限额会影响结果。[SWE-bench 官方评测指南](https://www.swebench.com/SWE-bench/guides/evaluation/)（未见发布日期）要求把补丁应用到仓库后执行测试，并说明空补丁、未评分和真正未解决的区别。本项目沿用已经固定的 harness 版本，不因文档更新而静默升级。

## 模型与协议准备

配置为 [reference-qwen38-max.toml](../configs/reference-qwen38-max.toml)：`qwen3.8-max-0902`、北京区、非流式文本 Chat Completions、`enable_thinking=false`、`max_completion_tokens`、temperature 0。固定 recent / chat / direct / loop_policy=off。不开摘要调用，不在首轮混入新上下文策略。

- [官方上架记录](https://help.aliyun.com/zh/model-studio/newly-released-models)确认该模型为 **2026-09-02** 快照。它是参考候选，不是本项目已证实的强基线。
- [当前 Coder 文档](https://help.aliyun.com/zh/model-studio/qwen-coder)已推荐新的通用模型。旧 Coder Plus 搜索缓存报价不能视为当前报价；本轮没有据此断言其下架。
- [官方兼容接口](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)支持本次选择的关闭思考、完整输出上限与文本接口。直接 HTTP 的 `enable_thinking` 位于请求顶层，非 SDK `extra_body`。旧北京域名仍可使用；业务空间专属域名可在实际执行前显式替换并重新冻结。
- 适配器新增可选布尔字段 `api_enable_thinking`；未配置时不发送，false/true 必须显式指定。这不是支持所有思考模型或保存思维链的承诺。API 类型、请求字段和用量对账已通过 CPU 假 HTTP 服务验证，实际账号权限、响应与可用性尚未验证。
- 使用 `byte` 作为保守输入估计，它不是该模型 tokenizer。真实 usage 才进入账本，模型返回名称必须记录；超出预留或无法核对会停止。官方注明输出上限可能有最多 10 token 偏差，本项目不会悄悄豁免超额以制造有效预算结果。

## 两阶段与验收门槛

| 阶段 | 任务 / 尝试 | 输入＋输出总额度 | 上下文 / 单次输出 / 步数 | 进入下一阶段条件 |
|---|---|---:|---|---|
| 协议修复 | slugify-digits，1 次 | 8,000 | 8,192 / 1,024 / 8 | 有有效动作、非空补丁、独立自建校验通过，账本已核对且无超额 |
| 真实参考 | astropy__astropy-12907，1 次 | 128,000 | 32,768 / 2,048 / 30 | 保存完整尝试和补丁，执行官方独立评分；成功与失败均报告 |

第一阶段失败就停止并分析，不自动切模型、重试或继续第二阶段。第二阶段属于反复使用过的开发题，不能成为独立最终测试；即使成功也只是参考链路验收。若成功，下一轮再预先冻结其他开发任务和多策略对照，不从结果中挑选任务。

以下命令只在**另行授权后的云端**执行，输出目录均须新建。阶段门槛由执行者审核现有原始记录，计划 JSON 不会自动启动或审批这两条命令：

```bash
python -m budget_agent run --config configs/reference-qwen38-max.toml --tasks tasks/local.jsonl --task slugify-digits --context-tokens 8192 --total-tokens 8000 --max-output-tokens 1024 --max-steps 8 --output runs/reference-protocol-NEW
# 仅在协议修复通过后执行：
python -m budget_agent run --config configs/reference-qwen38-max.toml --tasks tasks/swebench-pilot.jsonl --task astropy__astropy-12907 --output runs/reference-swe-NEW
python -m budget_agent swe-export --input runs/reference-swe-NEW --output runs/reference-swe-NEW/predictions.jsonl
# 先打印评分计划；核对独立评分环境后再显式追加 --execute：
python -m budget_agent swe-evaluate --predictions runs/reference-swe-NEW/predictions.jsonl --run-id reference-swe-NEW --max-workers 1 --instance-ids astropy__astropy-12907
```

评分完成后按现有 swe-attach 合同核验 patch.diff；空补丁需要 aggregate summary 和 export manifest，不能报成测试通过。保存所有原始用量、失败、模型名称与镜像实际 ID，不覆盖旧实验。新主机先运行官方正确/错误补丁评分对照；标准补丁与隐藏测试不得进入 Agent 工作区。

## 资源、费用与待满足条件

**不需要 GPU。** 模型在服务商侧推理，远端 Linux x86_64 主机只执行 Docker 沙箱和独立评分。计划主机规格为 8 vCPU、32 GB RAM、至少 100 GB 可用盘，评分并发 1；这是一项拟租配置，不是已核验的在售实例。Agent 容器仍使用当前代码的 2 CPU / 4 GB / 256 PID / 无网络约束，评分资源与镜像实际 ID另行记录并冻结。环境恢复、拉镜像和评分均计入租用时间；预检失败或时间不足时不进入模型阶段，不自动延时。

[官方北京价格](https://help.aliyun.com/zh/model-studio/model-pricing)在本轮检索时列出该快照：输入 **12 元/百万 token**，输出 **36 元/百万 token**，0–1M 输入区间。忽略优惠和缓存，按较高输出价对全部规划 token 估计：

`(8,000 + 128,000) × 36 / 1,000,000 = 4.896 元`。

建议申请 **API ≤5 元；云主机 ≤5 元/小时、最多 1 小时；合计 ≤10 元**。机器部分是采购上限，尚无已确认库存/报价；如找不到符合规格与价格的主机，不购买替代套餐。该计算是使用量不超额时的规划估计，不是供应商硬账单封顶；实际计费以服务商账单为准，有可用平台硬额度时同时配置。未知计费或实际超额会中止后续尝试，不自动重发。

代码原有费用字段以 USD 计价，本配置不把人民币 12/36 填入美元字段，不引入未经核实的汇率。实际报告的 USD 保持 unknown；执行后另按官方账单与人民币报价复核费用。

执行需要：本轮新的费用授权；已确认可运行 Docker 的云主机；云端 `DASHSCOPE_API_KEY` 环境变量以及对应北京模型权限。密钥不发到聊天或 GitHub。本轮只准备配置、计划哈希和 CPU 验收，未租机器、调用真实模型或训练。
