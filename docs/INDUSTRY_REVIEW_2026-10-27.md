# 2026-10-27 官方需求复核

观察日期为北京时间 2026-10-27。以下是本轮实际复核的有限来源，不是整个招聘市场的穷尽扫描；页面存在不等于具体 HC 已确认。

## 招聘要求是否变化

| 来源 | 来源日期与访问方式 | 与上一轮比较 |
|---|---|---|
| [百度 Coding Agent 策略算法实习生 J96208](https://talent.baidu.com/jobs/detail/INTERN/a77dfe3a-0f3a-45b5-95dc-c7478f9d43e4) | 2026-07-21；本轮直接页为导航壳，官方域名搜索索引正文可读 | 上下文策略、错误恢复、成本、效果和消融要求与昨日一致；不算十月新开岗位 |
| [百度代码智能体大模型架构实习生 J101404](https://talent.baidu.com/jobs/detail/INTERN/397771ff-709c-42f5-9fd7-2e4112e0768d) | 2026-07-21；官方正文可读 | 仍强调可复现沙箱、轨迹质量、后训练与 rollout，未确认新 HC |
| [字节 Seed 招聘项目](https://seed.bytedance.com/zh/seedearlycareer) | 页面未见发布日期；官方正文可读 | 仍包含 Code Agent RL、长程任务、Memory 和机器学习系统；用户毕业年份符合该页实习范围，不等于具体录用或名额 |

在这些来源中，没有发现足以改变项目主线的实质要求变化。上下文的技术价值仍应通过可靠实验和成本/修复效果来证明，不能仅凭新增记忆模块描述能力。

## 工程与研究依据

- [Anthropic：Effective context engineering for AI agents](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents)，2025-09-29。强调选择高信号信息、按需获取源码和引用。对本轮的约束是：有界保留、保留来源，同时测量输入代价，不预设保存越多越好。
- [The Complexity Trap，arXiv v3](https://arxiv.org/abs/2508.21433v3)，首发 2025-08-29，v3 为 **2025-10-27**，不是本轮当天的新论文。该工作在其 SWE-agent/SWE-bench Verified 设置中发现简单 observation masking 可以与模型摘要竞争，且有成本优势。本项目需要保留简单对照，不能直接借用论文结果为自己的新策略背书。
- [Anthropic：Quantifying infrastructure noise in agentic coding evals](https://www.anthropic.com/research/infrastructure-noise)，2026-02-05。基础设施配置可能改变评测结果；后续对照除了 token 和模型，还应固定 CPU/RAM、超时和并发条件。

## 本轮落实

本轮从真实轨迹中的隐藏后重读出发，实现 `mask-read` 候选并测量同历史前缀下的可见性与额外输入成本。候选只恢复原始完整源码行，没有引入向量数据库、额外摘要模型或多 Agent 编排。实现和验收见 [READ_MEMORY_VALIDATION.md](READ_MEMORY_VALIDATION.md)。

CPU 回放不能说明模型会停止重读、生成补丁或通过独立测试。后续进入真实实验前，仍需可用模型配置和相应费用授权；旧租约不存在，旧费用授权不自动重复使用。
