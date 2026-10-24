# v4：可靠交互与协议对照

日期：2026-10-22。本文保留本轮实施与云端执行计划。A、B 两组均已执行，四次尝试均为空补丁，实例已提前归还；实际结果与归还凭据见 [v4 验证](VALIDATION_V4.md)，失败原因见 [轨迹分析](V4_FAILURE_ANALYSIS.md)。下文的条件分支记录原计划，不表示它们全部已有正面结果。招聘核查见 [岗位调研](INDUSTRY_REQUIREMENTS_V4.md)，产品与工程资料见 [行业系统调研](INDUSTRY_SYSTEMS_V4.md)。

## 为什么选择这个方向

百度 J96208 的 Coding Agent 策略实习职责直接覆盖任务拆解、工具选择、上下文、多轮决策、错误恢复、成本和对照评测；J101404 则覆盖 sandbox、轨迹数据、模型后训练与推理系统。当前最适合的求职主线是 **Coding Agent 策略与执行系统 + 可信评测**。后训练作为后续深化方向，其起点应是可靠环境和有效轨迹。

大厂公开产品正在交付完整软件任务：从理解仓库和需求，到环境执行、局部修改、测试反馈、变更审阅、持续协作。我们的项目需要逐步成为这个闭环的一个可复现实现。界面、框架名称、Agent 数量和训练过程本身不能代替正确修复的证据。

现有项目已经有预算计量、容器执行、原始轨迹和官方评分基础，但还没有有效的真实任务修复。这次代码检查发现生产循环把历史模型动作与环境返回都编码为 user；v3B 有两次回复续写了 Action/Observation 事件格式。它是合理的候选原因，尚未得到因果验证。首次错误动作发生在没有历史的情况下，因此不能归因于历史角色。

新增公开研究也要求我们收窄贡献：2026 年 9 月的 [An Empirical Study of Harness Design for Coding Agents](https://arxiv.org/abs/2609.20804) 和 [Beyond the Model](https://arxiv.org/abs/2609.32459) 已研究上下文、harness 与模型/任务之间的关系。本轮只读取了两篇预印本的摘要，不能引用未读方法细节；它们足以说明“比较几个上下文策略”不是可以直接宣称的新研究。求职项目的价值应来自可信实现和明确问题，而不是包装新颖性。

## 已实施的工程范围

1. `history_format=chat|transcript`。默认 chat，保留旧 transcript 对照。原始模型回复进入 Event，包括非法 JSON；chat 使用原始 assistant 回复和独立 user 工具观察。这仍是文本工具协议，不声称原生 function calling。
2. 四种上下文策略按完整事件组处理。任务不裁，assistant 原文不裁，过长时只摘取 observation；无法容纳完整动作时显式停止。来源版本与 stale 信息继续保留，摘要调用继续计入预算。
3. 官方基础设施失败保留官方原值和原因，本地判定为空；空补丁从明确指定的官方汇总、原始 predictions 和 manifest 导入，显示 `official_empty_patch`、`tests_executed=false`，不伪造测试失败。
4. 一个专用协议诊断脚本：先存计划、同一模型仅运行两种格式各一次、拒绝复用目录与挑最好结果。汇总核验模型、源码、任务、环境和配置，并要求两组第一次模型输入完全相同。

本轮不修改 SYSTEM、工具反馈、模型采样参数，也不强迫无补丁的 Agent 继续运行。这样首次协议比较只改变历史表示及其实际 token 开销。摘要和 structured 的代码支持已完成，但不进入首轮协议实验。

## 云端实验 A：同模型协议诊断

**问题：**在同模型和预算下，chat 历史是否减少事件格式续写及重复失败，并产生可验证的修改？

| 项目 | 固定值 |
|---|---|
| 模型 | Qwen/Qwen2.5-Coder-7B-Instruct |
| revision | c03e6d358207e414f1eca0bb1891e29f1db0e242 |
| 任务 | astropy__astropy-12907，既有 dev，不是未见测试 |
| 策略 / 格式 | recent；transcript 与 chat 各一次 |
| 上下文 / 累计 token | 8192 / 32000，累计输入和输出都计费 |
| 单次最大输出 / 最大动作步 | 768 / 20 |
| 采样 | BF16、greedy、seed 42 |
| 工具执行 | 独立无网络 Docker 容器，沿用既有资源与超时 |
| 评分 | swebench 5.0.2，Agent 停止后独立评分 |

判读以整任务修复为最终结果；invalid_action、重复失败、成功编辑、diff/run 次数只是行为诊断。首次 prompt 应一致；如果不一致，汇总拒绝将它视为历史格式对照。

- chat 消除非法续写且完成修复：支持继续检验协议作用，但单题不能证明泛化。
- chat 减少格式错误但仍不能修复：协议有局部作用，模型能力/工具反馈仍可能受限。
- 两组都失败且轨迹近似：不要继续堆上下文机制；进入独立模型参考与任务难度检查。
- 环境、计量或评分失败：本轮没有可用的模型效果结论，先修复对应环节。

运行前准备缓存；驱动强制离线加载，缺缓存直接失败：

```bash
export HF_HOME=/home/featurize/work/budget-agent-assets/huggingface
python scripts/probe_history_format.py run \
  --config configs/cloud-qwen7b.toml --tasks tasks/swebench-pilot.jsonl \
  --task astropy__astropy-12907 --output runs/cloud-v4-protocol-dev-01
python scripts/probe_history_format.py summarize \
  --runs runs/cloud-v4-protocol-dev-01 \
  --output runs/cloud-v4-protocol-dev-01/protocol-before-grading.json
```

两种格式的子目录分别导出与评分，不能把同任务的两次尝试交给一次导出。官方评分后再次生成一个新名称的汇总。

```bash
# 默认只打印评分命令；确认评分环境与缓存后添加 --execute。
python scripts/grade_protocol_probe.py \
  --runs runs/cloud-v4-protocol-dev-01 \
  --grading-dir /home/featurize/benchmark-grading-v4 \
  --grader-python /home/featurize/swebench-env/bin/python \
  --run-id budget-agent-v4-protocol-01
```

评分脚本不启动 Agent 或模型；分别导出 predictions/manifest，调用独立评分环境，导入明确的官方汇总，然后生成报告。新的运行 ID 不得复用。

## 云端实验 B：独立模型参考

如果本轮时间与预算允许，再用 `configs/cloud-qwen3-4b.toml` 对同一个 dev 任务执行相同的两格式诊断，保存到独立目录。它是跨模型参考，不混入 A 的因果解释。

模型为 `Qwen/Qwen3-4B-Instruct-2507`，固定 revision `cdbee75f17c01a7cc42f958dc650907174af0554`，官方 non-thinking。2026-10-22 从 Hugging Face 官方 metadata 核验该 revision，原始 metadata 已保存。选择它因为可沿用当前加载接口与单卡资源，不是因为已证明它更强，也不称其为最新前沿模型。权重完整性核验后才启动；不把下载时间当推理耗时。

## 云端资源和停止条件

上一台 Featurize 实例已归还，用户确认没有运行实例。持久目录保存了旧模型、数据集缓存、镜像压缩包和依赖版本快照。本轮普通 4090 24GB 缺货，实际使用官方 PyTorch 2 镜像、单卡 4090 48GB、约 56GiB RAM；小时单价为 3 元。新实例为 Python 3.11.8、Docker 26.1.0、驱动 610.57.04；重新建立独立推理和评分环境，固定 PyTorch 2.8.0+cu128、Transformers 5.4.0 和 swebench 5.0.2。

用户已明确授权本轮按量租用：单价不超过 5 元/小时、最多 2 小时、总额不超过 10 元，不购买磁盘扩容或套餐。控制台记录 2026-10-22 北京时间 09:53 开始；设置了 11:43 的官方 CLI 释放定时器，工作负载截止为 11:33。实际在 10:46:25 提前调用官方 CLI 释放，并刷新控制台核验“未找到实例”；约 54 分钟、估算 2.7 元，未读取最终账单。本机只编辑、读取产物和进行 CPU 软件检查。

租用后立即记录开始时间并设置官方释放定时器，计划在第 110 分钟释放，预留最后 10 分钟处理异常。定时任务成功创建不等于实例已经释放；最后核验控制台无运行实例。模型进程在任务期限前停止；预算不足时跳过 B，保留 A 的全部结果。禁止为跑满计划自动续费或重新租一台。

恢复顺序：检查 GPU/磁盘/Docker → 安装固定推理和评分依赖到实例磁盘 → 校验持久模型与镜像 → 加载原镜像并确认 image ID → 恢复正负评分对照 → 运行 A → 独立评分与回传 → 有余量再运行 B → 保存产物并归还。评分数据保持在 Agent 项目目录之外；不把 gold patch 或隐藏测试放进 Agent 输入。

本轮镜像恢复先因 image ID 不同而停止，未启动模型尝试。原归档 SHA256 不变；逐字节核验其旧 index `a9be42bf…` → linux/amd64 manifest `87f608e6…` → config `a7cc2f99…` 的摘要链，且加载后 11 个 RootFS DiffID 与运行配置一致。因此 Docker 26 的严格预期 ID 校准为完整 config digest，而非忽略身份检查。初次失败日志及 `image-identity-mapping.json` 随回执保存。[Docker 官方文档](https://docs.docker.com/engine/storage/containerd/)说明新版存储支持镜像索引与 attestations，[OCI 配置规范](https://github.com/opencontainers/image-spec/blob/main/config.md)定义配置摘要和 DiffID。本轮内部对照共用同一恢复环境；不能据此声称 v3/v4 的宿主运行时完全一致。

## 完成本轮之后的路线

| 阶段 | 具体交付 | 决定是否继续的证据 |
|---|---|---|
| 真实修复基线 | 从问题定位、局部编辑到适配仓库的测试执行、补丁和验证说明 | 在事先确定的多个 dev 任务产生有效修复；逐个解释失败，不只展示成功样本 |
| 外部参考 | 固定 mini-swe-agent commit、模型与资源的独立运行 | 检查自有 harness 是否因接口不成熟而落后；未执行前不声称已复现 |
| 上下文研究 | recent 与 observation masking，再比较一种保留定位/编辑/测试证据的策略 | 至少任务确有上下文压力；全部摘要/检索计入预算，比较修复、重复工作和成本 |
| 确认实验 | 冻结实现、预算与未参与开发的多仓库任务集 | 样本与任务选择透明，报告不确定性；不能将反复调试的 Astropy 题作为未见成绩 |
| 可学习策略 | 从检查过的失败/恢复轨迹做 SFT 或上下文选择策略 | 与未训练、SFT、合理简单策略比较，确认提升不是额外 token 或数据泄漏 |
| Agentic RL | 独立 reward、有效 rollout、参数更新、留出评测 | 奖励和动作 token 对齐可靠，先取得非零且有差异的结果；现有无效 GRPO 不作初始化证据 |

前两阶段若仍全部失败，应优先借助能力更强的模型建立可运行参考，再决定自训目标。一个对所有任务都失败的底座不适合研究上下文收益；小模型非零成功也不等于已经达到产品交付能力。

用户到年底的学习重点依次是：能独立调试 Python/pytest/Git/Linux；读懂 Transformer 生成、tokenizer 与 KV cache；读懂完整 Agent 轨迹与测试框架；掌握 SFT、概率比、advantage 和 policy gradient 的含义；最后完整复现一个有可靠奖励的训练对照。面试中应能解释自己的失败、取舍与实验，而不是只背框架名称。
