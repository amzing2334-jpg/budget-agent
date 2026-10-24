# 可训练 Coding Agent（v2）

v2 在 v1 执行与评分环境上增加开源模型、LoRA、SFT 和真实的组相对策略梯度更新。商业模型 API 仍可用于 v1 推理对照；v2 训练不需要 API 密钥。

**执行约束（2026-10-21）：用户明确要求不使用本机训练。后续 SFT、RL 和 GPU 验证只在用户提供的云端机器执行。本文操作示例统一用于云端 Linux；本地电脑只进行代码编辑、材料检查和允许的 CPU 软件测试。**

## 实现范围

训练模型为固定版本 `Qwen/Qwen2.5-Coder-0.5B-Instruct`，仅更新 LoRA 参数。代码针对支持 `logits_to_keep` 的 Transformers 因果模型实现，已验证 Qwen2；并非任意模型后端的兼容层。

```text
训练任务 → 同一策略独立采样 G 条轨迹 → 各自的 Docker 工作区
       → read / write / run / finish → 独立私有测试给出 0/1 奖励
       → 组内相对 advantage → 按每一步真实上下文重算动作概率
       → 仅对动作 token 反向传播 → LoRA optimizer.step → 保存 adapter
```

基础设施失败会中止当前训练，不能当作奖励 0。模型生成非法动作则收到工具错误观察，可继续修复。每条轨迹都有独立的原始仓库副本，评分在新环境执行，只应用允许修改的源文件。

**这是精简的 on-policy GRPO 基线。** 每组新轨迹只执行一次 optimizer step；组内末端奖励减均值后除以总体标准差。轨迹等权，轨迹内部按动作 token 平均。`beta=0`，没有参考模型 KL，也没有 value model。虽然实现 clipped ratio 目标，每次更新前 ratio 通常约为 1，当前单步设置下 clip 基本不起约束作用。终局奖励广播给各决策，不代表已经解决逐步信用分配。

同组奖励相同时没有相对学习信号，**明确跳过更新**，记录 `equal_rewards`。评判“RL 实际执行”需要看 `optimizer_steps`、`grad_norm`、`parameter_delta_l2` 和参数哈希变化，不能只看训练命令退出码。

训练采用全历史短任务。v1 的摘要和结构化上下文策略仍可独立评测，尚未接入 v2 的训练采样。每一步保留真实 prompt IDs，因此后续引入上下文压缩时可以按实际决策上下文训练，无需把不同上下文拼成不存在的连续序列。

历史 v2 的 SFT 验证与 RL 试跑详见 [v2 验证记录](VALIDATION_V2.md)。`runs/learning-v2/grpo` 中受评分解析错误影响的更新仍然无效；代码修复不会追溯性地使旧更新变有效，后续必须从有效检查点在云端重新采样。当前真实仓库方向的 7B 推理验证与本节 0.5B 训练流程不同，推理或评分通过都不等于完成了 SFT 或 RL。

## 环境与命令

当前 Featurize 云端 Linux 已完成 RTX 4090 24GB 的 CUDA/BF16 运算和 Docker 执行检查，具体环境见 [云端配置记录](CLOUD_SETUP_FEATURIZE.md)。这些检查证明环境可用，不证明任何模型训练已经成功。

后续训练起步可先使用现有 0.5B 配置；扩大到 1.5B–3B、2048 上下文、group size 4、顺序 rollout 和 LoRA 时，需实际测量显存峰值再扩大规模。7B 推理可用不代表 7B 训练及长上下文同样能放进 24GB。更大模型或更长训练上下文应重新评估训练配置、GPU 和费用，不能直接沿用推理验证结论。

模型与数据需在云端准备；模型缓存放在实例本地磁盘，代码、日志和需要保留的 adapter 放在云端持久工作目录。云平台必须允许创建 Docker 工具沙箱，否则需要安排独立执行服务，不能直接把模型生成的 shell 命令放到宿主机运行。

历史 Windows 验证曾使用 `.venv-train`，通过 `--system-site-packages` 复用已有 CUDA PyTorch，相关版本记录在 `requirements-train.verified.txt`；它不是当前的训练执行入口，也不是完整依赖锁。普通本地 v1 环境 `.venv` 不需要安装训练依赖。重建其他云端 Linux 环境时，先安装匹配硬件的 CUDA PyTorch，再安装项目训练依赖；现有 Featurize 环境已配置，无需为了阅读这些示例重复安装。

以下命令在已连接的云端 Linux 终端执行，仅作为已获授权的训练操作参考，不延长当前租期或扩大预算。下一阶段实验应先根据实际任务、显存和时长估算所需费用。每次运行选择新输出目录；实例归还后，本地磁盘上的虚拟环境需要按配置记录重建。

```bash
source /home/featurize/budget-agent-env/bin/activate
cd /home/featurize/work/budget-agent

# 12 个训练任务、6 个评测任务；SFT 仅导出训练答案
python -m budget_agent.learning.cli curriculum --output data/curriculum

# 未训练的模型基线
python -m budget_agent.learning.cli evaluate --tasks data/curriculum/eval.jsonl --output runs/my-v2/base-eval

# 学习工具协议；这一步是 SFT
python -m budget_agent.learning.cli sft --data data/curriculum/sft.jsonl --steps 60 --output runs/my-v2/sft

# SFT 基线与 RL 必须分别评价
python -m budget_agent.learning.cli evaluate --tasks data/curriculum/eval.jsonl --adapter runs/my-v2/sft/adapter --output runs/my-v2/sft-eval

# 每个 group 4 条模型自己生成的轨迹，末端测试奖励驱动梯度更新
python -m budget_agent.learning.cli train --tasks data/curriculum/train.jsonl --adapter runs/my-v2/sft/adapter --output runs/my-v2/grpo

python -m budget_agent.learning.cli evaluate --tasks data/curriculum/eval.jsonl --adapter runs/my-v2/grpo/adapter --output runs/my-v2/grpo-eval
```

默认配置在 `configs/train-smoke.toml`，三个阶段用同一个推理预算：每次上下文 1536 token、累计输入加输出 8192 token、最多 5 次动作、每次最多生成 192 token。实际计数使用模型 tokenizer，重复输入的历史也计费。训练时反向传播计算不属于推理 token 预算，不能据此比较训练总成本。不同模型的 tokenizer 计数也不是相同 FLOPs。

评测每题用 `seed + task_index` 重置随机种子，三个阶段使用相同任务顺序与温度；每题只采一个样本，没有统计置信保证。全部评测任务保留，包括失败项。

## 数据与结果边界

当前课程只有两类合成 Python 边界 bug。训练用奇数常量，评测用偶数常量；模板相同。这只能验证参数分离、工具调用和优化闭环，不能证明跨仓库泛化。SFT 教师由确定性规则生成，训练轨迹由模型实际采样，二者明确分开。

训练工具看不到私有测试。私有测试是评分依据，不能因为有“私有”二字就认为这套极小课程有真实业务难度。真实 SWE-bench 的任务准备、补丁导出和官方评分仍由 v1 命令负责，本轮没有声称取得 SWE-bench 分数。

每个运行保存：

- `config.json`：模型版本、推理预算、训练参数。
- `metrics.jsonl`：SFT loss，或每组 RL 奖励、梯度、参数变化和跳过原因。
- 每个 episode 的 `trajectory.jsonl`、`episode.json`、`patch.diff`、`grade.json`：实际动作、token IDs、旧 log probabilities、补丁及独立测试结果。
- `adapter/`：LoRA 权重、tokenizer 和策略版本。
- `checkpoint.json`、`optimizer.pt`：参数哈希与优化器快照。当前 CLI 只支持加载 adapter 后启动新优化器，**不提供精确断点续训**。

训练程序不使用 TRL/verl。它把更新核心保持为可阅读的 PyTorch 实现，方便检查 token 对齐、奖励、预算和更新是否真实发生。未来更大规模的数据并行、推理批处理、异步采样和 checkpoint 恢复应迁移到成熟训练框架，而不是继续扩张此参考实现。

## 代码阅读顺序

`learning/schema.py` → `rollout.py` → `policy.py` → `trainer.py` → `cli.py`。

外部输入只在明确边界检查：配置入口、工具 JSON 解析、文件与执行权限、预算和独立评分。内部数据使用 dataclass 与直接字段访问。没有宽泛 `except Exception`、OOM 自动降级、设备回退或把程序错误转换成“正常负奖励”的逻辑。

算法实现可对照 [TRL 官方 GRPO 说明](https://huggingface.co/docs/trl/grpo_trainer)，适配器用法可对照 [PEFT LoRA 文档](https://huggingface.co/docs/peft/main/en/package_reference/lora)。本项目的精简更新设置不等同于这些框架的完整实现。
