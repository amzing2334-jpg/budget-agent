# 前缀共享的多 turn 打包训练

本文写于 2026-11-11，代码见 `grpo_core.pack_turns`、`grpo_core.chain_layout` 与 `scripts/grpo_harness.py --pack-turns`。

## 问题

GRPO 把一次尝试的每个 agent 请求当作一个训练 turn。一个 turn 的输入是"prompt + 已采样的回复"，而 `recent` 上下文策略在历史放得下时会把之前的全部内容原样重发，所以同一次尝试中，相邻 turn 的 prompt 大段重复。逐 turn 训练会把这些共享前缀反复编码，这是第二轮 GRPO 训练时间是 rollout 2 到 4 倍的主要原因。

## 做法

- **成链**：按时间顺序检查每个 turn。如果它的 prompt 在 token 层面以"上一个成员的 prompt + 采样回复"开头，就并入同一条链。
- **一次前向**：每条链只做一次前向，输入是最后一个 turn 的完整序列。通过 `logits_to_keep` 传入位置索引，只在"预测各段回复 token"的位置计算 logits。
- **为什么等价**：在因果注意力下，这些位置看到的上下文恰好是每个 turn 采样时的 prompt。所以逐 token 的 log probability、loss 和梯度都与逐 turn 前向相同。
- **适用范围**：只支持按 token 聚合 loss；与按序列聚合同时使用会直接报错。

## 验证

`tests/test_grpo_packing.py` 在 GPU 主机的 CPU 上运行，库版本与训练环境相同：torch 2.14、transformers 5.20、peft 0.21。

- 用一个随机初始化的小 Qwen2，加上非零的 LoRA，构造三个前缀一致的 turn。
- 打包前向与逐 turn 前向相比，log probability 和全部可训练参数的梯度都在 1e-5 以内一致。
- 前缀断开时会另起一条序列。

## 实测：能省多少

对第三轮前 4 步的真实 rollout（`docs/data/packing_round3_steps0-3.json`）运行 `scripts/bench_turn_packing.py`，只用 CPU 统计。

| 指标 | 数值 |
|---|---|
| 尝试 / turn | 256 / 4,619，prompt 摘要全部一致 |
| 逐 turn 编码的 token | 3,291 万 |
| 打包后编码的 token | 1,075 万 |
| **减少** | **3.06 倍**，训练的 completion token 完全相同（16.7 万） |
| 平均每条序列的 turn 数 | 4.86；有 26 条链覆盖了整次尝试（30 个 turn） |

链断开的原因（共 695 次）：

| 原因 | 次数 |
|---|---|
| 上下文策略改写了历史 | 672 |
| 模板对 agent 回复重新分词，与采样时的 token 不同 | 23（3.3%） |

## 含义

- **训练成本**：打包后训练全部 turn，花的编码量只有逐 turn 训练全部 turn 的三分之一。第三轮为了省钱每次尝试只抽 4 个 turn，打包后可以不抽样，从而去掉抽样带来的梯度方差。
- **重分词漂移**：在这里很少见（3.3%）。prompt 摘要的核对仍然是必要的护栏，因为一旦出现，训练就会落在策略没见过的上下文上。
- **上下文策略决定了打包效率**：链主要断在 `recent` 策略滑动窗口、改写历史的地方。同样的前缀也决定推理时的前缀缓存命中率。窗口滑动时一次性丢掉一大块历史，而不是每轮丢一条，可以让之后的 turn 重新共享长前缀，推理和训练两边都受益。这是下一步值得做的设计。
