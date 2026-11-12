| pool | split | tasks | attempts | pass@1 | pass@k | solved tasks | statuses | evaluation | edit success | attempts/h |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed-trained | eval | 96 | 768 | 0.06510416666666667 | {'pass@2': 0.10379464285714285, 'pass@4': 0.15223214285714284, 'pass@8': 0.19791666666666666} | 19 | {'finished': 131, 'max_steps': 219, 'stuck': 418} | {'resolved': 50, 'unresolved': 718} (timeouts 2) | {'edit_file': '1636/3186'} | 741.7 |

SFT data: {'examples': 658, 'tasks': 33, 'by_mode': {'last': 658}}
Training: {'model': '/root/autodl-tmp/coding-agent-rl/models/Qwen2.5-Coder-1.5B-Instruct', 'data': '/root/autodl-tmp/coding-agent-rl/outputs/round6/sft.jsonl', 'examples': 619, 'dropped': 39, 'trained_tokens': 25522, 'epochs': 2.0, 'lr': 0.0001, 'rank': 64, 'optimizer_steps': 155, 'seconds': 1043.7}
Loss (first, last): [(1, 0.2117)] … [(155, 0.0902)]

Paired eval: base 0.069 → candidate 0.065, delta -0.004, 95% CI [-0.021, +0.013], no detectable change
Paired eval_drop_flagged: base 0.069 → candidate 0.065, delta -0.004, 95% CI [-0.021, +0.013], no detectable change
Paired shift: base 0.059 → candidate 0.065, delta +0.007, 95% CI [-0.010, +0.023], no detectable change

Tamper audit fixed-trained: 0 resolved attempts with signals, 0 in graded files

GPU stage time: 1.39 h, cost ≈ ¥4.57
Stages: sft-data 0 min, train-sft 19 min, serve-student 1 min, eval 62 min, stop-vllm 0 min
