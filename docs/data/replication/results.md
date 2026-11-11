| pool | split | tasks | attempts | pass@1 | pass@k | solved tasks | statuses | evaluation | edit success | attempts/h |
|---|---|---|---|---|---|---|---|---|---|---|
| rep-grpo | eval | 96 | 384 | 0.09895833333333333 | {'pass@2': 0.13020833333333334, 'pass@4': 0.14583333333333334} | 14 | {'finished': 92, 'stuck': 147, 'max_steps': 145} | {'resolved': 38, 'unresolved': 346} (timeouts 0) | {'edit_file': '1142/2307'} | 787.6 |
| rep-sft | eval | 96 | 384 | 0.0703125 | {'pass@2': 0.10416666666666667, 'pass@4': 0.13541666666666666} | 13 | {'stuck': 205, 'max_steps': 134, 'finished': 45} | {'unresolved': 357, 'resolved': 27} (timeouts 0) | {'edit_file': '908/1685'} | 983.2 |

SFT data: {'examples': 0, 'tasks': 0, 'by_mode': {}}
Training: None
Loss (first, last): [] … []

Paired eval: base 0.070 → candidate 0.099, delta +0.029, 95% CI [+0.000, +0.057], no detectable change
Paired eval_drop_flagged: base 0.070 → candidate 0.099, delta +0.029, 95% CI [+0.000, +0.057], no detectable change
Paired noise_grpo: base 0.096 → candidate 0.099, delta +0.003, 95% CI [-0.023, +0.029], no detectable change
Paired noise_sft: base 0.062 → candidate 0.070, delta +0.008, 95% CI [-0.013, +0.026], no detectable change
Paired pooled: base 0.066 → candidate 0.098, delta +0.031, 95% CI [+0.005, +0.061], improved

Tamper audit rep-grpo: 0 resolved attempts with signals, 0 in graded files
Tamper audit rep-sft: 0 resolved attempts with signals, 0 in graded files

GPU stage time: 0.92 h, cost ≈ ¥3.03
Stages: serve-sft 1 min, rep-sft 23 min, serve-grpo 1 min, rep-grpo 29 min, stop-vllm 0 min
