| pool | split | tasks | attempts | pass@1 | pass@k | solved tasks | statuses | evaluation | edit success | attempts/h |
|---|---|---|---|---|---|---|---|---|---|---|
| grpo-eval | eval | 96 | 384 | 0.09635416666666667 | {'pass@2': 0.13368055555555555, 'pass@4': 0.16666666666666666} | 16 | {'finished': 92, 'stuck': 134, 'max_steps': 158} | {'resolved': 37, 'unresolved': 347} (timeouts 0) | {'edit_file': '1203/2331'} | 732.2 |

SFT data: {'examples': 0, 'tasks': 0, 'by_mode': {}}
Training: None
Loss (first, last): [] … []

Paired eval: base 0.062 → candidate 0.096, delta +0.034, 95% CI [+0.003, +0.070], improved
Paired eval_drop_flagged: base 0.062 → candidate 0.096, delta +0.034, 95% CI [+0.003, +0.070], improved

Tamper audit grpo-eval: 0 resolved attempts with signals, 0 in graded files

GPU stage time: 2.62 h, cost ≈ ¥8.58
Stages: serve-policy 1 min, grpo 122 min, serve-final 2 min, grpo-eval 31 min, stop-vllm 0 min
