| pool | split | tasks | attempts | pass@1 | pass@k | solved tasks | statuses | evaluation | edit success | attempts/h |
|---|---|---|---|---|---|---|---|---|---|---|
| eval-grpo | eval | 96 | 384 | 0.11979166666666667 | {'pass@2': 0.1579861111111111, 'pass@4': 0.1875} | 18 | {'finished': 156, 'max_steps': 92, 'stuck': 136} | {'resolved': 46, 'unresolved': 338} (timeouts 1) | {'edit_file': '1017/2302'} | 601.7 |
| eval-sft | eval | 96 | 384 | 0.06510416666666667 | {'pass@2': 0.09895833333333333, 'pass@4': 0.13541666666666666} | 13 | {'stuck': 194, 'max_steps': 138, 'finished': 52} | {'unresolved': 359, 'resolved': 25} (timeouts 1) | {'edit_file': '864/1663'} | 766.1 |

SFT data: {'examples': 0, 'tasks': 0, 'by_mode': {}}
Training: None
Loss (first, last): [] … []

Paired eval: base 0.065 → candidate 0.120, delta +0.055, 95% CI [+0.021, +0.091], improved
Paired eval_drop_flagged: base 0.065 → candidate 0.120, delta +0.055, 95% CI [+0.021, +0.091], improved
Paired harness_sft: base 0.062 → candidate 0.065, delta +0.003, 95% CI [-0.016, +0.021], no detectable change

Tamper audit eval-grpo: 0 resolved attempts with signals, 0 in graded files
Tamper audit eval-sft: 0 resolved attempts with signals, 0 in graded files

GPU stage time: 4.27 h, cost ≈ ¥14.02
Stages: serve-policy 1 min, grpo 184 min, serve-sft 1 min, eval-sft 30 min, serve-final 1 min, eval-grpo 38 min, stop-vllm 0 min
