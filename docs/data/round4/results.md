| pool | split | tasks | attempts | pass@1 | pass@k | solved tasks | statuses | evaluation | edit success | attempts/h |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | eval | 96 | 768 | 0.05859375 | {'pass@2': 0.09040178571428571, 'pass@4': 0.1193452380952381, 'pass@8': 0.13541666666666666} | 13 | {'finished': 92, 'max_steps': 245, 'stuck': 431} | {'resolved': 45, 'unresolved': 723} (timeouts 1) | {'edit_file': '1666/3304'} | 732.4 |
| relabel | eval | 96 | 768 | 0.06901041666666667 | {'pass@2': 0.10305059523809523, 'pass@4': 0.14092261904761905, 'pass@8': 0.16666666666666666} | 16 | {'stuck': 413, 'max_steps': 258, 'finished': 97} | {'unresolved': 715, 'resolved': 53} (timeouts 0) | {'edit_file': '1640/3295'} | 764.8 |

SFT data: {'examples': 0, 'tasks': 0, 'by_mode': {}}
Training: None
Loss (first, last): [] … []

Paired eval: base 0.069 → candidate 0.059, delta -0.010, 95% CI [-0.030, +0.009], no detectable change
Paired eval_drop_flagged: base 0.069 → candidate 0.059, delta -0.010, 95% CI [-0.030, +0.009], no detectable change

Tamper audit fixed: 0 resolved attempts with signals, 0 in graded files
Tamper audit relabel: 0 resolved attempts with signals, 0 in graded files

GPU stage time: 2.09 h, cost ≈ ¥6.85
Stages: serve-relabel 0 min, relabel 60 min, serve-fixed 1 min, fixed 62 min, stop-vllm 0 min
