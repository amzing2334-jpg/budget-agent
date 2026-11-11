| pool | split | tasks | attempts | pass@1 | pass@k | solved tasks | statuses | evaluation | edit success | attempts/h |
|---|---|---|---|---|---|---|---|---|---|---|
| student-base-eval | eval | 96 | 384 | 0.0026041666666666665 | {'pass@2': 0.005208333333333333, 'pass@4': 0.010416666666666666} | 1 | {'finished': 47, 'stuck': 206, 'max_steps': 131} | {'unresolved': 383, 'resolved': 1} (timeouts 0) | {'edit_file': '93/895', 'replace_lines': '0/3'} | 970.3 |
| student-base-train | train | 192 | 192 | 0.0 | {} | 0 | {'finished': 21, 'stuck': 108, 'max_steps': 63} | {'unresolved': 192} (timeouts 0) | {'edit_file': '58/375'} | 2300.8 |
| student-sft-eval | eval | 96 | 384 | 0.0625 | {'pass@2': 0.10069444444444443, 'pass@4': 0.14583333333333334} | 14 | {'finished': 42, 'stuck': 215, 'max_steps': 127} | {'resolved': 24, 'unresolved': 360} (timeouts 0) | {'edit_file': '792/1599'} | 904.3 |
| student-sft-train | train | 192 | 192 | 0.026455026455026454 | {} | 5 | {'stuck': 120, 'finished': 12, 'max_steps': 60} | {'unresolved': 184, 'resolved': 5, 'grading_error': 3} (timeouts 0) | {'edit_file': '384/802'} | 677.3 |
| teacher-eval | eval | 96 | 192 | 0.07291666666666667 | {'pass@2': 0.11458333333333333} | 11 | {'stuck': 127, 'max_steps': 47, 'finished': 16, 'model_error': 2} | {'unresolved': 178, 'resolved': 14} (timeouts 0) | {'edit_file': '474/922', 'replace_lines': '1/1'} | 670.2 |
| teacher-train | train | 192 | 768 | 0.07161458333333333 | {'pass@2': 0.11805555555555557, 'pass@4': 0.17647058823529413} | 33 | {'stuck': 458, 'max_steps': 251, 'finished': 56, 'model_error': 3} | {'unresolved': 707, 'resolved': 55, 'grading_error': 6} (timeouts 0) | {'edit_file': '1739/3697', 'replace_lines': '4/5', 'write_file': '1/10'} | 807.0 |

SFT data: {'examples': 658, 'tasks': 33, 'by_mode': {'last': 658}}
Training: {'model': '/root/autodl-tmp/coding-agent-rl/models/Qwen2.5-Coder-1.5B-Instruct', 'data': '/root/autodl-tmp/coding-agent-rl/outputs/round1/sft.jsonl', 'examples': 619, 'dropped': 39, 'trained_tokens': 25522, 'epochs': 2.0, 'lr': 0.0001, 'rank': 64, 'optimizer_steps': 155, 'seconds': 1010.0}
Loss (first, last): [(1, 0.2095)] … [(155, 0.0951)]

Paired eval: base 0.003 → candidate 0.062, delta +0.060, 95% CI [+0.029, +0.096], improved
Paired train: base 0.000 → candidate 0.026, delta +0.026, 95% CI [+0.005, +0.053], improved

Tamper audit student-base-eval: 0 resolved attempts with signals, 0 in graded files
Tamper audit student-base-train: 0 resolved attempts with signals, 0 in graded files
Tamper audit student-sft-eval: 0 resolved attempts with signals, 0 in graded files
Tamper audit student-sft-train: 0 resolved attempts with signals, 0 in graded files
Tamper audit teacher-eval: 0 resolved attempts with signals, 0 in graded files
Tamper audit teacher-train: 0 resolved attempts with signals, 0 in graded files

GPU stage time: 2.79 h, cost ≈ ¥9.16
Stages: serve-teacher 1 min, teacher-train 57 min, teacher-eval 17 min, serve-student 1 min, student-base-eval 23 min, student-base-train 5 min, sft-data 0 min, stop-vllm 0 min, train-sft 17 min, serve-student-sft 1 min, student-sft-eval 25 min, student-sft-train 17 min, stop-vllm-final 0 min
