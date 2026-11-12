| pool | split | tasks | attempts | pass@1 | pass@k | solved tasks | statuses | evaluation | edit success | attempts/h |
|---|---|---|---|---|---|---|---|---|---|---|
| eval-grpo | eval | 96 | 768 | 0.12369791666666667 | {'pass@2': 0.15438988095238096, 'pass@4': 0.18288690476190475, 'pass@8': 0.20212765957446807} | 20 | {'finished': 125, 'stuck': 215, 'max_steps': 428} | {'resolved': 94, 'unresolved': 672, 'infrastructure_error': 2} (timeouts 3) | {'edit_file': '2360/4805', 'write_file': '5/5'} | 538.5 |

GRPO: 29 steps, 116 updates, 1856 attempts (0 ungraded), 5536 turns in 2794 sequences, prompt mismatches 0, logprob gap [0.0017, 0.0028], TIS mean [0.9994, 1.0005] (capped at most 0.0), clip at most 0.0011, entropy 0.0825 → 0.116, train share 0.427

Paired eval: base 0.069 → candidate 0.124, delta +0.055, 95% CI [+0.022, +0.091], improved
Paired eval_drop_flagged: base 0.069 → candidate 0.124, delta +0.055, 95% CI [+0.022, +0.091], improved

Tamper audit eval-grpo: 0 resolved attempts with signals, 0 in graded files

GPU stage time: 4.5 h, cost ≈ ¥14.76
Stages: serve-policy 1 min, grpo 185 min, serve-final 1 min, serve-final 1 min, eval-grpo 80 min, stop-vllm 0 min
