| pool | split | tasks | attempts | pass@1 | pass@k | solved tasks | statuses | evaluation | edit success | attempts/h |
|---|---|---|---|---|---|---|---|---|---|---|
| eval-grpo | eval | 96 | 768 | 0.10807291666666667 | {'pass@2': 0.13058035714285715, 'pass@4': 0.15848214285714285, 'pass@8': 0.1875} | 18 | {'finished': 83, 'stuck': 193, 'max_steps': 492} | {'resolved': 83, 'unresolved': 685} (timeouts 0) | {'edit_file': '370/667', 'replace_lines': '2/2'} | 135.2 |

GRPO: 46 steps, 180 updates, 2944 attempts (0 ungraded), 6432 turns in 3027 sequences, prompt mismatches 0, logprob gap [0.0008, 0.0087], TIS mean [0.999, 1.0005] (capped at most 0.0), clip at most 0.2838, entropy first/last [0.103, 3.0437], train share 0.301, steps without an update 1

Paired eval: base 0.124 → candidate 0.108, delta -0.016, 95% CI [-0.047, +0.016], no detectable change
Paired eval_drop_flagged: base 0.124 → candidate 0.108, delta -0.016, 95% CI [-0.047, +0.016], no detectable change
Paired sft: base 0.069 → candidate 0.108, delta +0.039, 95% CI [+0.008, +0.076], improved

Tamper audit eval-grpo: 0 resolved attempts with signals, 0 in graded files

GPU stage time: 8.93 h, cost ≈ ¥29.29
Stages: serve-policy 1 min, grpo 192 min, serve-final 1 min, eval-grpo 340 min, stop-vllm 0 min
