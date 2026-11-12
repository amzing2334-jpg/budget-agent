| pool | split | tasks | attempts | pass@1 | pass@k | solved tasks | statuses | evaluation | edit success | attempts/h |
|---|---|---|---|---|---|---|---|---|---|---|
| eval-grpo | eval | 96 | 768 | 0.14322916666666666 | {'pass@2': 0.1636904761904762, 'pass@4': 0.18125, 'pass@8': 0.19791666666666666} | 19 | {'finished': 184, 'stuck': 447, 'max_steps': 137} | {'resolved': 110, 'unresolved': 658} (timeouts 0) | {'edit_file': '1692/4064'} | 698.4 |

GRPO: 29 steps, 116 updates, 1856 attempts (0 ungraded), 5248 turns in 2598 sequences, prompt mismatches 0, logprob gap [0.002, 0.0048], TIS mean [0.9992, 1.0018] (capped at most 0.0006), clip at most 0.0009, entropy first/last [0.0822, 0.0624], train share 0.476, steps without an update 0

Paired eval: base 0.124 → candidate 0.143, delta +0.020, 95% CI [-0.009, +0.052], no detectable change
Paired eval_drop_flagged: base 0.124 → candidate 0.143, delta +0.020, 95% CI [-0.009, +0.052], no detectable change
Paired sft: base 0.069 → candidate 0.143, delta +0.074, 95% CI [+0.034, +0.120], improved

Tamper audit eval-grpo: 0 resolved attempts with signals, 0 in graded files

GPU stage time: 3.85 h, cost ≈ ¥12.63
Stages: serve-policy 1 min, grpo 162 min, serve-final 1 min, eval-grpo 66 min, stop-vllm 0 min
