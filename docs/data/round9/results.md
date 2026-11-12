| pool | split | tasks | attempts | pass@1 | pass@k | solved tasks | statuses | evaluation | edit success | attempts/h |
|---|---|---|---|---|---|---|---|---|---|---|
| eval-v26 | eval | 96 | 768 | 0.10807291666666667 | {'pass@2': 0.13988095238095238, 'pass@4': 0.17380952380952383, 'pass@8': 0.20833333333333334} | 20 | {'finished': 100, 'max_steps': 422, 'stuck': 246} | {'resolved': 83, 'unresolved': 685} (timeouts 0) | {'edit_file': '1231/2911'} | 609.2 |

Paired continued: base 0.108 → candidate 0.108, delta +0.000, 95% CI [-0.023, +0.022], no detectable change
Paired eval: base 0.124 → candidate 0.108, delta -0.016, 95% CI [-0.044, +0.009], no detectable change
Paired eval_drop_flagged: base 0.124 → candidate 0.108, delta -0.016, 95% CI [-0.044, +0.009], no detectable change
Paired sft: base 0.069 → candidate 0.108, delta +0.039, 95% CI [+0.009, +0.073], improved

Tamper audit eval-v26: 0 resolved attempts with signals, 0 in graded files

GPU stage time: 1.28 h, cost ≈ ¥4.2
Stages: serve-final 1 min, eval-v26 75 min, stop-vllm 0 min
