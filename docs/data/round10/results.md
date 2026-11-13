| pool | split | tasks | attempts | pass@1 | pass@k | solved tasks | statuses | evaluation | edit success | attempts/h |
|---|---|---|---|---|---|---|---|---|---|---|
| eval-grpo | eval | 96 | 768 | 0.14583333333333334 | {'pass@2': 0.1752232142857143, 'pass@4': 0.21264880952380952, 'pass@8': 0.25} | 24 | {'finished': 205, 'stuck': 274, 'max_steps': 289} | {'resolved': 112, 'unresolved': 656} (timeouts 0) | {'edit_file': '2585/4710', 'replace_lines': '2/2'} | 619.3 |
| select | train | 159 | 636 | 0.036163522012578615 | {'pass@2': 0.06184486373165619, 'pass@4': 0.0880503144654088} | 14 | {'stuck': 337, 'max_steps': 135, 'finished': 164} | {'unresolved': 613, 'resolved': 23} (timeouts 10) | {'edit_file': '1275/3627'} | 1201.5 |

GRPO: 13 steps, 52 updates, 832 attempts (0 ungraded), 2200 turns in 1230 sequences, prompt mismatches 0, logprob gap [0.0014, 0.0061], TIS mean [0.9996, 1.0014] (capped at most 0.0002), clip at most 0.0028, entropy first/last [0.0609, 0.068], train share 0.449, steps without an update 0, invalid action share [0.0059, 0.023], halted None

Paired eval: base 0.143 → candidate 0.146, delta +0.003, 95% CI [-0.021, +0.023], no detectable change
Paired eval_drop_flagged: base 0.143 → candidate 0.146, delta +0.003, 95% CI [-0.021, +0.023], no detectable change
Paired round5: base 0.124 → candidate 0.146, delta +0.022, 95% CI [-0.012, +0.061], no detectable change
Paired sft: base 0.069 → candidate 0.146, delta +0.077, 95% CI [+0.038, +0.121], improved

Tamper audit eval-grpo: 0 resolved attempts with signals, 0 in graded files
Tamper audit select: 0 resolved attempts with signals, 0 in graded files

GPU stage time: 3.61 h, cost ≈ ¥11.83
Stages: serve-start 1 min, serve-start 0 min, select 31 min, serve-policy 1 min, grpo 106 min, serve-final 1 min, eval-grpo 74 min, stop-vllm 0 min
