# Read-memory fixed-prefix replay

Counter: `tiktoken:cl100k_base` — analysis estimate, not exact Qwen tokens or billing.

3 histories, 1 distinct task IDs; 39 prefixes, 39 paired constructed, 0 overflow.

Mean additional input inclusion cost per constructed prefix: 706.1538461538462 estimated tokens.

Known target line occurrences on paired read prefixes: 464; visible with mask: 4; visible with mask-read: 226.

These are repeated line occurrences across fixed prefixes, not independent source lines or tasks.

| Task ID | Recorded histories |
| --- | ---: |
| astropy__astropy-12907 | 3 |

Excluded runs:

- `runs/cloud-v5-a4000-20261022-01/adaptive-mask/astropy__astropy-12907__recent__20261022T072506Z-86c192b2`: Original strategy result=recent, config=recent; only mask is replayed
- `runs/cloud-v5-a4000-20261022-01/mini-baseline/astropy__astropy-12907__mini-swe-textbased`: External harness mini-swe-agent-textbased-budget-adapter; not an internal mask policy run
- `runs/cloud-v5-a4000-20261022-01/reserve-probe/reserve-1/astropy__astropy-12907__recent__20261022T071042Z-2c771740`: Original strategy result=recent, config=recent; only mask is replayed
- `runs/cloud-v5-a4000-20261022-01/reserve-probe/reserve-128/astropy__astropy-12907__recent__20261022T071302Z-adc44a6d`: Original strategy result=recent, config=recent; only mask is replayed

Limitations:

- Fixed recorded prefixes only: no new actions, model calls, repair successes, training or billed usage.
- Analysis token estimates use the selected encoding and estimated chat framing; they are not exact Qwen tokens or provider billing.
- Each prefix uses config.context_tokens - config.max_output_tokens, not the historical remaining total-budget reservation.
- Extra input inclusion costs do not establish whole-rollout savings or how a model would act after seeing different context.
- Overflow prefixes remain in the report with unknown visibility; they are not silently dropped or counted as zero.
- Repeated histories from the same task are not independent tasks; no statistical success-rate claim is made.
- Coverage means exact complete numbered source lines observed earlier in the current contiguous revision segment, not semantic usefulness.
