# Budget Agent · Evaluation report

Scripted smoke runs verify plumbing only; they are not LLM performance evidence. Unknown cost is not zero. Pending/ungraded tasks are not counted as failures or successes. Official empty patches have no executed tests; infrastructure failures are ungraded and counted separately. A success rate is shown only when every attempt has a verifier outcome and valid budget accounting. Budget overruns and unknown usage invalidate fixed-budget comparisons. Harnesses, configurations, models, splits and token-accounting modes are grouped separately; compare strategies only with matched task fingerprints and repetition labels. API seeds are repetition labels; the Transformers backend resets its RNG per attempt. Neither guarantees cross-hardware reproducibility.

## scripted_smoke · scripted-smoke-v1 · dev · da991ddec7

Context: 16000 · total: 64000 · tokenizer: byte

| Strategy | Success / graded / attempted | Infra errors | Empty patches | Rate | Tokens | Total USD | USD / success | Seconds / task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mask | 1 / 2 / 2 | 0 | 0 | 50.0% | 42,704 | — | — | 1.99 |

```json
{
  "actual_models": [
    "scripted-smoke-v1"
  ],
  "config": {
    "allow_local_execution": false,
    "api_key_env": "LLM_API_KEY",
    "api_max_tokens_field": "max_tokens",
    "base_url": "",
    "context_tokens": 16000,
    "decision_policy": "direct",
    "device": "cuda",
    "docker_image": "python:3.11-slim",
    "environment": "local",
    "history_format": "chat",
    "input_price_per_million": null,
    "loop_policy": "off",
    "max_cost_usd": null,
    "max_output_tokens": 1024,
    "max_steps": 12,
    "min_action_output_tokens": 128,
    "model": "",
    "model_dtype": "bfloat16",
    "model_revision": "",
    "output_price_per_million": null,
    "provider": "scripted",
    "temperature": 0.0,
    "timeout_seconds": 30,
    "tokenizer": "byte",
    "total_tokens": 64000
  },
  "evidence_kind": "scripted_smoke",
  "harness": null,
  "model": "scripted-smoke-v1",
  "model_metadata": null,
  "provider": "scripted",
  "source_fingerprint": "6234dcbaee573c4504b71dab55af8fb5316a46515f07f203bcd3bbe803c582df",
  "split": "dev",
  "task_kind": "local",
  "usage_sources": [
    "scripted_estimate"
  ]
}
```

## scripted_smoke · scripted-smoke-v1 · dev · 288221392a

Context: 16000 · total: 64000 · tokenizer: byte

| Strategy | Success / graded / attempted | Infra errors | Empty patches | Rate | Tokens | Total USD | USD / success | Seconds / task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mask | 1 / 2 / 2 | 0 | 0 | 50.0% | 28,469 | — | — | 6.38 |

```json
{
  "actual_models": [
    "scripted-smoke-v1"
  ],
  "config": {
    "allow_local_execution": false,
    "api_key_env": "LLM_API_KEY",
    "api_max_tokens_field": "max_tokens",
    "base_url": "",
    "context_tokens": 16000,
    "decision_policy": "direct",
    "device": "cuda",
    "docker_image": "python:3.11-slim",
    "environment": "local",
    "history_format": "chat",
    "input_price_per_million": null,
    "loop_policy": "recover",
    "max_cost_usd": null,
    "max_output_tokens": 1024,
    "max_steps": 12,
    "min_action_output_tokens": 128,
    "model": "",
    "model_dtype": "bfloat16",
    "model_revision": "",
    "output_price_per_million": null,
    "provider": "scripted",
    "temperature": 0.0,
    "timeout_seconds": 30,
    "tokenizer": "byte",
    "total_tokens": 64000
  },
  "evidence_kind": "scripted_smoke",
  "harness": null,
  "model": "scripted-smoke-v1",
  "model_metadata": null,
  "provider": "scripted",
  "source_fingerprint": "6234dcbaee573c4504b71dab55af8fb5316a46515f07f203bcd3bbe803c582df",
  "split": "dev",
  "task_kind": "local",
  "usage_sources": [
    "scripted_estimate"
  ]
}
```

## Run ledger

| Task | Strategy | Agent status | Evaluation status | Verified outcome |
|---|---|---|---|---|
| slugify-digits-edit-after-read | mask | finished | resolved | success |
| slugify-digits-edit-after-read | mask | finished | resolved | success |
| slugify-digits-persistent-read | mask | finished | unresolved | failed verification |
| slugify-digits-persistent-read | mask | stalled | unresolved | failed verification |

## Run behavior diagnostics

Tool success does not establish a correct repair or a passed test. Repeated observations have identical action, output and recorded revision; repetition does not necessarily indicate an error. unavailable means the result has no stored count. This report does not reanalyze old trajectories.

| Run | Task | Strategy | Successful read | Successful edit + write | Successful run | Successful diff | Invalid actions | Failed action repeats | Identical observation repeats |
|---|---|---|---|---|---|---|---|---|---|
| 20261026T093656Z-31763196 | slugify-digits-edit-after-read | mask | 2 | 1 | 1 | 1 | 0 | 0 | 1 |
| 20261026T093658Z-70e70a82 | slugify-digits-edit-after-read | mask | 2 | 1 | 1 | 1 | 0 | 0 | 1 |
| 20261026T093647Z-413f4990 | slugify-digits-persistent-read | mask | 6 | 0 | 0 | 0 | 0 | 0 | 5 |
| 20261026T093649Z-b6f86dcb | slugify-digits-persistent-read | mask | 3 | 0 | 0 | 0 | 0 | 0 | 2 |

## Read-loop control

Stored controller events only; no trajectory replay. A repeated visible read is not proof of faulty reasoning. off means not monitored; unavailable means the run predates controller logging.

| Run | Policy | Observed repeats | Warnings sent | Stopped | Read evidence |
|---|---|---|---|---|---|
| 20261026T093656Z-31763196 | off | not monitored | 0 | False | — |
| 20261026T093658Z-70e70a82 | recover | 1 | 1 | False | e0002 &lt;- e0001: slugify.py:1-6 |
| 20261026T093647Z-413f4990 | off | not monitored | 0 | False | — |
| 20261026T093649Z-b6f86dcb | recover | 2 | 1 | True | e0002 &lt;- e0001: slugify.py:1-6; e0003 &lt;- e0001: slugify.py:1-6 |
