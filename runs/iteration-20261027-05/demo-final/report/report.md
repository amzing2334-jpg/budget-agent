# Budget Agent · Evaluation report

Scripted smoke runs verify plumbing only; they are not LLM performance evidence. Unknown cost is not zero. Pending/ungraded tasks are not counted as failures or successes. Official empty patches have no executed tests; infrastructure failures are ungraded and counted separately. A success rate is shown only when every attempt has a verifier outcome and valid budget accounting. Budget overruns and unknown usage invalidate fixed-budget comparisons. Harnesses, configurations, models, splits and token-accounting modes are grouped separately; compare strategies only with matched task fingerprints and repetition labels. API seeds are repetition labels; the Transformers backend resets its RNG per attempt. Neither guarantees cross-hardware reproducibility.

## scripted_smoke · scripted-smoke-v1 · dev · 9d4f9076a9

Context: 16000 · total: 64000 · tokenizer: byte

| Strategy | Success / graded / attempted | Infra errors | Empty patches | Rate | Tokens | Total USD | USD / success | Seconds / task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recent | 1 / 1 / 1 | 0 | 0 | 100.0% | 24,560 | — | — | 0.40 |

```json
{
  "actual_models": [
    "scripted-smoke-v1"
  ],
  "config": {
    "allow_local_execution": false,
    "api_enable_thinking": null,
    "api_key_env": "LLM_API_KEY",
    "api_max_tokens_field": "max_tokens",
    "base_url": "",
    "context_tokens": 16000,
    "decision_policy": "direct",
    "device": "cuda",
    "docker_image": "python:3.11-slim",
    "environment": "local",
    "execution_policy": "restricted",
    "history_format": "chat",
    "input_price_per_million": null,
    "loop_policy": "off",
    "max_cost_usd": null,
    "max_output_tokens": 1536,
    "max_reflections": 1,
    "max_steps": 30,
    "min_action_output_tokens": 128,
    "model": "",
    "model_dtype": "bfloat16",
    "model_revision": "",
    "output_price_per_million": null,
    "provider": "scripted",
    "read_memory_tokens": 1024,
    "reflection_max_output_tokens": 256,
    "reflection_policy": "evidence",
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
  "source_fingerprint": "bd2aaca7319b96affa7afb5664d10f426c4a688a17b143762b7a801bafeae9d3",
  "split": "dev",
  "task_kind": "local",
  "usage_sources": [
    "scripted_estimate"
  ]
}
```

## scripted_smoke · scripted-smoke-v1 · dev · 98a895ca93

Context: 16000 · total: 64000 · tokenizer: byte

| Strategy | Success / graded / attempted | Infra errors | Empty patches | Rate | Tokens | Total USD | USD / success | Seconds / task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recent | 1 / 1 / 1 | 0 | 0 | 100.0% | 25,904 | — | — | 0.40 |

```json
{
  "actual_models": [
    "scripted-smoke-v1"
  ],
  "config": {
    "allow_local_execution": false,
    "api_enable_thinking": null,
    "api_key_env": "LLM_API_KEY",
    "api_max_tokens_field": "max_tokens",
    "base_url": "",
    "context_tokens": 16000,
    "decision_policy": "direct",
    "device": "cuda",
    "docker_image": "python:3.11-slim",
    "environment": "local",
    "execution_policy": "restricted",
    "history_format": "chat",
    "input_price_per_million": null,
    "loop_policy": "off",
    "max_cost_usd": null,
    "max_output_tokens": 1536,
    "max_reflections": 1,
    "max_steps": 30,
    "min_action_output_tokens": 128,
    "model": "",
    "model_dtype": "bfloat16",
    "model_revision": "",
    "output_price_per_million": null,
    "provider": "scripted",
    "read_memory_tokens": 1024,
    "reflection_max_output_tokens": 256,
    "reflection_policy": "failure",
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
  "source_fingerprint": "bd2aaca7319b96affa7afb5664d10f426c4a688a17b143762b7a801bafeae9d3",
  "split": "dev",
  "task_kind": "local",
  "usage_sources": [
    "scripted_estimate"
  ]
}
```

## scripted_smoke · scripted-smoke-v1 · dev · 42a8f28f57

Context: 16000 · total: 64000 · tokenizer: byte

| Strategy | Success / graded / attempted | Infra errors | Empty patches | Rate | Tokens | Total USD | USD / success | Seconds / task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recent | 1 / 1 / 1 | 0 | 0 | 100.0% | 21,743 | — | — | 0.42 |

```json
{
  "actual_models": [
    "scripted-smoke-v1"
  ],
  "config": {
    "allow_local_execution": false,
    "api_enable_thinking": null,
    "api_key_env": "LLM_API_KEY",
    "api_max_tokens_field": "max_tokens",
    "base_url": "",
    "context_tokens": 16000,
    "decision_policy": "direct",
    "device": "cuda",
    "docker_image": "python:3.11-slim",
    "environment": "local",
    "execution_policy": "restricted",
    "history_format": "chat",
    "input_price_per_million": null,
    "loop_policy": "off",
    "max_cost_usd": null,
    "max_output_tokens": 1536,
    "max_reflections": 1,
    "max_steps": 30,
    "min_action_output_tokens": 128,
    "model": "",
    "model_dtype": "bfloat16",
    "model_revision": "",
    "output_price_per_million": null,
    "provider": "scripted",
    "read_memory_tokens": 1024,
    "reflection_max_output_tokens": 256,
    "reflection_policy": "off",
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
  "source_fingerprint": "bd2aaca7319b96affa7afb5664d10f426c4a688a17b143762b7a801bafeae9d3",
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
| slugify-digits | recent | finished | resolved | success |
| slugify-digits | recent | finished | resolved | success |
| slugify-digits | recent | finished | resolved | success |

## Run behavior diagnostics

Tool success does not establish a correct repair or a passed test. Repeated observations have identical action, output and recorded revision; repetition does not necessarily indicate an error. unavailable means the result has no stored count. This report does not reanalyze old trajectories.

| Run | Task | Strategy | Successful read | Successful edit + write | Successful run | Successful diff | Invalid actions | Failed action repeats | Identical observation repeats |
|---|---|---|---|---|---|---|---|---|---|
| 20261027T070744Z-a4c8678c | slugify-digits | recent | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| 20261027T070745Z-8622c220 | slugify-digits | recent | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| 20261027T070744Z-059bde10 | slugify-digits | recent | 1 | 1 | 0 | 1 | 0 | 0 | 0 |

## Run acceptance contracts

Per-run recorded contracts only: not production certification. Safety covers direct tool authorization, not arbitrary behavior inside tests. Unknown evidence cannot pass; reflection text is a hypothesis. Scripted runs remain software checks, not model capability evidence.

| Run | Quality | Token ledger | Tool policy | Acceptance | Feedback policy | Memory notes | Reflection calls | Blocked tools |
|---|---|---|---|---|---|---|---|---|
| 20261027T070744Z-a4c8678c | pass | pass | pass | pass | evidence | 1 | 0 | 0 |
| 20261027T070745Z-8622c220 | pass | pass | pass | pass | failure | 1 | 1 | 0 |
| 20261027T070744Z-059bde10 | pass | pass | pass | pass | off | 0 | 0 | 0 |
