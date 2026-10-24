# Budget Agent · Evaluation report

Scripted smoke runs verify plumbing only; they are not LLM performance evidence. Unknown cost is not zero. Pending/ungraded tasks are not counted as failures or successes. Official empty patches have no executed tests; infrastructure failures are ungraded and counted separately. A success rate is shown only when every attempt has a verifier outcome and valid budget accounting. Budget overruns and unknown usage invalidate fixed-budget comparisons. Harnesses, configurations, models, splits and token-accounting modes are grouped separately; compare strategies only with matched task fingerprints and repetition labels. API seeds are repetition labels; the Transformers backend resets its RNG per attempt. Neither guarantees cross-hardware reproducibility.

## model_run · Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 · dev · 683c1965a4

Context: 8192 · total: 32000 · tokenizer: hf:Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554

| Strategy | Success / graded / attempted | Infra errors | Empty patches | Rate | Tokens | Total USD | USD / success | Seconds / task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mini-swe-textbased | 0 / 0 / 1 | 0 | 1 | ungraded | 26,800 | — | — | 88.91 |

```json
{
  "actual_models": [
    "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554"
  ],
  "config": {
    "allow_local_execution": false,
    "api_key_env": "LLM_API_KEY",
    "api_max_tokens_field": "max_tokens",
    "base_url": "",
    "context_policy": "upstream_full_history",
    "context_tokens": 8192,
    "device": "cuda",
    "docker_image": "python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce",
    "environment": "docker",
    "harness": "mini-swe-agent-textbased-budget-adapter",
    "history_format": "upstream_textbased",
    "input_price_per_million": null,
    "max_cost_usd": null,
    "max_output_tokens": 768,
    "max_steps": 20,
    "min_action_output_tokens": 128,
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "model_dtype": "bfloat16",
    "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "output_price_per_million": null,
    "provider": "transformers",
    "submission_policy": "explicit_upstream_submission",
    "temperature": 0.0,
    "timeout_seconds": 120,
    "tokenizer": "hf:Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
    "total_tokens": 32000,
    "upstream_sha": "a83fcae82d2a08f0ee0c688f9d137b3566c097f8"
  },
  "evidence_kind": "model_run",
  "harness": "mini-swe-agent-textbased-budget-adapter",
  "model": "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
  "model_metadata": {
    "device": "cuda",
    "dtype": "bfloat16",
    "generation": {
      "bos_token_id": null,
      "do_sample": false,
      "eos_token_id": [
        151645,
        151643
      ],
      "num_beams": 1,
      "pad_token_id": 151643,
      "repetition_penalty": 1.0,
      "temperature": 1.0,
      "top_k": 0,
      "top_p": 1.0,
      "use_cache": true
    },
    "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "seed_scope": "torch RNG reset once at each task/strategy attempt; not a cross-hardware determinism guarantee",
    "tokenizer_revision": "cdbee75f17c01a7cc42f958dc650907174af0554"
  },
  "provider": "transformers",
  "source_fingerprint": "36652ef463361827c7e5f48991ca7eb864fd7c34af70f49e9709ba5dd5f3aa1a",
  "split": "dev",
  "task_kind": "swebench",
  "usage_sources": [
    "tokenizer_exact"
  ]
}
```

## model_run · Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 · dev · 8b26a30aa2

Context: 8192 · total: 32000 · tokenizer: hf:Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554

| Strategy | Success / graded / attempted | Infra errors | Empty patches | Rate | Tokens | Total USD | USD / success | Seconds / task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mask | 0 / 0 / 1 | 0 | 1 | ungraded | 31,801 | — | — | 140.42 |

```json
{
  "actual_models": [
    "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554"
  ],
  "config": {
    "allow_local_execution": false,
    "api_key_env": "LLM_API_KEY",
    "api_max_tokens_field": "max_tokens",
    "base_url": "",
    "context_tokens": 8192,
    "decision_policy": "budget-aware",
    "device": "cuda",
    "docker_image": "python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce",
    "environment": "docker",
    "history_format": "chat",
    "input_price_per_million": null,
    "max_cost_usd": null,
    "max_output_tokens": 768,
    "max_steps": 20,
    "min_action_output_tokens": 128,
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "model_dtype": "bfloat16",
    "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "output_price_per_million": null,
    "provider": "transformers",
    "temperature": 0.0,
    "timeout_seconds": 120,
    "tokenizer": "hf:Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
    "total_tokens": 32000
  },
  "evidence_kind": "model_run",
  "harness": null,
  "model": "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
  "model_metadata": {
    "device": "cuda",
    "dtype": "bfloat16",
    "generation": {
      "bos_token_id": null,
      "do_sample": false,
      "eos_token_id": [
        151645,
        151643
      ],
      "num_beams": 1,
      "pad_token_id": 151643,
      "repetition_penalty": 1.0,
      "temperature": 1.0,
      "top_k": 0,
      "top_p": 1.0,
      "use_cache": true
    },
    "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "seed_scope": "torch RNG reset once at each task/strategy attempt; not a cross-hardware determinism guarantee",
    "tokenizer_revision": "cdbee75f17c01a7cc42f958dc650907174af0554"
  },
  "provider": "transformers",
  "source_fingerprint": "d3ca73f7db6c7ecd49832e1c04d9d87fa08b73092b7dd1189a19d83fe98a4528",
  "split": "dev",
  "task_kind": "swebench",
  "usage_sources": [
    "tokenizer_exact"
  ]
}
```

## model_run · Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 · dev · 3d654c65ae

Context: 8192 · total: 32000 · tokenizer: hf:Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554

| Strategy | Success / graded / attempted | Infra errors | Empty patches | Rate | Tokens | Total USD | USD / success | Seconds / task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mask | 0 / 0 / 1 | 0 | 1 | ungraded | 31,686 | — | — | 81.82 |

```json
{
  "actual_models": [
    "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554"
  ],
  "config": {
    "allow_local_execution": false,
    "api_key_env": "LLM_API_KEY",
    "api_max_tokens_field": "max_tokens",
    "base_url": "",
    "context_tokens": 8192,
    "decision_policy": "direct",
    "device": "cuda",
    "docker_image": "python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce",
    "environment": "docker",
    "history_format": "chat",
    "input_price_per_million": null,
    "max_cost_usd": null,
    "max_output_tokens": 768,
    "max_steps": 20,
    "min_action_output_tokens": 128,
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "model_dtype": "bfloat16",
    "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "output_price_per_million": null,
    "provider": "transformers",
    "temperature": 0.0,
    "timeout_seconds": 120,
    "tokenizer": "hf:Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
    "total_tokens": 32000
  },
  "evidence_kind": "model_run",
  "harness": null,
  "model": "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
  "model_metadata": {
    "device": "cuda",
    "dtype": "bfloat16",
    "generation": {
      "bos_token_id": null,
      "do_sample": false,
      "eos_token_id": [
        151645,
        151643
      ],
      "num_beams": 1,
      "pad_token_id": 151643,
      "repetition_penalty": 1.0,
      "temperature": 1.0,
      "top_k": 0,
      "top_p": 1.0,
      "use_cache": true
    },
    "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "seed_scope": "torch RNG reset once at each task/strategy attempt; not a cross-hardware determinism guarantee",
    "tokenizer_revision": "cdbee75f17c01a7cc42f958dc650907174af0554"
  },
  "provider": "transformers",
  "source_fingerprint": "d3ca73f7db6c7ecd49832e1c04d9d87fa08b73092b7dd1189a19d83fe98a4528",
  "split": "dev",
  "task_kind": "swebench",
  "usage_sources": [
    "tokenizer_exact"
  ]
}
```

## model_run · Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 · dev · 3ae6e69012

Context: 8192 · total: 32000 · tokenizer: hf:Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554

| Strategy | Success / graded / attempted | Infra errors | Empty patches | Rate | Tokens | Total USD | USD / success | Seconds / task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recent | 0 / 0 / 1 | 0 | 1 | ungraded | 31,427 | — | — | 113.75 |

```json
{
  "actual_models": [
    "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554"
  ],
  "config": {
    "allow_local_execution": false,
    "api_key_env": "LLM_API_KEY",
    "api_max_tokens_field": "max_tokens",
    "base_url": "",
    "context_tokens": 8192,
    "device": "cuda",
    "docker_image": "python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce",
    "environment": "docker",
    "history_format": "chat",
    "input_price_per_million": null,
    "max_cost_usd": null,
    "max_output_tokens": 768,
    "max_steps": 20,
    "min_action_output_tokens": 1,
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "model_dtype": "bfloat16",
    "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "output_price_per_million": null,
    "provider": "transformers",
    "temperature": 0.0,
    "timeout_seconds": 120,
    "tokenizer": "hf:Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
    "total_tokens": 32000
  },
  "evidence_kind": "model_run",
  "harness": null,
  "model": "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
  "model_metadata": {
    "device": "cuda",
    "dtype": "bfloat16",
    "generation": {
      "bos_token_id": null,
      "do_sample": false,
      "eos_token_id": [
        151645,
        151643
      ],
      "num_beams": 1,
      "pad_token_id": 151643,
      "repetition_penalty": 1.0,
      "temperature": 1.0,
      "top_k": 0,
      "top_p": 1.0,
      "use_cache": true
    },
    "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "seed_scope": "torch RNG reset once at each task/strategy attempt; not a cross-hardware determinism guarantee",
    "tokenizer_revision": "cdbee75f17c01a7cc42f958dc650907174af0554"
  },
  "provider": "transformers",
  "source_fingerprint": "e9642259b3f1e5d18db9cd75a3780b1dfdaa8012eb3baa6e1c4e8f4ee597f0a5",
  "split": "dev",
  "task_kind": "swebench",
  "usage_sources": [
    "tokenizer_exact"
  ]
}
```

## model_run · Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 · dev · bfa433b1db

Context: 8192 · total: 32000 · tokenizer: hf:Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554

| Strategy | Success / graded / attempted | Infra errors | Empty patches | Rate | Tokens | Total USD | USD / success | Seconds / task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mask | 0 / 0 / 1 | 0 | 1 | ungraded | 31,686 | — | — | 36.71 |
| recent | 0 / 0 / 1 | 0 | 1 | ungraded | 31,427 | — | — | 118.06 |

```json
{
  "actual_models": [
    "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554"
  ],
  "config": {
    "allow_local_execution": false,
    "api_key_env": "LLM_API_KEY",
    "api_max_tokens_field": "max_tokens",
    "base_url": "",
    "context_tokens": 8192,
    "device": "cuda",
    "docker_image": "python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce",
    "environment": "docker",
    "history_format": "chat",
    "input_price_per_million": null,
    "max_cost_usd": null,
    "max_output_tokens": 768,
    "max_steps": 20,
    "min_action_output_tokens": 128,
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "model_dtype": "bfloat16",
    "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "output_price_per_million": null,
    "provider": "transformers",
    "temperature": 0.0,
    "timeout_seconds": 120,
    "tokenizer": "hf:Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
    "total_tokens": 32000
  },
  "evidence_kind": "model_run",
  "harness": null,
  "model": "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
  "model_metadata": {
    "device": "cuda",
    "dtype": "bfloat16",
    "generation": {
      "bos_token_id": null,
      "do_sample": false,
      "eos_token_id": [
        151645,
        151643
      ],
      "num_beams": 1,
      "pad_token_id": 151643,
      "repetition_penalty": 1.0,
      "temperature": 1.0,
      "top_k": 0,
      "top_p": 1.0,
      "use_cache": true
    },
    "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "seed_scope": "torch RNG reset once at each task/strategy attempt; not a cross-hardware determinism guarantee",
    "tokenizer_revision": "cdbee75f17c01a7cc42f958dc650907174af0554"
  },
  "provider": "transformers",
  "source_fingerprint": "209a8cff1285870eb9456fdb1db5f38c1e02ef73a987b82ab53ffe1961a4ca4f",
  "split": "dev",
  "task_kind": "swebench",
  "usage_sources": [
    "tokenizer_exact"
  ]
}
```

## model_run · Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 · dev · bd7137c67b

Context: 8192 · total: 32000 · tokenizer: hf:Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554

| Strategy | Success / graded / attempted | Infra errors | Empty patches | Rate | Tokens | Total USD | USD / success | Seconds / task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recent | 0 / 0 / 1 | 0 | 1 | ungraded | 31,427 | — | — | 60.95 |

```json
{
  "actual_models": [
    "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554"
  ],
  "config": {
    "allow_local_execution": false,
    "api_key_env": "LLM_API_KEY",
    "api_max_tokens_field": "max_tokens",
    "base_url": "",
    "context_tokens": 8192,
    "device": "cuda",
    "docker_image": "python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce",
    "environment": "docker",
    "history_format": "chat",
    "input_price_per_million": null,
    "max_cost_usd": null,
    "max_output_tokens": 768,
    "max_steps": 20,
    "min_action_output_tokens": 128,
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "model_dtype": "bfloat16",
    "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "output_price_per_million": null,
    "provider": "transformers",
    "temperature": 0.0,
    "timeout_seconds": 120,
    "tokenizer": "hf:Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
    "total_tokens": 32000
  },
  "evidence_kind": "model_run",
  "harness": null,
  "model": "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
  "model_metadata": {
    "device": "cuda",
    "dtype": "bfloat16",
    "generation": {
      "bos_token_id": null,
      "do_sample": false,
      "eos_token_id": [
        151645,
        151643
      ],
      "num_beams": 1,
      "pad_token_id": 151643,
      "repetition_penalty": 1.0,
      "temperature": 1.0,
      "top_k": 0,
      "top_p": 1.0,
      "use_cache": true
    },
    "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "seed_scope": "torch RNG reset once at each task/strategy attempt; not a cross-hardware determinism guarantee",
    "tokenizer_revision": "cdbee75f17c01a7cc42f958dc650907174af0554"
  },
  "provider": "transformers",
  "source_fingerprint": "e9642259b3f1e5d18db9cd75a3780b1dfdaa8012eb3baa6e1c4e8f4ee597f0a5",
  "split": "dev",
  "task_kind": "swebench",
  "usage_sources": [
    "tokenizer_exact"
  ]
}
```

## Run ledger

| Task | Strategy | Agent status | Evaluation status | Verified outcome |
|---|---|---|---|---|
| astropy__astropy-12907 | mask | budget_exhausted | official_empty_patch | ungraded: official empty patch; tests not executed |
| astropy__astropy-12907 | mask | budget_exhausted | official_empty_patch | ungraded: official empty patch; tests not executed |
| astropy__astropy-12907 | mask | budget_exhausted | official_empty_patch | ungraded: official empty patch; tests not executed |
| astropy__astropy-12907 | recent | budget_exhausted | official_empty_patch | ungraded: official empty patch; tests not executed |
| astropy__astropy-12907 | mini-swe-textbased | budget_exhausted | official_empty_patch | ungraded: official empty patch; tests not executed |
| astropy__astropy-12907 | recent | budget_exhausted | official_empty_patch | ungraded: official empty patch; tests not executed |
| astropy__astropy-12907 | recent | budget_exhausted | official_empty_patch | ungraded: official empty patch; tests not executed |

## Run behavior diagnostics

Tool success does not establish a correct repair or a passed test. Repeated observations have identical action, output and recorded revision; repetition does not necessarily indicate an error. unavailable means the result has no stored count. This report does not reanalyze old trajectories.

| Run | Task | Strategy | Successful read | Successful edit + write | Successful run | Successful diff | Invalid actions | Failed action repeats | Identical observation repeats |
|---|---|---|---|---|---|---|---|---|---|
| 20261022T074654Z-43348b7b | astropy__astropy-12907 | mask | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20261022T074532Z-2df7bd07 | astropy__astropy-12907 | mask | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20261022T072704Z-bbb60603 | astropy__astropy-12907 | mask | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20261022T072506Z-86c192b2 | astropy__astropy-12907 | recent | 4 | 0 | 1 | 0 | 0 | 0 | 0 |
| astropy__astropy-12907__mini-swe-textbased | astropy__astropy-12907 | mini-swe-textbased | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 20261022T071042Z-2c771740 | astropy__astropy-12907 | recent | 4 | 0 | 1 | 0 | 0 | 0 | 0 |
| 20261022T071302Z-adc44a6d | astropy__astropy-12907 | recent | 4 | 0 | 1 | 0 | 0 | 0 | 0 |
