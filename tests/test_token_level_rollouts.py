"""Token-level rollout records: exact sampled IDs and behaviour log probabilities."""
import json
from pathlib import Path

import pytest

from budget_agent.budget import TokenCounter
from budget_agent.models import ModelError, OpenAICompatibleModel, token_digest
from budget_agent.runner import run_task
from budget_agent.schema import ModelResponse, RunConfig
from budget_agent.tasks import load_tasks
from test_models import server  # noqa: F401 (pytest fixture)


ROOT = Path(__file__).resolve().parents[1]
CONTENT = '{"tool":"finish"}'
PROMPT_IDS = list(range(100, 130))


def vllm_body(ids, logprobs, completion_tokens=None, finish="stop"):
    return {"model": "student", "usage": {"prompt_tokens": 30, "completion_tokens": completion_tokens or len(ids)},
            "prompt_token_ids": PROMPT_IDS,
            "choices": [{"message": {"content": CONTENT}, "finish_reason": finish, "token_ids": list(ids),
                         "logprobs": {"content": [{"token": f"token_id:{i}", "logprob": lp, "bytes": None,
                                                   "top_logprobs": []} for i, lp in zip(ids, logprobs, strict=True)]}}]}


def client(server, logprobs):  # noqa: F811 (fixture argument)
    return OpenAICompatibleModel(RunConfig(model="student", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                                           api_logprobs=logprobs))


def test_default_requests_and_records_are_unchanged(server):  # noqa: F811
    response = client(server, False).complete([{"role": "user", "content": "x"}], 16)
    assert not {"logprobs", "return_tokens_as_token_ids", "return_token_ids"} & set(server.requests[0])
    assert response.completion_token_ids is None and response.completion_logprobs is None
    assert response.prompt_token_digest is None


def test_token_ids_and_logprobs_are_parsed_exactly(server):  # noqa: F811
    server.body = vllm_body([123, 9, 151645], [-0.5, -0.01, -0.2])
    response = client(server, True).complete([{"role": "user", "content": "x"}], 16)
    body = server.requests[0]
    assert body["logprobs"] is True and body["return_tokens_as_token_ids"] is True and body["return_token_ids"] is True
    assert response.completion_token_ids == [123, 9, 151645]
    assert response.completion_logprobs == [-0.5, -0.01, -0.2]
    assert response.finish_reason == "stop"
    assert response.prompt_token_digest == token_digest(PROMPT_IDS)


def test_token_digest_is_fixed_across_platforms_and_versions():
    # Rollout-time and train-time digests must agree; changing this breaks old runs.
    assert token_digest([1, 2, 3]) == "b30a3c159d7e8193559a42a84f1fcae1"
    assert token_digest([1, 2, 3]) != token_digest([1, 2, 4]) != token_digest([1, 2])


@pytest.mark.parametrize("mutate", [
    lambda body: body["choices"][0].pop("logprobs"),
    lambda body: body["choices"][0]["logprobs"]["content"][0].update(token="hello"),
    lambda body: body["usage"].update(completion_tokens=4),
    lambda body: body["choices"][0].update(token_ids=[1, 2, 4]),
    lambda body: body.pop("prompt_token_ids"),
    lambda body: body.update(prompt_token_ids=PROMPT_IDS[:-1]),
    lambda body: body.update(prompt_token_ids=PROMPT_IDS[:-1] + [-1]),
], ids=["missing", "text-tokens", "usage-mismatch", "ids-disagree", "no-prompt-ids", "prompt-short", "prompt-negative"])
def test_incomplete_token_records_fail_loudly(server, mutate):  # noqa: F811
    body = vllm_body([1, 2, 3], [-1.0, -1.0, -1.0])
    mutate(body)
    server.body = body
    with pytest.raises(ModelError, match="token ID and log probability"):
        client(server, True).complete([{"role": "user", "content": "x"}], 16)


def test_runner_persists_token_records_only_when_present(tmp_path):
    task = load_tasks(ROOT / "tasks/local.jsonl", task_ids=["slugify-digits"])[0]

    class Recording:
        name = "student"

        def __init__(self, ids):
            self.ids, self.counter = ids, TokenCounter()

        def complete(self, messages, max_output_tokens, *, purpose="agent"):
            return ModelResponse(CONTENT, self.counter.messages(messages), 3, "provider", "student",
                                 completion_token_ids=self.ids, completion_logprobs=[-0.1] * 3 if self.ids else None)

    config = RunConfig(provider="openai-compatible", model="student", base_url="http://127.0.0.1:9/v1",
                       allow_local_execution=True)
    recorded = run_task(task, config, tmp_path / "a", model=Recording([5, 6, 7]))
    plain = run_task(task, config, tmp_path / "b", model=Recording(None))
    calls = [json.loads(Path(r["result_path"]).with_name("calls.jsonl").read_text().splitlines()[0]) for r in (recorded, plain)]
    assert calls[0]["response"]["completion_token_ids"] == [5, 6, 7]
    assert set(calls[1]["response"]) == {"content", "input_tokens", "output_tokens", "usage_source", "model"}
