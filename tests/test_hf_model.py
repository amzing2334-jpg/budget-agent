"""Backend boundary tests use doubles only: no model weights or torch imports."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import pytest

from budget_agent.budget import Budget
from budget_agent.context import ContextManager
from budget_agent.hf_model import HFTokenCounter, TransformersModel
from budget_agent.schema import Event, RunConfig


REVISION = "a" * 40


class Tokenizer:
    bos_token_id = None
    eos_token_id = 99
    pad_token_id = None

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode())

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert tokenize and add_generation_prompt and not return_dict
        # Transformers inspects conversation[0]; an empty conversation is invalid.
        if not messages:
            raise IndexError("list index out of range")
        return [1] + [n for message in messages for n in [2, *message["content"].encode(), 3]] + [4]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return bytes(i for i in ids if i != self.eos_token_id).decode()


class Tensor:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        return SimpleNamespace(tolist=lambda: self.rows[index])


@pytest.fixture
def backend(monkeypatch):
    tokenizer = Tokenizer()
    calls = []
    seeds = []

    class Model:
        # A non-default checkpoint sampler must not silently control this run.
        generation_config = SimpleNamespace(eos_token_id=99, top_p=0.2, top_k=7)
        device = "cuda"

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls.append(kwargs)
            output = [ord("x"), 99][:kwargs["max_new_tokens"]]
            return Tensor([kwargs["input_ids"].rows[0] + output])

    loaded = []

    def load_model(name, **kwargs):
        loaded.append((name, kwargs))
        return Model()

    fake_torch = SimpleNamespace(
        bfloat16="bf16", manual_seed=seeds.append,
        tensor=lambda rows, device: Tensor(rows), ones_like=lambda x: Tensor([[1] * len(x.rows[0])]),
        inference_mode=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda name, revision: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    ))
    config = RunConfig(provider="transformers", model="test/model", model_revision=REVISION,
                       context_tokens=256, total_tokens=512, max_output_tokens=8, min_action_output_tokens=8)
    return TransformersModel(config), config, calls, seeds, loaded


def test_exact_chat_template_and_eos_accounting(backend):
    model, config, calls, seeds, loaded = backend
    messages = [{"role": "user", "content": "repair"}]
    budget = Budget(model.counter, config)
    allowance = budget.reserve(messages, 8)
    response = model.complete(messages, allowance)
    budget.record(response)
    assert response.input_tokens == len(calls[0]["input_ids"].rows[0]) == model.counter.messages(messages)
    assert response.content == "x" and response.output_tokens == 2  # EOS is charged.
    assert response.usage_source == "tokenizer_exact"
    assert budget.to_dict()["usage_is_estimated"] is False
    assert loaded == [("test/model", {"revision": REVISION, "dtype": "bf16"})]
    generation = calls[0]["generation_config"]
    assert not generation.do_sample
    assert generation.top_p == 1.0 and generation.top_k == 0
    assert generation.repetition_penalty == 1.0


def test_remaining_budget_sets_actual_generation_cap(backend):
    model, config, calls, seeds, loaded = backend
    messages = [{"role": "user", "content": "repair"}]
    config.total_tokens = model.counter.messages(messages) + 1
    budget = Budget(model.counter, config)
    response = model.complete(messages, budget.reserve(messages, 8), purpose="summary")
    budget.record(response, purpose="summary")
    assert calls[0]["max_new_tokens"] == response.output_tokens == 1
    assert budget.remaining_tokens == 0
    assert budget.records[0]["purpose"] == "summary"


def test_task_seed_reset_is_explicit(backend):
    model, _, _, seeds, _ = backend
    model.reset_seed(42)
    model.reset_seed(42)
    assert seeds == [42, 42]


@pytest.mark.parametrize("strategy", ["recent", "summary", "structured"])
def test_compaction_uses_nonempty_hf_conversations(strategy):
    counter = HFTokenCounter(Tokenizer(), "test/model", REVISION)
    manager = ContextManager(strategy, counter, history_format="transcript")
    events = [
        Event("e1", 1, {"tool": "run"}, "setup logs\n" * 500 + "AssertionError: failed", "r1", False),
        Event("e2", 2, {"tool": "read_file", "path": "a.py"}, "current code", "r1", True),
    ]
    summary_calls = []

    def summarize(messages, max_output_tokens):
        assert counter.messages(messages) <= 1100
        summary_calls.append(messages)
        return "Event e1 failed at revision r1; verify after editing."

    messages = manager.build("system", "preserve the complete issue 中文", events, "r1", 1100, summarize)
    assert manager.stats["compactions"] == 1
    assert counter.messages(messages) <= 1100
    assert messages[1]["content"].endswith("preserve the complete issue 中文")
    assert bool(summary_calls) == (strategy == "summary")
    assert events[0].output.endswith("AssertionError: failed")


@pytest.mark.parametrize("revision", ["", "main", "latest", "a" * 39])
def test_transformers_requires_immutable_revision(revision):
    config = RunConfig(provider="transformers", model="test/model", model_revision=revision)
    with pytest.raises(ValueError, match="model_revision"):
        config.validate()
