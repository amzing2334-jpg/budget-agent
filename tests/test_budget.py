import pytest

from budget_agent.budget import Budget, BudgetExceeded, TokenCounter
from budget_agent.schema import ModelResponse, RunConfig


def config(**kwargs):
    return RunConfig(context_tokens=1000, total_tokens=2000, max_output_tokens=100, **kwargs)


def test_byte_counts_non_ascii_bytes_and_message_framing():
    counter = TokenCounter()
    assert counter.count("你好") == 6
    assert counter.messages([{"role": "user", "content": "你好"}]) > 6
    assert "estimate" in counter.description


def test_total_cap_includes_repeated_inputs_and_summaries():
    counter = TokenCounter()
    cfg = RunConfig(context_tokens=300, total_tokens=160, max_output_tokens=100)
    budget = Budget(counter, cfg)
    messages = [{"role": "user", "content": "x"}]
    inputs = counter.messages(messages)
    cap = budget.reserve(messages, 100)
    budget.record(ModelResponse("", inputs, cap, "scripted_estimate"), "summary")
    remaining = 160 - inputs - cap
    cap2 = budget.reserve(messages, 100)
    assert cap2 == remaining - inputs
    budget.record(ModelResponse("", inputs, cap2, "scripted_estimate"))
    assert budget.remaining_tokens == 0
    assert budget.calls == 2
    assert budget.records[0]["purpose"] == "summary"
    with pytest.raises(BudgetExceeded):
        budget.reserve(messages, 1)


def test_unknown_prices_are_not_zero_cost():
    budget = Budget(TokenCounter(), config())
    assert budget.cost_usd is None
    assert budget.to_dict()["cost_kind"] == "unknown_prices"


def test_cost_reservation_counts_input_and_caps_output():
    counter = TokenCounter()
    budget = Budget(counter, config(input_price_per_million=1, output_price_per_million=2, max_cost_usd=0.000045))
    messages = [{"role": "user", "content": "x"}]
    inputs = counter.messages(messages)
    cap = budget.reserve(messages, 100)
    assert cap == 10
    budget.record(ModelResponse("", inputs, cap))
    assert budget.cost_usd == pytest.approx(0.000045)
    with pytest.raises(BudgetExceeded):
        budget.reserve(messages, 1)


def test_provider_overrun_is_recorded_and_halts_followup():
    budget = Budget(TokenCounter(), config())
    budget.reserve([{"role": "user", "content": "x"}], 10)
    with pytest.raises(BudgetExceeded, match="output"):
        budget.record(ModelResponse("", 25, 11))
    assert budget.output_tokens == 11
    assert budget.calls == 1
    assert budget.to_dict()["overrun"]
    with pytest.raises(BudgetExceeded, match="Previous"):
        budget.reserve([], 1)


def test_context_cap_and_pending_request_are_enforced():
    counter = TokenCounter()
    budget = Budget(counter, config())
    messages = [{"role": "user", "content": "x" * 970}]
    assert budget.reserve(messages, 100) == 6
    with pytest.raises(RuntimeError, match="already reserved"):
        budget.reserve(messages, 1)


def test_invalid_usage_not_silently_accepted():
    budget = Budget(TokenCounter(), config())
    budget.reserve([], 10)
    with pytest.raises(ValueError):
        budget.record(ModelResponse("", -1, 3))
    assert budget.to_dict()["unreconciled_request"]
    assert budget.to_dict()["attempted_calls"] == 1
    assert budget.to_dict()["pending_reservation"] == {"estimated_input_tokens": 8, "reserved_output_tokens": 10}


def test_reservation_reconciles_lower_provider_usage():
    budget = Budget(TokenCounter(), config())
    budget.reserve([{"role": "user", "content": "x" * 100}], 100)
    budget.record(ModelResponse("ok", 30, 2))
    assert budget.remaining_tokens == 1968
    assert budget.to_dict()["usage_is_estimated"] is False


def test_explicit_tiktoken_counts_text_without_treating_special_strings_as_control():
    pytest.importorskip("tiktoken")
    counter = TokenCounter("cl100k_base")
    text = "中文 code 🙂 <|endoftext|>"
    assert 0 < counter.count(text) < len(text.encode("utf-8"))
    assert counter.messages([{"role": "user", "content": text}]) > counter.count(text)
    assert "framing estimate" in counter.description


@pytest.mark.parametrize("ceiling", ["total", "context", "cost"])
def test_minimum_output_reservation_stops_before_request(ceiling):
    counter = TokenCounter()
    messages = [{"role": "user", "content": "x"}]
    inputs = counter.messages(messages)
    cfg = config()
    if ceiling == "total":
        cfg.total_tokens = inputs + 49
    elif ceiling == "context":
        cfg.context_tokens = inputs + 49
    else:
        cfg.input_price_per_million = 0
        cfg.output_price_per_million = 1
        cfg.max_cost_usd = 0.000049
    budget = Budget(counter, cfg)
    with pytest.raises(BudgetExceeded, match="requires at least 50"):
        budget.reserve(messages, 100, min_output_tokens=50)
    assert budget.to_dict()["attempted_calls"] == 0
    assert not budget.to_dict()["unreconciled_request"]
    # A lower floor is an explicit experiment choice, not an automatic retry.
    assert budget.reserve(messages, 100, min_output_tokens=1) == 49


def test_minimum_reservation_does_not_require_long_generation():
    counter = TokenCounter()
    messages = [{"role": "user", "content": "x"}]
    budget = Budget(counter, config())
    assert budget.reserve(messages, 100, min_output_tokens=50) == 100
    budget.record(ModelResponse("ok", counter.messages(messages), 2))
    assert budget.output_tokens == 2


@pytest.mark.parametrize("minimum", [0, -1, 101])
def test_invalid_minimum_reservation_is_rejected(minimum):
    with pytest.raises(ValueError, match="min_output_tokens"):
        Budget(TokenCounter(), config()).reserve([], 100, min_output_tokens=minimum)



def test_tokenizer_counts_are_memoized_without_changing_results(monkeypatch):
    class Encoding:
        calls = 0

        def encode(self, text, disallowed_special=()):
            Encoding.calls += 1
            return text.split()

    counter = TokenCounter()
    counter._encoding = Encoding()
    monkeypatch.setattr(TokenCounter, "_CACHE_ENTRIES", 2)
    assert counter.count("a b c") == 3 and counter.count("a b c") == 3
    assert Encoding.calls == 1
    assert counter.count("lone \udcff surrogate") == 3  # digest keys accept any str
    assert counter.count("third entry") == 2  # clears the full cache instead of growing
    assert len(counter._counts) == 1
    assert counter.count("a b c") == 3 and Encoding.calls == 4


def test_byte_counter_has_no_cache_overhead():
    counter = TokenCounter()
    assert counter.count("abc") == 3 and counter._counts == {}
