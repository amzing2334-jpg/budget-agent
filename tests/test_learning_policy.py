"""Real tiny Qwen + LoRA tests; no model downloads or GPU required."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from budget_agent.context import ContextOverflow
from budget_agent.learning.policy import HFPolicy
from budget_agent.learning.schema import PolicySample


@pytest.fixture
def base_path(tmp_path):
    torch.manual_seed(7)
    vocab = {token: i for i, token in enumerate([
        "<pad>", "<bos>", "<eos>", "<unk>", "<user>", "<assistant>",
        "<system>", "<tool>", "hello", "code", "fixed", "broken"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>",
        unk_token="<unk>", pad_token="<pad>",
        chat_template="{% for m in messages %}{{ '<' + m['role'] + '> ' + m['content'] + ' <eos> ' }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}")
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=2,
        pad_token_id=0, attention_dropout=0.0))
    path = tmp_path / "base"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def policy(base_path):
    return HFPolicy(str(base_path), device="cpu", lora_r=2,
                    temperature=0.63, max_context_tokens=64)


def test_completion_alignment_and_prompt_mask(policy):
    messages = [{"role": "user", "content": "hello code"},
                {"role": "tool", "content": "broken"}]
    prompt = policy.encode(messages)
    completion = [10, 2]
    sample = PolicySample(prompt, completion, [], "fixed", 0)
    logits = policy.model(
        input_ids=torch.tensor([prompt + completion]), use_cache=False).logits[0]
    expected_logits = logits[len(prompt) - 1:-1].float()
    targets = torch.tensor(completion)
    expected = (expected_logits / policy.temperature).log_softmax(-1).gather(
        -1, targets[:, None]).squeeze(-1)
    torch.testing.assert_close(policy.log_probs(sample), expected)
    torch.testing.assert_close(policy.supervised_loss(prompt, completion),
                               torch.nn.functional.cross_entropy(expected_logits, targets))
    assert policy.log_probs(sample).shape == (len(completion),)


def test_sample_log_probs_match_differentiable_scoring(policy):
    torch.manual_seed(41)
    sample = policy.sample([{"role": "user", "content": "hello code"}], 5)
    assert sample.prompt_ids == policy.encode([{"role": "user", "content": "hello code"}])
    assert sample.policy_version == 0
    assert 1 <= len(sample.completion_ids) <= 5
    torch.testing.assert_close(policy.log_probs(sample), torch.tensor(sample.old_log_probs))
    assert policy.log_probs(sample).requires_grad


def test_eos_is_included_and_budget_does_not_clip(policy, monkeypatch):
    monkeypatch.setattr(torch, "multinomial", lambda probabilities, count:
                        torch.tensor([2], device=probabilities.device))
    sample = policy.sample([{"role": "user", "content": "hello"}], 5)
    assert sample.completion_ids == [2]
    assert len(sample.old_log_probs) == 1
    assert sample.text == ""
    with pytest.raises(ContextOverflow):
        policy.sample([{"role": "user", "content": "hello"}], 64)


def test_adapter_updates_frozen_base_and_reloads(policy, base_path, tmp_path):
    prompt, completion = policy.encode([{"role": "user", "content": "broken"}]), [10, 2]
    sample = PolicySample(prompt, completion, [], "fixed", 0)
    before = policy.log_probs(sample).detach().clone()
    frozen = {name: value.detach().clone() for name, value in policy.model.named_parameters()
              if not value.requires_grad}
    assert frozen
    assert all("lora_" in name for name, value in policy.model.named_parameters() if value.requires_grad)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.05)
    policy.supervised_loss(prompt, completion).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    optimizer.step()
    for name, value in policy.model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    after = policy.log_probs(sample).detach()
    assert not torch.allclose(before, after)
    adapter = tmp_path / "adapter"
    policy.version = 3
    policy.save(adapter)
    assert json.loads((adapter / "policy_state.json").read_text(encoding="utf-8")) == {
        "version": 3, "temperature": 0.63, "max_context_tokens": 64}
    loaded = HFPolicy(str(base_path), device="cpu", temperature=policy.temperature,
                      max_context_tokens=64, adapter_path=str(adapter))
    torch.testing.assert_close(loaded.log_probs(sample), after)
    assert loaded.version == 3
    assert list(loaded.parameters())
    assert loaded.encode([{"role": "user", "content": "broken"}]) == prompt


def test_components_keep_policy_version(policy):
    constructed = HFPolicy.from_components(
        policy.model, policy.tokenizer, temperature=0.63, max_context_tokens=64, version=3)
    sample = constructed.sample([{"role": "user", "content": "hello"}], 1)
    assert sample.policy_version == 3
