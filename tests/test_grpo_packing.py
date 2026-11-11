"""Prefix-packed GRPO sequences give the same log probabilities and gradients as per-turn forwards.

Needs the training stack (torch, transformers, peft), so it is skipped in the
lightweight test environment and runs on the GPU host's CPU.
"""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from budget_agent.learning.grpo_core import Turn  # noqa: E402

SPEC = importlib.util.spec_from_file_location("grpo_harness", Path(__file__).parents[1] / "scripts/grpo_harness.py")
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def tiny_policy():
    torch.manual_seed(0)
    config = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    model = peft.get_peft_model(transformers.Qwen2ForCausalLM(config).float(),
                                peft.LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"))
    with torch.no_grad():  # a non-zero adapter, so the LoRA path carries signal too
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0.0, 0.02)
    return model


def attempt(break_prefix=False):
    generator = torch.Generator().manual_seed(1)
    ids = lambda n: torch.randint(0, 64, (n,), generator=generator).tolist()  # noqa: E731
    p0, c0 = ids(10), ids(4)
    p1, c1 = p0 + c0 + ids(5), ids(3)
    p2, c2 = p1 + c1 + ids(4), ids(5)
    if break_prefix:
        p2 = [63] + p2[1:]  # the history was cut: no longer an extension of turn 1
    return [(p0, Turn([], c0, [-0.5] * 4, None)), (p1, Turn([], c1, [-0.4] * 3, None)), (p2, Turn([], c2, [-0.3] * 5, None))]


def test_a_prefix_chain_matches_per_turn_forwards_in_logprobs_and_gradients():
    model = tiny_policy()
    packed = harness.build_sequences(attempt(), 0.7, pack=True)
    separate = harness.build_sequences(attempt(), 0.7, pack=False)
    assert [s.turns for s in packed] == [3] and len(separate) == 3
    assert packed[0].behavior == [-0.5] * 4 + [-0.4] * 3 + [-0.3] * 5
    assert len(packed[0].inputs) < sum(len(s.inputs) for s in separate)
    trainable = [p for p in model.parameters() if p.requires_grad]
    logp_packed, _ = harness.sequence_logprobs(model, packed[0])
    logp_separate = torch.cat([harness.sequence_logprobs(model, s)[0] for s in separate])
    assert torch.allclose(logp_packed, logp_separate, atol=1e-5)
    grads_packed = torch.autograd.grad(logp_packed.sum(), trainable)
    grads_separate = torch.autograd.grad(logp_separate.sum(), trainable)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(grads_packed, grads_separate, strict=True))


def test_a_broken_prefix_starts_a_new_sequence():
    packed = harness.build_sequences(attempt(break_prefix=True), 0.7, pack=True)
    assert [s.turns for s in packed] == [2, 1]
