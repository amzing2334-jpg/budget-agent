"""Continuing GRPO from a saved adapter: the loaded LoRA is trainable and reproduces the saved policy.

Needs the training stack (torch, transformers, peft); it runs in the full-suite CI job.
"""
import copy
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

SPEC = importlib.util.spec_from_file_location("grpo_harness", Path(__file__).parents[1] / "scripts/grpo_harness.py")
assert SPEC is not None and SPEC.loader is not None
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def tiny_base():
    torch.manual_seed(0)
    config = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    return transformers.Qwen2ForCausalLM(config).float().eval()


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_a_fresh_lora_leaves_the_base_model_unchanged():
    base = tiny_base()
    ids = torch.tensor([[1, 5, 9, 2, 7]])
    reference = logits(base, ids)
    policy = harness.attach_lora(copy.deepcopy(base), rank=4)
    assert torch.equal(logits(policy, ids), reference)
    assert any(p.requires_grad for name, p in policy.named_parameters() if "lora_" in name)


def test_a_saved_adapter_reloads_trainable_and_reproduces_the_trained_policy(tmp_path):
    base = tiny_base()
    ids = torch.tensor([[3, 1, 4, 1, 5, 9, 2]])
    trained = harness.attach_lora(copy.deepcopy(base), rank=4)
    with torch.no_grad():  # stand-in for training: a non-zero adapter
        for name, param in trained.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.1)
    trained.save_pretrained(tmp_path / "v29")
    resumed = harness.attach_lora(copy.deepcopy(base), rank=4, init_adapter=tmp_path / "v29")
    assert torch.allclose(logits(resumed, ids), logits(trained, ids), atol=1e-6)
    assert not torch.allclose(logits(resumed, ids), logits(base, ids), atol=1e-3)
    trainable = [name for name, p in resumed.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)
