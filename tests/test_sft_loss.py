"""The SFT loss projects only trained positions and still equals the model's own loss.

Needs the training stack (torch, transformers, peft), so it is skipped in the
lightweight test environment and runs on the GPU host's CPU.
"""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

SPEC = importlib.util.spec_from_file_location("train_sft_lora", Path(__file__).parents[1] / "scripts/train_sft_lora.py")
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


def tiny_model(with_lora):
    torch.manual_seed(0)
    config = transformers.Qwen2Config(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = transformers.Qwen2ForCausalLM(config).float()
    if with_lora:
        model = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, target_modules="all-linear",
                                                           task_type="CAUSAL_LM"))
    return model


@pytest.mark.parametrize("with_lora", [False, True])
def test_assistant_loss_matches_the_full_vocabulary_loss_and_gradients(with_lora):
    model = tiny_model(with_lora)
    ids = torch.randint(0, 96, (40,), generator=torch.Generator().manual_seed(1)).tolist()
    labels = [-100] * 20 + ids[20:26] + [-100] * 8 + ids[34:38] + [-100] * 2  # two trained spans
    trainable = [p for p in model.parameters() if p.requires_grad]
    expected = model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels])).loss
    want = torch.autograd.grad(expected, trainable)
    got = train.assistant_loss(model, ids, labels)
    grads = torch.autograd.grad(got, trainable)
    assert torch.allclose(got, expected, atol=1e-6)
    assert all(torch.allclose(g, w, atol=1e-6) for g, w in zip(grads, want, strict=True))
