"""Pinned local Transformers inference; no adapters or optimizer are constructed.

Imports stay inside model construction so the ordinary CPU test environment does
not need torch or transformers. Prompt counting and generation share one encoder.
"""
from __future__ import annotations

from .schema import ModelResponse, RunConfig


class HFTokenCounter:
    def __init__(self, tokenizer, model: str, revision: str):
        self.tokenizer = tokenizer
        self.name = f"hf:{model}@{revision}"

    def encode_messages(self, messages: list[dict[str, str]]) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def messages(self, messages: list[dict[str, str]]) -> int:
        return len(self.encode_messages(messages))

    @property
    def description(self) -> str:
        return "Exact input IDs from the same pinned tokenizer and chat template used for generation"


class TransformersModel:
    def __init__(self, config: RunConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        config.validate()
        self.torch = torch
        self.name = f"{config.model}@{config.model_revision}"
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.model_revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model, revision=config.model_revision,
            dtype=getattr(torch, config.model_dtype)).to(config.device).eval()
        self.counter = HFTokenCounter(self.tokenizer, config.model, config.model_revision)
        # Use checkpoint stop IDs, but do not inherit sampling/penalty defaults.
        generation = {
            "do_sample": config.temperature > 0,
            "temperature": config.temperature if config.temperature > 0 else 1.0,
            "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0,
            "num_beams": 1, "use_cache": True,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.model.generation_config.eos_token_id,
            "pad_token_id": (self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                             else self.tokenizer.eos_token_id),
        }
        self.generation_config = GenerationConfig(**generation)
        self.metadata = {
            "model_revision": config.model_revision,
            "tokenizer_revision": config.model_revision,
            "device": config.device, "dtype": config.model_dtype,
            "generation": generation,
            "seed_scope": "torch RNG reset once at each task/strategy attempt; not a cross-hardware determinism guarantee",
        }

    def reset_seed(self, seed: int) -> None:
        self.torch.manual_seed(seed)

    def complete(self, messages: list[dict[str, str]], max_output_tokens: int,
                 *, purpose: str = "agent") -> ModelResponse:
        prompt_ids = self.counter.encode_messages(messages)
        inputs = self.torch.tensor([prompt_ids], device=self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                input_ids=inputs, attention_mask=self.torch.ones_like(inputs),
                generation_config=self.generation_config, max_new_tokens=max_output_tokens)
        completion_ids = generated[0].tolist()[len(prompt_ids):]
        return ModelResponse(
            self.tokenizer.decode(completion_ids, skip_special_tokens=True),
            len(prompt_ids), len(completion_ids), "tokenizer_exact", self.name)
