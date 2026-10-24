"""A trainable Qwen policy with exact per-decision action probabilities.

Requires a causal model implementing Transformers' ``logits_to_keep`` argument.
Only completion tokens enter the loss; every decision keeps its actual context.
"""
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..context import ContextOverflow
from .schema import PolicySample


class HFPolicy:
    def __init__(self, model_name: str, revision: str | None = None,
                 device: str = "cuda", lora_r: int = 8, temperature: float = 0.8,
                 max_context_tokens: int = 2048, adapter_path: str | None = None):
        version = json.loads((Path(adapter_path) / "policy_state.json").read_text(
            encoding="utf-8"))["version"] if adapter_path else 0
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path or model_name, revision=None if adapter_path else revision)
        base = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision,
            dtype=torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32)
        if adapter_path:
            model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        else:
            model = get_peft_model(base, LoraConfig(
                task_type="CAUSAL_LM", r=lora_r, lora_alpha=2 * lora_r,
                lora_dropout=0.0, target_modules="all-linear"))
        self._configure(model.to(device), tokenizer, temperature, max_context_tokens, version)

    @classmethod
    def from_components(cls, model, tokenizer, temperature: float = 0.8,
                        max_context_tokens: int = 2048, version: int = 0):
        """Use an already constructed model and tokenizer, including local tests."""
        policy = cls.__new__(cls)
        policy._configure(model, tokenizer, temperature, max_context_tokens, version)
        return policy

    def _configure(self, model, tokenizer, temperature, max_context_tokens, version):
        if temperature <= 0:
            raise ValueError("Sampled RL requires temperature > 0")
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_context_tokens = max_context_tokens
        self.version = version
        # eval disables dropout, while gradients remain available for updates.
        self.model.eval()

    def encode(self, messages: list[dict[str, str]]) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False)

    encode_messages = encode

    def _check_length(self, prompt_ids, completion_length):
        if not prompt_ids or completion_length < 1:
            raise ValueError("A decision needs a nonempty prompt and completion")
        if len(prompt_ids) + completion_length > self.max_context_tokens:
            raise ContextOverflow("Prompt and completion exceed the policy context budget")

    @torch.inference_mode()
    def sample(self, messages: list[dict[str, str]], max_new_tokens: int) -> PolicySample:
        prompt_ids = self.encode(messages)
        self._check_length(prompt_ids, max_new_tokens)
        self.model.eval()
        input_ids = torch.tensor([prompt_ids], device=self.model.device)
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        completion_ids, old_log_probs = [], []
        cache = None
        for _ in range(max_new_tokens):
            output = self.model(input_ids=input_ids, past_key_values=cache,
                                use_cache=True, logits_to_keep=1)
            log_probs = (output.logits[0, -1].float() / self.temperature).log_softmax(-1)
            token = torch.multinomial(log_probs.exp(), 1)
            completion_ids.append(token.item())
            old_log_probs.append(log_probs[token].item())
            if token.item() in eos_ids:
                break
            input_ids = token.reshape(1, 1)
            cache = output.past_key_values
        return PolicySample(
            prompt_ids, completion_ids, old_log_probs,
            self.tokenizer.decode(completion_ids, skip_special_tokens=True), self.version)

    def _completion_log_probs(self, prompt_ids, completion_ids, temperature):
        self._check_length(prompt_ids, len(completion_ids))
        self.model.eval()
        # The final prompt token predicts the first completion token. Restricting
        # output logits avoids materializing the large vocabulary at prompt positions.
        input_ids = torch.tensor([prompt_ids + completion_ids[:-1]], device=self.model.device)
        logits = self.model(input_ids=input_ids, use_cache=False,
                            logits_to_keep=len(completion_ids)).logits[0]
        targets = torch.tensor(completion_ids, device=logits.device)
        return (logits.float() / temperature).log_softmax(-1).gather(
            -1, targets[:, None]).squeeze(-1)

    def log_probs(self, sample: PolicySample) -> torch.Tensor:
        """Differentiable action log probabilities under the sampling temperature."""
        return self._completion_log_probs(
            sample.prompt_ids, sample.completion_ids, self.temperature)

    def supervised_loss(self, prompt_ids: list[int], completion_ids: list[int]) -> torch.Tensor:
        """Standard, untempered SFT cross entropy, with prompt/tool tokens masked."""
        return -self._completion_log_probs(prompt_ids, completion_ids, 1.0).mean()

    def parameters(self):
        return (parameter for parameter in self.model.parameters() if parameter.requires_grad)

    def save(self, path: str | Path) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (Path(path) / "policy_state.json").write_text(json.dumps({
            "version": self.version, "temperature": self.temperature,
            "max_context_tokens": self.max_context_tokens,
        }, indent=2) + "\n", encoding="utf-8")
