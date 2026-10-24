from dataclasses import asdict, dataclass


@dataclass
class PolicySample:
    prompt_ids: list[int]
    completion_ids: list[int]
    old_log_probs: list[float]
    text: str
    policy_version: int


@dataclass
class Episode:
    task_id: str
    split: str
    policy_version: int
    steps: list[PolicySample]
    reward: float
    resolved: bool | None
    status: str
    input_tokens: int
    output_tokens: int
    patch: str
    trace: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingConfig:
    model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    revision: str | None = None
    adapter_path: str | None = None
    device: str = "cuda"
    lora_r: int = 8
    temperature: float = 0.8
    context_tokens: int = 1536
    total_tokens: int = 8192
    max_steps: int = 5
    max_new_tokens: int = 192
    environment: str = "docker"
    docker_image: str = "python:3.11-slim"
    tool_timeout: int = 30
    allow_local_execution: bool = False
    group_size: int = 4
    updates: int = 3
    learning_rate: float = 5e-5
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    seed: int = 42

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("Sampled RL requires temperature > 0")
        if self.group_size < 2:
            raise ValueError("GRPO requires at least two rollouts per task")
        if min(self.context_tokens, self.total_tokens, self.max_steps,
               self.max_new_tokens, self.updates) <= 0:
            raise ValueError("Token budgets, steps and updates must be positive")
        if self.environment not in {"docker", "local"}:
            raise ValueError("environment must be docker or local")
        if self.environment == "local" and not self.allow_local_execution:
            raise ValueError("Local generated code execution requires allow_local_execution=true")
