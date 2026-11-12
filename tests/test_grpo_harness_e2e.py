"""End to end on CPU: the GRPO harness trains a tiny model from fake rollouts, synchronously and pipelined.

A stub stands in for the vLLM server (it records adapter loads and unloads) and a fake pool
launcher writes token-level attempt records the way rollout_pool.py does, with prompt digests
from the same tokenizer and chat template. Needs the training stack; runs in the full-suite job.
"""
import importlib.util
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import threading

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("grpo_harness", ROOT / "scripts/grpo_harness.py")
assert SPEC is not None and SPEC.loader is not None
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)

TEMPLATE = ("{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
            "{% if add_generation_prompt %}<assistant>{% endif %}")
LAUNCHER = """#!{python}
# Fake rollout pool: one agent call per attempt, alternating rewards inside each group.
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, {src!r})
from transformers import AutoTokenizer
from budget_agent.models import token_digest
parser = argparse.ArgumentParser()
for name in ("--config", "--tasks", "--split", "--samples", "--workers", "--output"):
    parser.add_argument(name)
parser.add_argument("--set", action="append", default=[])
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(os.environ["TINY_MODEL"])
out = Path(args.output)
out.mkdir(parents=True)
policy = dict(item.split("=", 1) for item in args.set)["model"]
(out / "policy.txt").write_text(policy)
rows = []
for task in map(json.loads, open(args.tasks)):
    for sample in range(int(args.samples)):
        run = out / "runs" / f"s{{sample:02d}}" / f"{{task['id']}}__run"
        run.mkdir(parents=True)
        messages = [{{"role": "system", "content": "fix the bug"}}, {{"role": "user", "content": "task " + task["id"]}}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        completion = tokenizer.encode("ok done>", add_special_tokens=False)
        response = {{"content": "ok done", "completion_token_ids": completion,
                     "completion_logprobs": [-1.5] * len(completion), "prompt_token_digest": token_digest(prompt)}}
        (run / "calls.jsonl").write_text(json.dumps({{"purpose": "agent", "messages": messages, "response": response}}) + "\\n")
        (run / "result.json").write_text("{{}}")
        rows.append({{"task_id": task["id"], "sample": sample, "split": "train", "resolved": sample % 2 == 0,
                      "result_path": str(run / "result.json")}})
(out / "attempts.jsonl").write_text("".join(json.dumps(row) + "\\n" for row in rows))
"""


class StubServer(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):  # noqa: N802 (http.server API)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubServer.requests.append((self.path.rsplit("/", 1)[-1], body["lora_name"]))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


def tiny_model(path):
    chars = sorted(set("abcdefghijklmnopqrstuvwxyz0123456789<> _-.T"))
    vocab = {ch: i for i, ch in enumerate(chars)}
    vocab["[UNK]"] = len(vocab)
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Split(tokenizers.Regex("."), behavior="isolated")
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token=">",
                                                pad_token="[UNK]")
    fast.chat_template = TEMPLATE
    fast.save_pretrained(path)
    torch.manual_seed(0)
    config = transformers.Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    transformers.Qwen2ForCausalLM(config).save_pretrained(path)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    tiny_model(tmp_path / "model")
    monkeypatch.setenv("TINY_MODEL", str(tmp_path / "model"))
    launcher = tmp_path / "launcher.py"
    launcher.write_text(LAUNCHER.format(python=sys.executable, src=str(ROOT / "src")))
    launcher.chmod(0o755)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("".join(json.dumps({"id": f"t{i}", "issue": "fix it", "split": "train", "kind": "local"}) + "\n"
                                for i in range(4)))
    server = HTTPServer(("127.0.0.1", 0), StubServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    StubServer.requests = []
    yield tmp_path, launcher, manifest, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def run_harness(tmp_path, launcher, manifest, url, monkeypatch, *extra):
    (tmp_path / "rollouts").mkdir()
    monkeypatch.setattr(sys, "argv", [
        "grpo_harness", "--base-model", str(tmp_path / "model"), "--tasks", str(manifest), "--config", "unused.toml",
        "--output", str(tmp_path / "out"), "--rollouts", str(tmp_path / "rollouts"), "--pool-launcher", str(launcher),
        "--server", url, "--steps", "3", "--tasks-per-step", "2", "--group", "2", "--workers", "1",
        "--max-len", "256", "--device", "cpu", *extra])
    assert harness.main() == 0
    records = [json.loads(line) for line in (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()]
    policies = [(tmp_path / "rollouts" / f"step-{step:04d}" / "policy.txt").read_text() for step in range(3)]
    return records, policies


def test_synchronous_steps_train_and_publish_each_new_policy(setup, monkeypatch):
    records, policies = run_harness(*setup, monkeypatch)
    assert [r["step"] for r in records] == [0, 1, 2] and [r["behaviour_policy"] for r in records] == [0, 1, 2]
    assert policies == ["policy-v0", "policy-v1", "policy-v2"]
    assert all(r["prompt_mismatch"] == 0 and r["turns"] == 4 for r in records)
    assert StubServer.requests == [("load_lora_adapter", "policy-v0"),
                                   ("load_lora_adapter", "policy-v1"), ("unload_lora_adapter", "policy-v0"),
                                   ("load_lora_adapter", "policy-v2"), ("unload_lora_adapter", "policy-v1"),
                                   ("load_lora_adapter", "policy-v3"), ("unload_lora_adapter", "policy-v2")]
    assert all((setup[0] / "out" / "adapters" / f"v{k}" / "adapter_config.json").exists() for k in range(4))


def test_pipelined_steps_sample_with_the_previous_policy_and_keep_it_loaded(setup, monkeypatch):
    records, policies = run_harness(*setup, monkeypatch, "--pipeline")
    assert [r["behaviour_policy"] for r in records] == [0, 0, 1]
    assert policies == ["policy-v0", "policy-v0", "policy-v1"]
    assert all(r["prompt_mismatch"] == 0 for r in records)
    # A policy is unloaded only once no running rollout samples from it.
    assert StubServer.requests == [("load_lora_adapter", "policy-v0"),
                                   ("load_lora_adapter", "policy-v1"),
                                   ("load_lora_adapter", "policy-v2"), ("unload_lora_adapter", "policy-v0"),
                                   ("load_lora_adapter", "policy-v3"), ("unload_lora_adapter", "policy-v1"),
                                   ("unload_lora_adapter", "policy-v2")]
