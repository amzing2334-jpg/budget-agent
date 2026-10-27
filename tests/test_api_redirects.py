"""Real loopback HTTP exchanges; all keys and model responses are test fixtures."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading

import pytest

from budget_agent.cli import main
from budget_agent.models import ModelError, OpenAICompatibleModel
from budget_agent.schema import RunConfig


@pytest.fixture
def endpoints(monkeypatch):
    monkeypatch.setenv("BUDGET_AGENT_TEST_KEY", "dummy-redirect-key")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.server.requests.append({"method": self.command, "path": self.path,
                "authorization": self.headers.get("Authorization"), "body": body})
            redirect = self.path == "/v1/chat/completions" and self.server.status != 200
            self.send_response(self.server.status if redirect else 200)
            if redirect:
                self.send_header("Location", self.server.location)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"model": "http-fixture", "choices": [
                {"message": {"content": '{"tool":"finish"}'}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3}}).encode())

        do_GET = do_POST

        def log_message(self, *args):
            pass

    servers = []
    try:
        for _ in range(2):
            server = HTTPServer(("127.0.0.1", 0), Handler)
            server.requests, server.status, server.location = [], 200, ""
            thread = threading.Thread(target=lambda s=server: s.serve_forever(poll_interval=0.01), daemon=True)
            thread.start()
            servers.append((server, thread))
        yield tuple(s for s, _ in servers)
    finally:
        for server, thread in servers:
            server.shutdown()
            server.server_close()
            thread.join()


def api_model(origin):
    return OpenAICompatibleModel(RunConfig(model="http-fixture", api_key_env="BUDGET_AGENT_TEST_KEY",
        base_url=f"http://127.0.0.1:{origin.server_port}/v1"))


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("cross_origin", [False, True])
def test_no_redirected_request_or_credentials(endpoints, status, cross_origin):
    origin, target = endpoints
    origin.status = status
    origin.location = (f"http://localhost:{target.server_port}/redirected" if cross_origin else "/redirected")
    with pytest.raises(ModelError, match=f"HTTP {status}.*usage may be unknown"):
        api_model(origin).complete([{"role": "user", "content": "fixture-prompt"}], 20)
    assert len(origin.requests) == 1
    assert origin.requests[0]["method"] == "POST"
    assert origin.requests[0]["authorization"] == "Bearer dummy-redirect-key"
    assert target.requests == []


def test_redirect_location_is_not_parsed_or_exposed(endpoints):
    origin, target = endpoints
    origin.status = 302
    origin.location = "https://[invalid-location-with-private-marker"
    with pytest.raises(ModelError, match="HTTP 302") as error:
        api_model(origin).complete([], 20)
    assert "private-marker" not in str(error.value)
    assert len(origin.requests) == 1 and target.requests == []


def test_direct_response_keeps_authorized_request_and_usage(endpoints):
    origin, target = endpoints
    response = api_model(origin).complete([{"role": "user", "content": "fixture-prompt"}], 20)
    assert (response.model, response.input_tokens, response.output_tokens) == ("http-fixture", 9, 3)
    request, = origin.requests
    assert request["authorization"] == "Bearer dummy-redirect-key"
    assert json.loads(request["body"])["messages"][0]["content"] == "fixture-prompt"
    assert target.requests == []


def test_redirect_stops_batch_and_preserves_unknown_usage(endpoints, tmp_path):
    origin, target = endpoints
    origin.status = 302
    origin.location = f"http://localhost:{target.server_port}/redirected?private-location-marker"
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "runs"
    code = main(["compare", "--provider", "openai-compatible", "--model", "http-fixture",
        "--base-url", f"http://127.0.0.1:{origin.server_port}/v1", "--api-key-env", "BUDGET_AGENT_TEST_KEY",
        "--tasks", str(root / "tasks/local.jsonl"), "--task", "slugify-digits",
        "--strategies", "recent", "mask", "--repeat", "2", "--environment", "local",
        "--allow-local-execution", "--output", str(output)])
    assert code == 1
    result_path, = output.glob("*/result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "model_error" and result["usage_unknown"] is True
    assert result["budget"]["unreconciled_request"] is True
    assert result["budget"]["attempted_calls"] == 1 and result["calls"] == 0
    assert result["cost_usd"] is None and result["usage_sources"] == []
    assert (output / "report/report.html").is_file()
    assert len(origin.requests) == 1 and target.requests == []
    for artifact in output.rglob("*"):
        if artifact.is_file() and artifact.suffix in {".json", ".jsonl", ".md", ".html"}:
            text = artifact.read_text(encoding="utf-8")
            assert "dummy-redirect-key" not in text and "private-location-marker" not in text
