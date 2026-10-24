import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from budget_agent.models import ModelError, OpenAICompatibleModel, parse_action
from budget_agent.schema import RunConfig


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.server.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(self.server.status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = self.server.body
            self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    httpd.requests = []
    httpd.status = 200
    httpd.body = {"model": "version-pinned", "choices": [{"message": {"content": '{"tool":"finish"}'}}], "usage": {"prompt_tokens": 30, "completion_tokens": 6}}
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def model(server):
    return OpenAICompatibleModel(RunConfig(model="requested-version", base_url=f"http://127.0.0.1:{server.server_port}/v1"))


def test_real_http_protocol_and_usage(server):
    response = model(server).complete([{"role": "user", "content": "test"}], 17)
    assert response.input_tokens == 30 and response.output_tokens == 6
    assert response.model == "version-pinned"
    assert server.requests[0]["max_tokens"] == 17
    assert server.requests[0]["stream"] is False


def test_missing_usage_is_not_free_inference(server):
    del server.body["usage"]
    with pytest.raises(ModelError, match="usage unknown"):
        model(server).complete([], 17)


def test_invalid_utf8_is_a_provider_error(server):
    server.body = b"\xff"
    with pytest.raises(ModelError, match="UnicodeDecodeError.*usage unknown"):
        model(server).complete([], 17)
    assert len(server.requests) == 1


@pytest.mark.parametrize("name", [None, 123, {}, ""])
def test_malformed_provider_model_name_is_rejected(server, name):
    server.body["model"] = name
    with pytest.raises(ModelError, match="string model name"):
        model(server).complete([], 17)


def test_absent_provider_model_name_uses_requested_version(server):
    del server.body["model"]
    assert model(server).complete([], 17).model == "requested-version"


def test_http_error_no_secret_echo_or_automatic_retry(server):
    server.status, server.body = 429, {"error": "secret-key-should-not-be-logged"}
    with pytest.raises(ModelError) as error:
        model(server).complete([], 17)
    assert "secret-key" not in str(error.value)
    assert len(server.requests) == 1


def test_credential_url_rejected():
    with pytest.raises(ValueError, match="credentials"):
        OpenAICompatibleModel(RunConfig(model="x", base_url="https://name:secret@example.com/v1"))


@pytest.mark.parametrize("text", ['[]', '{"tool":"delete_everything"}', 'prose {"tool":"finish"}', '{"tool":1}'])
def test_invalid_action_rejected(text):
    with pytest.raises(ValueError):
        parse_action(text)


def test_fenced_json_is_supported():
    assert parse_action('```json\n{"tool":"finish"}\n```') == {"tool": "finish"}


@pytest.mark.parametrize("action", [
    {"tool": "run"},
    {"tool": "write_file", "path": "solution.py", "content": {}},
    {"tool": "read_file", "path": ["solution.py"]},
    {"tool": "read_file", "path": "solution.py", "start_line": None},
    {"tool": "recall", "event_id": "e0001", "limit": True},
    {"tool": "list_files", "path": None},
    {"tool": "list_files", "offset": False},
    {"tool": "search", "query": "def", "limit": "50"},
    {"tool": "search", "query": "def", "path": []},
])
def test_malformed_tool_arguments_are_action_errors(action):
    with pytest.raises(ValueError):
        parse_action(json.dumps(action))


def test_optional_integer_fields_can_be_omitted():
    action = {"tool": "read_file", "path": "solution.py"}
    assert parse_action(json.dumps(action)) == action


@pytest.mark.parametrize("action", [
    {"tool": "list_files", "path": "src", "offset": 100, "limit": 20},
    {"tool": "search", "query": "def", "path": "src/utils.py", "offset": 2, "limit": 5},
])
def test_scoped_paginated_query_arguments_are_preserved(action):
    assert parse_action(json.dumps(action)) == action
