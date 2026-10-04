import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from trustllm.cli import main


def test_cli_help_and_tasks(capsys):
    assert main(["tasks"]) == 0
    assert "safety" in capsys.readouterr().out


def test_config_overrides(monkeypatch, tmp_path):
    config = tmp_path / "run.json"
    config.write_text('{"model":"m","task":"safety","limit":2}')
    received = {}

    def generate(**kwargs):
        received.update(kwargs)
        return {"successful": 1, "total": 1, "output_dir": "result"}

    monkeypatch.setattr("trustllm.cli.generate", generate)
    assert main(["generate", "--config", str(config), "--limit", "1"]) == 0
    assert received["limit"] == 1


def test_secret_in_config_rejected(tmp_path, capsys):
    config = tmp_path / "run.json"
    config.write_text('{"api_key":"secret-test-value"}')
    assert main(["generate", "--config", str(config)]) == 1
    assert "secret-test-value" not in capsys.readouterr().err


def test_missing_data_exits_nonzero(tmp_path):
    assert (
        main(["generate", "--model", "m", "--task", "safety", "--data", str(tmp_path / "missing")])
        == 1
    )


def test_real_cli_to_http_server_and_resume(tmp_path):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body, self.headers.get("Authorization")))
            payload = json.dumps({"choices": [{"message": {"content": "fixture reply"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = tmp_path / "input.json"
    source.write_text('[{"prompt":"hello", "label":1}]')
    output = tmp_path / "run"
    env = dict(os.environ, OPENAI_API_KEY="test-private-key")
    command = [
        sys.executable,
        "-m",
        "trustllm",
        "generate",
        "--backend",
        "api",
        "--model",
        "arbitrary/id",
        "--base-url",
        f"http://127.0.0.1:{server.server_port}/v1",
        "--task",
        "safety",
        "--data",
        str(source),
        "--output",
        str(output),
        "--limit",
        "1",
    ]
    try:
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=20)
        assert result.returncode == 0, result.stderr
        resumed = subprocess.run(
            command + ["--resume"], env=env, text=True, capture_output=True, timeout=20
        )
        assert resumed.returncode == 0, resumed.stderr
        assert len(calls) == 1
        assert calls[0][0] == "/v1/chat/completions"
        assert calls[0][1]["model"] == "arbitrary/id"
        assert calls[0][2] == "Bearer test-private-key"
        records = json.loads((output / "input.json").read_text())
        assert records[0]["res"] == "fixture reply"
        assert records[0]["label"] == 1
        for path in output.iterdir():
            assert "test-private-key" not in path.read_text()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
