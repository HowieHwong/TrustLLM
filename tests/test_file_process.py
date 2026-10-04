import json

import pytest
from trustllm.utils.file_process import load_json, save_json


def test_unicode_response_roundtrip(tmp_path):
    path = tmp_path / "responses.json"
    records = [{"prompt": "你好", "res": "Hello 🌍", "score": 0.5}, {"res": None}]
    save_json(records, path)
    assert load_json(path) == records
    assert "你好" in path.read_text(encoding="utf-8")


def test_malformed_input_is_reported(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_json(path)


def test_missing_input_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "missing.json")
