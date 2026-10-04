from unittest.mock import MagicMock

import pytest
import requests
from trustllm.models import APIModel


def response(status=200, text="hello"):
    res = MagicMock(status_code=status, ok=status < 400, headers={})
    res.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": {"completion_tokens": 1},
    }
    return res


def test_arbitrary_api_model_and_token_options(monkeypatch):
    post = MagicMock(return_value=response())
    monkeypatch.setattr(requests, "post", post)
    client = APIModel(
        "custom/reasoning-model",
        base_url="http://localhost:8000/v1/",
        api_key="test-secret",
        token_parameter="max_completion_tokens",
        omit_temperature=True,
    )
    assert client.generate("hi", temperature=1, max_new_tokens=7).text == "hello"
    args, kw = post.call_args
    assert args[0] == "http://localhost:8000/v1/chat/completions"
    assert kw["json"]["model"] == "custom/reasoning-model"
    assert kw["json"]["max_completion_tokens"] == 7
    assert "temperature" not in kw["json"]
    assert kw["headers"]["Authorization"] == "Bearer test-secret"
    assert "test-secret" not in str(client.describe())


def test_rate_limit_retry(monkeypatch):
    post = MagicMock(side_effect=[response(429), response()])
    sleep = MagicMock()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("trustllm.models.time.sleep", sleep)
    APIModel("m", base_url="http://localhost/v1").generate("hi", temperature=0, max_new_tokens=4)
    assert post.call_count == 2
    sleep.assert_called_once()


def test_auth_failure_not_retried_or_body_logged(monkeypatch):
    res = response(401)
    res.text = "private prompt and key"
    post = MagicMock(return_value=res)
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        APIModel("m", base_url="http://localhost/v1").generate(
            "hi", temperature=0, max_new_tokens=4
        )
    assert "private" not in str(error.value)
    assert post.call_count == 1


def test_timeout_retries_are_bounded(monkeypatch):
    post = MagicMock(side_effect=requests.Timeout("sensitive request"))
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("trustllm.models.time.sleep", lambda n: None)
    with pytest.raises(RuntimeError, match="timeout/connection"):
        APIModel("m", base_url="http://localhost/v1", retries=1).generate(
            "hi", temperature=0, max_new_tokens=4
        )
    assert post.call_count == 2


@pytest.mark.parametrize("text", [None, "", "  ", [{"text": "multimodal"}]])
def test_empty_or_unsupported_response_rejected(monkeypatch, text):
    monkeypatch.setattr(requests, "post", MagicMock(return_value=response(text=text)))
    with pytest.raises(RuntimeError, match="empty/non-text"):
        APIModel("m", base_url="http://localhost/v1").generate(
            "hi", temperature=0, max_new_tokens=4
        )


@pytest.mark.parametrize(
    "url", ["file:///tmp/key", "https://secret@example.com/v1", "https://x/v1?key=secret"]
)
def test_unsafe_endpoint_forms_rejected(url):
    with pytest.raises(ValueError, match="base_url"):
        APIModel("m", base_url=url)
