"""Load isolated copies of config so environment state cannot leak between tests."""

import runpy

from trustllm import config


def load_config():
    return runpy.run_path(config.__file__)


def test_environment_credentials(monkeypatch):
    values = {
        "OPENAI_API_KEY": ("openai_key", "test-key"),
        "OPENAI_BASE_URL": ("openai_api_base", "https://example.invalid/v1"),
        "ANTHROPIC_API_KEY": ("claude_api", "test-anthropic"),
        "REPLICATE_API_TOKEN": ("replicate_api", "test-replicate"),
        "AZURE_OPENAI_DEPLOYMENT": ("azure_engine", "test-deployment"),
    }
    for variable, (_, value) in values.items():
        monkeypatch.setenv(variable, value)
    settings = load_config()
    for attribute, value in values.values():
        assert settings[attribute] == value


def test_defaults_without_credentials(monkeypatch):
    for variable in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "AZURE_OPENAI_ENABLED"):
        monkeypatch.delenv(variable, raising=False)
    settings = load_config()
    assert settings["openai_key"] == ""
    assert settings["openai_api_base"] is None
    assert settings["azure_openai"] is False


def test_azure_boolean(monkeypatch):
    for value, expected in (
        ("true", True),
        ("YES", True),
        ("1", True),
        ("false", False),
        ("0", False),
    ):
        monkeypatch.setenv("AZURE_OPENAI_ENABLED", value)
        assert load_config()["azure_openai"] is expected


def test_python_overrides_remain_supported(monkeypatch):
    monkeypatch.setattr(config, "openai_key", "runtime-key")
    assert config.openai_key == "runtime-key"
