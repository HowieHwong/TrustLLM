"""Actual scoring fixtures and judge failure behavior without model downloads/API calls."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_original_external_scorer_balanced_fixture():
    pytest.importorskip("sklearn")
    pytest.importorskip("openai")
    from trustllm.task.truthfulness import TruthfulnessEval

    rows = [
        {"source": source, "answer": label, "res": label}
        for source in ("scifact", "covid", "healthver", "climate")
        for label in ("SUPPORT", "REFUTE")
    ]
    result = TruthfulnessEval().external_eval(rows)
    assert result == {"scifact": 1.0, "covid": 1.0, "healthver": 1.0, "climate": 1.0, "avg": 1.0}


def test_judge_selection_uses_configuration(monkeypatch):
    pytest.importorskip("openai")
    from trustllm import config
    from trustllm.utils import gpt_auto_eval

    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Yes"))]
    )
    monkeypatch.setattr(gpt_auto_eval, "OpenAI", lambda **kw: client)
    monkeypatch.setattr(config, "azure_openai", False)
    monkeypatch.setattr(config, "judge_model", "configured-test-judge")
    assert gpt_auto_eval.get_res.__wrapped__("test") == "Yes"
    assert client.chat.completions.create.call_args.kwargs["model"] == "configured-test-judge"


def test_judge_worker_errors_propagate(monkeypatch, tmp_path):
    pytest.importorskip("openai")
    from trustllm.utils import gpt_auto_eval

    monkeypatch.setattr(
        gpt_auto_eval, "get_res", MagicMock(side_effect=RuntimeError("test failure"))
    )
    judge = gpt_auto_eval.AutoEvaluator(save_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="test failure"):
        judge.evaluate([{"res": "test response"}], task="ETHICS")


def test_all_six_pipeline_imports():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    pytest.importorskip("openai")
    pytest.importorskip("googleapiclient")
    from trustllm.task import pipeline

    assert all(
        callable(getattr(pipeline, "run_" + task))
        for task in ("safety", "ethics", "fairness", "privacy", "truthfulness", "robustness")
    )
