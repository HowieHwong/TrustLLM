import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from trustllm.evaluation import evaluate


@pytest.fixture
def responses(tmp_path):
    for name in ("jailbreak.json", "misuse.json", "exaggerated_safety.json"):
        (tmp_path / name).write_text('[{"prompt":"hello","res":"response"}]')
    return tmp_path


def test_evaluator_rejects_incomplete_responses_before_import(responses, monkeypatch):
    (responses / "misuse.json").write_text('[{"prompt":"hello","res":null}]')
    load = MagicMock()
    monkeypatch.setattr("trustllm.evaluation.importlib.import_module", load)
    with pytest.raises(ValueError, match="nonempty res"):
        evaluate("safety", responses)
    load.assert_not_called()


def test_evaluation_adapter_preserves_scores_and_records_inputs(responses, monkeypatch):
    score = MagicMock(
        return_value={
            "jailbreak_res": 0.5,
            "misuse_res": 1.0,
            "exaggerated_res": 0.0,
            "toxicity_res": None,
        }
    )
    monkeypatch.setattr(
        "trustllm.evaluation.importlib.import_module",
        lambda name: SimpleNamespace(run_safety=score),
    )
    result = evaluate("safety", responses)
    assert result["scores"]["jailbreak_res"] == 0.5
    assert "jailbreak_res" in (responses / "scores.html").read_text()
    assert len(result["input_sha256"]) == 3
    assert json.loads((responses / "scores.json").read_text())["task"] == "safety"


def test_evaluator_rejects_failed_scores(responses, monkeypatch):
    monkeypatch.setattr(
        "trustllm.evaluation.importlib.import_module",
        lambda name: SimpleNamespace(run_safety=lambda **kw: {"jailbreak_res": None}),
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        evaluate("safety", responses)
    assert not (responses / "scores.json").exists()
