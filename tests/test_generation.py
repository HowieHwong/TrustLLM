import json
from unittest.mock import MagicMock

import pytest
from trustllm import runner
from trustllm.models import Response


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "input" / "safety"
    source.mkdir(parents=True)
    (source / "jailbreak.json").write_text(
        json.dumps([{"prompt": "one", "label": "keep"}, {"prompt": "two", "label": "also keep"}])
    )
    return source


@pytest.fixture
def adapter(monkeypatch):
    instance = MagicMock()
    instance.describe.return_value = {"backend": "api", "model": "arbitrary-model"}
    instance.generate.side_effect = lambda prompt, **kw: Response(
        "answer " + prompt, {"completion_tokens": 2}
    )
    monkeypatch.setattr(runner, "APIModel", MagicMock(return_value=instance))
    return instance


def run(dataset, tmp_path, **kwargs):
    return runner.generate(
        "arbitrary-model",
        task="safety",
        data_path=dataset,
        output_dir=tmp_path / "results",
        **kwargs,
    )


def test_outputs_keep_fields_and_manifest(dataset, tmp_path, adapter):
    result = run(dataset, tmp_path, concurrency=2)
    assert result["status"] == "completed"
    assert result["successful"] == 2
    records = json.loads((tmp_path / "results/jailbreak.json").read_text())
    assert records == [
        {"prompt": "one", "label": "keep", "res": "answer one"},
        {"prompt": "two", "label": "also keep", "res": "answer two"},
    ]
    assert result["settings"]["temperatures"]["jailbreak.json"] == 1
    assert "trustworthiness score" in (tmp_path / "results/report.html").read_text()
    assert result["settings"]["dataset_sha256"]["jailbreak.json"]


def test_limit_and_temperature(dataset, tmp_path, adapter):
    result = run(dataset, tmp_path, limit=1, temperature=0.2, max_new_tokens=13)
    assert result["total"] == 1
    adapter.generate.assert_called_once_with("one", temperature=0.2, max_new_tokens=13)


def test_resume_only_failed_samples(dataset, tmp_path, adapter):
    adapter.generate.side_effect = [Response("first"), RuntimeError("temporary failure")]
    with pytest.raises(runner.RunFailed):
        run(dataset, tmp_path)
    assert json.loads((tmp_path / "results/run.json").read_text())["failed"] == 1
    adapter.generate.reset_mock(side_effect=True)
    adapter.generate.return_value = Response("second")
    result = run(dataset, tmp_path, resume=True)
    adapter.generate.assert_called_once()
    assert result["successful"] == 2
    assert result["failed"] == 0
    assert json.loads((tmp_path / "results/jailbreak.json").read_text())[0]["res"] == "first"


def test_resume_completed_run_does_not_call_model(dataset, tmp_path, adapter):
    run(dataset, tmp_path)
    adapter.generate.reset_mock()
    run(dataset, tmp_path, resume=True)
    adapter.generate.assert_not_called()


def test_resume_rejects_changed_dataset(dataset, tmp_path, adapter):
    run(dataset, tmp_path)
    (dataset / "jailbreak.json").write_text('[{"prompt":"different"}]')
    adapter.generate.reset_mock()
    with pytest.raises(ValueError, match="changed"):
        run(dataset, tmp_path, resume=True)
    adapter.generate.assert_not_called()


def test_resume_rejects_changed_settings(dataset, tmp_path, adapter):
    run(dataset, tmp_path)
    with pytest.raises(ValueError, match="changed"):
        run(dataset, tmp_path, resume=True, temperature=0.9)


def test_torn_final_journal_line_is_discarded(dataset, tmp_path, adapter):
    run(dataset, tmp_path)
    with (tmp_path / "results/samples.jsonl").open("ab") as f:
        f.write(b'{"file":')
    run(dataset, tmp_path, resume=True)
    lines = (tmp_path / "results/samples.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line) for line in lines)


def test_existing_outputs_not_overwritten(dataset, tmp_path, adapter):
    run(dataset, tmp_path)
    with pytest.raises(FileExistsError):
        run(dataset, tmp_path)


def test_invalid_input_before_model_loading(dataset, tmp_path, adapter):
    (dataset / "jailbreak.json").write_text('[{"prompt":null}]')
    with pytest.raises(ValueError, match="row 0"):
        run(dataset, tmp_path)
    runner.APIModel.assert_not_called()


def test_failure_does_not_return_success(dataset, tmp_path, adapter):
    adapter.generate.side_effect = RuntimeError("bad response")
    with pytest.raises(runner.RunFailed):
        run(dataset, tmp_path)
    report = json.loads((tmp_path / "results/run.json").read_text())
    assert report["status"] == "failed"
    assert report["successful"] == 0


def test_html_escapes_model_and_names(tmp_path):
    from trustllm.reporting import write_report

    report = {
        "settings": {
            "task": "safety",
            "model": {"model": "<script>alert(1)</script>", "backend": "api"},
        },
        "total": 0,
        "successful": 0,
        "failed": 0,
        "files": {},
        "status": "completed",
    }
    path = tmp_path / "report.html"
    write_report(report, path)
    assert "<script>" not in path.read_text()
    assert "&lt;script&gt;" in path.read_text()
