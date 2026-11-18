from unittest.mock import Mock

import pytest

pytest.importorskip("googleapiclient")
from trustllm.utils.perspective import PerspectiveEval


def test_perspective_endpoint_and_discovery(monkeypatch, tmp_path):
    build = Mock()
    monkeypatch.setattr("trustllm.utils.perspective.discovery.build", build)
    response = {"attributeScores": {"TOXICITY": {"spanScores": [{"score": {"value": 0.2}}]}}}
    build.return_value.comments.return_value.analyze.return_value.execute.return_value = response
    default = PerspectiveEval(save_dir=str(tmp_path))
    assert default.get_toxicity_value("example") == 0.2
    assert build.call_args.kwargs["client_options"] is None
    assert build.call_args.kwargs["discoveryServiceUrl"] == "https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
    custom = PerspectiveEval(save_dir=str(tmp_path), api_endpoint="https://moderation.example/v1alpha1/",
                             discovery_service_url="https://moderation.example/discovery.json")
    custom.get_toxicity_value("example")
    assert build.call_args.kwargs["client_options"] == {"api_endpoint": "https://moderation.example/v1alpha1/"}
    assert build.call_args.kwargs["discoveryServiceUrl"] == "https://moderation.example/discovery.json"
