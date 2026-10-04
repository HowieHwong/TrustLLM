"""Validated entry point for TrustLLM's existing scoring pipelines."""

import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

from trustllm.runner import _versions, atomic_json
from trustllm.tasks import TASKS


def evaluate(task, data_path, output_path=None):
    """Score a complete dimension of generated response files.

    Scorers retain historical prompts and metric definitions. Some require a
    classifier, embedding service, or LLM judge. Generation reports are separate.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}")
    folder = Path(data_path)
    names = [name for name in TASKS[task] if name != "awareness.json"]
    fingerprints = {}
    for name in names:
        path = folder / name
        raw = path.read_bytes()
        records = json.loads(raw)
        if (
            not isinstance(records, list)
            or not records
            or any(
                not isinstance(row, dict)
                or not isinstance(row.get("res"), str)
                or not row["res"].strip()
                for row in records
            )
        ):
            raise ValueError(f"{path}: every sample must have a nonempty res before scoring")
        fingerprints[name] = hashlib.sha256(raw).hexdigest()
    destination = Path(output_path) if output_path else folder / "scores.json"
    if destination.suffix != ".json":
        raise ValueError("Score output must use a .json filename")
    if destination.resolve() in [
        (folder / name).resolve() for name in names
    ] or destination.name in {"run.json", "samples.jsonl", "report.html"}:
        raise ValueError("Score output must not overwrite generation artifacts")
    if destination.exists() or destination.with_suffix(".html").exists():
        raise FileExistsError(
            f"{destination} or its HTML report already exists; choose a new --output path"
        )
    try:
        pipeline = importlib.import_module("trustllm.task.pipeline")
    except ImportError:
        raise ImportError(
            "Scoring requires the 'eval' extra: pip install 'trustllm[eval]' (or the source-install equivalent)"
        ) from None
    scores = getattr(pipeline, f"run_{task}")(all_folder_path=str(folder))
    required = {
        key: value for key, value in scores.items() if key not in {"emotional_res", "toxicity_res"}
    }
    if not required or any(value is None for value in required.values()):
        raise RuntimeError(
            "Scorer returned incomplete results; no successful score file was written"
        )
    # Convert NumPy scalars without importing NumPy in the lightweight package.
    scores = json.loads(json.dumps(scores, default=lambda value: value.item(), allow_nan=False))
    from trustllm import config

    result = {
        "task": task,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "input_sha256": fingerprints,
        "versions": _versions(),
        "judge_model": config.azure_engine if config.azure_openai else config.judge_model,
    }
    atomic_json(destination, result)
    from trustllm.reporting import write_score_report

    write_score_report(result, destination.with_suffix(".html"))
    return result
