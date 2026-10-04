"""Reproducible text generation with sample checkpoints and compatible JSON outputs."""

import hashlib
import json
import logging
import os
import platform
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from trustllm.models import APIModel, LocalModel
from trustllm.reporting import write_report
from trustllm.tasks import TASKS

logger = logging.getLogger(__name__)


class RunFailed(RuntimeError):
    """A run finished with failed samples; successful samples are checkpointed."""


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            name = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def _datasets(task, data_path, limit, prompt_key):
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {', '.join(TASKS)}")
    root = Path(data_path)
    if root.is_file():
        paths = [root]
    else:
        folder = root / task if (root / task).is_dir() else root
        paths = [folder / name for name in TASKS[task] if (folder / name).is_file()]
    if not paths:
        raise FileNotFoundError(
            f"No {task} datasets found under {root}; run 'trustllm download' first"
        )
    datasets, fingerprints = {}, {}
    for path in paths:
        if path.name in {"run.json", "scores.json"}:
            raise ValueError(f"{path.name} is reserved for run metadata")
        raw = path.read_bytes()
        records = json.loads(raw)
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path}: expected a nonempty JSON array")
        records = records[:limit] if limit is not None else records
        for index, row in enumerate(records):
            if (
                not isinstance(row, dict)
                or not isinstance(row.get(prompt_key), str)
                or not row[prompt_key].strip()
            ):
                raise ValueError(
                    f"{path}, row {index}: missing nonempty string field {prompt_key!r}"
                )
        datasets[path.name] = [{k: v for k, v in row.items() if k != "res"} for row in records]
        fingerprints[path.name] = hashlib.sha256(raw).hexdigest()
    return datasets, fingerprints


def _versions():
    result = {"python": platform.python_version()}
    for package in ("trustllm", "requests", "torch", "transformers", "accelerate"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result


def _read_journal(path):
    completed = {}
    if path.exists():
        # A process can be killed during its final write. Discard only that torn line.
        with path.open("rb+") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    handle.truncate(offset)
                    break
                try:
                    event = json.loads(line)
                    key = (event["file"], event["index"])
                    if event["status"] == "success":
                        if not isinstance(event.get("res"), str) or not event["res"].strip():
                            raise ValueError("invalid response")
                        completed[key] = event
                except (ValueError, KeyError, TypeError):
                    raise ValueError(
                        f"Corrupt checkpoint in {path}; use a new output directory"
                    ) from None
    return completed


def generate(
    model,
    *,
    task,
    backend="api",
    data_path="data/dataset",
    output_dir=None,
    limit=None,
    concurrency=1,
    resume=False,
    temperature=None,
    max_new_tokens=512,
    prompt_key="prompt",
    base_url=None,
    api_key=None,
    timeout=120,
    retries=3,
    device="auto",
    dtype="auto",
    revision=None,
    trust_remote_code=False,
    seed=42,
    repetition_penalty=1.0,
    token_parameter="max_tokens",
    omit_temperature=False,
):
    """Generate responses with ``api`` or ``local`` and return a run summary.

    ``limit`` is per dataset file. Resume requires identical inputs and generation
    settings; only successful checkpoints are reused. Failed samples raise RunFailed
    after writing a report. API keys are never included in artifacts.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty model ID or local path")
    if backend not in {"api", "local"}:
        raise ValueError("backend must be api or local")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if concurrency < 1 or max_new_tokens < 1:
        raise ValueError("concurrency and max_new_tokens must be positive")
    if temperature is not None and not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if backend == "local" and concurrency != 1:
        raise ValueError(
            "Local generation uses concurrency=1; use a serving API for concurrent inference"
        )
    datasets, fingerprints = _datasets(task, data_path, limit, prompt_key)
    temperatures = {
        name: temperature if temperature is not None else TASKS[task].get(name, 0.0)
        for name in datasets
    }
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "--", model).strip(".-") or "model"
    output = Path(output_dir) if output_dir else Path("runs") / slug / task
    manifest_path = output / "run.json"
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"{output} is not empty; use --resume or a new output directory")
    if resume and not manifest_path.exists():
        raise FileNotFoundError(f"Cannot resume: {manifest_path} does not exist")
    # Build identity before loading a potentially large local model.
    if backend == "api":
        adapter = APIModel(
            model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            retries=retries,
            token_parameter=token_parameter,
            omit_temperature=omit_temperature,
        )
        identity = adapter.describe()
    else:
        adapter = None
        identity = {
            "backend": backend,
            "model": model,
            "device": device,
            "dtype": dtype,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "seed": seed,
            "repetition_penalty": repetition_penalty,
        }
    settings = {
        "model": identity,
        "task": task,
        "dataset_sha256": fingerprints,
        "limit": limit,
        "temperatures": temperatures,
        "max_new_tokens": max_new_tokens,
        "prompt_key": prompt_key,
    }
    environment = _versions()
    if resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("settings") != settings or previous.get("versions") != environment:
            raise ValueError(
                "Resume settings, dataset or dependency versions changed; use a new output directory"
            )
    else:
        previous = {"created_at": datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "samples.jsonl"
    completed = _read_journal(journal) if resume else {}
    expected = {(name, index) for name, rows in datasets.items() for index in range(len(rows))}
    if not set(completed).issubset(expected):
        raise ValueError("Checkpoint contains samples outside this run")
    manifest = {
        "schema_version": 1,
        "created_at": previous["created_at"],
        "settings": settings,
        "versions": environment,
        "status": "running",
        "concurrency": concurrency,
        "request_timeout": timeout if backend == "api" else None,
        "request_retries": retries if backend == "api" else None,
    }
    atomic_json(manifest_path, manifest)
    failures = {}
    interrupted = False
    if resume and previous.get("resolved_model"):
        manifest["resolved_model"] = previous["resolved_model"]

    def sample(name, index, row):
        try:
            result = adapter.generate(
                row[prompt_key], temperature=temperatures[name], max_new_tokens=max_new_tokens
            )
            if not isinstance(result.text, str) or not result.text.strip():
                raise RuntimeError("Model returned empty text")
            return {
                "file": name,
                "index": index,
                "status": "success",
                "res": result.text,
                "usage": result.usage,
            }
        except Exception as exc:
            # Adapter errors are sanitized. Unexpected library errors are identified by type only.
            message = (
                str(exc)
                if backend == "api" and isinstance(exc, RuntimeError)
                else type(exc).__name__
            )
            return {"file": name, "index": index, "status": "error", "error": message}

    try:
        if adapter is None and len(completed) < len(expected):
            adapter = LocalModel(
                model,
                device=device,
                dtype=dtype,
                revision=revision,
                trust_remote_code=trust_remote_code,
                seed=seed,
                repetition_penalty=repetition_penalty,
            )
            manifest["resolved_model"] = adapter.describe()
        pending = [
            (name, index, row)
            for name, rows in datasets.items()
            for index, row in enumerate(rows)
            if (name, index) not in completed
        ]
        with journal.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(sample, *item) for item in pending]
                for future in as_completed(futures):
                    event = future.result()
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                    handle.flush()
                    key = event["file"], event["index"]
                    if event["status"] == "success":
                        completed[key] = event
                    else:
                        failures[key] = event["error"]
                    logger.info(
                        "%s [%d/%d] %s",
                        event["file"],
                        len(completed) + len(failures),
                        len(expected),
                        event["status"],
                    )
    except BaseException:
        interrupted = True
        raise
    finally:
        file_summaries = {}
        for name, rows in datasets.items():
            records = [
                dict(
                    row, res=completed[(name, index)]["res"] if (name, index) in completed else None
                )
                for index, row in enumerate(rows)
            ]
            atomic_json(output / name, records)
            successful = sum((name, index) in completed for index in range(len(rows)))
            file_summaries[name] = {
                "total": len(rows),
                "successful": successful,
                "failed": sum(key[0] == name for key in failures),
                "pending": len(rows) - successful - sum(key[0] == name for key in failures),
            }
        manifest.update(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": "interrupted" if interrupted else "failed" if failures else "completed",
                "files": file_summaries,
                "successful": len(completed),
                "total": len(expected),
                "failed": len(failures),
                "output_dir": str(output.resolve()),
            }
        )
        atomic_json(manifest_path, manifest)
        write_report(manifest, output / "report.html")
    if failures:
        raise RunFailed(
            f"{len(failures)} samples failed; inspect {journal} and rerun with --resume"
        )
    return manifest
