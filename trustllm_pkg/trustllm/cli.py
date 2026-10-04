"""Command-line interface; also available as python -m trustllm."""

import argparse
import json
import logging
import sys
from pathlib import Path

from trustllm.dataset_download import download_dataset
from trustllm.runner import generate
from trustllm.tasks import TASKS


def parser():
    root = argparse.ArgumentParser(
        prog="trustllm", description="TrustLLM · Download → Generate → Evaluate"
    )
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("tasks", help="List benchmark dimensions and dataset files")
    download = sub.add_parser("download", help="Download and extract the benchmark")
    download.add_argument(
        "--output", default="data", help="Parent of the extracted dataset/ directory"
    )
    gen = sub.add_parser("generate", help="Generate responses with a local model or compatible API")
    gen.add_argument(
        "--config", type=Path, help="JSON configuration file; explicit CLI flags override it"
    )
    gen.add_argument("--model", default=argparse.SUPPRESS)
    gen.add_argument("--backend", choices=["api", "local"], default=argparse.SUPPRESS)
    gen.add_argument("--task", choices=list(TASKS), default=argparse.SUPPRESS)
    gen.add_argument("--data", dest="data_path", default=argparse.SUPPRESS)
    gen.add_argument("--output", dest="output_dir", default=argparse.SUPPRESS)
    gen.add_argument("--base-url", default=argparse.SUPPRESS)
    gen.add_argument(
        "--limit",
        type=int,
        default=argparse.SUPPRESS,
        help="Samples per dataset file (smoke test only)",
    )
    gen.add_argument("--concurrency", type=int, default=argparse.SUPPRESS)
    gen.add_argument("--max-new-tokens", type=int, default=argparse.SUPPRESS)
    gen.add_argument("--temperature", type=float, default=argparse.SUPPRESS)
    gen.add_argument("--timeout", type=float, default=argparse.SUPPRESS)
    gen.add_argument("--retries", type=int, default=argparse.SUPPRESS)
    gen.add_argument("--device", default=argparse.SUPPRESS)
    gen.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default=argparse.SUPPRESS
    )
    gen.add_argument("--revision", default=argparse.SUPPRESS)
    gen.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    gen.add_argument("--prompt-key", default=argparse.SUPPRESS)
    gen.add_argument("--repetition-penalty", type=float, default=argparse.SUPPRESS)
    gen.add_argument(
        "--token-parameter",
        choices=["max_tokens", "max_completion_tokens"],
        default=argparse.SUPPRESS,
    )
    for flag in ("resume", "trust-remote-code", "omit-temperature"):
        gen.add_argument(f"--{flag}", action="store_true", default=argparse.SUPPRESS)
    score = sub.add_parser(
        "evaluate", help="Score complete response files with the original evaluators"
    )
    score.add_argument("--task", choices=list(TASKS), required=True)
    score.add_argument("--data", required=True, help="Directory containing generated JSON files")
    score.add_argument("--output", help="Score JSON path (default: <data>/scores.json)")
    return root


def main(argv=None):
    root = parser()
    args = vars(root.parse_args(argv))
    command = args.pop("command")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if command == "tasks":
            for task, files in TASKS.items():
                print(f"{task:14} {len(files):2} files  " + ", ".join(files))
        elif command == "download":
            download_dataset(args["output"])
            print(f"Dataset ready: {Path(args['output']) / 'dataset'}")
        elif command == "evaluate":
            from trustllm.evaluation import evaluate

            print(json.dumps(evaluate(args["task"], args["data"], args["output"]), indent=2))
        else:
            config_file = args.pop("config")
            options = json.loads(config_file.read_text(encoding="utf-8")) if config_file else {}
            if not isinstance(options, dict):
                raise ValueError("Configuration must be a JSON object")
            if "api_key" in options:
                raise ValueError("Use OPENAI_API_KEY in the environment, not a saved config file")
            options.update(args)
            if not options.get("model") or not options.get("task"):
                raise ValueError("Provide --model and --task, or include them in --config")
            result = generate(**options)
            print(
                f"Completed {result['successful']}/{result['total']} samples. Report: {result['output_dir']}/report.html"
            )
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(f"trustllm: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "trustllm: interrupted; use --resume to recover checkpointed samples", file=sys.stderr
        )
        return 130
