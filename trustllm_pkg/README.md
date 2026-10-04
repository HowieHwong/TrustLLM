![TrustLLM](https://raw.githubusercontent.com/HowieHwong/TrustLLM/main/docs/assets/logo-transparent.png)

# TrustLLM

**Trustworthiness evaluation for large language models · ICML 2024**

[Documentation](https://howiehwong.github.io/TrustLLM/) ·
[Quickstart](https://howiehwong.github.io/TrustLLM/guides/running.html) ·
[Paper](https://arxiv.org/abs/2401.05561) ·
[GitHub](https://github.com/HowieHwong/TrustLLM)

Download the benchmark, generate responses with local or API models, and run the
original TrustLLM scoring pipelines across **truthfulness, safety, fairness,
robustness, privacy, and ethics**. Use the same workflow from Python, a terminal,
or an AI agent that orchestrates command-line tools.

## Install

Python 3.9+ and Git are required for source installation. Python 3.10–3.12 is recommended for local inference. Install directly from GitHub:

```bash
# Dataset download, API generation, checkpoints and HTML generation reports
python -m pip install "trustllm @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"

# Add local Hugging Face models and the original evaluation pipelines
python -m pip install "trustllm[local,eval] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
```

| Install | Included |
| --- | --- |
| `trustllm` | Lightweight API client, CLI, dataset downloader and run artifacts |
| `trustllm[local]` | PyTorch, Transformers, Accelerate and SentencePiece |
| `trustllm[eval]` | Original scorers, classifiers, judge SDK and metrics |
| `trustllm[local,eval]` | Local generation and scoring together |

A wheel from [GitHub Releases](https://github.com/HowieHwong/TrustLLM/releases) can also be installed with `python -m pip install /path/to/trustllm-<version>-py3-none-any.whl`, without Git. Replace the path with the downloaded wheel.

The base installation does not require PyTorch or a GPU. For GPU inference,
install a PyTorch build suitable for your hardware first. Pin the package and
model versions and save your environment when reporting research results.

## Download the benchmark

No repository clone is needed:

```bash
python -m trustllm download --output data
python -m trustllm tasks
```

The dataset is extracted to `data/dataset`. Downloading again overwrites matching
dataset files.

## Generate with an API model

Set `OPENAI_API_KEY` and, for another compatible service, `OPENAI_BASE_URL` in your
environment. Replace `your-model-id` with the model served by that endpoint.

```bash
trustllm generate --backend api --model your-model-id \
  --task safety --data data/dataset --limit 5 --output runs/api-smoke
```

The API backend supports text-only, non-streaming OpenAI-compatible
`/chat/completions`, including compatible local model servers. Native Anthropic,
Gemini, Azure deployment and Responses APIs require a compatible gateway.

`--limit 5` runs the first five samples **per dataset file** as a smoke test. Omit
it for a full run. To resume an interrupted run, repeat the same command with
`--resume`; completed samples are reused and incompatible settings are rejected.

## Generate with a local model

After installing `trustllm[local]`:

```bash
trustllm generate --backend local --model Qwen/Qwen2.5-0.5B-Instruct \
  --device auto --task safety --data data/dataset \
  --limit 5 --output runs/local-smoke
```

Use a Hugging Face causal-model ID or a directory containing saved model weights.
Weights from the Hub use the normal Hugging Face cache. CPU, CUDA and MPS device
selection are supported; actual compatibility depends on the model and your
PyTorch installation. Local inference is sequential.

## Use Python

```python
from trustllm import download_dataset, generate

download_dataset("data")
run = generate(
    model="your-model-id",
    backend="api",  # Use "local" for a Hugging Face causal model.
    task="safety",
    data_path="data/dataset",
    output_dir="runs/python-smoke",
    limit=5,
)
print(run["output_dir"])
```

Each generation run saves the original records with `res` responses, an append-only
sample journal, a manifest with settings and input hashes, and a standalone HTML
generation report. Failed runs produce a nonzero CLI exit code.

## Evaluate a complete run

After installing `trustllm[eval]`, generate a full set of responses without
`--limit`, then score that output directory:

```bash
trustllm evaluate --task safety --data runs/api-safety-full
```

Scoring writes `scores.json` and `scores.html` using the original metric names and
scales. Some scorers download classifiers or use paid judge/embedding APIs; check
the [evaluation guide](https://howiehwong.github.io/TrustLLM/guides/evaluation.html)
and configure an available `OPENAI_JUDGE_MODEL` before running them. Generation
reports describe completion, not trustworthiness scores. Smoke-test subsets are
not representative benchmark results, and no overall trust score is synthesized.

## Learn more

- [Configuration, resume and migration from 0.3](https://howiehwong.github.io/TrustLLM/guides/running.html)
- [AI agent integration](https://howiehwong.github.io/TrustLLM/guides/agents.html)
- [FAQ and troubleshooting](https://howiehwong.github.io/TrustLLM/faq.html)
- [Report an issue](https://github.com/HowieHwong/TrustLLM/issues)
- [Citation](https://github.com/HowieHwong/TrustLLM/blob/main/CITATION.bib)

The benchmark evaluates model responses, including responses generated by local
and API models. Agent orchestration does not add benchmarks for tool use or
multi-step agent trajectories. See the documentation for the tested scope.
