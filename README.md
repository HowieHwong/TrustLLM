<p align="center">
  <img src="images/logo.png" alt="TrustLLM — Trustworthiness in Large Language Models" width="760">
</p>
<p align="center"><sub>ICML 2024 &nbsp; · &nbsp; TRUSTWORTHINESS IN LARGE LANGUAGE MODELS</sub></p>

<p align="center">
  <a href="https://arxiv.org/abs/2401.05561">Paper</a> &nbsp; / &nbsp;
  <a href="https://howiehwong.github.io/TrustLLM/">Documentation</a> &nbsp; / &nbsp;
  <a href="https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset">Dataset</a> &nbsp; / &nbsp;
  <a href="https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html">Leaderboard</a>
</p>

<!-- Keep language links and code examples in sync across all README translations. -->
<p align="center">
  <strong>English</strong> &nbsp; / &nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp; / &nbsp;
  <a href="README.zh-TW.md">繁體中文</a> &nbsp; / &nbsp;
  <a href="README.ja.md">日本語</a> &nbsp; / &nbsp;
  <a href="README.ko.md">한국어</a> &nbsp; / &nbsp;
  <a href="README.es.md">Español</a> &nbsp; / &nbsp;
  <a href="README.fr.md">Français</a>
</p>

TrustLLM is an open research toolkit for evaluating the trustworthiness of large language models across **six dimensions**. Run the ICML 2024 benchmark with local weights or a model API, and keep the data, settings, and results together.

## Start here

**1 — Install.** The base package supports API generation and data downloads; no GPU libraries required.

```bash
python -m pip install "trustllm @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
```

These commands install the **0.4 source version**. The older `pip install trustllm` package on PyPI has not been updated by this change. For reproducible runs, replace `main` with a commit SHA.

**2 — Get the benchmark.**

```bash
python -m trustllm download --output data
python -m trustllm tasks
```

**3 — Try your API model.** Set the credentials and endpoint for your service:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"

python -m trustllm generate \
  --backend api --model your-model-id \
  --task safety --data data/dataset \
  --limit 5 --concurrency 4 --output runs/api-safety
```

Use your provider's actual API root in place of the example URL. For a local OpenAI-compatible server, use `http://localhost:8000/v1` and its served model ID; an API key is optional if the server does not require one. API mode uses text-only Chat Completions.

Open `runs/api-safety/report.html` to inspect completion counts. `--limit 5` selects five samples **per dataset file** for a smoke test; remove it and choose a new output directory for a full run. This report tracks generation, not benchmark scores.

## Local weights, same workflow

```bash
python -m pip install "trustllm[local] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"

python -m trustllm generate \
  --backend local --model Qwen/Qwen2.5-0.5B-Instruct \
  --task safety --data data/dataset \
  --device auto --limit 5 --output runs/local-safety
```

Hugging Face weights download on first use. Replace the model ID with `/path/to/checkpoint` to use existing weights. `--device cpu`, `cuda:0`, and `mps` select a device; `auto` uses Accelerate placement. Local loading supports Transformers causal language models and uses the tokenizer's chat template when available. Model access, hardware capacity, and architecture compatibility still apply.

## Prefer Python?

```python
from trustllm import download_dataset, generate

download_dataset("data")

run = generate(
    model="your-model-id",
    backend="api",                    # Switch to "local" for HF weights.
    task="safety",
    data_path="data/dataset",
    output_dir="runs/python-safety",
    limit=5,
)
print(run["status"], run["successful"], run["total"])
```

API settings come from `OPENAI_API_KEY` and `OPENAI_BASE_URL`, or explicit `api_key` and `base_url` Python arguments. See [usage and configuration](docs/guides/running.md) for JSON configs, retries, model revisions, token settings, and resuming a run.

## From responses to scores

Install the scoring dependencies, then evaluate a complete directory of generated responses:

```bash
python -m pip install "trustllm[eval] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
python -m trustllm evaluate --task safety --data runs/api-safety-full
```

Replace the path with your **full-run** output directory. Scores are written to `scores.json` and `scores.html`; incomplete responses are rejected. The six scoring pipelines retain the original benchmark methods. Depending on the task, scoring uses rules, a downloaded classifier, embeddings, or an API judge. Set `OPENAI_JUDGE_MODEL` for a judge available to your account. Judge calls can incur charges. Read the [scoring guide](docs/guides/evaluation.md) before a full run.

| Dimension | Evaluation focus |
| :--- | :--- |
| **Truthfulness** | Misinformation · hallucination · sycophancy |
| **Safety** | Jailbreaks · misuse · exaggerated safety |
| **Fairness** | Stereotypes · preferences · disparagement |
| **Robustness** | Adversarial perturbations · out-of-domain inputs |
| **Privacy** | Privacy awareness · information leakage |
| **Ethics** | Moral judgments · moral choices |

[Dataset and metric reference →](docs/benchmark.md)

<details>
<summary><b>What changed in 0.4?</b></summary>

- Arbitrary model IDs and a shared runner for API/local generation; no legacy model whitelist.
- Lightweight API install; optional `local`, `eval`, and archived `legacy` dependencies.
- `download`, `tasks`, `generate`, and `evaluate` commands, also through `python -m trustllm`.
- Bounded API retries, per-sample checkpoints, explicit failures, and guarded `--resume`.
- JSON response files remain compatible with the existing `res`-based evaluators.
- Saved dataset hashes, settings, dependency versions, usage where returned, and HTML generation reports.

The original generation engine is archived under `trustllm.generation.legacy`. Generation formatting and model integrations have changed: new runs are not automatically equivalent to the original paper's settings. See [migration notes](docs/guides/running.md#migration-from-03).

</details>

## Research & development

[CI checks](https://github.com/HowieHwong/TrustLLM/actions/workflows/ci.yml) · [Contributing](CONTRIBUTING.md) · [Design references](docs/design.md) · [Changelog](docs/changelog.md) · [Issues](https://github.com/HowieHwong/TrustLLM/issues)

The language links above translate this README. Detailed guides are currently in English; translations do not change benchmark prompts, datasets, or scoring methods.

If TrustLLM supports your research, please cite the [ICML 2024 paper](https://openreview.net/forum?id=bWUU0LwwMp). Full BibTeX: [CITATION.bib](CITATION.bib). Code: [MIT](LICENSE). Dataset terms remain with the original sources.
