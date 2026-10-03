<div align="center">
  <img src="images/logo.png" alt="TrustLLM" width="680">

# TrustLLM

**A benchmark and toolkit for trustworthiness in large language models.**

ICML 2024 · Six evaluation dimensions · 30+ datasets

[![CI](https://github.com/HowieHwong/TrustLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/HowieHwong/TrustLLM/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](trustllm_pkg/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ICML 2024](https://img.shields.io/badge/ICML-2024-6846C4)](https://arxiv.org/abs/2401.05561)

[Paper](https://arxiv.org/abs/2401.05561) ·
[Website](https://trustllmbenchmark.github.io/TrustLLM-Website/) ·
[Documentation](https://howiehwong.github.io/TrustLLM/) ·
[Dataset](https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset) ·
[Leaderboard](https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html)

</div>

---

TrustLLM provides datasets, response generation, and evaluation utilities for studying LLM trustworthiness. The original study evaluates 16 models across six dimensions:

| Dimension | What it measures |
| --- | --- |
| **Truthfulness** | Misinformation, hallucination, sycophancy, and factuality correction |
| **Safety** | Jailbreaks, misuse, toxicity, and exaggerated safety |
| **Fairness** | Stereotypes, preferences, and disparagement |
| **Robustness** | Adversarial perturbations and out-of-domain performance |
| **Privacy** | Privacy awareness and leakage |
| **Machine ethics** | Moral judgments, choices, and awareness |

See the [dataset and task reference](docs/benchmark.md) for individual datasets, metrics, and evaluation methods.

> **Project status:** This cleanup adds standard packaging, configuration, and offline checks. Generation and evaluation retain historical integrations and model IDs; live provider compatibility and GPU inference are not covered by CI. For newer work, see [TrustGen](https://trustgen.github.io/) and [TrustEval](https://github.com/TrustGen/TrustEval-toolkit).

## Quickstart

### 1. Install from source

Run these commands in a terminal (Python 3.9+):

```bash
git clone https://github.com/HowieHwong/TrustLLM.git
cd TrustLLM
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e './trustllm_pkg[dev]'
```

### 2. Run the offline checks

```bash
python -m pytest
python -m ruff check .
```

These tests cover configuration, JSON utilities, and dataset-download behavior using mocked HTTP responses. They do not download models or call paid APIs.

### 3. Prepare the dataset

The repository includes the benchmark archive, so no second download is needed:

```bash
python -m zipfile -e dataset/dataset.zip data
```

The dataset root is now `data/dataset`, with a subdirectory for each dimension. Alternatively, fetch the archive from GitHub:

```python
from trustllm.dataset_download import download_dataset

download_dataset(save_path="data")  # Extracts to data/dataset; overwrites existing dataset files.
```

## Run a benchmark

The lightweight install above does not include model runtimes. Install the historical benchmark dependencies before using generation or task modules:

```bash
python -m pip install -e './trustllm_pkg[benchmark]'
```

**Installation change:** Bare `pip install ./trustllm_pkg` now installs dataset utilities only. Existing generation/evaluation users should add `[benchmark]`. The extra preserves the previous SDK requirements; it is not a tested, locked environment for all providers. Python 3.9 was the original benchmark environment; the CI Python matrix applies to offline utilities only.

### Configure credentials

```bash
cp .env.example .env
```

Fill in only the providers you use, then load the file **before** importing TrustLLM configuration, generation, or evaluation modules:

```python
from dotenv import load_dotenv

load_dotenv()

from trustllm import config

# Existing Python configuration overrides remain supported:
# config.openai_key = "..."
```

You can also export environment variables directly. See [.env.example](.env.example) for supported names. Model aliases and task prompts remain in [config.py](trustllm_pkg/trustllm/config.py); historical provider IDs may need updating for your account. `OPENAI_BASE_URL` configures the evaluator's endpoint; legacy generation adapters do not all use this setting.

### Generate model responses

Example for a supported local model (model access and sufficient GPU memory required):

```python
from dotenv import load_dotenv

load_dotenv()

from trustllm.generation.generation import LLMGeneration

runner = LLMGeneration(
    model_path="lmsys/vicuna-7b-v1.3",
    test_type="safety",
    data_path="data/dataset",
    online_model=False,
    num_gpus=1,
    max_new_tokens=512,
    device="cuda",
)
runner.generation_results()
```

`test_type` accepts `truthfulness`, `safety`, `fairness`, `robustness`, `privacy`, or `ethics`. Use the dataset **root**, not an individual JSON file. Outputs are written under `generation_results/<model>/<dimension>/` relative to your working directory. See the [generation guide](docs/guides/generation_details.md) for historical adapters and settings.

### Evaluate responses

Pass **generated response files**, including their `res` fields, to the matching evaluator. Raw prompt datasets alone are not evaluation inputs.

```python
from dotenv import load_dotenv

load_dotenv()

from trustllm.task.pipeline import run_safety
from trustllm.utils.file_process import save_json

results = run_safety(
    all_folder_path="generation_results/vicuna-7b/safety",
)
save_json(results, "safety_results.json")
```

Some tasks use an LLM judge, embeddings, a downloaded classifier, or an external service. Check the [evaluation guide](docs/guides/evaluation.md) before running them; a complete benchmark can require network access, model downloads, and API charges. Preserve the commit, dependency versions, model IDs, generation settings, and judge settings with published results.

## Repository map

```text
TrustLLM/
├── trustllm_pkg/
│   ├── pyproject.toml       # Package metadata and dependency extras
│   └── trustllm/
│       ├── config.py        # Environment-backed runtime configuration
│       ├── dataset_download.py
│       ├── generation/      # Model response generation
│       ├── task/            # Six dimensions and evaluation pipelines
│       └── utils/           # Metrics, judges, embeddings, and JSON I/O
├── tests/                   # Offline regression tests
├── dataset/dataset.zip      # Bundled benchmark data
├── docs/                    # Guides and benchmark reference
└── .github/workflows/       # CI, documentation, and package publishing
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, and contribution expectations. To preview the documentation:

```bash
python -m pip install -e './trustllm_pkg[docs]'
python -m mkdocs serve
```

[Release history](docs/changelog.md) · [Report an issue](https://github.com/HowieHwong/TrustLLM/issues)

## Citation

If you use TrustLLM in your research, please cite the [ICML 2024 paper](https://openreview.net/forum?id=bWUU0LwwMp). The full author list and BibTeX entry are in [CITATION.bib](CITATION.bib).

## License

Code is released under the [MIT License](LICENSE). Consult the original dataset sources for their terms and attribution requirements.
