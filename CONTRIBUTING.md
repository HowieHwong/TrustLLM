# Contributing to TrustLLM

Contributions that make the benchmark easier to reproduce are welcome: bug reports, documentation, regression tests, and focused changes to model or evaluation adapters.

## Set up a development environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './trustllm_pkg[dev,docs]'
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
Add `local` for local generation or `eval` for scoring dependencies.
The base install and offline tests need neither API credentials nor a GPU. API integration tests use a loopback HTTP server. The optional CPU integration test constructs a tiny local model, runs real inference, and skips when local dependencies are absent.

## Validate a change

```bash
python -m ruff check .
python -m compileall -q trustllm_pkg/trustllm
python -m pytest
python -m build trustllm_pkg --outdir dist
python -m twine check dist/*
python -m mkdocs build --strict
```

The initial lint gate checks syntax and a small set of correctness rules across the repository. It does not claim the legacy code is fully lint-clean or reformat research code wholesale. New code should use four-space indentation, clear names, and docstrings for public behavior; `.editorconfig` captures the shared text conventions.

## Scope and testing

- Keep changes focused; preserve public function signatures and output keys unless a migration is documented.
- Add a regression test for a behavior change. Mock network requests and use temporary paths so default tests remain offline and do not spend API credits.
- Changes to prompts, labels, datasets, aggregation, or metric definitions can change benchmark scores. Explain the impact and compare against fixed fixtures.
- State exactly which live provider or GPU tests you ran, including model ID and environment. Offline CI alone is not evidence that these integrations work.
- Never commit credentials, downloaded models, private evaluation data, or generated output. Use `.env.example` as the configuration template.

## Submit a contribution

Open a focused pull request with the problem, resulting behavior, and validation. For bug reports, include the commit, Python version, relevant dependency versions, a minimal input, and the full traceback with credentials removed.

The Python package lives in `trustllm_pkg`; its `pyproject.toml` is the source of package metadata. The root `pyproject.toml` configures development tools. Release automation builds from `trustllm_pkg`, and documentation deployment runs separately from pull-request checks.

## README translations

The English `README.md` is the reference for translated READMEs in the repository root:
`README.zh-CN.md`, `README.zh-TW.md`, `README.ja.md`, `README.ko.md`,
`README.es.md`, and `README.fr.md`.

When changing installation commands, supported backends, output paths, version notes,
or limitations, update every language edition in the same change. Preserve command
flags, environment variable names, model IDs, links, and executable examples. Keep
the language switcher and original `images/logo.png` consistent across editions.
Translate explanatory prose and headings; a README translation does not imply
translated benchmark data or support for evaluating that language.
