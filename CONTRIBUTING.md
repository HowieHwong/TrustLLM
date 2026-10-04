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
python scripts/check_docs.py
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

## Package releases

The `python-publish.yml` workflow uses PyPI Trusted Publishing. A project owner
must configure the following GitHub publisher on the existing `trustllm` project
under **Manage → Publishing** on PyPI:

| PyPI publisher field | Value |
| --- | --- |
| Owner | `HowieHwong` |
| Repository | `TrustLLM` |
| Workflow filename | `python-publish.yml` |
| Environment | `pypi` |

This grants that repository workflow permission to publish the package without
storing a long-lived PyPI API token. Only the upload job requests an OIDC token;
building and testing run in a separate job. Keep repository write access limited
to maintainers and use GitHub's `pypi` environment protections as appropriate.
See [PyPI's publisher setup guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

1. Update the package version and PyPI-facing `trustllm_pkg/README.md`, record
   migration notes, and pass CI on the release commit.
2. Create a `v<version>` tag pointing to that commit. Run **Publish Python package**
   manually with that tag and `publish` unchecked to validate the release first.
3. Publish the GitHub release for that tag. The workflow checks the tag against
   package metadata, runs offline tests, builds an sdist and a wheel from it,
   checks metadata, and installs the wheel into a clean environment.
4. The upload job sends those tested artifacts to PyPI. The final job installs
   the exact version from PyPI and exercises both CLI entry points.
5. Confirm the version and artifacts on PyPI before announcing availability or
   replacing source-install instructions with commands for the new release.

For an authentication failure, correct the publisher configuration and rerun the
failed jobs. Manual dispatch with `publish` checked can also publish an existing
tag. PyPI versions are immutable: do not move a published tag, rebuild a changed
package under the same version, or use a new version to hide an upload failure.

## Documentation publishing

Write a unique `title` and `description` in each documentation page's frontmatter.
Use clear headings and accurate capability descriptions. Keep the original logo
and icon files unchanged. Retain existing page URLs or provide a useful page at
the old URL when reorganizing a guide.

`hooks/discovery.py` generates per-page Markdown, `llms.txt` and `llms-full.txt`
from the built articles. Edit the source pages, not generated files in `site/`.
Pages marked `archived: true` appear in the optional index section; pages marked
`agent_index: false` are omitted from the index. The build check validates local
links and anchors, canonical URLs, metadata, exports and example preservation.

The site uses static HTML, a sitemap and page descriptions for search discovery.
Agent exports are a convenience for clients that support them; they are not a
ranking guarantee. See [Google's AI search guidance](https://developers.google.com/search/docs/fundamentals/ai-optimization-guide),
the [llms.txt proposal](https://llmstxt.org/), and
[Mintlify's documentation export approach](https://www.mintlify.com/docs/ai/llmstxt).

GitHub Pages publishes this project under `/TrustLLM/`. Keep that prefix in
canonical and discovery URLs. A project-level `robots.txt` would not control
crawling for the host; do not add one as a substitute for root-host configuration.

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
