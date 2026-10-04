---
title: Changelog and migration notes
description: Track the TrustLLM 0.4 source workflow, documentation updates and historical releases. Find migration guidance and distinguish source changes from PyPI releases.
---

# Changelog

Changes to the maintained source repository are listed first. Historical package releases remain below for research reproducibility.

## Documentation update · October 2026

- Reorganized installation, backend selection, scoring and original API references.
- Refreshed the documentation layout, typography, search and mobile navigation while preserving the original TrustLLM logo and icon.
- Added AI agent integration instructions covering CLI/Python orchestration, exit codes, artifacts and benchmark scope.
- Added canonical URLs, sitemap, page descriptions, social previews and repository metadata.
- Added Markdown page exports, `llms.txt` and `llms-full.txt`, generated from the same content as the HTML documentation.
- Expanded troubleshooting and linked the seven README languages.

## Source update: 0.4.0

- Shared local/API generation runner, CLI and Python API.
- Dataset download without cloning, task discovery and JSON configurations.
- Per-sample checkpoints, guarded resume, bounded API retries and explicit failures.
- Input hashes, environment versions, response exports and HTML generation reports.
- Separate local/evaluation dependencies; original generation engine archived.
- Multilingual READMEs and CPU/HTTP integration tests.

This source update does not by itself publish a PyPI release. Install from source using the [current instructions](guides/running.md#install-the-components-you-use). Existing users should read the [migration guide](guides/running.md#migration-from-03).

## Historical releases

Provider and model support below describes the release at that time. It does not establish current availability or coverage by the maintained backend tests.

### Version 0.3.0

April 23, 2024

- Parallel embedding retrieval for AdvInstruction evaluation.
- Exception handling for partial evaluations and bug fixes.
- Added published results for ChatGLM3, GLM-4, Mixtral and Llama 3 models. See the [research leaderboard](https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html).

### Version 0.2.3 & 0.2.4

March 2024

- Bug fixes and Gemini API support in the historical generation engine.

### Version 0.2.2

February 1, 2024

- Awareness evaluation from [related research](https://arxiv.org/abs/2401.17882).
- Zhipu API support for GLM-4 and GLM-3-turbo.

### Version 0.2.1

January 26, 2024

- Historical Replicate, DeepInfra and Azure OpenAI integrations.
- Simplified evaluation pipelines.

### Version 0.2.0

January 20, 2024

- Added model generation and concurrent automatic evaluation.

### Version 0.1.0

January 10, 2024

- First release of the TrustLLM assessment toolkit, covering the evaluation methods from the initial paper.
