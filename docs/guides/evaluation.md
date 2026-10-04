---
title: Score LLM benchmark responses
description: Evaluate complete TrustLLM response sets, configure scoring dependencies and judges, and interpret per-task JSON and HTML benchmark results.
---

# Scoring model responses

Generation records what a model says. Evaluation applies the original TrustLLM scorers to those responses and writes **per-task metrics**. Complete generation before starting evaluation, and keep the original dataset fields and ordering.

## Prepare a complete run

Start with the [running guide](running.md). A generation smoke test with `--limit` is useful for checking setup, but may omit pairs or groups required by the scorers. Generate without a limit into a new directory for a full benchmark run.

```bash
python -m pip install "trustllm[eval] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
python -m trustllm evaluate --task safety --data runs/api-safety-full
```

Replace the example directory with an existing full response set. The CLI requires every registered response file for the task, excluding optional awareness, and a nonempty `res` for every record. It cannot prove benchmark coverage from arbitrary input files: retain the data hashes and run manifest, and verify the sample counts yourself.

## Configure scoring dependencies

The `eval` extra installs the original evaluation dependencies, including PyTorch, Transformers, metrics libraries and the judge SDK. Different tasks use a mixture of rules, classifiers, embeddings and language-model judges. Some may download model weights or call paid services.

For a scoring method that uses an API judge, set these variables **before** importing evaluation modules or running the CLI:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"
export OPENAI_JUDGE_MODEL="your-available-judge-model"
```

Replace the endpoint and judge placeholders with your provider's actual values. The original default judge ID is historical and may no longer be available. A generation endpoint's Chat Completions support does not guarantee compatibility with all original judge or embedding calls. Check the selected pipeline before launching a full experiment.

The standard ethics pipeline excludes optional awareness scoring. Safety toxicity is an opt-in feature of the original Python pipeline and uses Perspective API credentials; the CLI does not enable it by default. See the [original APIs](../reference/scoring.md) for these options.

## Read the score files

| File | Contents |
| :--- | :--- |
| `scores.json` | Task, nested score values, response-file hashes, dependency versions, judge model and creation time. |
| `scores.html` | A readable table of the same score values. |
| `run.json` | Generation settings and completion counts; retain it with the score files. |
| `report.html` | Generation progress and completion only. This is not a benchmark score report. |

```python
import json
from pathlib import Path

result = json.loads(Path("runs/api-safety-full/scores.json").read_text())
print(result["task"])
print(result["scores"])
```

Existing scores are not overwritten. To rescore with a different judge or environment, select a new output filename and record the changed settings:

```bash
python -m trustllm evaluate --task safety --data runs/api-safety-full \
  --output runs/api-safety-full/scores-rerun.json
```

Both the chosen JSON and its corresponding HTML filename must be unused. A failed or incomplete evaluation does not produce a successful score artifact; inspect the error and the relevant pipeline dependencies.

## Interpret and compare results

Metric directions and scales differ. For example, higher refusal-to-answer rates can be desirable on harmful prompts and undesirable on harmless prompts in exaggerated-safety evaluation. Use the [original metric reference](../benchmark.md#task-overview), and preserve its definitions when reporting results. TrustLLM does not define a single overall trust score.

Record the dataset and response hashes, sample counts, source commit, model revision, prompt formatting, generation settings, dependencies and judge configuration. Treat new chat templates or judge versions as experimental changes. A 100% generation completion rate says nothing about a model's trustworthiness.

Review [language limitations](../faq.md#language-bias) and the [validation scope](../design.md) before comparing with the published leaderboard. Current tests do not reproduce all paper results or validate every hosted provider.

## Original Python APIs

The detailed [scoring reference](../reference/scoring.md) retains the research APIs. These links also preserve bookmarks from earlier documentation.

<span id="start-your-evaluation"></span><span id="api-setting"></span><span id="easy-pipeline"></span>
[Configuration and pipeline setup](../reference/scoring.md#start-your-evaluation)

| Dimension | Pipeline | Task API |
| :--- | :--- | :--- |
| Truthfulness | <span id="truthfulness-evaluation"></span>[Pipeline](../reference/scoring.md#truthfulness-evaluation) | <span id="truthfulness"></span>[Task methods](../reference/scoring.md#truthfulness) |
| Safety | <span id="safety-evaluation"></span>[Pipeline](../reference/scoring.md#safety-evaluation) | <span id="safety"></span>[Task methods](../reference/scoring.md#safety) |
| Fairness | <span id="fairness-evaluation"></span>[Pipeline](../reference/scoring.md#fairness-evaluation) | <span id="fairness"></span>[Task methods](../reference/scoring.md#fairness) |
| Robustness | <span id="robustness-evaluation"></span>[Pipeline](../reference/scoring.md#robustness-evaluation) | <span id="robustness"></span>[Task methods](../reference/scoring.md#robustness) |
| Privacy | <span id="privacy-evaluation"></span>[Pipeline](../reference/scoring.md#privacy-evaluation) | <span id="privacy"></span>[Task methods](../reference/scoring.md#privacy) |
| Ethics | <span id="ethics-evaluation"></span>[Pipeline](../reference/scoring.md#ethics-evaluation) | <span id="machine-ethics"></span>[Task methods](../reference/scoring.md#machine-ethics) |
