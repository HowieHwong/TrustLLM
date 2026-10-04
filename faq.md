Source: https://howiehwong.github.io/TrustLLM/faq.html

# FAQ & troubleshooting

## Why does pip install trustllm behave differently?

The source repository contains the 0.4 workflow; the older package on PyPI has not been updated by this source change. Follow the [source installation commands](https://howiehwong.github.io/TrustLLM/guides/running.html#install-the-components-you-use). Use `python -m trustllm --help` to verify that your Python environment exposes `download`, `tasks`, `generate` and `evaluate`.

## Do I need a GPU?

API generation and dataset downloads use the base package without PyTorch. Local inference requires the `local` extra and enough memory for the selected model; CPU inference is supported. Scoring uses the `eval` extra and may load classifiers or call external judge/embedding services. Start with a small generation smoke test.

## Which APIs and local models work?

The maintained API backend implements text-only, non-streaming OpenAI-compatible Chat Completions. The local backend loads Transformers causal language models. Model IDs pass through to the chosen backend. Native Anthropic Messages, Gemini, Azure deployment endpoints and OpenAI Responses require a separate compatible bridge or adapter. See the [backend comparison](https://howiehwong.github.io/TrustLLM/guides/generation_details.md).

## Can I use this with an AI agent?

Yes. An agent can call the Python API or CLI and read JSON artifacts. Start with the [AI agent integration guide](https://howiehwong.github.io/TrustLLM/guides/agents.md) or the [agent documentation index](https://howiehwong.github.io/TrustLLM/llms.txt). The current benchmark scores model responses, not multi-step agent trajectories or tool-use correctness.

## Where did the dataset go?

`python -m trustllm download --output data` extracts the benchmark to `data/dataset`. Pass that directory to `--data`. The download helper also works through Python. Repeating the download overwrites matching files.

## Why did five samples produce more than five requests?

`--limit 5` selects the first five records of **each** dataset file in the selected task. Retries can add requests. A small prefix is a debugging aid; it is not a representative or paired benchmark subset.

## Can I resume after an API failure?

Repeat the exact generation command with `--resume`. Successful samples are reused and failed samples are retried. Dataset hashes, backend/model settings, generation options and dependency versions must match. Choose a new output directory if they change. Never share one output directory between simultaneous runs. Retry details are in the [running guide](https://howiehwong.github.io/TrustLLM/guides/running.html#api-models).

## Why is scoring rejecting my responses?

The CLI requires every registered response file for the selected dimension, excluding optional awareness, and a nonempty `res` for every record. Inspect `run.json` and `samples.jsonl`, resolve failed/pending responses, then resume. A limited smoke test can still miss groups required by an original scorer. Generate a full response set before reporting benchmark scores.

Existing score files are not overwritten. Use `--output runs/my-run/scores-rerun.json` when rescoring; the paired HTML filename must also be unused.

## Is a 100% completion report a perfect benchmark score?

No. `report.html` reports generation completion. `scores.json` and `scores.html` are produced separately by evaluation and contain per-task metrics. Their scales and directions differ; TrustLLM defines no overall trust score.

## Which judge model should I use?

Select a model available to your account with `OPENAI_JUDGE_MODEL` before starting evaluation. The original default is historical and may be unavailable. Record the model, endpoint and prompts when comparing results. Judge and embedding compatibility must be checked separately from generation compatibility. Consult the [scoring guide](https://howiehwong.github.io/TrustLLM/guides/evaluation.md).

## Language bias

Translated READMEs explain installation; they do not translate or validate the benchmark. As discussed in the paper, response language can affect evaluation. The original [Longformer refusal classifier](https://huggingface.co/LibrAI/longformer-harmful-ro) performs poorly on Chinese responses. The historical refusal-to-answer calculation excludes responses above a Chinese-character ratio threshold, with a default of 0.3. Do not interpret cross-language score differences without reviewing this filtering and the task assumptions.

## How should I report a bug?

Include the source commit or installed version, Python version, backend, sanitized command, task, traceback and a minimal example. Remove API keys and private responses. The [contribution guide](https://howiehwong.github.io/TrustLLM/development.md) explains how to run checks and submit focused fixes.
