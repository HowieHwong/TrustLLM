Source: https://howiehwong.github.io/TrustLLM/guides/agents.html

# Use TrustLLM from an AI agent

An AI coding or research agent can use TrustLLM to **evaluate an underlying language model**: download data, generate responses, inspect failures and invoke the original scorers. The same commands work from a terminal, a subprocess or an orchestration tool.

TrustLLM does not currently measure multi-step agent trajectories, tool-use correctness, memory, browser actions or end-to-end agent task success. This guide describes integration with an agent, not a new agent benchmark or MCP server.

## Give an agent the documentation

Start from the [documentation index](https://howiehwong.github.io/TrustLLM/llms.txt). It links to a Markdown version of each page. [Complete documentation](https://howiehwong.github.io/TrustLLM/llms-full.txt) is also available when a client needs one file.

Each HTML page advertises its Markdown alternative and the agent index through HTML link elements. These files are generated from the rendered documentation on every build, including code examples and tables.

A useful task to give your agent:

> Read https://howiehwong.github.io/TrustLLM/llms.txt and the agent integration guide. Set up a TrustLLM safety smoke test for the model and endpoint I provide. Use five samples per dataset file and a new output directory. Read credentials from the environment. Report the command, exit status and completion counts from run.json. Explain failures before retrying. Treat report.html as a generation report. Only run a full benchmark or paid scoring when those costs are authorized.

## Discover and install

```
python -m pip install "trustllm @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
python -m trustllm --help
python -m trustllm tasks
python -m trustllm download --output data
```

These commands install the maintained workflow directly from GitHub. Replace `main` with a reviewed commit SHA for a reproducible environment. Downloading again overwrites matching dataset files under `data/dataset`.

The six task IDs are `truthfulness`, `safety`, `fairness`, `robustness`, `privacy` and `ethics`. `tasks` prints a human-readable list of their registered dataset files; it does not emit JSON.

## Run a bounded smoke test

Configure `OPENAI_API_KEY` and `OPENAI_BASE_URL` in the environment, then replace `your-served-model` with the actual model ID:

```
python -m trustllm generate \
  --backend api --model your-served-model \
  --task safety --data data/dataset \
  --limit 5 --concurrency 2 --max-new-tokens 128 \
  --output runs/agent-safety-smoke
```

`--limit 5` means five records **per file**, not five requests for the entire task. Retries can add requests. API generation uses text-only Chat Completions; the endpoint must implement that protocol. For local generation, install the `local` extra, use `--backend local`, supply a Hugging Face model ID or local checkpoint path, and set `--concurrency 1`. See [model backends](https://howiehwong.github.io/TrustLLM/guides/generation_details.md).

### Python orchestration

```
from trustllm import generate

run = generate(
    model="your-served-model",
    backend="api",
    task="safety",
    data_path="data/dataset",
    output_dir="runs/agent-python-smoke",
    limit=5,
    concurrency=2,
    max_new_tokens=128,
)
print(run["status"], run["successful"], run["total"])
```

Python generation returns the run summary on success and raises on failure. Use a unique output directory for each configuration. Keep secrets in environment variables; saved JSON configs reject `api_key`.

## Read the artifacts

| Artifact | Agent usage |
| --- | --- |
| `run.json` | Read `status`, aggregate `successful`, `failed` and `total` counts, plus per-file `pending` counts under `files`, settings, versions and input hashes. |
| `samples.jsonl` | Inspect per-sample `success` or `error` events. Repeated attempts can produce more than one event for a sample. |
| Dataset-named `.json` files | Original input records plus `res`. Failed or pending responses are `null`. |
| `report.html` | Human-readable generation completion report. It contains no benchmark scores. |
| `scores.json` | After scoring: `task`, `scores`, input hashes, dependency versions and judge model. Preserve each metric's meaning. |
| `scores.html` | Human-readable score table generated alongside `scores.json`. |

Read JSON files instead of scraping terminal progress. Generation states are `running`, `completed`, `failed` and `interrupted`. Counts and per-file summaries are written when generation finishes or unwinds; a process killed before finalization may leave only the initial `running` manifest and journal. `pending` samples have not produced a successful response or recorded failure at the last saved update.

| CLI exit code | Meaning |
| --- | --- |
| `0` | The command completed successfully. |
| `1` | A handled setup, input, generation or scoring error; inspect stderr and available artifacts. |
| `2` | Argument parsing failed. |
| `130` | Keyboard interruption; completed samples can be resumed. |

Other nonzero exits may come from an unexpected exception or process termination. A setup failure can occur before artifacts exist; inspect the exit code first.

## Recover, then score a complete run

For a failed or interrupted generation run, repeat the **same** command with `--resume`. Successful checkpoints are reused and failed samples are retried. Changing input hashes, generation settings or dependency versions requires a new output directory. Never write to the same output directory from two processes. See [resume details](https://howiehwong.github.io/TrustLLM/guides/running.html#inputs-checkpoints-and-output-files).

For a benchmark result, generate again **without `--limit` into a new directory**, then install the `eval` extra and score the complete response set:

```
python -m pip install "trustllm[eval] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
python -m trustllm evaluate --task safety --data runs/agent-safety-full
```

The directory above must already contain the full run. A smoke test cannot establish benchmark performance. Scoring validates required files and nonempty responses, but cannot prove you ran the full benchmark; preserve the data and manifest. Some scoring pipelines download classifiers or call paid judge/embedding services. Set `OPENAI_JUDGE_MODEL` to an available judge where needed and review the [scoring guide](https://howiehwong.github.io/TrustLLM/guides/evaluation.md).

## Record evidence for a report

Include the source commit, model ID/revision, dataset hashes, full configuration, dependency versions, sample counts, failures, judge settings and the per-task score files. Label partial runs clearly. Generation completion rate is not a trustworthiness metric; TrustLLM does not define a single overall trust score.
