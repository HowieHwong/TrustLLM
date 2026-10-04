Source: https://howiehwong.github.io/TrustLLM/guides/running.html

# Run TrustLLM

## Install the components you use

The 0.4 source distribution has four installation modes:

| Extra | Purpose |
| --- | --- |
| Base install | Dataset download, CLI, compatible API generation, checkpoints, HTML reports |
| `local` | PyTorch, Transformers 4.x, Accelerate, SentencePiece |
| `eval` | Original scoring pipelines, classifier dependencies, API judge SDK and metrics |
| `legacy` | Archived generation engine and historical provider SDKs |

```
python -m pip install "trustllm @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
# For local generation and scoring together:
python -m pip install "trustllm[local,eval] @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
```

Git must be installed for VCS installation. Python 3.10–3.12 is recommended for local inference; offline utilities are tested on 3.9–3.12. GPU users should install a PyTorch build appropriate for their hardware first. For a source checkout, use `python -m pip install -e './trustllm_pkg[dev,local,eval]'`.

The PyPI release has not been changed by this source update. Pin a commit SHA instead of `main` and save your environment for published experiments.

## Download data

```
trustllm download --output data
trustllm tasks
```

Or:

```
from trustllm import download_dataset

download_dataset("data")
```

Data is extracted under `data/dataset`. Downloading again overwrites matching files. The helper raises on download or archive errors. No model libraries are imported by this command.

## API models

Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` in the environment. The default endpoint is `https://api.openai.com/v1`. Keys are optional for compatible local servers with authentication disabled.

```
trustllm generate --backend api --model your-served-model \
  --base-url http://localhost:8000/v1 \
  --task safety --data data/dataset --limit 5 --output runs/api-smoke
```

The transport sends text-only, non-streaming `POST /chat/completions` requests. Model IDs are passed through unchanged. Use an endpoint that implements this protocol (for example a vLLM-served model); this is not a native Anthropic Messages, Gemini, Azure deployment, or Responses API adapter.

- `--concurrency 4`: maximum simultaneous sample requests.
- `--timeout 120`: timeout in seconds per HTTP request.
- `--retries 3`: retries after the first attempt for connection errors, timeouts, HTTP 408/429/500/502/503/504. Other HTTP errors fail immediately.
- `--token-parameter max_completion_tokens`: use this instead of the default `max_tokens` if required by your endpoint.
- `--omit-temperature`: omit temperature entirely for models that do not accept it.

Retries may cause duplicate inference or billing if a provider completed a request before a connection failed. The client cannot guarantee provider-side idempotency.

## Local models

```
trustllm generate --backend local --model Qwen/Qwen2.5-0.5B-Instruct \
  --device auto --task safety --data data/dataset \
  --limit 5 --output runs/local-smoke
```

A Hugging Face model ID downloads weights using the normal HF cache. A filesystem path loads saved weights. `HF_TOKEN` and the normal Hugging Face authentication setup apply to gated models.

`AutoModelForCausalLM` and `AutoTokenizer` handle loading. A tokenizer chat template is applied when present; otherwise the prompt is used directly. The response excludes the input tokens.

- `--device cpu`, `cuda:0`, or `mps`: select one device.
- `--device auto`: use Accelerate model placement, potentially across devices.
- `--dtype auto|float32|float16|bfloat16`: model precision.
- `--revision <commit>`: pin a model revision.
- `--seed 42`: initialize local generation randomness.
- `--trust-remote-code`: opt in only when your chosen model requires custom code.

Local generation is sequential (`concurrency=1`). Use a serving API for concurrent requests. This implementation is text-only causal-LM inference; it does not implement multimodal, log-likelihood, tensor-parallel or GGUF loading itself. Reproducibility still depends on hardware, library versions and sampling; resumed stochastic local runs are not guaranteed to be bit-identical to an uninterrupted run.

## JSON configuration

```
trustllm generate --config examples/api-safety.json --model your-model-id
trustllm generate --config examples/local-safety.json
```

The example files are in the repository. Config keys match `trustllm.generate` keyword arguments, and explicit CLI flags override the JSON config. Keep secrets in environment variables; saved CLI configs reject `api_key`. Python callers can supply it directly.

## Inputs, checkpoints and output files

`--data` accepts a benchmark root, a dimension directory, or a single JSON file. A custom file must be a nonempty array of objects containing a nonempty `prompt` string (or the field selected by `--prompt-key`). All other input fields are preserved; existing `res` values in input data are regenerated. Dataset default temperatures match the original task registry unless overridden by `--temperature`.

`--limit` takes the first N samples of each file. It is a debugging aid, not a statistically representative subset, and some original scorers need complete groups or paired records.

```
runs/api-safety/
├── run.json                # Settings, input SHA-256, versions, completion counts
├── samples.jsonl           # Append-only sample successes and errors
├── report.html             # Self-contained generation report
├── jailbreak.json          # Original fields + res
├── misuse.json
└── exaggerated_safety.json
```

Outputs are ordered like the input, regardless of request completion order. Failed/pending responses are `null` and are rejected by the scoring entry point. A failed run produces artifacts and exits nonzero. Unexpected local failures are recorded by exception type to avoid persisting sensitive library error details.

To resume, repeat the original command and add `--resume`. Successful checkpoints are reused; failed samples are retried. Resume checks dataset hashes, model/backend configuration, generation settings and dependency versions. A changed run needs a new output directory. Do not run two processes into the same output directory. For existing local checkpoint directories, keep the files immutable: model-file contents are not hashed automatically.

## Score a full run

```
trustllm evaluate --task safety --data runs/api-safety-full
```

This calls the existing TrustLLM scorer after checking that all required response files exist and all responses are nonempty. It writes `scores.json` and a readable `scores.html` table. Metric directions and scales are preserved; no overall trust score is invented. The default ethics pipeline does not include the optional awareness scorer; safety toxicity remains an opt-in feature of the original Python pipeline. See the [scoring guide](https://howiehwong.github.io/TrustLLM/guides/evaluation.md) for individual tasks.

Judge model selection is configurable with `OPENAI_JUDGE_MODEL`. The original judge default is retained for compatibility, but historical IDs may no longer be served by your provider. Embedding and classifier choices remain in the original scoring code. Generation transport flexibility does not guarantee all scoring dependencies use the same provider.

## Migration from 0.3

`from trustllm.generation.generation import LLMGeneration` remains available and delegates to the new runner. `online_model=True` selects the API backend. Prefer explicit `backend='api'` or `backend='local'`; use actual model IDs, not historical aliases. `generation_results()` returns `"OK"` on success, and its summary is available as `.result`. It now raises on failure instead of silently printing an error and returning `None`.

Default outputs move from `generation_results/<alias>/<dimension>` to `runs/<sanitized-model>/<dimension>`. Specify `output_dir` to control this. `num_gpus` values other than 1 are rejected; use `device='auto'` for placement. The old fixed retry interval is replaced by bounded per-request backoff. Model chat templates can change prompt formatting relative to FastChat, so these are new experiment settings rather than exact reproductions of the paper.

For archived provider-specific integrations, import from `trustllm.generation.legacy` and install the `legacy` extra. Those providers are not modernized or covered by the new adapter tests.
