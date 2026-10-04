Source: https://howiehwong.github.io/TrustLLM/guides/generation_details.html

# Local models & API endpoints

Both backends use the same task registry, response format, checkpoint journal and generation reports. Choose a backend based on where inference runs.

|  | API backend | Local backend |
| --- | --- | --- |
| Installation | Base package | `local` extra |
| Model | Served model ID | Hugging Face causal-LM ID or local checkpoint path |
| Inference | Text-only `POST /chat/completions` | Transformers `AutoModelForCausalLM` |
| Hardware | Managed by the endpoint | Your CPU, CUDA GPU or Apple MPS device |
| Parallel samples | `--concurrency N` | Sequential; `--concurrency 1` |
| Authentication | `OPENAI_API_KEY` if required | Hugging Face authentication for gated weights |

## API generation

Set the endpoint and credentials in your environment. This example assumes a local server that implements the OpenAI-compatible Chat Completions protocol:

```
python -m trustllm generate \
  --backend api --model your-served-model \
  --base-url http://localhost:8000/v1 \
  --task safety --data data/dataset \
  --limit 5 --concurrency 4 --output runs/api-smoke
```

The model ID must match your server. For a hosted service, set `OPENAI_BASE_URL` and `OPENAI_API_KEY` or use `--base-url` for the endpoint. The default API root is `https://api.openai.com/v1`.

A vLLM server can expose this protocol; TrustLLM does not launch or manage the server itself. Compatible text responses are required. There are no native Anthropic Messages, Gemini, Azure deployment or Responses API adapters in the maintained runner. Support for generation through an endpoint does not establish compatibility with the original judge or embedding integrations.

Some endpoints require `--token-parameter max_completion_tokens` or `--omit-temperature`. Request timeouts and bounded retries are configurable; see [API options](https://howiehwong.github.io/TrustLLM/guides/running.html#api-models).

## Local inference

Install `local` using the [source installation instructions](https://howiehwong.github.io/TrustLLM/guides/running.html#install-the-components-you-use), then run:

```
python -m trustllm generate \
  --backend local --model Qwen/Qwen2.5-0.5B-Instruct \
  --device auto --task safety --data data/dataset \
  --limit 5 --output runs/local-smoke
```

Weights download through the Hugging Face cache on first use. Replace the model ID with a saved checkpoint directory to load existing weights. Tokenizers with a chat template use that template; otherwise the prompt is passed directly.

Use `--device cpu`, `cuda:0` or `mps` to choose one device. `auto` delegates placement to Accelerate. Use `--revision <commit>` to pin hosted model weights and `--seed 42` to initialize randomness. Hardware capacity and Transformers architecture compatibility still apply. GGUF, multimodal inference and log-likelihood evaluation are outside this backend.

## Inspect and resume

`--limit` is per dataset file. A smoke test checks the setup; it does not establish a benchmark score.

Open `report.html` in the output directory and inspect `run.json` for counts and settings. Repeat the original command with `--resume` to reuse successful checkpoints. A changed configuration or input requires a new output directory. Read the [artifact and resume contract](https://howiehwong.github.io/TrustLLM/guides/running.html#inputs-checkpoints-and-output-files) before automating runs.

## Migrate an existing experiment

The current compatibility class `trustllm.generation.generation.LLMGeneration` delegates to this runner. Use actual model IDs and explicit backends. See [migration from 0.3](https://howiehwong.github.io/TrustLLM/guides/running.html#migration-from-03).

The [archived generation guide](https://howiehwong.github.io/TrustLLM/guides/legacy_generation.md) documents the old provider SDKs and model aliases. Those integrations are retained for reference and are not covered by the maintained backend tests.
