# TrustLLM

Trustworthiness evaluation for large language models · ICML 2024.

The base package provides dataset download, OpenAI-compatible API generation,
sample checkpoints, and HTML generation reports. Install `trustllm[local]` for
Hugging Face causal-model inference, or `trustllm[eval]` for the original scorers.

```bash
python -m trustllm download --output data
python -m trustllm tasks
python -m trustllm generate --help
```

```python
from trustllm import download_dataset, generate

download_dataset("data")
run = generate(model="your-model-id", backend="api", task="safety", limit=5)
```

Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` for your service. See the
[quickstart](https://github.com/HowieHwong/TrustLLM#start-here),
[documentation](https://howiehwong.github.io/TrustLLM/), and
[paper](https://arxiv.org/abs/2401.05561).
