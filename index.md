Source: https://howiehwong.github.io/TrustLLM/index.html

![TrustLLM — Trustworthiness in Large Language Models](https://howiehwong.github.io/TrustLLM/assets/logo.png)

OPEN RESEARCH · ICML 2024

# LLM trustworthiness, measured.

Evaluate the models behind your applications across six dimensions. TrustLLM brings benchmark data, local and API generation, and the original research scorers into one Python workflow.

[Run your first evaluation](https://howiehwong.github.io/TrustLLM/guides/running.md)
[Integrate with an AI agent](https://howiehwong.github.io/TrustLLM/guides/agents.md)

## 01 / Set up

Install only the components you need. Download benchmark data from Python or the CLI.

[Installation & data →](https://howiehwong.github.io/TrustLLM/guides/running.md)

## 02 / Run a model

Use local Hugging Face weights or a compatible API. Keep checkpoints and resume interrupted runs.

[Model backends →](https://howiehwong.github.io/TrustLLM/guides/generation_details.md)

## 03 / Read the evidence

Score full response sets, inspect each metric, and keep the experiment settings alongside the results.

[Scoring & results →](https://howiehwong.github.io/TrustLLM/guides/evaluation.md)

## What does TrustLLM evaluate?

| Dimension | Questions the benchmark explores |
| --- | --- |
| **Truthfulness** | Does the model produce misinformation, hallucinate, or agree with a user's false premise? |
| **Safety** | How does it respond to jailbreaks, misuse requests, and harmless prompts that trigger excessive refusal? |
| **Fairness** | Does it express stereotypes, demographic preferences, or disparaging judgments? |
| **Robustness** | How do perturbations and out-of-distribution inputs affect its responses? |
| **Privacy** | Does it recognize privacy concerns or disclose sensitive information? |
| **Ethics** | How does it reason about moral judgments and choices? |

[Explore the datasets and original metrics](https://howiehwong.github.io/TrustLLM/benchmark.md). TrustLLM reports individual metrics with their original directions and scales; it does not collapse them into an overall trust score.

## One workflow, two model backends

```
python -m pip install "trustllm @ git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg"
python -m trustllm download --output data
python -m trustllm tasks
```

The source installation above provides the current **0.4 workflow**. The PyPI release may differ. API generation uses the lightweight base package; install the `local` extra for local inference and `eval` for scoring. The [first-run guide](https://howiehwong.github.io/TrustLLM/guides/running.md) covers credentials, examples and model requirements.

An AI agent can orchestrate this workflow through the CLI or Python, then read JSON artifacts. See the [agent integration guide](https://howiehwong.github.io/TrustLLM/guides/agents.md) for commands, exit codes and output contracts. The benchmark evaluates model responses; multi-step agent trajectories, tool-use correctness and agent memory are outside its current scope.

## Research and reproducibility

TrustLLM accompanies **TrustLLM: Trustworthiness in Large Language Models**, published at ICML 2024. Prompts and original scoring methods remain part of the research benchmark. New backends, model versions and judge settings can change results; record them when comparing experiments.

[Paper](https://arxiv.org/abs/2401.05561) · [Dataset](https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset) · [Published leaderboard](https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html) · [Validation scope](https://howiehwong.github.io/TrustLLM/design.md)

Documentation is in English. The repository also offers introductions in [简体中文](https://github.com/HowieHwong/TrustLLM/blob/main/README.zh-CN.md), [繁體中文](https://github.com/HowieHwong/TrustLLM/blob/main/README.zh-TW.md), [日本語](https://github.com/HowieHwong/TrustLLM/blob/main/README.ja.md), [한국어](https://github.com/HowieHwong/TrustLLM/blob/main/README.ko.md), [Español](https://github.com/HowieHwong/TrustLLM/blob/main/README.es.md) and [Français](https://github.com/HowieHwong/TrustLLM/blob/main/README.fr.md). README translations do not translate the benchmark data.
