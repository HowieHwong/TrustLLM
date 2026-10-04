Source: https://howiehwong.github.io/TrustLLM/design.html

# Workflow design and references

Reviewed on 2026-10-03. These references inform the workflow; TrustLLM does not claim feature parity with these larger frameworks or copy their benchmark definitions.

| Reference | Practice adopted in this update |
| --- | --- |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Separate model-backend dependencies, explicit task discovery, CLI and Python entry points |
| [Inspect logs](https://inspect.aisi.org.uk/eval-logs.html) and [parallelism](https://inspect.aisi.org.uk/parallelism.html) | Per-sample records, explicit run status, configurable concurrency and recoverable runs |
| [EvalScope quickstart](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html) | Side-by-side API/local examples, small-sample trials and reusable configuration |
| [Transformers chat templates](https://huggingface.co/docs/transformers/v4.57.1/chat_templating) | Format local-model inputs with the tokenizer's native chat template |

## What this update validates

Offline tests cover download errors, configuration, API transport, bounded retries, model-independent input/output handling, resume compatibility, incomplete-response rejection and report escaping. A real loopback HTTP server exercises the CLI request path. A tiny locally constructed causal model exercises CPU inference and saved-checkpoint loading without downloading external weights.

## What remains separate

The original six-dimension scoring methods and prompts remain. This update does not claim to reproduce every published benchmark score or validate every hosted model provider. Full GPU runs, paid judge/embedding integrations, multi-node inference, multimodal tasks and a live dashboard are outside these checks. A generation report's completion rate must never be interpreted as a trustworthiness score.
