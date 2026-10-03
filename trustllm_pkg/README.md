# TrustLLM

Toolkit for **TrustLLM: Trustworthiness in Large Language Models (ICML 2024)**.

The base installation includes dataset utilities. Install `trustllm[benchmark]`
for the historical generation and evaluation dependencies. For source installs,
run `python -m pip install -e "./trustllm_pkg[benchmark]"` from the repository root.

See the [repository quickstart](https://github.com/HowieHwong/TrustLLM#quickstart),
[documentation](https://howiehwong.github.io/TrustLLM/), and
[paper](https://arxiv.org/abs/2401.05561).

The benchmark integrations retain historical model IDs and SDK requirements.
Offline CI does not validate live providers or GPU inference.
