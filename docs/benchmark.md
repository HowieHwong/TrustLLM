---
title: LLM trustworthiness datasets and metrics
description: Explore the original TrustLLM benchmark datasets and metric definitions for truthfulness, safety, fairness, robustness, privacy and machine ethics.
---

# Benchmark reference

This reference records the original research benchmark. Counts and metrics below describe that benchmark, not a live inventory of every downloaded file. Use `python -m trustllm tasks` to inspect the current task registry and record dataset hashes for your experiment. See the [paper](https://arxiv.org/abs/2401.05561) for methodology and the [scoring guide](guides/evaluation.md) for the maintained workflow.

## ETHICS justice label correction · October 15, 2026

The bundled `dataset/ethics/implicit_ETHICS.json` now maps the original ETHICS
justice labels as `1 → not wrong` and `0 → wrong`. Its commonsense subset uses
the opposite numeric mapping, `1 → wrong` and `0 → not wrong`. Previously the
commonsense mapping had also been applied to justice.

The direction follows the [original ETHICS task definition and Figure 2](https://arxiv.org/html/2008.02275v6#S2.F2):
the figure's reasonable/unreasonable examples correspond to labels 1/0 in the
official CSV. The original [reader](https://github.com/hendrycks/ethics/blob/fa022dbc62dc6736c2f3780ced6412cae97efb90/utils.py#L83-L88)
and [training code](https://github.com/hendrycks/ethics/blob/fa022dbc62dc6736c2f3780ced6412cae97efb90/justice/tune.py#L54-L103)
use those labels without inversion.

All 500 bundled samples were checked against the official train, test, and
test-hard splits. Each matched one source record without conflicting labels:
100 train and 150 test records per topic. Exactly 250 justice labels changed;
all 250 commonsense records, prompts, input text, record order, and the other
30 ZIP members' contents are unchanged. The
[per-record correction list](https://github.com/HowieHwong/TrustLLM/blob/main/dataset/ethics_justice_corrections.tsv)
records zero-based JSON indices, source CSV record numbers (including the header),
source labels, and SHA256 hashes of the unchanged UTF-8 input strings.

There were 497 exact text matches. For indices 340, 341, and 419, source identity
was established by reversing UTF-8 bytes misdecoded as GBK:
`original.encode("gbk").decode("utf-8")`. Each recovered text matched one source
record, and converting it back reproduced the original string exactly. This was
used only for matching; the existing `boy鈥檚` and `toup茅e` text remains unchanged.

The [official ETHICS archive](https://people.eecs.berkeley.edu/~hendrycks/ethics.tar)
used for this check is 35,585,024 bytes, with SHA256
`40acbf1ac0da79a2aabef394d58889136b8d38b05be09482006de2453fb06333`.
Its source CSV hashes are:

| Source file | SHA256 |
| --- | --- |
| `justice/justice_train.csv` | `175c8255f5ee3d99ad381cd716473367274dc24b3832d5e3357f37f1779a7d3f` |
| `justice/justice_test.csv` | `d7cc506a1055915d455a9244b26de573c75a760d416a8508a9420d41cbf40250` |
| `commonsense/cm_train.csv` | `286cb4098a66b71819493f964cf2afe62cdbb6c6b41983b9c0b8fc484650c752` |
| `commonsense/cm_test.csv` | `a004f51b8f863d85eb5575d86e8723897f850fcb7fc8de38c1f30105dfb13c4d` |

### Reproducing an earlier dataset revision

This correction changes scoring. Historical published results have not been
recomputed, and scores using the two revisions are not interchangeable. The
downloader fetches the archive from `main`, so pinning the Python package alone
does not pin the dataset. For the previous revision, use the
[archive at the fixed pre-correction commit](https://raw.githubusercontent.com/HowieHwong/TrustLLM/c3edca246f39f778f666366c1e30cfdb95b2e43e/dataset/dataset.zip)
and verify the hash before extracting into a separate data directory.

| Artifact | Previous SHA256 | Corrected SHA256 |
| --- | --- | --- |
| `dataset/dataset.zip` | `0433f79bc8dc8a5869af253d4a70067eda82f5651b89b7235a0220a8bf775317` | `b9e3c7a83a74ce2e065627466c59ced4376cedcc0761b501c5d65665365871d8` |
| `dataset/ethics/implicit_ETHICS.json` | `07374119724cafe57d96b643283c0cc6180bd43d0151377944ef6db593cd5a95` | `75971ae958a854cfc280a01cad318d4b7eaee129bf85b0ba0a7fdd8fee5ba444` |

Generated response files keep the dataset fields they were created from,
including `label`, and the scorer reads labels from those files rather than
from the dataset archive. Rescoring existing `implicit_ETHICS.json` responses
after updating the dataset therefore still uses the old justice labels; relabel
them by index with the per-record correction list above, or generate them
again, before scoring. The maintained runner refuses `--resume` when a dataset
hash has changed, but the archived `trustllm.generation.legacy` engine has no
such check and resumes from an existing results file with its old labels.

## Dataset overview


*✓ the dataset is from prior work, and ✗ means the dataset is first proposed in our benchmark.*

| Dataset               | Description                                                                                                           | Num.     | Exist? | Section                |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------|----------|--------|------------------------|
| SQuAD2.0              | It combines questions in SQuAD1.1 with over 50,000 unanswerable questions.                                            | 100      | ✓      | Misinformation         |
| CODAH                 | It contains 28,000 commonsense questions.                                                                             | 100      | ✓      | Misinformation         |
| HotpotQA              | It contains 113k Wikipedia-based question-answer pairs for complex multi-hop reasoning.                               | 100      | ✓      | Misinformation         |
| AdversarialQA         | It contains 30,000 adversarial reading comprehension question-answer pairs.                                           | 100      | ✓      | Misinformation         |
| Climate-FEVER         | It contains 7,675 climate change-related claims manually curated by human fact-checkers.                              | 100      | ✓      | Misinformation         |
| SciFact               | It contains 1,400 expert-written scientific claims pairs with evidence abstracts.                                     | 100      | ✓      | Misinformation         |
| COVID-Fact            | It contains 4,086 real-world COVID claims.                                                                            | 100      | ✓      | Misinformation         |
| HealthVer             | It contains 14,330 health-related claims against scientific articles.                                                 | 100      | ✓      | Misinformation         |
| TruthfulQA            | The multiple-choice questions to evaluate whether a language model is truthful in generating answers to questions.     | 352      | ✓      | Hallucination          |
| HaluEval              | It contains 35,000 generated and human-annotated hallucinated samples.                                                | 300      | ✓      | Hallucination          |
| LM-exp-sycophancy     | A dataset consists of human questions with one sycophancy response example and one non-sycophancy response example.    | 179      | ✓      | Sycophancy             |
| Opinion pairs         | It contains 120 pairs of opposite opinions.                                                                           | 240, 120 | ✗      | Sycophancy, Preference |
| WinoBias              | It contains 3,160 sentences, split for development and testing, created by researchers familiar with the project.     | 734      | ✓      | Stereotype             |
| StereoSet             | It contains the sentences that measure model preferences across gender, race, religion, and profession.                | 734      | ✓      | Stereotype             |
| Adult                 | The dataset, containing attributes like sex, race, age, education, work hours, and work type, is utilized to predict salary levels for individuals. | 810      | ✓      | Disparagement          |
| Jailbreak Trigger     | The dataset contains the prompts based on 13 jailbreak attacks.                                                        | 1300     | ✗      | Jailbreak, Toxicity    |
| Misuse (additional)   | This dataset contains prompts crafted to assess how LLMs react when confronted by attackers or malicious users seeking to exploit the model for harmful purposes. | 261      | ✗      | Misuse                 |
| Do-Not-Answer         | It is curated and filtered to consist only of prompts to which responsible LLMs do not answer.                         | 344 + 95 | ✓      | Misuse, Stereotype     |
| AdvGLUE               | A multi-task dataset with different adversarial attacks.                                                               | 912      | ✓      | Natural Noise          |
| AdvInstruction        | 600 instructions generated by 11 perturbation methods.                                                                 | 600        | ✗      | Natural Noise          |
| ToolE                 | A dataset with the users' queries which may trigger LLMs to use external tools.                                        | 241      | ✓      | Out of Domain (OOD)    |
| Flipkart              | A product review dataset, collected starting from December 2022.                                                       | 400      | ✓      | Out of Domain (OOD)    |
| DDXPlus               | A 2022 medical diagnosis dataset comprising synthetic data representing about 1.3 million patient cases.               | 100      | ✓      | Out of Domain (OOD)    |
| ETHICS                | It contains numerous morally relevant scenarios descriptions and their moral correctness.                              | 500      | ✓      | Implicit Ethics        |
| Social Chemistry 101  | It contains various social norms, each consisting of an action and its label.                                          | 500      | ✓      | Implicit Ethics        |
| MoralChoice           | It consists of different contexts with morally correct and wrong actions.                                             | 668      | ✓      | Explicit Ethics        |
| ConfAIde              | It contains the description of how information is used.                                                               | 196      | ✓      | Privacy Awareness      |
| Privacy Awareness     | It includes different privacy information queries about various scenarios.                                            | 280      | ✗      | Privacy Awareness      |
| Enron Email           | It contains approximately 500,000 emails generated by employees of the Enron Corporation.                              | 400      | ✓      | Privacy Leakage        |
| Xstest                | It's a test suite for identifying exaggerated safety behaviors in LLMs.                                                | 200      | ✓      | Exaggerated Safety     |

## Task overview

*○ means evaluation through the automatic scripts (e.g., keywords matching), ● means the automatic evaluation by ChatGPT, GPT-4 or longformer, and ◐ means the mixture evaluation.*

*More trustworthy LLMs are expected to have a higher value of the metrics with ↑ and a lower value with ↓.*

| Task Name                                    | Metrics                                   | Type            | Eval | Section                  |
|----------------------------------------------|-------------------------------------------|-----------------|------|--------------------------|
| Closed-book QA                               | Accuracy (↑)                              | Generation      | ○    | Misinformation(Internal) |
| Fact-Checking                                | Macro F-1 (↑)                             | Classification  | ●    | Misinformation(External) |
| Multiple Choice QA                           | Accuracy (↑)                              | Classification  | ●    | Hallucination            |
| Hallucination Classification                 | Accuracy (↑)                              | Classification  | ●    | Hallucination            |
| Persona Sycophancy                           | Embedding similarity (↑)                  | Generation      | ◐    | Sycophancy               |
| Opinion Sycophancy                           | Percentage change (↓)                     | Generation      | ○    | Sycophancy               |
| Factuality Correction                        | Percentage change (↑)                     | Generation      | ○    | Adversarial Factuality   |
| Jailbreak Attack Evaluation                  | RtA (↑)                                   | Generation      | ○    | Jailbreak                |
| Toxicity Measurement                         | Toxicity Value (↓)                        | Generation      | ●    | Toxicity                 |
| Misuse Evaluation                            | RtA (↑)                                   | Generation      | ○    | Misuse                   |
| Exaggerated Safety Evaluation                | RtA (↓)                                   | Generation      | ○    | Exaggerated Safety       |
| Agreement on Stereotypes                     | Accuracy (↑)                              | Generation      | ◐    | Stereotype               |
| Recognition of Stereotypes                   | Agreement Percentage (↓)                  | Classification  | ◐    | Stereotype               |
| Stereotype Query Test                        | RtA (↑)                                   | Generation      | ○    | Stereotype               |
| Preference Selection                         | RtA (↑)                                   | Generation      | ○    | Preference               |
| Salary Prediction                            | p-value (↑)                               | Generation      | ●    | Disparagement            |
| Adversarial Perturbation in Downstream Tasks | ASR (↓), RS (↑)                           | Generation      | ◐    | Natural Noise            |
| Adversarial Perturbation in Open-Ended Tasks | Embedding similarity (↑)                  | Generation      | ◐    | Natural Noise            |
| OOD Detection                                | RtA (↑)                                   | Generation      | ○    | Out of Domain (OOD)      |
| OOD Generalization                           | Micro F1 (↑)                              | Classification  | ○    | Out of Domain (OOD)      |
| Agreement on Privacy Information             | Pearson's correlation (↑)                 | Classification  | ●    | Privacy Awareness        |
| Privacy Scenario Test                        | RtA (↑)                                   | Generation      | ○    | Privacy Awareness        |
| Probing Privacy Information Usage            | RtA (↑), Accuracy (↓)                     | Generation      | ◐    | Privacy Leakage          |
| Moral Action Judgement                       | Accuracy (↑)                              | Classification  | ◐    | Implicit Ethics          |
| Moral Reaction Selection (Low-Ambiguity)     | Accuracy (↑)                              | Classification  | ◐    | Explicit Ethics          |
| Moral Reaction Selection (High-Ambiguity)    | RtA (↑)                                   | Generation      | ○    | Explicit Ethics          |
| Emotion Classification                       | Accuracy (↑)                              | Classification  | ●    | Emotional Awareness      |
