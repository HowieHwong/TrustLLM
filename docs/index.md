---
hide:
  - navigation
---

# **TrustLLM: Trustworthiness in Large Language Models**

## **About**

TrustLLM is a comprehensive study of trustworthiness in large language models (LLMs), including principles for different dimensions of trustworthiness, established benchmark, evaluation, and analysis of trustworthiness for mainstream LLMs, and discussion of open challenges and future directions. The document explains how to use the trustllm python package to help you assess the performance of your LLM in trustworthiness more quickly. For more details about TrustLLM, please refer to [this link](https://trustllmbenchmark.github.io/TrustLLM-Website/).

<img src="https://raw.githubusercontent.com/TrustLLMBenchmark/TrustLLM-Website/main/img/logo.png" width="100%">

## **Before Evaluation**

### **Installation**

**Installation via `pip`:**

```shell
pip install trustllm
```

**Installation via `conda`:**

```sh
conda install -c conda-forge trustllm
```

**Installation via Github:**

```shell
git clone git@github.com:HowieHwong/TrustLLM.git
```

Creat a new environment:

```shell
conda create --name trustllm python=3.9
```

Install required packages:

```shell
cd trustllm_pkg
pip install .
```

### **Dataset Download**

1. Download TrustLLM dataset from Github:

```python
from trustllm.dataset_download import download_dataset

download_dataset(save_path='save_path')
```

2. Download TrustLLM dataset from [Hugginface]().

### **Generation**

!!! note

    Please note that the LLM you use for evaluation should have a certain level of utility. If its generation/NLP capabilities are weak, it may bias the evaluation results (for example, many evaluation samples may be considered invalid).


We have added generation section from [version 0.2.0](https://howiehwong.github.io/TrustLLM/changelog.html). Start your generation from [this page](https://howiehwong.github.io/TrustLLM/guides/generation_details.html).

[//]: # (The datasets are structured in JSON format, where each JSON file consists of a collection of `dict&#40;&#41;`. Within each `dict&#40;&#41;`, there is a key named `prompt`. Your should utilize the value of `prompt` key as the input for generation. After generation, you should store the output of LLMs as s new key named `res` within the same dictionary. Here is an example to generate answer from your LLM:)

[//]: # ()
[//]: # (For each dataset, we have configured the temperature setting during model generation. Please refer to [this page]&#40;guides/generation_details.md#generation-parameters&#41; for the settings.)

[//]: # ()
[//]: # (```python)

[//]: # (import json)

[//]: # ()
[//]: # (filename = 'dataset_path.json')

[//]: # ()
[//]: # (# Load the data from the file)

[//]: # (with open&#40;filename, 'r'&#41; as file:)

[//]: # (    data = json.load&#40;file&#41;)

[//]: # ()
[//]: # (# Process each dictionary and add the 'res' key with the generated output)

[//]: # (for element in data:)

[//]: # (    element['res'] = generation&#40;element['prompt']&#41;  # Replace 'generation' with your function)

[//]: # ()
[//]: # (# Write the modified data back to the file)

[//]: # (with open&#40;filename, 'w'&#41; as file:)

[//]: # (    json.dump&#40;data, file, indent=4&#41;)

[//]: # (```)

## **Start Your Evaluation**

See [this page](guides/evaluation.md) for more details.

## **Dataset & Task**

**Dataset overview**

| Dataset               | Description                                                                                                           | Num.    | Exist? | Section            |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------|---------|--------|--------------------|
| SQuAD2.0              | It combines questions in SQuAD1.1 with over 50,000 unanswerable questions.                                            | 100     | ✓      | Misinformation     |
| CODAH                 | It contains 28,000 commonsense questions.                                                                             | 100     | ✓      | Misinformation     |
| HotpotQA              | It contains 113k Wikipedia-based question-answer pairs for complex multi-hop reasoning.                               | 100     | ✓      | Misinformation     |
| AdversarialQA         | It contains 30,000 adversarial reading comprehension question-answer pairs.                                           | 100     | ✓      | Misinformation     |
| Climate-FEVER         | It contains 7,675 climate change-related claims manually curated by human fact-checkers.                              | 100     | ✓      | Misinformation     |
| SciFact               | It contains 1,400 expert-written scientific claims pairs with evidence abstracts.                                     | 100     | ✓      | Misinformation     |
| COVID-Fact            | It contains 4,086 real-world COVID claims.                                                                            | 100     | ✓      | Misinformation     |
| HealthVer             | It contains 14,330 health-related claims against scientific articles.                                                 | 100     | ✓      | Misinformation     |
| TruthfulQA            | The multiple-choice questions to evaluate whether a language model is truthful in generating answers to questions.     | 352     | ✓      | Hallucination      |
| HaluEval              | It contains 35,000 generated and human-annotated hallucinated samples.                                                | 300     | ✓      | Hallucination      |
| LM-exp-sycophancy     | A dataset consists of human questions with one sycophancy response example and one non-sycophancy response example.    | 179     | ✓      | Sycophancy         |
| Opinion pairs         | It contains 120 pairs of opposite opinions.                                                                           | 240     | ✗      | Sycophancy         |
| WinoBias              | It contains 3,160 sentences, split for development and testing, created by researchers familiar with the project.     | 734     | ✓      | Stereotype         |
| StereoSet             | It contains the sentences that measure model preferences across gender, race, religion, and profession.                | 734     | ✓      | Stereotype         |
| Adult                 | The dataset, containing attributes like sex, race, age, education, work hours, and work type, is utilized to predict salary levels for individuals. | 810     | ✓      | Disparagement      |
| Jailbraek Trigger     | The dataset contains the prompts based on 13 jailbreak attacks.                                                        | 1300    | ✗      | Jailbreak, Toxicity|
| Misuse (additional)   | This dataset contains prompts crafted to assess how LLMs react when confronted by attackers or malicious users seeking to exploit the model for harmful purposes. | 261     | ✗      | Misuse             |
| Do-Not-Answer         | It is curated and filtered to consist only of prompts to which responsible LLMs do not answer.                         | 344 + 95| ✓      | Misuse, Stereotype |
| AdvGLUE               | A multi-task dataset with different adversarial attacks.                                                               | 912     | ✓      | Natural Noise      |
| AdvInstruction        | 600 instructions generated by 11 perturbation methods.                                                                 | 1200    | ✗      | Natural Noise      |
| ToolE                 | A dataset with the users' queries which may trigger LLMs to use external tools.                                        | 241     | ✓      | Out of Domain (OOD)|
| Flipkart              | A product review dataset, collected starting from December 2022.                                                       | 400     | ✓      | Out of Domain (OOD)|
| DDXPlus               | A 2022 medical diagnosis dataset comprising synthetic data representing about 1.3 million patient cases.               | 100     | ✓      | Out of Domain (OOD)|
| ETHICS                | It contains numerous morally relevant scenarios descriptions and their moral correctness.                              | 500     | ✓      | Implicit Ethics    |
| Social Chemistry 101  | It contains various social norms, each consisting of an action and its label.                                          | 500     | ✓      | Implicit Ethics    |
| MoralChoice           | It consists of different contexts with morally correct and wrong actions.                                             | 668     | ✓      | Explicit Ethics    |
| ConfAIde              | It contains the description of how information is used.                                                               | 196     | ✓      | Privacy Awareness  |
| Privacy Awareness     | It includes different privacy information queries about various scenarios.                                            | 280     | ✗      | Privacy Awareness  |
| Enron Email           | It contains approximately 500,000 emails generated by employees of the Enron Corporation.                              | 400     | ✓      | Privacy Leakage    |
| Xstest                | It's a test suite for identifying exaggerated safety behaviors in LLMs.                                                | 200     | ✓      | Exaggerated Safety |

**Task overview**

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
| Agreement on Privacy Information             | Pearson’s correlation (↑)                 | Classification  | ●    | Privacy Awareness        |
| Privacy Scenario Test                        | RtA (↑)                                   | Generation      | ○    | Privacy Awareness        |
| Probing Privacy Information Usage            | RtA (↑), Accuracy (↓)                     | Generation      | ◐    | Privacy Leakage          |
| Moral Action Judgement                       | Accuracy (↑)                              | Classification  | ◐    | Implicit Ethics          |
| Moral Reaction Selection (Low-Ambiguity)     | Accuracy (↑)                              | Classification  | ◐    | Explicit Ethics          |
| Moral Reaction Selection (High-Ambiguity)    | RtA (↑)                                   | Generation      | ○    | Explicit Ethics          |
| Emotion Classification                       | Accuracy (↑)                              | Classification  | ●    | Emotional Awareness      |


## **Leaderboard**

If you want to view the performance of all models or upload the performance of your LLM, please refer to [this link](https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html).

## **Citation**

```text
@misc{sun2024trustllm,
      title={TrustLLM: Trustworthiness in Large Language Models}, 
      author={Lichao Sun and Yue Huang and Haoran Wang and Siyuan Wu and Qihui Zhang and Chujie Gao and Yixin Huang and Wenhan Lyu and Yixuan Zhang and Xiner Li and Zhengliang Liu and Yixin Liu and Yijue Wang and Zhikun Zhang and Bhavya Kailkhura and Caiming Xiong and Chaowei Xiao and Chunyuan Li and Eric Xing and Furong Huang and Hao Liu and Heng Ji and Hongyi Wang and Huan Zhang and Huaxiu Yao and Manolis Kellis and Marinka Zitnik and Meng Jiang and Mohit Bansal and James Zou and Jian Pei and Jian Liu and Jianfeng Gao and Jiawei Han and Jieyu Zhao and Jiliang Tang and Jindong Wang and John Mitchell and Kai Shu and Kaidi Xu and Kai-Wei Chang and Lifang He and Lifu Huang and Michael Backes and Neil Zhenqiang Gong and Philip S. Yu and Pin-Yu Chen and Quanquan Gu and Ran Xu and Rex Ying and Shuiwang Ji and Suman Jana and Tianlong Chen and Tianming Liu and Tianyi Zhou and Willian Wang and Xiang Li and Xiangliang Zhang and Xiao Wang and Xing Xie and Xun Chen and Xuyu Wang and Yan Liu and Yanfang Ye and Yinzhi Cao and Yong Chen and Yue Zhao},
      year={2024},
      eprint={2401.05561},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
