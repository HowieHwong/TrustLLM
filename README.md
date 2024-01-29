
<div align="center">


<img src="https://raw.githubusercontent.com/TrustLLMBenchmark/TrustLLM-Website/main/img/logo.png" width="100%">

# **TrustLLM: Trustworthiness in Large Language Models**


[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%8D-blue?style=flat-square)](https://trustllmbenchmark.github.io/TrustLLM-Website/)
[![Paper](https://img.shields.io/badge/Paper-%F0%9F%8E%93-lightgrey?style=flat-square)](https://arxiv.org/abs/2401.05561)
[![Dataset](https://img.shields.io/badge/Dataset-%F0%9F%92%BE-green?style=flat-square)](https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset)
[![Data Map](https://img.shields.io/badge/Data%20Map-%F0%9F%8D%9F-orange?style=flat-square)](https://atlas.nomic.ai/map/f64e87d3-c769-4a90-b15d-9dc833acc8ba/8e9d7045-503b-4ba0-bc64-7201cb7aacee?xs=-16.14086&xf=-1.88776&ys=-7.54937&yf=3.88213)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-%F0%9F%9A%80-brightgreen?style=flat-square)](https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html)
[![Toolkit Document](https://img.shields.io/badge/Toolkit%20Document-%F0%9F%93%9A-blueviolet?style=flat-square)](https://howiehwong.github.io/TrustLLM/)
[![Code](https://img.shields.io/badge/Code-%F0%9F%90%99-red?style=flat-square)](https://github.com/HowieHwong/TrustLLM)



<img src="https://img.shields.io/github/last-commit/HowieHwong/TrustLLM?style=flat-square&color=5D6D7E" alt="git-last-commit" />
<img src="https://img.shields.io/github/commit-activity/m/HowieHwong/TrustLLM?style=flat-square&color=5D6D7E" alt="GitHub commit activity" />
<img src="https://img.shields.io/github/languages/top/HowieHwong/TrustLLM?style=flat-square&color=5D6D7E" alt="GitHub top language" />
</div>

<div align="center">




</div>


## Updates & News

- [29/01/2024] :star: Version 0.2.1: trustllm toolkit now supports (1) Easy evaluation pipeline (2)LLMs in [replicate](https://replicate.com/) and [deepinfra](https://deepinfra.com/) (3) [Azure OpenAI API](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- [20/01/2024] :star: Version 0.2.0 of trustllm toolkit is released! See the [new features](https://howiehwong.github.io/TrustLLM/changelog.html#version-020).
- [12/01/2024] :surfer: The [dataset](https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset), [leaderboard](https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html), and [evaluation toolkit](https://howiehwong.github.io/TrustLLM/) are released!




**Table of Content**

- [About](#about)
- [Dataset & Task](#dataset--task)
  - [Dataset overview:](#dataset-overview)
  - [Task overview:](#task-overview)
- [Before Evaluation](#before-evaluation)
  - [Installation](#installation)
  - [Dataset Download](#dataset-download)
  - [Generation](#generation)
- [Evaluation](#evaluation)
- [Leaderboard](#leaderboard)
- [Contribution](#contribution)
- [Citation](#citation)
- [License](#license)

## **About**

We introduce TrustLLM, a comprehensive study of trustworthiness in LLMs, including principles for different dimensions of trustworthiness, established benchmark, evaluation, and analysis of trustworthiness for mainstream LLMs, and discussion of open challenges and future directions. Specifically, we first propose a set of principles for trustworthy LLMs that span eight different dimensions. Based on these principles, we further establish a benchmark across six dimensions including truthfulness, safety, fairness, robustness, privacy, and machine ethics. 
We then present a study evaluating 16 mainstream LLMs in TrustLLM, consisting of over 30 datasets. 
The [document](https://howiehwong.github.io/TrustLLM/#about) explains how to use the trustllm python package to help you assess the performance of your LLM in trustworthiness more quickly. For more details about TrustLLM, please refer to [project website](https://trustllmbenchmark.github.io/TrustLLM-Website/).

<div align="center">
<img src="https://raw.githubusercontent.com/TrustLLMBenchmark/TrustLLM-Website/main/img/benchmark_arch_00.png" width="100%">
</div>

## **Dataset & Task**

### **Dataset overview:**

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
| Jailbraek Trigger     | The dataset contains the prompts based on 13 jailbreak attacks.                                                        | 1300     | ✗      | Jailbreak, Toxicity    |
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

### **Task overview:**

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
| Agreement on Privacy Information             | Pearson’s correlation (↑)                 | Classification  | ●    | Privacy Awareness        |
| Privacy Scenario Test                        | RtA (↑)                                   | Generation      | ○    | Privacy Awareness        |
| Probing Privacy Information Usage            | RtA (↑), Accuracy (↓)                     | Generation      | ◐    | Privacy Leakage          |
| Moral Action Judgement                       | Accuracy (↑)                              | Classification  | ◐    | Implicit Ethics          |
| Moral Reaction Selection (Low-Ambiguity)     | Accuracy (↑)                              | Classification  | ◐    | Explicit Ethics          |
| Moral Reaction Selection (High-Ambiguity)    | RtA (↑)                                   | Generation      | ○    | Explicit Ethics          |
| Emotion Classification                       | Accuracy (↑)                              | Classification  | ●    | Emotional Awareness      |


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

Create a new environment:

```shell
conda create --name trustllm python=3.9
```

Install required packages:

```shell
cd trustllm_pkg
pip install .
```

### **Dataset Download**

Download TrustLLM dataset:

```python
from trustllm.dataset_download import download_huggingface_dataset

download_huggingface_dataset(save_path='save_path')
```

### **Generation**

We have added generation section from [version 0.2.0](https://howiehwong.github.io/TrustLLM/changelog.html). Start your generation from [this page](https://howiehwong.github.io/TrustLLM/guides/generation_details.html).


## **Evaluation**

See our [docs](https://howiehwong.github.io/TrustLLM/) for more details.

## **Leaderboard**

If you want to view the performance of all models or upload the performance of your LLM, please refer to [this link](https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html).


## **Contribution**

We welcome your contributions, including but not limited to the following:

- New evaluation datasets
- Research on trustworthy issues
- Improvements to the toolkit

If you intend to make improvements to the toolkit, please fork the repository first, make the relevant modifications to the code, and finally initiate a `pull request`.

## **⏰ TODO in Coming Versions**

- [x] Faster and simpler evaluation pipeline  (**Version 0.2.1**)
- [ ] Dynamic dataset  
- [ ] More fine-grained datasets
- [ ] Chinese output evaluation
- [ ] Downstream application evaluation


## **Citation**

```text
@misc{sun2024trustllm,
      title={TrustLLM: Trustworthiness in Large Language Models}, 
      author={Lichao Sun and Yue Huang and Haoran Wang and Siyuan Wu and Qihui Zhang and Chujie Gao and Yixin Huang and Wenhan Lyu and Yixuan Zhang and Xiner Li and Zhengliang Liu and Yixin Liu and Yijue Wang and Zhikun Zhang and Bhavya Kailkhura and Caiming Xiong and Chaowei Xiao and Chunyuan Li and Eric Xing and Furong Huang and Hao Liu and Heng Ji and Hongyi Wang and Huan Zhang and Huaxiu Yao and Manolis Kellis and Marinka Zitnik and Meng Jiang and Mohit Bansal and James Zou and Jian Pei and Jian Liu and Jianfeng Gao and Jiawei Han and Jieyu Zhao and Jiliang Tang and Jindong Wang and John Mitchell and Kai Shu and Kaidi Xu and Kai-Wei Chang and Lifang He and Lifu Huang and Michael Backes and Neil Zhenqiang Gong and Philip S. Yu and Pin-Yu Chen and Quanquan Gu and Ran Xu and Rex Ying and Shuiwang Ji and Suman Jana and Tianlong Chen and Tianming Liu and Tianyi Zhou and William Wang and Xiang Li and Xiangliang Zhang and Xiao Wang and Xing Xie and Xun Chen and Xuyu Wang and Yan Liu and Yanfang Ye and Yinzhi Cao and Yong Chen and Yue Zhao},
      year={2024},
      eprint={2401.05561},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HowieHwong/TrustLLM&type=Date)](https://star-history.com/#HowieHwong/TrustLLM&Date)



## **License**

The code in this repository is open source under the [MIT license](https://github.com/HowieHwong/TrustLLM/blob/main/LICENSE).
