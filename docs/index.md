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

Download TrustLLM dataset:

```python
from trustllm.dataset_download import download_huggingface_dataset

download_huggingface_dataset(save_path='save_path')
```


### **Generation**

!!! note

    Please note that the LLM you use for evaluation should have a certain level of utility. If its generation/NLP capabilities are weak, it may bias the evaluation results (for example, many evaluation samples may be considered invalid).


The datasets are structured in JSON format, where each JSON file consists of a collection of `dict()`. Within each `dict()`, there is a key named `prompt`. Your should utilize the value of `prompt` key as the input for generation. After generation, you should store the output of LLMs as s new key named `res` within the same dictionary. Here is an example to generate answer from your LLM:

For each dataset, we have configured the temperature setting during model generation. Please refer to [this page](guides/generation_details.md#generation-parameters) for the settings.

```python
import json

filename = 'dataset_path.json'

# Load the data from the file
with open(filename, 'r') as file:
    data = json.load(file)

# Process each dictionary and add the 'res' key with the generated output
for element in data:
    element['res'] = generation(element['prompt'])  # Replace 'generation' with your function

# Write the modified data back to the file
with open(filename, 'w') as file:
    json.dump(data, file, indent=4)
```

## **Start Your Evaluation**

See [this page](guides/evaluation.md) for more details.

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
