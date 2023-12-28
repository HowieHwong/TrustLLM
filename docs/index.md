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

Installation can be done using pypi:

```shell
pip install trustllm
```

### **Dataset Download**

Download TrustLLM dataset:

```python
from trustllm import dataset_download

download_huggingface_dataset(save_path='save_path')
```


### **Generation**

!!! note

    Please note that the LLM you use for evaluation should have a certain level of utility. If its generation/NLP capabilities are weak, it may bias the evaluation results (for example, many evaluation samples may be considered invalid).





The datasets are uniformly structured in JSON format, where each JSON file consists of a collection of dictionaries. Within each dictionary, there is a key named `prompt`. Your should utilize the value of `prompt` key as the input for generation. After generation, you should store the output of LLMs as s new key named `res` within the same dictionary. Here is an example to generate answer from your LLM:

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




### **API Setting**
Before starting the evaluation, you need to first set up your [OpenAI API](https://openai.com/product) (GPT-4-turbo) and [Perspective API](https://developers.perspectiveapi.com/s/docs-get-started?language=en_US) (used for measuring toxicity).

```python
from trustllm import config

config.openai_key = 'your-openai-api-key'

config.perspective_key = 'your-perspective-api-key'
```

### **Truthfulness**


### **Safety**

Four subsections in safety evaluation:

* Jailbreak: `jailbreak.json`
* Exaggerated Safety: `exaggerated_safety.json`
* Toxicity: `jailbreak.json`
* Misuse: `misuse.json`

Requirement:

* openai api (gpt-4-turbo)
* perspective api
* huggingface evaluator: LibrAI/longformer-harmful-ro

Preliminary:



```python
from trustllm.task import safety
from trustllm.utils import file_process
from trustllm import config

evaluator = safety.SafetyEval()
```

Jailbreak evaluation (`eval_type`: type of evaluation, either `total` or `single`):

```python
jailbreak_data = file_process.load_json('jailbreak_data_json_path')
print(evaluator.jailbreak_eval(jailbreak_data, eval_type='total')) # return overall RtA
print(evaluator.jailbreak_eval(jailbreak_data, eval_type='single')) # return RtA dict for each kind of jailbreak ways
```

Exaggerated evaluation:

```python
exaggerated_data = file_process.load_json('exaggerated_data_json_path')
print(evaluator.exaggerated_eval(exaggerated_data))
```

Toxicity evaluation:

!!! note

    Toxicity data for evaluation is based on the evaluated results of jailbreak data. You should first evaluate jailbreak data and then use evaluated results to conduct toxicity evaluation.
    

```python
# If you have not saved evaluated jailbreak results
jailbreak_data = file_process.load_json('jailbreak_data_json_path')
eval_data, _ = evaluator.jailbreak_eval(jailbreak_data, eval_type='total/single', return_dat=True)
print(evaluator.toxicity_eval(eval_data, perspective_api=config.perspective_key))

# If you have saved evaluated jailbreak results
toxicity_data = file_process.load_json('evaluated_jailbreak_results_json_path') # load eval data for toxicity evaluation
print(evaluator.toxicity_eval(toxicity_data, perspective_api=config.perspective_key))
```

Misuse evaluation:

```python
misuse_data = file_process.load_json('misuse_data.json')
print(evaluator.misuse_eval(misuse_data))
```

### **Fairness**

Three subsections in fairness evaluation:

* Stereotype: `stereotype_agreement.json`, `stereotype_recognition.json`, `stereotype_query_test.json`
* Disparagement: `disparagement.json`
* Preference: `preference_force.json`

Requirement:

* openai api (gpt-4-turbo)
* perspective api
* huggingface evaluator: LibrAI/longformer-harmful-ro

Preliminary:

```python
from trustllm.task import fairness
from trustllm.utils import file_process
from trustllm import config

evaluator = fairness.FairnessEval()
```

Stereotype evaluation:

```python
stereotype_recognition_data = file_process.load_json('stereotype_recognition_data_json_path')
print(evaluator.stereotype_recognition_eval(stereotype_recognition_data))

stereotype_agreement_data = file_process.load_json('stereotype_agreement_data_json_path')
print(evaluator.stereotype_agreement_eval(stereotype_agreement_data))

stereotype_query_test_data = file_process.load_json('stereotype_query_test_data_json_path')
print(evaluator.stereotype_query_eval(stereotype_query_test_data))
```

Disparagement evaluation:
```python
disparagement_data = file_process.load_json('disparagement_data_json_path')
print(evaluator.disparagement_eval(disparagement_data))
```

Preference evaluation:
```python
preference_data = file_process.load_json('preference_data_json_path')
evaluator.preference_eval(preference_data)
```

### **Robustness**



### **Privacy**


### **Machine Ethics**




## **Leaderboard**

If you want to view the performance of all models or upload the performance of your LLM, please refer to [this link](https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html).


## **Citation**
```shell

```


