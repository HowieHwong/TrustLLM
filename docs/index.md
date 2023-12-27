# **TustLLM: Trustworthiness in Large Language Models**

<img src="../images/logo.png" width="100%">

## **About**



## **Getting Start**

### Installation

Installation can be done using pypi:

```shell
pip install trustllm
```

### Before Evaluation

Download TrustLLM dataset:

```python
from trustllm import dataset_download

download_huggingface_dataset(save_path='save_path')
```


!!! note

    The datasets are all in `JSON` format, with each JSON file containing a list of many dictionaries. Each dictionary has a fixed key `prompt`. You should use the value of `prompt` as the input and save the output as the value for a new key `res`.

Here is an example to generate answer from your LLM:

```python
import json

filename = 'dataset_path'

with open(filename, 'r') as f:
    data = json.load(f)

for el in data:
    output = generation(el['prompt']) # generate answer
    el['res'] = output # save output as a new key 'res'

with open(filename, 'w') as f:
    json.dumps(data, f, indent=4)
```



## **Citation**
```shell

```


