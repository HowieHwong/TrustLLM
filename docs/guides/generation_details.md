

## **Generation Results**


The trustllm toolkit currently supports the generation of over a dozen models. 
You can use the trustllm toolkit to generate output results for specified models on the trustllm benchmark.

### **Supported LLMs**

- `Baichuan-13b`
- `Baichuan2-13b`
- `Yi-34b`
- `ChatGLM2 - 6B`
- `ChatGLM3 -6B`
- `Vicuna-13b`
- `Vicuna-7b`
- `Vicuna-33b`
- `Llama2-7b`
- `Llama2-13b`
- `Llama2-70b`
- `Koala-13b`
- `Oasst-12b`
- `Wizardlm-13b`
- `Mixtral-8x7B`
- `Mistral-7b`
- `Dolly-12b`
- `bison-001-text`
- `ERNIE`
- `ChatGPT (gpt-3.5-turbo)`
- `GPT-4`
- `Claude-2`
- ***other LLMs in huggingface***

### **Start Your Generation**

The `LLMGeneration` class is designed for result generation, supporting the use of both ***local*** and ***online*** models. It is used for evaluating the performance of models in different tasks such as ethics, privacy, fairness, truthfulness, robustness, and safety.

**API setting:**

If you need to evaluate an API LLM, please set the following API according to your requirements.

```python
from trustllm import config

config.deepinfra_api = "deepinfra api"

config.claude_api = "claude api"

config.openai_key = "openai api"

config.palm_api = "palm api"

config.ernie_client_id = "ernie client id"

config.ernie_client_secret = "ernie client secret"

config.ernie_api = "ernie api"
```

**Generation template:**

```python
from trustllm.generation.generation import LLMGeneration

llm_gen = LLMGeneration(
    model_path="your model name", 
    test_type="test section", 
    data_path="your dataset file path",
    model_name="", 
    online_model=False, 
    temperature=1.0, 
    repetition_penalty=1.0,
    num_gpus=1, 
    max_new_tokens=512, 
    debug=False,
    device='cuda:0'
)

llm_gen.generation_results()
```

**Args:**

- `model_path` (`Required`, `str`): Path to the local model. LLM list: 

```text
'baichuan-inc/Baichuan-13B-Chat', 
'baichuan-inc/Baichuan2-13B-chat', 
'01-ai/Yi-34B-Chat', 
'THUDM/chatglm2-6b', 
'THUDM/chatglm3-6b', 
'lmsys/vicuna-13b-v1.3', 
'lmsys/vicuna-7b-v1.3', 
'lmsys/vicuna-33b-v1.3', 
'meta-llama/Llama-2-7b-chat-hf', 
'meta-llama/Llama-2-13b-chat-hf', 
'TheBloke/koala-13B-HF', 
'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5', 
'WizardLM/WizardLM-13B-V1.2', 
'mistralai/Mixtral-8x7B-Instruct-v0.1', 
'meta-llama/Llama-2-70b-chat-hf', 
'mistralai/Mistral-7B-Instruct-v0.1', 
'databricks/dolly-v2-12b', 
'bison-001', 
'ernie', 
'chatgpt', 
'gpt-4', 
'claude-2'
... (other LLMs in huggingface)
```

- `test_type` (`Required`, `str`): Type of evaluation task, including `'robustness'`, `'truthfulness'`, `'fairness'`, `'ethics'`, `'safety'`, `'privacy'`.
- `data_path` (`Required`, `str`): Path to the dataset, default is 'TrustLLM'.
- `online_model` (`Optional`, `bool`): Whether to use an online model, default is False.
- `temperature` (`Optional`, `float`): Temperature setting for text generation, default is 1.0.
- `repetition_penalty` (`Optional`, `float`): Repetition penalty setting, default is 1.0.
- `num_gpus` (`Optional`, `int`): Number of GPUs to use, default is 1.
- `max_new_tokens` (`Optional`, `int`): Maximum number of new tokens in generated text, default is 512.
- `debug` (`Optional`, `bool`): Enable debug mode, default is False.
- `device` (`Optional`, `str`): Specify the device to use, default is 'cuda:0'.

Here is a usage example:

```python
from trustllm.generation.generation import LLMGeneration

llm_gen = LLMGeneration(
    model_path="meta-llama/Llama-2-7b-chat-hf", 
    test_type="safety", 
    data_path="TrustLLM",
)

llm_gen.generation_results()
```



### **Don't have enough computing resource?**

If you don't have sufficient computing resources to run HuggingFace models locally, we recommend using online models. We provide an online model interface through [deepinfra](https://deepinfra.com/), and currently supported online models include:

- `llama2-70b`
- `mistral-7b`
- `dolly-12b`
- `llama2-13b`
- `mixtral-8x7B`
- `yi-34b`




## **Generation Parameters**

We categorize the tasks in the benchmark into two main groups: *Generation* and *Classification*. 

We employ a temperature setting of **0** for classification tasks to ensure more precise outputs. Conversely, for generation tasks, we set the temperature to **1**, fostering a more diverse range of results and exploring potential worst-case scenarios.

The temperature setting of each dataset file is as follows:

```python
file_config = {
        "disparagement.json": 1.0,
        "preference_force.json": 1.0,
        "preference_plain.json": 1.0,
        "stereotype_agreement.json": 1.0,
        "stereotype_stereoset_classification.json": 0.0,
        "stereotype_query_test.json":1.0,
        "stereotype_recognition.json":0.0,
        "external.json": 0.0,
        "hallucination.json": 0.0,
        "golden_advfactuality.json": 1.0,
        "internal.json": 1.0,
        "sycophancy.json": 1.0,
        "ood_detection.json":1.0,
        "ood_generalization.json":0.0,
        "AdvGLUE.json":0.0,
        "AdvInstruction.json":1.0,
        "jailbreak.json":1.0,
        "exaggerated_safety.json": 1.0,
        "misuse.json":1.0,
        "privacy_awareness_confAIde.json":0.0,
        "privacy_awareness_query.json": 1.0,
        "privacy_leakage.json": 1.0,
        "emotional_awareness.json": 0.0,
        "implicit_ETHICS.json": 0.0,
        "implicit_SocialChemistry101.json": 0.0
    }
```


