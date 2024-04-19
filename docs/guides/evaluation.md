
## **Start Your Evaluation**




### **API Setting**
Before starting the evaluation, you need to first set up your [OpenAI API](https://openai.com/product) (GPT-4-turbo) and [Perspective API](https://developers.perspectiveapi.com/s/docs-get-started?language=en_US) (used for measuring toxicity).

```python
from trustllm import config

config.openai_key = 'your-openai-api-key'

config.perspective_key = 'your-perspective-api-key'
```

If you're using OpenAI API through [Azure](https://azure.microsoft.com/en-us/products/ai-services/openai-service), you should set up your Azure api:

```python
config.azure_openai = True

config.azure_engine = "your-azure-engine-name"

config.azure_api_base = "your-azure-api-url (openai.base_url)"
```



### Easy Pipeline

From [Version 0.2.1](https://howiehwong.github.io/TrustLLM/changelog.html#version-021), trustllm toolkit supports easy pipeline for evaluation.

We have provided pipelines for all six sections: `run_truthfulness`, `run_safety`, `run_fairness`, `run_robustness`, `run_privacy`, `run_ethics`.



#### Truthfulness Evaluation  

For truthfulness assessment, the `run_truthfulness` function is used. Provide JSON file paths for internal consistency, external consistency, hallucination scenarios, sycophancy evaluation, and adversarial factuality.  

```python  
truthfulness_results = run_truthfulness(  
    internal_path="path_to_internal_consistency_data.json",  
    external_path="path_to_external_consistency_data.json",  
    hallucination_path="path_to_hallucination_data.json",  
    sycophancy_path="path_to_sycophancy_data.json",
    advfact_path="path_to_advfact_data.json"
)
```

The function will return a dictionary containing results for internal consistency, external consistency, hallucinations, sycophancy (with persona and preference evaluations), and adversarial factuality.     

#### Safety Evaluation  

To assess the safety of your language model, use the `run_safety` function. You can provide paths to data for jailbreak scenarios, exaggerated safety situations, and misuse potential. Optionally, you can also evaluate for toxicity.  

```python  
safety_results = run_safety(  
    jailbreak_path="path_to_jailbreak_data.json",  
    exaggerated_safety_path="path_to_exaggerated_safety_data.json",  
    misuse_path="path_to_misuse_data.json",  
    toxicity_eval=True,  
    toxicity_path="path_to_toxicity_data.json",  
    jailbreak_eval_type="total"  
)  
```

The returned dictionary includes results for jailbreak, exaggerated safety, misuse, and toxicity evaluations.  

#### Fairness Evaluation     

To evaluate the fairness of your language model, use the `run_fairness` function. This function takes paths to JSON files containing data on stereotype recognition, stereotype agreement, stereotype queries, disparagement, and preference biases.     

```python
fairness_results = run_fairness(
    stereotype_recognition_path="path_to_stereotype_recognition_data.json",      
    stereotype_agreement_path="path_to_stereotype_agreement_data.json",      
    stereotype_query_test_path="path_to_stereotype_query_test_data.json",      
    disparagement_path="path_to_disparagement_data.json",      
    preference_path="path_to_preference_data.json"   
)  
```

The returned dictionary will include results for stereotype recognition, stereotype agreement, stereotype queries, disparagement, and preference bias evaluations.

#### Robustness Evaluation  

To evaluate the robustness of your language model, use the `run_robustness` function. This function accepts paths to JSON files for adversarial GLUE data, adversarial instruction data, out-of-distribution (OOD) detection, and OOD generalization.  

```python  
robustness_results = run_robustness(  
    advglue_path="path_to_advglue_data.json",  
    advinstruction_path="path_to_advinstruction_data.json",  
    ood_detection_path="path_to_ood_detection_data.json",  
    ood_generalization_path="path_to_ood_generalization_data.json"  
)  
```

The function returns a dictionary with the results of adversarial GLUE, adversarial instruction, OOD detection, and OOD generalization evaluations.  

#### Privacy Evaluation  

To conduct privacy evaluations, use the `run_privacy` function. It allows you to specify paths to datasets for privacy conformity, privacy awareness queries, and privacy leakage scenarios.  

```python  
privacy_results = run_privacy(  
    privacy_confAIde_path="path_to_privacy_confaide_data.json",  
    privacy_awareness_query_path="path_to_privacy_awareness_query_data.json",  
    privacy_leakage_path="path_to_privacy_leakage_data.json"  
)  
```

The function outputs a dictionary with results for privacy conformity AIde, normal and augmented privacy awareness queries, and privacy leakage evaluations.  

#### Ethics Evaluation  

To evaluate the ethical considerations of your language model, use the `run_ethics` function. You can specify paths to JSON files containing explicit ethics, implicit ethics, and awareness data.  

```python  
results = run_ethics(  
    explicit_ethics_path="path_to_explicit_ethics_data.json",  
    implicit_ethics_path="path_to_implicit_ethics_data.json",  
    awareness_path="path_to_awareness_data.json"  
)  
```

The function returns a dictionary containing the results of the explicit ethics evaluation (with low and high levels), implicit ethics evaluation (ETHICS and social norm types), and emotional awareness evaluation.  







### **Truthfulness**

Four subsections in truthfulness evaluation:

* Misinformation: `external.json`, `internal.json`
* Hallucination: `hallucination.json`
* Sycophancy: `sycophancy.json`
* Adversarial Factuality: `golden_advfactuality.json`


Requirement:

![OpenAI](https://img.shields.io/badge/OpenAI-blue)

* openai api (gpt-4-turbo)

Preliminary:

```python
from trustllm import truthfulness
from trustllm import file_process
from trustllm import config

evaluator = truthfulness.TruthfulnessEval()
```

Misinformation evaluation:

```python
misinformation_internal_data = file_process.load_json('misinformation_internal_data_json_path')
print(evaluator.internal_eval(misinformation_internal_data))

misinformation_external_data = file_process.load_json('misinformation_external_data_json_path')
print(evaluator.external_eval(misinformation_external_data))
```

Hallucination evaluation:

```python
hallucination_data = file_process.load_json('hallucination_data_json_path')
print(evaluator.hallucination_eval(hallucination_data))
```

Sycophancy evaluation (`eval_type`: type of evaluation, either `persona` or `preference`):
```python
sycophancy_data = file_process.load_json('sycophancy_data_json_path')
print(evaluator.sycophancy_eval(sycophancy_data, eval_type='persona'))
print(evaluator.sycophancy_eval(sycophancy_data, eval_type='preference'))
```

Adversarial factuality evaluation:
```python
adv_fact_data = file_process.load_json('adv_fact_data_json_path')
print(evaluator.advfact_eval(adv_fact_data))
```

### **Safety**

Four subsections in safety evaluation:

* Jailbreak: `jailbreak.json`
* Exaggerated Safety: `exaggerated_safety.json`
* Toxicity: `jailbreak.json`
* Misuse: `misuse.json`

Requirement:

![OpenAI](https://img.shields.io/badge/OpenAI-blue)
![Perspective](https://img.shields.io/badge/Perspective-purple)
![HuggingFace](https://img.shields.io/badge/HuggingFace-yellow) 

* openai api (gpt-4-turbo)
* perspective api
* huggingface evaluator: LibrAI/longformer-harmful-ro

Preliminary:

```python
from trustllm import safety
from trustllm import file_process
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
eval_data, _ = evaluator.jailbreak_eval(jailbreak_data, eval_type='total/single', return_data=True)
print(evaluator.toxicity_eval(eval_data))

# If you have saved evaluated jailbreak results
toxicity_data = file_process.load_json('evaluated_jailbreak_results_json_path') # load eval data for toxicity evaluation
print(evaluator.toxicity_eval(toxicity_data))
```

Misuse evaluation:

```python
misuse_data = file_process.load_json('misuse_data_json_path')
print(evaluator.misuse_eval(misuse_data))
```

### **Fairness**

Three subsections in fairness evaluation:

* Stereotype: `stereotype_agreement.json`, `stereotype_recognition.json`, `stereotype_query_test.json`
* Disparagement: `disparagement.json`
* Preference: `preference_force.json`

Requirement:

![OpenAI](https://img.shields.io/badge/OpenAI-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-yellow) 

* openai api (gpt-4-turbo)
* huggingface evaluator: LibrAI/longformer-harmful-ro

Preliminary:

```python
from trustllm import fairness
from trustllm import file_process
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
print(evaluator.preference_eval(preference_data))
```

### **Robustness**

Two subsections in robustness evaluation:

* Natural noise: `advglue.json`, `advinstruction.json`
* Out of distribution: `ood_generalization.json`, `ood_detection.json`


Requirement:


![HuggingFace](https://img.shields.io/badge/HuggingFace-yellow) 

* huggingface evaluator: LibrAI/longformer-harmful-ro

Preliminary:

```python
from trustllm import robustness
from trustllm import file_process
from trustllm import config

evaluator = robustness.RobustnessEval()
```

Natural noise evaluation:

```python
advglue_data = file_process.load_json('advglue_data_json_path')
print(evaluator.advglue_eval(advglue_data))

advinstruction_data = file_process.load_json('advinstruction_data_json_path')
print(evaluator.advglue_eval(advinstruction_data))
```

OOD evaluation:

```python
ood_detection_data = file_process.load_json('ood_detection_data_json_path')
print(evaluator.ood_detection(ood_detection_data))

ood_generalization_data = file_process.load_json('ood_generalization_data_json_path')
print(evaluator.ood_generalization(ood_generalization_data))
```


### **Privacy**


Two subsections in privacy evaluation:

* Privacy awareness: `privacy_awareness_confAIde.json`, `privacy_awareness_query.json`
* Privacy leakage: `privacy_leakage.json`

Requirement:


![HuggingFace](https://img.shields.io/badge/HuggingFace-yellow) 

* huggingface evaluator: LibrAI/longformer-harmful-ro

Preliminary:

```python
from trustllm import privacy
from trustllm import file_process
from trustllm import config

evaluator = privacy.PrivacyEval()
```

Privacy awareness:

```python
privacy_confAIde_data = file_process.load_json('privacy_confAIde_data_json_path')
print(evaluator.ConfAIDe_eval(privacy_confAIde_data))

privacy_awareness_query_data = file_process.load_json('privacy_awareness_query_data_json_path')
print(evaluator.awareness_query_eval(privacy_awareness_query_data, type='normal'))
print(evaluator.awareness_query_eval(privacy_awareness_query_data, type='aug'))
```

Privacy leakage:

```python
privacy_leakage_data = file_process.load_json('privacy_leakage_data_json_path')
print(evaluator.leakage_eval(privacy_leakage_data))
```



### **Machine Ethics**

Three subsections in machine ethics evaluation:

Implicit ethics: `implicit_ETHICS.json`, `implicit_SocialChemistry101.json`  
Explicit ethics: `explicit_moralchoice.json`  
Awareness: `awareness.json`  


Requirement:

![OpenAI](https://img.shields.io/badge/OpenAI-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-yellow) 

* openai api (gpt-4-turbo)
* huggingface evaluator: LibrAI/longformer-harmful-ro

Preliminary:

```python
from trustllm import ethics
from trustllm import file_process
from trustllm import config

evaluator = ethics.EthicsEval()
```

Explicit ethics:

```python
explicit_ethics_data = file_process.load_json('explicit_ethics_data_json_path')
print(evaluator.explicit_ethics_eval(explicit_ethics_data, eval_type='low'))
print(evaluator.explicit_ethics_eval(explicit_ethics_data, eval_type='high'))
```
Implicit ethics:

```python
implicit_ethics_data = file_process.load_json('implicit_ethics_data_json_path')
# evaluate ETHICS dataset
print(evaluator.implicit_ethics_eval(implicit_ethics_data, eval_type='ETHICS'))
# evaluate social_norm dataset
print(evaluator.implicit_ethics_eval(implicit_ethics_data, eval_type='social_norm'))
```

Awareness:

```python
awareness_data = file_process.load_json('awareness_data_json_path')
print(evaluator.awareness_eval(awareness_data))
```