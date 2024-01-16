

## **Generation Results**


The trustllm toolkit currently supports the generation of over a dozen models. 
You can use the trustllm toolkit to generate output results for specified models on the trustllm benchmark.

Supported LLMs:

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

### **Start Your Generation**

We offer both ***online*** and ***local LLMs*** generation pipeline. 


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


## **Dataset & Task**

**Dataset overview:**

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

**Task overview:**

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
