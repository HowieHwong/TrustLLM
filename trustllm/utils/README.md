# AutoEvaluator Class Documentation

## Overview
The `AutoEvaluator` class is designed for automatic evaluation of data. It supports resuming evaluations from previous progress or starting new evaluation tasks. The class also includes logging to track the evaluation process and any errors encountered.

## Initialization
To instantiate the `AutoEvaluator` class, use the following constructor:

```python
evaluator = AutoEvaluator()
```

Parameters:
- `save_dir` (str): Name of the directory to save evaluation progress. The directory will be created if it does not exist.

## Methods

### evaluate
Conduct evaluations or resume from previously saved progress.

```python
evaluated_data = evaluator.evaluate(data, task, resume=False, progress_filename='eval_progress.json')
```

Parameters:
- `data` (list): List of data to be evaluated. Each item in the list should be a dictionary.
- `task` (str): Identifier for the evaluation task.
- `resume` (bool, optional): Whether to continue from previously saved progress. Defaults to `False`.
- `progress_filename` (str, optional): Filename for saving or loading evaluation progress. Defaults to 'eval_progress.json'.

Returns:
- A list of evaluated data.

### save_progress
Saves the current evaluation progress to a file.

```python
evaluator.save_progress(data, filename='eval_progress.json')
```

Parameters:
- `data` (list): The evaluation data to be saved.
- `filename` (str, optional): Filename for saving the progress. Defaults to 'eval_progress.json'.

### load_progress
Loads evaluation progress from a file.

```python
data = evaluator.load_progress(filename='eval_progress.json')
```

Parameters:
- `filename` (str, optional): Filename from which to load progress. Defaults to 'eval_progress.json'.

Returns:
- A list of data loaded from the file.

## Usage Example

```python
from trustllm.utils.gpt_auto_eval import AutoEvaluator

# Create an evaluator instance
evaluator = AutoEvaluator()

# Prepare data to be evaluated
data = [
    {'prompt': 'prompt1', 'res': 'response1'}, 
    {'prompt': 'prompt2', 'res': 'response2'}, 
    ...
]

# Perform the evaluation
evaluated_data = evaluator.evaluate(data, task='your_task')

# View evaluation results
print(evaluated_data)
```

In the above example, we create an instance of `AutoEvaluator` and perform evaluations on some mock data. Here, `'your_task'` should be replaced with the actual task identifier.



# HuggingFaceEvaluator Class Documentation

## Overview
The `HuggingFaceEvaluator` class utilizes a model from the Hugging Face Transformers library to evaluate data. This class is particularly useful for processing and evaluating text data with pre-trained models available on Hugging Face.

## Initialization
To create an instance of the `HuggingFaceEvaluator` class, use the following constructor:

```python
evaluator = HuggingFaceEvaluator(model_name='LibrAI/longformer-harmful-ro', device='cuda:0', save_dir='your_save_directory')
```

Parameters:
- `model_name` (str): Name of the Hugging Face model to be used for evaluation.
- `device` (str): The computing device for the model (e.g., 'cuda:0' for GPU).
- `save_dir` (str): The directory where the evaluation progress will be saved.

## Methods

### evaluate
Perform evaluations on the given data or resume from previously saved progress.

```python
evaluated_data = evaluator.evaluate(data, resume=False, progress_filename='hf_eval_progress.json')
```

Parameters:
- `data` (list): The data to be evaluated. Each item should be a dictionary with a key 'res' containing the text to evaluate.
- `resume` (bool, optional): Flag to indicate whether to resume from saved progress. Defaults to `False`.
- `progress_filename` (str, optional): Filename for saving or loading evaluation progress. Defaults to 'hf_eval_progress.json'.

Returns:
- A list of dictionaries with evaluation results added.

## Usage Example

```python
from trustllm.utils.longformer import HuggingFaceEvaluator

# Create an instance of the evaluator
evaluator = HuggingFaceEvaluator(model_name='LibrAI/longformer-harmful-ro', device='cuda:0')

# Prepare data for evaluation
data = [
    {'prompt': 'prompt1', 'res': 'response1'}, 
    {'prompt': 'prompt2', 'res': 'response2'}, 
    ...
]
# Perform the evaluation
evaluated_data = evaluator.evaluate(data)

# Print the evaluation results
for item in evaluated_data:
    print(item)
```

In this example, we create an instance of `HuggingFaceEvaluator` and evaluate some sample text data. The results of the evaluation are added to the original data under the key 'eval_res'.

Note: Ensure that the `transformers` library and appropriate CUDA libraries are installed if using GPU acceleration.

