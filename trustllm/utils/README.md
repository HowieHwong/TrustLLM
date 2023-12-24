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
from auto_evaluator import AutoEvaluator

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
