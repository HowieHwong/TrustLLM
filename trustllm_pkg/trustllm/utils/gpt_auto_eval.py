import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from trustllm.utils import file_process
from tqdm import tqdm
import logging
import os
import trustllm

# Setting up basic logging configuration
logging.basicConfig(filename='autoevaluator.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


# Retry decorator with exponential backoff and stop condition for API calls
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_res(string, model='gpt-4-1106-preview', temp=0):
    """
    Retrieve a response from the OpenAI ChatCompletion API.

    Args:
        string (str): The input string to process.
        model (str): The model to use for generating the response. Default is 'gpt-4-1106-preview'.
        temp (float): The temperature setting for the API request. Default is 0 for deterministic output.

    Returns:
        str: The API response content.

    Raises:
        ValueError: If the API response is null or an empty string.
    """
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": string}],
        temperature=temp
    )

    if not completion.choices[0].message['content']:
        raise ValueError("The response from the API is NULL or an empty string!")

    return completion.choices[0].essage['content']


class AutoEvaluator:
    """
    A class for automating the evaluation of text using the OpenAI API.
    """

    def __init__(self, save_dir='saved_evaluations'):
        """
        Initialize the AutoEvaluator class.

        Args:
            save_dir (str): Directory for saving evaluation results.
        """
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        openai.api_key = trustllm.config.openai_key

    def save_progress(self, data, filename='auto_eval.json'):
        """
        Save evaluation progress to a JSON file.

        Args:
            data: Data to be saved.
            filename (str): Name of the file for saving the data.
        """
        save_path = os.path.join(self.save_dir, filename)
        file_process.save_json(data, save_path)
        logging.info("Progress saved to %s", save_path)

    def evaluate(self, data, task, resume=False, progress_filename='eval_progress.json', concat=True):
        """
        Evaluate a given dataset using a specified task.

        Args:
            data: Data to be evaluated.
            task (str): The task identifier for the evaluation.
            resume (bool): Flag to resume from saved progress. Default is False.
            progress_filename (str): The filename for saving or resuming progress.
            concat (bool): Flag to concatenate responses. Default is True.

        Returns:
            The evaluated data.
        """
        task_prompt_dict = file_process.load_json('trustllm/prompt/task_prompt.json')
        prompt_data = []

        if not concat:
            replace_dict = task_prompt_dict.get(task, {}).get('mapping', {})
            prompt = task_prompt_dict.get(task, {}).get('prompt', '')
            for el in data:
                single_prompt = prompt
                for k, v in replace_dict.items():
                    single_prompt = single_prompt.replace(k, str(el[v]))
                prompt_data.append(single_prompt)
        else:
            prompt = task_prompt_dict.get(task, {}).get('prompt', '')
            prompt_data = [prompt + item['res'] for item in data]

        if resume:
            load_path = os.path.join(self.save_dir, progress_filename)
            try:
                data = file_process.load_json(load_path)
                logging.info("Resuming evaluation from saved progress.")
            except FileNotFoundError:
                logging.warning("No saved progress file found at %s. Starting a new evaluation.", load_path)

        assert isinstance(data, list), "Data must be a list."
        assert task is not None, "Task must be specified for evaluation."

        print('Total data number: {}'.format(len(data)))
        print('Evaluating...')
        for item, el in tqdm(zip(prompt_data, data)):
            try:
                if 'eval_res' not in el:
                    print('Prompt: {}'.format(item))
                    eval_res = get_res(item)
                    print('Response: {}'.format(eval_res))
                    el['eval_res'] = eval_res
                    logging.info("Evaluated item: %s", item)
                    logging.info("Evaluated result: %s", eval_res)
            except Exception as e:
                logging.error("Error evaluating item %s: %s", item, str(e))
                self.save_progress(data, filename=progress_filename)
                raise

        self.save_progress(data, filename=progress_filename)
        return data
