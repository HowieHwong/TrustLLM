import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from trustllm.utils import file_process
from tqdm import tqdm
import logging
import os
from .. config import openai_key


# A decorator for retrying the `get_res` function with exponential backoff and a stop condition
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_res(string):
    """
    Function to get response from OpenAI's ChatCompletion API.

    Args:
    - string (str): The input string for which the completion is required.

    Returns:
    - str: The response from the API.

    Raises:
    - ValueError: If the API returns a null or empty response.
    """

    completion = openai.ChatCompletion.create(
        model='gpt-4-1106-preview',
        messages=[
            {"role": "user", "content": string}
        ],
        temperature=0,  # Setting the temperature to 0 for deterministic output
    )

    # Checking if the response is empty
    if not completion.choices[0].message['content']:
        raise ValueError("The response from the API is NULL or an empty string!")

    # Returning the API response
    return completion.choices[0].message['content']


# Set up logging
logging.basicConfig(filename='autoevaluator.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


class AutoEvaluator:
    def __init__(self, save_dir='saved_evaluations'):
        """
        Initialize the AutoEvaluator class.

        Args:
        - save_dir (str): Directory where the evaluation progress will be saved.
        """
        self.save_dir = save_dir
        # Create the directory if it does not exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        openai.api_key = openai_key

    def save_progress(self, data, filename='eval_progress.json'):
        """
        Save the current evaluation progress to a file.

        Args:
        - data (list): The data to be saved.
        - filename (str): The filename to save the progress.
        """
        save_path = os.path.join(self.save_dir, filename)
        # Use file_process module to save data as JSON
        file_process.save_json(data, save_path)
        logging.info("Progress saved to %s", save_path)

    def evaluate(self, data, task, resume=False, progress_filename='eval_progress.json'):
        """
        Evaluate the given data or resume evaluation from saved progress.

        Args:
        - data (list): The data to be evaluated.
        - task (str): The task identifier for evaluation.
        - resume (bool): Flag to indicate whether to resume from saved progress.
        - progress_filename (str): Filename to save/load the evaluation progress.

        Returns:
        - list: The evaluated data with results.
        """
        # If resume is True, load saved progress; otherwise, use provided data
        if resume:
            load_path = os.path.join(self.save_dir, progress_filename)
            try:
                data = file_process.load_json(load_path)
                logging.info("Resuming evaluation from saved progress.")
            except FileNotFoundError:
                logging.warning("No saved progress file found at %s. Starting a new evaluation.", load_path)
                # If no saved file found, start with the provided data

        assert isinstance(data, list), "Data must be a list."
        assert task is not None, "Task must be specified for evaluation."
        # Load the task prompts for evaluation
        task_prompt_dict = file_process.load_json('../prompt/task_prompt.json')
        prompt = task_prompt_dict.get(task, '')
        print('Evaluating...')

        for item in tqdm(data):
            try:
                # Append the response to the prompt and evaluate
                if 'eval_res' not in item:
                    prompt += item.get('res', '')
                    eval_res = get_res(prompt)
                    item['eval_res'] = eval_res
                    logging.info("Evaluated item: %s", item)
            except Exception as e:
                # Log the error and save the current progress
                logging.error("Error evaluating item %s: %s", item, str(e))
                self.save_progress(data, filename=progress_filename)
                raise  # Re-raise the exception to notify the caller

        # Save the final progress after evaluation completion
        self.save_progress(data, filename=progress_filename)
        return data