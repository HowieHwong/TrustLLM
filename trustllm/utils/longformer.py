from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging
import os
from trustllm.utils import file_process
from tqdm import tqdm

# Configure basic logging
logging.basicConfig(filename='huggingface_evaluator.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


class HuggingFaceEvaluator:
    """
    A class for evaluating text using a Hugging Face model.
    """
    def __init__(self, model_name='LibrAI/longformer-harmful-ro', device='cuda:0', save_dir='saved_evaluations'):
        """
        Initialize the HuggingFaceEvaluator class.

        Args:
            model_name (str): Name of the pre-trained model to use.
            device (str): The device to run the model on (e.g., 'cuda:0').
            save_dir (str): Directory to save the evaluation results.
        """
        # Load the model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)

        self.save_dir = save_dir
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def evaluate(self, data, resume=False, progress_filename='longformer_eval.json'):
        """
        Evaluate a dataset using the loaded Hugging Face model.

        Args:
            data: List of data to be evaluated.
            resume (bool): Whether to resume from a saved progress file. Default is False.
            progress_filename (str): Filename for saving or resuming progress.

        Returns:
            List: The evaluated data.
        """
        if resume:
            load_path = os.path.join(self.save_dir, progress_filename)
            try:
                data = file_process.load_json(load_path)
                logging.info("Resuming evaluation from saved progress.")
            except FileNotFoundError:
                logging.warning("No saved progress file found at %s. Starting a new evaluation.", load_path)

        assert isinstance(data, list), "Data must be a list."
        print('Total data number: {}'.format(len(data)))
        # Filter out invalid data
        data = [el for el in data if el['res'] is not None and el != ""]
        print('Valid data number: {}'.format(len(data)))
        print('Evaluating...')
        for item in tqdm(data):
            try:
                if 'eval_res' not in item:
                    text = item.get('res', '')
                    eval_res = self.classifier(text)[0]["label"]
                    item['eval_res'] = eval_res
                    logging.info("Evaluated item: %s", item)
            except Exception as e:
                logging.error("Error evaluating item %s: %s", item, str(e))
                file_process.save_json(data, os.path.join(self.save_dir, progress_filename))
                raise

        file_process.save_json(data, os.path.join(self.save_dir, progress_filename))
        return data
