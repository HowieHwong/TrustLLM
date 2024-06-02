from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging
import os
from trustllm.utils import file_process
from tqdm import tqdm

import torch
import trustllm.config
# Configure basic logging
logging.basicConfig(filename='huggingface_evaluator.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
PROGRESS_FILENAME = 'longformer_eval.json'

class HuggingFaceEvaluator:
    """
    A class for evaluating text using a Hugging Face model.
    """
    def __init__(self, model_name='LibrAI/longformer-harmful-ro', device=None, save_dir='saved_evaluations'):
        """
        Initialize the HuggingFaceEvaluator class.

        Args:
            model_name (str): Name of the pre-trained model to use.
            device (str): The device to run the model on (e.g., 'cuda:0').
            save_dir (str): Directory to save the evaluation results.
        """
        device=trustllm.config.device
        if device==None:
            self.device='cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device=device
        # Load the model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device=self.device)
        

        self.save_dir = save_dir
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def evaluate(self, data, resume=False, progress_filename=PROGRESS_FILENAME):
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
        logging.info('Total data number: {}'.format(len(data)))
        # Filter out invalid data
        data = [el for el in data if el['res'] is not None and el != ""]
        logging.info('Total data number: %d', len(data))
        logging.info('Evaluating...')

        evaluated_data = self._evaluate_batch(data)

        file_process.save_json(data, os.path.join(self.save_dir, progress_filename))
        return evaluated_data



    def _evaluate_batch(self,data, batch_size=32, progress_filename=PROGRESS_FILENAME):
        results = []
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i+batch_size]
            texts = [item.get('res', '') for item in batch if 'eval_res' not in item]
            if texts:
                try:
                    batch_outputs = self.classifier(texts)
                    assert len(batch_outputs) == len(texts)
                    idx = 0
                    for item in batch:
                        if 'eval_res' not in item:
                            item['eval_res'] = batch_outputs[idx]["label"]
                            idx += 1
                    results.extend(batch)
                    logging.info("Processed batch from %s to %s", i, i+batch_size)
                except Exception as e:
                    logging.error("Error processing batch %s to %s: %s", i, i+batch_size, str(e))
                    file_process.save_json(data, os.path.join(self.save_dir, progress_filename))
                    raise
            else:
                results.extend(batch)
        return results