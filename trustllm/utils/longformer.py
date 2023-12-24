from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging
import logging
import os
import file_process

# Configure logging
logging.basicConfig(filename='huggingface_evaluator.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


class HuggingFaceEvaluator:
    def __init__(self, model_name='LibrAI/longformer-harmful-ro', device='cuda:0', save_dir='saved_evaluations'):
        """
        Initialize the HuggingFaceEvaluator class.

        Args:
        - model_name (str): Name of the model to use for evaluation.
        - device (str): Device to run the model on (e.g., 'cuda:0' for GPU).
        - save_dir (str): Directory where evaluation progress will be saved.
        """
        # Download and cache the model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize the pipeline with the downloaded model and tokenizer
        self.classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def evaluate(self, data, resume=False, progress_filename='hf_eval_progress.json'):
        """
        Evaluate the given data or resume evaluation from saved progress.

        Args:
        - data (list): Data to be evaluated. Each item in the list should be a dictionary.
        - resume (bool): Flag to indicate whether to resume from saved progress.
        - progress_filename (str): Filename to save/load the evaluation progress.

        Returns:
        - list: Evaluated data with results.
        """
        if resume:
            load_path = os.path.join(self.save_dir, progress_filename)
            try:
                data = file_process.load_json(load_path)
                logging.info("Resuming evaluation from saved progress.")
            except FileNotFoundError:
                logging.warning("No saved progress file found at %s. Starting a new evaluation.", load_path)

        assert isinstance(data, list), "Data must be a list."
        print('Evaluating...')
        for item in data:
            try:
                if 'eval_res' not in item:
                    text = item.get('res', '')
                    eval_res = self.classifier(text)[0]["label"]
                    item['eval_res'] = eval_res
                    logging.info("Evaluated item: %s", item)
            except Exception as e:
                logging.error("Error evaluating item %s: %s", item, str(e))
                file_process.save_json(data, os.path.join(self.save_dir, progress_filename))
                raise  # Re-raise the exception to notify the caller

        file_process.save_json(data, os.path.join(self.save_dir, progress_filename))
        return data
