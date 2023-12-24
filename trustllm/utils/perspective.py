from googleapiclient import discovery
import logging
import os
import file_process


def get_toxicity_value(sentence, api_key):
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=api_key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {
        'comment': {'text': '{}'.format(sentence)},
        'requestedAttributes': {'TOXICITY': {}}
    }
    response = client.comments().analyze(body=analyze_request).execute()
    return response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']


# Configure logging
logging.basicConfig(filename='perspective_evaluator.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


class PerspectiveEval:
    def __init__(self, api_key, save_dir='saved_evaluations'):
        """
        Initialize the PerspectiveEval class.

        Args:
        - api_key (str): API key for accessing the Perspective API.
        - save_dir (str): Directory where evaluation progress will be saved.
        """
        self.api_key = api_key
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def get_toxicity_value(self, sentence):
        """
        Get the toxicity value of a sentence using the Perspective API.

        Args:
        - sentence (str): The sentence to analyze.

        Returns:
        - float: The toxicity score of the sentence.
        """
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

        analyze_request = {
            'comment': {'text': sentence},
            'requestedAttributes': {'TOXICITY': {}}
        }
        response = client.comments().analyze(body=analyze_request).execute()
        return response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']

    def evaluate(self, data, resume=False, progress_filename='perspective_eval_progress.json'):
        """
        Evaluate the given data for toxicity using the Perspective API.

        Args:
        - data (list): Data to be evaluated. Each item should be a dictionary with a key 'text' containing the text to evaluate.
        - resume (bool): Flag to indicate whether to resume from saved progress.
        - progress_filename (str): Filename to save/load the evaluation progress.

        Returns:
        - list: Evaluated data with toxicity scores.
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
                text = item.get('text', '')
                toxicity_score = self.get_toxicity_value(text)
                item['toxicity'] = toxicity_score
                logging.info("Evaluated item: %s", item)
            except Exception as e:
                logging.error("Error evaluating item %s: %s", item, str(e))
                file_process.save_json(data, os.path.join(self.save_dir, progress_filename))
                raise

        file_process.save_json(data, os.path.join(self.save_dir, progress_filename))
        return data
