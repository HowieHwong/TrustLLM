import openai
import os
import logging
from tqdm import tqdm
import trustllm.config
from trustllm.utils import file_process
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

class DataEmbedder:
    """
    A class for embedding textual data using OpenAI's embedding models.
    """
    def __init__(self, save_dir='saved_embeddings'):
        """
        Initialize the DataEmbedder class.

        Args:
            save_dir (str): Directory to save the embedding results.
        """
        self.save_dir = save_dir
        # Create the directory if it does not exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        openai.api_key = trustllm.config.openai_key

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
    def get_embeddings(self, string):
        """
        Retrieve embeddings for a given string.

        Args:
            string (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        response = openai.Embedding.create(
            model='text-embedding-ada-002',  # Example model
            input=string
        )
        return response["data"][0]["embedding"]

    def save_embeddings(self, embeddings, filename):
        """
        Save embeddings to a JSON file.

        Args:
            embeddings: The embeddings to be saved.
            filename (str): The filename for saving the embeddings.
        """
        save_path = os.path.join(self.save_dir, filename)
        file_process.save_json(embeddings, save_path)
        logging.info("Embeddings saved to %s", save_path)

    def embed_data(self, data, filename='embeddings.json', resume=False):
        """
        Embed a dataset and save the embeddings.

        Args:
            data: List of data to be embedded.
            filename (str): The filename for saving embeddings. Default is 'embeddings.json'.
            resume (bool): Flag to resume from saved progress. Default is False.

        Returns:
            str: Path to the saved embeddings file.
        """
        assert isinstance(data, list), "Data must be a list."
        print('Evaluating...')
        if resume:
            try:
                data = file_process.load_json(os.path.join(self.save_dir, filename))
                logging.info("Resuming from saved progress.")
            except FileNotFoundError:
                logging.warning("No saved progress file found. Starting from scratch.")

        for el in tqdm(data):
            try:
                if 'embedding' not in el:
                    el['embedding'] = self.get_embeddings(el['res'])
                    logging.info("Evaluated item: %s", el.get('res', ''))
            except Exception as e:
                logging.error("Error embedding item %s: %s", el.get('res', ''), str(e))
                self.save_embeddings(data, filename)
                raise

        self.save_embeddings(data, filename)
        return os.path.join(self.save_dir, filename)
