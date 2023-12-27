from datasets import load_dataset


def download_huggingface_dataset(dataset_name="TrustLLM/TrustLLM-dataset", save_path=None):
    """
    Download a dataset from Hugging Face and save it locally.

    Args:
    - dataset_name (str): The name of the dataset to download.
    - save_path (str, optional): The local path to save the dataset. If None, uses default path.

    Returns:
    - None
    """

    try:
        # Load the dataset
        dataset = load_dataset(dataset_name)

        # If a save path is provided, save the dataset to that path
        if save_path:
            dataset.save_to_disk(save_path)
            print(f"Dataset saved to {save_path}")
        else:
            print("Dataset loaded but not saved to disk.")

    except Exception as e:
        print(f"An error occurred: {e}")
