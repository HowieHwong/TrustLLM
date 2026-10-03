"""Download the benchmark archive without importing model dependencies."""

from pathlib import Path
from tempfile import TemporaryFile
from zipfile import ZipFile

import requests

DATASET_URL = "https://raw.githubusercontent.com/HowieHwong/TrustLLM/main/dataset/dataset.zip"


def download_dataset(save_path=None):
    """Extract the GitHub dataset archive under ``save_path`` (default: ``data``).

    The archive contains a ``dataset/`` directory, so the default dataset root
    for generation is ``data/dataset``. Existing dataset files are overwritten.
    Network, HTTP and ZIP errors propagate to the caller; partial downloads are
    discarded. Returns None for compatibility with the original helper.
    """
    destination = Path(save_path if save_path is not None else "data").resolve()
    with TemporaryFile() as archive:
        with requests.get(DATASET_URL, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                archive.write(chunk)
        archive.seek(0)
        with ZipFile(archive) as dataset:
            # Validate all paths before writing any archive member.
            for member in dataset.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
            destination.mkdir(parents=True, exist_ok=True)
            dataset.extractall(destination)
