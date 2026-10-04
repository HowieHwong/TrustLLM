"""TrustLLM public Python API. Model libraries load only when requested."""

from trustllm.dataset_download import download_dataset
from trustllm.runner import generate

__all__ = ["download_dataset", "generate"]
