"""Compatibility entry point backed by the unified generation runner."""

from trustllm.runner import generate


class LLMGeneration:
    """Use ``backend='local'`` or ``backend='api'`` with arbitrary model IDs.

    The original constructor parameters remain accepted. Archived Replicate
    integrations are available in ``trustllm.generation.legacy``.
    """

    def __init__(
        self,
        test_type,
        data_path,
        model_path,
        online_model=False,
        use_deepinfra=False,
        use_replicate=False,
        repetition_penalty=1.0,
        num_gpus=1,
        max_new_tokens=512,
        debug=False,
        device="",
        **options,
    ):
        if use_replicate:
            raise ValueError(
                "Use an OpenAI-compatible endpoint, or import LLMGeneration from trustllm.generation.legacy with the legacy extra"
            )
        if num_gpus != 1:
            raise ValueError(
                "Use device='auto' for model sharding; num_gpus no longer selects GPU count"
            )
        self.options = dict(options)
        self.options.setdefault("backend", "api" if online_model or use_deepinfra else "local")
        if use_deepinfra:
            import os

            self.options.setdefault("base_url", "https://api.deepinfra.com/v1/openai")
            self.options.setdefault("api_key", os.getenv("DEEPINFRA_API_TOKEN"))
        self.options.update(
            model=model_path,
            task=test_type,
            data_path=data_path,
            max_new_tokens=max_new_tokens,
            device=device or "auto",
            repetition_penalty=repetition_penalty,
        )
        self.result = None

    def generation_results(self, max_retries=3, retry_interval=3):
        """Run generation, raising on failure; the summary is available as ``result``.

        Retries are per API request; retry_interval is retained for source compatibility.
        """
        self.options.setdefault("retries", max_retries)
        self.result = generate(**self.options)
        return "OK"
