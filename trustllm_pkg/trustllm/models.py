"""Lazy model adapters: HTTP APIs work without any local inference libraries."""

import math
import os
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import requests


@dataclass
class Response:
    text: str
    usage: dict = field(default_factory=dict)


class APIModel:
    """Text-only, non-streaming OpenAI-compatible Chat Completions transport."""

    def __init__(
        self,
        model,
        *,
        base_url=None,
        api_key=None,
        timeout=120,
        retries=3,
        token_parameter="max_tokens",
        omit_temperature=False,
    ):
        self.model = model
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        url = urlsplit(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) API root without credentials, query or fragment"
            )
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        if url.hostname == "api.openai.com" and not self.api_key:
            raise ValueError("Set OPENAI_API_KEY before using the hosted OpenAI endpoint")
        if token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("token_parameter must be max_tokens or max_completion_tokens")
        if timeout <= 0 or retries < 0:
            raise ValueError("timeout must be positive and retries non-negative")
        self.timeout, self.retries = timeout, retries
        self.token_parameter, self.omit_temperature = token_parameter, omit_temperature

    def generate(self, prompt, *, temperature, max_new_tokens):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            self.token_parameter: max_new_tokens,
        }
        if not self.omit_temperature:
            payload["temperature"] = temperature
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.retries + 1):
            delay = min(2**attempt, 30)
            try:
                response = requests.post(
                    self.base_url + "/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.retries:
                    raise RuntimeError(
                        "API request failed after timeout/connection retries"
                    ) from None
            else:
                if response.ok:
                    try:
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                    except (ValueError, KeyError, IndexError, TypeError):
                        raise RuntimeError(
                            "API returned an invalid Chat Completions response"
                        ) from None
                    finally:
                        response.close()
                    if not isinstance(content, str) or not content.strip():
                        raise RuntimeError(
                            "API returned empty/non-text content; no successful sample was recorded"
                        )
                    return Response(content, data.get("usage") or {})
                # Never log provider bodies: they can contain credentials or private prompts.
                if (
                    response.status_code not in {408, 429, 500, 502, 503, 504}
                    or attempt == self.retries
                ):
                    response.close()
                    raise RuntimeError(
                        f"API request failed (HTTP {response.status_code}); check endpoint, model and credentials"
                    )
                try:
                    requested_delay = float(response.headers.get("Retry-After", delay))
                    if math.isfinite(requested_delay):
                        delay = max(0, min(requested_delay, 60))
                except ValueError:
                    pass
                finally:
                    response.close()
            time.sleep(delay)

    def describe(self):
        return {
            "backend": "api",
            "model": self.model,
            "base_url": self.base_url,
            "token_parameter": self.token_parameter,
            "omit_temperature": self.omit_temperature,
        }


class LocalModel:
    """Hugging Face causal language model, downloaded on first use or loaded from disk."""

    def __init__(
        self,
        model,
        *,
        device="auto",
        dtype="auto",
        revision=None,
        trust_remote_code=False,
        seed=42,
        repetition_penalty=1.0,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError:
            raise ImportError(
                "Local generation requires the 'local' extra. Install it with: "
                "python -m pip install 'trustllm[local] @ "
                "git+https://github.com/HowieHwong/TrustLLM.git@main#subdirectory=trustllm_pkg'"
            ) from None
        if dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be auto, float32, float16 or bfloat16")
        self.model_id, self.revision = model, revision
        self.device, self.dtype, self.seed = device, dtype, seed
        self.trust_remote_code = trust_remote_code
        self.repetition_penalty = repetition_penalty
        self.torch = torch
        set_seed(seed)
        options = {"revision": revision, "trust_remote_code": trust_remote_code}
        self.tokenizer = AutoTokenizer.from_pretrained(model, **options)
        load_options = dict(
            options, torch_dtype="auto" if dtype == "auto" else getattr(torch, dtype)
        )
        if device == "auto":
            load_options["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model, **load_options)
        if device != "auto":
            self.model.to(device)
        self.model.eval()

    def generate(self, prompt, *, temperature, max_new_tokens):
        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        options = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "repetition_penalty": self.repetition_penalty,
        }
        if temperature > 0:
            options["temperature"] = temperature
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is not None:
            options["pad_token_id"] = pad_id
        with self.torch.inference_mode():
            tokens = self.model.generate(**inputs, **options)
        prompt_length = inputs["input_ids"].shape[-1]
        completion = tokens[0][prompt_length:]
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        if not text.strip():
            raise RuntimeError("Local model returned empty text")
        return Response(
            text, {"prompt_tokens": prompt_length, "completion_tokens": len(completion)}
        )

    def describe(self):
        return {
            "backend": "local",
            "model": self.model_id,
            "revision": self.revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "device": self.device,
            "dtype": self.dtype,
            "seed": self.seed,
            "trust_remote_code": self.trust_remote_code,
            "repetition_penalty": self.repetition_penalty,
        }
