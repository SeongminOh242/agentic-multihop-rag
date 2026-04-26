from __future__ import annotations

import os
from typing import Protocol


class TextLLM(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        ...


_cached_llms: dict[tuple[str, str], TextLLM] = {}


class HFLocalLLM:
    """Local Hugging Face text-generation pipeline."""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self._pipeline = None
        self._load()

    def _hub_token(self) -> str | None:
        return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    def _load(self) -> None:
        from src.data_loader import ensure_hf_hub_env

        ensure_hf_hub_env()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        token = self._hub_token()

        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto",
                token=token,
            )
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=device,
                token=token,
            )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.max_length = None

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        tokenizer = self._pipeline.tokenizer
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        if getattr(tokenizer, "chat_template", None):
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = f"### User:\n{prompt}\n\n### Assistant:\n"

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        outputs = self._pipeline(
            rendered,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
        return self._extract_new_text(outputs, rendered).strip()

    @staticmethod
    def _extract_new_text(outputs: list, prompt_prefix: str) -> str:
        """Handle both string `generated_text` and chat list formats across Transformers versions."""
        if not outputs:
            return ""
        gen = outputs[0].get("generated_text")
        if gen is None:
            return ""
        if isinstance(gen, str):
            if gen.startswith(prompt_prefix):
                return gen[len(prompt_prefix) :]
            if prompt_prefix in gen:
                idx = gen.rindex(prompt_prefix) + len(prompt_prefix)
                return gen[idx:]
            return gen.strip()
        if isinstance(gen, list) and gen:
            last = gen[-1]
            if isinstance(last, dict) and last.get("role") == "assistant":
                return str(last.get("content", ""))
            if isinstance(last, dict) and "content" in last:
                return str(last["content"])
        return str(gen)


class OpenAILLM:
    """OpenAI chat-completions wrapper used for remote experiments."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise ImportError("openai is required for the OpenAI provider.") from exc

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider.")

        self.model_name = model_name
        self._client = OpenAI(api_key=resolved_api_key)

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content.strip()


class GeminiLLM:
    """Gemini wrapper for frontier-model experiments."""

    def __init__(self, model_name: str = "gemini-2.5-pro", api_key: str | None = None) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise ImportError("google-genai is required for the Gemini provider.") from exc

        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini provider.")

        self.model_name = model_name
        self._types = types
        self._client = genai.Client(api_key=resolved_api_key)

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        # thinking_budget=0 disables internal reasoning on gemini-2.5-* models so
        # response.text is never None; ignored by older non-thinking models.
        is_thinking_model = "2.5" in self.model_name
        config_kwargs: dict = {"temperature": 0, "max_output_tokens": max_new_tokens}
        if is_thinking_model:
            config_kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_budget=0)
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(**config_kwargs),
        )
        return (response.text or "").strip()


def get_llm(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    provider: str = "hf",
) -> TextLLM:
    """Return a cached LLM instance for the requested provider."""

    cache_key = (provider, model_name)
    cached = _cached_llms.get(cache_key)
    if cached is not None:
        return cached

    normalized_provider = provider.lower()
    if normalized_provider == "hf":
        llm: TextLLM = HFLocalLLM(model_name)
    elif normalized_provider == "openai":
        llm = OpenAILLM(model_name)
    elif normalized_provider == "gemini":
        llm = GeminiLLM(model_name)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    _cached_llms[cache_key] = llm
    return llm
