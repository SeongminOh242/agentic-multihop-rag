from __future__ import annotations

_cached_llm: HFLocalLLM | None = None


class HFLocalLLM:
    """Llama-3.1-8B-Instruct. Uses 4-bit NF4 on CUDA, float16 on MPS/CPU."""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self._pipeline = None
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
            )
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=device,
            )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        output = self._pipeline(messages, max_new_tokens=max_new_tokens)
        return output[0]["generated_text"][-1]["content"].strip()


def get_llm(model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> HFLocalLLM:
    """Return a cached HFLocalLLM instance (loaded once, shared across pipelines)."""
    global _cached_llm
    if _cached_llm is None:
        _cached_llm = HFLocalLLM(model_name)
    return _cached_llm
