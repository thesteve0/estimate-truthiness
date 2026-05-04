import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .base import ModelAdapter


class OLMo3Adapter(ModelAdapter):
    """Adapter for allenai/Olmo-3-7B-Instruct.

    OLMo 3 Instruct is natively supported in transformers >= 4.57.0 and does
    not require trust_remote_code. Uses SentencePiece tokenization (▁ space
    markers). Input is formatted as a single-turn user message via the model's
    chat template before being passed to the model.
    """

    HF_MODEL_ID = "allenai/Olmo-3-7B-Instruct"

    def __init__(self) -> None:
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._device: torch.device | None = None

    @property
    def model_name(self) -> str:
        return "OLMo 3 7B Instruct (allenai/Olmo-3-7B-Instruct)"

    def load(self, device: torch.device) -> None:
        self._device = device
        print(f"\nLoading {self.model_name}...")

        dtype = self._select_dtype()
        print(f"  dtype: {dtype}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_ID)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.HF_MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
        )
        self._model.eval()

        param_count = sum(p.numel() for p in self._model.parameters())
        print(f"  parameters: {param_count / 1e9:.2f}B")
        print(f"  loaded on: {next(self._model.parameters()).device}")

    def tokenize(self, text: str) -> torch.Tensor:
        assert self._tokenizer is not None, "Call load() before tokenize()"
        assert self._device is not None

        messages = [{"role": "user", "content": text}]
        encoding = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return encoding["input_ids"].to(self._device)

    def get_input_token_strings(self, input_ids: torch.Tensor) -> list[str]:
        assert self._tokenizer is not None

        ids_on_cpu = input_ids[0].tolist()
        return self._tokenizer.convert_ids_to_tokens(ids_on_cpu)

    def decode_token_id(self, token_id: int) -> str:
        assert self._tokenizer is not None
        tokens = self._tokenizer.convert_ids_to_tokens([token_id])
        return tokens[0] if tokens else "<unknown>"

    def get_user_content_start(self, text: str) -> int:
        assert self._tokenizer is not None

        # Use a sentinel to find exactly where the user's content lands in the
        # formatted string, then count tokens up to that point.
        sentinel = "\x00CONTENT_START\x00"
        formatted = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": sentinel + text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        sentinel_char_pos = formatted.index(sentinel)
        prefix = formatted[:sentinel_char_pos]
        prefix_ids = self._tokenizer(prefix, add_special_tokens=False)["input_ids"]
        return len(prefix_ids)

    def run_forward_pass(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self._model is not None, "Call load() before run_forward_pass()"

        with torch.no_grad():
            outputs = self._model(input_ids)

        return outputs.logits

    @staticmethod
    def _select_dtype() -> torch.dtype:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("  bfloat16 not supported on this device, falling back to float16")
        return torch.float16
