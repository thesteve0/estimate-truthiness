import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .base import ModelAdapter


class PythiaAdapter(ModelAdapter):
    """Adapter for EleutherAI/pythia-6.9b.

    Pythia uses byte-level BPE tokenization (GPT-NeoX style), so token strings
    use Ġ (U+0120) as a space prefix rather than the ▁ (U+2581) used by OLMo's
    SentencePiece tokenizer. This difference is visible in the output and is
    informative when comparing token distributions across models.

    Pythia was purpose-built for interpretability research and provides
    154 intermediate training checkpoints, making it uniquely suited for
    studying how probability distributions evolve during training.
    """

    HF_MODEL_ID = "EleutherAI/pythia-6.9b"

    def __init__(self) -> None:
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._device: torch.device | None = None

    @property
    def model_name(self) -> str:
        return "Pythia 6.9B (EleutherAI/pythia-6.9b)"

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

        encoding = self._tokenizer(text, return_tensors="pt")
        return encoding["input_ids"].to(self._device)

    def get_input_token_strings(self, input_ids: torch.Tensor) -> list[str]:
        assert self._tokenizer is not None

        ids_on_cpu = input_ids[0].tolist()
        return self._tokenizer.convert_ids_to_tokens(ids_on_cpu)

    def decode_token_id(self, token_id: int) -> str:
        assert self._tokenizer is not None
        tokens = self._tokenizer.convert_ids_to_tokens([token_id])
        return tokens[0] if tokens else "<unknown>"

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
