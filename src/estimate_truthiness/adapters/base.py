from abc import ABC, abstractmethod

import torch


def select_compute_device() -> torch.device:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {device_name}")
        return torch.device("cuda")
    print("No GPU found, using CPU. Inference will be slow.")
    return torch.device("cpu")


class ModelAdapter(ABC):
    """Common interface for all model adapters.

    Each subclass handles the model-specific details of loading and tokenizing
    so that logit_inspector and display code never need to know which model
    is running.
    """

    @abstractmethod
    def load(self, device: torch.device) -> None:
        """Download (if needed) and load the model and tokenizer into memory."""

    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text and return input_ids on the model's device."""

    @abstractmethod
    def get_input_token_strings(self, input_ids: torch.Tensor) -> list[str]:
        """Decode input token IDs to their raw string forms (subword markers visible)."""

    @abstractmethod
    def decode_token_id(self, token_id: int) -> str:
        """Decode a single token ID to its string form, for displaying top-k candidates."""

    @abstractmethod
    def run_forward_pass(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return logits of shape [1, seq_len, vocab_size]."""

    def get_user_content_start(self, text: str) -> int:
        """Return the token index where the user's prompt content begins.

        Base models are given raw text, so position 0 is always the start of
        the user's content. Instruct model adapters override this to skip over
        the chat template preamble (system prompt, role markers, etc.) so that
        the display only shows predictions for the user's actual words.
        """
        return 0

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name for display headers."""
