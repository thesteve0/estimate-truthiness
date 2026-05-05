import torch

from .adapters.base import ModelAdapter

# The vocab dimension is always the last dimension in the logits tensor.
VOCAB_DIM = -1

# Exact stop tokens: model's end-of-turn markers
EXACT_STOP_TOKENS = {
    "<|im_end|>",      # OLMo's end-of-turn token
    "<|endoftext|>",   # GPT-style end-of-text token
    "**ĊĊ",    # For some reason models doesn't always use stop punctuation and it should stop here with Mary had a little lamb
}

# Stop patterns: tokens ending with sentence-ending punctuation + newlines.
# Be specific to avoid false positives like ":ĊĊ" (formatting, not sentence end).
STOP_SUFFIXES = [
    ".ĊĊ",   # Period + double newline
    "!ĊĊ",   # Exclamation + double newline
    "?ĊĊ",   # Question + double newline
    ".Ċ",    # Period + newline
    "!Ċ",    # Exclamation + newline
    "?Ċ",    # Question + newline
    "Ġ.",    # Space + period

]


def generate_with_logits(
    adapter: ModelAdapter,
    input_ids: torch.Tensor,
    top_k: int,
    max_new_tokens: int = 100,
) -> list[tuple[str, list[tuple[str, float]]]]:
    """Generate a response one token at a time, recording top-k candidates at each step.

    At each step, runs a full forward pass on the current sequence, looks only
    at the last-position logits (the distribution over what comes next), picks
    the highest-probability token (greedy decoding), appends it, and repeats.

    Stops when a sentence-ending token is generated or max_new_tokens is reached.

    Args:
        adapter: Model adapter providing forward pass and token decoding.
        input_ids: The fully formatted prompt as token IDs, shape [1, seq_len].
        top_k: Number of top candidates to record at each generation step.
        max_new_tokens: Hard limit on generated tokens to prevent runaway output.

    Returns:
        A list of (generated_token_string, top_k_predictions) pairs, one per
        generated token. top_k_predictions is a list of (token_string, probability)
        tuples ordered from most to least likely.
    """
    current_ids = input_ids.clone()
    generation_steps: list[tuple[str, list[tuple[str, float]]]] = []

    for _ in range(max_new_tokens):
        logits = adapter.run_forward_pass(current_ids)

        # We only care about the very last position — the next token to generate.
        # logits shape: [1, seq_len, vocab_size]; index [0, -1, :] gives [vocab_size].
        next_token_logits = logits[0, -1, :]

        probabilities = torch.softmax(next_token_logits, dim=VOCAB_DIM)
        top_probs, top_indices = torch.topk(probabilities, k=top_k)

        top_predictions: list[tuple[str, float]] = [
            (adapter.decode_token_id(int(idx)), float(prob))
            for idx, prob in zip(top_indices.cpu(), top_probs.cpu())
        ]

        # Greedy: always pick the most probable token.
        next_token_id = int(top_indices[0])
        generated_token = adapter.decode_token_id(next_token_id)

        generation_steps.append((generated_token, top_predictions))

        next_token_tensor = torch.tensor([[next_token_id]], device=current_ids.device)
        current_ids = torch.cat([current_ids, next_token_tensor], dim=1)

        # Check exact stop tokens
        if generated_token in EXACT_STOP_TOKENS:
            break

        # Check suffix patterns
        if any(generated_token.endswith(suffix) for suffix in STOP_SUFFIXES):
            break

    return generation_steps