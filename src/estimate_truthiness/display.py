from .adapters.base import ModelAdapter


def print_model_header(adapter: ModelAdapter, prompt: str, response: str) -> None:
    """Print a header identifying the model, prompt, and response."""
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"Model:  {adapter.model_name}")
    print(f'Prompt: "{prompt}"')
    print(f'Response: "{response}"')
    print(separator)


def print_generation_steps(
    steps: list[tuple[str, list[tuple[str, float]]]],
    top_n: int = 5,
) -> None:
    """Print the top-n candidates considered at each generation step.

    Each step shows the token the model actually chose (greedy) along with the
    top-n ranked alternatives, so you can see both what the model said
    and what it was considering instead.

    Args:
        steps: Output from generate_with_logits — a list of
            (generated_token, top_k_predictions) pairs. The generated_token is
            always the first entry in top_k_predictions (highest probability).
        top_n: Number of top candidates to display (default: 5).
    """
    for step, (generated_token, top_predictions) in enumerate(steps):
        # Remove Ġ character for display
        clean_generated = generated_token.replace("Ġ", "")
        print(f"\n  Step {step} | generated: {clean_generated!r}")
        print("  " + "-" * 50)

        # Only show top_n predictions
        for rank, (token_string, probability) in enumerate(
            top_predictions[:top_n], start=1
        ):
            # Remove Ġ character for display
            clean_token = token_string.replace("Ġ", "")
            probability_percent = probability * 100
            print(f"  {rank:2d}. {clean_token!r:<20}  {probability_percent:6.2f}%")
