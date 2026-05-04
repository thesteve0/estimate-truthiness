from .adapters.base import ModelAdapter


def print_model_header(adapter: ModelAdapter, prompt: str) -> None:
    """Print a header identifying the model and the prompt being used."""
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"Model:  {adapter.model_name}")
    print(f'Prompt: "{prompt}"')
    print(separator)


def print_generation_steps(
    steps: list[tuple[str, list[tuple[str, float]]]],
) -> None:
    """Print the top-k candidates considered at each generation step.

    Each step shows the token the model actually chose (greedy) along with the
    full ranked list of alternatives, so you can see both what the model said
    and what it was considering instead.

    Args:
        steps: Output from generate_with_logits — a list of
            (generated_token, top_k_predictions) pairs. The generated_token is
            always the first entry in top_k_predictions (highest probability).
    """
    for step, (generated_token, top_predictions) in enumerate(steps):
        print(f"\n  Step {step} | generated: {generated_token!r}")
        print("  " + "-" * 50)

        for rank, (token_string, probability) in enumerate(top_predictions, start=1):
            probability_percent = probability * 100
            print(f"  {rank:2d}. {token_string!r:<20}  {probability_percent:6.2f}%")
