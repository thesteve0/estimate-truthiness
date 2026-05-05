"""Entry point for token probability exploration.

Generates a one-sentence response from each active model and prints the top-10
candidate tokens considered at each generation step. This lets you see not just
what the model said, but what it was considering at every word.

Test prompts are chosen to reveal different distribution behaviors:
  - High-confidence: model has seen this pattern many times; expect peaked distributions
  - Hallucination-prone: requires specific factual recall; watch for confident wrong answers
    or spread uncertainty across many candidates

To add a model, uncomment its adapter in ACTIVE_ADAPTERS.
To change the number of top candidates shown, adjust TOP_K_PREDICTIONS.
"""

from src.estimate_truthiness.adapters.base import select_compute_device
from src.estimate_truthiness.adapters.olmo3 import OLMo3Adapter

# from src.estimate_truthiness.adapters.olmo2 import OLMo2Adapter
# from src.estimate_truthiness.adapters.pythia import PythiaAdapter
from src.estimate_truthiness.display import print_generation_steps, print_model_header
from src.estimate_truthiness.logit_inspector import generate_with_logits

ACTIVE_ADAPTERS = [
    OLMo3Adapter(),
    # OLMo2Adapter(),
    # PythiaAdapter(),
]

TOP_K_PREDICTIONS = 10

TEST_PROMPTS = [
    # High-confidence: completion of a well-known nursery rhyme.
    # Expect dominant probability on "lamb" and the rest of the rhyme.
    "Please complete this sentence: Mary had a little",
    # Hallucination-prone: requires recall of a specific count.
    # Watch whether the model is uncertain (spread distribution)
    # or confidently wrong (peaked distribution on an incorrect number).
    "In one sentence, please tell me how many steps lead up to my front door?",
]


def run_for_adapter(adapter, device):
    adapter.load(device)

    for prompt in TEST_PROMPTS:
        input_ids = adapter.tokenize(prompt)
        steps = generate_with_logits(adapter, input_ids, TOP_K_PREDICTIONS)

        # Construct full response from generated tokens
        response = "".join(token for token, _ in steps).replace("Ġ", " ").strip()
        # Remove stop token from response
        response = response.replace("<|im_end|>", "").strip()

        print_model_header(adapter, prompt, response)
        print_generation_steps(steps)


def main():
    device = select_compute_device()

    for adapter in ACTIVE_ADAPTERS:
        run_for_adapter(adapter, device)


if __name__ == "__main__":
    main()
