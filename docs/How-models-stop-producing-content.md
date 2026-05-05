# How Models Stop Producing Content

## Overview

Large Language Models (LLMs) don't inherently "know" when to stop generating tokens. They are autoregressive systems that produce probability distributions over the next token indefinitely. **The inference framework**, not the model itself, determines when generation should stop.

## The Model's Perspective

At each generation step, the model:
1. Takes the current sequence of tokens
2. Runs a forward pass through its layers
3. Produces a probability distribution over all possible next tokens
4. Returns logits (unnormalized scores) for every token in the vocabulary

The model has **no built-in stopping mechanism**. It will continue producing predictions forever if we let it.

## Stop Tokens: Signals, Not Commands

Models are trained to **generate** special tokens (like `<|im_end|>`, `</s>`, `<|endoftext|>`) when they want to signal "I'm done with my turn." However:

- **Generating a stop token ≠ stopping generation**
- The stop token is just another token in the vocabulary
- Without intervention, the model would keep generating tokens after the stop token

Think of stop tokens as the model saying "I'm finished" — but it's **our code's responsibility** to listen and actually halt the generation loop.

## How This Project Handles Stopping

### In `logit_inspector.py`

```python
# Stop tokens we recognize
STOP_TOKENS = {"<|im_end|>"}

def generate_with_logits(...):
    for _ in range(max_new_tokens):
        # 1. Generate next token
        generated_token = adapter.decode_token_id(next_token_id)
        
        # 2. Record the token (including stop tokens)
        generation_steps.append((generated_token, top_predictions))
        
        # 3. Append to sequence
        current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
        
        # 4. Check if we should stop
        if generated_token in STOP_TOKENS:
            break  # ← OUR CODE stops the loop
```

**Key insight**: The stop token **is included** in `generation_steps` because we want to study that the model generated it. We remove it later when displaying the response.

### In `run_inference.py`

```python
# Build response from all generated tokens
response = "".join(token for token, _ in steps).replace("Ġ", " ").strip()

# Remove stop token from human-readable output
response = response.replace("<|im_end|>", "").strip()
```

## How Other Inference Systems Handle Stopping

### vLLM

```python
response = llm.generate(
    prompts=["Hello!"],
    stop=["<|im_end|>", "\n\n"],          # Stop conditions
    max_tokens=100,                        # Hard limit
    include_stop_str_in_output=False       # Whether to include stop token in response
)
```

vLLM's generation loop:
1. Generate next token
2. Check if token matches any stop condition
3. If yes → break and return response
4. If no → append token and continue
5. Repeat until stop condition or `max_tokens`

### HuggingFace Transformers

```python
output = model.generate(
    input_ids,
    max_new_tokens=100,
    eos_token_id=tokenizer.eos_token_id,  # End-of-sequence token
    stopping_criteria=StoppingCriteriaList([...])  # Custom stop conditions
)
```

### OpenAI API

```python
response = openai.ChatCompletion.create(
    messages=[...],
    stop=["\n\n", "END"],  # Stop sequences
    max_tokens=100
)
```

### Anthropic API

```python
response = anthropic.messages.create(
    messages=[...],
    stop_sequences=["</answer>", "\n\n"],  # Stop sequences
    max_tokens=100
)
```

## Universal Pattern

**All LLM inference systems follow the same pattern:**

```
while tokens_generated < max_tokens:
    1. Get next token from model
    2. Check stopping conditions:
       - Is this a stop token?
       - Have we hit max_tokens?
       - Custom stopping criteria met?
    3. If stop condition met → BREAK
    4. Else → append token and continue
```

## Types of Stop Conditions

### 1. Stop Tokens
Specific tokens that signal end-of-turn:
- `<|im_end|>` (OLMo, many instruct models)
- `</s>` (Llama, Mistral)
- `<|endoftext|>` (GPT-2, GPT-3)

### 2. Stop Sequences
Multi-token patterns:
- `"\n\n"` (paragraph break)
- `"END"` (literal string)
- `"."` (period — for single-sentence generation)

### 3. Hard Limits
Safety mechanisms to prevent runaway generation:
- `max_tokens` / `max_new_tokens`
- Time limits
- Memory limits

### 4. Custom Criteria
Application-specific conditions:
- Repetition detection (model is looping)
- Semantic completion (answer is complete)
- Safety filters (toxic content detected)

## What Happens Without Stop Conditions?

If you **remove the `break` statement** from the generation loop:

```python
if generated_token in STOP_TOKENS:
    # break  ← commented out
    pass
```

**Result**: The model keeps generating until `max_new_tokens` is reached. Output might look like:

```
Response: "I don't have access to that information.<|im_end|>However, I can tell you that typically residential homes have between 1 and 5 steps leading to the front door.<|im_end|>Would you like me to explain how steps are typically designed?<|im_end|>"
```

The model happily generates multiple responses, stop tokens included, with no awareness that it "finished" several times.

## Streaming and Stop Conditions

When streaming tokens to users (like in vLLM or OpenAI's API):

```python
for chunk in llm.generate(..., stream=True):
    print(chunk.text, end="", flush=True)  # Send token immediately
    # Framework monitors for stop conditions in background
```

**The framework:**
1. Sends each token to the user immediately (streaming)
2. Simultaneously checks stop conditions
3. Halts the stream when a stop condition is detected
4. Closes the connection

The user sees tokens appearing in real-time, but the stream ends cleanly when the model signals completion.

## Best Practices

1. **Always set a `max_tokens` limit** — Prevents runaway generation if stop tokens fail
2. **Know your model's stop tokens** — Check the model card or tokenizer config
3. **Test stop conditions** — Verify they actually halt generation as expected
4. **Consider multi-token sequences** — Single tokens can be unreliable (e.g., period might appear mid-sentence)
5. **Log when you hit `max_tokens`** — Helps debug cases where stop tokens weren't generated

## Summary

- **Models generate tokens endlessly** — They have no built-in stop mechanism
- **Stop tokens are signals** — The model generates them to indicate "I'm done"
- **Inference code must act** — The generation loop must detect stop tokens and break
- **This pattern is universal** — All LLM inference systems work this way
- **Multiple stop conditions** — Combine stop tokens, sequences, and hard limits for robust stopping

The model's job is to predict. Our job is to know when to stop asking.
