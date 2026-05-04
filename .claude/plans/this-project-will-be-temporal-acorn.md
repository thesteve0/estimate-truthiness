# Plan: Initial LLM Logit Exploration — Multi-Model Setup + First Inference

## Context

The project will study token probability distributions (logits) from LLMs to understand patterns in how models assign probability to words. The first milestone is: implement a model-agnostic inference framework that supports three truly open-source 4-7B models, run one sentence through all three, and display the top 10 predicted tokens per position with their probabilities. The user wants to learn by reading and writing code — comparing the same sentence across models is part of that learning.

**Models (all truly open source — training data + methodology published):**
1. `allenai/OLMo-3-7B` — newest (Nov 2025), 9.3T token corpus, start here
2. `EleutherAI/pythia-6.9b` — purpose-built for interpretability, 154 training checkpoints, GPT-NeoX architecture
3. `allenai/OLMo-2-1124-7B` — proven (Nov 2024), good baseline for OLMo 2 vs 3 comparison

**Quantization decision:** Skip quantization. ROCm setup (~96GB unified memory) handles 7B bfloat16 (~14GB) comfortably. Running full precision means patterns in the distributions are real, not artifacts.

## Architecture: Adapter Pattern

The visualization and analysis code must be insulated from model-specific loading and calling conventions. Each model differs in: tokenizer class, loading parameters (some need `trust_remote_code=True`), and tokenizer output format (SentencePiece markers vs GPT-NeoX byte-level BPE). An adapter layer handles all of that.

```
run_inference.py                           ← entry point: picks adapter, orchestrates
src/estimate-truthiness/
├── adapters/
│   ├── base.py                            ← abstract ModelAdapter base class
│   ├── olmo3.py                           ← OLMo 3 adapter
│   ├── olmo2.py                           ← OLMo 2 adapter
│   └── pythia.py                          ← Pythia adapter
├── logit_inspector.py                     ← model-agnostic: forward pass + top-k extraction
└── display.py                             ← model-agnostic: output formatting
```

`logit_inspector.py` and `display.py` never import from any adapter module directly — they only call methods defined in `base.py`.

## Dependencies to Add

```bash
uv add transformers accelerate safetensors
```

## Implementation Plan

### `src/estimate-truthiness/adapters/base.py`

Abstract base class that defines the interface all adapters must implement. Using Python's `abc` module.

```python
class ModelAdapter(ABC):
    @abstractmethod
    def load(self, device: torch.device) -> None: ...
    
    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor: ...
    # Returns input_ids tensor, moved to the correct device
    
    @abstractmethod
    def get_input_token_strings(self, input_ids: torch.Tensor) -> list[str]: ...
    # Decodes input token IDs to human-readable strings (with subword markers visible)
    
    @abstractmethod
    def decode_token_id(self, token_id: int) -> str: ...
    # Decodes a single predicted token ID to its string form (for top-k display)
    
    @abstractmethod
    def run_forward_pass(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    # Returns logits tensor of shape [1, seq_len, vocab_size], inside torch.no_grad()
    
    @property
    @abstractmethod
    def model_name(self) -> str: ...
    # Human-readable name for display headers
```

Also includes `select_compute_device() -> torch.device` as a module-level function (not a method — it's shared logic, not model-specific).

### `src/estimate-truthiness/adapters/olmo3.py`

```python
class OLMo3Adapter(ModelAdapter):
    HF_MODEL_ID = "allenai/OLMo-3-7B"
```

Loading details:
- Verify on model card whether `trust_remote_code=True` is required (OLMo 3 may need it; OLMo 2 did not)
- `torch_dtype=torch.bfloat16` with fallback to `float16` if `torch.cuda.is_bf16_supported()` returns False
- `device_map="auto"` via accelerate
- `model.eval()` before returning
- Prints model parameter count and actual dtype loaded

Tokenizer: Uses SentencePiece (`▁` space markers). `convert_ids_to_tokens()` shows these markers — leave them visible.

### `src/estimate-truthiness/adapters/olmo2.py`

```python
class OLMo2Adapter(ModelAdapter):
    HF_MODEL_ID = "allenai/OLMo-2-1124-7B"
```

Same loading approach as OLMo3 but `trust_remote_code=False`. Nearly identical to OLMo3Adapter — may share logic via a common OLMo base class if the duplication becomes obvious, but don't abstract prematurely.

### `src/estimate-truthiness/adapters/pythia.py`

```python
class PythiaAdapter(ModelAdapter):
    HF_MODEL_ID = "EleutherAI/pythia-6.9b"
```

Key differences from OLMo adapters:
- Uses `GPTNeoXTokenizerFast` (byte-level BPE, not SentencePiece). Token strings will look different (e.g., `Ġ` instead of `▁` for space prefix). Display should still use `convert_ids_to_tokens()` — just note in output that token format differs.
- `trust_remote_code=False`
- `AutoModelForCausalLM` still works (transformers handles the architecture)
- Same bfloat16/float16 logic

### `src/estimate-truthiness/logit_inspector.py`

Two functions, both fully model-agnostic (take `adapter: ModelAdapter` as parameter):

**`extract_top_predictions(logits: torch.Tensor, adapter: ModelAdapter, top_k: int) -> list[list[tuple[str, float]]]`**
- `torch.softmax(logits[0], dim=-1)` — softmax over vocab dimension (dim=-1)
- `torch.topk(probabilities, k=top_k, dim=-1)` — top-k over vocab dimension
- Loop over positions; call `adapter.decode_token_id(id)` for each candidate
- Move to CPU before converting to Python floats
- Returns: outer list = positions, inner list = (token_string, probability) tuples

### `src/estimate-truthiness/display.py`

**`print_model_header(adapter: ModelAdapter) -> None`**  
Prints model name and a separator line before showing its results.

**`print_position_predictions(input_tokens: list[str], predictions: list[list[tuple[str, float]]], top_k: int) -> None`**  
For each position: position index + input token as header, then ranked candidates with probability as percentage (`23.41%`, not `23%`). Leave subword markers visible — they convey tokenizer behavior.

### `run_inference.py` (entry point)

```python
from src.estimate_truthiness.adapters.olmo3 import OLMo3Adapter
from src.estimate_truthiness.adapters.olmo2 import OLMo2Adapter
from src.estimate_truthiness.adapters.pythia import PythiaAdapter

ADAPTERS = [OLMo3Adapter()]  # OLMo2Adapter(), PythiaAdapter() — uncomment when ready
TOP_K_PREDICTIONS = 10
TEST_SENTENCES = [
    "Mary had a little",           # high-confidence: near-deterministic next tokens
    "The exact number of stairs in the Eiffel Tower is",  # hallucination-prone: specific factual recall
]

def run_for_adapter(adapter, device):
    adapter.load(device)
    print_model_header(adapter)
    for sentence in TEST_SENTENCES:
        input_ids = adapter.tokenize(sentence)
        input_tokens = adapter.get_input_token_strings(input_ids)
        logits = adapter.run_forward_pass(input_ids)
        predictions = extract_top_predictions(logits, adapter, TOP_K_PREDICTIONS)
        print_position_predictions(input_tokens, predictions, TOP_K_PREDICTIONS)

def main():
    device = select_compute_device()
    for adapter in ADAPTERS:
        run_for_adapter(adapter, device)
```

On first run, only OLMo 3 will be active (comment out the other two until ready). All three constants are defined so the pattern is visible from the start.

## Key Technical Notes

- **Logit shape**: `outputs.logits` is `[batch, seq_len, vocab_size]`. Operate on `logits[0]` (shape `[seq_len, vocab_size]`) since we run single sentences without batching.
- **HuggingFace cache**: Set `HF_HOME=/workspaces/estimate-truthiness/.cache/huggingface` before the first run so model downloads (~14GB each) go into the project directory and survive container rebuilds. The `.cache/` dir is already gitignored.
- **`model.eval()` + `torch.no_grad()`**: Both required. `model.eval()` disables dropout (affects distributions). `torch.no_grad()` saves memory during forward pass.
- **Token display**: Use `convert_ids_to_tokens()` not `decode()`. Shows subword markers (`▁`, `Ġ`) which are meaningful for understanding tokenizer behavior and comparing across models.
- **Module naming**: The `src/estimate-truthiness/` directory name has a hyphen, which Python can't import directly. Either rename to `estimate_truthiness` or use `importlib`. The existing `src/estimate-truthiness/__init__.py` exists but would need the directory renamed. Rename to `src/estimate_truthiness/` for clean imports.

## Test Sentences

Two sentences chosen to reveal different distribution behaviors:

**1. High-confidence sentence (near-deterministic):**
```
"Mary had a little"
```
Rationale: Extremely common in training data. At "little", the model should assign dominant probability mass to "lamb" — very sharp, peaked distribution. At "had", it should strongly favor "a". This gives us a baseline for what high-confidence looks like in the logits.

**2. Hallucination-prone sentence (specific factual recall):**
```
"The exact number of stairs in the Eiffel Tower is"
```
Rationale: Requires recall of a precise count the model likely has imprecise knowledge of. Two interesting outcomes are possible: (a) spread-out distribution across many numbers = model knows it doesn't know, or (b) high confidence on a specific wrong number = dangerous hallucination. Both are informative about how the model handles factual uncertainty.

## Output Format (Simplified)

For each token position, print:
- Position index and the input token at that position
- Top 10 predicted next tokens with their probabilities (as percentages)

Nothing else in the initial version. No entropy, no charts, no aggregate statistics — those come later.

## Verification

1. `uv add transformers accelerate safetensors` completes without breaking ROCm packages
2. `python run_inference.py` with only OLMo 3 active: downloads model, prints top-10 next-token predictions per position for both sentences
3. "Mary had a little" output: position for "little" shows "lamb" as dominant prediction with high probability
4. Eiffel Tower sentence: output reveals whether model is confident or uncertain about the specific number
5. Enable OLMo 2 and Pythia adapters one at a time: compare output across models for the same sentences
6. `ruff check src/` passes
