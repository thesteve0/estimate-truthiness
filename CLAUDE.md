# CLAUDE.md

This file provides context to Claude Code when working on this project.

NEVER use PIP in this project, not even with `uv pip`. All commands dealing with Python package management must either use `uv` or read the TOML file.

You will also keep all your memory files and other files for Claude in the `.claude` directory in this project rather than the `.claude` directory in the default user's home directory. We are working in a devcontainer which limits what you can do. But this also means if the devcontainer needs to be rebuilt or if we need to move to a different machine all files will be lost. This is why any files including memory files must be in the git repository.

## Project Overview

**Purpose**: Explore and visualize token probability distributions (logits) from LLMs. The project generates model responses autoregressively and inspects the top-k candidate tokens and their probabilities at each generation step — revealing not just what the model said, but what it was considering.

**Problem Domain**: LLM interpretability — understanding how models assign probability to tokens, what uncertainty looks like in logit distributions, and how different models (or model versions) differ in their predictions.

**Key Technologies**:
- PyTorch (ROCm-accelerated via devcontainer)
- HuggingFace `transformers` (≥ 5.7.0), `accelerate`, `safetensors`
- OLMo 3 7B Instruct (`allenai/Olmo-3-7B-Instruct`) — primary model
- OLMo 2 7B (`allenai/OLMo-2-1124-7B`) — comparison baseline
- Pythia 6.9B (`EleutherAI/pythia-6.9b`) — interpretability-focused, 154 training checkpoints

All three models are **truly open source**: training data, training code, and methodology are published (not just weights).

## Codebase Structure

```
run_inference.py                        # Entry point — run this to execute
src/estimate_truthiness/
├── adapters/
│   ├── base.py                         # Abstract ModelAdapter + select_compute_device()
│   ├── olmo3.py                        # OLMo 3 Instruct adapter (active)
│   ├── olmo2.py                        # OLMo 2 base adapter (commented out in entry point)
│   └── pythia.py                       # Pythia 6.9B adapter (commented out in entry point)
├── logit_inspector.py                  # Autoregressive generation with logit capture
└── display.py                          # Formatted output
.claude/
├── MEMORY.md                           # Index of memory files
├── memory/                             # Individual memory files
└── plans/                              # Implementation plans
```

**Key files**:
- `run_inference.py` — entry point; controls which models and prompts are active
- `src/estimate_truthiness/adapters/base.py` — `ModelAdapter` abstract class; all analysis code depends only on this interface, never on a concrete adapter
- `src/estimate_truthiness/logit_inspector.py` — `generate_with_logits()`: autoregressive generation that captures top-k distributions at every step
- `src/estimate_truthiness/display.py` — `print_generation_steps()`: renders the per-step output

## Development Workflow

**IMPORTANT — Package Management**: This project uses `uv` exclusively. Never use `pip install` directly.

```bash
uv add <package>            # Add a runtime dependency (updates pyproject.toml + uv.lock)
uv add --dev <package>      # Add a dev-only dependency
uv remove <package>         # Remove a dependency
uv sync                     # Install/update all dependencies from uv.lock
```

ROCm packages (torch, numpy, etc.) are provided by `/opt/venv` via the `.pth` bridge and are excluded from `uv` installs automatically. Do not `uv add torch` or similar — they are already available.

**Running**:
```bash
uv run python run_inference.py
```

**Linting/Formatting**:
```bash
ruff check src/        # Lint
ruff format src/       # Format
ruff check --fix src/  # Auto-fix
```

**Testing**:
```bash
pytest tests/
```

## Architectural Decisions

**Adapter pattern**: All model-specific logic (loading, tokenization, chat template formatting) is encapsulated in adapter classes under `src/estimate_truthiness/adapters/`. The `logit_inspector` and `display` modules depend only on the `ModelAdapter` abstract interface. Adding a new model means writing a new adapter file — nothing else changes.

**Autoregressive generation with logit inspection**: Rather than a single forward pass over the full input, `generate_with_logits()` generates response tokens one at a time. At each step it runs a full forward pass, looks only at the last-position logits, records the top-k distribution, picks the highest-probability token (greedy decoding), appends it, and repeats. This makes the model's decision process visible at every word of its response.

**No quantization**: All models run in bfloat16 (float16 fallback if bfloat16 unsupported). The ROCm hardware (Ryzen AI Max+ 395, ~96GB unified memory) handles 7B models comfortably. Quantization would distort the logit distributions we are studying.

**Instruct vs. base models**: OLMo 3 uses the instruct variant (`Olmo-3-7B-Instruct`) so prompts can be written as natural questions. Instruct models format input via `apply_chat_template` with `add_generation_prompt=True`. OLMo 2 and Pythia are base models — their adapters pass raw text.

**Stop token**: Generation stops on `<|im_end|>` (OLMo's end-of-turn marker). This is correct for instruct models; sentence-ending punctuation (`.`, `!`) is not used as a stop condition because instruct models often open with acknowledgments like "Sure!" before giving the actual answer.

## Known Issues and Gotchas

- `apply_chat_template` with `return_tensors="pt"` returns a `BatchEncoding`, not a plain tensor. Must use `return_dict=True` and access `encoding["input_ids"]` explicitly.
- The `src/estimate_truthiness/` directory was renamed from `src/estimate-truthiness/` (hyphen → underscore) for Python import compatibility.
- ROCm prints attention kernel warnings (`TRITON_ENABLE_EXPERIMENTAL=1`, `sdp_utils.cpp`) to stderr during inference. These are informational, not errors — inference proceeds normally.
- HuggingFace model cache: `HF_HOME` is set to `/workspaces/estimate-truthiness/.cache/huggingface` in the devcontainer environment. Models download there (~14GB each) and persist in the project directory. The `.cache/` directory is gitignored.

## External Dependencies

- **Pretrained models**: Downloaded from HuggingFace Hub on first run, cached in `.cache/huggingface/hub/`. Models are NOT committed to git.
- **No external APIs**: All inference runs locally on the ROCm GPU.

## Testing Strategy

No automated tests yet. Manual verification: run `run_inference.py` and inspect output for plausible distributions (high confidence on "lamb" after "Mary had a little", uncertainty on factual recall prompts).

---

**Note**: This is a ROCm devcontainer project. For ROCm-specific troubleshooting (GPU access, dependency conflicts, Python version issues), see `template_docs/CLAUDE.md`.
