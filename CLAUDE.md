# CLAUDE.md

This file provides context to Claude Code when working on this project.

NEVER use PIP in this project, not even with `uv pip`. All commands dealing with Python package management must either use `uv` or read the TOML file. 
You will also keep all your memory files and other files for claude in the .claude directory in this project  rather than the .claude directory in the 
default user's home directory. We are working in a devcontainer which limits what you can do. But this also means if the devcontainer needs to be rebuilt or if we need to move to a different machine all files will be lost. This is why any files including memory files must be in the git repository. 

## Project Overview

**Purpose**: We are going to experiment with the logits returned from the predictions of LLMs and try to see what patterns we can dervie. 

**Problem Domain**: LLM word prediction and accuracy

**Key Technologies**:
- PyTorch (ROCm-accelerated via devcontainer)
- [transformers, SmolLM]
- [IDEA]

## Codebase Structure

```
src/
├── data/           # Data loading, preprocessing, augmentation
├── models/         # Model architectures and training logic
├── utils/          # Helper functions, logging, visualization
└── experiments/    # Experiment scripts and configurations
```

**Key files**:
- `src/models/model.py` - Main model architecture
- `configs/base_config.yaml` - Default configuration
- [Add your critical files with brief descriptions]

## Development Workflow

**IMPORTANT — Package Management**: This project uses `uv` exclusively. Never use `pip install`
directly. Always use `uv` commands:

```bash
uv add <package>            # Add a runtime dependency (updates pyproject.toml + uv.lock)
uv add --dev <package>      # Add a dev-only dependency
uv remove <package>         # Remove a dependency
uv sync                     # Install/update all dependencies from uv.lock
uv pip install <package>    # One-off install without adding to pyproject.toml (rare)
```

ROCm packages (torch, numpy, etc.) are provided by `/opt/venv` via the `.pth` bridge and are
excluded from `uv` installs automatically. Do not `uv add torch` or similar — they are already
available.

**Common commands**:
```bash
# Training
python src/train.py --config configs/experiment1.yaml

# Evaluation
python src/evaluate.py --checkpoint models/best-model.pth

# Data preprocessing
python src/data/preprocess.py --input datasets/raw --output datasets/processed
```

**Testing**:
```bash
pytest tests/
```

**Linting/Formatting**:
```bash
ruff check src/        # Lint code
ruff format src/       # Format code
ruff check --fix src/  # Auto-fix linting issues
```

## Architectural Decisions

Document key design choices that Claude should understand:

- **Data loading strategy**: [e.g., "Using PyTorch DataLoader with custom Dataset class", "Streaming large datasets from /data"]
- **Model architecture**: [e.g., "Fine-tuning BERT-base", "Custom CNN with ResNet backbone"]
- **Training approach**: [e.g., "Mixed precision training with gradient accumulation", "Distributed training across 2 GPUs"]
- **Experiment tracking**: [e.g., "MLflow for metrics, model versioning in models/"]

## Important Patterns

**Configuration management**:
[Explain how configs work - YAML files, Hydra, argparse, etc.]

**Model checkpointing**:
[Explain checkpoint naming convention, where they're saved, how to load them]

**Data pipeline**:
[Explain data flow from raw → preprocessed → DataLoader → model]

## Known Issues and Gotchas

- [e.g., "Dataset has class imbalance - must use weighted loss"]
- [e.g., "Large models may OOM on integrated GPU - reduce batch size"]
- [e.g., "Preprocessing requires 32GB RAM - run on host if container OOMs"]

## External Dependencies

- **Data sources**: [Where data comes from, how to refresh it]
- **Pretrained models**: [Which models are downloaded, where cached]
- **APIs/Services**: [Any external services the project calls]

## Testing Strategy

[Describe test coverage, what's tested, what's not]

---

**Note**: This is a ROCm devcontainer project. For ROCm-specific troubleshooting (GPU access, dependency conflicts, Python version issues), see `template_docs/CLAUDE.md`.
