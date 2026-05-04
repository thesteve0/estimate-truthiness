---
name: Feedback and working preferences
description: Corrections and confirmed approaches from working sessions — how to behave on this project
type: feedback
---

**Keep all Claude files in the project `.claude/` directory, not `~/.claude/`.**
**Why:** Devcontainer environment — home directory is lost on rebuild. Everything must be in the git repo at `/workspaces/estimate-truthiness/.claude/`.
**How to apply:** Memory files, plans, settings — always write to `.claude/` inside the project. Never write to `/home/stpousty-devcontainer/.claude/`.

---

**Never use pip, including `uv pip`.**
**Why:** Project uses `uv` exclusively for package management. Even `uv pip` is disallowed.
**How to apply:** Always use `uv add`, `uv remove`, `uv sync`. Read `pyproject.toml` for dependency info.

---

**Don't use sentence-ending punctuation as a stop token for instruct models.**
**Why:** Instruct models often open responses with acknowledgments like "Sure!" — stopping on `!` terminates generation before the actual answer. Use `<|im_end|>` (the model's end-of-turn marker) instead.
**How to apply:** For any instruct model adapter, `STOP_TOKENS` should contain `<|im_end|>`, not `.`, `?`, `!`.

---

**`apply_chat_template` with `return_tensors="pt"` returns a `BatchEncoding`, not a plain tensor.**
**Why:** Discovered when `input_ids[0].tolist()` raised `AttributeError: 'tokenizers.Encoding' object has no attribute 'tolist'`.
**How to apply:** Always pair with `return_dict=True` and access `encoding["input_ids"]` explicitly.

---

**The user will modify code directly** (e.g., changed test prompts in `run_inference.py` without asking).
**Why:** He wants to participate in the work, not just review Claude's output.
**How to apply:** Check for user modifications before overwriting files. Don't be surprised by changes between turns.
