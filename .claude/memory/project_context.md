---
name: Project context — estimate-truthiness
description: Goals, current state, model choices, and key technical decisions for this project
type: project
---

**Goal**: Study token probability distributions (logits) from LLMs. Generate responses autoregressively and inspect the top-k candidate tokens at each generation step to understand how models assign probability, where they are confident vs. uncertain, and what hallucination looks like in logit space.

**Current state (as of 2026-05-04)**: Working implementation with OLMo 3 7B Instruct active. Two test prompts in use:
1. "Please complete this sentence: Mary had a little" — high-confidence, near-deterministic completion
2. "In one sentence, please tell me how many steps lead up to my front door?" — hallucination-prone, specific factual recall

**Why:** The user wants to learn about LLM internals through exploration and experiment, not just run existing tools. This is a research/learning project.

**How to apply:** When suggesting new experiments or features, frame them in terms of what they reveal about model behavior. Expect iterative discussion before implementation.

**Three models planned** (all truly open source — training data + methodology published):
1. `allenai/Olmo-3-7B-Instruct` — active, instruct variant, newest (Nov 2025)
2. `allenai/OLMo-2-1124-7B` — base model, comparison baseline, commented out in run_inference.py
3. `EleutherAI/pythia-6.9b` — base model, 154 training checkpoints (useful for studying how distributions change during training), commented out

**No quantization**: Running bfloat16 on ROCm (Ryzen AI Max+ 395, ~96GB unified memory). Quantization would distort logit distributions.

**Generation approach**: Autoregressive with greedy decoding. Each step: full forward pass → last-position logits → top-k → pick top token → append → repeat. Stops at `<|im_end|>`.
