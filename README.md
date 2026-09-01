# Continual Unlearning with Memory Consolidation in LLMs

> A framework for selective unlearning and memory consolidation in Large Language Models — balancing the ability to forget specific knowledge while retaining essential capabilities.

---

## Overview

This project addresses a critical challenge in deployed LLMs: how to selectively **unlearn** specific data (e.g., harmful content, private information) without triggering **catastrophic forgetting** of retained knowledge.

The framework integrates:
- **Selective Unlearning** via gradient ascent on forget-set samples
- **Memory Consolidation** via Elastic Weight Consolidation (EWC)
- **Knowledge Auditor** to verify successful erasure
- **Stability-Plasticity Controller** to balance new learning vs. retention

---

## Architecture

```
Input (Forget / Retain / Eval Split)
        │
        ▼
┌─────────────────────────┐
│   Knowledge Auditor     │  ← Identifies what to forget
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Selective Unlearning   │  ← Gradient ascent on forget set
│  (Gradient Ascent)      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Memory Consolidation   │  ← EWC regularization
│  (EWC)                  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Stability-Plasticity    │  ← Controls forgetting vs. retention
│ Controller              │
└─────────────────────────┘
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| LLM Base Model | GPT-2 (via Hugging Face) |
| Fine-tuning | LoRA (Low-Rank Adaptation) |
| Datasets | TOFU, HateXplain |
| Regularization | Elastic Weight Consolidation (EWC) |

---

## Datasets

- **TOFU** — Fictional author dataset for unlearning evaluation
- **HateXplain** — Hate speech dataset with human rationale annotations

Each dataset is split into:
- `retain` — knowledge the model must keep
- `forget` — knowledge targeted for erasure
- `eval` — held-out set for unlearning verification

---

## Getting Started

### Installation

```bash
git clone <this-repo>
cd continual-unlearning-llm
pip install -r requirements.txt
```

### Quick start

```bash
# Run the full pipeline on TOFU with GPT-2 + LoRA (default config)
python train.py --config configs/config.yaml

# Swap datasets / base model from the CLI
python train.py --config configs/config.yaml --dataset hatexplain
python train.py --config configs/config.yaml --base-model gpt2-medium --no-lora

# Audit a saved checkpoint on its own
python scripts/evaluate.py --checkpoint outputs/checkpoints/round_3
```

### Run the test suite

The tests exercise the full pipeline (auditor → gradient-ascent unlearning
→ EWC consolidation → stability-plasticity controller) against a tiny,
randomly-initialized GPT-2-architecture model and synthetic data, so they
run in seconds with no GPU and no model download required:

```bash
pip install pytest
python -m pytest tests/ -v
```

### Project layout

```
configs/config.yaml          All hyperparameters (model, dataset, unlearning,
                              EWC, auditor, controller, logging)
src/data/datasets.py         TOFU / HateXplain loaders + retain/forget/eval
                              splitting, with a synthetic offline fallback
src/models/model_utils.py    Base model + LoRA loading; tiny no-download
                              test model for CI
src/unlearning/              Selective unlearning: gradient_ascent,
                              gradient_difference, and npo variants
src/consolidation/ewc.py     Online Elastic Weight Consolidation
src/auditor/                 Knowledge Auditor: perplexity, exact-match,
                              and membership-inference-gap metrics
src/controller/              Stability-Plasticity Controller: PI feedback
                              loop that tunes the EWC lambda each round
src/pipeline.py              Orchestrates the full round-based loop
train.py                     CLI entrypoint
scripts/evaluate.py          Standalone checkpoint auditing
tests/test_pipeline.py       End-to-end tests on a tiny local model
```

### Notes on running with real pretrained weights

By default `model.base_model: "gpt2"` pulls weights from the Hugging Face
Hub the first time you run `train.py`. If you're working in a
network-restricted sandbox without Hub access, use
`--dataset synthetic` together with `src.models.model_utils.build_tiny_test_model()`
(as the test suite does) to validate the pipeline logic offline, then move
to a real base model once Hub access is available.

---

## Key Results

> *(Results will be updated as experiments complete)*

| Metric | Value |
|---|---|
| Forget-set accuracy drop | TBD |
| Retain-set accuracy | TBD |
| Catastrophic forgetting reduction | TBD |

---

## Project Status

🔬 **Active Research** — June 2026 – Present

Manuscript in preparation for submission.

---

## Author

**Harini B.**
MSc Data Science, CHRIST (Deemed to be University), Bengaluru
[LinkedIn](https://linkedin.com/in/harini-b-870b712b6) • [GitHub](https://github.com/harinids)

---

## Citation

```bibtex
@article{harini2026continual,
  title={Continual Unlearning with Memory Consolidation in Large Language Models},
  author={Harini, B. et al.},
  year={2026},
  note={Manuscript in Preparation}
}
```
