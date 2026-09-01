#!/usr/bin/env python3
"""
Standalone auditing script: run the Knowledge Auditor against a saved
checkpoint without re-running the full unlearning pipeline.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/round_3 \
        --config configs/config.yaml
"""
from __future__ import annotations
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[evaluate] Loading checkpoint from {args.checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from src.data.datasets import load_unlearning_dataset
    from src.pipeline import make_loader
    from src.auditor.knowledge_auditor import KnowledgeAuditor

    dataset = load_unlearning_dataset(cfg)
    forget_loader = make_loader(dataset.forget, tokenizer, cfg, cfg.auditor.eval_batch_size, shuffle=False)
    retain_loader = make_loader(dataset.retain, tokenizer, cfg, cfg.auditor.eval_batch_size, shuffle=False)
    eval_loader = make_loader(dataset.eval, tokenizer, cfg, cfg.auditor.eval_batch_size, shuffle=False)

    auditor = KnowledgeAuditor(model, tokenizer, device, cfg)
    report = auditor.audit(forget_loader, retain_loader, eval_loader, raw_forget_examples=dataset.forget)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
