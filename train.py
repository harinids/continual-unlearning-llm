#!/usr/bin/env python3
"""
CLI entrypoint for the continual unlearning + memory consolidation pipeline.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --dataset hatexplain --no-lora
"""
from __future__ import annotations
import argparse
import sys

from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Continual Unlearning with Memory Consolidation")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--dataset", type=str, default=None, choices=["tofu", "hatexplain", "synthetic"],
                   help="Override dataset.name from the config.")
    p.add_argument("--base-model", type=str, default=None, help="Override model.base_model.")
    p.add_argument("--no-lora", action="store_true", help="Disable LoRA, fine-tune full model.")
    p.add_argument("--max-rounds", type=int, default=None, help="Override controller.max_rounds.")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.dataset:
        cfg.dataset.name = args.dataset
    if args.base_model:
        cfg.model.base_model = args.base_model
    if args.no_lora:
        cfg.model.use_lora = False
    if args.max_rounds:
        cfg.controller.max_rounds = args.max_rounds
    if args.seed:
        cfg.seed = args.seed

    set_seed(cfg.seed)

    print(f"[train] Loading dataset '{cfg.dataset.name}'...")
    from src.data.datasets import load_unlearning_dataset
    dataset = load_unlearning_dataset(cfg)
    print(f"[train] {dataset}")

    print(f"[train] Loading model '{cfg.model.base_model}' (LoRA={cfg.model.use_lora})...")
    from src.models.model_utils import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    from src.pipeline import UnlearningPipeline
    pipeline = UnlearningPipeline(cfg, model, tokenizer, dataset)
    results = pipeline.run()

    print("\n[train] === Final Summary ===")
    print(f"Baseline audit: {results['baseline']}")
    print(f"Final audit:    {results['final_audit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
