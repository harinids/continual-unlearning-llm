"""
End-to-end smoke tests using a tiny, randomly-initialized GPT-2-architecture
model (no Hub download required) and synthetic data. These validate that
the unlearning / EWC / auditor / controller code paths are wired correctly,
independent of real pretrained weights or network access.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.utils.config import Config
from src.utils.seed import set_seed


def make_test_config(tmp_path):
    return Config({
        "seed": 0,
        "device": "cpu",
        "model": {
            "base_model": "tiny-test",
            "use_lora": False,
            "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["c_attn"]},
            "max_length": 32,
        },
        "dataset": {
            "name": "synthetic",
            "forget_split": "forget", "retain_split": "retain", "eval_split": "eval",
            "cache_dir": str(tmp_path / "cache"),
            "num_forget_samples": 8, "num_retain_samples": 16, "num_eval_samples": 8,
        },
        "unlearning": {
            "method": "gradient_ascent", "lr": 1e-3, "epochs": 1, "batch_size": 2,
            "grad_clip": 1.0, "forget_loss_weight": 1.0, "retain_loss_weight": 1.0,
        },
        "ewc": {
            "enabled": True, "lambda_init": 10.0, "fisher_num_samples": 8,
            "fisher_batch_size": 2, "online": True,
        },
        "auditor": {
            "metrics": ["perplexity", "membership_inference"],
            "forget_threshold": 0.3, "retain_threshold": 0.85, "eval_batch_size": 2,
        },
        "controller": {
            "strategy": "adaptive", "plasticity_init": 1.0, "stability_init": 1.0,
            "kp": 0.5, "ki": 0.1, "max_rounds": 2, "patience": 2,
        },
        "logging": {
            "output_dir": str(tmp_path / "outputs"),
            "log_dir": str(tmp_path / "outputs/logs"),
            "checkpoint_dir": str(tmp_path / "outputs/checkpoints"),
            "save_every_round": False, "use_tensorboard": False,
        },
    })


@pytest.fixture
def cfg(tmp_path):
    set_seed(0)
    return make_test_config(tmp_path)


def test_synthetic_dataset_loads(cfg):
    from src.data.datasets import load_unlearning_dataset
    ds = load_unlearning_dataset(cfg)
    assert len(ds.forget) == 8
    assert len(ds.retain) == 16
    assert len(ds.eval) == 8


def test_tiny_model_builds():
    from src.models.model_utils import build_tiny_test_model
    model, tokenizer = build_tiny_test_model()
    assert model is not None
    assert tokenizer.pad_token is not None


def test_ewc_penalty_is_zero_at_snapshot(cfg):
    """Right after fitting, EWC penalty should be exactly 0 (params == snapshot)."""
    from src.models.model_utils import build_tiny_test_model
    from src.pipeline import make_loader
    from src.consolidation.ewc import EWC
    from src.data.datasets import load_unlearning_dataset

    model, tokenizer = build_tiny_test_model()
    dataset = load_unlearning_dataset(cfg)
    loader = make_loader(dataset.retain, tokenizer, cfg, cfg.ewc.fisher_batch_size)

    ewc = EWC(model, loader, device="cpu", online=cfg.ewc.online)
    penalty = ewc.penalty(model).item()
    assert penalty == pytest.approx(0.0, abs=1e-5)


def test_gradient_ascent_increases_forget_loss(cfg):
    """After a round of gradient-ascent unlearning, loss on the forget set should rise."""
    import torch
    from src.models.model_utils import build_tiny_test_model
    from src.pipeline import make_loader
    from src.unlearning.gradient_ascent import SelectiveUnlearner
    from src.auditor.knowledge_auditor import KnowledgeAuditor
    from src.data.datasets import load_unlearning_dataset

    cfg.unlearning.epochs = 2
    model, tokenizer = build_tiny_test_model()
    dataset = load_unlearning_dataset(cfg)

    forget_loader = make_loader(dataset.forget, tokenizer, cfg, cfg.unlearning.batch_size)
    retain_loader = make_loader(dataset.retain, tokenizer, cfg, cfg.unlearning.batch_size)

    auditor = KnowledgeAuditor(model, tokenizer, "cpu", cfg)
    ppl_before = auditor._perplexity(forget_loader)

    unlearner = SelectiveUnlearner(model, tokenizer, "cpu", cfg, ewc=None)
    unlearner.run_round(forget_loader, retain_loader, ewc_lambda=0.0)

    ppl_after = auditor._perplexity(forget_loader)
    assert ppl_after >= ppl_before  # forgetting should not decrease forget-set perplexity


def test_controller_lowers_lambda_when_forget_leaks(cfg):
    from src.controller.stability_plasticity import StabilityPlasticityController

    controller = StabilityPlasticityController(cfg)
    lambda_before = controller.lambda_ewc
    # Simulate a report where forget-set leakage is above threshold.
    report = {"forget_exact_match": 0.9, "retain_ppl": 10.0}
    baseline = {"retain_ppl": 10.0}
    controller.step(report, baseline=baseline)
    assert controller.lambda_ewc <= lambda_before


def test_full_pipeline_runs_end_to_end(cfg):
    """Integration test: run the full pipeline for 2 rounds on the tiny model."""
    from src.models.model_utils import build_tiny_test_model
    from src.data.datasets import load_unlearning_dataset
    from src.pipeline import UnlearningPipeline

    model, tokenizer = build_tiny_test_model()
    dataset = load_unlearning_dataset(cfg)

    pipeline = UnlearningPipeline(cfg, model, tokenizer, dataset)
    results = pipeline.run()

    assert "baseline" in results
    assert "final_audit" in results
    assert len(results["rounds"]) >= 1
