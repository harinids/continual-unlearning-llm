"""
GeometryAwareUnlearner — Phase 2's actual novel mechanism.

Same interface as SelectiveUnlearner (run_round(forget_loader, retain_loader,
ewc_lambda, epochs, on_step)) so it's a drop-in swap in pipeline.py /
sequential_pipeline.py. Before each optimizer.step(), projects the forget
gradient to remove any component in the retain subspace (estimated once
per round from a small retain-gradient sample), so retain-relevant
directions are structurally protected rather than penalized after the
fact via EWC.

EWC penalty is kept ON by default alongside this (cfg.geometry.keep_ewc) --
these are complementary, not exclusive: EWC penalizes moving along
Fisher-important directions, subspace projection prevents moving along
retain-gradient directions at all. Set keep_ewc=False for an ablation
isolating the geometry mechanism alone.
"""
from __future__ import annotations
from src.unlearning.gradient_ascent import SelectiveUnlearner, _lm_loss
from src.consolidation.retain_subspace import (
    RetainSubspaceEstimator, _flatten_grads, _scatter_like_grads,
)


class GeometryAwareUnlearner(SelectiveUnlearner):
    def __init__(self, model, tokenizer, device, cfg, ewc=None):
        super().__init__(model, tokenizer, device, cfg, ewc=ewc)
        geo_cfg = getattr(cfg, "geometry", None)
        self.subspace_dim = getattr(geo_cfg, "subspace_dim", 10) if geo_cfg else 10
        self.subspace_samples = getattr(geo_cfg, "subspace_samples", 16) if geo_cfg else 16
        self.keep_ewc = getattr(geo_cfg, "keep_ewc", True) if geo_cfg else True
        self.estimator = RetainSubspaceEstimator(cfg, self.subspace_dim, self.subspace_samples)

    def run_round(self, forget_loader, retain_loader, ewc_lambda: float,
                   epochs: int | None = None, on_step=None):
        import torch
        import itertools

        # Fit the retain subspace once at the start of this round, from
        # the *current* model state -- cheap (subspace_samples backward
        # passes on single examples), done before any forget-set training.
        print(f"[geometry] Fitting retain subspace (k={self.subspace_dim}, "
              f"samples={self.subspace_samples})...")
        self.estimator.fit(self.model, retain_loader, self.device)

        effective_ewc_lambda = ewc_lambda if self.keep_ewc else 0.0

        epochs = epochs or self.cfg.unlearning.epochs
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.unlearning.lr,
        )

        retain_iter = itertools.cycle(retain_loader) if retain_loader is not None else None
        history = []
        global_step = 0
        aborted = False

        self.model.train()
        for epoch in range(epochs):
            if aborted:
                break
            for forget_batch in forget_loader:
                retain_batch = next(retain_iter) if retain_iter else None
                optimizer.zero_grad()
                loss, metrics = self._step_loss(forget_batch, retain_batch, effective_ewc_lambda)
                loss.backward()

                # --- geometry-aware projection ---
                flat = _flatten_grads(self.model)
                projected = self.estimator.project_out(flat)
                _scatter_like_grads(self.model, projected)
                # ----------------------------------

                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.unlearning.grad_clip,
                )
                optimizer.step()
                global_step += 1
                metrics["epoch"] = epoch
                metrics["global_step"] = global_step
                history.append(metrics)

                if on_step is not None and on_step(global_step):
                    metrics["aborted"] = True
                    aborted = True
                    break

        if on_step is not None and not aborted and global_step > 0:
            if on_step(global_step, force=True):
                if history:
                    history[-1]["aborted"] = True

        self.model.eval()
        return history
