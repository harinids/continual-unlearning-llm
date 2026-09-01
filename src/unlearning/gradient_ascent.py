"""
Selective unlearning via gradient ascent on the forget set, regularized
by an EWC penalty (memory consolidation) so retained knowledge is
protected while forget-set knowledge is actively erased.

Supported methods:
  - "gradient_ascent": maximize forget-set loss (i.e. minimize -loss)
  - "gradient_difference": maximize forget-set loss while simultaneously
    minimizing retain-set loss in the same step (Liu et al., 2022 style)
  - "npo": Negative Preference Optimization — a bounded alternative to
    plain gradient ascent that avoids the unboundedness/divergence that
    vanilla GA can suffer from (Zhang et al., 2024)
"""
from __future__ import annotations


def _lm_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch.get("labels", input_ids).to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return out.loss


class SelectiveUnlearner:
    def __init__(self, model, tokenizer, device, cfg, ewc=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = cfg
        self.ewc = ewc  # src.consolidation.ewc.EWC instance, or None
        self.method = cfg.unlearning.method
        self.forget_weight = cfg.unlearning.forget_loss_weight
        self.retain_weight = cfg.unlearning.retain_loss_weight

    def _step_loss(self, forget_batch, retain_batch, ewc_lambda: float):
        import torch

        forget_loss = _lm_loss(self.model, forget_batch, self.device)

        if self.method == "gradient_ascent":
            task_loss = -self.forget_weight * forget_loss

        elif self.method == "gradient_difference":
            retain_loss = _lm_loss(self.model, retain_batch, self.device) if retain_batch else torch.tensor(0.0, device=self.device)
            task_loss = -self.forget_weight * forget_loss + self.retain_weight * retain_loss

        elif self.method == "npo":
            # NPO: bounded negative-preference loss, beta controls steepness.
            beta = 0.1
            task_loss = (2.0 / beta) * torch.nn.functional.softplus(beta * forget_loss)
            # NPO's objective already grows as forget_loss shrinks below the
            # reference; here we approximate with softplus of the loss itself
            # since we don't retain a frozen reference-model pass in this
            # lightweight implementation. Swap in a true reference-model
            # log-ratio for the full NPO formulation.
        else:
            raise ValueError(f"Unknown unlearning method: {self.method}")

        ewc_loss = torch.tensor(0.0, device=self.device)
        if self.ewc is not None and ewc_lambda > 0:
            ewc_loss = ewc_lambda * self.ewc.penalty(self.model)

        total = task_loss + ewc_loss
        return total, {
            "forget_loss": forget_loss.item(),
            "task_loss": task_loss.item(),
            "ewc_loss": float(ewc_loss.item() if hasattr(ewc_loss, "item") else ewc_loss),
        }

    def run_round(self, forget_loader, retain_loader, ewc_lambda: float, epochs: int | None = None):
        """Run one round of unlearning (a handful of epochs over the forget set)."""
        import torch
        import itertools

        epochs = epochs or self.cfg.unlearning.epochs
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.unlearning.lr,
        )

        retain_iter = itertools.cycle(retain_loader) if retain_loader is not None else None
        history = []

        self.model.train()
        for epoch in range(epochs):
            for forget_batch in forget_loader:
                retain_batch = next(retain_iter) if retain_iter else None
                optimizer.zero_grad()
                loss, metrics = self._step_loss(forget_batch, retain_batch, ewc_lambda)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.unlearning.grad_clip,
                )
                optimizer.step()
                metrics["epoch"] = epoch
                history.append(metrics)

        self.model.eval()
        return history
