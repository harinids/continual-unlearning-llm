"""
Selective unlearning via gradient ascent on the forget set, regularized
by an EWC penalty, with an on_step hook for intra-round abort. Also
forces one extra check at the very end of the round (force=True) so
degradation completing in the last few steps between regular checks
cannot slip through unnoticed.
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
        self.ewc = ewc
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
            beta = 0.1
            task_loss = (2.0 / beta) * torch.nn.functional.softplus(beta * forget_loss)
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

    def run_round(self, forget_loader, retain_loader, ewc_lambda: float,
                   epochs: int | None = None, on_step=None):
        import torch
        import itertools

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
                loss, metrics = self._step_loss(forget_batch, retain_batch, ewc_lambda)
                loss.backward()
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

        # Force one final check at the true end of the round, regardless of
        # check_every_n_steps alignment -- this is what catches degradation
        # that completed in the last few steps between regular checkpoints
        # (previously this could slip through and get permanently baked in
        # by the next EWC/Fisher consolidation step).
        if on_step is not None and not aborted and global_step > 0:
            if on_step(global_step, force=True):
                if history:
                    history[-1]["aborted"] = True
                aborted = True

        self.model.eval()
        return history
