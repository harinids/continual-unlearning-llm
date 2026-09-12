"""
Intra-round circuit breaker.
"""
from __future__ import annotations
import math
import itertools


def snapshot_trainable(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, snapshot):
    import torch
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snapshot:
                p.copy_(snapshot[n])


def quick_perplexity(model, loader, device, max_batches: int = 2) -> float:
    import torch

    if loader is None:
        return float("nan")

    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in itertools.islice(loader, max_batches):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels", input_ids).to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(out.loss.item())
    if was_training:
        model.train()

    if not losses:
        return float("nan")
    mean_loss = sum(losses) / len(losses)
    try:
        return math.exp(min(mean_loss, 50))
    except OverflowError:
        return float("inf")


class IntraRoundCircuitBreaker:
    def __init__(self, check_every_n_steps: int = 10, spike_multiplier: float = 2.0,
                 emergency_lambda_multiplier: float = 3.0):
        self.check_every_n_steps = check_every_n_steps
        self.spike_multiplier = spike_multiplier
        self.emergency_lambda_multiplier = emergency_lambda_multiplier
        self.round_start_forget_ppl = None
        self.round_start_retain_ppl = None

    @staticmethod
    def _safe(x):
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return x

    def start_round(self, forget_ppl, retain_ppl):
        self.round_start_forget_ppl = self._safe(forget_ppl)
        self.round_start_retain_ppl = self._safe(retain_ppl)

    def check(self, step: int, forget_ppl, retain_ppl) -> bool:
        if step % self.check_every_n_steps != 0:
            return False
        if self.round_start_forget_ppl is None or self.round_start_retain_ppl is None:
            return False

        forget_ppl = self._safe(forget_ppl)
        retain_ppl = self._safe(retain_ppl)
        if forget_ppl is None or retain_ppl is None:
            return True

        retain_spiked = retain_ppl > self.spike_multiplier * self.round_start_retain_ppl
        forget_spiked_with_retain = (
            retain_spiked
            and forget_ppl > self.spike_multiplier * self.round_start_forget_ppl
        )
        return retain_spiked or forget_spiked_with_retain

    def emergency_lambda(self, lambda_ewc: float) -> float:
        return lambda_ewc * self.emergency_lambda_multiplier
