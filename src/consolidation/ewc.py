"""
Elastic Weight Consolidation (EWC) for memory consolidation.

After the model has learned/retained a body of knowledge, we estimate a
diagonal Fisher Information Matrix (FIM) over the trainable parameters
using the retain set. During subsequent unlearning steps, an EWC penalty
    L_ewc = (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2
discourages moving parameters that were important for retained knowledge,
which is what keeps gradient-ascent unlearning from causing catastrophic
forgetting of everything else.

"online": if True, Fisher estimates accumulate (running average) across
consolidation rounds rather than being overwritten, per Online EWC
(Schwarz et al., 2018).
"""
from __future__ import annotations
import copy


class EWC:
    def __init__(self, model, dataloader, device, online: bool = True, gamma: float = 1.0):
        self.online = online
        self.gamma = gamma
        self.device = device
        self.fisher: dict = {}
        self.opt_params: dict = {}
        self.update(model, dataloader)

    def _named_trainable(self, model):
        return [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def update(self, model, dataloader):
        """(Re-)estimate the Fisher Information from `dataloader` and snapshot params."""
        import torch

        model.eval()
        new_fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self._named_trainable(model)}

        n_batches = 0
        for batch in dataloader:
            model.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            for n, p in self._named_trainable(model):
                if p.grad is not None:
                    new_fisher[n] += p.grad.detach() ** 2
            n_batches += 1

        n_batches = max(n_batches, 1)
        for n in new_fisher:
            new_fisher[n] /= n_batches

        if self.online and self.fisher:
            for n in new_fisher:
                prev = self.fisher.get(n, torch.zeros_like(new_fisher[n]))
                new_fisher[n] = self.gamma * prev + new_fisher[n]

        self.fisher = new_fisher
        self.opt_params = {n: p.detach().clone() for n, p in self._named_trainable(model)}
        model.zero_grad()

    def penalty(self, model):
        """Compute the EWC quadratic penalty against the stored optimal parameters."""
        loss = 0.0
        for n, p in self._named_trainable(model):
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.opt_params[n]) ** 2).sum()
        return loss

    def fisher_norm(self) -> float:
        """Summary statistic: total Fisher 'mass', useful for logging/controller feedback."""
        total = 0.0
        for f in self.fisher.values():
            total += f.sum().item()
        return total
