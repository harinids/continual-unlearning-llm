"""
RetainSubspaceEstimator.

Estimates a k-dimensional basis for the subspace of trainable-parameter
gradient-space that matters for retained knowledge, from a sample of
retain-set gradients. Uses the standard trick for tall-skinny matrices:
for gradient matrix G (m samples x n params, m << n for LoRA), the top-k
right singular vectors of G (i.e. directions in param space) are obtained
via eigendecomposition of the small m x m Gram matrix G @ G.T, avoiding
ever forming an n x n or n x k SVD directly on param-space.
"""
from __future__ import annotations
import torch


def _flatten_grads(model) -> torch.Tensor:
    """Flatten all trainable params' .grad into one 1D vector, in a fixed order."""
    parts = []
    for p in model.parameters():
        if p.requires_grad:
            g = p.grad
            parts.append((g if g is not None else torch.zeros_like(p)).reshape(-1))
    return torch.cat(parts)


def _scatter_like_grads(model, flat: torch.Tensor):
    """Write a flat vector back into each trainable param's .grad, same order as _flatten_grads."""
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            n = p.numel()
            if p.grad is None:
                p.grad = flat[offset:offset + n].view_as(p).clone()
            else:
                p.grad.copy_(flat[offset:offset + n].view_as(p))
            offset += n


class RetainSubspaceEstimator:
    def __init__(self, cfg, subspace_dim: int = 10, num_samples: int = 16):
        self.cfg = cfg
        self.subspace_dim = subspace_dim
        self.num_samples = num_samples
        self.basis: torch.Tensor | None = None  # shape [n_params, k], orthonormal columns

    def fit(self, model, retain_loader, device) -> torch.Tensor:
        """
        Collects up to self.num_samples per-example retain-loss gradients
        (one backward pass per example, param grads flattened + zeroed
        between samples), stacks into G [m, n], and returns the top-k
        right singular vectors as an [n, k] orthonormal basis.
        """
        model.eval()
        grads = []
        count = 0
        for batch in retain_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels", input_ids).to(device)

            # one example at a time within the batch, for per-example gradients
            for i in range(input_ids.size(0)):
                if count >= self.num_samples:
                    break
                model.zero_grad(set_to_none=True)
                out = model(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1],
                    labels=labels[i:i+1],
                )
                out.loss.backward()
                grads.append(_flatten_grads(model).detach().clone())
                count += 1
            if count >= self.num_samples:
                break
        model.zero_grad(set_to_none=True)
        model.train()

        if len(grads) < 2:
            raise RuntimeError(f"RetainSubspaceEstimator got only {len(grads)} gradient samples — need >=2")

        G = torch.stack(grads, dim=0)  # [m, n]
        k = min(self.subspace_dim, G.shape[0] - 1)

        gram = G @ G.T  # [m, m], cheap
        eigvals, eigvecs = torch.linalg.eigh(gram)  # ascending order
        top_idx = torch.argsort(eigvals, descending=True)[:k]
        top_eigvals = eigvals[top_idx].clamp(min=1e-12)
        top_eigvecs = eigvecs[:, top_idx]  # [m, k]

        # v_i = G.T @ u_i / sqrt(eigval_i)  -> orthonormal directions in param space
        V = (G.T @ top_eigvecs) / top_eigvals.sqrt().unsqueeze(0)  # [n, k]
        V = V / V.norm(dim=0, keepdim=True).clamp(min=1e-12)  # re-normalize for numerical safety

        self.basis = V
        return V

    def project_out(self, flat_grad: torch.Tensor) -> torch.Tensor:
        """Remove the component of flat_grad lying in the retain subspace."""
        if self.basis is None:
            return flat_grad
        coeffs = self.basis.T @ flat_grad  # [k]
        return flat_grad - self.basis @ coeffs
