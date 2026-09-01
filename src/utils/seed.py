"""Reproducibility helpers."""
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy, and torch (if available) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preferred: str = "cuda"):
    """Return an available torch device, falling back gracefully."""
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
