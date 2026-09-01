"""
Dataset loading and splitting for the unlearning pipeline.

Supports:
  - TOFU (fictional-author QA unlearning benchmark)
  - HateXplain (hate speech + rationale annotations)
  - A synthetic fallback generator, used when the HF Hub is unreachable
    (e.g. offline / sandboxed environments) so the rest of the pipeline
    can still be developed and unit-tested end to end.

Each loader returns a dict with keys "forget", "retain", "eval", where
every value is a list of {"text": str, "label": Optional[str]} records
already formatted for causal-LM fine-tuning / unlearning.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class Example:
    text: str
    label: Optional[str] = None
    meta: Optional[dict] = None


class UnlearningDataset:
    """Container for retain/forget/eval splits."""

    def __init__(self, forget: list[Example], retain: list[Example], eval_set: list[Example]):
        self.forget = forget
        self.retain = retain
        self.eval = eval_set

    def __repr__(self):
        return (
            f"UnlearningDataset(forget={len(self.forget)}, "
            f"retain={len(self.retain)}, eval={len(self.eval)})"
        )


def _truncate(items: list, n: Optional[int]) -> list:
    if n is None:
        return items
    return items[:n]


def load_tofu(cfg) -> UnlearningDataset:
    """
    Load the TOFU benchmark (locuslab/TOFU on the HF Hub).

    TOFU is organized as question/answer pairs about fictional authors.
    The "forgetXX" splits mark the subset that should be unlearned; the
    complementary "retainXX" split is the remainder of the full author set.
    """
    from datasets import load_dataset

    forget_split = cfg.dataset.forget_split
    retain_split = cfg.dataset.retain_split

    forget_ds = load_dataset("locuslab/TOFU", forget_split, cache_dir=cfg.dataset.cache_dir)["train"]
    retain_ds = load_dataset("locuslab/TOFU", retain_split, cache_dir=cfg.dataset.cache_dir)["train"]
    # TOFU's held-out "real world"/"world facts" split works well as an eval
    # sanity-check set that should be unaffected by unlearning.
    try:
        eval_ds = load_dataset("locuslab/TOFU", "real_authors_perturbed", cache_dir=cfg.dataset.cache_dir)["train"]
    except Exception:
        eval_ds = retain_ds

    def fmt(row):
        q = row.get("question", "").strip()
        a = row.get("answer", "").strip()
        return Example(text=f"Question: {q}\nAnswer: {a}", meta={"question": q, "answer": a})

    forget = [fmt(r) for r in forget_ds]
    retain = [fmt(r) for r in retain_ds]
    evalset = [fmt(r) for r in eval_ds]

    return UnlearningDataset(
        forget=_truncate(forget, cfg.dataset.num_forget_samples),
        retain=_truncate(retain, cfg.dataset.num_retain_samples),
        eval_set=_truncate(evalset, cfg.dataset.num_eval_samples),
    )


def load_hatexplain(cfg) -> UnlearningDataset:
    """
    Load HateXplain and construct forget/retain/eval splits.

    Framing: examples labeled "hatespeech" with high annotator agreement
    are treated as the "forget" set (memorized toxic spans a deployed
    model should not reproduce). Neutral/offensive-but-non-hateful
    examples form the "retain" set, so we can check that general language
    understanding survives unlearning. A held-out slice forms "eval".
    """
    from datasets import load_dataset

    ds = load_dataset("Hate-speech-CNERG/hatexplain", revision="refs/convert/parquet", cache_dir=cfg.dataset.cache_dir)["train"]

    label_names = ds.features["annotators"]  # nested; label ints resolved below
    id2label = {0: "hatespeech", 1: "normal", 2: "offensive"}

    def majority_label(row):
        labels = row["annotators"]["label"]
        return max(set(labels), key=labels.count)

    def text_of(row):
        return " ".join(row["post_tokens"])

    forget, retain, evalset = [], [], []
    for row in ds:
        lbl_id = majority_label(row)
        lbl = id2label.get(lbl_id, "unknown")
        ex = Example(text=text_of(row), label=lbl)
        if lbl == "hatespeech":
            forget.append(ex)
        elif lbl == "normal":
            retain.append(ex)
        else:
            evalset.append(ex)

    random.Random(cfg.seed).shuffle(forget)
    random.Random(cfg.seed).shuffle(retain)
    random.Random(cfg.seed).shuffle(evalset)

    return UnlearningDataset(
        forget=_truncate(forget, cfg.dataset.num_forget_samples),
        retain=_truncate(retain, cfg.dataset.num_retain_samples),
        eval_set=_truncate(evalset, cfg.dataset.num_eval_samples),
    )


def load_synthetic(cfg) -> UnlearningDataset:
    """
    Deterministic synthetic dataset requiring no network access.

    Useful for: CI, unit tests, and offline development of the pipeline
    logic (unlearning / EWC / auditor / controller) before pointing it
    at the real TOFU or HateXplain data.
    """
    rng = random.Random(cfg.seed)
    fake_authors = [f"Author_{i}" for i in range(20)]
    fake_facts = ["was born in Kelvora", "wrote 'The Glass Orchard'", "won the Ilmen Prize",
                  "lives in Nord Haven", "studied under Master Voss"]

    def make(n, tag):
        out = []
        for i in range(n):
            a = rng.choice(fake_authors)
            f = rng.choice(fake_facts)
            out.append(Example(text=f"Question: What do we know about {a}?\nAnswer: {a} {f}. [{tag}#{i}]"))
        return out

    n_forget = cfg.dataset.num_forget_samples or 40
    n_retain = cfg.dataset.num_retain_samples or 120
    n_eval = cfg.dataset.num_eval_samples or 40

    return UnlearningDataset(
        forget=make(n_forget, "forget"),
        retain=make(n_retain, "retain"),
        eval_set=make(n_eval, "eval"),
    )


LOADERS = {
    "tofu": load_tofu,
    "hatexplain": load_hatexplain,
    "synthetic": load_synthetic,
}


def load_unlearning_dataset(cfg) -> UnlearningDataset:
    name = cfg.dataset.name.lower()
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(LOADERS)}.")
    try:
        return LOADERS[name](cfg)
    except Exception as e:
        # Network-restricted / offline environments (no Hub access) fall
        # back to synthetic data so development isn't blocked.
        print(f"[data] Could not load '{name}' from the Hub ({e}). "
              f"Falling back to synthetic data.")
        return load_synthetic(cfg)
