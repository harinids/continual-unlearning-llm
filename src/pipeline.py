"""
End-to-end pipeline, matching the architecture in README.md:

    Input (Forget / Retain / Eval Split)
            |
            v
    Knowledge Auditor        <- baseline audit
            |
            v
    Selective Unlearning      <- gradient ascent on forget set
    (Gradient Ascent)
            |
            v
    Memory Consolidation      <- EWC regularization
    (EWC)
            |
            v
    Stability-Plasticity      <- adjusts lambda_ewc from auditor feedback,
    Controller                   loops back for further rounds if needed
"""
from __future__ import annotations
import json
import os
import time


class DictDataset:
    """Wraps a list[Example] into a torch Dataset of tokenized tensors."""

    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex.text if hasattr(ex, "text") else ex["text"]
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        item["labels"][item["attention_mask"] == 0] = -100
        return item


def make_loader(examples, tokenizer, cfg, batch_size, shuffle=True):
    from torch.utils.data import DataLoader

    if not examples:
        return None
    ds = DictDataset(examples, tokenizer, cfg.model.max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class UnlearningPipeline:
    def __init__(self, cfg, model, tokenizer, dataset):
        from src.utils.seed import get_device
        from src.auditor.knowledge_auditor import KnowledgeAuditor
        from src.controller.stability_plasticity import StabilityPlasticityController
        from src.unlearning.gradient_ascent import SelectiveUnlearner

        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = get_device(cfg.device)
        self.model.to(self.device)

        self.forget_loader = make_loader(dataset.forget, tokenizer, cfg, cfg.unlearning.batch_size, shuffle=True)
        self.retain_loader = make_loader(dataset.retain, tokenizer, cfg, cfg.unlearning.batch_size, shuffle=True)
        self.eval_loader = make_loader(dataset.eval, tokenizer, cfg, cfg.auditor.eval_batch_size, shuffle=False)
        self.fisher_loader = make_loader(
            dataset.retain[: cfg.ewc.fisher_num_samples], tokenizer, cfg, cfg.ewc.fisher_batch_size, shuffle=True,
        )

        self.auditor = KnowledgeAuditor(model, tokenizer, self.device, cfg)
        self.controller = StabilityPlasticityController(cfg)
        self.ewc = None
        self.unlearner = SelectiveUnlearner(model, tokenizer, self.device, cfg, ewc=None)

        os.makedirs(cfg.logging.output_dir, exist_ok=True)
        os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)

    def _consolidate(self):
        from src.consolidation.ewc import EWC

        if self.ewc is None:
            self.ewc = EWC(self.model, self.fisher_loader, self.device, online=self.cfg.ewc.online)
        else:
            self.ewc.update(self.model, self.fisher_loader)
        self.unlearner.ewc = self.ewc

    def run(self) -> dict:
        results = {"rounds": []}

        # 1. Knowledge Auditor: baseline, before any unlearning.
        print("[pipeline] Running baseline audit...")
        baseline_report = self.auditor.audit(
            self.forget_loader, self.retain_loader, self.eval_loader,
            raw_forget_examples=self.dataset.forget,
        )
        results["baseline"] = baseline_report
        print(f"[pipeline] Baseline: {baseline_report}")

        # 2. Memory Consolidation (initial EWC fit on retain set) — this
        #    happens *before* unlearning so we have something to protect.
        if self.cfg.ewc.enabled:
            print("[pipeline] Fitting initial Fisher Information on retain set...")
            self._consolidate()

        max_rounds = self.cfg.controller.max_rounds
        for round_idx in range(max_rounds):
            print(f"\n[pipeline] === Round {round_idx + 1}/{max_rounds} "
                  f"(lambda_ewc={self.controller.lambda_ewc:.2f}) ===")

            # 3. Selective Unlearning (gradient ascent on forget set,
            #    regularized by the current EWC penalty).
            history = self.unlearner.run_round(
                self.forget_loader, self.retain_loader,
                ewc_lambda=self.controller.lambda_ewc if self.cfg.ewc.enabled else 0.0,
            )

            # 4. Memory Consolidation refresh (Online EWC accumulates
            #    Fisher information after each round of change).
            if self.cfg.ewc.enabled:
                self._consolidate()

            # Re-audit to see the effect of this round.
            report = self.auditor.audit(
                self.forget_loader, self.retain_loader, self.eval_loader,
                raw_forget_examples=self.dataset.forget,
            )
            print(f"[pipeline] Round {round_idx + 1} audit: {report}")

            # 5. Stability-Plasticity Controller: adjust lambda_ewc for
            #    next round based on the auditor's feedback.
            control = self.controller.step(report, baseline=baseline_report)
            print(f"[pipeline] Controller: {control}")

            results["rounds"].append({
                "round": round_idx + 1,
                "train_history": history,
                "audit": report,
                "control": control,
            })

            if self.cfg.logging.save_every_round:
                self._save_checkpoint(round_idx + 1)

            if self.controller.should_stop():
                print("[pipeline] Controller signaled convergence (no improvement). Stopping early.")
                break

        results["final_audit"] = results["rounds"][-1]["audit"] if results["rounds"] else baseline_report
        self._save_results(results)
        return results

    def _save_checkpoint(self, round_idx: int):
        path = os.path.join(self.cfg.logging.checkpoint_dir, f"round_{round_idx}")
        try:
            self.model.save_pretrained(path)
        except Exception as e:
            print(f"[pipeline] Warning: could not save checkpoint ({e})")

    def _save_results(self, results: dict):
        path = os.path.join(self.cfg.logging.output_dir, f"results_{int(time.time())}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[pipeline] Results saved to {path}")
