"""
End-to-end pipeline.
"""
from __future__ import annotations
import json
import os
import time


class DictDataset:
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
        from src.controller.circuit_breaker import IntraRoundCircuitBreaker
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

        # Fixed, unshuffled subsets dedicated to the circuit breaker's probes.
        # Using the same examples every call (rather than sampling from the
        # shuffled main loaders) removes estimator variance as a source of
        # missed spikes -- a random 2-batch sample can read "fine" purely by
        # chance even when the true full-set perplexity has already blown up.
        breaker_probe_size = 20
        self.breaker_forget_loader = make_loader(
            dataset.forget[:breaker_probe_size], tokenizer, cfg, cfg.unlearning.batch_size, shuffle=False,
        )
        self.breaker_retain_loader = make_loader(
            dataset.retain[:breaker_probe_size], tokenizer, cfg, cfg.unlearning.batch_size, shuffle=False,
        )

        self.auditor = KnowledgeAuditor(model, tokenizer, self.device, cfg)
        self.controller = StabilityPlasticityController(cfg)

        breaker_cfg = getattr(cfg, "breaker", None)
        self.breaker = IntraRoundCircuitBreaker(
            check_every_n_steps=getattr(breaker_cfg, "check_every_n_steps", 10) if breaker_cfg else 10,
            spike_multiplier=getattr(breaker_cfg, "spike_multiplier", 2.0) if breaker_cfg else 2.0,
            emergency_lambda_multiplier=getattr(breaker_cfg, "emergency_lambda_multiplier", 3.0) if breaker_cfg else 3.0,
        )

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
        from src.controller.circuit_breaker import snapshot_trainable, restore_trainable, quick_perplexity

        results = {"rounds": []}

        print("[pipeline] Running baseline audit...")
        baseline_report = self.auditor.audit(
            self.forget_loader, self.retain_loader, self.eval_loader,
            raw_forget_examples=self.dataset.forget,
        )
        results["baseline"] = baseline_report
        print(f"[pipeline] Baseline: {baseline_report}")

        if self.cfg.ewc.enabled:
            print("[pipeline] Fitting initial Fisher Information on retain set...")
            self._consolidate()

        max_rounds = self.cfg.controller.max_rounds
        for round_idx in range(max_rounds):
            print(f"\n[pipeline] === Round {round_idx + 1}/{max_rounds} "
                  f"(lambda_ewc={self.controller.lambda_ewc:.2f}) ===")

            snapshot = snapshot_trainable(self.model)
            start_forget_ppl = quick_perplexity(self.model, self.breaker_forget_loader, self.device, max_batches=5)
            start_retain_ppl = quick_perplexity(self.model, self.breaker_retain_loader, self.device, max_batches=5)
            self.breaker.start_round(start_forget_ppl, start_retain_ppl)
            breaker_state = {"tripped": False}

            def on_step(step, _snapshot=snapshot, _state=breaker_state, force=False):
                if not force and step % self.breaker.check_every_n_steps != 0:
                    return False
                cur_forget_ppl = quick_perplexity(self.model, self.breaker_forget_loader, self.device, max_batches=5)
                cur_retain_ppl = quick_perplexity(self.model, self.breaker_retain_loader, self.device, max_batches=5)
                if self.breaker.check(step, cur_forget_ppl, cur_retain_ppl):
                    restore_trainable(self.model, _snapshot)
                    old_lambda = self.controller.lambda_ewc
                    self.controller.lambda_ewc = min(
                        self.breaker.emergency_lambda(old_lambda),
                        getattr(self.cfg.ewc, "lambda_max", 1e5),
                    )
                    _state["tripped"] = True
                    print(f"[pipeline] CIRCUIT BREAKER tripped at step {step}: "
                          f"forget_ppl={cur_forget_ppl:.2f} (start={start_forget_ppl:.2f}), "
                          f"retain_ppl={cur_retain_ppl:.2f} (start={start_retain_ppl:.2f}) "
                          f"-> reverted weights, lambda_ewc {old_lambda:.2f} -> {self.controller.lambda_ewc:.2f}")
                    return True
                return False

            history = self.unlearner.run_round(
                self.forget_loader, self.retain_loader,
                ewc_lambda=self.controller.lambda_ewc if self.cfg.ewc.enabled else 0.0,
                on_step=on_step,
            )

            if self.cfg.ewc.enabled:
                self._consolidate()

            report = self.auditor.audit(
                self.forget_loader, self.retain_loader, self.eval_loader,
                raw_forget_examples=self.dataset.forget,
            )
            print(f"[pipeline] Round {round_idx + 1} audit: {report}")

            control = self.controller.step(report, baseline=baseline_report)
            self.controller.decay_if_stable(breaker_state["tripped"])
            print(f"[pipeline] Controller: {control} (lambda after decay check: {self.controller.lambda_ewc:.2f})")

            results["rounds"].append({
                "round": round_idx + 1,
                "train_history": history,
                "audit": report,
                "control": control,
                "breaker_tripped": breaker_state["tripped"],
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
