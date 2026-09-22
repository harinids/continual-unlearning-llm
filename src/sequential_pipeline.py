"""
Sequential/continual unlearning orchestrator for Phase 1.
"""
from __future__ import annotations
import json
import os
import time


class SequentialUnlearningPipeline:
    def __init__(self, cfg, model, tokenizer, dataset, scheduler, enable_safety: bool = True):
        from src.pipeline import UnlearningPipeline
        self.cfg = cfg
        self.scheduler = scheduler
        self.dataset = dataset
        self.enable_safety = enable_safety  # False = plain GA control run, no breaker/PID/decay/hard-stop
        self.pipeline = UnlearningPipeline(cfg, model, tokenizer, dataset)
        self.persistence_matrix: dict = {}
        self.request_results: list = []

    def _make_forget_loader(self, examples):
        from src.pipeline import make_loader
        return make_loader(examples, self.pipeline.tokenizer, self.cfg,
                            self.cfg.unlearning.batch_size, shuffle=True)

    def run(self) -> dict:
        from src.controller.circuit_breaker import snapshot_trainable, restore_trainable, quick_perplexity

        p = self.pipeline
        overall_baseline = p.auditor.audit(
            p.forget_loader, p.retain_loader, p.eval_loader,
            raw_forget_examples=self.dataset.forget,
        )
        print(f"[sequential] Overall baseline (full forget10 pool): {overall_baseline}")

        if self.cfg.ewc.enabled:
            p._consolidate()

        seen_requests = []

        for req in self.scheduler:
            print(f"\n[sequential] ===== Forget Request {req.request_id} ({len(req)} examples) =====")

            req_forget_loader = self._make_forget_loader(req.examples)
            req_baseline = p.auditor.audit(
                req_forget_loader, p.retain_loader, p.eval_loader,
                raw_forget_examples=req.examples,
            )

            report = req_baseline
            max_rounds = self.cfg.controller.max_rounds
            for round_idx in range(max_rounds):
                print(f"[sequential]   -- round {round_idx + 1}/{max_rounds} "
                      f"(lambda_ewc={p.controller.lambda_ewc:.2f}) --")

                breaker_state = {"tripped": False}

                if self.enable_safety:
                    snapshot = snapshot_trainable(p.model)
                    start_forget_ppl = quick_perplexity(p.model, req_forget_loader, p.device)
                    start_retain_ppl = quick_perplexity(p.model, p.retain_loader, p.device)
                    p.breaker.start_round(start_forget_ppl, start_retain_ppl)

                    def on_step(step, _snapshot=snapshot, _state=breaker_state, force=False):
                        if not force and step % p.breaker.check_every_n_steps != 0:
                            return False
                        cur_forget_ppl = quick_perplexity(p.model, req_forget_loader, p.device, max_batches=2)
                        cur_retain_ppl = quick_perplexity(p.model, p.retain_loader, p.device, max_batches=2)
                        if p.breaker.check(step, cur_forget_ppl, cur_retain_ppl):
                            restore_trainable(p.model, _snapshot)
                            old_lambda = p.controller.lambda_ewc
                            p.controller.lambda_ewc = min(
                                p.breaker.emergency_lambda(old_lambda),
                                getattr(self.cfg.ewc, "lambda_max", 1e5),
                            )
                            _state["tripped"] = True
                            print(f"[sequential]   CIRCUIT BREAKER tripped at step {step}")
                            return True
                        return False
                else:
                    on_step = None

                p.unlearner.run_round(
                    req_forget_loader, p.retain_loader,
                    ewc_lambda=(p.controller.lambda_ewc if (self.enable_safety and self.cfg.ewc.enabled) else 0.0),
                    on_step=on_step,
                )

                if self.enable_safety and self.cfg.ewc.enabled:
                    p._consolidate()

                report = p.auditor.audit(
                    req_forget_loader, p.retain_loader, p.eval_loader,
                    raw_forget_examples=req.examples,
                )

                if self.enable_safety:
                    p.controller.step(report, baseline=req_baseline)
                    p.controller.decay_if_stable(breaker_state["tripped"], report=report, baseline=req_baseline)
                    if p.controller.check_hard_stop(report, req_baseline):
                        print(f"[sequential]   HARD STOP: {p.controller._hard_stop_reason}")
                    if p.controller.should_stop():
                        print("[sequential]   Converged for this request. Moving to next.")
                        break

            print(f"[sequential]   Request {req.request_id} final: {report}")

            seen_requests.append((req.request_id, req.examples, req_forget_loader))
            row = {}
            for (prev_id, prev_examples, prev_loader) in seen_requests:
                audit = p.auditor.audit(
                    prev_loader, p.retain_loader, p.eval_loader,
                    raw_forget_examples=prev_examples,
                )
                row[prev_id] = audit
                print(f"[sequential]   persistence[{req.request_id}][{prev_id}]: "
                      f"forget_ppl={audit.get('forget_ppl'):.2f}, "
                      f"forget_exact_match={audit.get('forget_exact_match')}")

            self.persistence_matrix[req.request_id] = row
            self.request_results.append({
                "request_id": req.request_id, "num_examples": len(req),
                "baseline": req_baseline, "final": report,
            })
            self._save_intermediate()

        return {
            "overall_baseline": overall_baseline,
            "request_results": self.request_results,
            "persistence_matrix": self.persistence_matrix,
        }

    def _save_intermediate(self):
        os.makedirs("results/phase1_sequential", exist_ok=True)
        path = f"results/phase1_sequential/persistence_{int(time.time())}.json"
        with open(path, "w") as f:
            json.dump({"request_results": self.request_results,
                       "persistence_matrix": self.persistence_matrix}, f, indent=2, default=str)
        print(f"[sequential] Intermediate results saved to {path}")
