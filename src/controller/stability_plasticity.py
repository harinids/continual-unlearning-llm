"""
Stability-Plasticity Controller.

Each round, the Knowledge Auditor reports how much forget-set knowledge
remains (plasticity concern: are we changing the model enough?) and how
much retain-set/eval performance has degraded relative to baseline
(stability concern: are we changing it too much / too broadly?).

This controller adjusts the EWC penalty strength (lambda) round-to-round
using simple PI (proportional-integral) feedback:

  - If forget-set leakage is still high  -> lower lambda (more plasticity,
    let gradient ascent move parameters more freely to finish erasing).
  - If retain/eval performance has dropped too far from baseline -> raise
    lambda (more stability, clamp parameters closer to the consolidated
    "safe" values).

This keeps the unlearning/consolidation trade-off from being a single
fixed hyperparameter that has to be hand-tuned per dataset/model.
"""
from __future__ import annotations


class StabilityPlasticityController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lambda_ewc = cfg.ewc.lambda_init
        self.plasticity = cfg.controller.plasticity_init
        self.stability = cfg.controller.stability_init
        self.kp = cfg.controller.kp
        self.ki = cfg.controller.ki
        self._integral_error = 0.0
        self.history: list[dict] = []
        self._best_score = float("-inf")
        self._rounds_no_improve = 0

    def _score(self, report: dict, baseline: dict) -> float:
        """
        Composite score to track for early stopping: higher is better.
        Rewards low forget-set retention and low degradation of retain/eval.
        """
        forget_leak = report.get("forget_exact_match", report.get("forget_ppl", 0.0))
        # Normalize forget leakage: lower is better for unlearning, so invert.
        forget_term = -(forget_leak if forget_leak is not None else 0.0)

        retain_ppl = report.get("retain_ppl")
        baseline_retain_ppl = baseline.get("retain_ppl") if baseline else None
        stability_term = 0.0
        if retain_ppl is not None and baseline_retain_ppl:
            # Penalize retain perplexity inflation relative to baseline.
            stability_term = -max(0.0, (retain_ppl - baseline_retain_ppl) / max(baseline_retain_ppl, 1e-6))

        return forget_term + stability_term

    def step(self, report: dict, baseline: dict | None = None) -> dict:
        """
        Consume this round's audit report and baseline (pre-unlearning)
        report, update lambda_ewc, and return the updated control signals.
        """
        forget_threshold = self.cfg.auditor.forget_threshold
        retain_threshold = self.cfg.auditor.retain_threshold

        forget_metric = report.get("forget_exact_match")
        forget_error = (forget_metric - forget_threshold) if forget_metric is not None else 0.0
        # forget_error > 0 => still leaking too much knowledge => need more plasticity => lower lambda

        retain_ppl = report.get("retain_ppl")
        baseline_ppl = (baseline or {}).get("retain_ppl")
        retain_degradation = 0.0
        if retain_ppl is not None and baseline_ppl:
            retain_degradation = max(0.0, (retain_ppl / baseline_ppl) - 1.0)  # fractional inflation
        # retain_degradation > (1 - retain_threshold) => losing too much stability => need more lambda

        self._integral_error += forget_error

        # PI update: push lambda down when forget leakage is high,
        # push lambda up when retain degradation exceeds tolerance.
        # Both terms are clipped to a bounded fractional step per round —
        # without this, an early large retain_degradation spike causes an
        # unbounded multiplicative explosion in lambda_ewc across rounds
        # (observed empirically: 1000 -> 1.7e32 within 3 rounds pre-fix).
        delta = -self.kp * forget_error - self.ki * self._integral_error
        delta = max(-0.5, min(0.5, delta))
        stability_penalty_budget = 1.0 - retain_threshold
        retain_degradation = min(retain_degradation, 5.0)
        if retain_degradation > stability_penalty_budget:
            delta += min(0.5, self.kp * (retain_degradation - stability_penalty_budget))

        self.lambda_ewc = max(0.0, self.lambda_ewc + delta * self.lambda_ewc)
        self.plasticity = 1.0 / (1.0 + self.lambda_ewc / max(self.cfg.ewc.lambda_init, 1e-6))
        self.stability = 1.0 - self.plasticity

        score = self._score(report, baseline)
        improved = score > self._best_score
        self._best_score = max(self._best_score, score)
        self._rounds_no_improve = 0 if improved else self._rounds_no_improve + 1

        record = {
            "lambda_ewc": self.lambda_ewc,
            "plasticity": self.plasticity,
            "stability": self.stability,
            "score": score,
            "improved": improved,
        }
        self.history.append(record)
        return record

    def should_stop(self) -> bool:
        return self._rounds_no_improve >= self.cfg.controller.patience
