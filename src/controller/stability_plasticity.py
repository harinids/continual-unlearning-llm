"""
Stability-Plasticity Controller (PID + lambda cap).
"""
from __future__ import annotations
import math


class StabilityPlasticityController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lambda_ewc = cfg.ewc.lambda_init
        self.plasticity = cfg.controller.plasticity_init
        self.stability = cfg.controller.stability_init
        self.kp = cfg.controller.kp
        self.ki = cfg.controller.ki
        self.kd = getattr(cfg.controller, "kd", 0.0)
        self._integral_error = 0.0
        self._prev_retain_degradation = 0.0
        self.history: list[dict] = []
        self._best_score = float("-inf")
        self._rounds_no_improve = 0

    def _score(self, report: dict, baseline: dict) -> float:
        forget_leak = report.get("forget_exact_match", report.get("forget_ppl", 0.0))
        forget_term = -(forget_leak if forget_leak is not None else 0.0)

        retain_ppl = report.get("retain_ppl")
        baseline_retain_ppl = baseline.get("retain_ppl") if baseline else None
        stability_term = 0.0
        if retain_ppl is not None and baseline_retain_ppl:
            stability_term = -max(0.0, (retain_ppl - baseline_retain_ppl) / max(baseline_retain_ppl, 1e-6))

        return forget_term + stability_term

    def step(self, report: dict, baseline: dict | None = None) -> dict:
        forget_threshold = self.cfg.auditor.forget_threshold
        retain_threshold = self.cfg.auditor.retain_threshold

        forget_metric = report.get("forget_exact_match")
        forget_metric_valid = forget_metric is not None and not (
            isinstance(forget_metric, float) and math.isnan(forget_metric)
        )
        forget_error = (forget_metric - forget_threshold) if forget_metric_valid else 0.0

        retain_ppl = report.get("retain_ppl")
        baseline_ppl = (baseline or {}).get("retain_ppl")
        retain_degradation = 0.0
        if retain_ppl is not None and baseline_ppl:
            retain_degradation = max(0.0, (retain_ppl / baseline_ppl) - 1.0)

        self._integral_error += forget_error

        delta = -self.kp * forget_error - self.ki * self._integral_error
        delta = max(-0.5, min(0.5, delta))
        stability_penalty_budget = 1.0 - retain_threshold
        retain_degradation = min(retain_degradation, 5.0)
        if retain_degradation > stability_penalty_budget:
            delta += min(0.5, self.kp * (retain_degradation - stability_penalty_budget))

        d_retain = retain_degradation - self._prev_retain_degradation
        self._prev_retain_degradation = retain_degradation
        if d_retain > 0:
            delta += min(0.5, self.kd * d_retain)

        self.lambda_ewc = max(0.0, self.lambda_ewc + delta * self.lambda_ewc)
        self.lambda_ewc = min(self.lambda_ewc, getattr(self.cfg.ewc, "lambda_max", 1e5))
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

    def decay_if_stable(self, breaker_tripped: bool, decay_factor: float = 0.7,
                         stable_rounds_needed: int = 2):
        """
        If the circuit breaker hasn't tripped for `stable_rounds_needed`
        consecutive rounds, ease lambda_ewc back down. Without this, lambda
        only ever escalates (every trip multiplies it up, nothing brings it
        back down), so once a rough patch pushes lambda to lambda_max the
        controller stays pinned there permanently -- plasticity never
        recovers even after the model has clearly stabilized, which is
        exactly what produced zero net forgetting across all 12 rounds of
        the Qwen2.5 stress test (every round tripped, so this never
        triggered there -- this addresses the case where trips eventually
        stop but lambda never comes back down to let unlearning resume).
        """
        if not hasattr(self, "_stable_round_count"):
            self._stable_round_count = 0

        if breaker_tripped:
            self._stable_round_count = 0
        else:
            self._stable_round_count += 1

        if self._stable_round_count >= stable_rounds_needed:
            self.lambda_ewc = max(self.cfg.ewc.lambda_init * 0.1, self.lambda_ewc * decay_factor)
            self._stable_round_count = 0  # reset so decay is gradual, not immediate re-decay next round

    def should_stop(self) -> bool:
        return self._rounds_no_improve >= self.cfg.controller.patience
