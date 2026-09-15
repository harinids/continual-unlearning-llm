"""
Stability-Plasticity Controller (PID + lambda cap + baseline-aware decay
+ audit-level baseline-referenced hard-stop).
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
        self._stable_round_count = 0
        self.history: list[dict] = []
        self._best_score = float("-inf")
        self._rounds_no_improve = 0

        # --- audit-level baseline-referenced hard-stop ---
        self.hard_drift_threshold = getattr(cfg.controller, "hard_drift_threshold", 3.0)
        self.hard_stop_action = getattr(cfg.controller, "hard_stop_action", "abort")  # "abort" | "emergency_lambda"
        self._hard_stop_triggered = False
        self._hard_stop_reason = None

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

    def decay_if_stable(self, breaker_tripped: bool, report: dict | None = None,
                         baseline: dict | None = None, decay_factor: float = 0.7,
                         stable_rounds_needed: int = 2, max_baseline_drift: float = 0.5):
        """
        Ease lambda_ewc down only if BOTH (a) breaker hasn't tripped for
        `stable_rounds_needed` consecutive rounds, AND (b) this round's
        audit is still within `max_baseline_drift` fractional degradation
        of the ORIGINAL baseline (not just the previous round).

        Condition (a) alone was the bug in the lr=2e-6 Qwen2.5-0.5B run:
        it eased lambda during rounds that were "stable" only by the
        breaker's local, round-start-relative frame of reference, while
        forget/retain/eval ppl kept climbing round-over-round against the
        true baseline. Condition (b) closes that gap. Note this still
        does not prevent lambda from being stuck at max while the model
        keeps getting worse once it's already capped -- that residual
        gap is covered by check_hard_stop() below.
        """
        within_baseline_budget = True
        if report is not None and baseline is not None:
            for key in ("forget_ppl", "retain_ppl", "eval_ppl"):
                cur = report.get(key)
                base = baseline.get(key)
                if cur is not None and base and base > 0:
                    drift = (cur - base) / base
                    if drift > max_baseline_drift:
                        within_baseline_budget = False
                        break

        if breaker_tripped or not within_baseline_budget:
            self._stable_round_count = 0
        else:
            self._stable_round_count += 1

        if self._stable_round_count >= stable_rounds_needed:
            self.lambda_ewc = max(self.cfg.ewc.lambda_init * 0.1, self.lambda_ewc * decay_factor)
            self._stable_round_count = 0

    def check_hard_stop(self, report: dict, baseline: dict | None) -> bool:
        """
        Audit-level, ORIGINAL-baseline-referenced safety check. Independent
        of should_stop()/patience and of the intra-round breaker.

        Only retain_ppl / eval_ppl drift counts as a genuine failure signal
        -- forget_ppl rising is the intended effect of gradient-ascent
        unlearning and must not trigger this alone (same reasoning as the
        intra-round breaker's forget_spiked_with_retain logic). forget_ppl
        only contributes context if retain or eval has ALSO drifted past
        threshold, i.e. the model is failing broadly, not just
        successfully forgetting.
        """
        if baseline is None:
            return False

        def drift_of(key):
            cur, base = report.get(key), baseline.get(key)
            if cur is not None and base and base > 0:
                return cur / base
            return 0.0

        retain_drift = drift_of("retain_ppl")
        eval_drift = drift_of("eval_ppl")
        forget_drift = drift_of("forget_ppl")

        stability_failed = retain_drift > self.hard_drift_threshold or eval_drift > self.hard_drift_threshold
        worst_stability_drift = max(retain_drift, eval_drift)
        worst_key = "retain_ppl" if retain_drift >= eval_drift else "eval_ppl"

        if stability_failed:
            self._hard_stop_triggered = True
            self._hard_stop_reason = (
                f"{worst_key} drifted {worst_stability_drift:.2f}x from original baseline "
                f"(threshold {self.hard_drift_threshold}x)"
                + (f"; forget_ppl also drifted {forget_drift:.2f}x" if forget_drift > self.hard_drift_threshold else "")
            )
            if self.hard_stop_action == "emergency_lambda":
                self.lambda_ewc = getattr(self.cfg.ewc, "lambda_max", 1e5)
            return True
        return False

    def should_stop(self) -> bool:
        if self._hard_stop_triggered and self.hard_stop_action == "abort":
            return True
        return self._rounds_no_improve >= self.cfg.controller.patience
