"""
Knowledge Auditor: verifies whether unlearning actually erased the
targeted knowledge, and whether retained knowledge/capabilities survived.

Metrics:
  - perplexity: on forget / retain / eval splits. Forget-set perplexity
    should rise after unlearning; retain/eval perplexity should stay flat.
  - exact_match: for QA-style data (e.g. TOFU), checks whether the
    greedy-decoded continuation still reproduces the target answer.
  - membership_inference: a lightweight loss-threshold MIA — if the
    model's loss on a forget example is statistically indistinguishable
    from its loss on held-out (never-seen) data, we treat the example as
    successfully "forgotten" in the membership-inference sense.
"""
from __future__ import annotations
import math


class KnowledgeAuditor:
    def __init__(self, model, tokenizer, device, cfg):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = cfg

    def _perplexity(self, dataloader) -> float:
        import torch

        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                n_tok = attention_mask.sum().item()
                total_loss += out.loss.item() * n_tok
                total_tokens += n_tok
        if total_tokens == 0:
            return float("nan")
        return math.exp(min(total_loss / total_tokens, 20))  # cap to avoid overflow

    def _exact_match(self, examples, max_new_tokens: int = 16) -> float:
        """Greedy-generate a completion for the question and compare to the target answer."""
        import torch

        self.model.eval()
        hits = 0
        n = 0
        with torch.no_grad():
            for ex in examples:
                meta = ex.get("meta") if isinstance(ex, dict) else getattr(ex, "meta", None)
                if not meta or "question" not in meta:
                    continue
                prompt = f"Question: {meta['question']}\nAnswer:"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                out_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                gen = self.tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                target = meta["answer"].strip().lower()
                if target and target[:20] in gen.strip().lower():
                    hits += 1
                n += 1
        return hits / n if n else float("nan")

    def _membership_inference_gap(self, forget_loader, eval_loader) -> float:
        """
        |mean loss on forget set - mean loss on eval set|.
        Near 0 => forget examples now look statistically like unseen data
        (good — the model can no longer be distinguished from one that
        never saw them). Large gap => still detectably memorized.
        """
        import torch

        def mean_loss(loader):
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch.get("labels", input_ids).to(self.device)
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    losses.append(out.loss.item())
            return sum(losses) / len(losses) if losses else float("nan")

        return abs(mean_loss(forget_loader) - mean_loss(eval_loader))

    def audit(self, forget_loader, retain_loader, eval_loader, raw_forget_examples=None) -> dict:
        report = {}
        if "perplexity" in self.cfg.auditor.metrics:
            report["forget_ppl"] = self._perplexity(forget_loader)
            report["retain_ppl"] = self._perplexity(retain_loader)
            report["eval_ppl"] = self._perplexity(eval_loader)
        if "exact_match" in self.cfg.auditor.metrics and raw_forget_examples:
            report["forget_exact_match"] = self._exact_match(raw_forget_examples)
        if "membership_inference" in self.cfg.auditor.metrics:
            report["mia_gap"] = self._membership_inference_gap(forget_loader, eval_loader)

        report["forget_ok"] = report.get("forget_exact_match", 1.0) <= self.cfg.auditor.forget_threshold \
            if "forget_exact_match" in report else None
        report["retain_ok"] = None  # filled in relative to a pre-unlearning baseline by the pipeline
        return report
