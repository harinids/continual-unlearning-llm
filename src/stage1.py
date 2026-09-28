"""Stage-1 fine-tune: teach the model TOFU before any unlearning, save a small LoRA adapter."""
import math, os, json, torch
from peft import LoraConfig, get_peft_model, TaskType

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.datasets import load_unlearning_dataset
from src.models.model_utils import load_model_and_tokenizer
from src.pipeline import make_loader
from src.unlearning.gradient_ascent import _lm_loss
from src.utils.utility_eval import utility_before_after


def stage1_finetune(cfg_path, tag, targets, r=64, alpha=128, lr=3e-4, epochs=5, bs=16, seed=0):
    cfg = load_config(cfg_path)
    set_seed(seed)
    cfg.model.use_lora = False              # plain model; we attach our own stage-1 LoRA
    cfg.dataset.num_forget_samples = 10**6  # take everything: forget10 (400) ...
    cfg.dataset.num_retain_samples = 10**6  # ... + retain90 (3600) = full TOFU
    ds = load_unlearning_dataset(cfg)
    forget, retain = list(ds.forget), list(ds.retain)
    holdout = next((list(getattr(ds, a)) for a in ("eval", "holdout", "eval_set")
                    if getattr(ds, a, None)), None)
    print(f"forget={len(forget)} retain={len(retain)} holdout={len(holdout) if holdout else None}")

    model, tok = load_model_and_tokenizer(cfg)
    device = cfg.device
    model.to(device)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05,
                                             target_modules=targets, task_type=TaskType.CAUSAL_LM))
    model.print_trainable_parameters()

    @torch.no_grad()
    def ppl(examples, n):
        model.eval()
        tot, k = 0.0, 0
        for b in make_loader(examples[:n], tok, cfg, 8, shuffle=False):
            tot += _lm_loss(model, b, device).item(); k += 1
        model.train()
        return math.exp(tot / k)

    def report(label):
        out = {"forget": ppl(forget, 200), "retain": ppl(retain, 500)}
        if holdout:
            out["holdout"] = ppl(holdout, 200)
        print(label, {k: round(v, 2) for k, v in out.items()})
        return out

    before = report("before:")
    loader = make_loader(forget + retain, tok, cfg, bs, shuffle=True)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.0)
    total = epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / 50) * max(0.0, 1 - st / total))
    model.train()
    for ep in range(epochs):
        run, n = 0.0, 0
        for batch in loader:
            loss = _lm_loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += loss.item(); n += 1
        print(f"epoch {ep+1}/{epochs}  train_loss={run/n:.3f}", flush=True)
    after = report("after: ")

    util = utility_before_after(model, tok, device, f"stage1_{tag}")   # WikiText-2 cost of stage 1
    out_dir = f"results/stage1/{tag}"
    model.save_pretrained(out_dir)
    with open(f"{out_dir}/stage1_meta.json", "w") as f:
        json.dump({"before": before, "after": after, "wikitext2": util, "r": r, "alpha": alpha,
                   "lr": lr, "epochs": epochs, "bs": bs, "seed": seed, "targets": targets}, f, indent=2)
    assert os.path.exists(f"{out_dir}/adapter_model.safetensors"), "adapter save failed"
    print("saved", out_dir)
    return before, after
