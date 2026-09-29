"""Run one sequential-unlearning experiment, measure WikiText-2 utility, save results, assert they exist."""
import time, json, os, gc, torch

from src.utils.config import load_config
from src.data.datasets import load_unlearning_dataset
from src.models.model_utils import load_model_and_tokenizer
from src.scheduler.forget_request_scheduler import ForgetRequestScheduler
from src.sequential_pipeline import SequentialUnlearningPipeline
from src.utils.utility_eval import utility_before_after, save_adapter


def run_with_utility(cfg_path, tag, method=None, seed=42, max_rounds=5,
                     num_requests=8, stage1=None):
    cfg = load_config(cfg_path)
    if method:
        cfg.unlearning.method = method
    if stage1:
        cfg.model.stage1_adapter = stage1     # merged into base by load_model_and_tokenizer
    cfg.controller.max_rounds = max_rounds
    cfg.seed = seed
    # read this line first; interrupt if it is not the config you meant
    print(f"[{tag}] method={cfg.unlearning.method} | ewc.enabled={cfg.ewc.enabled} | "
          f"geometry={getattr(cfg, 'geometry', None)} | seed={cfg.seed} | "
          f"max_rounds={cfg.controller.max_rounds} | stage1={getattr(cfg.model, 'stage1_adapter', None)}")

    model, tok = load_model_and_tokenizer(cfg)
    data = load_unlearning_dataset(cfg)
    sched = ForgetRequestScheduler(data.forget, num_requests=num_requests, seed=cfg.seed)
    pipe = SequentialUnlearningPipeline(cfg, model, tok, data, sched)

    t0 = time.time()
    results = pipe.run()
    print(f"[{tag}] run done in {(time.time()-t0)/60:.1f} min")

    os.makedirs("results/phase3_utility", exist_ok=True)
    run_path = f"results/phase3_utility/{tag}_run.json"
    with open(run_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    pp = pipe.pipeline
    res = utility_before_after(pp.model, tok, pp.device, tag)
    save_adapter(pp.model, tag)
    assert os.path.exists(run_path) and os.path.exists(f"results/adapters/{tag}.pt"), "save failed"
    print(f"[{tag}] confirmed on disk")

    del pipe, model, pp
    gc.collect(); torch.cuda.empty_cache()
    return res

def run_with_utility_patience(cfg_path, tag, method=None, seed=42, max_rounds=5,
                               num_requests=8, stage1=None, patience=5):
    """Same as run_with_utility, with a patience override for testing the stopping rule."""
    from src.utils.config import load_config
    from src.data.datasets import load_unlearning_dataset
    from src.models.model_utils import load_model_and_tokenizer
    from src.scheduler.forget_request_scheduler import ForgetRequestScheduler
    from src.sequential_pipeline import SequentialUnlearningPipeline
    from src.utils.utility_eval import utility_before_after, save_adapter
    import time, json, os, gc, torch

    cfg = load_config(cfg_path)
    if method:
        cfg.unlearning.method = method
    if stage1:
        cfg.model.stage1_adapter = stage1
    cfg.controller.max_rounds = max_rounds
    cfg.controller.patience = patience
    cfg.seed = seed
    print(f"[{tag}] method={cfg.unlearning.method} | patience={cfg.controller.patience} | "
          f"max_rounds={cfg.controller.max_rounds} | stage1={stage1}")

    model, tok = load_model_and_tokenizer(cfg)
    data = load_unlearning_dataset(cfg)
    sched = ForgetRequestScheduler(data.forget, num_requests=num_requests, seed=cfg.seed)
    pipe = SequentialUnlearningPipeline(cfg, model, tok, data, sched)

    t0 = time.time()
    results = pipe.run()
    print(f"[{tag}] run done in {(time.time()-t0)/60:.1f} min")

    os.makedirs("results/phase3_utility", exist_ok=True)
    with open(f"results/phase3_utility/{tag}_run.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    pp = pipe.pipeline
    res = utility_before_after(pp.model, tok, pp.device, tag)
    save_adapter(pp.model, tag)
    del pipe, model, pp
    gc.collect(); torch.cuda.empty_cache()
    return res
