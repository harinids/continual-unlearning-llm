"""General-utility check: WikiText-2 perplexity of the base model vs the adapted model."""
import os, json, math
import torch

_wt_cache = {}

def _wikitext_ids(tokenizer):
    from datasets import load_dataset
    key = tokenizer.name_or_path
    if key not in _wt_cache:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        _wt_cache[key] = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    return _wt_cache[key]

@torch.no_grad()
def wikitext2_ppl(model, tokenizer, device, max_len=512, bs=8):
    ids = _wikitext_ids(tokenizer)
    n_windows = ids.numel() // max_len
    was_training = model.training
    model.eval()
    nll, n_tok = 0.0, 0
    for s in range(0, n_windows, bs):
        chunk = torch.stack([ids[i * max_len:(i + 1) * max_len]
                             for i in range(s, min(s + bs, n_windows))]).to(device)
        loss = model(input_ids=chunk, labels=chunk).loss
        cnt = chunk.size(0) * (chunk.size(1) - 1)
        nll += loss.item() * cnt
        n_tok += cnt
    model.train(was_training)
    return math.exp(nll / n_tok)

def utility_before_after(model, tokenizer, device, tag):
    after = wikitext2_ppl(model, tokenizer, device)
    with model.disable_adapter():
        before = wikitext2_ppl(model, tokenizer, device)
    res = {"tag": tag,
           "wikitext2_ppl_before": before,
           "wikitext2_ppl_after": after,
           "rel_change_pct": 100 * (after / before - 1)}
    os.makedirs("results/phase3_utility", exist_ok=True)
    with open(f"results/phase3_utility/{tag}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(res)
    return res

def save_adapter(model, tag):
    os.makedirs("results/adapters", exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(sd, f"results/adapters/{tag}.pt")
