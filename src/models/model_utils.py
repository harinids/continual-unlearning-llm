"""Model + tokenizer loading, with optional LoRA wrapping via PEFT."""
from __future__ import annotations


def load_model_and_tokenizer(cfg):
    """
    Load the base causal LM and tokenizer, optionally wrapping the model
    in a LoRA adapter (recommended: keeps unlearning updates cheap and
    reversible/inspectable, and plays nicely with EWC since the Fisher
    Information only needs to be tracked over the small adapter param set).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg.model.base_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if cfg.model.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_cfg = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.alpha,
            lora_dropout=cfg.model.lora.dropout,
            target_modules=list(cfg.model.lora.target_modules),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def build_tiny_test_model():
    """
    Construct a tiny, randomly-initialized GPT-2-architecture model.

    This requires no network / Hub access (no weight download) and is
    used by the test suite to validate the unlearning / EWC / auditor /
    controller logic end-to-end without needing real pretrained weights.
    """
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    config = GPT2Config(
        vocab_size=1000,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)

    # Minimal character-level tokenizer built from scratch (no download,
    # no external vocab files needed) using the `tokenizers` library
    # directly, then wrapped as a HF-compatible fast tokenizer.
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    chars = [chr(c) for c in range(32, 127)]  # printable ASCII
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    trainer = trainers.WordLevelTrainer(
        special_tokens=["<pad>", "<|endoftext|>", "<unk>"],
        vocab_size=1000,
    )
    tok.train_from_iterator([chars, ["<pad>", "<|endoftext|>", "<unk>"]], trainer=trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>",
    )
    return model, tokenizer


def get_trainable_params(model):
    """Return the list of (name, param) currently requiring grad (e.g. LoRA weights)."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]
