"""Fine-tune GPT-2 on full TOFU before unlearning. Run once, then point
train.py's --base-model at the saved checkpoint dir."""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

MODEL = "gpt2"
OUT = "./outputs/gpt2_tofu_finetuned"
MAX_LEN = 256

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL).to("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("locuslab/TOFU", "full")["train"]

def fmt(row):
    text = f"Question: {row['question'].strip()}\nAnswer: {row['answer'].strip()}"
    enc = tok(text, truncation=True, max_length=MAX_LEN, padding="max_length")
    return enc

ds = ds.map(fmt, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=20,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to=[],
)

collator = DataCollatorForLanguageModeling(tok, mlm=False)
trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
trainer.train()

model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"Saved fine-tuned model to {OUT}")
