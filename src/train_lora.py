"""
LoRA fine-tuning of a small open instruct model on PLABA sentence-simplification
pairs. CPU-only by design (see README / case study for why). Run a smoke test
first with --max_train_samples 64 --epochs 1 to confirm the loop runs without
OOM and to measure per-step time before committing to a full run.

Usage:
    python src/train_lora.py --smoke_test
    python src/train_lora.py --epochs 3 --output_dir checkpoints/qwen05b_lora
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

from prompting import build_chat_messages

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LEN = 320


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class SimplificationDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=MAX_LEN):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        source, target = row["source"], row["target"]

        full_messages = build_chat_messages(source, target)
        prompt_messages = build_chat_messages(source, None)

        full_text = self.tok.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.tok.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        full_ids = self.tok(full_text, add_special_tokens=False, truncation=True,
                             max_length=self.max_len)["input_ids"]
        prompt_ids = self.tok(prompt_text, add_special_tokens=False, truncation=True,
                               max_length=self.max_len)["input_ids"]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--train_path", default=str(PROJECT_ROOT / "data/processed/train.jsonl"))
    ap.add_argument("--output_dir", default=str(PROJECT_ROOT / "checkpoints/lora_adapter"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--smoke_test", action="store_true",
                     help="Tiny run: 64 examples, 1 epoch, to validate the loop.")
    args = ap.parse_args()

    if args.smoke_test:
        args.max_train_samples = 64
        args.epochs = 1
        args.output_dir = str(PROJECT_ROOT / "checkpoints/smoke_test_adapter")

    print(f"Loading tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    rows = load_jsonl(args.train_path)
    if args.max_train_samples:
        rows = rows[: args.max_train_samples]
    print(f"Training on {len(rows)} examples")

    dataset = SimplificationDataset(rows, tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "checkpoints/_trainer_tmp"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        dataloader_num_workers=0,
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s for {len(rows)} examples, "
          f"{args.epochs} epoch(s) -> {elapsed / max(len(rows) * args.epochs, 1):.3f} s/example")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
