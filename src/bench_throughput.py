"""Quick standalone benchmark: forward+backward throughput for the LoRA setup
under different batch sizes / thread counts, without full Trainer overhead.
Used once, ad hoc, to size the real training run's hyperparameters."""

import argparse
import json
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from prompting import build_chat_messages

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--n_batches", type=int, default=3)
args = ap.parse_args()

torch.set_num_threads(args.threads)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
model.config.use_cache = False
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.train()

rows = []
with open("../data/processed/train.jsonl", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))
rows = rows[: args.batch_size * args.n_batches]

examples = []
for row in rows:
    full_messages = build_chat_messages(row["source"], row["target"])
    prompt_messages = build_chat_messages(row["source"], None)
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(full_ids))):
        labels[i] = -100
    examples.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})

collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

start = time.time()
n_examples = 0
for b in range(args.n_batches):
    batch = examples[b * args.batch_size : (b + 1) * args.batch_size]
    inputs = collator(batch)
    optimizer.zero_grad()
    out = model(**inputs)
    loss = out.loss
    loss.backward()
    optimizer.step()
    n_examples += len(batch)
    print(
        f"batch {b + 1}/{args.n_batches} loss={loss.item():.3f} seq_len={inputs['input_ids'].shape[1]} elapsed={time.time() - start:.1f}s"
    )

elapsed = time.time() - start
print(
    f"threads={args.threads} batch_size={args.batch_size}: {elapsed:.1f}s for {n_examples} examples "
    f"-> {elapsed / n_examples:.3f} s/example"
)
