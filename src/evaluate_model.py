"""
Evaluate base (zero-shot) vs. LoRA-fine-tuned model on the PLABA held-out test
split. Generates outputs for both, then scores both against the same set of
human reference simplifications using:
  - SARI (via `evaluate` easse-based implementation)
  - sacreBLEU
  - ROUGE-L
  - Flesch-Kincaid Grade Level delta (target readability drop vs. source)

Usage:
    python src/evaluate_model.py --adapter_dir checkpoints/lora_adapter \
        --test_path data/processed/test.jsonl --limit 200 \
        --out reports/eval_results.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import textstat
import evaluate as hf_evaluate

from prompting import build_chat_messages

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def generate(model, tokenizer, source, max_new_tokens=100):
    messages = build_chat_messages(source, None)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    # Guard against run-on generations: keep first line only.
    result = result.split("\n")[0].strip()
    return result


def run_generation(model, tokenizer, rows, tag, limit=None, cache_path=None):
    if limit:
        rows = rows[:limit]

    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("n") == len(rows):
            print(f"[{tag}] loaded {len(rows)} cached generations from {cache_path}")
            return cached["predictions"], rows

    outputs = []
    start = time.time()
    for i, row in enumerate(rows):
        pred = generate(model, tokenizer, row["source"])
        outputs.append(pred)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            print(f"  [{tag}] {i + 1}/{len(rows)} generated ({elapsed:.1f}s elapsed)")
    elapsed = time.time() - start
    print(f"[{tag}] generated {len(rows)} outputs in {elapsed:.1f}s "
          f"({elapsed / max(len(rows), 1):.2f} s/example)")

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"n": len(rows), "predictions": outputs}, f, ensure_ascii=False)
        print(f"[{tag}] cached generations to {cache_path}")

    return outputs, rows


def score(rows, predictions):
    sari_metric = hf_evaluate.load("sari")
    bleu_metric = hf_evaluate.load("sacrebleu")
    rouge_metric = hf_evaluate.load("rouge")

    sources = [r["source"] for r in rows]
    references_list = [r["references"] for r in rows]  # list of lists

    # SARI expects: sources (list[str]), predictions (list[str]),
    # references (list[list[str]])
    sari_result = sari_metric.compute(
        sources=sources, predictions=predictions, references=references_list
    )

    # sacrebleu (unlike SARI) requires every example to have the same number
    # of references. PLABA test rows have 1 or 2 references depending on how
    # many annotators adapted that article, so pad shorter reference lists by
    # repeating their own last reference (does not invent new reference text,
    # just satisfies the metric's fixed-width requirement).
    max_refs = max(len(r) for r in references_list)
    padded_references = [r + [r[-1]] * (max_refs - len(r)) for r in references_list]
    bleu_result = bleu_metric.compute(predictions=predictions, references=padded_references)

    # ROUGE: use the first reference per example (rouge implementation expects
    # single-reference-per-pred by default in this simple usage; report against
    # the best/first human reference).
    first_refs = [r[0] for r in references_list]
    rouge_result = rouge_metric.compute(predictions=predictions, references=first_refs)

    # Readability: Flesch-Kincaid Grade Level. Report source grade, prediction
    # grade, and (for reference) the human reference's own grade level, so the
    # "did it get simpler" question is answered against both.
    src_fk = [textstat.flesch_kincaid_grade(s) for s in sources if s.strip()]
    pred_fk = [textstat.flesch_kincaid_grade(p) if p.strip() else src_fk[i]
               for i, p in enumerate(predictions)]
    ref_fk = [textstat.flesch_kincaid_grade(r[0]) for r in references_list if r[0].strip()]

    import statistics as stats
    results = {
        "n_examples": len(rows),
        "sari": sari_result["sari"],
        "bleu": bleu_result["score"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "fk_grade_source_mean": stats.mean(src_fk),
        "fk_grade_prediction_mean": stats.mean(pred_fk),
        "fk_grade_reference_mean": stats.mean(ref_fk),
        "fk_grade_delta_prediction_vs_source": stats.mean(pred_fk) - stats.mean(src_fk),
        "fk_grade_delta_reference_vs_source": stats.mean(ref_fk) - stats.mean(src_fk),
    }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--adapter_dir", default=str(PROJECT_ROOT / "checkpoints/lora_adapter"))
    ap.add_argument("--test_path", default=str(PROJECT_ROOT / "data/processed/test.jsonl"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports/eval_results.json"))
    ap.add_argument("--examples_out", default=str(PROJECT_ROOT / "reports/qualitative_examples.json"))
    args = ap.parse_args()

    rows = load_jsonl(args.test_path)
    print(f"Loaded {len(rows)} test rows (using first {args.limit})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cache_dir = Path(args.out).parent / "_gen_cache"
    baseline_cache = cache_dir / f"baseline_{args.limit}.json"
    ft_cache = cache_dir / f"finetuned_{args.limit}.json"

    print("Loading base model for zero-shot baseline...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    base_model.eval()
    base_preds, used_rows = run_generation(base_model, tokenizer, rows, "baseline", args.limit,
                                            cache_path=baseline_cache)
    base_scores = score(used_rows, base_preds)
    print("Baseline scores:", json.dumps(base_scores, indent=2))

    print("Loading LoRA-adapted model for fine-tuned eval...")
    del base_model
    ft_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    ft_model = PeftModel.from_pretrained(ft_model, args.adapter_dir)
    ft_model.eval()
    ft_preds, _ = run_generation(ft_model, tokenizer, used_rows, "fine-tuned", args.limit,
                                  cache_path=ft_cache)
    ft_scores = score(used_rows, ft_preds)
    print("Fine-tuned scores:", json.dumps(ft_scores, indent=2))

    final = {
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "test_path": args.test_path,
        "n_examples": len(used_rows),
        "baseline_zero_shot": base_scores,
        "fine_tuned": ft_scores,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"Wrote {args.out}")

    examples = []
    for row, bp, fp in zip(used_rows, base_preds, ft_preds):
        examples.append({
            "source": row["source"],
            "human_reference": row["references"][0],
            "baseline_output": bp,
            "fine_tuned_output": fp,
        })
    with open(args.examples_out, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    print(f"Wrote {args.examples_out}")


if __name__ == "__main__":
    main()
