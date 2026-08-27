# Can LoRA Fine-Tuning Teach a Tiny Open LLM to Simplify Clinical Text?

*A CPU-only LoRA fine-tuning case study: adapting `Qwen2.5-0.5B-Instruct` to
rewrite biomedical sentences in plain language, then scoring it against PLABA
— a real, professionally human-written plain-language dataset from NIH/NLM —
rather than asserting the fine-tune "worked."*

## The question

Clinical and biomedical writing is dense with jargon that's routine for
practitioners and opaque for patients. Large frontier LLMs can simplify this
kind of text reasonably well out of the box, but that isn't the interesting
question for a portfolio project — the interesting question is narrower and
more falsifiable: **can a small (~0.5B parameter) open model, fine-tuned with
LoRA on a CPU in a modest amount of wall-clock time, get measurably better at
this one task than the same base model prompted zero-shot?** This case study
builds that fine-tune end to end and scores it against real held-out human
references, reporting whatever the numbers actually say.

## The data

- **[PLABA](https://bionlp.nlm.nih.gov/plaba2024/)** (Attal et al. 2023, *A
  dataset for plain language adaptation of biomedical abstracts*, *Scientific
  Data* — [PMC9811873](https://pmc.ncbi.nlm.nih.gov/articles/PMC9811873/)): 75
  consumer-health topics (drawn from real MedlinePlus user questions), ~10
  PubMed abstracts per topic (749 abstracts total), with every abstract
  sentence adapted into plain language by one or more professional annotators
  from NLM and an external annotation company. Downloaded directly from the
  dataset's official OSF release (`https://osf.io/download/4kp7v`), linked
  from the live PLABA-2024 TREC shared task page
  (`https://bionlp.nlm.nih.gov/plaba2024/`) — this is the same public release
  the shared task itself points participants to for training data.
- **Why not the official TREC-2024 test split?** That split is held behind
  NIST's "active participants" area
  (`https://trec.nist.gov/act_part/tracks2024.html`) and requires completing
  shared-task registration through `ir.nist.gov/evalbase` — a multi-step,
  identity-bound process that can't be completed non-interactively. Rather than
  fabricate a substitute test set, this project uses the same public release
  the shared task trains on and creates its own split.
- **Split methodology** (`src/data_prep.py`, deterministic, seed 13): topics —
  not sentences or articles — are shuffled and partitioned 52 train / 11 val /
  12 test topics, so no topic's vocabulary or subject matter appears in more
  than one split. Flattening to sentence level: **6,202 training pairs**,
  **1,085 validation pairs**, **1,242 test pairs**. Articles with more than one
  independent annotator adaptation (170 of 749 articles) contribute one
  training row per annotator; validation/test rows instead keep *every*
  available reference per source sentence (312 of 1,242 test rows have 2
  references) for proper multi-reference SARI/BLEU scoring, since a "correct"
  simplification isn't unique.
- **Integrity check on load**: of 9,320 (article, adaptation, sentence)
  triples, only 1 had a sentence-id present in an adaptation but missing from
  its abstract (skipped) — the data is essentially perfectly aligned.

## Modeling approach

**CPU-only by design, validated before committing.** This machine is a 4-core /
8-thread Intel i5-8365U laptop CPU with 8GB RAM — no CUDA GPU, and the same
class of hardware that produced an OOM crash on a past end-to-end training
attempt for this user. Rather than assume a CPU LoRA loop would just work, a
`--smoke_test` run (`src/train_lora.py --smoke_test`: 64 examples, 1 epoch,
default `batch_size=4`/`grad_accum=4`) was run first. It completed cleanly —
`train_loss: 1.97`, peak process memory ~1.9GB (fp32 weights for a 496M-param
model, well within the 8GB budget), no OOM — but at **8.8 s/example**, which
made the plan's original target of 3 epochs over 6,202 examples (≈45 hours)
obviously infeasible.

**A short, bounded throughput benchmark** (`src/bench_throughput.py`, run
directly against the training forward/backward pass, no full-Trainer
overhead) checked whether a different batch size or thread count would help
before committing to a final configuration:

| threads | batch size | s / example |
|---|---|---|
| 8 | 8 | 11.40 |
| 4 (default) | 4 | 8.81 |
| 4 | 2 | 5.61 |
| 4 | 1 | **5.28** |

Larger batches and extra (hyperthreaded) threads were both *slower* per
example on this CPU, not faster — more padding waste per batch and thread
contention outweighed any BLAS efficiency gain. `batch_size=1`,
`gradient_accumulation_steps=8` (effective batch 8), default 4 threads was
the fastest configuration found and is what the real run used. This
benchmarking was intentionally kept short (a handful of short runs, each a
few minutes) — the goal was "pick a safe, bounded configuration," not search
exhaustively for an optimum.

- **Model:** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
  (Apache 2.0, 496M params including the LM head).
- **Method:** LoRA (`peft`), rank 16 / alpha 32 / dropout 0.05, targeting
  `q_proj, k_proj, v_proj, o_proj` — **2,162,688 trainable parameters, 0.44%**
  of the full 496,195,456. Not full fine-tuning — the base model's weights
  never change.
- **Prompt format:** a fixed system instruction plus `Sentence: {source}`,
  identical at train and inference time (`src/prompting.py` is the single
  source of truth both scripts import, so there's no risk of prompt drift
  between training and eval).
- **Training objective:** standard causal-LM cross-entropy, with loss masked
  to the assistant's response tokens only (the prompt/system/user tokens
  contribute no gradient).

**Real run, reduced scope, honestly documented:** at ~5.3s/example, even one
full epoch over all 6,202 training examples would take roughly 9 hours —
still not a reasonable CPU wall-clock budget for this project. Per the
project's own ground rules ("reduce scope further rather than silently
switch to fabricated results"), the real fine-tuning run used **800 training
examples (a fixed prefix of the shuffled-by-topic training split, not
cherry-picked), 1 epoch, batch_size=1, grad_accum=8, lr=2e-4** — roughly
70-90 minutes of CPU time. This is a materially smaller training set than
the full 6,202-example split, and that's a real limitation carried into the
results below, not a hidden one.

## Evaluation

Both the base model prompted zero-shot and the LoRA-adapted model generate
greedy (`do_sample=False`, beam=1) outputs for the same held-out test
sentences, using the identical prompt. Both sets of outputs are scored against
the same human references with:

- **SARI** — the standard text-simplification metric, rewarding words correctly
  added/kept/deleted relative to the source *and* the human references (unlike
  BLEU/ROUGE, it's specifically designed for simplification, not paraphrase or
  translation).
- **sacreBLEU** and **ROUGE-L** — n-gram/longest-common-subsequence overlap
  with the human reference, standard for reporting even though neither is
  simplification-specific.
- **Flesch-Kincaid Grade Level**, computed on source / prediction / reference,
  to check whether outputs are actually *reading* easier, not just closer to
  the reference by token overlap.

Evaluation ran on the **first 200 of the 1,242 test sentences** (a fixed
prefix of the topic-held-out split, not cherry-picked) — chosen to keep
evaluation wall-clock time bounded (generation is autoregressive and slow on
this CPU: ~5.5s/example, so 200 examples × 2 models ≈ 37 minutes). Both
models used identical greedy decoding (`do_sample=False`, `num_beams=1`,
`max_new_tokens=100`) so any difference in output is attributable to the
weights, not sampling noise. sacreBLEU requires an equal reference count per
example; PLABA rows with only 1 human reference (156/200 in this slice) were
padded by repeating that single reference to match rows with 2 — a standard
workaround for the metric's fixed-width requirement, not fabricated text.

## Results

Real numbers, computed by `src/evaluate_model.py`, sourced to
`reports/eval_results.json`:

| Metric | Zero-shot baseline | LoRA fine-tuned | Change |
|---|---|---|---|
| **SARI** (higher = better simplification) | 29.04 | **34.11** | **+5.06** |
| **sacreBLEU** (higher = closer to reference) | 9.48 | **17.40** | **+7.92 (+83%)** |
| ROUGE-1 | 0.334 | **0.414** | +0.080 |
| ROUGE-2 | 0.129 | **0.210** | +0.080 |
| ROUGE-L | 0.295 | **0.380** | +0.084 |
| FK grade, source (mean) | 14.60 | 14.60 | — (same 200 sources) |
| FK grade, model output (mean) | 12.24 | **11.08** | -1.16 (lower = simpler) |
| FK grade, human reference (mean) | 13.20 | 13.20 | — (same references) |
| FK grade delta, output vs. source | -2.36 | **-3.51** | more aggressive simplification |
| FK grade delta, human reference vs. source | -1.40 | -1.40 | (for comparison) |

**The fine-tune helped, unambiguously, on every metric measured.** This
wasn't a given going in — LoRA on 800 examples for one epoch is a small
amount of supervision, and it was entirely possible the run would show no
effect or a mixed picture across metrics. Instead SARI improved by just over
5 points, BLEU nearly doubled (+83% relative), and every ROUGE variant moved
in the same direction by a similar, consistent margin. Readability moved
further too: the fine-tuned model's outputs average grade 11.08 (vs. 12.24
zero-shot) against source sentences averaging grade 14.6.

## Qualitative examples

Real, unedited outputs from `reports/qualitative_examples.json` (rows 1, 2,
and 10 of the 200 scored):

**A clean win** — fine-tuned nearly matches the human reference's brevity and phrasing; baseline over-elaborates and drifts ("individuals suffering from type 2 diabetes" — the source doesn't say type 2):
> **Source:** Currently, only one half of the people who have diabetes mellitus have been diagnosed.
> **Human reference:** Currently, only half of those with diabetes mellitus have been identified.
> **Baseline:** Currently, less than half of individuals suffering from type 2 diabetes have been identified as having the condition.
> **Fine-tuned:** Only half of those with diabetes have been diagnosed.

**A modest, real improvement** — both outputs are reasonable, but the fine-tuned one is closer to the source's own wording and slightly more concise:
> **Source:** These changes provide an easier and more reliable means of diagnosing persons at risk of complications from hyperglycemia.
> **Human reference:** These changes help improve identification of those at risk from effects of high blood sugar.
> **Baseline:** These adjustments make it easier to identify people who might have problems with high blood sugar earlier on.
> **Fine-tuned:** These changes make it easier to diagnose people who may have problems with high blood sugar.

**A real failure mode, reported honestly** — the fine-tuned model drops the actual clinical content and produces a generic, unfaithful paraphrase; the baseline, while wordier, at least keeps the source's meaning intact:
> **Source:** It is clear that these changes in diagnostic criteria have major importance both for individuals and for resource planning at a national level.
> **Human reference:** It is clear that these changes in identification criteria have major importance both for individuals and for planning at a national level.
> **Baseline:** These changes in diagnostic criteria are crucial for both individual patients and overall resource management at a national scale.
> **Fine-tuned:** It's important to change how doctors diagnose patients so they can better treat them.

## What surprised us

**The fine-tune helped on every single metric, with no mixed picture to
explain away.** Going in, the more likely outcome (given how little
supervision 800 examples / 1 epoch really is) seemed to be a split result —
maybe better BLEU/ROUGE from picking up PLABA's vocabulary and sentence
patterns, but no real readability gain, or vice versa. That's not what
happened: SARI, BLEU, all three ROUGE variants, and the FK grade delta all
moved the same direction by a consistent, non-marginal amount.

**The fine-tuned model simplifies *more aggressively* than the human
annotators do, in terms of grade level** (-3.51 grades vs. source, versus the
human references' own -1.40), and that isn't a free win. The qualitative
spot-check above shows the mechanism: some of that extra "simplification" is
genuinely better, tighter phrasing (examples 1 and 2 above), but some of it
is the model dropping real content and defaulting to vaguer, more generic
phrasing that happens to score reasonably on n-gram overlap metrics without
actually preserving the source's meaning (example 3). A lower FK-grade output
is not automatically a *faithful* one — this is a real instance of a small
model, given a modest amount of supervision, learning "PLABA sentences tend
to be shorter and use common words" about as much as it learns "how to
preserve the specific clinical content while doing so."

**A specific hallucination worth flagging directly**: one fine-tuned output
(row 0 in `reports/qualitative_examples.json`) truncates and appears to
invent a numeric threshold ("...lowered from 140 mg/dL to 12") that isn't in
the source sentence at all. This is exactly the kind of factual-faithfulness
failure that automatic n-gram/SARI metrics are not designed to catch, and
it's the single strongest argument in this project for why "the metrics went
up" is not the same claim as "this model is safe to deploy for real
patient-facing text" — see Limitations.

## Limitations & next steps

- **Held-out split is self-constructed, not the official TREC test set** —
  see "The data" above for why. Numbers here are not directly comparable to
  published PLABA-2024 leaderboard results, which use the gated official split
  and (in most submissions) much larger models.
- **A single 0.5B model, one LoRA configuration.** No hyperparameter sweep
  (rank, target modules, learning rate) and no comparison against a larger
  model in the same family (e.g. Qwen2.5-1.5B-Instruct) — the smoke test's
  timing numbers determined the model-size ceiling for this CPU-only setup;
  see "Modeling approach."
- **Automatic metrics only, and they can be fooled.** SARI/BLEU/ROUGE/FKGL are
  standard but imperfect proxies for whether a simplification is actually
  clearer *and* factually faithful to the source; no clinician or lay-reader
  panel reviewed outputs. The qualitative spot-check above surfaces a real
  case of the fine-tuned model dropping source content in favor of vaguer
  phrasing, and a separate case of an apparently invented numeric value —
  neither would necessarily tank SARI/BLEU/ROUGE, which is exactly the risk
  of trusting automatic metrics alone for a clinical-text use case.
- **Evaluation used 200 of 1,242 test sentences**, not the full test split —
  a deliberate scope reduction (generation is slow and autoregressive on this
  CPU) documented here rather than silently applied; the 200 are a fixed,
  unfiltered prefix of the split, not selected to favor either model.
- **Sentence-level, not document-level.** PLABA's ground truth is annotated
  sentence-by-sentence; a real plain-language rewrite of a full abstract would
  also need to manage document-level coherence and redundancy, which this
  project doesn't test.
