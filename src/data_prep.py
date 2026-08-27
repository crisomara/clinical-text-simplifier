"""
Data preparation for the Clinical Text Simplifier project.

Source: the public PLABA dataset release (Attal et al. 2023, "A dataset for
plain language adaptation of biomedical abstracts", Scientific Data), downloaded
directly from the official OSF repository linked from the PLABA-2024 TREC shared
task page (https://bionlp.nlm.nih.gov/plaba2024/):
    https://osf.io/download/4kp7v  ->  data/raw/plaba_data.json

This is the real, human-authored dataset: 75 consumer-health topics, ~10 PubMed
abstracts per topic, each abstract sentence adapted into plain language by one or
more professional annotators. We do NOT use the official TREC test split, because
that lives behind NIST's gated "active participants" area and requires shared-task
registration we can't complete non-interactively. Instead we carve our own
train/val/test split out of the public release, splitting by TOPIC (not by
sentence or article) so that no topic's vocabulary/subject matter leaks between
splits. This substitution is documented here and in the README/report.

Output: data/processed/{train,val,test}.jsonl
  train.jsonl: one row per (source_sentence, single reference) pair. Articles
    with multiple independent annotator adaptations contribute one training row
    per annotator (legitimate additional supervision, not duplication of the
    exact same pair).
  val.jsonl / test.jsonl: one row per source sentence, with ALL available
    reference simplifications collected into a list (multi-reference eval is
    the standard, less noisy way to score SARI/BLEU against PLABA, since good
    simplifications are not unique).
"""
import json
import random
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "plaba_data.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SEED = 13
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remainder (~0.15) is test


def load_raw():
    with open(RAW_PATH, encoding="utf-8") as f:
        return json.load(f)


def iter_articles(raw):
    """Yield (topic_id, article_id, article_dict) for every real article."""
    for topic_id, arts in raw.items():
        for art_id, art in arts.items():
            if art_id in ("question", "question_type"):
                continue
            yield topic_id, art_id, art


def split_topics(raw, seed=SEED):
    topics = sorted(raw.keys(), key=lambda t: int(t))
    rng = random.Random(seed)
    rng.shuffle(topics)
    n = len(topics)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    train_topics = set(topics[:n_train])
    val_topics = set(topics[n_train:n_train + n_val])
    test_topics = set(topics[n_train + n_val:])
    return train_topics, val_topics, test_topics


def build_splits():
    raw = load_raw()
    train_topics, val_topics, test_topics = split_topics(raw)
    print(f"topics -> train {len(train_topics)}, val {len(val_topics)}, test {len(test_topics)}")

    train_rows = []
    val_rows = []
    test_rows = []

    n_articles = 0
    n_sentences_seen = 0
    n_mismatched_ids = 0

    for topic_id, art_id, art in iter_articles(raw):
        n_articles += 1
        abstract = art.get("abstract", {})
        adaptations = art.get("adaptations", {})
        title = art.get("Title", "")

        # Collect all references per sentence id across every annotator adaptation.
        refs_by_sid = {}
        for _aname, adapt in adaptations.items():
            for sid, source_sent in abstract.items():
                n_sentences_seen += 1
                if sid not in adapt:
                    n_mismatched_ids += 1
                    continue
                ref = adapt[sid].strip()
                if not ref:
                    continue
                refs_by_sid.setdefault(sid, {"source": source_sent.strip(), "refs": []})
                refs_by_sid[sid]["refs"].append(ref)

        if topic_id in train_topics:
            for sid, entry in refs_by_sid.items():
                for ref in entry["refs"]:
                    train_rows.append({
                        "topic": topic_id,
                        "article_id": art_id,
                        "title": title,
                        "sentence_id": sid,
                        "source": entry["source"],
                        "target": ref,
                    })
        else:
            bucket = val_rows if topic_id in val_topics else test_rows
            for sid, entry in refs_by_sid.items():
                bucket.append({
                    "topic": topic_id,
                    "article_id": art_id,
                    "title": title,
                    "sentence_id": sid,
                    "source": entry["source"],
                    "references": entry["refs"],
                })

    # Basic hygiene: drop degenerate rows (empty/very short source or target)
    train_rows = [r for r in train_rows if len(r["source"]) > 3 and len(r["target"]) > 3]
    val_rows = [r for r in val_rows if len(r["source"]) > 3 and r["references"]]
    test_rows = [r for r in test_rows if len(r["source"]) > 3 and r["references"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        path = OUT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows -> {path}")

    print(f"articles processed: {n_articles}")
    print(f"sentence-id mismatches skipped: {n_mismatched_ids} / {n_sentences_seen}")

    return train_rows, val_rows, test_rows


if __name__ == "__main__":
    build_splits()
