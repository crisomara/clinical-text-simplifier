"""Tests for the PLABA topic-level split, run on a small synthetic release."""

import json

import pytest

import data_prep


def _synthetic_raw(n_topics: int = 20) -> dict:
    raw = {}
    for t in range(1, n_topics + 1):
        raw[str(t)] = {
            "question": f"Question for topic {t}?",
            "question_type": "general",
            f"art{t}": {
                "Title": f"Article {t}",
                "abstract": {
                    "1": f"Topic {t} source sentence one is long enough.",
                    "2": f"Topic {t} source sentence two is long enough.",
                    "3": "ok",  # too short: dropped by the hygiene filter
                },
                "adaptations": {
                    "annotator_a": {"1": f"Plain one for {t}.", "2": f"Plain two for {t}.", "3": "ok"},
                    # Second annotator did not adapt sentence 2.
                    "annotator_b": {"1": f"Another plain one for {t}.", "3": "ok"},
                },
            },
        }
    return raw


def test_split_topics_is_a_deterministic_partition():
    raw = _synthetic_raw()
    train, val, test = data_prep.split_topics(raw)
    assert train | val | test == set(raw)
    assert not (train & val or train & test or val & test)
    assert (len(train), len(val), len(test)) == (14, 3, 3)
    assert data_prep.split_topics(raw) == (train, val, test)


def test_iter_articles_skips_topic_metadata():
    raw = _synthetic_raw(2)
    ids = [art_id for _topic, art_id, _art in data_prep.iter_articles(raw)]
    assert ids == ["art1", "art2"]


@pytest.fixture
def built_splits(tmp_path, monkeypatch):
    raw_path = tmp_path / "plaba_data.json"
    raw_path.write_text(json.dumps(_synthetic_raw()), encoding="utf-8")
    monkeypatch.setattr(data_prep, "RAW_PATH", raw_path)
    monkeypatch.setattr(data_prep, "OUT_DIR", tmp_path / "processed")
    return data_prep.build_splits(), tmp_path / "processed"


def test_train_rows_have_one_row_per_reference(built_splits):
    (train, _val, _test), _out = built_splits
    # 14 train topics x (2 refs for sentence 1 + 1 ref for sentence 2).
    assert len(train) == 14 * 3
    assert all({"source", "target"} <= set(r) for r in train)


def test_eval_rows_keep_every_reference(built_splits):
    (_train, val, test), _out = built_splits
    for rows in (val, test):
        by_sid = {r["sentence_id"]: r for r in rows}
        assert set(by_sid) == {"1", "2"}  # sentence 3 filtered as too short
        assert len(by_sid["1"]["references"]) == 2
        assert len(by_sid["2"]["references"]) == 1


def test_no_topic_leaks_across_splits(built_splits):
    (train, val, test), _out = built_splits
    topics = [{r["topic"] for r in rows} for rows in (train, val, test)]
    assert not (topics[0] & topics[1] or topics[0] & topics[2] or topics[1] & topics[2])


def test_jsonl_files_are_written(built_splits):
    (train, val, test), out = built_splits
    for name, rows in (("train", train), ("val", val), ("test", test)):
        lines = (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == rows
