import numpy as np
import pandas as pd
import pytest

from sdpo.nlp.score import VaderScorer, probs_to_labels, probs_to_score, score_to_parquet

PROSUS = {0: "positive", 1: "negative", 2: "neutral"}


def test_probs_to_score_uses_label_names_not_positions():
    probs = np.array([[0.7, 0.2, 0.1]])  # ProsusAI order: positive, negative, neutral
    assert probs_to_score(probs, PROSUS)[0] == pytest.approx(0.5)


def test_probs_to_score_case_insensitive():
    probs = np.array([[0.1, 0.3, 0.6]])
    assert probs_to_score(probs, {0: "Negative", 1: "Neutral", 2: "Positive"})[0] == \
        pytest.approx(0.5)


def test_probs_to_labels_maps_to_unified_ids():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    assert probs_to_labels(probs, PROSUS).tolist() == [2, 0, 1]  # pos, neg, neu


def _news(n):
    return pd.DataFrame({"ticker": ["A"] * n, "headline": [f"h{i}" for i in range(n)],
                         "date": pd.date_range("2012-01-02", periods=n)})


def test_score_to_parquet_resumes(tmp_path):
    calls = []

    def fake(texts):
        calls.append(len(texts))
        return np.zeros(len(texts))

    score_to_parquet(_news(5), fake, tmp_path, chunk_size=2)  # 3 chunks
    assert calls == [2, 2, 1]
    calls.clear()
    (tmp_path / "chunk_00002.parquet").unlink()
    out = score_to_parquet(_news(5), fake, tmp_path, chunk_size=2)
    assert calls == [1] and len(out) == 5
    assert list(out.columns) == ["ticker", "date", "score"]


def test_vader_labels():
    assert VaderScorer().predict_labels(["great profits soar", "terrible losses crash"]).tolist() \
        == [2, 0]


def test_score_to_parquet_rescores_stale_chunks(tmp_path):
    # e.g. the universe changed and stage 01 was re-run: existing chunks no longer match
    old = _news(4)
    new = _news(4).assign(ticker="B")
    score_to_parquet(old, lambda t: np.zeros(len(t)), tmp_path, chunk_size=2)
    out = score_to_parquet(new, lambda t: np.ones(len(t)), tmp_path, chunk_size=2)
    assert (out["ticker"] == "B").all() and (out["score"] == 1).all()
