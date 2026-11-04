"""Headline sentiment scoring: VADER and any Hugging Face 3-class classifier.

Score convention: P(positive) - P(negative) in [-1, 1].
Label convention (unified): 0 = negative, 1 = neutral, 2 = positive.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

UNIFIED = {"negative": 0, "neutral": 1, "positive": 2}


def _columns(id2label: dict[int, str]) -> dict[str, int]:
    """Map label name -> probability column, case-insensitively."""
    cols = {str(name).lower(): int(i) for i, name in id2label.items()}
    missing = set(UNIFIED) - set(cols)
    if missing:
        raise ValueError(f"model labels {id2label} lack {sorted(missing)}")
    return cols


def probs_to_score(probs: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    c = _columns(id2label)
    return probs[:, c["positive"]] - probs[:, c["negative"]]


def probs_to_labels(probs: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    c = _columns(id2label)
    order = [c["negative"], c["neutral"], c["positive"]]  # unified id -> model column
    return probs[:, order].argmax(axis=1)


class HFScorer:
    """Batch scorer for a 3-class sequence-classification model (hub name or local path)."""

    def __init__(self, model_name_or_path: str, batch_size: int = 256, max_length: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.id2label = self.model.config.id2label
        self.batch_size, self.max_length = batch_size, max_length

    def probs(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.float()
                out.append(logits.softmax(-1).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, 3))

    def __call__(self, texts: list[str]) -> np.ndarray:
        return probs_to_score(self.probs(texts), self.id2label)

    def predict_labels(self, texts: list[str]) -> np.ndarray:
        return probs_to_labels(self.probs(texts), self.id2label)


class VaderScorer:
    """Lexicon baseline. Score = compound; label pos >= 0.05, neg <= -0.05, else neutral."""

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.sia = SentimentIntensityAnalyzer()

    def __call__(self, texts: list[str]) -> np.ndarray:
        return np.array([self.sia.polarity_scores(t)["compound"] for t in texts])

    def predict_labels(self, texts: list[str]) -> np.ndarray:
        s = self(texts)
        return np.where(s >= 0.05, 2, np.where(s <= -0.05, 0, 1))


def score_to_parquet(news: pd.DataFrame, scorer: Callable[[list[str]], np.ndarray],
                     out_dir: Path, chunk_size: int = 50_000) -> pd.DataFrame:
    """Score headlines in chunks, one Parquet file per chunk; existing chunks are skipped,
    so a crashed multi-hour run resumes where it stopped. Chunking assumes `news` order is
    stable between runs (it is: load_kaggle_news sorts)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for k, start in enumerate(range(0, len(news), chunk_size)):
        path = out_dir / f"chunk_{k:05d}.parquet"
        chunk = news.iloc[start:start + chunk_size]
        keys = chunk[["ticker", "date"]].reset_index(drop=True)
        cached = pd.read_parquet(path) if path.exists() else None
        # Reuse only a chunk scored from these exact rows; the news set may have changed
        # (e.g. a re-run of stage 01 with a different universe).
        if cached is None or not cached[["ticker", "date"]].reset_index(drop=True).equals(keys):
            cached = keys.assign(
                score=np.asarray(scorer(chunk["headline"].tolist()), dtype="float32"))
            cached.to_parquet(path)
            log.info("scored chunk %d (%d rows)", k, len(chunk))
        parts.append(cached)
    return pd.concat(parts, ignore_index=True)
