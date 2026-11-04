"""Stage 5: weekly feature panels (one per NLP model) plus dashboard sentiment tables.

Writes artifacts/panel_<model>.parquet, sentiment_weekly.parquet, headlines.parquet.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from sdpo import config

log = logging.getLogger(__name__)
from sdpo.features import build_weekly_panel, week_ends, weekly_sentiment_raw

MODELS = ("vader", "finbert", "finetuned")


def best_model(artifacts: Path, available: list[str]) -> str:
    """The NLP model with the best held-out macro-F1, if it was scored; else first available."""
    path = artifacts / "nlp_eval.json"
    best = json.loads(path.read_text()).get("best") if path.exists() else None
    return best if best in available else available[0]


def main(processed: Path = config.PROCESSED_DIR, artifacts: Path = config.ARTIFACTS_DIR) -> None:
    prices = pd.read_parquet(processed / "prices.parquet")
    available = [m for m in MODELS if (artifacts / f"scores_{m}.parquet").exists()]
    if not available:
        raise SystemExit("No artifacts/scores_<model>.parquet found; run scripts/04_score.py")

    for m in available:
        panel = build_weekly_panel(pd.read_parquet(artifacts / f"scores_{m}.parquet"), prices)
        panel.to_parquet(artifacts / f"panel_{m}.parquet")
        log.info("panel_%s: %d weeks x %d tickers", m,
                     panel.index.get_level_values("week").nunique(), prices.shape[1])

    best = best_model(artifacts, available)
    scores = pd.read_parquet(artifacts / f"scores_{best}.parquet")
    sent = weekly_sentiment_raw(scores, prices)
    ends = week_ends(prices.index)
    sent["price"] = prices.loc[ends].stack().reindex(sent.index).to_numpy()
    sent.attrs["model"] = best
    sent.to_parquet(artifacts / "sentiment_weekly.parquet")

    news_path = processed / "news.parquet"
    if news_path.exists():  # score_to_parquet preserves news order, so rows align
        news = pd.read_parquet(news_path)
        news.assign(score=scores["score"].to_numpy()).to_parquet(artifacts / "headlines.parquet")
    print(f"panels: {available}; dashboard sentiment from '{best}'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
