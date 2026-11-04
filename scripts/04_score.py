"""Stage 4: score every universe headline with one model; resumable.

Usage: python scripts/04_score.py --model {vader,finbert,finetuned}
Writes artifacts/scores_<model>.parquet (ticker, date, score).
"""

import argparse
import logging

import pandas as pd

from sdpo import config
from sdpo.nlp.score import HFScorer, VaderScorer, score_to_parquet

MODELS = {"finbert": "ProsusAI/finbert",
          "finetuned": str(config.ARTIFACTS_DIR / "models" / "finetuned")}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["vader", "finbert", "finetuned"], required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    news = pd.read_parquet(config.PROCESSED_DIR / "news.parquet")
    scorer = VaderScorer() if args.model == "vader" else HFScorer(MODELS[args.model],
                                                                  batch_size=args.batch_size)
    chunks = config.ARTIFACTS_DIR / "score_chunks" / args.model
    scores = score_to_parquet(news, scorer, chunks)
    out = config.ARTIFACTS_DIR / f"scores_{args.model}.parquet"
    scores.to_parquet(out)
    print(f"{len(scores)} headlines scored -> {out}")
    print(scores["score"].describe().to_string())


if __name__ == "__main__":
    main()
