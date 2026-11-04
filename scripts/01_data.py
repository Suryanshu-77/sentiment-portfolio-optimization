"""Stage 1: load Kaggle news, download prices, select the universe.

Writes data/processed/{news.parquet, prices.parquet, universe.json}.
"""

import json
import logging
import sys

import pandas as pd

from sdpo import config
from sdpo.data.news import load_kaggle_news
from sdpo.data.prices import download_prices
from sdpo.data.universe import select_universe

CSV_NAMES = ["raw_analyst_ratings.csv", "raw_partner_headlines.csv"]
N_CANDIDATES = 80  # bounds the price download to the best-covered tickers


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = [config.RAW_DIR / n for n in CSV_NAMES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(
            "Missing Kaggle files:\n  " + "\n  ".join(map(str, missing)) +
            "\nDownload 'Massive Stock News Analysis DB for NLP/Backtests' from Kaggle "
            f"(kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests)"
            f"\nand put the CSVs in {config.RAW_DIR}"
        )

    news = load_kaggle_news(paths)
    window = news["date"].between(*map(pd.Timestamp, config.UNIVERSE_WINDOW))
    counts = news.loc[window, "ticker"].value_counts()
    counts = counts[~counts.index.isin(config.EXCLUDE_TICKERS)]
    candidates = counts.head(N_CANDIDATES).index.tolist()
    prices = download_prices(candidates, config.PRICE_START, config.PRICE_END)

    full_range = (config.PRICE_START, "2020-06-30")
    universe = select_universe(news, prices, config.N_UNIVERSE, config.UNIVERSE_WINDOW,
                               full_range, exclude=config.EXCLUDE_TICKERS)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    news[news["ticker"].isin(universe)].to_parquet(config.PROCESSED_DIR / "news.parquet")
    prices[universe].to_parquet(config.PROCESSED_DIR / "prices.parquet")
    (config.PROCESSED_DIR / "universe.json").write_text(json.dumps(universe, indent=2))

    counts = news[news["ticker"].isin(universe)]["ticker"].value_counts()
    print(f"Universe ({len(universe)}): {universe}")
    print(f"Headlines in universe: {int(counts.sum())}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
