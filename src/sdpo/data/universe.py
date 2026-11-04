"""Universe selection: most-covered tickers in an early window, with complete price history."""

import logging

import pandas as pd

log = logging.getLogger(__name__)


def select_universe(news: pd.DataFrame, prices: pd.DataFrame, n: int,
                    window: tuple[str, str], full_range: tuple[str, str],
                    exclude: set[str] = frozenset()) -> list[str]:
    """Top-`n` tickers by headline count inside `window`, among those with no missing price
    over `full_range`. Ranking uses only the window, so later coverage cannot leak in."""
    in_window = news["date"].between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))
    counts = news.loc[in_window, "ticker"].value_counts()
    counts = counts[~counts.index.isin(exclude)]

    span = prices.loc[full_range[0]:full_range[1]]
    complete = set(span.columns[span.notna().all()])
    dropped = [t for t in counts.index if t not in complete]
    if dropped:
        log.info("universe: %d ranked tickers lack complete prices: %s", len(dropped),
                 dropped[:20])

    ranked = (counts[counts.index.isin(complete)].rename_axis("ticker").reset_index(name="n")
              .sort_values(["n", "ticker"], ascending=[False, True]))
    if len(ranked) < n:
        raise ValueError(f"only {len(ranked)} tickers qualify, need {n}")
    return sorted(ranked["ticker"].head(n))
