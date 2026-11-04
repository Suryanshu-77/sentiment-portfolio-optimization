"""Live suggested weights: score recent news, build this week's features, run Strategy A."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from sdpo.backtest import context_at
from sdpo.features import build_weekly_panel
from sdpo.strategies import EqualWeight, StrategyA


def suggest_weights(news: pd.DataFrame, prices: pd.DataFrame,
                    scorer: Callable[[list[str]], np.ndarray],
                    model_path: Path) -> pd.Series:
    """Strategy A weights for the latest week. `prices` needs about 80 weeks of daily history
    (52-week covariance + 12-week sequences + warm-up). Tickers without news this window are
    treated as no-news weeks. Falls back to equal weight if the optimizer has no solution."""
    scores = news[["ticker", "date"]].assign(
        score=np.asarray(scorer(news["headline"].tolist()), dtype=float))
    ctx = context_at(build_weekly_panel(scores, prices))
    w = StrategyA.load(model_path).weights(ctx)
    return w if w is not None else EqualWeight().weights(ctx)
