"""Portfolio strategies. Each implements sdpo.backtest.Strategy: fit once per test year,
then weights(ctx) every week (None = no solution, the engine keeps last week's weights)."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from sdpo import config
from sdpo.backtest import Split, Strategy, WeekContext
from sdpo.portfolio.black_litterman import implied_prior
from sdpo.portfolio.optimize import cap_weights, max_sharpe_weights, shrunk_cov


class _NoFit:
    def fit(self, train: Split, val: Split) -> None:
        pass


class EqualWeight(_NoFit):
    def weights(self, ctx: WeekContext) -> pd.Series:
        return pd.Series(1 / len(ctx.tickers), index=ctx.tickers)


class MeanVariance(_NoFit):
    """Classic max-Sharpe on trailing-52w mean returns and Ledoit-Wolf covariance."""

    def weights(self, ctx: WeekContext) -> pd.Series | None:
        mu = ctx.trailing_returns.mean()
        return max_sharpe_weights(mu, shrunk_cov(ctx.trailing_returns), config.MAX_WEIGHT)


class BLNoViews(_NoFit):
    """Black-Litterman with no views: the equal-weight equilibrium prior, then max-Sharpe."""

    def weights(self, ctx: WeekContext) -> pd.Series | None:
        sigma = shrunk_cov(ctx.trailing_returns)
        n = len(ctx.tickers)
        pi = pd.Series(implied_prior(sigma.to_numpy(), np.full(n, 1 / n)), index=ctx.tickers)
        return max_sharpe_weights(pi, sigma, config.MAX_WEIGHT)


class SentimentTilt(_NoFit):
    """Strategy B: w_i proportional to max(0, 1 + k * z(sent_mean)_i), capped at 10%."""

    def __init__(self, k: float = config.TILT_K):
        self.k = k

    def weights(self, ctx: WeekContext) -> pd.Series:
        raw = (1 + self.k * ctx.features["sent_mean"]).clip(lower=0)
        if raw.sum() <= 0:
            return pd.Series(1 / len(ctx.tickers), index=ctx.tickers)
        return cap_weights(raw / raw.sum(), config.MAX_WEIGHT)


BASELINES: dict[str, Callable[[], Strategy]] = {
    "EW": EqualWeight,
    "MVO": MeanVariance,
    "BL-no-views": BLNoViews,
    "B-tilt": SentimentTilt,
}
