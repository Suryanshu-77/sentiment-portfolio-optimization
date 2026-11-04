"""Shared fixtures: a synthetic market where sentiment *does* predict next-week returns."""

import numpy as np
import pandas as pd
import pytest


def synthetic_market(n_tickers: int = 30, n_weeks: int = 400, signal: float = 0.3,
                     seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (scores[ticker, date, score], daily prices[date x ticker]).

    Each ticker-week has a latent s ~ N(0,1); 3 headlines dated that week's Tuesday score
    tanh(s + N(0, 0.3)). The *next* week's excess log-return is 0.01*signal*s + N(0, 0.02),
    plus a common market return, spread evenly across that week's trading days.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-04", periods=n_weeks * 5)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    s = rng.standard_normal((n_weeks, n_tickers))

    excess = np.zeros((n_weeks, n_tickers))
    excess[1:] = 0.01 * signal * s[:-1] + rng.normal(0, 0.02, (n_weeks - 1, n_tickers))
    market = rng.normal(0.001, 0.01, (n_weeks, 1))
    daily = np.repeat((excess + market) / 5, 5, axis=0)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(daily, axis=0)), index=days, columns=tickers)

    tuesdays = days[1::5]
    rows = []
    for k, day in enumerate(tuesdays):
        noisy = np.tanh(s[k][None, :] + rng.normal(0, 0.3, (3, n_tickers)))
        rows += [(t, day, noisy[j, i]) for j in range(3) for i, t in enumerate(tickers)]
    scores = pd.DataFrame(rows, columns=["ticker", "date", "score"])
    return scores, prices


@pytest.fixture
def synthetic():
    return synthetic_market()


@pytest.fixture
def synthetic_panel():
    from sdpo.features import build_weekly_panel

    return build_weekly_panel(*synthetic_market(signal=0.5))


def make_ctx(panel: pd.DataFrame, i: int = 100):
    """WeekContext at the i-th sequence week, assembled exactly as walk_forward does."""
    from sdpo import config
    from sdpo.backtest import WeekContext
    from sdpo.features import make_sequences

    X, _, _, weeks, tickers = make_sequences(panel)
    t = weeks[i]
    wide_ret = panel["fwd_ret"].unstack("ticker")[tickers]
    wide_ex = panel["fwd_excess"].unstack("ticker")[tickers]
    past = wide_ret.index < t
    return WeekContext(t, tickers, panel.loc[t, config.FEATURES].reindex(tickers), X[i],
                       wide_ret[past].tail(config.COV_WINDOW), wide_ex[past])


@pytest.fixture
def synthetic_ctx(synthetic_panel):
    return make_ctx(synthetic_panel)


@pytest.fixture
def synthetic_splits(synthetic_panel):
    """Chronological 60/20/20 split of the planted-signal sequences (targets: y_excess)."""
    from sdpo.features import make_sequences

    X, y, _, _, _ = make_sequences(synthetic_panel)
    ok = np.isfinite(y).all(axis=1)
    X, y = X[ok], y[ok]
    a, b = int(0.6 * len(X)), int(0.8 * len(X))
    return X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]
