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
def fake_artifacts(tmp_path):
    """A minimal artifacts/ directory in the formats scripts 03-06 write."""
    import json

    from sdpo.metrics import perf_metrics, sharpe_ci

    art = tmp_path / "artifacts"
    art.mkdir()
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2016-01-01", "2017-12-29", freq="W-FRI")
    names = ["EW", "MVO", "A-lstm@finetuned"]
    rets = pd.DataFrame(rng.normal(0.002, 0.02, (len(weeks), 3)), weeks, names)
    turn = pd.DataFrame(0.1, weeks, names)
    rets.to_parquet(art / "returns.parquet")
    turn.to_parquet(art / "turnover.parquet")
    metrics = {n: {**perf_metrics(rets[n], turn[n]), "sharpe_ci": sharpe_ci(rets[n])}
               for n in names}
    (art / "metrics.json").write_text(json.dumps(metrics))
    (art / "ic.json").write_text(json.dumps({"A-lstm@finetuned": {"mean_ic": 0.03,
                                                                  "t_stat": 2.1}}))
    conf = [[300, 30, 17], [60, 1400, 106], [10, 40, 425]]
    (art / "nlp_eval.json").write_text(json.dumps(
        {m: {"accuracy": 0.8, "macro_f1": 0.75, "confusion": conf}
         for m in ("vader", "finbert", "finetuned")} | {"best": "finetuned"}))
    idx = pd.MultiIndex.from_product([weeks, ["AAPL", "MSFT"]], names=["week", "ticker"])
    pd.DataFrame({"sent_mean": rng.uniform(-1, 1, len(idx)), "sent_std": 0.1,
                  "news_count": rng.integers(0, 9, len(idx)).astype(float),
                  "price": 100 + rng.normal(0, 1, len(idx)).cumsum()},
                 index=idx).to_parquet(art / "sentiment_weekly.parquet")
    pd.DataFrame({"ticker": ["AAPL", "AAPL", "MSFT"], "headline": ["Up", "Down", "Flat"],
                  "date": pd.to_datetime(["2016-02-01", "2016-03-01", "2016-04-01"]),
                  "score": [0.9, -0.8, 0.0]}).to_parquet(art / "headlines.parquet")
    return art


@pytest.fixture
def synthetic_splits(synthetic_panel):
    """Chronological 60/20/20 split of the planted-signal sequences (targets: y_excess)."""
    from sdpo.features import make_sequences

    X, y, _, _, _ = make_sequences(synthetic_panel)
    ok = np.isfinite(y).all(axis=1)
    X, y = X[ok], y[ok]
    a, b = int(0.6 * len(X)), int(0.8 * len(X))
    return X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]
