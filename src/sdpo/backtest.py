"""Walk-forward backtest: retrain yearly, rebalance weekly, charge turnover costs.

For test year Y: train on TRAIN_START..end of Y-2, validate on Y-1, test on Y. Each split
drops weeks whose forward-return label is realized after the split ends (purging), so no
label leaks across a boundary.
"""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from sdpo import config
from sdpo.features import make_sequences

log = logging.getLogger(__name__)


@dataclass
class Split:
    X: np.ndarray  # [T,N,L,F]
    y_excess: np.ndarray  # [T,N]
    y_ret: np.ndarray  # [T,N]
    weeks: pd.DatetimeIndex
    tickers: list[str]
    panel: pd.DataFrame  # panel rows for these weeks


@dataclass
class WeekContext:
    week: pd.Timestamp
    tickers: list[str]
    features: pd.DataFrame  # N x F at `week`
    X_seq: np.ndarray  # [N,L,F] ending at `week`
    trailing_returns: pd.DataFrame  # <= COV_WINDOW realized weekly returns, weeks < `week`
    past_excess: pd.DataFrame  # realized fwd_excess for weeks < `week`


class Strategy(Protocol):
    def fit(self, train: Split, val: Split) -> None: ...

    def weights(self, ctx: WeekContext) -> pd.Series | None: ...


@dataclass
class BacktestResult:
    returns: pd.DataFrame  # week x strategy, net of costs
    weights: pd.DataFrame  # (week, strategy) x ticker
    turnover: pd.DataFrame  # week x strategy, L1
    strategies: dict = field(default_factory=dict)  # (year, name) -> fitted strategy


def _sequences(panel: pd.DataFrame, seq_len: int):
    X, y_ex, y_ret, weeks, tickers = make_sequences(panel, seq_len)
    next_week = pd.Series(weeks[1:].append(pd.DatetimeIndex([pd.NaT])), index=weeks)

    def split(start: pd.Timestamp, end: pd.Timestamp) -> Split:
        m = ((weeks >= start) & (weeks <= end) & (next_week <= end)).to_numpy()
        w = weeks[m]
        return Split(X[m], y_ex[m], y_ret[m], w, tickers,
                     panel.loc[panel.index.get_level_values("week").isin(w)])

    return X, y_ret, weeks, tickers, split


def _year_splits(split, year: int) -> tuple[Split, Split]:
    return (split(pd.Timestamp(config.TRAIN_START), pd.Timestamp(f"{year - 2}-12-31")),
            split(pd.Timestamp(f"{year - 1}-01-01"), pd.Timestamp(f"{year - 1}-12-31")))


def make_splits(panel: pd.DataFrame, year: int,
                seq_len: int = config.SEQ_LEN) -> tuple[Split, Split]:
    """(train, val) used for test year `year`; year=2021 gives the final live model's data."""
    return _year_splits(_sequences(panel, seq_len)[-1], year)


def _check(w: pd.Series, max_weight: float, name: str, week) -> None:
    assert w.notna().all(), f"{name} {week}: NaN weights"
    assert (w >= -1e-9).all(), f"{name} {week}: negative weight"
    assert abs(w.sum() - 1) < 1e-6, f"{name} {week}: weights sum to {w.sum()}"
    assert w.max() <= max_weight + 1e-6, f"{name} {week}: weight {w.max()} > {max_weight}"


def walk_forward(panel: pd.DataFrame, factories: dict[str, Callable[[], Strategy]],
                 test_years: Iterable[int], max_weight: float = config.MAX_WEIGHT,
                 seq_len: int = config.SEQ_LEN) -> BacktestResult:
    X, y_ret, weeks, tickers, split = _sequences(panel, seq_len)
    wide_ret = panel["fwd_ret"].unstack("ticker")[tickers]
    wide_ex = panel["fwd_excess"].unstack("ticker")[tickers]

    rets, wts, turns, fitted = {}, {}, {}, {}
    prev = {name: pd.Series(0.0, index=tickers) for name in factories}
    for year in test_years:
        train, val = _year_splits(split, year)
        strategies = {}
        for name, make in factories.items():
            strategies[name] = fitted[(year, name)] = make()
            strategies[name].fit(train, val)
        log.info("year %d: fitted %s (train %d weeks, val %d)", year, list(factories),
                 len(train.weeks), len(val.weeks))

        test_idx = np.flatnonzero((weeks.year == year) & np.isfinite(y_ret).all(axis=1))
        for i in test_idx:
            t = weeks[i]
            past = wide_ret.index < t
            ctx = WeekContext(t, tickers, panel.loc[t, config.FEATURES].reindex(tickers),
                              X[i], wide_ret[past].tail(config.COV_WINDOW), wide_ex[past])
            for name, strat in strategies.items():
                w = strat.weights(ctx)
                if w is None:
                    log.info("%s %s: no weights, keeping previous", name, t.date())
                    w = prev[name] if prev[name].sum() > 0 else pd.Series(1 / len(tickers),
                                                                          index=tickers)
                w = w.reindex(tickers).astype(float)
                _check(w, max_weight, name, t.date())
                turnover = float((w - prev[name]).abs().sum())
                rets.setdefault(name, {})[t] = float(w @ y_ret[i]) - \
                    config.COST_BPS / 1e4 * turnover
                turns.setdefault(name, {})[t] = turnover
                wts[(t, name)] = w
                prev[name] = w

    weights = pd.DataFrame(wts).T
    weights.index = weights.index.set_names(["week", "strategy"])
    return BacktestResult(pd.DataFrame(rets), weights, pd.DataFrame(turns), fitted)
