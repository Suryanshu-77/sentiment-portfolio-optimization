from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from sdpo import config
from sdpo.backtest import Split, WeekContext, walk_forward


def _panel(start="2010-01-01", end="2016-12-31", tickers=("A", "B"), seed=0):
    weeks = pd.date_range(start, end, freq="W-FRI")
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([weeks, list(tickers)], names=["week", "ticker"])
    df = pd.DataFrame(0.0, index=idx, columns=config.FEATURES)
    df["fwd_ret"] = rng.normal(0.002, 0.02, len(idx))
    df["fwd_excess"] = df["fwd_ret"] - df.groupby(level="week")["fwd_ret"].transform("mean")
    df.loc[weeks[-1], ["fwd_ret", "fwd_excess"]] = np.nan
    return df


class Fixed:
    """Alternates [0.6, 0.4] and [0.4, 0.6] by week parity; optional None weeks."""

    def __init__(self, none_on=()):
        self.none_on, self.n = set(none_on), 0

    def fit(self, train, val):
        pass

    def weights(self, ctx):
        self.n += 1
        if self.n in self.none_on:
            return None
        w = [0.6, 0.4] if self.n % 2 else [0.4, 0.6]
        return pd.Series(w, index=ctx.tickers)


def test_hand_computed_two_asset_portfolio():
    panel = _panel()
    res = walk_forward(panel, {"F": Fixed}, test_years=[2016], max_weight=1.0, seq_len=1)
    test_weeks = res.returns.index
    r = panel["fwd_ret"].unstack("ticker").loc[test_weeks]
    w = np.array([[0.6, 0.4] if i % 2 == 0 else [0.4, 0.6] for i in range(len(test_weeks))])
    prev = np.vstack([np.zeros(2), w[:-1]])  # first week buys in from cash
    cost = 0.001 * np.abs(w - prev).sum(axis=1)
    expected = (w * r.to_numpy()).sum(axis=1) - cost
    np.testing.assert_allclose(res.returns["F"].to_numpy(), expected)
    np.testing.assert_allclose(res.turnover["F"].to_numpy(), np.abs(w - prev).sum(axis=1))


def test_none_keeps_previous_weights():
    res = walk_forward(_panel(), {"F": lambda: Fixed(none_on={3})}, test_years=[2016],
                       max_weight=1.0, seq_len=1)
    w = res.weights.xs("F", level="strategy")
    pd.testing.assert_series_equal(w.iloc[2], w.iloc[1], check_names=False)
    assert res.turnover["F"].iloc[2] == 0


def test_none_on_first_week_falls_back_to_equal_weight():
    res = walk_forward(_panel(), {"F": lambda: Fixed(none_on={1})}, test_years=[2016],
                       max_weight=1.0, seq_len=1)
    np.testing.assert_allclose(res.weights.xs("F", level="strategy").iloc[0], [0.5, 0.5])


class Recorder:
    seen: ClassVar[dict] = {}

    def fit(self, train: Split, val: Split):
        Recorder.seen.update(train=train.weeks, val=val.weeks)

    def weights(self, ctx: WeekContext):
        Recorder.seen.setdefault("ctx", []).append(ctx)
        return pd.Series(0.5, index=ctx.tickers)


def test_split_dates_follow_spec_and_labels_never_cross_boundaries():
    Recorder.seen = {}
    res = walk_forward(_panel(), {"R": Recorder}, test_years=[2016], max_weight=1.0,
                       seq_len=1)
    train, val = Recorder.seen["train"], Recorder.seen["val"]
    assert train.min() >= pd.Timestamp(config.TRAIN_START)
    # last label of train/val is realized one week later, still inside its own split
    assert (train + pd.Timedelta(weeks=1)).max() <= pd.Timestamp("2014-12-31")
    assert val.min().year == 2015 and (val + pd.Timedelta(weeks=1)).max() <= \
        pd.Timestamp("2015-12-31")
    assert (res.returns.index.year == 2016).all()


def test_context_contains_only_the_past():
    Recorder.seen = {}
    walk_forward(_panel(), {"R": Recorder}, test_years=[2016], max_weight=1.0, seq_len=1)
    for ctx in Recorder.seen["ctx"]:
        assert ctx.trailing_returns.index.max() < ctx.week
        assert ctx.past_excess.index.max() < ctx.week
        assert len(ctx.trailing_returns) <= config.COV_WINDOW


def test_last_week_without_forward_return_is_skipped():
    res = walk_forward(_panel(), {"F": Fixed}, test_years=[2016], max_weight=1.0, seq_len=1)
    assert res.returns["F"].notna().all()


def test_invariant_violation_raises():
    class Bad(Fixed):
        def weights(self, ctx):
            return pd.Series([0.8, 0.2], index=ctx.tickers)

    with pytest.raises(AssertionError):
        walk_forward(_panel(), {"B": Bad}, test_years=[2016], max_weight=0.5, seq_len=1)


def test_make_splits_matches_walk_forward_and_returns_fitted_strategies():
    from sdpo.backtest import make_splits

    Recorder.seen = {}
    res = walk_forward(_panel(), {"R": Recorder}, test_years=[2016], max_weight=1.0,
                       seq_len=1)
    train, val = make_splits(_panel(), 2016, seq_len=1)
    assert train.weeks.equals(Recorder.seen["train"]) and val.weeks.equals(Recorder.seen["val"])
    assert isinstance(res.strategies[(2016, "R")], Recorder)
