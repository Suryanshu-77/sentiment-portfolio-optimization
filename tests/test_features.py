import numpy as np
import pandas as pd

from sdpo import config
from sdpo.features import build_weekly_panel, make_sequences, week_ends


def test_no_lookahead_property(synthetic):
    scores, prices = synthetic
    t = week_ends(prices.index)[200]
    base = build_weekly_panel(scores, prices).loc[:t, config.FEATURES]
    # Perturb each ticker differently: a uniform change (e.g. all prices x3) is invisible to
    # cross-sectional z-scores, so it would let a leak through.
    rng = np.random.default_rng(7)
    s2 = scores.copy()
    later = s2.date > t
    s2.loc[later, "score"] = rng.uniform(-1, 1, later.sum())
    p2 = prices.copy()
    p2.loc[p2.index > t] *= rng.uniform(0.3, 3.0, prices.shape[1])
    after = build_weekly_panel(s2, p2).loc[:t, config.FEATURES]
    pd.testing.assert_frame_equal(base, after)


def _flat_prices(start="2012-01-02", end="2012-06-29", tickers=("A", "B", "C")):
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(1)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), len(tickers))),
                                               axis=0)), index=idx, columns=list(tickers))


def test_headline_usable_only_from_next_trading_day():
    prices = _flat_prices()
    scores = pd.DataFrame({"ticker": ["A"], "date": [pd.Timestamp("2012-03-09")],
                           "score": [0.9]})  # a Friday
    panel = build_weekly_panel(scores, prices)
    a = panel.xs("A", level="ticker")["log_news"]
    assert a.loc["2012-03-09"] == a.loc["2012-03-02"]  # nothing new in its own week
    assert a.loc["2012-03-16"] > a.loc["2012-03-09"]  # first visible the week after


def test_single_or_zero_headlines_give_no_nan():
    prices = _flat_prices()
    scores = pd.DataFrame({"ticker": ["A", "A", "B"],
                           "date": pd.to_datetime(["2012-02-07", "2012-03-06", "2012-03-06"]),
                           "score": [0.5, -0.2, 0.3]})  # C never has news
    panel = build_weekly_panel(scores, prices)
    assert not panel[config.FEATURES].isna().any().any()


def test_good_friday_week_uses_thursday():
    days = pd.bdate_range("2012-04-02", "2012-04-13").drop(pd.Timestamp("2012-04-06"))
    ends = week_ends(days)
    assert pd.Timestamp("2012-04-05") in ends and pd.Timestamp("2012-04-13") in ends
    assert len(ends) == 2


def test_features_are_cross_sectionally_standardized(synthetic):
    panel = build_weekly_panel(*synthetic)
    m = panel.groupby(level="week")[config.FEATURES].mean()
    assert np.allclose(m.values, 0, atol=1e-9)


def test_forward_returns_and_excess(synthetic):
    scores, prices = synthetic
    panel = build_weekly_panel(scores, prices)
    ends = week_ends(prices.index)
    w0, w1 = panel.index.get_level_values("week").unique()[[10, 11]]
    expected = prices.loc[ends[ends.get_loc(w1)], "T00"] / prices.loc[w0, "T00"] - 1
    assert np.isclose(panel.loc[(w0, "T00"), "fwd_ret"], expected)
    assert np.isclose(panel.loc[w0, "fwd_excess"].mean(), 0, atol=1e-12)
    last = panel.index.get_level_values("week").max()
    assert panel.loc[last, "fwd_ret"].isna().all()


def test_make_sequences_shapes(synthetic):
    panel = build_weekly_panel(*synthetic)
    X, ye, yr, weeks, tickers = make_sequences(panel)
    assert X.shape[1:] == (30, 12, 7) and ye.shape == X.shape[:2] == yr.shape
    assert len(weeks) == X.shape[0] and tickers == sorted(tickers)
    # the last step of each sequence is that week's own features
    w = weeks[5]
    np.testing.assert_allclose(X[5, :, -1, :], panel.loc[w, config.FEATURES].to_numpy())
