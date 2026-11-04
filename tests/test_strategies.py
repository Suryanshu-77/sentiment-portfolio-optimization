import dataclasses

import numpy as np
import pandas as pd
import pytest

from sdpo.strategies import BASELINES, BLNoViews, EqualWeight, SentimentTilt


@pytest.mark.parametrize("name", list(BASELINES))
def test_baseline_invariants(name, synthetic_ctx):
    w = BASELINES[name]().weights(synthetic_ctx)
    assert w is None or (w.ge(-1e-9).all() and abs(w.sum() - 1) < 1e-6
                         and w.max() <= 0.10 + 1e-6)
    if w is not None:
        assert list(w.index) == synthetic_ctx.tickers


def test_equal_weight():
    ctx = dataclasses.replace(_dummy_ctx(4), tickers=list("abcd"))
    np.testing.assert_allclose(EqualWeight().weights(ctx), 0.25)


def _dummy_ctx(n):
    from sdpo.backtest import WeekContext

    t = [f"T{i}" for i in range(n)]
    return WeekContext(pd.Timestamp("2015-01-02"), t, pd.DataFrame(index=t), np.zeros(0),
                       pd.DataFrame(), pd.DataFrame())


def test_tilt_overweights_high_sentiment(synthetic_ctx):
    feats = synthetic_ctx.features.copy()
    feats["sent_mean"] = np.linspace(-2, 2, len(feats))
    w = SentimentTilt().weights(dataclasses.replace(synthetic_ctx, features=feats))
    assert w.iloc[-1] > w.iloc[0]
    assert w.iloc[0] == 0  # 1 + 0.5 * (-2) = 0 -> clipped to zero weight


def test_bl_no_views_is_close_to_equal_weight(synthetic_ctx):
    # with an equal-weight prior and no views, max-Sharpe on pi = delta*Sigma*w_eq
    # recovers w_eq when the cap does not bind
    w = BLNoViews().weights(synthetic_ctx)
    np.testing.assert_allclose(w.to_numpy(), 1 / 30, atol=0.02)
