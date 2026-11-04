import dataclasses

import numpy as np
import pandas as pd
from conftest import make_ctx

from sdpo import config
from sdpo.backtest import Split, walk_forward
from sdpo.features import make_sequences
from sdpo.metrics import perf_metrics
from sdpo.strategies import EqualWeight, StrategyA


def _fitted(panel, kind="ridge", n_train=200, n_val=52):
    X, ye, yr, weeks, tickers = make_sequences(panel)
    tr, va = slice(0, n_train), slice(n_train, n_train + n_val)
    split = lambda s: Split(X[s], ye[s], yr[s], weeks[s], tickers, panel)
    a = StrategyA(kind, device="cpu")
    a.fit(split(tr), split(va))
    return a, n_train + n_val


def test_omega_uses_only_realized_past(synthetic_panel):
    a, start = _fitted(synthetic_panel)
    for i in range(start, start + 30):  # accumulate 30 weeks of stored forecasts
        a.weights(make_ctx(synthetic_panel, i))
    ctx = make_ctx(synthetic_panel, start + 30)
    base = a.omega_diag(ctx)
    wide_ex = synthetic_panel["fwd_excess"].unstack("ticker")[ctx.tickers]
    leaky = dataclasses.replace(ctx, past_excess=wide_ex)  # includes weeks >= ctx.week
    np.testing.assert_array_equal(base, a.omega_diag(leaky))
    assert (base >= 1e-6).all()


def test_omega_falls_back_to_validation_errors_early(synthetic_panel):
    a, start = _fitted(synthetic_panel)
    a.weights(make_ctx(synthetic_panel, start))
    np.testing.assert_allclose(a.omega_diag(make_ctx(synthetic_panel, start + 1)),
                               np.maximum(a.val_sq_err, 1e-6))


def test_weights_satisfy_invariants(synthetic_panel):
    a, start = _fitted(synthetic_panel)
    w = a.weights(make_ctx(synthetic_panel, start))
    assert w is not None and abs(w.sum() - 1) < 1e-6 and w.max() <= config.MAX_WEIGHT + 1e-6


def test_save_load_roundtrip(synthetic_panel, tmp_path):
    a, start = _fitted(synthetic_panel, kind="lstm")
    a.save(tmp_path / "a.pt")
    b = StrategyA.load(tmp_path / "a.pt")
    ctx = make_ctx(synthetic_panel, start)
    pd.testing.assert_series_equal(a.weights(ctx), b.weights(ctx))


def test_planted_signal_end_to_end_beats_ew(synthetic_panel):
    factories = {"EW": EqualWeight,
                 "A-lstm": lambda: StrategyA("lstm", device="cpu"),
                 "A-ridge": lambda: StrategyA("ridge", device="cpu")}
    res = walk_forward(synthetic_panel, factories, test_years=[2016, 2017])
    s = {k: perf_metrics(res.returns[k])["sharpe"] for k in res.returns}
    assert s["A-lstm"] > s["EW"] and s["A-ridge"] > s["EW"], s
