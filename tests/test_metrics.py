import numpy as np
import pandas as pd
import pytest

from sdpo.metrics import perf_metrics, rank_ic, sharpe_ci


def test_max_drawdown_hand_series():
    # wealth: 1.1, 0.55, 0.66 -> worst peak-to-trough = 0.55/1.1 - 1
    assert perf_metrics(pd.Series([0.1, -0.5, 0.2]))["max_dd"] == pytest.approx(-0.5)


def test_sharpe_constant_mean():
    r = pd.Series([0.01, 0.03] * 50)
    assert perf_metrics(r)["sharpe"] == pytest.approx(0.02 / r.std() * np.sqrt(52))


def test_cagr_and_turnover():
    r = pd.Series([0.01] * 52)
    m = perf_metrics(r, turnover=pd.Series([0.2] * 52))
    assert m["cagr"] == pytest.approx(1.01 ** 52 - 1)
    assert m["avg_turnover"] == pytest.approx(0.2)


def test_sharpe_ci_brackets_point_estimate():
    r = pd.Series(np.random.default_rng(0).normal(0.003, 0.02, 260))
    lo, hi = sharpe_ci(r)
    s = perf_metrics(r)["sharpe"]
    assert lo < s < hi


def test_sharpe_ci_is_reproducible():
    r = pd.Series(np.random.default_rng(1).normal(0.002, 0.02, 200))
    assert sharpe_ci(r) == sharpe_ci(r)


def test_rank_ic_perfect_and_inverse():
    a = np.arange(10.0)[None, :]
    assert rank_ic(a, a)[0] == pytest.approx(1.0)
    assert rank_ic(a, -a)[0] == pytest.approx(-1.0)


def test_rank_ic_skips_nan_pairs():
    pred = np.array([[1.0, 2.0, np.nan, 4.0]])
    actual = np.array([[10.0, 20.0, 30.0, 40.0]])
    assert rank_ic(pred, actual)[0] == pytest.approx(1.0)
