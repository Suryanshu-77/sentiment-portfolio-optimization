import numpy as np
import pandas as pd
import pytest
from pypfopt import BlackLittermanModel

from sdpo.portfolio.black_litterman import bl_posterior, implied_prior
from sdpo.portfolio.optimize import cap_weights, max_sharpe_weights, shrunk_cov


@pytest.fixture
def bl_inputs():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 0.02, (5, 5))
    S = A @ A.T + np.eye(5) * 1e-4  # random positive-definite weekly covariance
    pi = implied_prior(S, np.full(5, 0.2))
    P = np.array([[1, 0, 0, 0, 0], [0, 1, -1, 0, 0]], dtype=float)  # absolute + relative view
    Q = np.array([0.004, -0.002])
    omega = np.diag([1e-4, 2e-4])
    return S, pi, P, Q, omega


def test_implied_prior_is_delta_sigma_w():
    S = np.array([[0.04, 0.01], [0.01, 0.09]])
    np.testing.assert_allclose(implied_prior(S, np.array([0.5, 0.5]), delta=2.0),
                               2.0 * S @ [0.5, 0.5])


def test_bl_matches_pypfopt(bl_inputs):
    S, pi, P, Q, omega = bl_inputs
    ours = bl_posterior(S, pi, P, Q, omega, tau=0.05)
    cols = list("abcde")
    ref = BlackLittermanModel(pd.DataFrame(S, cols, cols), pi=pd.Series(pi, cols), P=P, Q=Q,
                              omega=omega, tau=0.05).bl_returns()
    np.testing.assert_allclose(ours, ref.to_numpy(), rtol=1e-8)


def test_bl_huge_omega_returns_prior(bl_inputs):
    S, pi, P, Q, _ = bl_inputs
    out = bl_posterior(S, pi, P, Q, np.eye(2) * 1e12, tau=0.05)
    np.testing.assert_allclose(out, pi, atol=1e-8)


def test_bl_tiny_omega_returns_views(bl_inputs):
    S, pi, _, _, _ = bl_inputs
    Q = np.array([0.01, -0.01, 0.0, 0.005, 0.002])
    out = bl_posterior(S, pi, np.eye(5), Q, np.eye(5) * 1e-12, tau=0.05)
    np.testing.assert_allclose(out, Q, atol=1e-6)


def test_cap_weights():
    w = cap_weights(pd.Series([0.5, 0.3] + [0.2 / 28] * 28), 0.10)
    assert w.max() <= 0.10 + 1e-9 and w.sum() == pytest.approx(1) and (w >= 0).all()
    assert w.iloc[2:].nunique() == 1  # excess spread pro-rata over equal uncapped names


def _market(n=30, weeks=104, seed=0):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n)]
    rets = pd.DataFrame(rng.normal(0.002, 0.03, (weeks, n)), columns=tickers)
    return tickers, rets


def test_max_sharpe_respects_bounds():
    tickers, rets = _market()
    sigma = shrunk_cov(rets)
    mu = pd.Series(np.linspace(0.001, 0.006, 30), index=tickers)
    w = max_sharpe_weights(mu, sigma)
    assert w is not None and list(w.index) == tickers
    assert w.sum() == pytest.approx(1, abs=1e-6) and (w >= -1e-9).all()
    assert w.max() <= 0.10 + 1e-6


def test_max_sharpe_binding_cap_is_never_exceeded():
    # several names pinned at the cap: rounding + renormalizing must not push them over
    # diagonal covariance -> w proportional to mu: 9 names at the cap, 21 share 0.1 equally
    # (0.0047619...), which 5-decimal rounding truncates
    tickers, _ = _market()
    sigma = pd.DataFrame(np.eye(30) * 1e-3, tickers, tickers)
    mu = pd.Series([0.05] * 9 + [0.001] * 21, index=tickers)
    w = max_sharpe_weights(mu, sigma)
    assert (w > 0.0999).sum() == 9
    assert w.max() <= 0.10 + 1e-9 and w.sum() == pytest.approx(1, abs=1e-9)


def test_max_sharpe_all_negative_returns_none():
    tickers, rets = _market()
    assert max_sharpe_weights(pd.Series(-0.01, index=tickers), shrunk_cov(rets)) is None


def test_shrunk_cov_is_psd_and_labelled():
    tickers, rets = _market()
    S = shrunk_cov(rets)
    assert list(S.index) == tickers and np.linalg.eigvalsh(S.to_numpy()).min() > 0
