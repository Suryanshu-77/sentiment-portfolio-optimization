"""Covariance estimation and constrained long-only max-Sharpe optimization."""

import logging

import numpy as np
import pandas as pd
from cvxpy.error import SolverError
from pypfopt import CovarianceShrinkage, EfficientFrontier
from pypfopt.exceptions import OptimizationError

log = logging.getLogger(__name__)


def shrunk_cov(weekly_returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance of weekly returns (weekly units, not annualized)."""
    return CovarianceShrinkage(weekly_returns, returns_data=True, frequency=1).ledoit_wolf()


def cap_weights(w: pd.Series, max_w: float) -> pd.Series:
    """Clip weights at max_w and hand the excess pro-rata to uncapped names, repeating until
    nothing exceeds the cap. Assumes len(w) * max_w >= 1 (true for N=30, cap 10%)."""
    w = w.clip(lower=0) / w.clip(lower=0).sum()
    for _ in range(len(w)):
        over = w > max_w + 1e-12
        if not over.any():
            break
        excess = (w[over] - max_w).sum()
        w[over] = max_w
        free = w < max_w - 1e-12
        w[free] += excess * w[free] / w[free].sum()
    return w


def max_sharpe_weights(mu: pd.Series, sigma: pd.DataFrame,
                       max_w: float = 0.10) -> pd.Series | None:
    """Long-only max-Sharpe (rf = 0) with per-name cap. Returns None if the problem has no
    solution (e.g. every expected return <= 0) so the caller can fall back."""
    try:
        ef = EfficientFrontier(mu, sigma, weight_bounds=(0, max_w))
        ef.max_sharpe(risk_free_rate=0.0)
        w = pd.Series(ef.clean_weights(), dtype=float).reindex(mu.index).fillna(0.0)
    except (ValueError, OptimizationError, SolverError) as e:  # all mean "no portfolio"
        log.warning("max_sharpe failed: %s", e)
        return None
    if not np.isfinite(w).all() or w.sum() <= 0:
        return None
    return w / w.sum()
