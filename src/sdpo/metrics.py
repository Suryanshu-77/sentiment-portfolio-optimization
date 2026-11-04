"""Performance metrics on weekly returns (annualized with 52, rf = 0) and forecast IC."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PERIODS = 52


def _sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(PERIODS)) if sd > 0 else 0.0


def perf_metrics(r: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = r.dropna()
    x = r.to_numpy()
    wealth = np.cumprod(1 + x)
    years = len(x) / PERIODS
    cagr = float(wealth[-1] ** (1 / years) - 1) if len(x) else 0.0
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    max_dd = float((wealth / peak - 1).min()) if len(x) else 0.0
    downside = np.sqrt(np.mean(np.minimum(x, 0) ** 2)) if len(x) else 0.0
    return {
        "cagr": cagr,
        "vol": float(x.std(ddof=1) * np.sqrt(PERIODS)),
        "sharpe": _sharpe(x),
        "sortino": float(x.mean() / downside * np.sqrt(PERIODS)) if downside > 0 else 0.0,
        "max_dd": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else 0.0,
        "avg_turnover": float(turnover.mean()) if turnover is not None else float("nan"),
    }


def sharpe_ci(r: pd.Series, block: int = 4, n_boot: int = 2000, seed: int = 42,
              alpha: float = 0.05) -> tuple[float, float]:
    """Circular block bootstrap CI for the annualized Sharpe ratio. Blocks keep short-range
    autocorrelation that an i.i.d. bootstrap would destroy."""
    x = r.dropna().to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n] % n
    sims = x[idx]
    sd = sims.std(axis=1, ddof=1)
    sharpes = np.where(sd > 0, sims.mean(axis=1) / np.where(sd > 0, sd, 1) * np.sqrt(PERIODS),
                       0.0)
    lo, hi = np.quantile(sharpes, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def rank_ic(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Per-row Spearman correlation, ignoring NaN pairs (NaN if fewer than 3 valid pairs)."""
    out = np.full(pred.shape[0], np.nan)
    for i, (p, a) in enumerate(zip(pred, actual, strict=True)):
        ok = ~(np.isnan(p) | np.isnan(a))
        if ok.sum() >= 3:
            out[i] = spearmanr(p[ok], a[ok]).statistic
    return out
