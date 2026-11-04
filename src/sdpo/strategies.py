"""Portfolio strategies. Each implements sdpo.backtest.Strategy: fit once per test year,
then weights(ctx) every week (None = no solution, the engine keeps last week's weights)."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from sdpo import config
from sdpo.backtest import Split, Strategy, WeekContext
from sdpo.models.forecaster import LSTMForecaster, RidgeForecaster, predict, train_lstm
from sdpo.portfolio.black_litterman import bl_posterior, implied_prior
from sdpo.portfolio.optimize import cap_weights, max_sharpe_weights, shrunk_cov


class _NoFit:
    def fit(self, train: Split, val: Split) -> None:
        pass


class EqualWeight(_NoFit):
    def weights(self, ctx: WeekContext) -> pd.Series:
        return pd.Series(1 / len(ctx.tickers), index=ctx.tickers)


class MeanVariance(_NoFit):
    """Classic max-Sharpe on trailing-52w mean returns and Ledoit-Wolf covariance."""

    def weights(self, ctx: WeekContext) -> pd.Series | None:
        mu = ctx.trailing_returns.mean()
        return max_sharpe_weights(mu, shrunk_cov(ctx.trailing_returns), config.MAX_WEIGHT)


class BLNoViews(_NoFit):
    """Black-Litterman with no views: the equal-weight equilibrium prior, then max-Sharpe."""

    def weights(self, ctx: WeekContext) -> pd.Series | None:
        sigma = shrunk_cov(ctx.trailing_returns)
        n = len(ctx.tickers)
        pi = pd.Series(implied_prior(sigma.to_numpy(), np.full(n, 1 / n)), index=ctx.tickers)
        return max_sharpe_weights(pi, sigma, config.MAX_WEIGHT)


class SentimentTilt(_NoFit):
    """Strategy B: w_i proportional to max(0, 1 + k * z(sent_mean)_i), capped at 10%."""

    def __init__(self, k: float = config.TILT_K):
        self.k = k

    def weights(self, ctx: WeekContext) -> pd.Series:
        raw = (1 + self.k * ctx.features["sent_mean"]).clip(lower=0)
        if raw.sum() <= 0:
            return pd.Series(1 / len(ctx.tickers), index=ctx.tickers)
        return cap_weights(raw / raw.sum(), config.MAX_WEIGHT)


BASELINES: dict[str, Callable[[], Strategy]] = {
    "EW": EqualWeight,
    "MVO": MeanVariance,
    "BL-no-views": BLNoViews,
    "B-tilt": SentimentTilt,
}


class StrategyA:
    """Strategy A: forecast next-week excess returns, use them as Black-Litterman views whose
    uncertainty is each ticker's recent out-of-sample forecast error, then max-Sharpe."""

    def __init__(self, forecaster: Literal["lstm", "ridge"] = "lstm",
                 device: str | None = None):
        self.kind, self.device = forecaster, device
        self.model = None
        self.val_sq_err: pd.Series | None = None  # by ticker: live may drop names
        self.preds: dict[pd.Timestamp, np.ndarray] = {}

    def fit(self, train: Split, val: Split) -> None:
        if self.kind == "lstm":
            self.model = train_lstm(train.X, train.y_excess, val.X, val.y_excess,
                                    device=self.device)
        else:
            self.model = RidgeForecaster().fit(train.X, train.y_excess)
        err = (predict(self.model, val.X) - val.y_excess) ** 2
        self.val_sq_err = pd.Series(np.nanmean(err, axis=0), index=val.tickers)

    def omega_diag(self, ctx: WeekContext) -> np.ndarray:
        """Mean squared forecast error per ticker over the last OMEGA_WINDOW weeks whose
        outcome is already realized (weeks < ctx.week); validation errors until then."""
        realized = ctx.past_excess[ctx.past_excess.index < ctx.week]
        weeks = [w for w in sorted(self.preds) if w < ctx.week and w in realized.index]
        weeks = weeks[-config.OMEGA_WINDOW:]
        if len(weeks) < config.OMEGA_WINDOW:
            # Match by name: tickers unseen in validation get the median error.
            v = self.val_sq_err
            err = v.reindex(ctx.tickers).fillna(v.median()).to_numpy()
        else:
            pred = np.stack([self.preds[w] for w in weeks])
            err = np.nanmean((pred - realized.loc[weeks, ctx.tickers].to_numpy()) ** 2, axis=0)
        return np.maximum(np.nan_to_num(err, nan=1.0), 1e-6)

    def weights(self, ctx: WeekContext) -> pd.Series | None:
        forecast = predict(self.model, ctx.X_seq)
        self.preds[ctx.week] = forecast
        sigma = shrunk_cov(ctx.trailing_returns)
        S = sigma.to_numpy()
        n = len(ctx.tickers)
        pi = implied_prior(S, np.full(n, 1 / n))
        # The model forecasts *excess* (relative) return, so each view is the stock's own prior
        # plus that excess: a zero forecast reproduces the prior exactly (equal weight), and any
        # tilt comes from the forecasts. (Anchoring at mean(pi) instead built in a low-vol tilt.)
        Q = pi + forecast
        mu = bl_posterior(S, pi, np.eye(n), Q, np.diag(self.omega_diag(ctx)))
        return max_sharpe_weights(pd.Series(mu, index=ctx.tickers), sigma, config.MAX_WEIGHT)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        model = self.model.state_dict() if self.kind == "lstm" else self.model
        torch.save({"kind": self.kind, "model": model, "val_sq_err": self.val_sq_err}, path)

    @classmethod
    def load(cls, path: Path) -> "StrategyA":
        # weights_only=False: the ridge variant pickles a sklearn object; local artifact only.
        blob = torch.load(path, weights_only=False)
        a = cls(blob["kind"], device="cpu")
        if a.kind == "lstm":
            a.model = LSTMForecaster()
            a.model.load_state_dict(blob["model"])
            a.model.eval()
        else:
            a.model = blob["model"]
        a.val_sq_err = blob["val_sq_err"]
        return a


FACTORIES_A: dict[str, Callable[[], Strategy]] = {
    "A-lstm": lambda: StrategyA("lstm"),
    "A-ridge": lambda: StrategyA("ridge"),
}
