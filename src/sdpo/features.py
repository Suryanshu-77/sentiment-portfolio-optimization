"""Weekly per-ticker feature panel, built so that features at week t use only data known at
the close of week t. Forward returns (the labels) are the only columns that look ahead."""

import numpy as np
import pandas as pd

from sdpo import config

VOL_DAYS = 20


def week_ends(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each W-FRI week (Thursday when Friday is a holiday)."""
    days = pd.Series(trading_days, index=trading_days)
    return pd.DatetimeIndex(days.groupby(trading_days.to_period("W-FRI")).max().to_numpy())


def _news_week(dates: pd.Series, days: pd.DatetimeIndex, ends: pd.DatetimeIndex) -> pd.Series:
    """Week-end at which a headline dated d is first usable: its effective date is the first
    trading day strictly after d, assigned to the first week-end on or after that day."""
    pos = days.searchsorted(dates.to_numpy(), side="right")
    ok = pos < len(days)
    eff = days[pos[ok]]
    week = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    week[ok] = ends[ends.searchsorted(eff, side="left")]
    return week


def _zscore(wide: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per row; a row with zero dispersion becomes all zeros."""
    mu = wide.mean(axis=1)
    sd = wide.std(axis=1, ddof=0)
    # Identical values give sd ~1e-16 from rounding, not 0; dividing by it would turn noise
    # into +-1. The mask is applied per row explicitly (a bare Series would align on columns).
    ok = (sd > 1e-12 * (1 + mu.abs())).to_numpy()[:, None]
    z = wide.sub(mu, axis=0).div(sd, axis=0)
    return z.where(np.broadcast_to(ok, z.shape), 0.0)


def weekly_sentiment_raw(scores: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Un-standardized weekly sentiment on the full (week, ticker) grid: sent_mean,
    sent_std (ddof=0), news_count. Weeks without news are 0, never NaN."""
    days = pd.DatetimeIndex(prices.index)
    ends = week_ends(days)
    tickers = sorted(prices.columns)
    s = scores[scores["ticker"].isin(tickers)].copy()
    s["week"] = _news_week(s["date"], days, ends)
    agg = s.dropna(subset=["week"]).groupby(["week", "ticker"])["score"].agg(
        sent_mean="mean", sent_std=lambda x: x.std(ddof=0), news_count="size")
    grid = pd.MultiIndex.from_product([ends, tickers], names=["week", "ticker"])
    return agg.reindex(grid).fillna(0.0).astype(float)


def build_weekly_panel(scores: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    days = pd.DatetimeIndex(prices.index)
    ends = week_ends(days)
    tickers = sorted(prices.columns)
    prices = prices[tickers]
    sent = weekly_sentiment_raw(scores, prices)

    def wide(col: str) -> pd.DataFrame:
        return sent[col].unstack("ticker").reindex(index=ends, columns=tickers)

    sent_mean = wide("sent_mean")
    pw = prices.loc[ends]
    daily = prices.pct_change()
    raw = {
        "sent_mean": sent_mean,
        "sent_std": wide("sent_std"),
        "log_news": np.log1p(wide("news_count")),
        "sent_change": sent_mean - sent_mean.shift(1).rolling(4).mean(),
        "ret_1w": pw.pct_change(fill_method=None),
        "ret_4w": pw.pct_change(4, fill_method=None),
        "vol_4w": daily.rolling(VOL_DAYS).std().loc[ends],
    }
    # Drop the warm-up weeks at the start where any feature is still undefined.
    ready = pd.concat([raw[f].notna().all(axis=1) for f in config.FEATURES], axis=1).all(axis=1)
    first = ready.idxmax() if ready.any() else ends[-1]
    keep = ends[ends >= first]

    feats = {f: _zscore(raw[f].loc[keep]) for f in config.FEATURES}
    fwd_ret = (pw.shift(-1) / pw - 1).loc[keep]
    feats["fwd_ret"] = fwd_ret
    feats["fwd_excess"] = fwd_ret.sub(fwd_ret.mean(axis=1), axis=0)

    panel = pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1)
    panel.index = panel.index.set_names(["week", "ticker"])
    return panel


def make_sequences(panel: pd.DataFrame, seq_len: int = config.SEQ_LEN):
    """X [T,N,L,F], y_excess [T,N], y_ret [T,N], weeks [T], tickers [N].
    The sequence for week t ends at t inclusive; the first seq_len-1 weeks are dropped."""
    weeks = panel.index.get_level_values("week").unique().sort_values()
    tickers = sorted(panel.index.get_level_values("ticker").unique())
    full = panel.reindex(pd.MultiIndex.from_product([weeks, tickers],
                                                    names=["week", "ticker"]))
    W, N, F = len(weeks), len(tickers), len(config.FEATURES)
    feats = full[config.FEATURES].to_numpy().reshape(W, N, F)
    win = np.lib.stride_tricks.sliding_window_view(feats, seq_len, axis=0)  # [T,N,F,L]
    X = np.ascontiguousarray(win.transpose(0, 1, 3, 2))
    y_excess = full["fwd_excess"].to_numpy().reshape(W, N)[seq_len - 1:]
    y_ret = full["fwd_ret"].to_numpy().reshape(W, N)[seq_len - 1:]
    return X, y_excess, y_ret, weeks[seq_len - 1:], tickers
