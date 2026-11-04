"""Daily adjusted close prices from yfinance."""

import logging
import time

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

RETRIES = 3


def _close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Wide daily adjusted close (DatetimeIndex x ticker).

    yfinance can fail for single tickers inside a batch without raising (transient
    cookie/rate-limit errors), so empty tickers are retried before being dropped.
    """
    close = _close(tickers, start, end)
    for attempt in range(1, RETRIES + 1):
        empty = [t for t in tickers if t not in close.columns or close[t].isna().all()]
        if not empty:
            break
        if attempt == RETRIES:
            log.warning("prices: no data after %d attempts for %s", RETRIES, empty)
            close = close.drop(columns=[t for t in empty if t in close.columns])
            break
        time.sleep(2 * attempt)
        retry = _close(empty, start, end)
        close = close.drop(columns=[t for t in empty if t in close.columns]).join(retry,
                                                                                  how="outer")
    close = close[[t for t in tickers if t in close.columns]]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    return close.sort_index()
