"""News loading: the Kaggle historical archive (and, later, the live Finnhub client)."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

REQUIRED = ("headline", "date", "stock")
EASTERN = "US/Eastern"
_TZ_SUFFIX = r"(?:[+-]\d{2}:?\d{2}|Z)$"


def _to_eastern_date(raw: pd.Series) -> pd.Series:
    """Parse mixed naive / tz-aware strings to a naive US/Eastern calendar date (NaT if bad)."""
    s = raw.astype("string").str.strip()
    aware = s.str.contains(_TZ_SUFFIX, regex=True, na=False)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if aware.any():
        parsed = pd.to_datetime(s[aware], utc=True, errors="coerce", format="mixed")
        out[aware] = parsed.dt.tz_convert(EASTERN).dt.tz_localize(None)
    if (~aware).any():
        # Naive timestamps in the archive are already US/Eastern.
        out[~aware] = pd.to_datetime(s[~aware], errors="coerce", format="mixed")
    return out.dt.normalize()


def load_kaggle_news(paths: list[Path]) -> pd.DataFrame:
    """Load and validate Kaggle headline CSVs into (ticker, headline, date), sorted by date."""
    frames = []
    for path in paths:
        df = pd.read_csv(path, usecols=lambda c: c in REQUIRED)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing required column(s) {missing}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)

    df = df.rename(columns={"stock": "ticker"})
    df["headline"] = df["headline"].astype("string").str.strip()
    df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
    df["date"] = _to_eastern_date(df["date"])

    bad = df["headline"].isna() | (df["headline"] == "") | df["ticker"].isna() | (
        df["ticker"] == "") | df["date"].isna()
    df = df[~bad]
    n_bad = int(bad.sum())
    df = df.drop_duplicates(["ticker", "headline", "date"])
    n_dup = n0 - n_bad - len(df)
    log.info("news: %d rows read, %d invalid dropped, %d duplicates dropped", n0, n_bad, n_dup)

    df = df[["ticker", "headline", "date"]].astype({"ticker": str, "headline": str})
    return df.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


class FinnhubError(Exception):
    """Finnhub request failed (HTTP error, rate limit, or malformed payload)."""


FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
NEWS_COLUMNS = ["ticker", "headline", "date"]


def fetch_finnhub_news(ticker: str, start: date, end: date, api_key: str,
                       session: requests.Session | None = None) -> pd.DataFrame:
    """Company news for one ticker as (ticker, headline, date[US/Eastern calendar day]).

    The key goes in the X-Finnhub-Token header, never the URL, so it stays out of logs."""
    http = session or requests
    resp = http.get(FINNHUB_URL, params={"symbol": ticker, "from": start.isoformat(),
                                         "to": end.isoformat()},
                    headers={"X-Finnhub-Token": api_key}, timeout=10)
    if resp.status_code != 200:
        raise FinnhubError(f"{ticker}: HTTP {resp.status_code}")
    try:
        items = resp.json()
    except ValueError as e:
        raise FinnhubError(f"{ticker}: invalid JSON") from e
    if not isinstance(items, list):
        raise FinnhubError(f"{ticker}: unexpected payload {str(items)[:100]}")
    rows = [(ticker, it.get("headline", ""), it.get("datetime")) for it in items
            if it.get("headline") and it.get("datetime")]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                             zip(NEWS_COLUMNS, ["str", "str", "datetime64[ns]"], strict=True)})
    df = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    df["date"] = (pd.to_datetime(df["date"], unit="s", utc=True).dt.tz_convert(EASTERN)
                  .dt.tz_localize(None).dt.normalize())
    return df
