import numpy as np
import pandas as pd

from sdpo.data import prices as prices_mod


def _fake_download(fail_first: set[str]):
    calls = []

    def download(tickers, **kwargs):
        calls.append(list(tickers))
        idx = pd.bdate_range("2010-01-04", periods=3)
        data = {t: (np.nan if (t in fail_first and len(calls) == 1) else 1.0) for t in tickers}
        close = pd.DataFrame({t: [v] * 3 for t, v in data.items()}, index=idx)
        return pd.concat({"Close": close}, axis=1)

    return download, calls


def test_transient_per_ticker_failure_is_retried(monkeypatch):
    fake, calls = _fake_download(fail_first={"AAPL"})
    monkeypatch.setattr(prices_mod.yf, "download", fake)
    monkeypatch.setattr(prices_mod.time, "sleep", lambda s: None)
    out = prices_mod.download_prices(["AAPL", "MSFT"], "2010-01-01", "2010-02-01")
    assert list(out.columns) == ["AAPL", "MSFT"] and out.notna().all().all()
    assert calls[1] == ["AAPL"]  # only the failed ticker is re-requested


def test_permanently_missing_ticker_dropped(monkeypatch):
    def download(tickers, **kwargs):
        idx = pd.bdate_range("2010-01-04", periods=3)
        close = pd.DataFrame({t: [np.nan if t == "GONE" else 1.0] * 3 for t in tickers}, index=idx)
        return pd.concat({"Close": close}, axis=1)

    monkeypatch.setattr(prices_mod.yf, "download", download)
    monkeypatch.setattr(prices_mod.time, "sleep", lambda s: None)
    out = prices_mod.download_prices(["GONE", "MSFT"], "2010-01-01", "2010-02-01")
    assert list(out.columns) == ["MSFT"]
