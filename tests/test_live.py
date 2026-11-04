import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_market
from streamlit.testing.v1 import AppTest

from sdpo.backtest import make_splits
from sdpo.data.news import FinnhubError, fetch_finnhub_news
from sdpo.features import build_weekly_panel
from sdpo.live import suggest_weights
from sdpo.strategies import StrategyA

APP = str(Path(__file__).parents[1] / "app" / "dashboard.py")
D0, D1 = dt.date(2026, 10, 1), dt.date(2026, 10, 29)


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, status, payload):
        self.status, self.payload, self.calls = status, payload, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return FakeResponse(self.status, self.payload)


def test_parses_articles_and_keeps_key_out_of_url():
    s = FakeSession(200, [{"datetime": 1793000000, "headline": "Shares jump", "id": 1}])
    df = fetch_finnhub_news("AAPL", D0, D1, "secret", session=s)
    assert list(df.columns) == ["ticker", "headline", "date"] and len(df) == 1
    assert df["ticker"].iloc[0] == "AAPL"
    assert s.calls[0]["headers"] == {"X-Finnhub-Token": "secret"}
    assert "secret" not in str(s.calls[0]["params"]) + s.calls[0]["url"]


def test_empty_news_is_empty_frame():
    df = fetch_finnhub_news("AAPL", D0, D1, "k", session=FakeSession(200, []))
    assert df.empty and list(df.columns) == ["ticker", "headline", "date"]


@pytest.mark.parametrize("status,payload", [(429, {}), (200, {"error": "bad"}), (500, None)])
def test_http_or_payload_error_raises(status, payload):
    with pytest.raises(FinnhubError, match="AAPL"):
        fetch_finnhub_news("AAPL", D0, D1, "k", session=FakeSession(status, payload))


@pytest.fixture
def tmp_model(tmp_path):
    scores, prices = synthetic_market(signal=0.5)
    a = StrategyA("ridge", device="cpu")
    a.fit(*make_splits(build_weekly_panel(scores, prices), 2016))
    a.save(tmp_path / "A.pt")
    return tmp_path / "A.pt"


def test_suggest_weights_with_no_news_ticker(tmp_model):
    scores, prices = synthetic_market(signal=0.5)
    recent = scores[scores["date"] > prices.index[-20]]
    news = recent[recent["ticker"] != "T05"].assign(headline="x")[["ticker", "headline", "date"]]
    w = suggest_weights(news, prices, lambda texts: np.zeros(len(texts)), tmp_model)
    assert w is not None and set(w.index) == set(prices.columns)
    assert abs(w.sum() - 1) < 1e-6 and w.max() <= 0.10 + 1e-6 and (w >= 0).all()


def test_live_tab_without_key_warns(fake_artifacts, monkeypatch):
    monkeypatch.setenv("SDPO_ARTIFACTS", str(fake_artifacts))
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    at = AppTest.from_file(APP).run(timeout=60)
    assert not at.exception
    assert any("FINNHUB_API_KEY" in w.value for w in at.warning)


def test_suggest_weights_with_no_news_at_all(tmp_model):
    from sdpo.data.news import NEWS_COLUMNS

    _, prices = synthetic_market(signal=0.5)
    empty = pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                          zip(NEWS_COLUMNS, ["str", "str", "datetime64[ns]"], strict=True)})
    w = suggest_weights(empty, prices, lambda texts: np.zeros(len(texts)), tmp_model)
    assert abs(w.sum() - 1) < 1e-6 and w.max() <= 0.10 + 1e-6


def test_suggest_weights_when_a_ticker_is_dropped(tmp_model):
    # the live tab drops tickers without recent prices (delisted, yfinance glitch); the model
    # was trained on all 30, so its per-ticker validation errors must be matched by name
    scores, prices = synthetic_market(signal=0.5)
    recent = scores[scores["date"] > prices.index[-20]]
    news = recent[recent["ticker"] != "T05"].assign(headline="x")[["ticker", "headline", "date"]]
    w = suggest_weights(news, prices.drop(columns=["T05"]), lambda texts: np.zeros(len(texts)),
                        tmp_model)
    assert "T05" not in w.index and len(w) == 29 and abs(w.sum() - 1) < 1e-6
