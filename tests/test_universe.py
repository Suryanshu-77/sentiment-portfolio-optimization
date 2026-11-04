import numpy as np
import pandas as pd
import pytest

from sdpo.data.universe import select_universe

WINDOW = ("2010-01-01", "2011-12-31")
FULL = ("2010-01-01", "2016-12-31")


def _news(counts: dict[str, list[tuple[str, int]]]) -> pd.DataFrame:
    rows = [(t, f"h{t}{d}{i}", pd.Timestamp(d)) for t, spec in counts.items()
            for d, n in spec for i in range(n)]
    return pd.DataFrame(rows, columns=["ticker", "headline", "date"])


def _prices(tickers, nan_ticker=None):
    idx = pd.bdate_range(*FULL)
    p = pd.DataFrame(100.0, index=idx, columns=tickers)
    if nan_ticker:
        p.loc["2016-03-01":"2016-03-10", nan_ticker] = np.nan
    return p


def test_ranks_only_inside_window():
    news = _news({"X": [("2015-05-01", 100), ("2010-05-01", 1)], "Y": [("2010-05-01", 5)]})
    assert select_universe(news, _prices(["X", "Y"]), n=1, window=WINDOW,
                           full_range=FULL) == ["Y"]


def test_excludes_incomplete_price_history():
    news = _news({"Z": [("2010-05-01", 50)], "A": [("2010-05-01", 3)],
                  "B": [("2010-05-01", 2)]})
    out = select_universe(news, _prices(["Z", "A", "B"], nan_ticker="Z"), n=2, window=WINDOW,
                          full_range=FULL)
    assert out == ["A", "B"]


def test_excludes_ticker_without_prices():
    news = _news({"GONE": [("2010-05-01", 50)], "A": [("2010-05-01", 3)]})
    assert select_universe(news, _prices(["A"]), n=1, window=WINDOW, full_range=FULL) == ["A"]


def test_raises_when_too_few_qualify():
    news = _news({"A": [("2010-05-01", 3)]})
    with pytest.raises(ValueError):
        select_universe(news, _prices(["A"]), n=50, window=WINDOW, full_range=FULL)


def test_excluded_tickers_are_skipped_and_backfilled():
    news = _news({"QQQ": [("2010-05-01", 90)], "A": [("2010-05-01", 3)],
                  "B": [("2010-05-01", 2)]})
    out = select_universe(news, _prices(["QQQ", "A", "B"]), n=2, window=WINDOW,
                          full_range=FULL, exclude={"QQQ"})
    assert out == ["A", "B"]
