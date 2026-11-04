import pandas as pd
import pytest

from sdpo.data.news import load_kaggle_news


def _csv(tmp_path, rows, columns=("headline", "date", "stock"), name="news.csv"):
    path = tmp_path / name
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)
    return path


def test_parses_mixed_date_formats(tmp_path):
    csv = _csv(tmp_path, [("Up big", "2020-06-05 10:30:54-04:00", "AAPL"),
                          ("Down", "2011-04-27 00:00:00", "MSFT")])
    df = load_kaggle_news([csv])
    assert df["date"].tolist() == [pd.Timestamp("2011-04-27"), pd.Timestamp("2020-06-05")]
    assert df["ticker"].tolist() == ["MSFT", "AAPL"]


def test_late_utc_timestamp_maps_to_eastern_date(tmp_path):
    csv = _csv(tmp_path, [("Late news", "2020-06-06 02:00:00+00:00", "AAPL")])
    assert load_kaggle_news([csv])["date"].iloc[0] == pd.Timestamp("2020-06-05")


def test_drops_bad_rows_and_duplicates(tmp_path):
    csv = _csv(tmp_path, [
        ("Valid", "2012-01-03", "IBM"),
        ("", "2012-01-03", "IBM"),
        ("Bad date", "n/a", "IBM"),
        ("No ticker", "2012-01-03", None),
        ("Valid", "2012-01-03", "IBM"),
    ])
    assert len(load_kaggle_news([csv])) == 1


def test_missing_column_raises(tmp_path):
    csv = _csv(tmp_path, [("x", "2012-01-03")], columns=("headline", "date"))
    with pytest.raises(ValueError, match="stock"):
        load_kaggle_news([csv])


def test_extra_columns_ignored_and_files_combined(tmp_path):
    a = _csv(tmp_path, [("A", "2012-01-03", "IBM", "u")],
             columns=("headline", "date", "stock", "url"), name="a.csv")
    b = _csv(tmp_path, [("B", "2012-01-04", "IBM")], name="b.csv")
    df = load_kaggle_news([a, b])
    assert list(df.columns) == ["ticker", "headline", "date"] and len(df) == 2
