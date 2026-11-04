from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app" / "dashboard.py")


def test_dashboard_renders_with_artifacts(fake_artifacts, monkeypatch):
    monkeypatch.setenv("SDPO_ARTIFACTS", str(fake_artifacts))
    at = AppTest.from_file(APP).run(timeout=60)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Backtest", "Sentiment", "NLP models", "Live"]
    assert len(at.dataframe) >= 2  # metrics table + NLP table


def test_dashboard_without_artifacts_shows_info(tmp_path, monkeypatch):
    monkeypatch.setenv("SDPO_ARTIFACTS", str(tmp_path / "missing"))
    at = AppTest.from_file(APP).run(timeout=60)
    assert not at.exception and any("scripts" in i.value for i in at.info)


def test_changing_ticker_rerenders(fake_artifacts, monkeypatch):
    monkeypatch.setenv("SDPO_ARTIFACTS", str(fake_artifacts))
    at = AppTest.from_file(APP).run(timeout=60)
    at.selectbox(key="ticker").select("MSFT").run(timeout=60)
    assert not at.exception
