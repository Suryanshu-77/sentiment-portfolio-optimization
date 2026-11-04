"""End-to-end smoke test of scripts 05 and 06 on the planted-signal market (CPU, no network)."""

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
from conftest import synthetic_market

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
features_script = importlib.import_module("05_features")
backtest_script = importlib.import_module("06_backtest")


def test_pipeline_end_to_end(tmp_path):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    processed.mkdir()
    artifacts.mkdir()
    scores, prices = synthetic_market(signal=0.5)
    prices.to_parquet(processed / "prices.parquet")
    for m in ("vader", "finetuned"):
        scores[["ticker", "date", "score"]].to_parquet(artifacts / f"scores_{m}.parquet")
    (artifacts / "nlp_eval.json").write_text(json.dumps({"best": "finetuned"}))

    features_script.main(processed, artifacts)
    metrics = backtest_script.main(artifacts, test_years=[2016, 2017], device="cpu")

    assert {"EW", "MVO", "BL-no-views", "A-lstm@finetuned", "A-ridge@finetuned",
            "B-tilt@finetuned", "A-lstm@vader"} <= set(metrics)
    assert metrics["A-lstm@finetuned"]["sharpe"] > metrics["EW"]["sharpe"]
    ic = json.loads((artifacts / "ic.json").read_text())
    assert ic["A-lstm@finetuned"]["mean_ic"] > 0.1
    assert (artifacts / "models" / "A-lstm.pt").exists()
    sent = pd.read_parquet(artifacts / "sentiment_weekly.parquet")
    assert {"sent_mean", "news_count", "price"} <= set(sent.columns)
    assert sent["price"].notna().all()
