"""Stage 6: walk-forward backtest of every strategy; metrics, IC, final live model.

Baselines run once on the best NLP model's panel; sentiment strategies run once per NLP
model (named '<strategy>@<model>'), which is the NLP ablation.
Writes artifacts/{returns,weights,turnover}.parquet, metrics.json, ic.json, results.md,
models/A-lstm.pt.
"""

import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sdpo import config
from sdpo.backtest import make_splits, walk_forward
from sdpo.metrics import perf_metrics, rank_ic, sharpe_ci
from sdpo.strategies import BASELINES, SentimentTilt, StrategyA

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
best_model = importlib.import_module("05_features").best_model


def sentiment_factories(model: str, device: str | None) -> dict:
    return {f"B-tilt@{model}": SentimentTilt,
            f"A-lstm@{model}": lambda: StrategyA("lstm", device=device),
            f"A-ridge@{model}": lambda: StrategyA("ridge", device=device)}


def information_coefficient(res, panel: pd.DataFrame) -> dict:
    realized = panel["fwd_excess"].unstack("ticker")
    out = {}
    for name in res.returns:
        preds = {}
        for (_, n), strat in res.strategies.items():
            if n == name and isinstance(strat, StrategyA):
                preds.update(strat.preds)
        if not preds:
            continue
        weeks = sorted(preds)
        ic = rank_ic(np.stack([preds[w] for w in weeks]), realized.loc[weeks].to_numpy())
        ic = ic[np.isfinite(ic)]
        out[name] = {"mean_ic": float(ic.mean()),
                     "t_stat": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))}
    return out


def results_table(metrics: dict) -> str:
    rows = ["| Strategy | CAGR | Vol | Sharpe [95% CI] | Max DD | Turnover/wk |",
            "|---|---|---|---|---|---|"]
    for name, m in sorted(metrics.items(), key=lambda kv: -kv[1]["sharpe"]):
        lo, hi = m["sharpe_ci"]
        rows.append(f"| {name} | {m['cagr']:.1%} | {m['vol']:.1%} | {m['sharpe']:.2f} "
                    f"[{lo:.2f}, {hi:.2f}] | {m['max_dd']:.1%} | {m['avg_turnover']:.2f} |")
    return "\n".join(rows)


def main(artifacts: Path = config.ARTIFACTS_DIR, test_years=config.TEST_YEARS,
         device: str | None = None) -> dict:
    available = [m for m in ("vader", "finbert", "finetuned")
                 if (artifacts / f"panel_{m}.parquet").exists()]
    if not available:
        raise SystemExit("No artifacts/panel_<model>.parquet found; run scripts/05_features.py")
    best = best_model(artifacts, available)
    panels = {m: pd.read_parquet(artifacts / f"panel_{m}.parquet") for m in available}

    runs = []
    for m in [best] + [x for x in available if x != best]:
        factories = sentiment_factories(m, device)
        if m == best:
            factories = {k: v for k, v in BASELINES.items() if k != "B-tilt"} | factories
            if importlib.util.find_spec("sdpo.models.e2e"):  # optional Strategy C
                from sdpo.strategies import FACTORIES_C
                factories |= {f"{k}@{m}": v for k, v in FACTORIES_C.items()}
        log.info("backtesting %s on panel_%s", list(factories), m)
        runs.append((m, walk_forward(panels[m], factories, test_years)))

    returns = pd.concat([r.returns for _, r in runs], axis=1)
    turnover = pd.concat([r.turnover for _, r in runs], axis=1)
    weights = pd.concat([r.weights for _, r in runs])
    returns.to_parquet(artifacts / "returns.parquet")
    turnover.to_parquet(artifacts / "turnover.parquet")
    weights.to_parquet(artifacts / "weights.parquet")

    metrics = {n: {**perf_metrics(returns[n], turnover[n]), "sharpe_ci": sharpe_ci(returns[n])}
               for n in returns}
    ic = {}
    for m, r in runs:
        ic |= information_coefficient(r, panels[m])
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (artifacts / "ic.json").write_text(json.dumps(ic, indent=2))

    # Final model for the live tab: validate on the last year with data (2020 for the
    # Kaggle archive), train on everything before it.
    last_year = panels[best].index.get_level_values("week").max().year
    final = StrategyA("lstm", device=device)
    final.fit(*make_splits(panels[best], last_year + 1))
    final.save(artifacts / "models" / "A-lstm.pt")

    table = results_table(metrics)
    (artifacts / "results.md").write_text(table + "\n")
    print(table)
    print("\nForecast IC:", json.dumps(ic, indent=2))
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
