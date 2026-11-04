# Sentiment-Driven Portfolio Optimization

<!-- After pushing to GitHub, add: ![CI](https://github.com/<you>/<repo>/actions/workflows/ci.yml/badge.svg) -->

News headlines are scored with transformer sentiment models (including one fine-tuned here),
turned into weekly per-stock features, and fed to two kinds of portfolio construction:

- **Strategy A**: an LSTM forecasts next-week *excess* returns. The forecasts become
  **Black-Litterman views** whose confidence (Ω) is each stock's own recent out-of-sample forecast
  error, so stocks where the model has been wrong lately fall back toward the equilibrium prior.
- **Strategy B**: a transparent sentiment tilt on equal weights (baseline).

Everything is evaluated in a **look-ahead-free, purged walk-forward backtest** (yearly retraining,
weekly rebalancing, 10 bps costs) against equal-weight, mean-variance and Black-Litterman-without-views
benchmarks, with block-bootstrap confidence intervals on Sharpe. A Streamlit dashboard presents
the results and live suggested weights from current news.

## Architecture

```
Kaggle headlines (2010-2020) --+
Finnhub live news (dashboard) -+
                               v
                 NLP scoring (GPU, cached Parquet, resumable)
                 VADER  vs  FinBERT  vs  fine-tuned RoBERTa
                               v
yfinance prices --> weekly per-ticker features (cross-sectionally z-scored,
                    news usable from the next trading day; no look-ahead)
                               v
      Strategy A (LSTM -> Black-Litterman)   Strategy B (tilt)   EW / MVO / BL-no-views
                               v
      Walk-forward backtest (train <= Y-2, validate Y-1, test Y; purged splits)
                               v
                 metrics + IC  -->  Streamlit dashboard
```

| Module | Responsibility |
|---|---|
| `sdpo/data/` | Kaggle loader with schema validation, yfinance with retry, universe selection |
| `sdpo/nlp/` | Batch scoring, fine-tuning, evaluation |
| `sdpo/features.py` | Weekly feature panel and model sequences |
| `sdpo/models/forecaster.py` | Pooled LSTM and ridge forecasters |
| `sdpo/portfolio/` | Black-Litterman posterior, capped max-Sharpe optimizer |
| `sdpo/strategies.py` | All strategies behind one `fit` / `weights` interface |
| `sdpo/backtest.py` | Walk-forward engine |
| `sdpo/metrics.py` | Performance metrics, Sharpe bootstrap, rank IC |

## Results

### Sentiment models (held-out Twitter financial-news set, 2,388 headlines)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| VADER (lexicon) | 0.494 | 0.447 |
| ProsusAI FinBERT | 0.725 | 0.668 |
| **Fine-tuned RoBERTa (this project)** | **0.897** | **0.872** |

The fine-tuned model saw the Twitter training split, so part of its edge is domain match. That
match is also why it suits short news headlines. FinBERT was not evaluated on PhraseBank because
it was trained on it.

### Backtest (out-of-sample 2016 to mid-2020)

Produced by `scripts/06_backtest.py` (`artifacts/results.md`): run the pipeline below to
regenerate it. *Table to be pasted here after the first full real-data run.*

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU
.venv/Scripts/pip install -e .[dev]                                                 # after torch
```

1. Download the Kaggle dataset *Massive Stock News Analysis DB for NLP/Backtests* and put
   `raw_analyst_ratings.csv` and `raw_partner_headlines.csv` in `data/raw/`.
2. Run the stages in order:

```bash
python scripts/01_data.py                 # universe + prices
python scripts/02_finetune.py             # GPU, a few minutes
python scripts/03_evaluate_nlp.py         # NLP comparison
python scripts/04_score.py --model vader  # repeat with finbert, finetuned (resumable)
python scripts/05_features.py
python scripts/06_backtest.py
streamlit run app/dashboard.py
```

Tests (`pytest`) need no data, network or GPU. They include a **planted-signal market** where
sentiment does predict returns, and the full pipeline must find it.

## Methodology notes

- **No look-ahead:** a headline dated *d* is usable from the close of the next trading day.
  A property test perturbs all data after week *t* and requires features at *t* to be identical.
  The test was itself checked by injecting leaks, which it catches.
- **Purged splits:** a training week's label is next week's return, so the last week of each split
  is dropped; no label crosses into validation or test.
- **Views in relative terms:** the forecaster predicts cross-sectional excess return, so views are
  `Q = mean(π) + forecast`. They move relative expectations, not the overall level.
- **Universe:** the 30 tickers with the most headlines in 2010–2011 (before any test year) with
  complete price history.

## Known limitations

- Survivorship bias: the Kaggle dataset covers tickers that existed in 2020.
- The BL prior is equal-weight, not market-cap (free point-in-time market caps are unavailable).
- About 230 test weeks; Sharpe confidence intervals will be wide.
- The live tab applies 2010–2020-trained models to current news (distribution shift).
- DeBERTa-v3 was the first fine-tuning choice but collapsed to the majority class under
  transformers 5.x, while RoBERTa trained normally on identical data, so RoBERTa is used.

*Research project, not investment advice.*
