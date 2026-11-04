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

30 US large caps, 232 out-of-sample weeks (Jan 2016 to Jun 2020, including the COVID crash),
net of 10 bps costs. Produced by `scripts/06_backtest.py`.

| Strategy | CAGR | Vol | Sharpe [95% CI] | Max DD | Turnover/wk |
|---|---|---|---|---|---|
| MVO | 12.4% | 18.6% | 0.73 [-0.14, 1.79] | -30.3% | 0.15 |
| A-lstm@vader | 10.1% | 21.2% | 0.56 [-0.42, 1.72] | -38.6% | 0.51 |
| A-lstm@finbert | 9.2% | 20.6% | 0.53 [-0.46, 1.73] | -39.1% | 0.43 |
| EW | 10.6% | 24.9% | 0.53 [-0.45, 1.62] | -45.1% | 0.00 |
| BL-no-views | 10.6% | 24.9% | 0.53 [-0.45, 1.62] | -45.1% | 0.00 |
| A-lstm@finetuned | 9.1% | 20.6% | 0.53 [-0.46, 1.67] | -38.3% | 0.47 |
| B-tilt@finbert | 9.6% | 23.8% | 0.50 [-0.45, 1.58] | -43.0% | 0.47 |
| B-tilt@finetuned | 9.0% | 23.8% | 0.48 [-0.48, 1.54] | -43.6% | 0.47 |
| B-tilt@vader | 9.1% | 24.6% | 0.48 [-0.46, 1.56] | -44.0% | 0.49 |
| A-ridge@finetuned | 6.6% | 21.8% | 0.40 [-0.60, 1.45] | -41.8% | 1.10 |
| A-ridge@vader | 4.1% | 22.9% | 0.29 [-0.68, 1.42] | -48.0% | 1.10 |
| A-ridge@finbert | 3.2% | 21.8% | 0.25 [-0.73, 1.34] | -47.0% | 1.04 |

Forecast quality (weekly rank IC of the LSTM's excess-return forecasts):
VADER +0.019 (t = 1.5), FinBERT +0.011 (t = 0.8), fine-tuned +0.012 (t = 0.9).

**What this shows**

- **No statistically detectable sentiment alpha at a weekly horizon.** Forecast ICs are about 0.01
  and insignificant; the raw sentiment features alone have ICs of 0.008 to 0.019. Every Sharpe
  confidence interval overlaps every other and includes zero.
- **Better NLP did not mean better trading.** The fine-tuned model beats FinBERT by 0.20 macro-F1
  at classifying sentiment, yet its signal predicts returns no better. Classification accuracy and
  return predictability are different targets.
- **Where the gains came from was portfolio construction, not sentiment.** Strategy A (LSTM) cuts
  volatility (20.6% vs 24.9%) and max drawdown (-38% vs -45%) relative to equal weight at a similar
  Sharpe. Covariance-aware optimization (MVO, Sharpe 0.73) did best, consistent with the strongest
  raw feature being volatility (IC -0.035, the low-volatility effect).
- **Costs matter.** The ridge forecaster trades about 110% of the book weekly and loses about
  3 points of CAGR against the LSTM version, which trades about 45%.

These are the results of a pre-registered design (spec written before any real data was seen),
not of tuning until something looked good. Natural next steps: a daily horizon, where news effects
are documented to be stronger, a wider universe, and event-based (earnings) news.

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

- Survivorship bias: the Kaggle dataset covers tickers that existed in 2020, and 25 of the 80
  best-covered tickers (e.g. VMware, Tiffany, Monsanto) have no price history on yfinance today,
  so the universe can only contain survivors. Three ETFs (QQQ, SLV, EPS) were excluded so the
  universe is companies only.
- The BL prior is equal-weight, not market-cap (free point-in-time market caps are unavailable).
- About 230 test weeks; Sharpe confidence intervals will be wide.
- The live tab applies 2010–2020-trained models to current news (distribution shift).
- DeBERTa-v3 was the first fine-tuning choice but collapsed to the majority class under
  transformers 5.x, while RoBERTa trained normally on identical data, so RoBERTa is used.

*Research project, not investment advice.*
