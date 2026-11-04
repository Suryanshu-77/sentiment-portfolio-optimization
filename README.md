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

30 US large caps, 234 out-of-sample weeks (Jan 2016 to Jun 2020, including the COVID crash),
net of 10 bps costs. Produced by `scripts/06_backtest.py`.

| Strategy | CAGR | Vol | Sharpe [95% CI] | Max DD | Turnover/wk |
|---|---|---|---|---|---|
| MVO | 12.4% | 18.6% | 0.73 [-0.14, 1.79] | -30.3% | 0.15 |
| EW | 10.6% | 24.9% | 0.53 [-0.45, 1.62] | -45.1% | 0.00 |
| BL-no-views | 10.6% | 24.9% | 0.53 [-0.45, 1.62] | -45.1% | 0.00 |
| A-lstm@vader | 10.1% | 24.3% | 0.52 [-0.44, 1.60] | -43.7% | 0.55 |
| B-tilt@finbert | 9.6% | 23.8% | 0.50 [-0.45, 1.58] | -43.0% | 0.47 |
| A-lstm@finbert | 9.3% | 24.2% | 0.49 [-0.50, 1.59] | -44.5% | 0.47 |
| B-tilt@finetuned | 9.0% | 23.8% | 0.48 [-0.48, 1.54] | -43.6% | 0.47 |
| A-lstm@finetuned | 9.1% | 24.0% | 0.48 [-0.49, 1.55] | -42.8% | 0.52 |
| B-tilt@vader | 9.1% | 24.6% | 0.48 [-0.46, 1.56] | -44.0% | 0.49 |
| A-ridge@finetuned | 6.3% | 24.5% | 0.37 [-0.63, 1.39] | -45.8% | 1.13 |
| A-ridge@vader | 3.4% | 25.7% | 0.26 [-0.70, 1.35] | -51.8% | 1.14 |
| A-ridge@finbert | 3.1% | 24.8% | 0.25 [-0.75, 1.26] | -51.0% | 1.07 |

Forecast quality (weekly rank IC of the LSTM's excess-return forecasts):
VADER +0.019 (t = 1.5), FinBERT +0.011 (t = 0.8), fine-tuned +0.012 (t = 0.9).

BL-no-views matches EW exactly by construction: the max-Sharpe portfolio of the equal-weight
equilibrium prior is the equal-weight portfolio. It is kept as a check on the BL machinery.

**What this shows**

- **No statistically detectable sentiment alpha at a weekly horizon.** Forecast ICs are about 0.01
  and insignificant. No sentiment strategy beats equal weight, and every Sharpe confidence
  interval overlaps every other and includes zero.
- **Better NLP did not mean better trading.** The fine-tuned model beats FinBERT by 0.20 macro-F1
  at classifying sentiment, yet its signal predicts returns no better. Classification accuracy and
  return predictability are different targets.
- **Risk-aware construction beat sentiment.** Only MVO (no sentiment at all) improved on equal
  weight (Sharpe 0.73, max drawdown -30% vs -45%), consistent with the low-volatility effect.
- **Costs matter.** The ridge forecaster trades about 113% of the book weekly against 52% for the
  LSTM; the extra turnover alone costs about 3 points of return a year at 10 bps.

**A bug the final review caught.** An earlier version anchored the views at the *average* prior
(`Q = mean(π) + forecast`). That quietly tilted every portfolio toward low-volatility stocks even
when the forecast was zero, and made Strategy A look like it cut volatility and drawdown. With
views anchored at each stock's own prior, a zero forecast reproduces equal weight (now tested), and
Strategy A's apparent risk reduction disappears. The table above is from the corrected code.

These are the results of a design written before any real data was seen, not of tuning until
something looked good. Natural next steps: a daily horizon, where news effects are documented to
be stronger, a wider universe, and event-based (earnings) news.

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
- **Views in relative terms:** the forecaster predicts cross-sectional excess return, so each
  view is `Q_i = π_i + forecast_i`. A zero forecast reproduces the prior (equal weight)
  exactly, a property the tests check, so any tilt comes from the forecasts.
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
