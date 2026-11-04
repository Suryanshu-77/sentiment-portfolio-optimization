# Sentiment-Driven Portfolio Optimization — Design Spec

Date: 2026-11-04
Status: approved design, pending spec review

## 1. Goal

A resume project that shows quant finance, ML/DL, and a usable dashboard in one system:
news sentiment is extracted with transformer models, turned into return forecasts and
portfolio weights, and evaluated in a look-ahead-free walk-forward backtest against
standard benchmarks.

Success criteria:

1. A reproducible pipeline (scripts run in order) from raw data to a results table.
2. NLP comparison: VADER vs off-the-shelf FinBERT vs a self-fine-tuned model on a held-out set.
3. Three sentiment strategies (A, B, C) plus three benchmarks, compared on out-of-sample
   metrics with a bootstrap confidence interval on Sharpe.
4. A Streamlit dashboard showing results, sentiment, NLP comparison, and live suggested weights.
5. A pytest suite, including no-look-ahead and planted-signal tests, running in GitHub Actions.

Non-goals: live trading or broker integration, intraday data, user accounts, non-US markets,
a market-cap-weighted prior (see §9), and production deployment (Streamlit Cloud is a stretch goal).

## 2. Environment

- Python 3.13, Windows 11, NVIDIA RTX 4060 Laptop (8 GB VRAM), PyTorch CUDA build.
- Main libraries: pandas, numpy, pyarrow, yfinance, torch, transformers, datasets,
  vaderSentiment, PyPortfolioOpt, cvxpy, scikit-learn, streamlit, plotly, requests, pytest, ruff.

## 3. Data

| Source | Use | Notes |
|---|---|---|
| Kaggle "Massive Stock News Analysis DB for NLP/Backtests" (`raw_analyst_ratings.csv`, `raw_partner_headlines.csv`) | Historical headlines, about 2009 to mid-2020 | Columns: headline, date, stock. Downloaded manually into `data/raw/`. |
| yfinance | Daily adjusted close, 2010-01 to 2020-07, plus recent prices for the live tab | |
| Financial PhraseBank (`takala/financial_phrasebank`, `sentences_allagree`) | Fine-tuning training data | CC BY-NC-SA 3.0, fine for non-commercial use |
| Twitter Financial News Sentiment (`zeroshot/twitter-financial-news-sentiment`) | Fine-tuning train split; its validation split is **the held-out NLP test set** | Labels bearish/bullish/neutral map to neg/pos/neu |
| Finnhub `/company-news` (free tier) | Live tab only: last 4 weeks of news | API key from env `FINNHUB_API_KEY` or `.streamlit/secrets.toml` (gitignored) |

The NLP test set is the Twitter validation split, because ProsusAI FinBERT was trained on
PhraseBank and testing it there would leak.

**Universe:** the 30 tickers with the most headlines in 2010-01-01..2011-12-31, among those
with complete yfinance price history over 2010-01..2020-06. Tickers that fail are logged
and excluded.

## 4. Architecture

```
src/sdpo/
  config.py              constants: dates, N=30, costs, lambdas, seeds, paths
  data/news.py           Kaggle loader + schema validation; Finnhub client
  data/prices.py         yfinance download → Parquet
  data/universe.py       universe selection (§3)
  nlp/score.py           batch scoring with VADER / FinBERT / fine-tuned; chunked, resumable
  nlp/finetune.py        HF Trainer fine-tune of microsoft/deberta-v3-base
  nlp/evaluate.py        accuracy, macro-F1, confusion matrix on the held-out set
  features.py            weekly per-ticker feature panel (§5)
  models/forecaster.py   Strategy A LSTM + ridge ablation
  models/e2e.py          Strategy C end-to-end network
  portfolio/black_litterman.py   hand-written BL posterior
  portfolio/optimize.py  constrained max-Sharpe via PyPortfolioOpt/cvxpy
  strategies.py          EW, MVO, BL-no-views, B, A, C — common interface
  backtest.py            walk-forward engine
  metrics.py             performance metrics + block bootstrap
scripts/                 01_prices.py … 07_backtest.py, run in order, write to artifacts/
app/dashboard.py         Streamlit app (reads artifacts/ only, except the live tab)
tests/
data/  artifacts/        gitignored
```

Strategy interface: `weights(t, history) -> pd.Series` indexed by ticker, non-negative,
summing to 1, each ≤ 0.10. `history` contains only data available at Friday close of week t.

## 5. NLP and features

- **Headline score** = P(positive) − P(negative), in [−1, 1]. VADER uses its compound score.
- **Models compared:** VADER, `ProsusAI/finbert`, and fine-tuned `microsoft/deberta-v3-base`
  (bf16, max length 64, 3 epochs, trained on PhraseBank allagree + Twitter train).
  The best model on held-out macro-F1 is used for features. All three are scored and cached so
  the backtest can be repeated per model (ablation).
- **Timing:** a headline dated d (date only, or a timestamp converted to US/Eastern) is first
  usable at the close of trading day d+1.
- **Weekly features**, per ticker, at Friday close:
  sent_mean, sent_std, log(1+news_count), sent_change (= this week's sent_mean − mean of the
  previous 4 weeks), ret_1w, ret_4w, vol_4w (std of daily returns).
  A week with no news gets sent_mean=0, sent_std=0, news_count=0.
  Each feature is z-scored **across tickers within the same week** (no time pooling).
- **Model input:** the last L=12 weeks of features per ticker.

## 6. Strategies

All strategies: long-only, fully invested, max 10% per stock, weekly rebalance at Friday close.

| Name | Description |
|---|---|
| EW | 1/N |
| MVO | max-Sharpe on trailing-52w mean returns and Ledoit-Wolf covariance |
| BL-no-views | BL posterior with no views (= prior), then the same optimizer |
| **B: sentiment tilt** | w_i ∝ max(0, 1 + 0.5·z(sent_mean)_i), then cap and renormalize |
| **A: LSTM → BL** | see below |
| **C: end-to-end** | see below |

**Strategy A**
- Target: next-week return minus the cross-sectional mean (excess return).
- Model: one LSTM shared across tickers (1 layer, hidden 32, dropout 0.2) → linear head, MSE loss,
  Adam, early stopping on validation. Ablation: ridge regression on the flattened 12×7 input.
- Forecast quality: weekly Spearman rank IC, reporting mean IC and its t-statistic.
- Black-Litterman: Σ = Ledoit-Wolf on trailing 52 weeks; δ = 2.5; τ = 0.05; π = δΣw_eq;
  P = I; Q = forecast; Ω = diag(mean squared out-of-sample forecast error per ticker over the
  trailing 26 weeks). For the first 26 test weeks, Ω uses validation-year errors.
  Posterior μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹[(τΣ)⁻¹π + PᵀΩ⁻¹Q], then max-Sharpe with constraints.

**Strategy C** (optional, built last: the project is complete without it. Every other strategy,
the dashboard, and the tests must work whether or not C exists.)
- Shared per-ticker LSTM encoder (same shape as A) → scalar score per ticker → softmax across
  tickers → weights. The 10% cap is applied after the softmax (clip and renormalize) at inference.
- Training loss on overlapping 52-week chunks (stride 4 weeks, shuffled each epoch):
  `−mean(r_p)/std(r_p) + λ_eq·mean‖w_t − 1/N‖²`, where
  `r_p,t = w_t·r_{t+1} − 0.001·‖w_t − w_{t−1}‖₁`; λ_eq is tuned on the validation year from {0.1, 1, 10}.
- Adam, early stopping on validation Sharpe.

## 7. Backtest

- Test years Y = 2016..2020 (2020 ends at the last full week of data, about June).
- For each Y: train on 2010-01..end of Y−2, validate on Y−1, test on Y. Models are retrained each Y.
- Weekly holding period, Friday close to Friday close. Transaction cost: 10 bps × ‖w_t − w_{t−1}‖₁.
- If the optimizer is infeasible or fails, keep the previous week's weights and log the date.
- **Metrics:** CAGR, annualized volatility, Sharpe (rf = 0), Sortino, max drawdown, Calmar,
  average weekly turnover. Sharpe 95% CI from a circular block bootstrap (block length 4,
  2000 resamples).
- Outputs in `artifacts/`: `returns.parquet` (weekly net returns per strategy), `weights.parquet`,
  `metrics.json`, `ic.json`, `nlp_eval.json`, `sentiment_weekly.parquet`, trained models.

## 8. Dashboard (Streamlit, Plotly)

1. **Backtest results:** equity curves, metrics table with Sharpe CI, drawdowns, test-year filter.
2. **Sentiment explorer:** per-ticker weekly sentiment vs price, plus top-scored headlines.
3. **NLP comparison:** metrics table and confusion matrices.
4. **Live portfolio:** Finnhub news for the last 4 weeks → best NLP model → features with recent
   yfinance prices → latest A model (and C, if its artifacts exist) → suggested weights. Cached for one hour. Banner:
   "Research demo — not investment advice." Delisted tickers are dropped with a note. A missing
   key or API error shows a message; the other tabs are unaffected.

Tabs 1–3 read only `artifacts/` and need no GPU or network.

## 9. Known limitations (documented in README)

- Survivorship bias: the Kaggle dataset covers tickers that existed in 2020.
- The BL prior is equal-weight, not market-cap (free point-in-time market caps are unavailable).
- About 230 test weeks; Sharpe confidence intervals will be wide.
- The live tab applies 2010–2020-trained models to current news (distribution shift).

## 10. Testing and error handling

pytest on synthetic data, no network or GPU:

- BL matches PyPortfolioOpt `BlackLittermanModel`; Ω→∞ gives the prior; Ω→0 gives the views.
- No look-ahead: changing any input after date t leaves features at t identical; a headline dated
  d is absent from features before d+1.
- Backtest: a hand-computed toy portfolio; turnover-cost arithmetic; weight invariants for every strategy.
- Metrics: Sharpe and max drawdown on hand-checked series.
- Strategy C: outputs non-negative and summing to 1; gradients reach all parameters; it can overfit
  a tiny dataset.
- Planted signal: a synthetic market where sentiment predicts next-week excess return. A has
  mean IC > 0 and A beats EW on Sharpe (and C beats EW too, once C is built).

Error handling: Kaggle schema validation with counted drops; failed tickers logged and excluded;
NLP scoring writes Parquet chunks and resumes; optimizer fallback (§7); fixed seeds; stdlib logging.

Tooling: `pyproject.toml`, `requirements.txt`, ruff, git, GitHub Actions running ruff + pytest (CPU).
