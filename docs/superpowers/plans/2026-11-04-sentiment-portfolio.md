# Sentiment-Driven Portfolio Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible pipeline that scores news headlines with transformer models, turns sentiment into portfolio weights (Strategies A and B, with C optional), evaluates them in a look-ahead-free walk-forward backtest, and presents the results in a Streamlit dashboard.

**Architecture:** A `src/sdpo/` package with one module per stage (data → nlp → features → models/portfolio → backtest → metrics). Numbered `scripts/` run the stages in order and write Parquet/JSON to `artifacts/`. `app/dashboard.py` reads only `artifacts/`, except the live tab, which calls Finnhub.

**Tech Stack:** Python 3.13, pandas, numpy, pyarrow, yfinance, PyTorch (CUDA), transformers, datasets, vaderSentiment, PyPortfolioOpt, cvxpy, scikit-learn, Streamlit, Plotly, requests, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-04-sentiment-portfolio-design.md`

## Global Constraints

- Python 3.13; the package is installed editable (`pip install -e .[dev]`) from a `.venv`.
- Every strategy's weights: non-negative, sum to 1 (tol 1e-6), each ≤ 0.10 (tol 1e-6). N = 30.
- Weekly rebalance at the last trading day of each W-FRI week; transaction cost 10 bps × ‖w_t − w_{t−1}‖₁.
- A headline dated d is first usable at the close of the next trading day after d.
- BL: δ = 2.5, τ = 0.05, Σ = Ledoit-Wolf on trailing 52 weekly returns, Ω window 26 weeks.
- Walk-forward: test Y ∈ 2016..2020; train 2010-01-01..end of Y−2; validate Y−1.
- Sequence length L = 12; features exactly `["sent_mean","sent_std","log_news","sent_change","ret_1w","ret_4w","vol_4w"]`.
- Seeds: 42 everywhere (`sdpo.config.SEED`).
- Tests never touch the network or the GPU. `data/`, `artifacts/`, `.streamlit/secrets.toml` are never committed.
- Finnhub key is read only from env `FINNHUB_API_KEY` or `st.secrets["FINNHUB_API_KEY"]`.
- Strategy C is optional: nothing outside Task 16 may import `sdpo.models.e2e`.

## Review Focus

1. **A ticker with one or zero headlines in a week:** pandas `std` returns NaN for a single value, and a model fed NaN silently produces NaN weights. Expected: `sent_std` = 0 and no NaN anywhere in the panel. Test in Task 6.
2. **A week whose Friday is a market holiday (e.g. Good Friday):** expected to use that week's last trading day as week-end, not drop the week. Test in Task 6.
3. **`max_sharpe` with all expected returns ≤ 0:** PyPortfolioOpt raises. Expected: the optimizer returns `None` and the engine keeps last week's weights (EW on the first week). Tests in Tasks 8 and 9.
4. **Mixed Kaggle date formats** (`2020-06-05 10:30:54-04:00` and `2011-04-27 00:00:00`): both must parse to a US/Eastern calendar date. Test in Task 2.
5. **Live tab with a ticker that has zero Finnhub articles:** expected to be treated as a no-news week, not an error. Test in Task 15.

## Learning contributions (for the human)

Two pieces are reserved for the user to write (Learn by Doing): `bl_posterior` (Task 8) and the Sharpe loss (Task 16). The implementer leaves exactly one `TODO(human)` at a time and waits.

---

## Phase 0: Setup

### Task 1: Project scaffolding, config, CI

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `src/sdpo/__init__.py`, `src/sdpo/config.py`, `src/sdpo/{data,nlp,models,portfolio}/__init__.py`, `tests/test_config.py`, `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `sdpo.config` constants: `SEED=42`, `N_UNIVERSE=30`, `UNIVERSE_WINDOW=("2010-01-01","2011-12-31")`, `PRICE_START="2010-01-01"`, `PRICE_END="2020-07-01"`, `TEST_YEARS=range(2016,2021)`, `TRAIN_START="2010-01-01"`, `SEQ_LEN=12`, `FEATURES` (list above), `MAX_WEIGHT=0.10`, `COST_BPS=10`, `BL_DELTA=2.5`, `BL_TAU=0.05`, `COV_WINDOW=52`, `OMEGA_WINDOW=26`, `TILT_K=0.5`, `LAMBDA_EQ_GRID=(0.1,1.0,10.0)`, `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `ARTIFACTS_DIR` (Paths relative to repo root, resolved from `config.py`'s location).

- [ ] **Step 1: Create a venv and install the CUDA build of torch**

Run: `python -m venv .venv && .venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu126`
Then: `.venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"` → `True`

- [ ] **Step 2: Write `pyproject.toml` (setuptools, `src` layout, package `sdpo`) and `requirements.txt`**

Runtime deps: pandas, numpy, pyarrow, yfinance, transformers, datasets, sentencepiece, accelerate, vaderSentiment, PyPortfolioOpt, cvxpy, scikit-learn, streamlit, plotly, requests. `[dev]` extra: pytest, ruff. torch is not listed in requirements (index-specific install). Configure `[tool.ruff] line-length = 100` and `[tool.pytest.ini_options] testpaths = ["tests"]`.

- [ ] **Step 3: Write the failing test `tests/test_config.py`**

```python
from sdpo import config

def test_constants_match_spec():
    assert config.N_UNIVERSE * config.MAX_WEIGHT >= 1  # cap is feasible
    assert list(config.TEST_YEARS) == [2016, 2017, 2018, 2019, 2020]
    assert len(config.FEATURES) == 7 and config.SEQ_LEN == 12
    assert (config.BL_DELTA, config.BL_TAU, config.COST_BPS) == (2.5, 0.05, 10)
    assert config.ARTIFACTS_DIR.name == "artifacts"
```

- [ ] **Step 4: Install, implement `config.py`, and run the test**

Run: `.venv/Scripts/pip install -e .[dev] && .venv/Scripts/pytest tests/test_config.py -v` → PASS

- [ ] **Step 5: Write `.github/workflows/ci.yml`**

ubuntu-latest, Python 3.13, `pip install torch --index-url https://download.pytorch.org/whl/cpu`, `pip install -e .[dev]`, `ruff check .`, `pytest -q`.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "chore: scaffold sdpo package, config, CI"`

---

## Phase 1: Data

### Task 2: Kaggle news loader with schema validation

**Files:**
- Create: `src/sdpo/data/news.py`, `tests/test_news.py`

**Interfaces:**
- Produces: `load_kaggle_news(paths: list[Path]) -> pd.DataFrame` with columns `ticker: str`, `headline: str`, `date: datetime64[ns]` (naive, normalized US/Eastern calendar date), deduplicated on (ticker, headline, date), sorted by date. It logs, via `logging.getLogger(__name__)`, how many rows were dropped and why.
- Each CSV must contain `headline`, `date`, `stock`; extra columns are ignored; a missing required column raises `ValueError` naming the file and the column.

- [ ] **Step 1: Write failing tests** (CSVs written to `tmp_path`)

```python
def test_parses_mixed_date_formats(tmp_path):
    # rows: ("Up big","2020-06-05 10:30:54-04:00","AAPL"), ("Down","2011-04-27 00:00:00","MSFT")
    df = load_kaggle_news([csv])
    assert df["date"].tolist() == [pd.Timestamp("2011-04-27"), pd.Timestamp("2020-06-05")]

def test_late_utc_timestamp_maps_to_eastern_date(tmp_path):
    # "2020-06-06 02:00:00+00:00" is 2020-06-05 22:00 US/Eastern
    assert load_kaggle_news([csv])["date"].iloc[0] == pd.Timestamp("2020-06-05")

def test_drops_bad_rows_and_duplicates(tmp_path):
    # rows: valid, empty headline, unparseable date "n/a", missing ticker, exact duplicate of row 1
    assert len(load_kaggle_news([csv])) == 1

def test_missing_column_raises(tmp_path):
    with pytest.raises(ValueError, match="stock"):
        load_kaggle_news([csv_without_stock])
```

- [ ] **Step 2: Run** `pytest tests/test_news.py -v` → FAIL (ImportError)
- [ ] **Step 3: Implement `load_kaggle_news`.** Parse with `pd.to_datetime(..., utc=True, errors="coerce", format="mixed")`. Naive values are treated as US/Eastern: parse naive and tz-aware strings separately, localize naive ones to US/Eastern, then `tz_convert("US/Eastern").dt.tz_localize(None).dt.normalize()`.
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** — `feat(data): Kaggle news loader with validation`

### Task 3: Prices and universe selection

**Files:**
- Create: `src/sdpo/data/prices.py`, `src/sdpo/data/universe.py`, `scripts/01_data.py`, `tests/test_universe.py`

**Interfaces:**
- Consumes: `load_kaggle_news` (Task 2).
- Produces:
  - `download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame`: wide daily adjusted close, DatetimeIndex × ticker columns, via `yfinance.download(auto_adjust=True)`. Columns that are entirely NaN are dropped and logged.
  - `select_universe(news: pd.DataFrame, prices: pd.DataFrame, n: int, window: tuple[str,str], full_range: tuple[str,str]) -> list[str]`: ranks by headline count inside `window` only, keeps tickers whose prices have no NaN over `full_range`, returns the top `n` sorted alphabetically. Raises `ValueError` if fewer than `n` qualify.
  - Files: `data/processed/news.parquet` (universe tickers only), `prices.parquet`, `universe.json`.

- [ ] **Step 1: Write failing tests**

```python
def test_ranks_only_inside_window():
    # ticker X: 100 headlines in 2015, 1 in 2010; Y: 5 in 2010 → Y ranks above X
    assert select_universe(news, prices, n=1, window=("2010-01-01","2011-12-31"), full_range=...) == ["Y"]

def test_excludes_incomplete_price_history():
    # Z has the most headlines but NaN prices in 2016
    assert "Z" not in select_universe(news, prices, n=2, ...)

def test_raises_when_too_few_qualify():
    with pytest.raises(ValueError):
        select_universe(news, prices, n=50, ...)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement both functions.** **Step 4: Run** → PASS.
- [ ] **Step 5: Write `scripts/01_data.py`.** It loads both Kaggle CSVs from `config.RAW_DIR`, downloads prices for the top 80 tickers by window count (this bounds the downloads), selects the universe, and writes the three files. It prints the user's manual download instruction if the CSVs are missing.
- [ ] **Step 6: Manual run (needs Kaggle files in `data/raw/`):** `python scripts/01_data.py` → prints 30 tickers and the headline count per ticker. Record the total headline count in the commit message.
- [ ] **Step 7: Commit** — `feat(data): prices download and universe selection`

---

## Phase 2: NLP

### Task 4: Headline scoring (VADER, FinBERT, any HF classifier), resumable

**Files:**
- Create: `src/sdpo/nlp/score.py`, `scripts/04_score.py`, `tests/test_score.py`

**Interfaces:**
- Produces:
  - `probs_to_score(probs: np.ndarray, id2label: dict[int,str]) -> np.ndarray`: P(positive) − P(negative), with label names matched case-insensitively.
  - `class HFScorer(model_name_or_path: str, batch_size: int = 256, max_length: int = 64)` with `__call__(texts: list[str]) -> np.ndarray` (fp16 on CUDA if available) and `predict_labels(texts) -> np.ndarray` in the unified ids {0: neg, 1: neu, 2: pos}.
  - `class VaderScorer` with the same two methods: score = compound; label pos if ≥ 0.05, neg if ≤ −0.05, otherwise neu.
  - `score_to_parquet(news: pd.DataFrame, scorer: Callable[[list[str]], np.ndarray], out_dir: Path, chunk_size: int = 50_000) -> pd.DataFrame`: writes `chunk_00000.parquet`, … (columns ticker, date, score), skips chunks that already exist, and returns the concatenation.
  - Files: `artifacts/scores_{vader,finbert,finetuned}.parquet`.

- [ ] **Step 1: Write failing tests**

```python
def test_probs_to_score_uses_label_names_not_positions():
    probs = np.array([[0.7, 0.2, 0.1]])  # ProsusAI order: positive, negative, neutral
    assert probs_to_score(probs, {0:"positive",1:"negative",2:"neutral"})[0] == pytest.approx(0.5)

def test_score_to_parquet_resumes(tmp_path):
    calls = []
    fake = lambda texts: calls.append(len(texts)) or np.zeros(len(texts))
    score_to_parquet(news_5_rows, fake, tmp_path, chunk_size=2)   # 3 chunks
    calls.clear()
    (tmp_path / "chunk_00002.parquet").unlink()
    out = score_to_parquet(news_5_rows, fake, tmp_path, chunk_size=2)
    assert calls == [1] and len(out) == 5

def test_vader_labels():
    assert VaderScorer().predict_labels(["great profits soar", "terrible losses crash"]).tolist() == [2, 0]
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Write `scripts/04_score.py --model {vader,finbert,finetuned}`.** finbert = `ProsusAI/finbert`; finetuned = `artifacts/models/finetuned` (from Task 5). Concatenated output goes to `artifacts/scores_<model>.parquet`.
- [ ] **Step 6: Commit** — `feat(nlp): resumable headline scoring`

### Task 5: Fine-tuning and NLP evaluation

**Files:**
- Create: `src/sdpo/nlp/finetune.py`, `src/sdpo/nlp/evaluate.py`, `scripts/02_finetune.py`, `scripts/03_evaluate_nlp.py`, `tests/test_nlp_eval.py`

**Interfaces:**
- Consumes: `HFScorer`, `VaderScorer` (Task 4).
- Produces:
  - `load_finetune_data() -> datasets.DatasetDict` with splits `train` (PhraseBank `sentences_allagree` + Twitter `train`) and `test` (Twitter `validation`), columns `text`, `label` in the unified ids. Mapping constants: `PHRASEBANK_MAP = {0:0, 1:1, 2:2}` (neg, neu, pos), `TWITTER_MAP = {0:0, 1:2, 2:1}` (Bearish→neg, Bullish→pos, Neutral→neu).
  - `finetune(out_dir: Path, base: str = "microsoft/deberta-v3-base") -> None`: HF `Trainer`, bf16, max_length 64, 3 epochs, lr 2e-5, batch 32, seed 42, 10% of train held out for eval. It saves the model with `id2label={0:"negative",1:"neutral",2:"positive"}`.
  - `classification_report_dict(y_true, y_pred) -> dict` with keys `accuracy`, `macro_f1`, `confusion` (3×3 list, rows = true).
  - File: `artifacts/nlp_eval.json` = `{model_name: report}` for vader, finbert, finetuned.

- [ ] **Step 1: Write failing tests** (pure functions only; no downloads)

```python
def test_twitter_label_mapping():
    assert [TWITTER_MAP[i] for i in (0, 1, 2)] == [0, 2, 1]

def test_report_known_predictions():
    r = classification_report_dict([0,1,2,2], [0,1,2,1])
    assert r["accuracy"] == 0.75 and r["confusion"][2] == [0, 1, 1]
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Manual check that the datasets load:** `python -c "from sdpo.nlp.finetune import load_finetune_data as f; print(f())"`. If `takala/financial_phrasebank` fails because the `datasets` version no longer runs loading scripts, load it with `revision="refs/convert/parquet"` instead and note this in a code comment.
- [ ] **Step 6: Manual GPU run:** `python scripts/02_finetune.py`, then `python scripts/03_evaluate_nlp.py` → prints the three reports. Expected: fine-tuned macro-F1 > FinBERT macro-F1 on the Twitter test set. If it isn't, record that honestly; the pipeline still uses the best model by macro-F1.
- [ ] **Step 7: Run scoring for all three models:** `python scripts/04_score.py --model vader` (repeat for finbert and finetuned). This is the multi-hour step; it resumes if interrupted.
- [ ] **Step 8: Commit** — `feat(nlp): fine-tune DeBERTa and evaluate VADER/FinBERT/fine-tuned`

---

## Phase 3: Features

### Task 6: Weekly feature panel and sequences

**Files:**
- Create: `src/sdpo/features.py`, `tests/conftest.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: scores frame `(ticker, date, score)` (Task 4), wide daily prices (Task 3).
- Produces:
  - `tests/conftest.py::synthetic_market(n_tickers=30, n_weeks=400, signal=0.3, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]` returning (scores, daily prices). Each ticker-week has a latent s ~ N(0,1) and 3 headlines dated that week's Tuesday with score `tanh(s + N(0,0.3))`. The next week's excess log-return is `0.01·signal·s + N(0, 0.02)`, spread evenly across that week's trading days. Prices start at 100 on `2010-01-04` over business days.
  - `week_ends(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex`: the last trading day of each W-FRI week.
  - `build_weekly_panel(scores: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame`: MultiIndex (`week`, `ticker`); columns = `config.FEATURES` (cross-sectionally z-scored per week; a column with zero cross-sectional std becomes 0) plus raw `fwd_ret` (simple return to the next week-end) and `fwd_excess` (fwd_ret minus its weekly cross-sectional mean). The final week has NaN `fwd_*`. Features contain no NaN after the first 4 weeks, which are dropped.
  - `make_sequences(panel: pd.DataFrame, seq_len: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]`: returns `X [T,N,L,F]`, `y_excess [T,N]`, `y_ret [T,N]`, `weeks [T]`, `tickers [N]`. The sequence for week t ends at t inclusive.
- News timing: each headline's effective date = the first trading day strictly after `date`. It is assigned to the week whose week-end is the first one ≥ the effective date.

- [ ] **Step 1: Write failing tests**

```python
def test_no_lookahead_property(synthetic):
    scores, prices = synthetic
    t = week_ends(prices.index)[200]
    base = build_weekly_panel(scores, prices).loc[:t, config.FEATURES]
    s2 = scores.copy(); s2.loc[s2.date > t, "score"] = 0.99
    p2 = prices.copy(); p2.loc[p2.index > t] *= 3.0
    after = build_weekly_panel(s2, p2).loc[:t, config.FEATURES]
    pd.testing.assert_frame_equal(base, after)

def test_headline_usable_only_from_next_trading_day():
    # one headline on Friday 2012-03-09 → appears in week ending 2012-03-16, not 2012-03-09

def test_single_or_zero_headlines_give_no_nan(synthetic):
    # drop all but one headline for ticker T0 in week w, and all headlines for T1
    assert not panel[config.FEATURES].isna().any().any()

def test_good_friday_week_uses_thursday():
    days = pd.bdate_range("2012-04-02", "2012-04-13").drop(pd.Timestamp("2012-04-06"))
    assert pd.Timestamp("2012-04-05") in week_ends(days)

def test_features_are_cross_sectionally_standardized(synthetic):
    m = panel.groupby(level="week")[config.FEATURES].mean()
    assert np.allclose(m.values, 0, atol=1e-9)

def test_make_sequences_shapes(synthetic):
    X, ye, yr, weeks, tickers = make_sequences(panel)
    assert X.shape[1:] == (30, 12, 7) and ye.shape == X.shape[:2]
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** Raw features before z-scoring: sent_mean and sent_std (ddof=0) of the week's scores; log_news = log1p(count); sent_change = sent_mean − mean(sent_mean over the previous 4 weeks); ret_1w and ret_4w from week-end prices; vol_4w = std of daily returns over the last 20 trading days. A week with no news gets sent_mean = sent_std = 0. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `feat: weekly feature panel with look-ahead tests`

---

## Phase 4: Portfolio core

### Task 7: Metrics and bootstrap

**Files:**
- Create: `src/sdpo/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `perf_metrics(r: pd.Series, turnover: pd.Series | None = None) -> dict` with keys `cagr, vol, sharpe, sortino, max_dd, calmar, avg_turnover`. Weekly data, annualized with 52, rf = 0. `max_dd` is negative.
  - `sharpe_ci(r: pd.Series, block: int = 4, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]`: circular block bootstrap.
  - `rank_ic(pred: np.ndarray, actual: np.ndarray) -> np.ndarray`: per-row Spearman correlation, NaN-safe (skips NaN pairs).

- [ ] **Step 1: Write failing tests**

```python
def test_max_drawdown_hand_series():
    assert perf_metrics(pd.Series([0.1, -0.5, 0.2]))["max_dd"] == pytest.approx(-0.5)

def test_sharpe_constant_mean():
    r = pd.Series([0.01, 0.03] * 50)
    assert perf_metrics(r)["sharpe"] == pytest.approx(0.02 / r.std() * np.sqrt(52))

def test_sharpe_ci_brackets_point_estimate():
    r = pd.Series(np.random.default_rng(0).normal(0.003, 0.02, 260))
    lo, hi = sharpe_ci(r); s = perf_metrics(r)["sharpe"]
    assert lo < s < hi

def test_rank_ic_perfect():
    a = np.arange(10.0)[None, :]
    assert rank_ic(a, a)[0] == pytest.approx(1.0)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** — `feat: performance metrics and Sharpe bootstrap`

### Task 8: Black-Litterman and constrained optimizer

**Files:**
- Create: `src/sdpo/portfolio/black_litterman.py`, `src/sdpo/portfolio/optimize.py`, `tests/test_portfolio.py`

**Interfaces:**
- Produces:
  - `implied_prior(sigma: np.ndarray, w: np.ndarray, delta: float = 2.5) -> np.ndarray` = δΣw.
  - `bl_posterior(sigma, pi, P, Q, omega, tau=0.05) -> np.ndarray`: the posterior mean from spec §6. **Body reserved for the human (Learn by Doing):** the implementer writes the signature, docstring, and `TODO(human)` and stops.
  - `cap_weights(w: pd.Series, max_w: float) -> pd.Series`: iteratively clip at max_w and redistribute the excess pro-rata to uncapped names.
  - `shrunk_cov(weekly_returns: pd.DataFrame) -> pd.DataFrame`: `pypfopt.risk_models.CovarianceShrinkage(..., returns_data=True, frequency=1).ledoit_wolf()`.
  - `max_sharpe_weights(mu: pd.Series, sigma: pd.DataFrame, max_w: float = 0.10) -> pd.Series | None`: `EfficientFrontier(mu, sigma, weight_bounds=(0, max_w)).max_sharpe(risk_free_rate=0.0)`, then `clean_weights`, then renormalize. Returns `None` on any exception (logged), including when every mu ≤ 0.

- [ ] **Step 1: Write failing tests**

```python
def test_bl_matches_pypfopt(rand_cov):           # 5 assets, random PSD sigma
    ours = bl_posterior(S, pi, P, Q, omega, tau=0.05)
    ref = BlackLittermanModel(S, pi=pi, P=P, Q=Q, omega=omega, tau=0.05).bl_returns()
    np.testing.assert_allclose(ours, ref.values, rtol=1e-8)

def test_bl_huge_omega_returns_prior(): ...      # omega = 1e12·I → ≈ pi (atol 1e-8)
def test_bl_tiny_omega_returns_views(): ...      # P = I, omega = 1e-12·I → ≈ Q (atol 1e-6)

def test_cap_weights():
    w = cap_weights(pd.Series([0.5, 0.3] + [0.2 / 28] * 28), 0.10)
    assert w.max() <= 0.10 + 1e-9 and w.sum() == pytest.approx(1)

def test_max_sharpe_respects_bounds(): ...       # 30 assets → sum 1, 0 ≤ w ≤ 0.10
def test_max_sharpe_all_negative_returns_none():
    assert max_sharpe_weights(pd.Series(-0.01, index=tickers), sigma) is None
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement everything except the `bl_posterior` body. Insert `TODO(human)` and hand off the Learn by Doing request** (use `np.linalg.solve`, not `inv`).
- [ ] **Step 4: After the human implements it, run** `pytest tests/test_portfolio.py -v` → PASS.
- [ ] **Step 5: Commit** — `feat(portfolio): Black-Litterman posterior and capped max-Sharpe`

### Task 9: Walk-forward backtest engine

**Files:**
- Create: `src/sdpo/backtest.py`, `tests/test_backtest.py`

**Interfaces:**
- Consumes: `make_sequences` (Task 6), `perf_metrics` (Task 7).
- Produces:
  - `@dataclass WeekContext`: `week: pd.Timestamp`, `tickers: list[str]`, `features: pd.DataFrame` (N×F at week), `X_seq: np.ndarray` [N,L,F], `trailing_returns: pd.DataFrame` (≤52 past weekly simple returns ending at week, rows = weeks, cols = tickers), `past_excess: pd.DataFrame` (realized fwd_excess for weeks strictly before week).
  - `class Strategy(Protocol)`: `fit(train: Split, val: Split) -> None`; `weights(ctx: WeekContext) -> pd.Series | None`.
  - `@dataclass Split`: `X [T,N,L,F]`, `y_excess [T,N]`, `y_ret [T,N]`, `weeks`, `tickers`, `panel` (the panel rows for those weeks).
  - `walk_forward(panel: pd.DataFrame, factories: dict[str, Callable[[], Strategy]], test_years: Iterable[int]) -> BacktestResult`, with `@dataclass BacktestResult`: `returns: pd.DataFrame` (week × strategy, net of costs), `weights: pd.DataFrame` (MultiIndex week, strategy × ticker), `turnover: pd.DataFrame`.
- Engine rules:
  - A fresh strategy is built per test year and `fit` is called once on the train and validation splits.
  - If `weights` returns `None`, the engine keeps the previous weights (EW for the first week of the first test year) and logs the week.
  - The engine asserts the weight invariants from Global Constraints.
  - Cost = 0.001 × L1 turnover, charged in the week the trade happens.
  - Net return = w·fwd_ret − cost.

- [ ] **Step 1: Write failing tests**

```python
def test_hand_computed_two_asset_portfolio():
    # 2 tickers, fwd_ret alternates; a fixed-weights strategy [0.6, 0.4], then [0.4, 0.6]
    # assert returns equal hand-computed w·r − 0.001·|Δw|₁ for each week

def test_none_keeps_previous_weights(): ...          # strategy returns None in week 3 → weights[3] == weights[2]
def test_split_dates_follow_spec(): ...              # for Y = 2016: train ends 2014-12-31, val = 2015, test = 2016
def test_invariant_violation_raises(): ...           # weight 0.2 → AssertionError
```

Use a `max_weight` argument (default `config.MAX_WEIGHT`) so the 2-asset test can pass `max_weight=1.0`.

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** — `feat: walk-forward backtest engine`

### Task 10: Baseline strategies (EW, MVO, BL-no-views, B tilt)

**Files:**
- Create: `src/sdpo/strategies.py`, `tests/test_strategies.py`

**Interfaces:**
- Consumes: `WeekContext`, `Strategy`, `Split` (Task 9); `cap_weights`, `shrunk_cov`, `max_sharpe_weights`, `implied_prior`, `bl_posterior` (Task 8).
- Produces: `EqualWeight`, `MeanVariance`, `BLNoViews`, `SentimentTilt(k=0.5)`, each implementing `Strategy` (`fit` is a no-op). `BASELINES: dict[str, Callable[[], Strategy]] = {"EW":…, "MVO":…, "BL-no-views":…, "B-tilt":…}`.
- `SentimentTilt` reads `ctx.features["sent_mean"]` (already z-scored): w ∝ max(0, 1 + k·z), then `cap_weights`.
- `MeanVariance`: μ = mean of `ctx.trailing_returns`, Σ = `shrunk_cov(ctx.trailing_returns)`, then `max_sharpe_weights`. `BLNoViews`: μ = `implied_prior(Σ, 1/N)` with the same Σ, then `max_sharpe_weights`.
- Add the fixture `synthetic_ctx` to `tests/conftest.py`: a `WeekContext` built from the synthetic panel at week index 100.

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("name", list(BASELINES))
def test_baseline_invariants(name, synthetic_ctx):
    w = BASELINES[name]().weights(synthetic_ctx)
    assert w is None or (w.ge(0).all() and abs(w.sum() - 1) < 1e-6 and w.max() <= 0.10 + 1e-6)

def test_tilt_overweights_high_sentiment(synthetic_ctx):
    w = SentimentTilt().weights(ctx_with_sent_mean_sorted)
    assert w.iloc[-1] > w.iloc[0]
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** — `feat: baseline strategies`

---

## Phase 5: Strategy A

### Task 11: LSTM and ridge forecasters

**Files:**
- Create: `src/sdpo/models/forecaster.py`, `tests/test_forecaster.py`

**Interfaces:**
- Produces:
  - `class LSTMForecaster(nn.Module)`: `__init__(n_features=7, hidden=32, dropout=0.2)`; `forward(x: Tensor[B,L,F]) -> Tensor[B]`.
  - `train_lstm(X_tr, y_tr, X_va, y_va, seed=42, max_epochs=50, patience=5, lr=1e-3, batch=256, device=None) -> LSTMForecaster`: flattens [T,N,L,F] to [T·N,L,F], skips NaN targets, uses Adam + MSE, restores the best validation-loss weights.
  - `class RidgeForecaster`: `fit(X, y)` / `predict(X) -> np.ndarray` on the flattened L·F input, `sklearn.linear_model.Ridge(alpha=1.0)`.
  - `predict(model, X: np.ndarray[..., L, F]) -> np.ndarray[...]`: any leading dims, works for both model types.
  - Add the fixture `synthetic_splits(signal=0.5)` to `tests/conftest.py`: `synthetic_market` → `build_weekly_panel` → `make_sequences`, split chronologically 60/20/20 into `(X_tr, y_tr, X_va, y_va, X_te, y_te)` using `y_excess`.

- [ ] **Step 1: Write failing tests**

```python
def test_lstm_recovers_planted_signal(synthetic_splits):   # CPU, max_epochs = 15
    X_tr, y_tr, X_va, y_va, X_te, y_te = synthetic_splits
    m = train_lstm(X_tr, y_tr, X_va, y_va, max_epochs=15, device="cpu")
    assert np.nanmean(rank_ic(predict(m, X_te), y_te)) > 0.1

def test_ridge_recovers_planted_signal(synthetic_split): ...   # same threshold
def test_lstm_is_deterministic_with_seed(): ...                # two runs → identical predictions (CPU)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS (should take under 60 s on CPU). **Step 5: Commit** — `feat(models): LSTM and ridge forecasters`

### Task 12: Strategy A (forecast → BL → optimizer)

**Files:**
- Modify: `src/sdpo/strategies.py`
- Test: `tests/test_strategy_a.py`

**Interfaces:**
- Consumes: Task 11 models; Task 8 portfolio functions; Task 9 engine.
- Produces: `class StrategyA(forecaster: Literal["lstm","ridge"] = "lstm")` implementing `Strategy`.
  - `fit` trains the forecaster on train and stores validation-year squared errors per ticker.
  - `weights(ctx)`:
    1. Q = forecast(ctx.X_seq).
    2. Store Q under `ctx.week`.
    3. Ω_ii = mean of squared (stored forecast − `ctx.past_excess`) over the last 26 weeks with both available; before 26 such weeks exist, use the validation errors. Floor Ω_ii at 1e-6.
    4. Σ = `shrunk_cov(ctx.trailing_returns)`; π = `implied_prior(Σ, 1/N)`; μ = `bl_posterior(Σ, π, I, Q, diag(Ω))`.
    5. Return `max_sharpe_weights(μ, Σ)`.
- `StrategyA.save(path: Path) -> None` / `StrategyA.load(path: Path) -> StrategyA`: `torch.save` of `{"kind", "state_dict" or pickled ridge, "val_sq_err": np.ndarray[N], "tickers"}`. Used by Task 13 (save) and Task 15 (load).
- Register `FACTORIES_A = {"A-lstm": lambda: StrategyA("lstm"), "A-ridge": lambda: StrategyA("ridge")}`.

- [ ] **Step 1: Write failing tests**

```python
def test_omega_uses_only_realized_past(): ...
    # feed past_excess with weeks < t only; Ω computed for week t must not change when future rows are appended

def test_planted_signal_end_to_end_beats_ew(synthetic_panel):   # signal = 0.5, test_years = last 2 synthetic years
    res = walk_forward(panel, {"EW": EqualWeight, **FACTORIES_A}, test_years=[...])
    s = {k: perf_metrics(res.returns[k])["sharpe"] for k in res.returns}
    assert s["A-lstm"] > s["EW"] and s["A-ridge"] > s["EW"]
```

The synthetic market starts in 2010 with 400 weeks (about 7.7 years), so use test years 2016 and 2017.

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** — `feat: Strategy A (LSTM → Black-Litterman)`

---

## Phase 6: Real-data results

### Task 13: Pipeline scripts, artifacts, README

**Files:**
- Create: `scripts/05_features.py`, `scripts/06_backtest.py`, `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `artifacts/panel_<model>.parquet` for each NLP model.
  - `artifacts/returns.parquet` (week × strategy; strategy names suffixed `@<nlp_model>` only for the sentiment strategies run per NLP model).
  - `artifacts/weights.parquet`, `artifacts/metrics.json` (`{strategy: {...perf_metrics, sharpe_ci: [lo, hi]}}`), `artifacts/ic.json` (`{strategy: {"mean_ic": float, "t_stat": float}}`).
  - `artifacts/models/A-lstm.pt` (trained on everything up to 2019, validated on 2020; used by the live tab).
  - `artifacts/sentiment_weekly.parquet` (raw, un-z-scored sent_mean and news_count per week and ticker, for the dashboard).
- Strategy C is added to `06_backtest.py` only if `importlib.util.find_spec("sdpo.models.e2e")` finds it.

- [ ] **Step 1: Write the scripts.** `06_backtest.py` runs baselines once (sentiment from the best NLP model) and runs A-lstm, A-ridge, and B-tilt for each of vader, finbert, and finetuned (the ablation).
- [ ] **Step 2: Manual run:** `python scripts/05_features.py && python scripts/06_backtest.py` → prints the metrics table. Spot-check that EW's CAGR roughly matches an equal-weight buy of the 30 tickers.
- [ ] **Step 3: Write `README.md`:** a one-paragraph pitch, an architecture diagram (copied from the spec), how to reproduce (Kaggle download → scripts 01–06 → `streamlit run app/dashboard.py`), the results table pasted from `metrics.json`, the NLP comparison table, the limitations (spec §9) verbatim, and a CI badge.
- [ ] **Step 4: Commit** — `feat: end-to-end pipeline, results, README`

---

## Phase 7: Dashboard

### Task 14: Dashboard tabs 1–3

**Files:**
- Create: `app/dashboard.py`, `tests/test_dashboard.py`, `tests/fixtures/artifacts/` (tiny fake artifacts written by a fixture, not committed as binaries)

**Interfaces:**
- Consumes: `artifacts/*` (Task 13 formats). The artifacts directory comes from env `SDPO_ARTIFACTS` (default `config.ARTIFACTS_DIR`) so tests can point it at a temp directory.
- Produces: `st.tabs(["Backtest", "Sentiment", "NLP models", "Live"])`. Loaders are wrapped in `@st.cache_data`. Plotly charts: cumulative returns, drawdown, sentiment vs price (two y-axes), and confusion-matrix heatmaps. If `artifacts/` is missing, show `st.info("Run scripts 01–06 first")` instead of a traceback.

- [ ] **Step 1: Write failing tests** with `streamlit.testing.v1.AppTest`

```python
def test_dashboard_renders_with_artifacts(fake_artifacts, monkeypatch):
    monkeypatch.setenv("SDPO_ARTIFACTS", str(fake_artifacts))
    at = AppTest.from_file("app/dashboard.py").run(timeout=30)
    assert not at.exception

def test_dashboard_without_artifacts_shows_info(tmp_path, monkeypatch):
    monkeypatch.setenv("SDPO_ARTIFACTS", str(tmp_path / "missing"))
    at = AppTest.from_file("app/dashboard.py").run(timeout=30)
    assert not at.exception and any("scripts" in i.value for i in at.info)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Manual check:** `streamlit run app/dashboard.py` on the real artifacts; click every tab.
- [ ] **Step 6: Commit** — `feat(app): dashboard results, sentiment, NLP tabs`

### Task 15: Live tab (Finnhub)

**Files:**
- Modify: `src/sdpo/data/news.py`, `app/dashboard.py`
- Create: `src/sdpo/live.py`, `tests/test_live.py`

**Interfaces:**
- Produces:
  - `class FinnhubError(Exception)`.
  - `fetch_finnhub_news(ticker: str, start: date, end: date, api_key: str, session: requests.Session | None = None) -> pd.DataFrame`: calls `GET https://finnhub.io/api/v1/company-news?symbol=&from=&to=&token=` (timeout 10 s) and returns columns ticker, headline, date (from the `datetime` unix field, converted to a US/Eastern date). A non-200 response or invalid JSON raises `FinnhubError`; an empty list returns an empty frame with those columns.
  - `suggest_weights(news: pd.DataFrame, prices: pd.DataFrame, scorer, model_path: Path) -> pd.Series`:
    1. score the headlines
    2. `build_weekly_panel`
    3. take the last available week
    4. run `StrategyA.load(model_path)`; Ω = its stored `val_sq_err`
    5. return the weights

    It also looks for C's weights when `artifacts/models/C.pt` exists and the e2e module is importable.
- Live tab behavior:
  - The key comes from env or `st.secrets`. If it's missing, show `st.warning` with setup instructions.
  - On `FinnhubError`, show `st.error` naming the ticker.
  - Delisted tickers (no price data from yfinance) are dropped and listed with `st.caption`.
  - Results are cached with `st.cache_data(ttl=3600)`.
  - The banner text is exactly "Research demo — not investment advice."

- [ ] **Step 1: Write failing tests** (mocked session, no network)

```python
def test_empty_news_is_empty_frame():
    df = fetch_finnhub_news("AAPL", d0, d1, "k", session=fake_session(200, []))
    assert df.empty and list(df.columns) == ["ticker", "headline", "date"]

def test_http_error_raises():
    with pytest.raises(FinnhubError):
        fetch_finnhub_news("AAPL", d0, d1, "k", session=fake_session(429, {}))

def test_suggest_weights_with_no_news_ticker(synthetic, tmp_model): ...   # one ticker has no news → valid weights

def test_live_tab_without_key_warns(fake_artifacts, monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    at = AppTest.from_file("app/dashboard.py").run(timeout=30)
    assert not at.exception and at.warning
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Manual check:** set `FINNHUB_API_KEY`, open the Live tab, and confirm weights appear and sum to 100%.
- [ ] **Step 6: Commit** — `feat(app): live suggested weights from Finnhub news`

---

## Phase 8 (optional): Strategy C — end-to-end

Skip this phase entirely if time is short; nothing earlier depends on it.

### Task 16: End-to-end Sharpe-trained portfolio network

**Files:**
- Create: `src/sdpo/models/e2e.py`, `tests/test_e2e.py`
- Modify: `src/sdpo/strategies.py` (this is the only place outside `e2e.py` that imports it, inside a guarded `try/except ImportError` registering `FACTORIES_C`)

**Interfaces:**
- Produces:
  - `class E2EPortfolioNet(nn.Module)`: `__init__(n_features=7, hidden=32, dropout=0.2)`; `forward(x: Tensor[B,N,L,F]) -> Tensor[B,N]` (shared LSTM per ticker → linear → softmax over N).
  - `sharpe_loss(w: Tensor[T,N], r_next: Tensor[T,N], lambda_eq: float, cost: float = 0.001) -> Tensor`: spec §6 formula; w_{−1} = w_0, so the first step has no cost. **Body reserved for the human (Learn by Doing).**
  - `train_e2e(train: Split, val: Split, lambda_eq: float, seed=42, max_epochs=100, patience=10, lr=1e-3, device=None) -> E2EPortfolioNet`: overlapping 52-week chunks with stride 4, shuffled each epoch; early stopping on validation Sharpe.
  - `class StrategyC` implementing `Strategy`. `fit` grid-searches λ_eq over `config.LAMBDA_EQ_GRID` by validation Sharpe. `weights` = model output, then `cap_weights`. `FACTORIES_C = {"C-e2e": StrategyC}`.
  - File: `artifacts/models/C.pt`.

- [ ] **Step 1: Write failing tests**

```python
def test_outputs_are_simplex():
    w = E2EPortfolioNet()(torch.randn(4, 30, 12, 7))
    assert (w >= 0).all() and torch.allclose(w.sum(-1), torch.ones(4))

def test_gradients_reach_all_params():
    loss = sharpe_loss(net(x), r, lambda_eq=1.0); loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())

def test_overfits_tiny_dataset(): ...          # 60 weeks, 5 tickers, 300 epochs → train Sharpe > 3
def test_cost_term_penalizes_turnover(): ...   # same returns, alternating weights → higher loss than constant weights
def test_planted_signal_beats_ew(synthetic_panel): ...   # walk_forward with C-e2e, Sharpe > EW
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement everything except the `sharpe_loss` body. Insert `TODO(human)` and hand off the Learn by Doing request.**
- [ ] **Step 4: After the human implements it, run** `pytest tests/test_e2e.py -v` → PASS.
- [ ] **Step 5: Re-run** `python scripts/06_backtest.py` → C-e2e rows appear; save `artifacts/models/C.pt`; update the README results table.
- [ ] **Step 6: Commit** — `feat(models): optional end-to-end Sharpe-trained Strategy C`

---

## Final verification

- [ ] `ruff check . && pytest -q` → all pass, with no network access (temporarily disable Wi-Fi to confirm once).
- [ ] Fresh clone → follow the README reproduce steps up to the dashboard using the existing artifacts.
- [ ] Spec §1 success criteria 1–5 each checked off against an artifact or test.
