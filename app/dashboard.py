"""Streamlit dashboard. Tabs 1-3 read only precomputed artifacts (no GPU, no network)."""

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sdpo import config
from sdpo.metrics import perf_metrics

ART = Path(os.environ.get("SDPO_ARTIFACTS", config.ARTIFACTS_DIR))
LABELS = ["negative", "neutral", "positive"]

st.set_page_config(page_title="Sentiment Portfolio", layout="wide")
st.title("Sentiment-Driven Portfolio Optimization")


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((ART / name).read_text())


@st.cache_data
def load_parquet(name: str) -> pd.DataFrame | None:
    path = ART / name
    return pd.read_parquet(path) if path.exists() else None


def backtest_tab() -> None:
    returns, turnover = load_parquet("returns.parquet"), load_parquet("turnover.parquet")
    ic = load_json("ic.json") if (ART / "ic.json").exists() else {}
    years = sorted(returns.index.year.unique())
    lo, hi = st.select_slider("Test years", options=years, value=(years[0], years[-1]))
    defaults = [c for c in returns.columns if "@" not in c or "finetuned" in c]
    chosen = st.multiselect("Strategies", list(returns.columns), default=defaults)
    if not chosen:
        st.warning("Pick at least one strategy.")
        return
    r = returns.loc[(returns.index.year >= lo) & (returns.index.year <= hi), chosen]

    rows = []
    for name in chosen:
        m = perf_metrics(r[name], turnover[name].reindex(r.index))
        rows.append({"Strategy": name, "CAGR": m["cagr"], "Vol": m["vol"], "Sharpe": m["sharpe"],
                     "Sortino": m["sortino"], "Max DD": m["max_dd"],
                     "Turnover/wk": m["avg_turnover"],
                     "Mean IC": ic.get(name, {}).get("mean_ic", float("nan"))})
    table = pd.DataFrame(rows).set_index("Strategy").sort_values("Sharpe", ascending=False)
    st.dataframe(table.style.format({"CAGR": "{:.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}",
                                     "Sortino": "{:.2f}", "Max DD": "{:.1%}",
                                     "Turnover/wk": "{:.2f}", "Mean IC": "{:.3f}"},
                                    na_rep="–"))
    if lo == years[0] and hi == years[-1]:
        full = load_json("metrics.json")
        st.caption("Full-period Sharpe 95% CI (block bootstrap): " + ", ".join(
            f"{n} [{full[n]['sharpe_ci'][0]:.2f}, {full[n]['sharpe_ci'][1]:.2f}]"
            for n in chosen if n in full))

    wealth = (1 + r).cumprod()
    st.plotly_chart(px.line(wealth, labels={"value": "Growth of $1", "index": "Week",
                                            "variable": "Strategy"},
                            title="Cumulative net returns"), use_container_width=True)
    drawdown = wealth / wealth.cummax() - 1
    # Lines, not px.area: area charts stack series, which would add drawdowns together.
    st.plotly_chart(px.line(drawdown, labels={"value": "Drawdown", "index": "Week",
                                              "variable": "Strategy"},
                            title="Drawdown").update_yaxes(tickformat=".0%"),
                    use_container_width=True)


def sentiment_tab() -> None:
    sent = load_parquet("sentiment_weekly.parquet")
    if sent is None:
        st.info("No sentiment table yet; run scripts/05_features.py.")
        return
    tickers = sorted(sent.index.get_level_values("ticker").unique())
    ticker = st.selectbox("Ticker", tickers, key="ticker")
    s = sent.xs(ticker, level="ticker")
    fig = go.Figure()
    fig.add_bar(x=s.index, y=s["sent_mean"], name="Weekly sentiment", yaxis="y2",
                opacity=0.4)
    fig.add_scatter(x=s.index, y=s["price"], name="Price", mode="lines")
    fig.update_layout(title=f"{ticker}: price vs weekly news sentiment",
                      yaxis={"title": "Price"},
                      yaxis2={"title": "Sentiment", "overlaying": "y", "side": "right",
                              "range": [-1, 1]})
    st.plotly_chart(fig, use_container_width=True)

    heads = load_parquet("headlines.parquet")
    if heads is not None:
        h = heads[heads["ticker"] == ticker].sort_values("score")
        left, right = st.columns(2)
        left.markdown("**Most negative headlines**")
        left.dataframe(h.head(10)[["date", "headline", "score"]], hide_index=True)
        right.markdown("**Most positive headlines**")
        right.dataframe(h.tail(10).iloc[::-1][["date", "headline", "score"]], hide_index=True)


def nlp_tab() -> None:
    ev = load_json("nlp_eval.json")
    models = [m for m in ev if m != "best"]
    st.dataframe(pd.DataFrame({m: {"Accuracy": ev[m]["accuracy"], "Macro-F1": ev[m]["macro_f1"]}
                               for m in models}).T.style.format("{:.3f}"))
    st.caption(f"Held-out set: Twitter financial-news validation split. "
               f"Used for features: **{ev.get('best', '?')}**.")
    for col, m in zip(st.columns(len(models)), models, strict=True):
        fig = px.imshow(ev[m]["confusion"], x=LABELS, y=LABELS, text_auto=True,
                        labels={"x": "Predicted", "y": "True"}, title=m,
                        color_continuous_scale="Blues")
        col.plotly_chart(fig, use_container_width=True)


def live_tab() -> None:
    st.info("Live suggested weights are added in the next build step.")


if not (ART / "metrics.json").exists():
    st.info(f"No results in {ART}. Run scripts 01–06 first (see README).")
    st.stop()

tabs = st.tabs(["Backtest", "Sentiment", "NLP models", "Live"])
with tabs[0]:
    backtest_tab()
with tabs[1]:
    sentiment_tab()
with tabs[2]:
    nlp_tab()
with tabs[3]:
    live_tab()
