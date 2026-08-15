"""BSE 1000 Momentum Scanner - Streamlit dashboard.

Run with:  streamlit run app.py

Lets you tune every screening/scoring parameter live, run a scan against
your Kite Connect account, and drill into any candidate's chart and
conviction-score breakdown. Designed for a 1-15 trading day swing/momentum
horizon.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.auth import get_authenticated_kite
from src.config import app_password, load_config
from src.data_fetcher import fetch_benchmark_history, fetch_universe_history, resolve_benchmark_token
from src.instruments import build_universe_mapping, load_mapping
from src.scanner import run_scan

st.set_page_config(page_title="BSE 1000 Momentum Scanner", layout="wide")


# ---------------------------------------------------------------------------
# Optional app-level password gate (APP_PASSWORD in .env). Skipped entirely
# if you don't set one.
# ---------------------------------------------------------------------------
def check_password() -> bool:
    required = app_password()
    if not required:
        return True
    if st.session_state.get("authed"):
        return True
    pw = st.text_input("App password", type="password")
    if pw == required and pw:
        st.session_state["authed"] = True
        return True
    if pw:
        st.error("Incorrect password")
    return False


if not check_password():
    st.stop()


# ---------------------------------------------------------------------------
# Config state
# ---------------------------------------------------------------------------
if "config" not in st.session_state:
    st.session_state.config = load_config()

cfg = st.session_state.config


def sidebar_controls(cfg: dict) -> dict:
    st.sidebar.header("Scan parameters")

    st.sidebar.subheader("Liquidity")
    cfg["filters"]["liquidity"]["enabled"] = st.sidebar.checkbox(
        "Enable liquidity filter", cfg["filters"]["liquidity"]["enabled"]
    )
    cfg["filters"]["liquidity"]["min_price"] = st.sidebar.number_input(
        "Min price (Rs)", value=float(cfg["filters"]["liquidity"]["min_price"]), step=5.0
    )
    cfg["filters"]["liquidity"]["min_avg_turnover_cr"] = st.sidebar.number_input(
        "Min 20d avg turnover (Rs cr)", value=float(cfg["filters"]["liquidity"]["min_avg_turnover_cr"]), step=0.5
    )

    st.sidebar.subheader("Volume")
    cfg["filters"]["volume"]["enabled"] = st.sidebar.checkbox(
        "Enable volume surge filter", cfg["filters"]["volume"]["enabled"]
    )
    cfg["filters"]["volume"]["surge_multiplier"] = st.sidebar.slider(
        "Volume surge (x avg)", 1.0, 5.0, float(cfg["filters"]["volume"]["surge_multiplier"]), 0.1
    )

    st.sidebar.subheader("Moving average breakouts")
    cfg["filters"]["dma50_breakout"]["enabled"] = st.sidebar.checkbox(
        "Require 50DMA breakout (recent)", cfg["filters"]["dma50_breakout"]["enabled"]
    )
    cfg["filters"]["dma50_breakout"]["lookback_days"] = st.sidebar.slider(
        "50DMA breakout lookback (days)", 1, 20, int(cfg["filters"]["dma50_breakout"]["lookback_days"])
    )
    cfg["filters"]["dma200_breakout"]["enabled"] = st.sidebar.checkbox(
        "Require 200DMA breakout (recent)", cfg["filters"]["dma200_breakout"]["enabled"]
    )
    cfg["filters"]["dma200_breakout"]["lookback_days"] = st.sidebar.slider(
        "200DMA breakout lookback (days)", 1, 30, int(cfg["filters"]["dma200_breakout"]["lookback_days"])
    )
    cfg["filters"]["trend_filter"]["enabled"] = st.sidebar.checkbox(
        "Require golden-cross regime (50 > 200 DMA)", cfg["filters"]["trend_filter"]["enabled"]
    )

    st.sidebar.subheader("RSI")
    cfg["filters"]["rsi"]["enabled"] = st.sidebar.checkbox("Enable RSI filter", cfg["filters"]["rsi"]["enabled"])
    rsi_range = st.sidebar.slider(
        "RSI range",
        0,
        100,
        (int(cfg["filters"]["rsi"]["min_rsi"]), int(cfg["filters"]["rsi"]["max_rsi"])),
    )
    cfg["filters"]["rsi"]["min_rsi"], cfg["filters"]["rsi"]["max_rsi"] = rsi_range
    cfg["filters"]["rsi"]["flag_above"] = st.sidebar.slider(
        "Flag as overbought above", 50, 100, int(cfg["filters"]["rsi"]["flag_above"])
    )

    st.sidebar.subheader("Breakout / trend strength")
    cfg["filters"]["donchian_breakout"]["enabled"] = st.sidebar.checkbox(
        "Require N-day high breakout", cfg["filters"]["donchian_breakout"]["enabled"]
    )
    cfg["filters"]["donchian_breakout"]["period"] = st.sidebar.slider(
        "Breakout lookback (days)", 5, 60, int(cfg["filters"]["donchian_breakout"]["period"])
    )
    cfg["filters"]["adx"]["enabled"] = st.sidebar.checkbox("Enable ADX filter", cfg["filters"]["adx"]["enabled"])
    cfg["filters"]["adx"]["min_adx"] = st.sidebar.slider(
        "Min ADX (trend strength)", 0, 50, int(cfg["filters"]["adx"]["min_adx"])
    )

    st.sidebar.header("Conviction score weights")
    st.sidebar.caption("Relative weights - don't need to sum to 100, they're normalized automatically.")
    weights = cfg["conviction"]["weights"]
    for key in list(weights.keys()):
        weights[key] = st.sidebar.slider(key.replace("_", " ").title(), 0, 40, int(weights[key]))

    st.sidebar.header("Penalties")
    cfg["conviction"]["extended_penalty"]["pct_above_pivot_threshold"] = st.sidebar.slider(
        "Extended-above-pivot threshold (%)",
        0,
        30,
        int(cfg["conviction"]["extended_penalty"]["pct_above_pivot_threshold"]),
    )
    cfg["conviction"]["rsi_overbought_penalty"]["threshold"] = st.sidebar.slider(
        "RSI overbought penalty threshold", 60, 100, int(cfg["conviction"]["rsi_overbought_penalty"]["threshold"])
    )

    return cfg


cfg = sidebar_controls(cfg)
st.session_state.config = cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("BSE 1000 Momentum Scanner")
st.caption(
    "Rule-based, fully transparent multi-factor screener for the 1-15 trading day momentum horizon. "
    "Not financial advice - use alongside your own risk management."
)

col1, col2, col3 = st.columns([1, 1, 2])
run_clicked = col1.button("Run scan", type="primary")
refresh_mapping_clicked = col2.button("Refresh instrument mapping")

if refresh_mapping_clicked:
    with st.spinner("Logging in and rebuilding BSE->Kite instrument mapping..."):
        kite = get_authenticated_kite()
        st.session_state.kite = kite
        mapping = build_universe_mapping(kite, force_refresh_instruments=True)
        st.session_state.mapping = mapping
    st.success(f"Mapping rebuilt: {mapping['resolved'].sum()}/{len(mapping)} constituents resolved.")

if run_clicked:
    with st.spinner("Logging in to Kite..."):
        kite = get_authenticated_kite()
        st.session_state.kite = kite

    with st.spinner("Loading universe mapping..."):
        mapping = load_mapping(refresh_with_kite=kite)
        resolved = mapping[mapping["resolved"] == True]  # noqa: E712
        st.session_state.mapping = mapping

    with st.spinner("Fetching benchmark index history..."):
        bench_token = resolve_benchmark_token(kite)
        benchmark = fetch_benchmark_history(kite, bench_token, cfg["data"]["history_days"])

    progress_bar = st.progress(0.0, text="Fetching stock history...")

    def progress(done, total):
        progress_bar.progress(done / total, text=f"Fetching stock history... {done}/{total}")

    with st.spinner("Fetching OHLCV history (rate-limited - this can take a few minutes for ~1000 names)..."):
        results = fetch_universe_history(
            kite,
            resolved,
            history_days=cfg["data"]["history_days"],
            max_requests_per_second=cfg["data"]["max_requests_per_second"],
            cache_ttl_hours=cfg["data"]["cache_ttl_hours"],
            progress_callback=progress,
        )
    progress_bar.empty()

    history = {r.bse_code: r.bars for r in results if r.ok}
    n_failed = sum(1 for r in results if not r.ok)
    if n_failed:
        st.warning(f"{n_failed} stocks failed to fetch and were skipped.")

    with st.spinner("Scoring conviction..."):
        result_df, enriched_cache, skipped = run_scan(history, resolved, cfg, benchmark)

    st.session_state.result_df = result_df
    st.session_state.enriched_cache = enriched_cache
    st.session_state.scan_time = datetime.now()

if "result_df" in st.session_state:
    result_df = st.session_state.result_df
    scan_time = st.session_state.scan_time
    st.caption(f"Last scan: {scan_time.strftime('%Y-%m-%d %H:%M:%S')} - {len(result_df)} stocks scored")

    tier_counts = result_df["conviction_tier"].value_counts()
    tiers_order = ["Very High Conviction", "High Conviction", "Moderate Conviction", "Watchlist"]
    cols = st.columns(len(tiers_order))
    for c, tier in zip(cols, tiers_order):
        c.metric(tier, int(tier_counts.get(tier, 0)))

    show_candidates_only = st.checkbox("Show only stocks passing all filters", value=True)
    min_score = st.slider("Minimum conviction score", 0, 100, 0)

    view = result_df.copy()
    if show_candidates_only:
        view = view[view["passes_all_filters"]]
    view = view[view["conviction_score"] >= min_score]

    display_cols = [
        "company_name",
        "tradingsymbol",
        "exchange",
        "sector",
        "conviction_score",
        "conviction_tier",
        "close",
        "rsi",
        "adx",
        "volume_surge",
        "pct_above_sma50",
        "roc5",
        "roc10",
        "relative_return_20d",
        "donchian_breakout",
        "notes",
    ]
    st.dataframe(view[display_cols], use_container_width=True, height=450)

    st.download_button(
        "Download this view as CSV",
        view.to_csv(index=False).encode("utf-8"),
        file_name=f"bse1000_scan_{scan_time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.subheader("Stock detail")
    if len(view):
        selected = st.selectbox(
            "Pick a stock",
            options=view["bse_code"].tolist(),
            format_func=lambda code: f"{view.set_index('bse_code').loc[code, 'company_name']} ({view.set_index('bse_code').loc[code, 'tradingsymbol']})",
        )
        row = result_df.set_index("bse_code").loc[selected]
        enriched = st.session_state.enriched_cache.get(selected)

        score_col, chart_col = st.columns([1, 3])
        with score_col:
            st.metric("Conviction score", f"{row['conviction_score']:.1f}", row["conviction_tier"])
            for label, key in [
                ("Trend", "score_trend"),
                ("Momentum", "score_momentum"),
                ("Volume", "score_volume"),
                ("Breakout quality", "score_breakout_quality"),
                ("Relative strength", "score_relative_strength"),
                ("Sector strength", "score_sector_strength"),
            ]:
                st.progress(min(row[key] / 100, 1.0), text=f"{label}: {row[key]:.0f}")
            if row["notes"]:
                st.info(row["notes"])

        with chart_col:
            if enriched is not None and not enriched.empty:
                tail = enriched.tail(150)
                fig = make_subplots(
                    rows=3,
                    cols=1,
                    shared_xaxes=True,
                    row_heights=[0.55, 0.2, 0.25],
                    vertical_spacing=0.03,
                    subplot_titles=("Price / 50DMA / 200DMA", "Volume", "RSI"),
                )
                fig.add_trace(
                    go.Candlestick(
                        x=tail["date"], open=tail["open"], high=tail["high"], low=tail["low"], close=tail["close"],
                        name="Price",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(go.Scatter(x=tail["date"], y=tail["sma50"], name="50DMA", line=dict(width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=tail["date"], y=tail["sma200"], name="200DMA", line=dict(width=1.5)), row=1, col=1)
                fig.add_trace(go.Bar(x=tail["date"], y=tail["volume"], name="Volume"), row=2, col=1)
                fig.add_trace(go.Scatter(x=tail["date"], y=tail["rsi"], name="RSI"), row=3, col=1)
                fig.add_hline(y=70, line_dash="dot", row=3, col=1)
                fig.add_hline(y=30, line_dash="dot", row=3, col=1)
                fig.update_layout(height=650, showlegend=False, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No stocks match the current filters/score threshold.")
else:
    st.info("Click **Run scan** to fetch live data from Kite and score the BSE 1000 universe.")
    st.caption(
        "First run will also build the BSE→Kite instrument mapping (data/universe_mapping.csv). "
        "Review that file afterwards for any low match_confidence rows."
    )
