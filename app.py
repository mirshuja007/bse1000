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

from src.auth import (
    AuthError,
    exchange_request_token,
    get_login_url,
    load_cached_access_token,
    login_automated,
    new_kite_client,
)
from src.config import KiteCredentials, app_password, deep_merge, load_config
from src.data_fetcher import fetch_benchmark_history, fetch_universe_history, resolve_benchmark_token
from src.instruments import build_nse_mapping, build_universe_mapping, combine_mappings, load_mapping, load_nse_mapping
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
# Kite login panel - two explicit methods, since the automated TOTP flow can
# break if Zerodha tweaks their login pages, and the manual flow can't use a
# blocking input() prompt inside Streamlit like the CLI does.
#
# Streamlit resets st.session_state on every new browser tab/session, so
# without this, you'd have to click through a login every single time you
# open the app - even minutes after someone else (or you, in another tab)
# already logged in today. @st.cache_resource is shared across ALL sessions
# hitting this same running app process, so the very first successful
# login of the day is reused silently by everyone until it expires -
# no button click needed for the common case.
# ---------------------------------------------------------------------------
@st.cache_resource(ttl=1800, show_spinner=False)
def _shared_auto_kite_client(cache_key: str):
    creds = KiteCredentials()
    cached_token = load_cached_access_token()
    if cached_token and creds.api_key:
        kite = new_kite_client(creds.api_key)
        kite.set_access_token(cached_token)
        return kite
    if not creds.is_complete():
        return None
    try:
        return login_automated(creds)
    except AuthError:
        # Cached as "unavailable" for the ttl window too, so a blocked/failed
        # attempt doesn't get retried on every single widget interaction.
        return None


def render_login_panel() -> None:
    st.sidebar.header("Kite Connect Login")
    creds = KiteCredentials()

    if "kite" not in st.session_state:
        auto_kite = _shared_auto_kite_client(datetime.now().strftime("%Y-%m-%d-%H"))
        if auto_kite is not None:
            st.session_state.kite = auto_kite
            st.session_state.login_method = "auto"

    if "kite" in st.session_state:
        st.sidebar.success(f"Logged in ({st.session_state.get('login_method', 'unknown')})")
        if st.sidebar.button("Log out"):
            for key in ("kite", "login_method"):
                st.session_state.pop(key, None)
            st.rerun()
        return

    cached_token = load_cached_access_token()
    if cached_token and creds.api_key:
        st.sidebar.info("A cached session token from earlier today was found.")
        if st.sidebar.button("Use cached session"):
            kite = login_automated(creds) if creds.is_complete() else None
            if kite is None:
                kite = new_kite_client(creds.api_key)
                kite.set_access_token(cached_token)
            st.session_state.kite = kite
            st.session_state.login_method = "cached"
            st.rerun()

    method = st.sidebar.radio(
        "Login method", ["Automated (TOTP)", "Manual (paste token)"], key="login_method_choice"
    )

    if method == "Automated (TOTP)":
        st.sidebar.caption("Uses KITE_USER_ID / KITE_PASSWORD / KITE_TOTP_SECRET from your .env")
        if st.sidebar.button("Login with TOTP", type="primary"):
            if not creds.is_complete():
                st.sidebar.error("Missing in .env: " + ", ".join(creds.missing_fields()))
            else:
                try:
                    with st.spinner("Logging in via automated TOTP flow..."):
                        kite = login_automated(creds)
                    st.session_state.kite = kite
                    st.session_state.login_method = "TOTP"
                    st.rerun()
                except AuthError as exc:
                    st.sidebar.error(f"Automated login failed: {exc}")

    else:
        if not creds.api_key or not creds.api_secret:
            st.sidebar.error("KITE_API_KEY / KITE_API_SECRET missing from .env")
        else:
            try:
                login_url = get_login_url(creds)
                st.sidebar.markdown(f"**1.** [Log in on Kite]({login_url})")
            except AuthError as exc:
                st.sidebar.error(str(exc))
                login_url = None

            st.sidebar.caption(
                "2. After logging in you'll land on a URL containing `request_token=...` "
                "- paste that token, or the whole URL, below."
            )
            token_input = st.sidebar.text_input("request_token", key="manual_request_token")
            if st.sidebar.button("Submit token", type="primary"):
                if not token_input.strip():
                    st.sidebar.error("Paste the request_token (or redirect URL) first.")
                else:
                    try:
                        with st.spinner("Exchanging request token..."):
                            kite = exchange_request_token(token_input, creds)
                        st.session_state.kite = kite
                        st.session_state.login_method = "manual token"
                        st.rerun()
                    except Exception as exc:  # kiteconnect raises assorted exception types
                        st.sidebar.error(f"Login failed: {exc}")


render_login_panel()


# ---------------------------------------------------------------------------
# Which constituent list(s) to scan. Kept separate from the trading-style
# preset below since it's a "what universe" choice, not a "how to score it"
# choice. BSE 1000 and Nifty 500 overlap heavily but aren't identical -
# picking "Both" scans the union, deduped so a dual-listed company that
# resolves to the same NSE instrument from both lists is only scored once
# (see combine_mappings in src/instruments.py).
# ---------------------------------------------------------------------------
UNIVERSE_OPTIONS = {
    "BSE 1000": "bse1000",
    "Nifty 500": "nifty500",
    "Both (deduped)": "both",
}


def render_universe_selector() -> str:
    st.sidebar.header("Universe")
    choice = st.sidebar.radio("Scan which constituents?", list(UNIVERSE_OPTIONS.keys()), key="universe_choice")
    st.sidebar.caption(
        "Nifty 500 resolves faster and more reliably (exact NSE symbol match) than BSE 1000 "
        "(fuzzy name match, since BSE codes aren't NSE symbols)."
    )
    return UNIVERSE_OPTIONS[choice]


selected_universe = render_universe_selector()


# ---------------------------------------------------------------------------
# Swing vs positional presets - a curated starting point for each trading
# style, so someone doesn't have to understand and hand-tune 15+ sliders
# before getting a sensible scan. Power users can still adjust anything
# below after picking a preset; picking "Custom" just leaves whatever is
# currently in the config alone.
# ---------------------------------------------------------------------------
SWING_PRESET = {
    "filters": {
        "liquidity": {"min_price": 30, "min_avg_turnover_cr": 5},
        "volume": {"surge_multiplier": 1.8},
        "dma50_breakout": {"lookback_days": 5},
        "dma200_breakout": {"enabled": False},
        "trend_filter": {"require_50_above_200": False},
        "rsi": {"min_rsi": 55, "max_rsi": 85, "flag_above": 70},
        "donchian_breakout": {"period": 15},
        "adx": {"min_adx": 18},
        "supertrend": {"period": 7, "multiplier": 2.0, "flip_lookback_days": 2},
    },
    "conviction": {
        "weights": {
            "trend": 15,
            "momentum": 25,
            "volume": 20,
            "breakout_quality": 25,
            "relative_strength": 10,
            "sector_strength": 5,
        },
        "extended_penalty": {"pct_above_pivot_threshold": 8},
        "rsi_overbought_penalty": {"threshold": 78},
    },
}

POSITIONAL_PRESET = {
    "filters": {
        "liquidity": {"min_price": 30, "min_avg_turnover_cr": 5},
        "volume": {"surge_multiplier": 1.5},
        "dma50_breakout": {"lookback_days": 10},
        "dma200_breakout": {"enabled": True},
        "trend_filter": {"require_50_above_200": True},
        "rsi": {"min_rsi": 60, "max_rsi": 80, "flag_above": 70},
        "donchian_breakout": {"period": 50},
        "adx": {"min_adx": 22},
        "supertrend": {"period": 10, "multiplier": 3.0, "flip_lookback_days": 5},
    },
    "conviction": {
        "weights": {
            "trend": 15,
            "momentum": 25,
            "volume": 20,
            "breakout_quality": 25,
            "relative_strength": 10,
            "sector_strength": 5,
        },
        "extended_penalty": {"pct_above_pivot_threshold": 15},
        "rsi_overbought_penalty": {"threshold": 82},
    },
}

# Mirrors a specific 3-rule Chartink screener a user shared:
#   Daily Close > 1-day-ago Max(Daily High, 125)   -> 125-day high breakout
#   Daily RSI(14) < 70                             -> not yet deeply overbought
#   Daily Volume > 1-day-ago SMA(Daily Volume, 125) -> above-average participation
# Everything else our scanner normally requires (50/200DMA structure, ADX
# trend strength, Supertrend flip) is switched OFF here so this preset
# behaves like that screener rather than "that screener plus all our other
# filters stacked on top." The one addition is a minimum-liquidity floor
# (price/turnover) that the original screener didn't have - kept on so
# results aren't dominated by illiquid names, called out here rather than
# left silent. Note our RSI/volume comparisons use <= / >= at the boundary
# where Chartink's are strictly < / > - only matters if a value lands
# exactly on 70 or exactly on the average, which is rare.
CHARTINK_BREAKOUT_PRESET = {
    "filters": {
        "liquidity": {"enabled": True, "min_price": 20, "min_avg_turnover_cr": 3},
        "volume": {"enabled": True, "surge_multiplier": 1.0, "lookback": 125},
        "dma50_breakout": {"enabled": False},
        "dma200_breakout": {"enabled": False},
        "trend_filter": {"enabled": False},
        "rsi": {"enabled": True, "min_rsi": 0, "max_rsi": 70, "flag_above": 70},
        "donchian_breakout": {"enabled": True, "period": 125},
        "adx": {"enabled": False},
        "supertrend": {"enabled": False},
    },
}

PRESET_LABELS = {
    "Custom (manual)": None,
    "Swing (3-7 days)": SWING_PRESET,
    "Positional (8-15 days)": POSITIONAL_PRESET,
    "125-Day Breakout (Chartink-style)": CHARTINK_BREAKOUT_PRESET,
}

# Presets for what to overlay on the Stock Detail price chart - independent
# of the scan-side trading-style preset above, since which lines are useful
# to look at is a matter of personal chart-reading taste, not strategy.
CHART_VIEW_PRESETS = {
    "Full (DMAs + Supertrend)": {"dma": True, "supertrend": True},
    "Supertrend focus": {"dma": False, "supertrend": True},
    "DMA focus": {"dma": True, "supertrend": False},
}


def render_preset_selector() -> None:
    st.sidebar.header("Trading style preset")
    choice = st.sidebar.selectbox("Preset", list(PRESET_LABELS.keys()), key="preset_choice")
    st.sidebar.caption(
        "Applies a curated set of filter thresholds and conviction weights for the chosen "
        "holding period. You can still fine-tune anything below afterward."
    )
    if choice != st.session_state.get("_applied_preset"):
        preset = PRESET_LABELS[choice]
        st.session_state.config = deep_merge(load_config(), preset) if preset else load_config()
        st.session_state._applied_preset = choice
        st.rerun()


# ---------------------------------------------------------------------------
# Config state
# ---------------------------------------------------------------------------
if "config" not in st.session_state:
    st.session_state.config = load_config()
    st.session_state._applied_preset = "Custom (manual)"

render_preset_selector()

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

    st.sidebar.subheader("Supertrend")
    cfg["filters"]["supertrend"]["enabled"] = st.sidebar.checkbox(
        "Require a recent bullish Supertrend flip", cfg["filters"]["supertrend"]["enabled"]
    )
    cfg["filters"]["supertrend"]["period"] = st.sidebar.slider(
        "Supertrend ATR period", 5, 20, int(cfg["filters"]["supertrend"]["period"])
    )
    cfg["filters"]["supertrend"]["multiplier"] = st.sidebar.slider(
        "Supertrend ATR multiplier", 1.0, 5.0, float(cfg["filters"]["supertrend"]["multiplier"]), 0.5
    )
    cfg["filters"]["supertrend"]["flip_lookback_days"] = st.sidebar.slider(
        "Count a flip as \"fresh\" within (days)", 1, 15, int(cfg["filters"]["supertrend"]["flip_lookback_days"])
    )
    st.sidebar.caption(
        "Lower period/multiplier = more reactive, more signals (better for short swings). "
        "Higher = smoother, fewer but more established signals (better for positional holds)."
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

def load_selected_mapping(kite, universe_choice: str, force_refresh: bool = False) -> pd.DataFrame:
    """Build/load the mapping(s) for whichever universe(s) the sidebar has
    selected, combining them (deduped) when both are chosen."""
    mappings = []
    if universe_choice in ("bse1000", "both"):
        mappings.append(
            build_universe_mapping(kite, force_refresh_instruments=force_refresh)
            if force_refresh
            else load_mapping(refresh_with_kite=kite)
        )
    if universe_choice in ("nifty500", "both"):
        mappings.append(
            build_nse_mapping(kite, force_refresh_instruments=force_refresh)
            if force_refresh
            else load_nse_mapping(refresh_with_kite=kite)
        )
    return combine_mappings(*mappings)


col1, col2, col3 = st.columns([1, 1, 2])
run_clicked = col1.button("Run scan", type="primary")
refresh_mapping_clicked = col2.button("Refresh instrument mapping")

if (run_clicked or refresh_mapping_clicked) and "kite" not in st.session_state:
    st.warning("Log in to Kite first using the panel in the sidebar.")
    run_clicked = refresh_mapping_clicked = False

if refresh_mapping_clicked:
    kite = st.session_state.kite
    with st.spinner("Rebuilding universe->Kite instrument mapping..."):
        mapping = load_selected_mapping(kite, selected_universe, force_refresh=True)
        st.session_state.mapping = mapping
    st.success(f"Mapping rebuilt: {mapping['resolved'].sum()}/{len(mapping)} constituents resolved.")

if run_clicked:
    kite = st.session_state.kite

    with st.spinner("Loading universe mapping..."):
        mapping = load_selected_mapping(kite, selected_universe)
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

    history = {r.security_code: r.bars for r in results if r.ok}
    n_failed = sum(1 for r in results if not r.ok)

    with st.spinner("Scoring conviction..."):
        result_df, enriched_cache, skipped = run_scan(history, resolved, cfg, benchmark)

    st.session_state.result_df = result_df
    st.session_state.enriched_cache = enriched_cache
    st.session_state.scan_time = datetime.now()
    st.session_state.scan_diagnostics = {
        "universe_total": len(mapping),
        "resolved": len(resolved),
        "fetch_ok": len(history),
        "fetch_failed": n_failed,
        "skipped_insufficient_history": len(skipped),
        "sample_fetch_errors": [r.error for r in results if not r.ok][:5],
    }

if "result_df" in st.session_state:
    result_df = st.session_state.result_df
    scan_time = st.session_state.scan_time
    diag = st.session_state.get("scan_diagnostics", {})
    st.caption(f"Last scan: {scan_time.strftime('%Y-%m-%d %H:%M:%S')} - {len(result_df)} stocks scored")

    if diag:
        with st.expander("Scan diagnostics", expanded=result_df.empty):
            st.write(
                f"Universe: {diag.get('universe_total', '?')} constituents · "
                f"Resolved to a Kite instrument: {diag.get('resolved', '?')} · "
                f"History fetched OK: {diag.get('fetch_ok', '?')} · "
                f"Fetch failures: {diag.get('fetch_failed', '?')} · "
                f"Skipped (insufficient history): {diag.get('skipped_insufficient_history', '?')}"
            )
            if diag.get("sample_fetch_errors"):
                st.write("Sample fetch errors:")
                for err in diag["sample_fetch_errors"]:
                    st.code(err)

    if result_df.empty:
        st.error(
            "The scan returned zero stocks. Check the diagnostics above - the most common causes are "
            "instrument mapping resolving 0 stocks (try 'Refresh instrument mapping'), every history "
            "fetch failing (often an expired/invalid Kite session - log out and back in), or "
            "`data.history_days` in the config being too short for the 200DMA to compute."
        )
    else:
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
            "universe",
            "sector",
            "conviction_score",
            "conviction_tier",
            "close",
            "stop_loss",
            "target",
            "rsi",
            "adx",
            "volume_surge",
            "pct_above_sma50",
            "roc5",
            "roc10",
            "relative_return_20d",
            "donchian_breakout",
            "supertrend_bullish",
            "supertrend_flip_recent",
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
                options=view["security_code"].tolist(),
                format_func=lambda code: f"{view.set_index('security_code').loc[code, 'company_name']} ({view.set_index('security_code').loc[code, 'tradingsymbol']})",
            )
            row = result_df.set_index("security_code").loc[selected]
            enriched = st.session_state.enriched_cache.get(selected)

            if row.get("explanation"):
                st.info(f"**In plain terms:** {row['explanation']}")

            score_col, chart_col = st.columns([1, 3])
            with score_col:
                st.metric("Conviction score", f"{row['conviction_score']:.1f}", row["conviction_tier"])

                if pd.notna(row.get("stop_loss")) and pd.notna(row.get("target")):
                    risk_col, reward_col = st.columns(2)
                    risk_col.metric("Suggested stop", f"₹{row['stop_loss']:.2f}", f"-{row['risk_pct']:.1f}%")
                    reward_col.metric("Suggested target", f"₹{row['target']:.2f}", f"+{row['reward_pct']:.1f}%")
                    st.caption(
                        "ATR-based (1.5x risk / 3x reward from close) - a starting frame, "
                        "not a recommendation. Size and confirm your own risk before entering."
                    )

                if pd.notna(row.get("supertrend_stop")):
                    st.metric("Supertrend trailing stop", f"₹{row['supertrend_stop']:.2f}")
                    st.caption(
                        "Follows price up as the trend continues; a daily close below this "
                        "line is the Supertrend's signal to exit or trail your stop tighter."
                    )

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

                    chart_view = st.radio(
                        "Chart overlays",
                        list(CHART_VIEW_PRESETS.keys()),
                        horizontal=True,
                        key="chart_view_preset",
                    )
                    show_dma = CHART_VIEW_PRESETS[chart_view]["dma"]
                    show_supertrend = CHART_VIEW_PRESETS[chart_view]["supertrend"] and "supertrend" in tail.columns

                    price_title = " / ".join(
                        ["Price"]
                        + (["50DMA", "200DMA"] if show_dma else [])
                        + (["Supertrend"] if show_supertrend else [])
                    )
                    fig = make_subplots(
                        rows=3,
                        cols=1,
                        shared_xaxes=True,
                        row_heights=[0.55, 0.2, 0.25],
                        vertical_spacing=0.03,
                        subplot_titles=(price_title, "Volume", "RSI"),
                    )
                    fig.add_trace(
                        go.Candlestick(
                            x=tail["date"], open=tail["open"], high=tail["high"], low=tail["low"], close=tail["close"],
                            name="Price",
                        ),
                        row=1,
                        col=1,
                    )
                    if show_dma:
                        fig.add_trace(go.Scatter(x=tail["date"], y=tail["sma50"], name="50DMA", line=dict(width=1.5)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=tail["date"], y=tail["sma200"], name="200DMA", line=dict(width=1.5)), row=1, col=1)
                    if show_supertrend:
                        st_bullish = tail["supertrend"].where(tail["supertrend_bullish"])
                        st_bearish = tail["supertrend"].where(~tail["supertrend_bullish"])
                        fig.add_trace(
                            go.Scatter(
                                x=tail["date"], y=st_bullish, name="Supertrend (bullish)",
                                line=dict(width=1.5, color="green"), connectgaps=False,
                            ),
                            row=1, col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=tail["date"], y=st_bearish, name="Supertrend (bearish)",
                                line=dict(width=1.5, color="red"), connectgaps=False,
                            ),
                            row=1, col=1,
                        )
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
