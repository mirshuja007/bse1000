"""Breakout Radar: a two-tier, multi-timeframe watchlist for catching a
move earlier than "wait for the breakout candle."

- **Fresh Breakout**: already breaking out today - reuses the same
  donchian_breakout + volume_surge signals the main scanner already
  computes. "It's happening now."
- **Pre-Breakout Watch**: hasn't broken out yet, but shows the classic
  pre-breakout signature - contracting volatility (ATR% AND Bollinger
  Band width both shrinking) while price coils tightly just below its
  N-day high.

Read this before treating either tier as a forecast: no indicator can see
the future. This screens for a higher-probability *setup* (a documented,
widely-used technical pattern - volatility contraction before expansion),
not a predicted outcome. `historical_hit_rate_for_timeframe()` gives an
honest, explicitly-small-sample check of how often this specific
signature has actually preceded a double-digit move in this app's own
scanned universe and history window - not a rigorous multi-cycle
backtest, and it says so in its own output.

Reuses src/indicators.py's compute_indicators() on resampled weekly/
monthly bars (via src/timeframe.py) rather than duplicating indicator
logic - a "weekly RSI(14)" or "monthly Donchian(20)" is standard
multi-timeframe practice: same periods, applied to bars spanning more
calendar time.

Known limitation: monthly bars need ~20+ months of history for a
Donchian(20)/ATR(14)-based signal to have enough data to compute at all.
With this app's default `data.history_days` (400 calendar days = ~13
monthly bars), monthly precursor signals will typically read as
insufficient data - flagged explicitly (`insufficient_data: True`), never
silently guessed at. Daily and weekly both have enough history by
default. To get real monthly signals, raise `data.history_days` to
~1500-2000 (calendar days) - at the cost of slower scans / more Kite API
calls per stock.
"""
from __future__ import annotations

import pandas as pd

from src.indicators import compute_indicators, sma
from src.timeframe import resample_ohlcv

TIMEFRAME_RULES = {"daily": None, "weekly": "W-FRI", "monthly": "ME"}
MIN_BARS_FOR_SIGNAL = 30  # enough for a period-20 rolling window plus a lookback buffer


def bollinger_band_width(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """(upper band - lower band) / middle band - a standard measure of how
    tight/wide a stock's trading range is. A shrinking width ("squeeze")
    is a classic pre-breakout signature, independent of ATR contraction
    (which measures true-range volatility rather than closing-price
    dispersion) - requiring both to agree is a stricter bar than either
    alone."""
    middle = sma(close, period)
    std = close.rolling(period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return (upper - lower) / middle


def compute_precursor_signals(enriched: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Adds bb_width / bb_width_chg20 / pct_below_pivot / is_watch to a
    frame already run through compute_indicators() (needs donchian_high
    and atr_pct_chg20 to already be present)."""
    out = enriched.copy()
    r = config["filters"]["breakout_radar"]

    out["bb_width"] = bollinger_band_width(out["close"], r["bollinger_period"])
    out["bb_width_chg20"] = out["bb_width"] - out["bb_width"].shift(20)
    out["pct_below_pivot"] = (out["donchian_high"] - out["close"]) / out["donchian_high"] * 100

    not_yet_broken_out = out["close"] <= out["donchian_high"]
    coiling_tight = out["pct_below_pivot"] <= r["max_pct_below_pivot"]
    volatility_contracting = (out["atr_pct_chg20"] < 0) & (out["bb_width_chg20"] < 0)

    out["is_watch"] = (not_yet_broken_out & coiling_tight & volatility_contracting).fillna(False)
    return out


def _prepare_timeframe_df(raw_ohlcv: pd.DataFrame, timeframe: str, config: dict) -> pd.DataFrame | None:
    """Resample + run the indicator pipeline + precursor signals. Returns
    None (not an empty/partial frame) when there isn't enough history to
    compute a period-20 rolling window reliably - an explicit "don't know"
    rather than a misleading NaN-filled result."""
    rule = TIMEFRAME_RULES[timeframe]
    bars = raw_ohlcv if rule is None else resample_ohlcv(raw_ohlcv, rule)
    if len(bars) < MIN_BARS_FOR_SIGNAL:
        return None
    enriched = compute_indicators(bars, config, benchmark=None)
    return compute_precursor_signals(enriched, config)


def _bars_available(raw_ohlcv: pd.DataFrame, timeframe: str) -> int:
    rule = TIMEFRAME_RULES[timeframe]
    return len(raw_ohlcv) if rule is None else len(resample_ohlcv(raw_ohlcv, rule))


def evaluate_timeframe(raw_ohlcv: pd.DataFrame, timeframe: str, config: dict) -> dict:
    """Latest-bar snapshot for one stock at one timeframe."""
    prepared = _prepare_timeframe_df(raw_ohlcv, timeframe, config)
    if prepared is None:
        return {
            "timeframe": timeframe,
            "insufficient_data": True,
            "bars_available": _bars_available(raw_ohlcv, timeframe),
        }

    latest = prepared.iloc[-1]
    surge_threshold = config["filters"]["volume"]["surge_multiplier"]
    volume_surge = latest.get("volume_surge")
    is_fresh_breakout = bool(
        latest.get("donchian_breakout") and pd.notna(volume_surge) and volume_surge >= surge_threshold
    )

    def _num(key, decimals):
        val = latest.get(key)
        return None if pd.isna(val) else round(float(val), decimals)

    return {
        "timeframe": timeframe,
        "insufficient_data": False,
        "bars_available": len(prepared),
        "is_fresh_breakout": is_fresh_breakout,
        "is_watch": bool(latest.get("is_watch")),
        "pct_below_pivot": _num("pct_below_pivot", 1),
        "atr_pct_chg20": _num("atr_pct_chg20", 2),
        "bb_width_chg20": _num("bb_width_chg20", 4),
        "volume_surge": _num("volume_surge", 2),
    }


def scan_breakout_radar(
    raw_history: dict[str, pd.DataFrame], mapping: pd.DataFrame, config: dict, timeframes: list[str] | None = None
) -> pd.DataFrame:
    """raw_history: {security_code: raw daily OHLCV df} - e.g. reuse the
    main scan's enriched_cache directly; resample_ohlcv only touches the
    open/high/low/close/volume/date columns, so the extra indicator
    columns already on those frames are simply ignored."""
    timeframes = timeframes or list(TIMEFRAME_RULES.keys())
    name_by_code = mapping.set_index("security_code")["company_name_raw"].to_dict()
    tsym_by_code = mapping.set_index("security_code")["tradingsymbol"].to_dict()
    exch_by_code = mapping.set_index("security_code")["exchange"].to_dict()

    rows = []
    for code, df in raw_history.items():
        if df is None or df.empty:
            continue
        row = {
            "security_code": code,
            "company_name": name_by_code.get(code, code),
            "tradingsymbol": tsym_by_code.get(code, code),
            "exchange": exch_by_code.get(code, "NSE"),
        }
        watch_tfs, breakout_tfs = [], []
        for tf in timeframes:
            result = evaluate_timeframe(df, tf, config)
            for key, val in result.items():
                if key == "timeframe":
                    continue
                row[f"{tf}_{key}"] = val
            if result.get("is_watch"):
                watch_tfs.append(tf)
            if result.get("is_fresh_breakout"):
                breakout_tfs.append(tf)
        row["watch_timeframes"] = ", ".join(watch_tfs)
        row["breakout_timeframes"] = ", ".join(breakout_tfs)
        row["n_watch_timeframes"] = len(watch_tfs)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values("n_watch_timeframes", ascending=False).reset_index(drop=True)
    return result_df


def _historical_hit_rate_from_prepared(prepared: pd.DataFrame, forward_days: int, target_return_pct: float) -> dict:
    """Pure function on an already-prepared (resampled + indicator +
    precursor-signal) frame - no I/O, cheap to unit test with synthetic
    data. For each historical bar, checks whether the forward `high`
    reached at least `target_return_pct`% above that bar's close within
    the following `forward_days` bars - a "did it rally" check, not just
    a fixed-day-N close comparison, since a rally that peaks mid-window
    and pulls back afterward should still count as a hit."""
    reversed_high = prepared["high"][::-1]
    forward_max_high = reversed_high.shift(1).rolling(forward_days, min_periods=forward_days).max()[::-1]
    forward_return_pct = (forward_max_high - prepared["close"]) / prepared["close"] * 100

    hit = forward_return_pct >= target_return_pct
    flagged = prepared["is_watch"].fillna(False)
    valid = forward_return_pct.notna()  # excludes the trailing bars with incomplete forward data

    flagged_valid = flagged & valid
    baseline_valid = (~flagged) & valid

    n_flagged = int(flagged_valid.sum())
    n_flagged_hit = int((hit & flagged_valid).sum())
    n_baseline = int(baseline_valid.sum())
    n_baseline_hit = int((hit & baseline_valid).sum())

    return {
        "n_flagged": n_flagged,
        "n_flagged_hit": n_flagged_hit,
        "flagged_hit_rate_pct": round(n_flagged_hit / n_flagged * 100, 1) if n_flagged else None,
        "n_baseline": n_baseline,
        "n_baseline_hit": n_baseline_hit,
        "baseline_hit_rate_pct": round(n_baseline_hit / n_baseline * 100, 1) if n_baseline else None,
    }


def historical_hit_rate_for_timeframe(
    raw_ohlcv: pd.DataFrame, timeframe: str, config: dict, forward_days: int = 15, target_return_pct: float = 10.0
) -> dict | None:
    """None when there isn't enough history to prepare this timeframe at
    all (same insufficient-data guard as evaluate_timeframe)."""
    prepared = _prepare_timeframe_df(raw_ohlcv, timeframe, config)
    if prepared is None:
        return None
    return _historical_hit_rate_from_prepared(prepared, forward_days, target_return_pct)


def aggregate_historical_hit_rate(
    raw_history: dict[str, pd.DataFrame], timeframe: str, config: dict, forward_days: int = 15, target_return_pct: float = 10.0
) -> dict:
    """Sums per-stock hit-rate counts across the scanned universe, for one
    timeframe. Still just one ~1-2 year period for whatever history was
    fetched - not a multi-cycle backtest. `n_stocks_included` and
    `n_flagged` are reported prominently precisely so a small sample
    isn't mistaken for a validated edge."""
    totals = {"n_flagged": 0, "n_flagged_hit": 0, "n_baseline": 0, "n_baseline_hit": 0}
    n_stocks_included = 0

    for code, df in raw_history.items():
        if df is None or df.empty:
            continue
        result = historical_hit_rate_for_timeframe(df, timeframe, config, forward_days, target_return_pct)
        if result is None:
            continue
        n_stocks_included += 1
        for key in ("n_flagged", "n_flagged_hit", "n_baseline", "n_baseline_hit"):
            totals[key] += result[key]

    totals["n_stocks_included"] = n_stocks_included
    totals["flagged_hit_rate_pct"] = (
        round(totals["n_flagged_hit"] / totals["n_flagged"] * 100, 1) if totals["n_flagged"] else None
    )
    totals["baseline_hit_rate_pct"] = (
        round(totals["n_baseline_hit"] / totals["n_baseline"] * 100, 1) if totals["n_baseline"] else None
    )
    return totals
