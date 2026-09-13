"""False-Move Filter: a SECOND PASS on top of whichever trading-style
preset already selected a candidate (Manual / Swing / Positional /
Chartink-style) - not a replacement for that preset, an extra check
layered on top of it, same philosophy as src/fundamentals.py and
src/growth_screen.py.

Flags two specific setups, both built from data this app already fetched
from Kite for the main scan (no new data source, unlike the yfinance-based
fundamentals/growth screens):

- **distribution_risk**: a fresh breakout (close > N-day high, the same
  `donchian_breakout` the main scan already computes) where, within the
  last `pattern_lookback_days`, a classic top-reversal candle also printed
  (Bearish Engulfing / Evening Star / Shooting Star) AND the weekly RSI
  isn't confirming the daily strength (see `classify_direction_zone`
  below). Read together: price pushed to a new high, a reversal candle
  showed up right at that high, and the bigger timeframe doesn't endorse
  it - the textbook signature of smart money distributing into a breakout
  that retail is chasing.
- **accumulation_signal**: the mirror image. A fresh breakdown (close <
  N-day low - this module's own `donchian_low`, since the main scan only
  tracks the high side) where a bottom-reversal candle printed (Bullish
  Engulfing / Morning Star / Hammer) AND the weekly RSI is holding up
  better than the daily breakdown would suggest - the textbook signature
  of smart money accumulating into a breakdown that retail is panicking
  out of.

Why the candle pattern is only checked in a short window around the
breakout/breakdown, not anywhere on the chart: a reversal candle in the
middle of nowhere is noise; the same candle right at a level is a signal.
That's a deliberate design choice carried over from the source material
this was built from, not an assumption buried in the code.

Every pattern definition and threshold below is the standard, commonly
cited version of that pattern (documented on each function) - this is
ordinary, backtestable technical analysis, unlike the Gann panel, and it
can be checked against real price history like Breakout Radar's hit-rate
check. It is still a documented setup, not a certainty: `false_move_verdict`
labels a read of the chart, not a prediction of what happens next.
"""
from __future__ import annotations

import pandas as pd

from src.indicators import rsi
from src.timeframe import resample_ohlcv

_BASE_OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def donchian_low(low: pd.Series, period: int) -> pd.Series:
    """Prior N-day low, excluding today - symmetric to
    src/indicators.py's donchian_high. A breakdown is close < this."""
    return low.shift(1).rolling(period, min_periods=period).min()


def _candle_geometry(df: pd.DataFrame) -> dict:
    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    return {"body": body, "upper_wick": upper_wick, "lower_wick": lower_wick}


def is_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Yesterday green, today red, today's real body fully engulfs
    yesterday's - the standard two-candle top-reversal shape."""
    prior_green = df["close"].shift(1) > df["open"].shift(1)
    today_red = df["close"] < df["open"]
    engulfs = (df["open"] >= df["close"].shift(1)) & (df["close"] <= df["open"].shift(1))
    return (prior_green & today_red & engulfs).fillna(False)


def is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Mirror of is_bearish_engulfing: yesterday red, today green, today's
    body fully engulfs yesterday's."""
    prior_red = df["close"].shift(1) < df["open"].shift(1)
    today_green = df["close"] > df["open"]
    engulfs = (df["open"] <= df["close"].shift(1)) & (df["close"] >= df["open"].shift(1))
    return (prior_red & today_green & engulfs).fillna(False)


def is_shooting_star(df: pd.DataFrame, min_upper_wick_ratio: float = 2.0, max_lower_wick_ratio: float = 1.0) -> pd.Series:
    """Small body near the bottom of the range with a long upper wick -
    the standard single-candle top-reversal shape. Ratios are relative to
    the candle's own body (upper wick at least 2x the body, lower wick no
    bigger than the body itself - the commonly cited version of this
    pattern)."""
    g = _candle_geometry(df)
    has_body = g["body"] > 0
    return (
        (g["upper_wick"] >= min_upper_wick_ratio * g["body"]) & (g["lower_wick"] <= max_lower_wick_ratio * g["body"]) & has_body
    ).fillna(False)


def is_hammer(df: pd.DataFrame, min_lower_wick_ratio: float = 2.0, max_upper_wick_ratio: float = 1.0) -> pd.Series:
    """Mirror of is_shooting_star: small body near the top of the range
    with a long lower wick - the standard single-candle bottom-reversal
    shape."""
    g = _candle_geometry(df)
    has_body = g["body"] > 0
    return (
        (g["lower_wick"] >= min_lower_wick_ratio * g["body"]) & (g["upper_wick"] <= max_upper_wick_ratio * g["body"]) & has_body
    ).fillna(False)


def is_evening_star(df: pd.DataFrame, small_body_ratio: float = 0.3) -> pd.Series:
    """3-candle top reversal: a large green candle, then a small-bodied
    (indecision) candle, then a red candle closing back below the first
    candle's body midpoint - the standard version of this pattern."""
    c1_open, c1_close = df["open"].shift(2), df["close"].shift(2)
    c1_body = c1_close - c1_open  # positive == green
    c2_body = (df["close"].shift(1) - df["open"].shift(1)).abs()
    c1_green = c1_body > 0
    c2_small = c2_body <= small_body_ratio * c1_body
    c3_red = df["close"] < df["open"]
    c3_closes_below_midpoint = df["close"] < (c1_open + c1_close) / 2
    return (c1_green & c2_small & c3_red & c3_closes_below_midpoint).fillna(False)


def is_morning_star(df: pd.DataFrame, small_body_ratio: float = 0.3) -> pd.Series:
    """Mirror of is_evening_star: a large red candle, a small-bodied
    candle, then a green candle closing back above the first candle's
    body midpoint."""
    c1_open, c1_close = df["open"].shift(2), df["close"].shift(2)
    c1_body = c1_open - c1_close  # positive == red
    c2_body = (df["close"].shift(1) - df["open"].shift(1)).abs()
    c1_red = c1_body > 0
    c2_small = c2_body <= small_body_ratio * c1_body
    c3_green = df["close"] > df["open"]
    c3_closes_above_midpoint = df["close"] > (c1_open + c1_close) / 2
    return (c1_red & c2_small & c3_green & c3_closes_above_midpoint).fillna(False)


BEARISH_PATTERNS = {
    "bearish_engulfing": is_bearish_engulfing,
    "evening_star": is_evening_star,
    "shooting_star": is_shooting_star,
}
BULLISH_PATTERNS = {
    "bullish_engulfing": is_bullish_engulfing,
    "morning_star": is_morning_star,
    "hammer": is_hammer,
}


def classify_rsi_zone(value: float | None, bullish_min: float, bearish_max: float) -> str | None:
    if value is None or pd.isna(value):
        return None
    if value >= bullish_min:
        return "bullish"
    if value < bearish_max:
        return "bearish"
    return "neutral"


def classify_direction_zone(weekly_rsi: float | None, daily_rsi: float | None, config: dict) -> str:
    """Mirrors the weekly-RSI x daily-RSI "Direction" matrix from the DSRD
    framework this was built from: 60+ is bullish, below 40 is bearish,
    40-60 is neutral on each timeframe. Only the five combinations that
    matrix names get a specific label; anything else (e.g. the two
    timeframes flatly disagreeing) is reported as "neutral" rather than
    inventing a label the source material didn't give."""
    f = config["filters"]["false_move_filter"]
    w = classify_rsi_zone(weekly_rsi, f["rsi_bullish_min"], f["rsi_bearish_max"])
    d = classify_rsi_zone(daily_rsi, f["rsi_bullish_min"], f["rsi_bearish_max"])
    if w is None or d is None:
        return "unknown"
    if w == "bullish" and d == "bullish":
        return "strong_bullish"
    if w == "bullish" and d == "neutral":
        return "bullish_pullback"
    if w == "bearish" and d == "bearish":
        return "strong_bearish"
    if w == "bearish" and d == "neutral":
        return "bearish_bounce"
    return "neutral"


def evaluate_false_move(enriched_df: pd.DataFrame, config: dict) -> dict:
    """enriched_df: a stock's OHLCV already run through compute_indicators()
    (e.g. straight from the main scan's enriched_cache) - reuses its
    existing `rsi` and `donchian_breakout` columns rather than
    recomputing them, so this never re-runs compute_indicators() (the
    duplicate-column pitfall Breakout Radar and the Gann panel both guard
    against doesn't apply here, since this module only reads existing
    columns and adds new ones of its own)."""
    f = config["filters"]["false_move_filter"]
    period = config["filters"]["donchian_breakout"]["period"]
    lookback = f["pattern_lookback_days"]

    df = enriched_df.reset_index(drop=True)
    if len(df) < period + lookback + 1 or "rsi" not in df.columns or "donchian_breakout" not in df.columns:
        return {"false_move_verdict": "insufficient_data"}

    base = df[_BASE_OHLCV_COLUMNS]

    d_low = donchian_low(base["low"], period)
    recent_breakdown = bool((base["close"] < d_low).tail(lookback).any())
    recent_breakout = bool(df["donchian_breakout"].tail(lookback).any())

    bearish_found = [name for name, fn in BEARISH_PATTERNS.items() if fn(base).tail(lookback).any()]
    bullish_found = [name for name, fn in BULLISH_PATTERNS.items() if fn(base).tail(lookback).any()]

    daily_rsi = df["rsi"].iloc[-1]
    weekly_bars = resample_ohlcv(base, "W-FRI")
    weekly_rsi_series = rsi(weekly_bars["close"], 14) if len(weekly_bars) >= 15 else pd.Series(dtype=float)
    weekly_rsi = weekly_rsi_series.iloc[-1] if not weekly_rsi_series.empty else None

    direction_zone = classify_direction_zone(weekly_rsi, daily_rsi, config)

    distribution_risk = recent_breakout and bool(bearish_found) and direction_zone in ("bearish_bounce", "strong_bearish")
    accumulation_signal = recent_breakdown and bool(bullish_found) and direction_zone in ("bullish_pullback", "strong_bullish")

    if distribution_risk:
        verdict = "distribution_risk"
    elif accumulation_signal:
        verdict = "accumulation_signal"
    else:
        verdict = "none"

    return {
        "weekly_rsi": round(float(weekly_rsi), 1) if weekly_rsi is not None and pd.notna(weekly_rsi) else None,
        "daily_rsi": round(float(daily_rsi), 1) if pd.notna(daily_rsi) else None,
        "direction_zone": direction_zone,
        "recent_breakout": recent_breakout,
        "recent_breakdown": recent_breakdown,
        "bearish_patterns": ", ".join(bearish_found),
        "bullish_patterns": ", ".join(bullish_found),
        "false_move_verdict": verdict,
    }


FALSE_MOVE_COLUMNS = [
    "weekly_rsi", "daily_rsi", "direction_zone", "recent_breakout", "recent_breakdown",
    "bearish_patterns", "bullish_patterns", "false_move_verdict",
]


def annotate_with_false_move_check(result_df: pd.DataFrame, config: dict, enriched_cache: dict) -> pd.DataFrame:
    """Second-pass enrichment, same pattern as
    growth_screen.annotate_with_growth_screen: only evaluates candidates
    that already passed every technical filter for whichever trading-style
    preset (Manual/Swing/Positional/Chartink) is currently active - this
    is a further filter ON TOP of that preset, not a replacement for it."""
    out = result_df.copy()
    for col in FALSE_MOVE_COLUMNS:
        out[col] = None

    candidates = out[out["passes_all_filters"] == True]  # noqa: E712
    for idx, row in candidates.iterrows():
        enriched_df = enriched_cache.get(row["security_code"])
        if enriched_df is None:
            out.at[idx, "false_move_verdict"] = "insufficient_data"
            continue
        result = evaluate_false_move(enriched_df, config)
        for col in FALSE_MOVE_COLUMNS:
            out.at[idx, col] = result.get(col)

    return out
