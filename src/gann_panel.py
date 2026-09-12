"""Gann Price-Time Squaring panel: for each stock, projects forward "time
windows" from its most recent significant swing high/low, and reports the
Square-of-Nine price levels associated with each window.

Read this before treating any of this as a signal:

- This is a documented but NOT empirically validated technical framework -
  unlike every other panel in this app (Breakout Radar's hit-rate check,
  the RSI/ADX/Donchian filters), there is no objective way to backtest
  "did the market respect this time window," because there is no single
  agreed-upon rule for which anchor, which angles, or which price-scaling
  convention is "correct." Different Gann practitioners disagree on all
  three. This module implements one commonly-cited convention at each
  step (documented inline) - it does not claim that convention is the
  only correct one, and it never predicts direction from the time window
  alone.
- "Time gives the alert, price gives the signal": a time window is a date
  to *watch*, not a trade. A window only gets called "bullish_confirmed"
  or "bearish_confirmed" once price actually breaks the signal candle's
  high/low - see `confirmation` in the output.
- Time cycles are projected as calendar days from the anchor (1 degree ==
  1 calendar day, so a 360-degree cycle == 360 days) - the most commonly
  cited Gann time convention, not the only one in circulation.
- Square-of-Nine levels use the standard add/subtract convention: each
  180 degrees of rotation shifts sqrt(price) by 1.0. See
  `square_of_nine_level` for the exact formula.
- Swing pivots are auto-detected (see `find_recent_swing_pivots`) and
  require `swing_lookback` bars of price action AFTER the pivot to
  confirm it - so the most recent detectable pivot always lags "today" by
  that many bars. This is deliberate: an unconfirmed peak/trough isn't a
  pivot yet, it's just the latest bar.
"""
from __future__ import annotations

import math

import pandas as pd

_BASE_OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def square_of_nine_level(anchor_price: float, angle_degrees: float, direction: str) -> float | None:
    """Standard Square-of-Nine add/subtract convention: each 180 degrees
    of rotation around the spiral shifts sqrt(price) by 1.0. Returns None
    if the result would be non-positive (not a meaningful price)."""
    if anchor_price is None or anchor_price <= 0:
        return None
    root = math.sqrt(anchor_price)
    delta = angle_degrees / 180.0
    new_root = root + delta if direction == "up" else root - delta
    if new_root <= 0:
        return None
    return round(new_root**2, 2)


def project_time_window(anchor_date: pd.Timestamp, angle_degrees: float) -> pd.Timestamp:
    """Degree == calendar day (360 degrees ~ 360 days ~ one year) - the
    most common Gann time-cycle convention."""
    return anchor_date + pd.Timedelta(days=angle_degrees)


def _local_extrema(series: pd.Series, lookback: int, mode: str) -> pd.Series:
    """Boolean mask: True where series[i] is the max (mode='high') or min
    (mode='low') within the symmetric window [i-lookback, i+lookback] - a
    classic swing-pivot definition. Bars within `lookback` of either end
    of the series can never be flagged (not enough bars on both sides to
    confirm them yet)."""
    window = 2 * lookback + 1
    if mode == "high":
        roll = series.rolling(window, center=True).max()
    else:
        roll = series.rolling(window, center=True).min()
    return series == roll


def find_recent_swing_pivots(df: pd.DataFrame, swing_lookback: int, anchor_window_days: int) -> dict:
    """Returns {"swing_high": {"date", "price"} | None, "swing_low": {...} | None}
    - the most recent CONFIRMED swing high/low (a local extreme with
    `swing_lookback` bars of price action on both sides) within the last
    `anchor_window_days` calendar days of available history. None means no
    qualifying pivot was found in that window - an explicit "don't know",
    never a guessed anchor."""
    d = df.sort_values("date").reset_index(drop=True)
    cutoff = d["date"].max() - pd.Timedelta(days=anchor_window_days)
    in_window = d["date"] >= cutoff

    result = {}
    for mode, col, key in (("high", "high", "swing_high"), ("low", "low", "swing_low")):
        is_extreme = _local_extrema(d[col], swing_lookback, mode)
        candidates = d[is_extreme.fillna(False) & in_window]
        if candidates.empty:
            result[key] = None
        else:
            last = candidates.iloc[-1]
            result[key] = {"date": last["date"], "price": float(last[col])}
    return result


def _nearest_trading_bar_on_or_after(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.Series | None:
    candidates = df[df["date"] >= target_date]
    return candidates.iloc[0] if not candidates.empty else None


def _check_confirmation(df: pd.DataFrame, signal_idx: int, lookforward: int) -> str:
    """"bullish_confirmed" / "bearish_confirmed" / "unconfirmed" -
    unconfirmed also covers "too recent to know yet" (fewer than
    `lookforward` bars have elapsed since the signal candle)."""
    signal = df.iloc[signal_idx]
    end_idx = min(signal_idx + lookforward, len(df) - 1)
    if end_idx <= signal_idx:
        return "unconfirmed"
    forward = df.iloc[signal_idx + 1 : end_idx + 1]
    if (forward["close"] > signal["high"]).any():
        return "bullish_confirmed"
    if (forward["close"] < signal["low"]).any():
        return "bearish_confirmed"
    return "unconfirmed"


def evaluate_gann_setup(raw_ohlcv: pd.DataFrame, config: dict) -> list[dict]:
    """One row per (anchor, angle) time window that falls within the
    configured recent-past-to-upcoming range, for a single stock."""
    g = config["filters"]["gann_panel"]
    df = raw_ohlcv[_BASE_OHLCV_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    pivots = find_recent_swing_pivots(df, g["swing_lookback"], g["anchor_window_days"])
    today = df["date"].max()
    horizon_end = today + pd.Timedelta(days=g["upcoming_horizon_days"])
    recent_past_start = today - pd.Timedelta(days=g["recent_past_days"])
    tolerance = pd.Timedelta(days=g["window_tolerance_days"])

    rows = []
    for anchor_type, pivot in pivots.items():
        if pivot is None:
            continue
        for angle in g["cycle_angles_degrees"]:
            window_date = project_time_window(pivot["date"], angle)
            if window_date < recent_past_start or window_date > horizon_end:
                continue

            row = {
                "anchor_type": anchor_type,
                "anchor_date": pivot["date"],
                "anchor_price": pivot["price"],
                "angle_degrees": angle,
                "window_date": window_date,
                "days_to_window": (window_date - today).days,
                "level_up": square_of_nine_level(pivot["price"], angle, "up"),
                "level_down": square_of_nine_level(pivot["price"], angle, "down"),
            }

            if window_date <= today + tolerance:
                signal_bar = _nearest_trading_bar_on_or_after(df, window_date - tolerance)
                if signal_bar is not None:
                    signal_idx = int(df.index[df["date"] == signal_bar["date"]][0])
                    row["signal_date"] = signal_bar["date"]
                    row["signal_high"] = float(signal_bar["high"])
                    row["signal_low"] = float(signal_bar["low"])
                    row["confirmation"] = _check_confirmation(df, signal_idx, g["confirmation_lookforward_days"])
                else:
                    row["signal_date"] = None
                    row["signal_high"] = None
                    row["signal_low"] = None
                    row["confirmation"] = "unconfirmed"
            else:
                row["signal_date"] = None
                row["signal_high"] = None
                row["signal_low"] = None
                row["confirmation"] = "upcoming"

            rows.append(row)

    rows.sort(key=lambda r: r["window_date"])
    return rows


def scan_gann_panel(raw_history: dict[str, pd.DataFrame], mapping: pd.DataFrame, config: dict) -> pd.DataFrame:
    """raw_history: {security_code: raw daily OHLCV df} - e.g. reuse the
    main scan's enriched_cache directly; only the base OHLCV columns are
    used, so any extra indicator columns already on those frames are
    ignored. One output row per (stock, time window)."""
    name_by_code = mapping.set_index("security_code")["company_name_raw"].to_dict()
    tsym_by_code = mapping.set_index("security_code")["tradingsymbol"].to_dict()
    exch_by_code = mapping.set_index("security_code")["exchange"].to_dict()

    rows = []
    for code, df in raw_history.items():
        if df is None or df.empty:
            continue
        for window in evaluate_gann_setup(df, config):
            rows.append(
                {
                    "security_code": code,
                    "company_name": name_by_code.get(code, code),
                    "tradingsymbol": tsym_by_code.get(code, code),
                    "exchange": exch_by_code.get(code, "NSE"),
                    **window,
                }
            )

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values("window_date").reset_index(drop=True)
    return result_df
