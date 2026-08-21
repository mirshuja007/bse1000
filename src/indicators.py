"""Technical indicator library, all pure pandas/numpy - no external TA
dependency, so the exact formula for every number the app shows is visible
and auditable here.

All functions take a daily OHLCV DataFrame sorted ascending by date with
columns: date, open, high, low, close, volume - and return either an
enriched copy of that DataFrame or a derived Series.

Smoothing convention: RSI / ATR / ADX use Wilder's smoothing (the original,
standard definition for these three), matching what most charting platforms
including Kite/TradingView show by default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(100)  # avg_loss == 0 -> maximally strong


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_smooth = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    plus_di = 100 * (plus_dm_smooth / atr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm_smooth / atr_smooth.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_line})


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def donchian_high(high: pd.Series, period: int = 20) -> pd.Series:
    """Prior N-day high, excluding today - a breakout is close > this."""
    return high.shift(1).rolling(period, min_periods=period).max()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend: an ATR-based trend/trailing-stop line (Olivier Seban).

    Basic bands are the textbook formula: midpoint +/- multiplier*ATR. On
    their own those basic bands are noisy - the indicator's actual behavior
    (the smooth, sticky line every platform shows) comes from a second,
    recursive step most summaries skip: each *final* band only ever moves
    in the direction that tightens around price (never loosens), and the
    line itself only flips from tracking the upper band to the lower band
    (or back) when price actually closes through it. That flip is what
    "bullish"/"bearish" means below.

    Uses this module's Wilder-smoothed ATR (the modern-platform default for
    Supertrend), not a plain moving average of true range.

    Returns a DataFrame aligned to `df`'s index with:
      - supertrend: the line's price level
      - supertrend_bullish: True while the line is acting as support below
        price (uptrend / green), False while it's resistance above price
        (downtrend / red)
    """
    atr_series = atr(df, period)
    mid = (df["high"] + df["low"]) / 2
    basic_upper = (mid + multiplier * atr_series).to_numpy()
    basic_lower = (mid - multiplier * atr_series).to_numpy()
    close = df["close"].to_numpy()

    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    line = np.full(n, np.nan)
    bullish = np.zeros(n, dtype=bool)

    first_valid = period  # ATR's min_periods=period, so bands are NaN before this
    if first_valid >= n:
        return pd.DataFrame({"supertrend": line, "supertrend_bullish": bullish}, index=df.index)

    final_upper[first_valid] = basic_upper[first_valid]
    final_lower[first_valid] = basic_lower[first_valid]
    line[first_valid] = final_upper[first_valid]
    bullish[first_valid] = False

    for i in range(first_valid + 1, n):
        if np.isnan(basic_upper[i]) or np.isnan(basic_lower[i]):
            continue

        final_upper[i] = (
            basic_upper[i]
            if (basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            basic_lower[i]
            if (basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )

        if line[i - 1] == final_upper[i - 1]:
            if close[i] <= final_upper[i]:
                line[i], bullish[i] = final_upper[i], False
            else:
                line[i], bullish[i] = final_lower[i], True
        else:
            if close[i] >= final_lower[i]:
                line[i], bullish[i] = final_lower[i], True
            else:
                line[i], bullish[i] = final_upper[i], False

    return pd.DataFrame({"supertrend": line, "supertrend_bullish": bullish}, index=df.index)


def flipped_bullish_within(bullish: pd.Series, lookback: int) -> pd.Series:
    """True on any day where `bullish` turned True (a red-to-green flip) at
    some point in the trailing `lookback` sessions (inclusive of today) -
    the actual entry-timing trigger, as opposed to merely "currently
    bullish" which says nothing about how stale the signal is."""
    prev = bullish.shift(1, fill_value=False)
    flipped = bullish & ~prev
    return flipped.rolling(lookback, min_periods=1).max().astype(bool)


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def crossed_above_within(price: pd.Series, level: pd.Series, lookback: int) -> pd.Series:
    """True on any day where `price` crossed above `level` at some point in
    the trailing `lookback` sessions (inclusive of today)."""
    crossed = (price > level) & (price.shift(1) <= level.shift(1))
    return crossed.rolling(lookback, min_periods=1).max().astype(bool)


def volume_surge_ratio(volume: pd.Series, lookback: int = 20) -> pd.Series:
    baseline = volume.rolling(lookback, min_periods=lookback).mean().shift(1)
    return volume / baseline.replace(0, np.nan)


def close_strength(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """Average of (close-low)/(high-low) over `window` days - close to 1
    means the stock is repeatedly closing near the day's high (buyers in
    control into the close), close to 0 means closing near the low."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    position = (df["close"] - df["low"]) / rng
    return position.rolling(window, min_periods=1).mean()


def turnover_cr(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Average daily traded value in INR crore over `lookback` days."""
    turnover = df["close"] * df["volume"]
    return turnover.rolling(lookback, min_periods=1).mean() / 1e7


def relative_strength_line(close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    return close / benchmark_close


def compute_indicators(df: pd.DataFrame, config: dict, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach every indicator this app uses to a copy of the raw OHLCV frame."""
    out = df.copy().reset_index(drop=True)

    rsi_period = config["filters"]["rsi"]["period"]
    atr_period = config["filters"]["adx"]["period"]
    adx_period = config["filters"]["adx"]["period"]
    donchian_period = config["filters"]["donchian_breakout"]["period"]
    vol_lookback = config["filters"]["volume"]["lookback"]
    turnover_lookback = config["filters"]["liquidity"]["turnover_lookback"]
    dma50_lookback = config["filters"]["dma50_breakout"]["lookback_days"]
    dma200_lookback = config["filters"]["dma200_breakout"]["lookback_days"]
    st_cfg = config["filters"]["supertrend"]

    out["sma50"] = sma(out["close"], 50)
    out["sma200"] = sma(out["close"], 200)
    out["rsi"] = rsi(out["close"], rsi_period)
    out["atr"] = atr(out, atr_period)
    out["atr_pct"] = out["atr"] / out["close"] * 100

    adx_df = adx(out, adx_period)
    out = pd.concat([out, adx_df], axis=1)

    macd_df = macd(out["close"])
    out = pd.concat([out, macd_df], axis=1)

    out["donchian_high"] = donchian_high(out["high"], donchian_period)
    out["donchian_breakout"] = out["close"] > out["donchian_high"]

    st_df = supertrend(out, st_cfg["period"], st_cfg["multiplier"])
    out["supertrend"] = st_df["supertrend"]
    out["supertrend_bullish"] = st_df["supertrend_bullish"]
    out["supertrend_flip_recent"] = flipped_bullish_within(
        out["supertrend_bullish"], st_cfg["flip_lookback_days"]
    )

    out["obv"] = obv(out)
    out["obv_slope20"] = out["obv"].diff(20)

    out["volume_surge"] = volume_surge_ratio(out["volume"], vol_lookback)
    out["close_strength"] = close_strength(out)
    out["turnover_cr"] = turnover_cr(out, turnover_lookback)

    out["dma50_breakout_recent"] = crossed_above_within(out["close"], out["sma50"], dma50_lookback)
    out["dma200_breakout_recent"] = crossed_above_within(out["close"], out["sma200"], dma200_lookback)
    out["pct_above_sma50"] = (out["close"] / out["sma50"] - 1) * 100
    out["pct_above_sma200"] = (out["close"] / out["sma200"] - 1) * 100

    out["roc3"] = out["close"].pct_change(3) * 100
    out["roc5"] = out["close"].pct_change(5) * 100
    out["roc10"] = out["close"].pct_change(10) * 100

    # Volatility contraction: today's ATR% vs 20 sessions ago. Negative =
    # volatility has been contracting into the current move (classic
    # pre-breakout "coil" pattern).
    out["atr_pct_chg20"] = out["atr_pct"] - out["atr_pct"].shift(20)

    if benchmark is not None and not benchmark.empty:
        bench = benchmark[["date", "close"]].rename(columns={"close": "bench_close"})
        out = out.merge(bench, on="date", how="left")
        out["bench_close"] = out["bench_close"].ffill()
        out["rs_line"] = relative_strength_line(out["close"], out["bench_close"])
        out["rs_line_high63"] = out["rs_line"].rolling(63, min_periods=20).max()
        out["rs_new_high"] = out["rs_line"] >= out["rs_line_high63"]
        out["rs_roc20"] = out["rs_line"].pct_change(20) * 100
        stock_roc20 = out["close"].pct_change(20) * 100
        bench_roc20 = out["bench_close"].pct_change(20) * 100
        out["relative_return_20d"] = stock_roc20 - bench_roc20
    else:
        out["rs_line"] = np.nan
        out["rs_new_high"] = False
        out["rs_roc20"] = np.nan
        out["relative_return_20d"] = np.nan

    return out


def latest_snapshot(enriched_df: pd.DataFrame) -> dict:
    """Pull the most recent row of an indicator-enriched frame into a flat
    dict, plus a couple of derived convenience fields, for scanner/conviction
    consumption."""
    if enriched_df.empty:
        return {}
    row = enriched_df.iloc[-1].to_dict()
    row["as_of"] = enriched_df["date"].iloc[-1]
    row["golden_cross"] = bool(
        pd.notna(row.get("sma50")) and pd.notna(row.get("sma200")) and row["sma50"] > row["sma200"]
    )
    row["price_above_50"] = bool(pd.notna(row.get("sma50")) and row["close"] > row["sma50"])
    row["price_above_200"] = bool(pd.notna(row.get("sma200")) and row["close"] > row["sma200"])
    return row
