"""Offline unit tests for src/btst_check.py - synthetic data only."""
import numpy as np
import pandas as pd

from src import btst_check as bc
from src.config import load_config
from src.indicators import compute_indicators


def make_ohlcv(closes: list[float], highs=None, lows=None, volumes=None, start="2022-01-03") -> pd.DataFrame:
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    closes = np.array(closes, dtype=float)
    highs = np.array(highs, dtype=float) if highs is not None else closes * 1.01
    lows = np.array(lows, dtype=float) if lows is not None else closes * 0.99
    if volumes is None:
        volumes = np.full(n, 100_000.0)
    return pd.DataFrame(
        {"date": dates, "open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes}
    )


def _cfg(**overrides):
    config = load_config()
    config["filters"]["btst_check"].update(overrides)
    return config


def test_weekly_rsi_no_lookahead_only_uses_completed_weeks():
    rng = np.random.default_rng(0)
    closes = 100 + rng.normal(0, 1, 200).cumsum() * 0.05
    df = make_ohlcv(list(closes))
    weekly_rsi = bc._weekly_rsi_no_lookahead(df[["date", "open", "high", "low", "close", "volume"]])

    from src.timeframe import resample_ohlcv
    from src.indicators import rsi as rsi_fn

    weekly = resample_ohlcv(df, "W-FRI")
    full_weekly_rsi = rsi_fn(weekly["close"], 14)
    last_completed_friday = weekly["date"].iloc[-2]  # the week before the most recent one
    expected_value = full_weekly_rsi.iloc[-2]

    # Any daily bar strictly within the week AFTER last_completed_friday (but
    # before the next Friday) should see exactly that completed week's RSI.
    check_date = last_completed_friday + pd.Timedelta(days=1)
    idx = df[df["date"] >= check_date].index[0]
    assert weekly_rsi.iloc[idx] == expected_value


def test_weekly_rsi_no_lookahead_is_nan_before_any_completed_week():
    df = make_ohlcv(list(np.linspace(100, 105, 5)))
    weekly_rsi = bc._weekly_rsi_no_lookahead(df[["date", "open", "high", "low", "close", "volume"]])
    assert weekly_rsi.isna().all()


def test_compute_btst_signal_flags_the_defined_setup():
    rng = np.random.default_rng(1)
    closes = 100 + rng.normal(0, 1, 150).cumsum() * 0.05
    # close_strength averages the last 3 days, so all three (not just the
    # final one) need to close near their day's high for the average to
    # clear the 0.7 threshold.
    closes = list(closes)
    closes[-3:] = [closes[-4] * 1.005, closes[-4] * 1.012, closes[-4] * 1.02]
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    for i in (-3, -2, -1):
        highs[i] = closes[i] * 1.002
        lows[i] = closes[i] * 0.97
    volumes = [100_000.0] * 149 + [400_000.0]
    df = make_ohlcv(closes, highs=highs, lows=lows, volumes=volumes)

    config = _cfg(min_weekly_rsi=0.0)  # isolate the close-strength/volume part of the signal for this test
    enriched = compute_indicators(df, config, benchmark=None)
    signal = bc.compute_btst_signal(enriched, config)
    assert bool(signal["is_btst_setup"].iloc[-1]) is True


def test_compute_btst_signal_never_flags_a_weak_low_volume_close():
    rng = np.random.default_rng(2)
    closes = list(100 + rng.normal(0, 1, 150).cumsum() * 0.05)
    df = make_ohlcv(closes)  # default highs/lows -> close sits mid-range, default volume -> no surge
    config = _cfg()
    enriched = compute_indicators(df, config, benchmark=None)
    signal = bc.compute_btst_signal(enriched, config)
    assert signal["is_btst_setup"].iloc[-1] == False  # noqa: E712


def test_btst_hit_rate_for_stock_returns_none_for_short_history():
    df = make_ohlcv(list(np.linspace(100, 110, 20)))
    config = _cfg()
    enriched = compute_indicators(df, config, benchmark=None)
    assert bc.btst_hit_rate_for_stock(enriched, config) is None


def test_btst_hit_rate_from_prepared_counts_a_known_hit_and_excludes_last_bar():
    n = 30
    dates = pd.bdate_range("2022-01-03", periods=n)
    closes = np.full(n, 100.0)
    highs = closes * 1.001
    lows = closes * 0.999
    # Day 10 is flagged; day 11's high reaches +5% (a hit), close only +1%.
    highs[11] = closes[10] * 1.05
    closes[11] = closes[10] * 1.01
    prepared = pd.DataFrame({"date": dates, "close": closes, "high": highs, "low": lows})
    prepared["is_btst_setup"] = False
    prepared.loc[10, "is_btst_setup"] = True
    prepared.loc[n - 1, "is_btst_setup"] = True  # last bar - must be excluded (no "tomorrow" to measure)

    result = bc._btst_hit_rate_from_prepared(prepared, target_return_pct=4.5, stop_loss_pct=2.0)
    assert result["n_flagged"] == 1  # the last-bar flag is excluded
    assert result["n_flagged_hit_target"] == 1


def test_aggregate_btst_hit_rate_skips_stocks_with_insufficient_history():
    rng = np.random.default_rng(3)
    long_history = make_ohlcv(list(100 + rng.normal(0, 1, 150).cumsum() * 0.05))
    short_history = make_ohlcv(list(np.linspace(100, 105, 20)))
    config = _cfg()
    enriched_history = {
        "LONG": compute_indicators(long_history, config, benchmark=None),
        "SHORT": compute_indicators(short_history, config, benchmark=None),
    }
    result = bc.aggregate_btst_hit_rate(enriched_history, config)
    assert result["n_stocks_included"] == 1
    assert result["target_return_pct"] == config["filters"]["btst_check"]["target_return_pct"]


def test_aggregate_btst_hit_rate_handles_zero_flags_without_crashing():
    rng = np.random.default_rng(4)
    # Very tight, flat range - unlikely to ever trigger a volume surge or strong close.
    closes = 100 + rng.normal(0, 0.01, 150)
    df = make_ohlcv(list(closes))
    config = _cfg(min_close_strength=0.99, min_volume_surge=10.0)  # deliberately near-impossible to trigger
    enriched = compute_indicators(df, config, benchmark=None)
    result = bc.aggregate_btst_hit_rate({"A1": enriched}, config)
    assert result["n_stocks_included"] == 1
    assert result["hit_target_rate_pct"] is None  # n_flagged == 0 -> None, never a misleading 0% or divide-by-zero
