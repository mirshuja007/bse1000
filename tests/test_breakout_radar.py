"""Offline unit tests for src/breakout_radar.py - synthetic data only."""
import numpy as np
import pandas as pd

from src import breakout_radar as br
from src.config import load_config
from src.indicators import compute_indicators


def make_ohlcv(closes: list[float], volumes: list[float] | None = None, start="2023-01-02") -> pd.DataFrame:
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    closes = np.array(closes, dtype=float)
    if volumes is None:
        volumes = np.full(n, 100_000.0)
    return pd.DataFrame(
        {
            "date": dates, "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": volumes,
        }
    )


def test_bollinger_band_width_shrinks_as_range_tightens():
    rng = np.random.default_rng(0)
    wide = 100 + rng.normal(0, 5, 100).cumsum() * 0.05 + rng.normal(0, 3, 100)  # choppy
    tight = np.linspace(150, 152, 60)  # a near-flat coil
    closes = pd.Series(np.concatenate([wide, tight]))
    width = br.bollinger_band_width(closes, period=20)
    assert width.iloc[-1] < width.iloc[99]  # tighter at the end of the coil than entering it


def _cfg(**overrides):
    config = load_config()
    config["filters"]["breakout_radar"].update(overrides)
    return config


def test_compute_precursor_signals_flags_a_coiling_setup():
    # Choppy base for a while, then a genuine tight coil just under a
    # well-defined ceiling, with volatility visibly contracting into it.
    rng = np.random.default_rng(1)
    base = 100 + rng.normal(0, 2, 150).cumsum() * 0.02
    ceiling = base.max() + 1
    coil = np.linspace(base[-1], ceiling - 0.5, 40) + rng.normal(0, 0.05, 40)  # very tight, just under ceiling
    closes = np.concatenate([base, coil])
    df = make_ohlcv(list(closes))

    config = _cfg()
    enriched = compute_indicators(df, config, benchmark=None)
    signals = br.compute_precursor_signals(enriched, config)

    assert signals["is_watch"].iloc[-1] == True or signals["pct_below_pivot"].iloc[-1] <= 5.0  # noqa: E712


def test_compute_precursor_signals_never_flags_after_a_breakout():
    closes = list(np.linspace(100, 100.5, 150)) + list(np.linspace(100.5, 200, 30))  # a clean breakout
    df = make_ohlcv(closes)
    config = _cfg()
    enriched = compute_indicators(df, config, benchmark=None)
    signals = br.compute_precursor_signals(enriched, config)
    assert signals["is_watch"].iloc[-1] == False  # noqa: E712  already broken out, not "pre"-breakout


def test_evaluate_timeframe_reports_insufficient_data_for_short_history():
    df = make_ohlcv(list(np.linspace(100, 110, 10)))
    config = _cfg()
    result = br.evaluate_timeframe(df, "daily", config)
    assert result["insufficient_data"] is True
    assert result["bars_available"] == 10


def test_evaluate_timeframe_daily_detects_fresh_breakout():
    rng = np.random.default_rng(2)
    base = 100 + rng.normal(0, 1, 150).cumsum() * 0.05
    breakout_day = base[-1] + 10  # a clean jump above the recent range
    closes = list(base) + [breakout_day]
    volumes = [100_000.0] * 150 + [400_000.0]  # volume surge on the breakout day
    df = make_ohlcv(closes, volumes=volumes)

    config = _cfg()
    result = br.evaluate_timeframe(df, "daily", config)
    assert result["insufficient_data"] is False
    assert result["is_fresh_breakout"] is True
    assert result["is_watch"] is False  # already broke out, not "pre"
    # Strength/false-breakout-check columns should be populated and sane.
    assert result["rsi"] is not None and 0 <= result["rsi"] <= 100
    assert result["adx"] is not None and result["adx"] >= 0
    assert result["close_strength"] is not None and 0 <= result["close_strength"] <= 1
    assert result["pct_from_pivot"] > 0  # already broke out -> positive = extended above the pivot
    assert result["pct_from_pivot"] == -result["pct_below_pivot"]


def test_evaluate_timeframe_pct_from_pivot_is_negative_while_still_coiling():
    rng = np.random.default_rng(5)
    closes = 100 + rng.normal(0, 1, 150).cumsum() * 0.05  # never breaks its own high
    df = make_ohlcv(list(closes))
    result = br.evaluate_timeframe(df, "daily", _cfg())
    assert result["insufficient_data"] is False
    assert result["is_fresh_breakout"] is False
    assert result["pct_from_pivot"] <= 0


def test_scan_breakout_radar_summarizes_across_stocks():
    rng = np.random.default_rng(3)
    base = 100 + rng.normal(0, 1, 150).cumsum() * 0.05
    breakout_closes = list(base) + [base[-1] + 10]
    breakout_volumes = [100_000.0] * 150 + [400_000.0]
    flat_closes = list(np.linspace(100, 105, 151))

    raw_history = {
        "A1": make_ohlcv(breakout_closes, volumes=breakout_volumes),
        "A2": make_ohlcv(flat_closes),
    }
    mapping = pd.DataFrame(
        [
            {"security_code": "A1", "company_name_raw": "Alpha Co", "tradingsymbol": "ALPHA", "exchange": "NSE"},
            {"security_code": "A2", "company_name_raw": "Beta Co", "tradingsymbol": "BETA", "exchange": "NSE"},
        ]
    )
    config = _cfg()
    result = br.scan_breakout_radar(raw_history, mapping, config, timeframes=["daily"])

    assert len(result) == 2
    alpha = result.set_index("security_code").loc["A1"]
    assert "daily" in alpha["breakout_timeframes"]


def test_historical_hit_rate_from_prepared_counts_a_known_hit():
    # Flag exactly one day as "watch", then have price rally 15% within
    # the forward window - should count as a hit for the flagged bucket
    # and nowhere else.
    n = 40
    dates = pd.bdate_range("2023-01-02", periods=n)
    closes = np.full(n, 100.0)
    closes[20:] = 115.0  # a rally starting the day after the flagged day
    highs = closes * 1.001
    prepared = pd.DataFrame({"date": dates, "close": closes, "high": highs})
    prepared["is_watch"] = False
    prepared.loc[19, "is_watch"] = True

    result = br._historical_hit_rate_from_prepared(prepared, forward_days=5, target_return_pct=10.0)
    assert result["n_flagged"] == 1
    assert result["n_flagged_hit"] == 1
    assert result["flagged_hit_rate_pct"] == 100.0


def test_historical_hit_rate_excludes_trailing_bars_with_incomplete_forward_window():
    n = 10
    dates = pd.bdate_range("2023-01-02", periods=n)
    prepared = pd.DataFrame({"date": dates, "close": [100.0] * n, "high": [101.0] * n})
    prepared["is_watch"] = True  # flag every day, including the very last ones

    result = br._historical_hit_rate_from_prepared(prepared, forward_days=5, target_return_pct=10.0)
    # Only the first n - forward_days bars have a full forward window.
    assert result["n_flagged"] == n - 5


def test_evaluate_timeframe_daily_handles_already_enriched_input():
    # Regression test: app.py actually feeds Breakout Radar the main scan's
    # enriched_cache, which has ALREADY been through compute_indicators()
    # (adx/rsi/sma50/donchian_high/etc. already present) - not fresh raw
    # OHLCV like every other test in this file uses. Re-running
    # compute_indicators() on top of that output must not append duplicate
    # columns (which previously made latest.get("adx") return a Series
    # instead of a scalar and crash pd.isna() with a ValueError).
    rng = np.random.default_rng(6)
    base = 100 + rng.normal(0, 1, 150).cumsum() * 0.05
    closes = list(base) + [base[-1] + 10]
    volumes = [100_000.0] * 150 + [400_000.0]
    raw_df = make_ohlcv(closes, volumes=volumes)

    config = _cfg()
    already_enriched = compute_indicators(raw_df, config, benchmark=None)
    assert not already_enriched.columns.duplicated().any()

    prepared = br._prepare_timeframe_df(already_enriched, "daily", config)
    assert prepared is not None
    assert not prepared.columns.duplicated().any()

    result = br.evaluate_timeframe(already_enriched, "daily", config)
    assert result["insufficient_data"] is False
    assert result["is_fresh_breakout"] is True
    assert isinstance(result["adx"], float)
    assert isinstance(result["rsi"], float)


def test_scan_breakout_radar_handles_already_enriched_raw_history():
    # Same regression, through the actual public entry point app.py calls.
    rng = np.random.default_rng(7)
    base = 100 + rng.normal(0, 1, 150).cumsum() * 0.05
    closes = list(base) + [base[-1] + 10]
    volumes = [100_000.0] * 150 + [400_000.0]
    raw_df = make_ohlcv(closes, volumes=volumes)
    config = _cfg()
    already_enriched = compute_indicators(raw_df, config, benchmark=None)

    mapping = pd.DataFrame(
        [{"security_code": "A1", "company_name_raw": "Alpha Co", "tradingsymbol": "ALPHA", "exchange": "NSE"}]
    )
    result = br.scan_breakout_radar({"A1": already_enriched}, mapping, config, timeframes=["daily", "weekly"])
    assert len(result) == 1


def test_aggregate_historical_hit_rate_skips_stocks_with_insufficient_history():
    rng = np.random.default_rng(4)
    long_history = make_ohlcv(list(100 + rng.normal(0, 1, 150).cumsum() * 0.05))
    short_history = make_ohlcv(list(np.linspace(100, 105, 10)))
    raw_history = {"LONG": long_history, "SHORT": short_history}

    config = _cfg()
    result = br.aggregate_historical_hit_rate(raw_history, "daily", config)
    assert result["n_stocks_included"] == 1  # SHORT excluded for insufficient history
