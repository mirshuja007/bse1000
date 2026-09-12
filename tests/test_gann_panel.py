"""Offline unit tests for src/gann_panel.py - synthetic data only."""
import numpy as np
import pandas as pd

from src import gann_panel as gp
from src.config import load_config


def make_ohlcv(closes: list[float], start="2023-01-02") -> pd.DataFrame:
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    closes = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": dates, "open": closes, "high": closes * 1.005, "low": closes * 0.995,
            "close": closes, "volume": np.full(n, 100_000.0),
        }
    )


def _cfg(**overrides):
    config = load_config()
    config["filters"]["gann_panel"].update(overrides)
    return config


def test_square_of_nine_level_matches_known_arithmetic():
    # anchor 100 -> sqrt(100) = 10; 180 degrees -> delta 1.0
    assert gp.square_of_nine_level(100.0, 180, "up") == 121.0
    assert gp.square_of_nine_level(100.0, 180, "down") == 81.0


def test_square_of_nine_level_returns_none_for_non_positive_anchor():
    assert gp.square_of_nine_level(0.0, 90, "up") is None
    assert gp.square_of_nine_level(-5.0, 90, "up") is None


def test_square_of_nine_level_returns_none_when_result_would_be_non_positive():
    # A large enough "down" delta can push sqrt(price) below zero.
    assert gp.square_of_nine_level(0.01, 360, "down") is None


def test_project_time_window_adds_calendar_days():
    anchor = pd.Timestamp("2024-01-01")
    assert gp.project_time_window(anchor, 90) == anchor + pd.Timedelta(days=90)


def test_find_recent_swing_pivots_detects_a_confirmed_high_and_low():
    rng = np.random.default_rng(0)
    up = 100 + np.linspace(0, 20, 40) + rng.normal(0, 0.2, 40)
    down = up[-1] - np.linspace(0, 15, 40) + rng.normal(0, 0.2, 40)
    tail = down[-1] + np.linspace(0, 5, 30) + rng.normal(0, 0.2, 30)  # confirms the low with bars after it
    closes = np.concatenate([up, down, tail])
    df = make_ohlcv(list(closes))

    pivots = gp.find_recent_swing_pivots(df, swing_lookback=10, anchor_window_days=250)
    assert pivots["swing_high"] is not None
    assert pivots["swing_low"] is not None
    assert pivots["swing_low"]["date"] > pivots["swing_high"]["date"]


def test_find_recent_swing_pivots_ignores_an_unconfirmed_recent_extreme():
    # A brand-new high right at the end of the series has no bars after it
    # to confirm it as a pivot yet - must not be reported.
    closes = list(np.linspace(100, 120, 60)) + [130.0]  # sharp new high on the very last bar
    df = make_ohlcv(closes)
    pivots = gp.find_recent_swing_pivots(df, swing_lookback=10, anchor_window_days=250)
    if pivots["swing_high"] is not None:
        assert pivots["swing_high"]["price"] < 130.0


def test_find_recent_swing_pivots_returns_none_outside_anchor_window():
    df = make_ohlcv(list(np.linspace(100, 110, 60)))
    pivots = gp.find_recent_swing_pivots(df, swing_lookback=10, anchor_window_days=1)
    assert pivots["swing_high"] is None
    assert pivots["swing_low"] is None


def test_evaluate_gann_setup_flags_bullish_confirmation_after_a_breakout():
    # Build a low pivot, then place a strong rally exactly around the
    # 90-degree window date so the signal candle's high gets broken soon
    # after - should read as bullish_confirmed for that window.
    rng = np.random.default_rng(1)
    down = 150 - np.linspace(0, 20, 40) + rng.normal(0, 0.2, 40)
    up_after = down[-1] + np.linspace(0, 10, 30) + rng.normal(0, 0.2, 30)  # confirms the low
    df = make_ohlcv(list(down) + list(up_after))

    pivots = gp.find_recent_swing_pivots(df, swing_lookback=10, anchor_window_days=250)
    low_date = pivots["swing_low"]["date"]
    window_date = gp.project_time_window(low_date, 90)

    # Extend history out to and past the 90-degree window with a rally
    # that breaks decisively above whatever bar lands on that date.
    extra_dates = pd.bdate_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=80)
    extra_closes = df["close"].iloc[-1] + np.linspace(0, 40, 80)  # steady rally through and past the window
    extra = pd.DataFrame(
        {
            "date": extra_dates, "open": extra_closes, "high": extra_closes * 1.01,
            "low": extra_closes * 0.99, "close": extra_closes, "volume": np.full(80, 100_000.0),
        }
    )
    full_df = pd.concat([df, extra], ignore_index=True)

    config = _cfg(cycle_angles_degrees=[90], recent_past_days=200, upcoming_horizon_days=1)
    windows = gp.evaluate_gann_setup(full_df, config)
    matching = [w for w in windows if w["anchor_type"] == "swing_low" and w["angle_degrees"] == 90]
    assert len(matching) == 1
    assert matching[0]["confirmation"] == "bullish_confirmed"
    assert matching[0]["window_date"] == window_date


def test_evaluate_gann_setup_marks_future_windows_as_upcoming():
    rng = np.random.default_rng(2)
    closes = 100 + rng.normal(0, 1, 60).cumsum() * 0.05
    df = make_ohlcv(list(closes))
    config = _cfg(cycle_angles_degrees=[360], upcoming_horizon_days=400, recent_past_days=30)
    windows = gp.evaluate_gann_setup(df, config)
    assert all(w["confirmation"] == "upcoming" for w in windows)
    assert all(w["days_to_window"] > 0 for w in windows)


def test_evaluate_gann_setup_excludes_windows_outside_configured_range():
    rng = np.random.default_rng(3)
    closes = 100 + rng.normal(0, 1, 60).cumsum() * 0.05
    df = make_ohlcv(list(closes))
    config = _cfg(cycle_angles_degrees=[360], upcoming_horizon_days=5, recent_past_days=1)
    windows = gp.evaluate_gann_setup(df, config)
    assert windows == []  # a 360-day-out window is far past a 5-day horizon


def test_scan_gann_panel_aggregates_across_stocks():
    rng = np.random.default_rng(4)
    closes_a = 100 + rng.normal(0, 1, 200).cumsum() * 0.05
    closes_b = 200 + rng.normal(0, 1, 200).cumsum() * 0.05
    raw_history = {"A1": make_ohlcv(list(closes_a)), "A2": make_ohlcv(list(closes_b))}
    mapping = pd.DataFrame(
        [
            {"security_code": "A1", "company_name_raw": "Alpha Co", "tradingsymbol": "ALPHA", "exchange": "NSE"},
            {"security_code": "A2", "company_name_raw": "Beta Co", "tradingsymbol": "BETA", "exchange": "NSE"},
        ]
    )
    config = _cfg(upcoming_horizon_days=400)
    result = gp.scan_gann_panel(raw_history, mapping, config)
    if not result.empty:
        assert set(result["security_code"]).issubset({"A1", "A2"})
        assert list(result["window_date"]) == sorted(result["window_date"])


def test_scan_gann_panel_handles_already_enriched_raw_history():
    # Regression guard matching Breakout Radar's fix: app.py feeds this
    # panel the main scan's enriched_cache (already run through
    # compute_indicators()), so extra pre-existing columns must be
    # tolerated, not just fresh raw OHLCV.
    from src.indicators import compute_indicators

    rng = np.random.default_rng(5)
    closes = 100 + rng.normal(0, 1, 200).cumsum() * 0.05
    raw_df = make_ohlcv(list(closes))
    config = _cfg(upcoming_horizon_days=400)
    already_enriched = compute_indicators(raw_df, config, benchmark=None)

    mapping = pd.DataFrame(
        [{"security_code": "A1", "company_name_raw": "Alpha Co", "tradingsymbol": "ALPHA", "exchange": "NSE"}]
    )
    result = gp.scan_gann_panel({"A1": already_enriched}, mapping, config)
    assert isinstance(result, pd.DataFrame)
