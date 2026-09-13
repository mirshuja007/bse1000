"""Offline unit tests for src/false_move_filter.py - synthetic data only."""
import numpy as np
import pandas as pd

from src import false_move_filter as fmf
from src.config import load_config
from src.indicators import compute_indicators


def _candle(o, h, l, c, date):
    return {"date": date, "open": o, "high": h, "low": l, "close": c, "volume": 100_000.0}


def make_ohlcv(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _dates(n, start="2023-01-02"):
    return list(pd.bdate_range(start, periods=n))


def _flat_series(n, price=100.0, start="2023-01-02"):
    dates = _dates(n, start)
    return [_candle(price, price * 1.001, price * 0.999, price, d) for d in dates]


def _cfg(**overrides):
    config = load_config()
    config["filters"]["false_move_filter"].update(overrides)
    return config


def test_donchian_low_matches_known_rolling_minimum():
    low = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    result = fmf.donchian_low(low, period=3)
    # At index 5 (value 5), prior 3 lows (indices 2,3,4 -> 8,7,6) -> min 6
    assert result.iloc[5] == 6


def test_is_bearish_engulfing_detects_known_pattern():
    rows = _flat_series(5)
    rows.append(_candle(100, 102, 99.5, 101.5, _dates(6)[5]))  # green day
    rows.append(_candle(102, 102.5, 98, 98.5, _dates(7)[6]))  # red day engulfing prior body
    df = make_ohlcv(rows)
    result = fmf.is_bearish_engulfing(df)
    assert result.iloc[-1] == True  # noqa: E712
    assert not result.iloc[:-1].any()


def test_is_bullish_engulfing_detects_known_pattern():
    rows = _flat_series(5)
    rows.append(_candle(102, 102.5, 99.5, 100, _dates(6)[5]))  # red day
    rows.append(_candle(99.5, 103, 99, 102.5, _dates(7)[6]))  # green day engulfing prior body
    df = make_ohlcv(rows)
    result = fmf.is_bullish_engulfing(df)
    assert result.iloc[-1] == True  # noqa: E712


def test_is_shooting_star_detects_long_upper_wick_small_body():
    rows = _flat_series(5)
    rows.append(_candle(100, 110, 99.8, 100.5, _dates(6)[5]))  # tiny body, huge upper wick, minimal lower wick
    df = make_ohlcv(rows)
    result = fmf.is_shooting_star(df)
    assert result.iloc[-1] == True  # noqa: E712


def test_is_hammer_detects_long_lower_wick_small_body():
    rows = _flat_series(5)
    rows.append(_candle(100, 100.2, 90, 100.5, _dates(6)[5]))  # tiny body, huge lower wick, minimal upper wick
    df = make_ohlcv(rows)
    result = fmf.is_hammer(df)
    assert result.iloc[-1] == True  # noqa: E712


def test_is_evening_star_detects_three_candle_pattern():
    rows = _flat_series(5)
    dates = _dates(8)
    rows.append(_candle(100, 110, 99.5, 109, dates[5]))  # big green candle
    rows.append(_candle(109.2, 110.5, 108.8, 109.5, dates[6]))  # small-bodied indecision candle
    rows.append(_candle(109, 109.2, 102, 103, dates[7]))  # red candle closing below candle1 midpoint (~104.5)
    df = make_ohlcv(rows)
    result = fmf.is_evening_star(df)
    assert result.iloc[-1] == True  # noqa: E712


def test_is_morning_star_detects_three_candle_pattern():
    rows = _flat_series(5)
    dates = _dates(8)
    rows.append(_candle(109, 109.5, 99, 100, dates[5]))  # big red candle
    rows.append(_candle(99.8, 100.5, 99.5, 100.2, dates[6]))  # small-bodied indecision candle
    rows.append(_candle(100.5, 107, 100.3, 106, dates[7]))  # green candle closing above candle1 midpoint (~104.5)
    df = make_ohlcv(rows)
    result = fmf.is_morning_star(df)
    assert result.iloc[-1] == True  # noqa: E712


def test_classify_direction_zone_matches_the_source_matrix():
    config = _cfg()
    assert fmf.classify_direction_zone(65, 65, config) == "strong_bullish"
    assert fmf.classify_direction_zone(65, 50, config) == "bullish_pullback"
    assert fmf.classify_direction_zone(30, 30, config) == "strong_bearish"
    assert fmf.classify_direction_zone(30, 50, config) == "bearish_bounce"
    assert fmf.classify_direction_zone(50, 50, config) == "neutral"
    assert fmf.classify_direction_zone(None, 50, config) == "unknown"


def test_evaluate_false_move_flags_distribution_risk_on_a_weak_breakout():
    # A long downtrend (keeps weekly RSI bearish) followed by a modest
    # 20-day bounce that's just enough to make a fresh 20-day high (daily
    # RSI recovers to neutral, not bullish - the weekly timeframe isn't
    # confirming the bounce), then a bearish engulfing right on the
    # breakout day - the "false breakout" signature this module targets.
    rng = np.random.default_rng(0)
    down = 150 - np.linspace(0, 50, 110) + rng.normal(0, 0.3, 110)
    bounce = down[-1] + np.linspace(0, 8, 20) + rng.normal(0, 0.1, 20)
    base = np.concatenate([down, bounce])
    dates = _dates(len(base) + 2)
    rows = [_candle(c, c * 1.005, c * 0.995, c, d) for c, d in zip(base, dates[: len(base)])]
    rows.append(_candle(base[-1], base[-1] + 3, base[-1] - 0.2, base[-1] + 2.8, dates[len(base)]))  # breakout day
    rows.append(_candle(base[-1] + 2.9, base[-1] + 3.0, base[-1] - 1, base[-1] - 0.5, dates[len(base) + 1]))  # bearish engulfing
    df = make_ohlcv(rows)

    config = _cfg()
    enriched = compute_indicators(df, config, benchmark=None)
    result = fmf.evaluate_false_move(enriched, config)

    assert result["recent_breakout"] is True
    assert "bearish_engulfing" in result["bearish_patterns"]
    assert result["direction_zone"] == "bearish_bounce"
    assert result["false_move_verdict"] == "distribution_risk"


def test_evaluate_false_move_reports_none_for_a_clean_breakout_with_no_reversal_candle():
    rng = np.random.default_rng(1)
    base = 100 + np.linspace(0, 30, 150) + rng.normal(0, 0.3, 150)  # strong, steady uptrend
    dates = _dates(150)
    rows = [_candle(c, c * 1.01, c * 0.99, c, d) for c, d in zip(base, dates)]
    df = make_ohlcv(rows)

    config = _cfg()
    enriched = compute_indicators(df, config, benchmark=None)
    result = fmf.evaluate_false_move(enriched, config)
    assert result["false_move_verdict"] == "none"


def test_evaluate_false_move_returns_insufficient_data_for_short_history():
    rows = _flat_series(20)
    df = make_ohlcv(rows)
    config = _cfg()
    enriched = compute_indicators(df, config, benchmark=None)
    result = fmf.evaluate_false_move(enriched, config)
    assert result["false_move_verdict"] == "insufficient_data"


def test_annotate_with_false_move_check_only_evaluates_passing_candidates():
    rng = np.random.default_rng(2)
    base = 100 + np.linspace(0, 30, 150) + rng.normal(0, 0.3, 150)
    dates = _dates(150)
    rows = [_candle(c, c * 1.01, c * 0.99, c, d) for c, d in zip(base, dates)]
    df = make_ohlcv(rows)
    config = _cfg()
    enriched = compute_indicators(df, config, benchmark=None)

    result_df = pd.DataFrame(
        [
            {"security_code": "A1", "passes_all_filters": True},
            {"security_code": "A2", "passes_all_filters": False},
        ]
    )
    enriched_cache = {"A1": enriched, "A2": enriched}
    out = fmf.annotate_with_false_move_check(result_df, config, enriched_cache)

    assert out.loc[out["security_code"] == "A1", "false_move_verdict"].iloc[0] == "none"
    assert out.loc[out["security_code"] == "A2", "false_move_verdict"].iloc[0] is None
