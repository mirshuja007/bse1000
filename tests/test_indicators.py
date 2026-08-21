"""Offline unit tests for src/indicators.py - synthetic data only, no
network/Kite access required."""
import numpy as np
import pandas as pd
import pytest

from src import indicators as ind


def make_ohlcv(closes: list[float], volumes: list[float] | None = None, start="2024-01-01") -> pd.DataFrame:
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    closes = np.array(closes, dtype=float)
    highs = closes * 1.01
    lows = closes * 0.99
    opens = closes * 1.0
    if volumes is None:
        volumes = np.full(n, 100_000.0)
    return pd.DataFrame(
        {"date": dates, "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    )


def test_rsi_bounds_and_trend():
    # Monotonically rising series should push RSI high (>70), never outside [0, 100].
    closes = np.linspace(100, 200, 60)
    df = make_ohlcv(list(closes))
    r = ind.rsi(df["close"], period=14)
    assert r.dropna().between(0, 100).all()
    assert r.iloc[-1] > 70


def test_rsi_falling_series_is_low():
    closes = np.linspace(200, 100, 60)
    df = make_ohlcv(list(closes))
    r = ind.rsi(df["close"], period=14)
    assert r.iloc[-1] < 30


def test_sma_matches_pandas_rolling():
    closes = list(np.random.default_rng(0).normal(100, 5, 250).cumsum())
    df = make_ohlcv(closes)
    s50 = ind.sma(df["close"], 50)
    expected = df["close"].rolling(50).mean()
    pd.testing.assert_series_equal(s50, expected, check_names=False)


def test_atr_non_negative():
    closes = list(np.random.default_rng(1).normal(0, 2, 100).cumsum() + 100)
    df = make_ohlcv(closes)
    a = ind.atr(df, period=14)
    assert (a.dropna() >= 0).all()


def test_adx_bounds():
    closes = list(np.linspace(100, 160, 120))
    df = make_ohlcv(closes)
    adx_df = ind.adx(df, period=14)
    assert adx_df["adx"].dropna().between(0, 100).all()


def test_macd_hist_equals_macd_minus_signal():
    closes = list(np.random.default_rng(2).normal(0, 1, 100).cumsum() + 100)
    df = make_ohlcv(closes)
    m = ind.macd(df["close"])
    diff = (m["macd"] - m["macd_signal"]) - m["macd_hist"]
    assert diff.abs().max() < 1e-9


def test_donchian_breakout_flags_new_high():
    # Flat at 100 for 30 days, then a clean breakout above the 20-day high.
    closes = [100.0] * 30 + [130.0]
    df = make_ohlcv(closes)
    dh = ind.donchian_high(df["high"], period=20)
    breakout = df["close"] > dh
    assert breakout.iloc[-1] == True  # noqa: E712


def test_crossed_above_within_detects_recent_cross():
    price = pd.Series([10, 10, 10, 12, 12, 12])
    level = pd.Series([11, 11, 11, 11, 11, 11])
    result = ind.crossed_above_within(price, level, lookback=3)
    assert result.iloc[3] == True  # crossed exactly on this bar  # noqa: E712
    assert result.iloc[5] == True  # still within lookback window  # noqa: E712
    assert result.iloc[1] == False  # noqa: E712


def test_volume_surge_ratio():
    volumes = [100] * 20 + [300]
    closes = [100.0] * 21
    df = make_ohlcv(closes, volumes=volumes)
    surge = ind.volume_surge_ratio(df["volume"], lookback=20)
    assert surge.iloc[-1] == pytest.approx(3.0, rel=1e-6)


def test_close_strength_range():
    closes = list(np.random.default_rng(3).normal(0, 1, 50).cumsum() + 100)
    df = make_ohlcv(closes)
    cs = ind.close_strength(df)
    assert cs.dropna().between(0, 1).all()


def test_compute_indicators_integration_has_expected_columns():
    rng = np.random.default_rng(42)
    closes = list(100 + rng.normal(0, 1, 260).cumsum())
    volumes = list(rng.integers(50_000, 200_000, 260))
    df = make_ohlcv(closes, volumes=volumes)
    bench_closes = list(1000 + rng.normal(0, 5, 260).cumsum())
    bench = make_ohlcv(bench_closes)

    config = {
        "filters": {
            "rsi": {"period": 14},
            "adx": {"period": 14},
            "donchian_breakout": {"period": 20},
            "volume": {"lookback": 20},
            "liquidity": {"turnover_lookback": 20},
            "dma50_breakout": {"lookback_days": 5},
            "dma200_breakout": {"lookback_days": 10},
            "supertrend": {"period": 10, "multiplier": 3.0, "flip_lookback_days": 3},
        }
    }
    enriched = ind.compute_indicators(df, config, benchmark=bench)
    for col in [
        "sma50",
        "sma200",
        "rsi",
        "atr",
        "adx",
        "macd",
        "donchian_breakout",
        "rs_line",
        "supertrend",
        "supertrend_bullish",
        "supertrend_flip_recent",
    ]:
        assert col in enriched.columns
    snap = ind.latest_snapshot(enriched)
    assert "golden_cross" in snap
    assert "price_above_50" in snap


def test_supertrend_line_stays_below_price_in_a_clean_uptrend():
    # A smooth, strong uptrend should settle into a bullish Supertrend with
    # the line acting as support (below price), not resistance.
    closes = list(np.linspace(100, 200, 120))
    df = make_ohlcv(closes)
    st = ind.supertrend(df, period=10, multiplier=3.0)
    tail = st.iloc[-20:]
    assert tail["supertrend_bullish"].all()
    assert (df["close"].iloc[-20:].reset_index(drop=True) > tail["supertrend"].reset_index(drop=True)).all()


def test_supertrend_flips_bullish_after_a_sharp_reversal():
    # Flat/declining, then a sharp sustained rally - the line should flip
    # from resistance (bearish) to support (bullish) at some point.
    declining = list(np.linspace(150, 100, 60))
    rallying = list(np.linspace(100, 180, 60))
    df = make_ohlcv(declining + rallying)
    st = ind.supertrend(df, period=10, multiplier=3.0)
    assert not st["supertrend_bullish"].iloc[30]  # still in the decline
    assert st["supertrend_bullish"].iloc[-1]  # rallied hard enough to flip


def test_flipped_bullish_within_detects_recent_flip_only():
    bullish = pd.Series([False, False, False, True, True, True, True])
    result = ind.flipped_bullish_within(bullish, lookback=2)
    assert result.iloc[3] == True  # noqa: E712  the flip itself
    assert result.iloc[4] == True  # noqa: E712  still within lookback=2
    assert result.iloc[6] == False  # noqa: E712  flip is now 3 bars stale
