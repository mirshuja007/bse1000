"""Offline unit tests for src/timeframe.py - synthetic data only."""
import pandas as pd

from src.timeframe import resample_ohlcv


def test_resample_weekly_aggregates_ohlcv_correctly():
    # Two full trading weeks (Mon-Fri), simple increasing values so each
    # field's aggregation is unambiguous to check by hand.
    dates = pd.bdate_range("2024-01-01", periods=10)  # Mon 1/1 .. Fri 1/12
    df = pd.DataFrame(
        {
            "date": dates,
            "open": range(100, 110),
            "high": range(110, 120),
            "low": range(90, 100),
            "close": range(105, 115),
            "volume": [1000] * 10,
        }
    )
    weekly = resample_ohlcv(df, "W-FRI")

    assert len(weekly) == 2
    week1 = weekly.iloc[0]
    assert week1["open"] == 100  # Monday's open
    assert week1["high"] == 114  # max high across the week
    assert week1["low"] == 90  # min low across the week
    assert week1["close"] == 109  # Friday's close
    assert week1["volume"] == 5000  # summed

    week2 = weekly.iloc[1]
    assert week2["open"] == 105
    assert week2["close"] == 114


def test_resample_monthly_aggregates_ohlcv_correctly():
    dates = pd.bdate_range("2024-01-01", periods=45)  # spans Jan and into Feb/Mar
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 45,
            "high": [110.0] * 45,
            "low": [90.0] * 45,
            "close": [105.0] * 45,
            "volume": [1000] * 45,
        }
    )
    monthly = resample_ohlcv(df, "ME")
    assert len(monthly) >= 2
    assert (monthly["high"] == 110.0).all()
    assert (monthly["low"] == 90.0).all()


def test_resample_drops_periods_with_no_trades():
    # A gap of missing dates shouldn't produce a phantom all-NaN row.
    dates = list(pd.bdate_range("2024-01-01", periods=5)) + list(pd.bdate_range("2024-03-01", periods=5))
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 10,
            "high": [110.0] * 10,
            "low": [90.0] * 10,
            "close": [105.0] * 10,
            "volume": [1000] * 10,
        }
    )
    monthly = resample_ohlcv(df, "ME")
    assert monthly["open"].notna().all()  # no phantom rows for the gap month(s)
