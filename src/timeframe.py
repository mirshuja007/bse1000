"""Resample daily OHLCV into weekly/monthly bars, so the exact same
indicator pipeline in src/indicators.py can run at higher timeframes
without duplicating any indicator logic - a "weekly RSI(14)" or "monthly
Donchian(20)" is standard multi-timeframe practice: same periods, just
applied to bars that each span more calendar time.
"""
from __future__ import annotations

import pandas as pd

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """rule: 'W-FRI' for weekly (week ending Friday, matching the NSE
    trading week) or 'ME' for calendar-month-end. Drops any trailing
    partial period with no trades in it."""
    dated = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["date"]), name="date"))
    resampled = dated[["open", "high", "low", "close", "volume"]].resample(rule).agg(_AGG)
    resampled = resampled.dropna(subset=["open"])
    return resampled.reset_index()
