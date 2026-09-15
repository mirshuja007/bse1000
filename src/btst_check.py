"""BTST (Buy Today Sell Tomorrow) signature check: an honest historical
hit-rate test of one specific, precisely-defined setup - it does NOT claim
90% accuracy or a guaranteed 4-5% return. It reports whatever the real
number is, whatever that turns out to be.

The setup being tested
-----------------------
"Strong close + volume confirmation + weekly RSI not fighting the daily
move" - a stock qualifies as an `is_btst_setup` day when, using only data
that would have been known by that day's close:

1. **Strong close** (`close_strength`, already computed by the main scan)
   is at or above `min_close_strength` - the stock closed repeatedly near
   its day's high over the last few sessions, not near the low.
2. **Volume confirmation** (`volume_surge`, already computed by the main
   scan) is at or above `min_volume_surge` - real participation behind the
   move, not a thin/illiquid print.
3. **Weekly RSI doesn't disagree** - the most recently COMPLETED week's
   RSI (never the current, still-forming week - see `_weekly_rsi_no_lookahead`
   for why) is at or above `min_weekly_rsi`.

What "hit" means, honestly
---------------------------
This is a ONE-DAY forward test, matching what a BTST trade actually is:
bought at today's close, exited tomorrow. Two separate numbers are
reported because they answer different questions:

- `hit_target_rate_pct`: how often did TOMORROW'S HIGH reach
  `target_return_pct` above today's close - the best case if you sold
  intraday at the best possible moment. This is an upper bound, not what
  a real fixed-time exit would capture.
- `avg_next_close_return_pct` / `pct_days_positive`: what you'd have
  actually realized selling at tomorrow's close instead of the intraday
  high - a much more honest number for "what does this setup actually
  return."
- `hit_stop_rate_pct`: how often tomorrow's LOW fell `stop_loss_pct`
  below today's close - the downside that a target-only number hides.

Every rate is reported alongside a baseline (unflagged days) so a
"66% hit rate" that's barely above the 60% baseline rate is visibly not
much of an edge. `n_flagged` is always shown for the same reason as
Breakout Radar's hit-rate check: a high percentage from 8 occurrences
is not evidence of anything.

No lookahead: forward returns use shift(-1) (only valid on historical
bars, since "tomorrow" has already happened), and the weekly RSI used for
today's signal is deliberately last week's completed reading, never the
current week's in-progress one.
"""
from __future__ import annotations

import pandas as pd

from src.indicators import rsi
from src.timeframe import resample_ohlcv

_BASE_OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
MIN_BARS_FOR_SIGNAL = 100  # enough trading days for ~15+ completed weekly bars plus a lookback buffer


def _weekly_rsi_no_lookahead(base: pd.DataFrame, period: int = 14) -> pd.Series:
    """Weekly RSI aligned back onto the daily calendar, using ONLY weekly
    bars that were already fully complete as of each daily date. A weekly
    bar labeled with its closing Friday becomes "known" starting the very
    next calendar day - so every day within the FOLLOWING week (through
    and including that week's own Friday) sees this completed week's RSI,
    never its own still-forming week's value. Returns a Series indexed
    exactly like `base` (one value per daily row, in `base`'s original row
    order regardless of whether it arrived pre-sorted), with NaN wherever
    no completed weekly RSI exists yet."""
    weekly = resample_ohlcv(base, "W-FRI")
    weekly_known = pd.DataFrame(
        {
            "date": weekly["date"] + pd.Timedelta(days=1),  # available starting the day after that Friday
            "weekly_rsi": rsi(weekly["close"], period).to_numpy(),
        }
    )
    daily = base[["date"]].reset_index(names="_orig_idx").sort_values("date")
    merged = pd.merge_asof(daily, weekly_known.sort_values("date"), on="date", direction="backward")
    return merged.set_index("_orig_idx")["weekly_rsi"].reindex(base.index)


def compute_btst_signal(enriched_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """enriched_df: one stock's OHLCV already run through compute_indicators()
    - e.g. straight from the main scan's enriched_cache, same convention as
    src/false_move_filter.py. Adds `weekly_rsi_known` and `is_btst_setup`,
    reusing the already-present `close_strength`/`volume_surge`/`rsi`
    columns rather than recomputing them (so this never re-runs
    compute_indicators() and can't hit the duplicate-column pitfall
    Breakout Radar and the Gann panel both guard against)."""
    b = config["filters"]["btst_check"]
    out = enriched_df.copy()
    base = out[_BASE_OHLCV_COLUMNS]

    out["weekly_rsi_known"] = _weekly_rsi_no_lookahead(base)
    out["is_btst_setup"] = (
        (out["close_strength"] >= b["min_close_strength"])
        & (out["volume_surge"] >= b["min_volume_surge"])
        & (out["weekly_rsi_known"] >= b["min_weekly_rsi"])
    ).fillna(False)
    return out


def _prepare_btst_df(enriched_df: pd.DataFrame, config: dict) -> pd.DataFrame | None:
    """None (not a partial frame) when there isn't enough history for a
    completed weekly RSI plus a lookback buffer, or `enriched_df` hasn't
    actually been through compute_indicators() yet - an explicit "don't
    know" rather than a confusing KeyError deep inside pandas."""
    if len(enriched_df) < MIN_BARS_FOR_SIGNAL or "close_strength" not in enriched_df.columns:
        return None
    return compute_btst_signal(enriched_df, config)


def _btst_hit_rate_from_prepared(prepared: pd.DataFrame, target_return_pct: float, stop_loss_pct: float) -> dict:
    """Pure function on an already-prepared frame - no I/O, cheap to unit
    test with synthetic data."""
    next_close_return_pct = (prepared["close"].shift(-1) - prepared["close"]) / prepared["close"] * 100
    next_high_return_pct = (prepared["high"].shift(-1) - prepared["close"]) / prepared["close"] * 100
    next_low_return_pct = (prepared["low"].shift(-1) - prepared["close"]) / prepared["close"] * 100

    valid = next_close_return_pct.notna()  # excludes the last bar - no "tomorrow" to measure yet
    flagged = prepared["is_btst_setup"].fillna(False) & valid
    baseline = (~prepared["is_btst_setup"].fillna(False)) & valid

    hit_target = next_high_return_pct >= target_return_pct
    hit_stop = next_low_return_pct <= -stop_loss_pct

    return {
        "n_flagged": int(flagged.sum()),
        "n_flagged_hit_target": int((hit_target & flagged).sum()),
        "n_flagged_hit_stop": int((hit_stop & flagged).sum()),
        "sum_next_close_return_pct_flagged": float(next_close_return_pct[flagged].sum()),
        "n_flagged_positive": int((next_close_return_pct[flagged] > 0).sum()),
        "n_baseline": int(baseline.sum()),
        "n_baseline_hit_target": int((hit_target & baseline).sum()),
        "sum_next_close_return_pct_baseline": float(next_close_return_pct[baseline].sum()),
    }


def btst_hit_rate_for_stock(enriched_df: pd.DataFrame, config: dict) -> dict | None:
    """None when there isn't enough history to evaluate this stock at all."""
    b = config["filters"]["btst_check"]
    prepared = _prepare_btst_df(enriched_df, config)
    if prepared is None:
        return None
    return _btst_hit_rate_from_prepared(prepared, b["target_return_pct"], b["stop_loss_pct"])


def aggregate_btst_hit_rate(enriched_history: dict[str, pd.DataFrame], config: dict) -> dict:
    """enriched_history: {security_code: enriched OHLCV df} - e.g. reuse
    the main scan's enriched_cache directly. Sums per-stock counts across
    the scanned universe. Still just whatever history got fetched (~1-2
    years by default) - not a multi-cycle backtest across different market
    regimes. `n_stocks_included` and `n_flagged` are surfaced prominently
    for the same reason as Breakout Radar's hit-rate check: a headline
    percentage from a handful of occurrences isn't evidence of a
    90%-accuracy system, or any system."""
    totals = {
        "n_flagged": 0, "n_flagged_hit_target": 0, "n_flagged_hit_stop": 0,
        "sum_next_close_return_pct_flagged": 0.0, "n_flagged_positive": 0,
        "n_baseline": 0, "n_baseline_hit_target": 0, "sum_next_close_return_pct_baseline": 0.0,
    }
    n_stocks_included = 0

    for code, df in enriched_history.items():
        if df is None or df.empty:
            continue
        result = btst_hit_rate_for_stock(df, config)
        if result is None:
            continue
        n_stocks_included += 1
        for key in totals:
            totals[key] += result[key]

    b = config["filters"]["btst_check"]
    n_flagged = totals["n_flagged"]
    n_baseline = totals["n_baseline"]

    return {
        "n_stocks_included": n_stocks_included,
        "n_flagged": n_flagged,
        "hit_target_rate_pct": round(totals["n_flagged_hit_target"] / n_flagged * 100, 1) if n_flagged else None,
        "hit_stop_rate_pct": round(totals["n_flagged_hit_stop"] / n_flagged * 100, 1) if n_flagged else None,
        "avg_next_close_return_pct": round(totals["sum_next_close_return_pct_flagged"] / n_flagged, 2) if n_flagged else None,
        "pct_days_positive": round(totals["n_flagged_positive"] / n_flagged * 100, 1) if n_flagged else None,
        "n_baseline": n_baseline,
        "baseline_hit_target_rate_pct": (
            round(totals["n_baseline_hit_target"] / n_baseline * 100, 1) if n_baseline else None
        ),
        "avg_baseline_next_close_return_pct": (
            round(totals["sum_next_close_return_pct_baseline"] / n_baseline, 2) if n_baseline else None
        ),
        "target_return_pct": b["target_return_pct"],
        "stop_loss_pct": b["stop_loss_pct"],
    }
