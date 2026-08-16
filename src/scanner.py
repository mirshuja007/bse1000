"""Orchestrates a full scan: apply configurable filters to every stock's
latest indicator snapshot, compute sector-level breadth, score conviction,
and return one ranked DataFrame.
"""
from __future__ import annotations

import pandas as pd

from src.conviction import compute_conviction
from src.indicators import compute_indicators, latest_snapshot


def compute_sector_strength(snapshots: dict[str, dict], mapping: pd.DataFrame) -> dict[str, float]:
    """Rank sectors by median 20-day relative return, map to a 0-100 score
    per sector so stocks in currently-leading sectors get a boost."""
    sector_by_code = mapping.set_index("bse_code")["sector"].to_dict()

    sector_returns: dict[str, list[float]] = {}
    for code, snap in snapshots.items():
        sector = sector_by_code.get(code, "Unclassified")
        rel = snap.get("relative_return_20d")
        if rel is not None and rel == rel:  # not NaN
            sector_returns.setdefault(sector, []).append(rel)

    medians = {
        sector: sorted(vals)[len(vals) // 2] for sector, vals in sector_returns.items() if vals
    }
    if not medians:
        return {sector: 50.0 for sector in sector_by_code.values()}

    ranked = sorted(medians.items(), key=lambda kv: kv[1])
    n = len(ranked)
    percentile_score = {
        sector: (i / max(n - 1, 1)) * 100 for i, (sector, _) in enumerate(ranked)
    }
    return percentile_score


def compute_risk_levels(snap: dict, config: dict) -> dict:
    """ATR-based suggested stop-loss/target, so a ranked candidate comes
    with an actionable entry/exit frame instead of just a score. Not a
    recommendation to size or place any specific trade."""
    close = snap.get("close")
    atr = snap.get("atr")
    if not close or not atr or atr != atr:  # missing or NaN
        return {"stop_loss": None, "target": None, "risk_pct": None, "reward_pct": None}

    risk_cfg = config["risk"]
    stop_loss = close - risk_cfg["atr_stop_multiplier"] * atr
    target = close + risk_cfg["atr_target_multiplier"] * atr
    return {
        "stop_loss": round(stop_loss, 2),
        "target": round(target, 2),
        "risk_pct": round((close - stop_loss) / close * 100, 2),
        "reward_pct": round((target - close) / close * 100, 2),
    }


def passes_filters(snap: dict, config: dict) -> tuple[bool, list[str]]:
    f = config["filters"]
    reasons = []

    if f["liquidity"]["enabled"]:
        if (snap.get("close") or 0) < f["liquidity"]["min_price"]:
            reasons.append("price_below_min")
        if (snap.get("turnover_cr") or 0) < f["liquidity"]["min_avg_turnover_cr"]:
            reasons.append("turnover_below_min")

    if f["volume"]["enabled"]:
        surge = snap.get("volume_surge")
        if surge is None or surge != surge or surge < f["volume"]["surge_multiplier"]:
            reasons.append("no_volume_surge")

    if f["dma50_breakout"]["enabled"]:
        if not snap.get("dma50_breakout_recent"):
            reasons.append("no_50dma_breakout")
        elif (snap.get("pct_above_sma50") or 0) < f["dma50_breakout"]["min_pct_above"]:
            reasons.append("50dma_breakout_too_shallow")

    if f["dma200_breakout"]["enabled"]:
        if not snap.get("dma200_breakout_recent"):
            reasons.append("no_200dma_breakout")

    if f["rsi"]["enabled"]:
        rsi = snap.get("rsi")
        if rsi is None or rsi != rsi:
            reasons.append("rsi_unavailable")
        elif not (f["rsi"]["min_rsi"] <= rsi <= f["rsi"]["max_rsi"]):
            reasons.append("rsi_out_of_range")

    if f["donchian_breakout"]["enabled"]:
        if not snap.get("donchian_breakout"):
            reasons.append("no_donchian_breakout")

    if f["trend_filter"]["enabled"]:
        tf = f["trend_filter"]
        if tf["require_50_above_200"] and not snap.get("golden_cross"):
            reasons.append("no_golden_cross")
        if tf["require_price_above_50"] and not snap.get("price_above_50"):
            reasons.append("price_below_50dma")
        if tf["require_price_above_200"] and not snap.get("price_above_200"):
            reasons.append("price_below_200dma")

    if f["adx"]["enabled"]:
        adx_val = snap.get("adx")
        if adx_val is None or adx_val != adx_val or adx_val < f["adx"]["min_adx"]:
            reasons.append("adx_below_min")

    return (len(reasons) == 0, reasons)


def run_scan(
    history: dict[str, pd.DataFrame],
    mapping: pd.DataFrame,
    config: dict,
    benchmark: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    history: {bse_code: raw OHLCV DataFrame}
    mapping: universe mapping DataFrame (bse_code, company_name_raw, sector, tradingsymbol, exchange, ...)
    Returns one row per stock with indicator values, filter pass/fail, and conviction score.
    """
    name_by_code = mapping.set_index("bse_code")["company_name_raw"].to_dict()
    sector_by_code = mapping.set_index("bse_code")["sector"].to_dict()
    tsym_by_code = mapping.set_index("bse_code")["tradingsymbol"].to_dict()
    exch_by_code = mapping.set_index("bse_code")["exchange"].to_dict()

    snapshots: dict[str, dict] = {}
    enriched_cache: dict[str, pd.DataFrame] = {}
    skipped: list[dict] = []

    min_bars_needed = 60  # need at least this many bars for RSI/ADX/MACD to be meaningful
    for code, df in history.items():
        if df is None or len(df) < min_bars_needed:
            skipped.append({"bse_code": code, "reason": "insufficient_history"})
            continue
        enriched = compute_indicators(df, config, benchmark)
        enriched_cache[code] = enriched
        snapshots[code] = latest_snapshot(enriched)

    sector_scores = compute_sector_strength(snapshots, mapping)

    rows = []
    for code, snap in snapshots.items():
        sector = sector_by_code.get(code, "Unclassified")
        passed, fail_reasons = passes_filters(snap, config)
        result = compute_conviction(snap, sector_scores.get(sector, 50.0), config)
        risk = compute_risk_levels(snap, config)

        rows.append(
            {
                "bse_code": code,
                "company_name": name_by_code.get(code, code),
                "sector": sector,
                "tradingsymbol": tsym_by_code.get(code, code),
                "exchange": exch_by_code.get(code, "BSE"),
                "as_of": snap.get("as_of"),
                "close": snap.get("close"),
                "rsi": snap.get("rsi"),
                "adx": snap.get("adx"),
                "pct_above_sma50": snap.get("pct_above_sma50"),
                "pct_above_sma200": snap.get("pct_above_sma200"),
                "volume_surge": snap.get("volume_surge"),
                "turnover_cr": snap.get("turnover_cr"),
                "donchian_breakout": snap.get("donchian_breakout"),
                "dma50_breakout_recent": snap.get("dma50_breakout_recent"),
                "dma200_breakout_recent": snap.get("dma200_breakout_recent"),
                "golden_cross": snap.get("golden_cross"),
                "relative_return_20d": snap.get("relative_return_20d"),
                "rs_new_high": snap.get("rs_new_high"),
                "roc5": snap.get("roc5"),
                "roc10": snap.get("roc10"),
                "passes_all_filters": passed,
                "fail_reasons": ", ".join(fail_reasons),
                "conviction_score": round(result.total, 1),
                "conviction_tier": result.tier,
                "score_trend": round(result.trend, 1),
                "score_momentum": round(result.momentum, 1),
                "score_volume": round(result.volume, 1),
                "score_breakout_quality": round(result.breakout_quality, 1),
                "score_relative_strength": round(result.relative_strength, 1),
                "score_sector_strength": round(result.sector_strength, 1),
                "penalties": result.penalties,
                "notes": " | ".join(result.notes),
                "explanation": result.explanation,
                "stop_loss": risk["stop_loss"],
                "target": risk["target"],
                "risk_pct": risk["risk_pct"],
                "reward_pct": risk["reward_pct"],
            }
        )

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values("conviction_score", ascending=False).reset_index(drop=True)
    return result_df, enriched_cache, skipped
