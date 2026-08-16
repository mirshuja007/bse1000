"""Conviction scoring engine.

This is a transparent, rule-based multi-factor scorecard - not a machine
learning model and not a promise of future returns. It encodes, in code,
the same checklist a discretionary momentum/swing trader would run through
by hand: trend structure, momentum quality, volume confirmation, breakout
quality/setup, relative strength vs the market, and sector participation.
Every sub-score and the final weighted total are returned so the UI can show
*why* a stock scored the way it did, not just a bare number.

Nothing here predicts direction with certainty - it ranks candidates by how
many well-established momentum characteristics they currently exhibit, for
a 1-15 trading day holding horizon. Always pair this with your own risk
management (position sizing, stop-loss, sector exposure limits).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return max(lo, min(hi, value))


def _linear(value: float, x0: float, x1: float, y0: float = 0.0, y1: float = 100.0) -> float:
    """Linearly map value from range [x0,x1] to [y0,y1], clamped at the ends."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    if x1 == x0:
        return y0
    t = (value - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


def score_trend(snap: dict) -> float:
    score = 0.0
    if snap.get("price_above_50"):
        score += 40
    if snap.get("price_above_200"):
        score += 30
    if snap.get("golden_cross"):
        score += 30
    return _clip(score)


def score_momentum(snap: dict) -> float:
    rsi = snap.get("rsi", float("nan"))
    if rsi is None or (isinstance(rsi, float) and math.isnan(rsi)):
        rsi_score = 0.0
    elif rsi < 50:
        rsi_score = 0.0
    elif rsi < 60:
        rsi_score = _linear(rsi, 50, 60, 0, 70)
    elif rsi <= 75:
        rsi_score = 100.0
    elif rsi <= 85:
        rsi_score = _linear(rsi, 75, 85, 100, 60)
    else:
        rsi_score = 40.0

    macd_hist = snap.get("macd_hist", 0) or 0
    macd_score = 100.0 if macd_hist > 0 else 0.0

    roc10 = snap.get("roc10", 0) or 0
    roc_score = _clip(roc10 * 5)  # +20% over 10d -> full marks

    return _clip(0.5 * rsi_score + 0.25 * macd_score + 0.25 * roc_score)


def score_volume(snap: dict) -> float:
    surge = snap.get("volume_surge", 0) or 0
    if surge < 1:
        surge_score = _linear(surge, 0, 1, 0, 50)
    elif surge < 1.5:
        surge_score = _linear(surge, 1, 1.5, 50, 75)
    elif surge < 2.5:
        surge_score = _linear(surge, 1.5, 2.5, 75, 100)
    else:
        surge_score = 100.0

    obv_slope = snap.get("obv_slope20", 0) or 0
    obv_score = 100.0 if obv_slope > 0 else 30.0

    return _clip(0.7 * surge_score + 0.3 * obv_score)


def score_breakout_quality(snap: dict) -> float:
    score = 0.0
    if snap.get("donchian_breakout"):
        score += 50
    if snap.get("dma50_breakout_recent"):
        score += 25
    atr_chg = snap.get("atr_pct_chg20", 0) or 0
    if atr_chg < 0:
        score += 15  # volatility was contracting into this move
    close_strength = snap.get("close_strength", 0) or 0
    score += _clip(close_strength * 10, 0, 10)
    return _clip(score)


def score_relative_strength(snap: dict) -> float:
    score = 0.0
    if snap.get("rs_new_high"):
        score += 50
    rel_return = snap.get("relative_return_20d", 0) or 0
    score += _linear(rel_return, 0, 20, 0, 50)
    return _clip(score)


def compute_pivot_distance_pct(snap: dict) -> float:
    """% the current close is above the relevant breakout reference (the
    Donchian pivot if it just triggered, else the 50DMA) - used to penalize
    chasing a name that has already run too far from its trigger."""
    if snap.get("donchian_breakout") and snap.get("donchian_high"):
        return (snap["close"] / snap["donchian_high"] - 1) * 100
    return snap.get("pct_above_sma50", 0) or 0


def _build_explanation(snap: dict, rsi_overbought_threshold: float) -> str:
    """Turn the same signals the score is built from into a short plain-
    English paragraph, so the conviction score isn't just a bare number -
    someone without a technical-analysis background can read this and
    understand why a stock ranked where it did."""
    sentences: list[str] = []

    surge = snap.get("volume_surge") or 0
    vol_clause = f" on {surge:.1f}x average volume" if surge >= 1.3 else ""
    if snap.get("donchian_breakout"):
        sentences.append(f"Broke out to a fresh multi-week high{vol_clause}.")
    elif snap.get("dma50_breakout_recent"):
        sentences.append(f"Recently reclaimed its 50-day moving average{vol_clause}.")
    elif vol_clause:
        sentences.append(f"Trading{vol_clause}, though without a fresh breakout trigger.")

    if snap.get("golden_cross") and snap.get("price_above_200"):
        sentences.append("Trading above both its 50- and 200-day averages in an established uptrend.")
    elif snap.get("price_above_50") and not snap.get("price_above_200"):
        sentences.append("Above its short-term trend but still below its longer-term 200-day average.")
    elif not snap.get("price_above_50"):
        sentences.append("Currently below its 50-day average - short-term momentum has cooled.")

    rsi = snap.get("rsi")
    if rsi is not None and rsi == rsi:  # not NaN
        if rsi > rsi_overbought_threshold:
            sentences.append(f"RSI of {rsi:.0f} is deeply overbought, raising pullback risk.")
        elif rsi >= 60:
            sentences.append(f"RSI of {rsi:.0f} shows strong, healthy momentum.")
        elif rsi >= 50:
            sentences.append(f"RSI of {rsi:.0f} shows momentum building but not yet strong.")
        else:
            sentences.append(f"RSI of {rsi:.0f} shows little momentum right now.")

    rel = snap.get("relative_return_20d")
    if rel is not None and rel == rel:
        if rel > 5:
            sentences.append(f"Outperforming the broader market by {rel:.1f}% over the past month.")
        elif rel < -5:
            sentences.append(f"Lagging the broader market by {abs(rel):.1f}% over the past month.")

    if not sentences:
        sentences.append("No standout momentum signal right now - sitting in a neutral zone.")

    return " ".join(sentences)


@dataclass
class ConvictionResult:
    trend: float
    momentum: float
    volume: float
    breakout_quality: float
    relative_strength: float
    sector_strength: float
    penalties: float
    total: float
    tier: str
    notes: list[str] = field(default_factory=list)
    explanation: str = ""


def compute_conviction(snap: dict, sector_strength_score: float, config: dict) -> ConvictionResult:
    cfg = config["conviction"]
    weights = cfg["weights"]

    trend = score_trend(snap)
    momentum = score_momentum(snap)
    volume = score_volume(snap)
    breakout_quality = score_breakout_quality(snap)
    relative_strength = score_relative_strength(snap)
    sector_strength = _clip(sector_strength_score)

    weighted_total = (
        trend * weights["trend"]
        + momentum * weights["momentum"]
        + volume * weights["volume"]
        + breakout_quality * weights["breakout_quality"]
        + relative_strength * weights["relative_strength"]
        + sector_strength * weights["sector_strength"]
    ) / sum(weights.values())

    penalties = 0.0
    notes: list[str] = []

    ext_cfg = cfg["extended_penalty"]
    if ext_cfg["enabled"]:
        pivot_dist = compute_pivot_distance_pct(snap)
        if pivot_dist > ext_cfg["pct_above_pivot_threshold"]:
            penalties += ext_cfg["penalty_points"]
            notes.append(
                f"Extended {pivot_dist:.1f}% above pivot - chasing risk, consider waiting for a pullback."
            )

    rsi_cfg = cfg["rsi_overbought_penalty"]
    if rsi_cfg["enabled"]:
        rsi = snap.get("rsi", 0) or 0
        if rsi > rsi_cfg["threshold"]:
            penalties += rsi_cfg["penalty_points"]
            notes.append(f"RSI {rsi:.0f} deeply overbought - elevated mean-reversion risk.")

    total = _clip(weighted_total - penalties)

    tiers = cfg["tiers"]
    if total >= tiers["very_high_conviction"]:
        tier = "Very High Conviction"
    elif total >= tiers["high_conviction"]:
        tier = "High Conviction"
    elif total >= tiers["moderate_conviction"]:
        tier = "Moderate Conviction"
    else:
        tier = "Watchlist"

    if snap.get("golden_cross") and snap.get("donchian_breakout"):
        notes.append("Golden-cross regime + fresh Donchian breakout - textbook momentum setup.")
    if not snap.get("rs_new_high", False):
        notes.append("Not yet at a new relative-strength high vs benchmark - lagging the tape.")

    explanation = _build_explanation(snap, rsi_cfg["threshold"])

    return ConvictionResult(
        trend=trend,
        momentum=momentum,
        volume=volume,
        breakout_quality=breakout_quality,
        relative_strength=relative_strength,
        sector_strength=sector_strength,
        penalties=penalties,
        total=total,
        tier=tier,
        notes=notes,
        explanation=explanation,
    )
