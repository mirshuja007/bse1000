from src.conviction import compute_conviction, score_momentum, score_trend
from src.config import load_config


def base_snapshot(**overrides) -> dict:
    snap = {
        "close": 105.0,
        "sma50": 100.0,
        "sma200": 95.0,
        "price_above_50": True,
        "price_above_200": True,
        "golden_cross": True,
        "rsi": 65.0,
        "macd_hist": 0.5,
        "roc10": 8.0,
        "volume_surge": 2.0,
        "obv_slope20": 1000.0,
        "donchian_breakout": True,
        "donchian_high": 104.0,
        "dma50_breakout_recent": True,
        "atr_pct_chg20": -0.5,
        "close_strength": 0.8,
        "rs_new_high": True,
        "relative_return_20d": 10.0,
        "pct_above_sma50": 5.0,
    }
    snap.update(overrides)
    return snap


def test_score_trend_perfect():
    snap = base_snapshot()
    assert score_trend(snap) == 100.0


def test_score_trend_zero_when_all_false():
    snap = base_snapshot(price_above_50=False, price_above_200=False, golden_cross=False)
    assert score_trend(snap) == 0.0


def test_momentum_peaks_in_ideal_rsi_zone():
    strong = score_momentum(base_snapshot(rsi=65))
    weak = score_momentum(base_snapshot(rsi=40))
    assert strong > weak


def test_momentum_extreme_overbought_scores_lower_than_ideal_zone():
    ideal = score_momentum(base_snapshot(rsi=68))
    extreme = score_momentum(base_snapshot(rsi=95))
    assert ideal > extreme


def test_compute_conviction_total_in_bounds():
    config = load_config()
    snap = base_snapshot()
    result = compute_conviction(snap, sector_strength_score=80, config=config)
    assert 0 <= result.total <= 100
    assert result.tier in {
        "Very High Conviction",
        "High Conviction",
        "Moderate Conviction",
        "Watchlist",
    }


def test_strong_setup_reaches_very_high_conviction_tier():
    config = load_config()
    snap = base_snapshot()
    result = compute_conviction(snap, sector_strength_score=90, config=config)
    assert result.tier in {"Very High Conviction", "High Conviction"}


def test_extended_penalty_reduces_score():
    config = load_config()
    normal = compute_conviction(base_snapshot(), sector_strength_score=80, config=config)
    extended_snap = base_snapshot(donchian_breakout=True, donchian_high=90.0, close=105.0)  # ~16.7% above pivot
    extended = compute_conviction(extended_snap, sector_strength_score=80, config=config)
    assert extended.penalties > 0
    assert extended.total < normal.total


def test_rsi_overbought_penalty_applies():
    config = load_config()
    result = compute_conviction(base_snapshot(rsi=95), sector_strength_score=80, config=config)
    assert result.penalties > 0
    assert any("overbought" in note.lower() for note in result.notes)


def test_weak_setup_lands_in_watchlist():
    config = load_config()
    weak_snap = base_snapshot(
        price_above_50=False,
        price_above_200=False,
        golden_cross=False,
        rsi=45,
        macd_hist=-0.2,
        roc10=-3,
        volume_surge=0.8,
        obv_slope20=-500,
        donchian_breakout=False,
        dma50_breakout_recent=False,
        rs_new_high=False,
        relative_return_20d=-5,
    )
    result = compute_conviction(weak_snap, sector_strength_score=20, config=config)
    assert result.tier == "Watchlist"
