from src.config import load_config
from src.scanner import compute_risk_levels


def test_compute_risk_levels_uses_configured_atr_multiples():
    config = load_config()
    config["risk"]["atr_stop_multiplier"] = 1.5
    config["risk"]["atr_target_multiplier"] = 3.0
    snap = {"close": 100.0, "atr": 4.0}
    risk = compute_risk_levels(snap, config)
    assert risk["stop_loss"] == 94.0  # 100 - 1.5*4
    assert risk["target"] == 112.0  # 100 + 3.0*4
    assert risk["risk_pct"] == 6.0
    assert risk["reward_pct"] == 12.0


def test_compute_risk_levels_handles_missing_atr():
    config = load_config()
    snap = {"close": 100.0, "atr": None}
    risk = compute_risk_levels(snap, config)
    assert risk["stop_loss"] is None
    assert risk["target"] is None
