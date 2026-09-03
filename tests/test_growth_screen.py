"""Offline unit tests for src/growth_screen.py - synthetic data only, no
network access."""
import numpy as np
import pandas as pd
import pytest

from src import growth_screen as gs
from src.config import load_config


def _income_stmt(revenue: list[float], net_income: list[float]) -> pd.DataFrame:
    """Mimics yfinance's Ticker.income_stmt shape: rows are line items,
    columns are fiscal year-end dates, most recent first (left)."""
    years = len(revenue)
    columns = [pd.Timestamp(f"{2024 - i}-03-31") for i in range(years)]
    return pd.DataFrame({"Total Revenue": revenue, "Net Income": net_income}, index=columns).T


def test_compute_growth_metrics_computes_yoy_pct_correctly():
    # Latest year revenue 120 vs prior 100 -> 20%; PAT 140 vs 100 -> 40%.
    stmt = _income_stmt(revenue=[120.0, 100.0, 90.0], net_income=[140.0, 100.0, 80.0])
    metrics = gs.compute_growth_metrics(stmt)
    assert metrics["sales_growth_pct"] == 20.0
    assert metrics["pat_growth_pct"] == 40.0


def test_compute_growth_metrics_none_when_prior_year_is_a_loss():
    # PAT swinging from a loss to a profit isn't a meaningful "growth %".
    stmt = _income_stmt(revenue=[120.0, 100.0], net_income=[50.0, -20.0])
    metrics = gs.compute_growth_metrics(stmt)
    assert metrics["sales_growth_pct"] == 20.0
    assert metrics["pat_growth_pct"] is None


def test_compute_growth_metrics_none_when_row_missing():
    stmt = pd.DataFrame({"Total Revenue": [100.0]}, index=[pd.Timestamp("2024-03-31")]).T
    metrics = gs.compute_growth_metrics(stmt)
    assert metrics["sales_growth_pct"] is None  # only one year of data
    assert metrics["pat_growth_pct"] is None  # row doesn't exist at all


def test_compute_growth_metrics_none_when_income_stmt_is_none():
    metrics = gs.compute_growth_metrics(None)
    assert metrics == {"sales_growth_pct": None, "pat_growth_pct": None}


def test_compute_price_volatility_low_for_a_steady_grind_up():
    # Smooth, near-constant daily gain -> very low realized volatility.
    closes = 100 * (1.001 ** np.arange(300))
    df = pd.DataFrame({"close": closes})
    vol = gs.compute_price_volatility(df)
    assert vol is not None
    assert vol < 5  # a near-straight line has ~0 realized SD


def test_compute_price_volatility_high_for_a_choppy_series():
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 3, 300))
    closes = np.clip(closes, 10, None)  # keep prices positive
    df = pd.DataFrame({"close": closes})
    vol = gs.compute_price_volatility(df)
    assert vol is not None
    assert vol > 20


def test_compute_price_volatility_none_with_insufficient_history():
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    assert gs.compute_price_volatility(df) is None


def test_compute_price_volatility_none_with_no_data():
    assert gs.compute_price_volatility(None) is None


def test_compute_peg_basic():
    assert gs.compute_peg(trailing_pe=30.0, pat_growth_pct=40.0) == 0.75


def test_compute_peg_none_when_growth_non_positive():
    assert gs.compute_peg(trailing_pe=30.0, pat_growth_pct=-5.0) is None
    assert gs.compute_peg(trailing_pe=30.0, pat_growth_pct=0.0) is None


def test_compute_peg_none_when_pe_missing_or_non_positive():
    assert gs.compute_peg(trailing_pe=None, pat_growth_pct=40.0) is None
    assert gs.compute_peg(trailing_pe=-5.0, pat_growth_pct=40.0) is None


def _cfg(**overrides):
    config = load_config()
    config["filters"]["growth_quality"].update(overrides)
    return config


def _metrics(**overrides):
    m = {"sales_growth_pct": 25.0, "pat_growth_pct": 45.0, "annualized_volatility_pct": 20.0, "peg": 0.6}
    m.update(overrides)
    return m


def test_evaluate_growth_screen_passes_when_all_four_criteria_met():
    result = gs.evaluate_growth_screen(_metrics(), _cfg())
    assert result["passes_growth_screen"] is True
    assert result["data_complete"] is True
    assert result["growth_screen_note"] == ""


def test_evaluate_growth_screen_fails_on_insufficient_sales_growth():
    result = gs.evaluate_growth_screen(_metrics(sales_growth_pct=10.0), _cfg())
    assert result["sales_growth_ok"] is False
    assert result["passes_growth_screen"] is False


def test_evaluate_growth_screen_fails_on_high_volatility():
    result = gs.evaluate_growth_screen(_metrics(annualized_volatility_pct=60.0), _cfg())
    assert result["volatility_ok"] is False
    assert result["passes_growth_screen"] is False


def test_evaluate_growth_screen_fails_on_peg_too_high():
    result = gs.evaluate_growth_screen(_metrics(peg=1.5), _cfg())
    assert result["peg_ok"] is False
    assert result["passes_growth_screen"] is False


def test_evaluate_growth_screen_missing_metric_is_unverified_not_failed_by_default():
    result = gs.evaluate_growth_screen(_metrics(peg=None), _cfg())
    assert result["peg_ok"] is None
    assert result["data_complete"] is False
    assert "peg" in result["growth_screen_note"]
    assert result["passes_growth_screen"] is True  # every known metric passed


def test_evaluate_growth_screen_require_complete_data_fails_on_any_missing_metric():
    result = gs.evaluate_growth_screen(_metrics(peg=None), _cfg(require_complete_data=True))
    assert result["passes_growth_screen"] is False


def test_evaluate_growth_screen_all_missing_never_passes():
    metrics = {"sales_growth_pct": None, "pat_growth_pct": None, "annualized_volatility_pct": None, "peg": None}
    result = gs.evaluate_growth_screen(metrics, _cfg())
    assert result["passes_growth_screen"] is False


def test_annotate_with_growth_screen_only_evaluates_passing_candidates(monkeypatch):
    calls = []

    def fake_fetch_and_evaluate_growth(tradingsymbol, exchange, enriched_df, config):
        calls.append(tradingsymbol)
        return {
            "sales_growth_pct": 25.0, "pat_growth_pct": 45.0, "trailing_pe": 20.0,
            "annualized_volatility_pct": 15.0, "peg": 0.5,
            "sales_growth_ok": True, "pat_growth_ok": True, "volatility_ok": True, "peg_ok": True,
            "data_complete": True, "passes_growth_screen": True, "growth_screen_note": "",
        }

    monkeypatch.setattr(gs, "fetch_and_evaluate_growth", fake_fetch_and_evaluate_growth)

    result_df = pd.DataFrame(
        [
            {"security_code": "A1", "tradingsymbol": "AAA", "exchange": "NSE", "passes_all_filters": True},
            {"security_code": "B1", "tradingsymbol": "BBB", "exchange": "NSE", "passes_all_filters": False},
        ]
    )
    annotated = gs.annotate_with_growth_screen(result_df, _cfg(), enriched_cache={})

    assert calls == ["AAA"]
    assert annotated.loc[0, "passes_growth_screen"] == True  # noqa: E712
    assert pd.isna(annotated.loc[1, "passes_growth_screen"])
