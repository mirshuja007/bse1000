"""Offline unit tests for src/fundamentals.py - synthetic yfinance-shaped
dicts only, no network access."""
import pandas as pd

from src import fundamentals as fnd
from src.config import load_config


def test_to_yahoo_symbol_maps_exchange_to_suffix():
    assert fnd.to_yahoo_symbol("RELIANCE", "NSE") == "RELIANCE.NS"
    assert fnd.to_yahoo_symbol("RELIANCE", "BSE") == "RELIANCE.BO"


def test_extract_metrics_converts_units_correctly():
    info = {
        "marketCap": 45_000_000_000,  # 4500 crore
        "ebitdaMargins": 0.187,       # -> 18.7%
        "returnOnEquity": 0.162,      # -> 16.2%
        "debtToEquity": 45.2,         # Yahoo's percentage-scaled convention -> ratio 0.45
    }
    metrics = fnd.extract_metrics(info)
    assert metrics["market_cap_cr"] == 4500.0
    assert metrics["ebitda_margin_pct"] == 18.7
    assert metrics["roe_pct"] == 16.2
    assert metrics["debt_to_equity"] == 0.45


def test_extract_metrics_leaves_missing_fields_as_none():
    metrics = fnd.extract_metrics({})
    assert metrics == {
        "market_cap_cr": None,
        "ebitda_margin_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
    }


def _cfg(**overrides):
    config = load_config()
    config["filters"]["fundamentals"].update(overrides)
    return config


def test_evaluate_fundamentals_passes_when_all_four_criteria_met():
    metrics = {"market_cap_cr": 3000.0, "ebitda_margin_pct": 20.0, "roe_pct": 18.0, "debt_to_equity": 1.2}
    result = fnd.evaluate_fundamentals(metrics, _cfg())
    assert result["passes_fundamentals"] is True
    assert result["data_complete"] is True
    assert result["fundamentals_note"] == ""


def test_evaluate_fundamentals_fails_on_market_cap_too_large():
    metrics = {"market_cap_cr": 8000.0, "ebitda_margin_pct": 20.0, "roe_pct": 18.0, "debt_to_equity": 1.2}
    result = fnd.evaluate_fundamentals(metrics, _cfg())
    assert result["market_cap_ok"] is False
    assert result["passes_fundamentals"] is False


def test_evaluate_fundamentals_fails_on_ebitda_margin_out_of_band():
    too_low = fnd.evaluate_fundamentals(
        {"market_cap_cr": 3000.0, "ebitda_margin_pct": 10.0, "roe_pct": 18.0, "debt_to_equity": 1.2}, _cfg()
    )
    too_high = fnd.evaluate_fundamentals(
        {"market_cap_cr": 3000.0, "ebitda_margin_pct": 30.0, "roe_pct": 18.0, "debt_to_equity": 1.2}, _cfg()
    )
    assert too_low["ebitda_margin_ok"] is False
    assert too_high["ebitda_margin_ok"] is False


def test_evaluate_fundamentals_missing_metric_is_unverified_not_failed_by_default():
    # roe_pct missing (e.g. Yahoo had no data) - default config doesn't
    # require complete data, so this should be flagged, not auto-failed,
    # as long as every metric that WAS available passes.
    metrics = {"market_cap_cr": 3000.0, "ebitda_margin_pct": 20.0, "roe_pct": None, "debt_to_equity": 1.2}
    result = fnd.evaluate_fundamentals(metrics, _cfg())
    assert result["roe_ok"] is None
    assert result["data_complete"] is False
    assert "roe" in result["fundamentals_note"]
    assert result["passes_fundamentals"] is True  # every known metric passed


def test_evaluate_fundamentals_require_complete_data_fails_on_any_missing_metric():
    metrics = {"market_cap_cr": 3000.0, "ebitda_margin_pct": 20.0, "roe_pct": None, "debt_to_equity": 1.2}
    result = fnd.evaluate_fundamentals(metrics, _cfg(require_complete_data=True))
    assert result["passes_fundamentals"] is False


def test_evaluate_fundamentals_all_missing_never_passes():
    metrics = {"market_cap_cr": None, "ebitda_margin_pct": None, "roe_pct": None, "debt_to_equity": None}
    result = fnd.evaluate_fundamentals(metrics, _cfg())
    assert result["passes_fundamentals"] is False
    assert result["data_complete"] is False


def test_annotate_with_fundamentals_only_fetches_for_passing_candidates(monkeypatch):
    calls = []

    def fake_fetch_and_evaluate(tradingsymbol, exchange, config):
        calls.append(tradingsymbol)
        return {
            "market_cap_cr": 1000.0, "ebitda_margin_pct": 20.0, "roe_pct": 18.0, "debt_to_equity": 1.0,
            "market_cap_ok": True, "ebitda_margin_ok": True, "roe_ok": True, "debt_to_equity_ok": True,
            "data_complete": True, "passes_fundamentals": True, "fundamentals_note": "",
        }

    monkeypatch.setattr(fnd, "fetch_and_evaluate", fake_fetch_and_evaluate)

    result_df = pd.DataFrame(
        [
            {"tradingsymbol": "AAA", "exchange": "NSE", "passes_all_filters": True},
            {"tradingsymbol": "BBB", "exchange": "NSE", "passes_all_filters": False},
        ]
    )
    annotated = fnd.annotate_with_fundamentals(result_df, _cfg())

    assert calls == ["AAA"]  # never fetched for the row that failed technical filters
    assert annotated.loc[0, "passes_fundamentals"] == True  # noqa: E712
    assert pd.isna(annotated.loc[1, "passes_fundamentals"])  # never evaluated
