"""Offline unit tests for src/portfolio_metrics.py - synthetic data only."""
import pandas as pd

from src import portfolio_metrics as pm


def make_row(code, score, passes_all=True, passes_growth=None, passes_fund=None, **overrides):
    row = {
        "security_code": code,
        "tradingsymbol": code,
        "sector": overrides.pop("sector", "Energy"),
        "conviction_score": score,
        "passes_all_filters": passes_all,
        "passes_growth_screen": passes_growth,
        "passes_fundamentals": passes_fund,
        "sales_growth_pct": overrides.pop("sales_growth_pct", 25.0),
        "pat_growth_pct": overrides.pop("pat_growth_pct", 45.0),
        "peg": overrides.pop("peg", 0.8),
        "trailing_pe": overrides.pop("trailing_pe", 30.0),
        "market_cap_cr": overrides.pop("market_cap_cr", 5000.0),
        "turnover_cr": overrides.pop("turnover_cr", 10.0),
    }
    row.update(overrides)
    return row


def test_build_portfolio_selects_top_n_by_conviction_among_passing_stocks():
    rows = [make_row(f"S{i}", score=float(100 - i)) for i in range(30)]
    result_df = pd.DataFrame(rows)
    portfolio = pm.build_portfolio(result_df, portfolio_size=25)
    assert portfolio["n_holdings"] == 25
    assert list(portfolio["holdings"]["security_code"]) == [f"S{i}" for i in range(25)]


def test_build_portfolio_excludes_stocks_failing_the_base_technical_filter():
    rows = [
        make_row("A", score=99.0, passes_all=False),  # highest score but fails base filter
        make_row("B", score=80.0, passes_all=True),
    ]
    result_df = pd.DataFrame(rows)
    portfolio = pm.build_portfolio(result_df, portfolio_size=25)
    assert portfolio["n_holdings"] == 1
    assert portfolio["holdings"]["security_code"].iloc[0] == "B"


def test_build_portfolio_excludes_stocks_failing_an_enabled_growth_screen():
    rows = [
        make_row("A", score=99.0, passes_growth=False),  # growth screen ran and failed
        make_row("B", score=80.0, passes_growth=True),
    ]
    result_df = pd.DataFrame(rows)
    portfolio = pm.build_portfolio(result_df, portfolio_size=25)
    assert portfolio["n_holdings"] == 1
    assert portfolio["holdings"]["security_code"].iloc[0] == "B"


def test_build_portfolio_does_not_exclude_when_growth_screen_never_ran():
    # passes_growth_screen is null (not False) for every row - the screen was off.
    rows = [make_row("A", score=99.0, passes_growth=None), make_row("B", score=80.0, passes_growth=None)]
    result_df = pd.DataFrame(rows)
    portfolio = pm.build_portfolio(result_df, portfolio_size=25)
    assert portfolio["n_holdings"] == 2


def test_build_portfolio_is_smaller_when_fewer_than_requested_pass():
    rows = [make_row("A", score=90.0), make_row("B", score=80.0, passes_all=False)]
    result_df = pd.DataFrame(rows)
    portfolio = pm.build_portfolio(result_df, portfolio_size=25)
    assert portfolio["n_holdings"] == 1
    assert portfolio["requested_size"] == 25


def test_build_portfolio_weight_pct_is_equal_and_sums_to_100():
    rows = [make_row(f"S{i}", score=float(i)) for i in range(4)]
    result_df = pd.DataFrame(rows)
    portfolio = pm.build_portfolio(result_df, portfolio_size=4)
    weights = portfolio["holdings"]["weight_pct"]
    assert (weights == 25.0).all()
    assert round(weights.sum(), 2) == 100.0


def test_build_portfolio_metrics_average_only_over_non_missing_values():
    rows = [
        make_row("A", score=90.0, sales_growth_pct=20.0),
        make_row("B", score=80.0, sales_growth_pct=None),
        make_row("C", score=70.0, sales_growth_pct=40.0),
    ]
    result_df = pd.DataFrame(rows)
    portfolio = pm.build_portfolio(result_df, portfolio_size=25)
    m = portfolio["metrics"]["avg_sales_growth_pct"]
    assert m["n_with_data"] == 2
    assert m["value"] == 30.0  # mean of 20 and 40, B excluded


def test_build_portfolio_handles_missing_metric_column_entirely():
    rows = [{"security_code": "A", "tradingsymbol": "A", "conviction_score": 90.0, "passes_all_filters": True}]
    result_df = pd.DataFrame(rows)
    portfolio = pm.build_portfolio(result_df, portfolio_size=25)
    assert portfolio["metrics"]["avg_sales_growth_pct"] == {"value": None, "n_with_data": 0}
    assert portfolio["n_sectors"] is None


def test_build_portfolio_counts_distinct_sectors():
    rows = [
        make_row("A", score=90.0, sector="Energy"),
        make_row("B", score=80.0, sector="IT"),
        make_row("C", score=70.0, sector="Energy"),
    ]
    result_df = pd.DataFrame(rows)
    portfolio = pm.build_portfolio(result_df, portfolio_size=25)
    assert portfolio["n_sectors"] == 2


def test_build_portfolio_beta_is_exact_average_under_equal_weighting():
    rows = [make_row("A", score=90.0, beta=0.8), make_row("B", score=80.0, beta=1.2)]
    result_df = pd.DataFrame(rows)
    portfolio = pm.build_portfolio(result_df, portfolio_size=25)
    assert portfolio["metrics"]["portfolio_beta"]["value"] == 1.0
    assert portfolio["metrics"]["portfolio_beta"]["n_with_data"] == 2


def test_build_portfolio_handles_zero_qualifying_stocks():
    rows = [make_row("A", score=90.0, passes_all=False)]
    result_df = pd.DataFrame(rows)
    portfolio = pm.build_portfolio(result_df, portfolio_size=25)
    assert portfolio["n_holdings"] == 0
    assert portfolio["holdings"].empty


def test_build_portfolio_sets_quarterly_review_cadence():
    from datetime import timedelta

    result_df = pd.DataFrame([make_row("A", score=90.0)])
    portfolio = pm.build_portfolio(result_df, portfolio_size=25)
    assert portfolio["review_frequency"] == "Quarterly"
    assert portfolio["next_review_date"] - portfolio["last_review_date"] == timedelta(days=pm.REVIEW_CADENCE_DAYS)
