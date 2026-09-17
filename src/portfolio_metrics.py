"""Portfolio Metrics: carves a concentrated, equal-weighted N-stock
portfolio out of the main scan's results (the "start broad, then build a
focused N-stock portfolio" framework), plus a summary dashboard of its
growth/valuation/liquidity attributes.

Selection: the top `portfolio_size` candidates by conviction_score among
stocks that ALREADY pass every currently-ENABLED screen (the base
technical filters, Growth & Quality if enabled, Fundamentals if enabled).
This is a genuinely screened, concentrated portfolio - deliberately
different from Trade Cards, which has NO such requirement by design (that
feature exists to surface every strong-scoring name for manual review,
borderline ones included; this one exists to build an actual candidate
portfolio from names that have already cleared every active screen). If
fewer than `portfolio_size` stocks pass, the portfolio is simply
smaller - it is never padded with non-passing names to hit a target count.

A screen that never ran (its column is all-null, because e.g. Growth &
Quality was switched off for this scan) does not exclude anything - only
a screen that actually ran and returned False excludes a stock. This
matters because passes_growth_screen/passes_fundamentals are null (not
False) for every row when their filter is disabled.

What this does NOT do (yet): lock membership between reviews. Every call
re-selects fresh top-N holdings from whatever result_df it's given -
it does not persist "these are this quarter's N stocks" and hold them
steady until the next review date arrives. `next_review_date` is a
suggested cadence to re-run this, not an enforced rebalance lock. A real
locked-membership quarterly rebalance (persisting holdings, only
reshuffling on an actual review date) would be a separate, bigger follow-up.

Averages are computed only over holdings with a non-missing value for
that metric - `n_with_data` is reported alongside every average so a mean
over, say, 6 of 25 holdings (because Fundamentals wasn't enabled for this
scan) is never mistaken for a mean over the full portfolio.

`portfolio_beta` (average of holdings' own per-stock betas, from
src/growth_screen.py's compute_beta) IS an exact, real portfolio
statistic under equal weighting - beta is linear in weights regardless of
correlation structure between holdings. Portfolio-level Standard
Deviation and Sharpe/Treynor ratios are NOT included, and can't be built
this cheaply: real portfolio SD needs the full covariance matrix between
holdings (diversification typically makes it lower than any simple
average), and Sharpe/Treynor need a multi-year simulated portfolio return
series plus a risk-free rate - both a separate, bigger follow-up with a
genuine survivorship-bias caveat (see README).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

REVIEW_CADENCE_DAYS = 91  # ~quarterly


def _screened_candidates(result_df: pd.DataFrame) -> pd.DataFrame:
    mask = result_df["passes_all_filters"] == True  # noqa: E712
    for screen_col in ("passes_growth_screen", "passes_fundamentals"):
        if screen_col in result_df.columns:
            ran = result_df[screen_col].notna()  # null = screen was off, not a fail
            mask &= (~ran) | (result_df[screen_col] == True)  # noqa: E712
    return result_df[mask]


def _avg(df: pd.DataFrame, col: str) -> dict:
    if col not in df.columns:
        return {"value": None, "n_with_data": 0}
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    return {"value": round(float(series.mean()), 2) if len(series) else None, "n_with_data": int(len(series))}


def build_portfolio(result_df: pd.DataFrame, portfolio_size: int = 25) -> dict:
    """Pure function - no I/O - so it's cheap to unit test. `holdings` is
    the selected slice of result_df with an added `weight_pct` column
    (equal weight = 100 / n_holdings, so it always sums to ~100%
    regardless of how many holdings actually qualified)."""
    screened = _screened_candidates(result_df)
    holdings = screened.sort_values("conviction_score", ascending=False).head(portfolio_size).copy()
    n = len(holdings)
    holdings["weight_pct"] = round(100.0 / n, 2) if n else None

    metrics = {
        "avg_sales_growth_pct": _avg(holdings, "sales_growth_pct"),
        "avg_pat_growth_pct": _avg(holdings, "pat_growth_pct"),
        "avg_peg": _avg(holdings, "peg"),
        "avg_trailing_pe": _avg(holdings, "trailing_pe"),
        "avg_market_cap_cr": _avg(holdings, "market_cap_cr"),
        "avg_turnover_cr": _avg(holdings, "turnover_cr"),
        # Under EQUAL weighting, portfolio beta is exactly the average of
        # constituent betas (beta is linear in weights) - this is a true
        # portfolio statistic, not an approximation. Average volatility,
        # by contrast, is NOT portfolio volatility (diversification/
        # correlation effects mean real portfolio SD is typically lower
        # than the simple average) - deliberately not computed here to
        # avoid implying it's the same thing. A true portfolio SD needs
        # the full covariance matrix - see the Sharpe/Beta/Treynor
        # follow-up noted in the module docstring.
        "portfolio_beta": _avg(holdings, "beta"),
    }
    n_sectors = int(holdings["sector"].nunique()) if "sector" in holdings.columns and n else None

    today = datetime.now().date()
    return {
        "requested_size": portfolio_size,
        "n_holdings": n,
        "holdings": holdings,
        "metrics": metrics,
        "n_sectors": n_sectors,
        "review_frequency": "Quarterly",
        "last_review_date": today,
        "next_review_date": today + timedelta(days=REVIEW_CADENCE_DAYS),
    }
