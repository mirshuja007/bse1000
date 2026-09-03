#!/usr/bin/env python3
"""One-off diagnostic: check what yfinance's annual income statement
actually returns for real Indian small/mid-cap tickers, since
src/growth_screen.py was built with no network route to Yahoo Finance -
the "Total Revenue" / "Net Income" row labels come from yfinance's own
source code (confirmed via its own deprecation-warning text), not a live
response.

Run from the project root (needs real internet access to Yahoo Finance):
    python debug_growth_financials.py RELIANCE TCS <symbols you care about>

Paste the full output back so this can be corrected against real data
before you rely on it.
"""
import sys

from src.config import load_config
from src.fundamentals import to_yahoo_symbol
from src.growth_screen import compute_growth_metrics, compute_peg, evaluate_growth_screen

DEFAULT_TICKERS = ["RELIANCE", "TCS", "CYIENTDLM", "PRECWIRE"]

tickers = sys.argv[1:] or DEFAULT_TICKERS
config = load_config()

import yfinance as yf  # noqa: E402

for symbol in tickers:
    yahoo_symbol = to_yahoo_symbol(symbol, "NSE")
    print(f"\n{'=' * 60}\n{symbol} -> {yahoo_symbol}\n{'=' * 60}")

    try:
        ticker = yf.Ticker(yahoo_symbol)
        income_stmt = ticker.income_stmt
        info = ticker.info or {}
    except Exception as exc:
        print(f"  FETCH FAILED: {exc}")
        continue

    if income_stmt is None or income_stmt.empty:
        print("  income_stmt is empty/None - no annual financials available for this ticker.")
        continue

    print(f"  income_stmt shape: {income_stmt.shape}, columns (fiscal years): {list(income_stmt.columns)}")
    print(f"  Row labels present (first 15): {list(income_stmt.index)[:15]}")
    print(f"  'Total Revenue' in index: {'Total Revenue' in income_stmt.index}")
    print(f"  'Net Income' in index: {'Net Income' in income_stmt.index}")
    if "Total Revenue" in income_stmt.index:
        print(f"  Total Revenue row: {income_stmt.loc['Total Revenue'].to_dict()}")
    if "Net Income" in income_stmt.index:
        print(f"  Net Income row: {income_stmt.loc['Net Income'].to_dict()}")

    metrics = compute_growth_metrics(income_stmt)
    trailing_pe = info.get("trailingPE")
    metrics["trailing_pe"] = trailing_pe
    metrics["annualized_volatility_pct"] = None  # not computed here - comes from this app's own OHLCV in real use
    metrics["peg"] = compute_peg(trailing_pe, metrics["pat_growth_pct"])
    print(f"  Computed: {metrics}")

print(
    "\nDone. If 'Total Revenue' or 'Net Income' aren't in the row labels above for a "
    "real ticker, tell me the actual label yfinance used and I'll fix the row lookup."
)
