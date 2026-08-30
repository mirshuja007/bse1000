#!/usr/bin/env python3
"""One-off diagnostic: check what yfinance actually returns for real Indian
small/mid-cap tickers, since src/fundamentals.py was built with no network
route to Yahoo Finance available - the field names come from reading
yfinance's own source code, not from a live response. Two things this
needs to confirm before the fundamentals filter can be trusted:

1. Coverage - are marketCap/ebitdaMargins/returnOnEquity/debtToEquity
   actually populated for names in the Rs.500cr-5000cr range, or mostly
   empty?
2. The debtToEquity unit - src/fundamentals.py assumes Yahoo returns it
   percentage-scaled (divides by 100 to get a ratio). If that's wrong for
   real data, extract_metrics() needs fixing.

Run from the project root (needs internet access to Yahoo Finance, which
this box has and the sandbox that built this did not):
    python debug_fundamentals.py RELIANCE TCS <a few small/mid-caps you care about>

Paste the full output back so the filter can be corrected against real
data before you rely on it.
"""
import sys

from src.config import load_config
from src.fundamentals import evaluate_fundamentals, extract_metrics, to_yahoo_symbol

DEFAULT_TICKERS = ["RELIANCE", "TCS", "CYIENTDLM", "PRECWIRE"]  # last two are small/mid-caps from your own scan output

tickers = sys.argv[1:] or DEFAULT_TICKERS
config = load_config()

import yfinance as yf  # noqa: E402  (import after argv parsing so --help-style typos fail fast above)

for symbol in tickers:
    yahoo_symbol = to_yahoo_symbol(symbol, "NSE")
    print(f"\n{'=' * 60}\n{symbol} -> {yahoo_symbol}\n{'=' * 60}")

    try:
        info = yf.Ticker(yahoo_symbol).info or {}
    except Exception as exc:
        print(f"  FETCH FAILED: {exc}")
        continue

    print(f"  .info returned {len(info)} keys total")

    raw_fields = ["marketCap", "ebitdaMargins", "returnOnEquity", "debtToEquity"]
    print("  Raw fields this module reads:")
    for field in raw_fields:
        print(f"    {field} = {info.get(field)!r}")

    metrics = extract_metrics(info)
    print(f"  After unit conversion: {metrics}")

    result = evaluate_fundamentals(metrics, config)
    print(f"  Evaluated: passes_fundamentals={result['passes_fundamentals']} "
          f"data_complete={result['data_complete']} note={result['fundamentals_note']!r}")

print(
    "\nDone. If debtToEquity above looks like a raw ratio already (e.g. 0.45 "
    "instead of 45.2) for a stock whose real D/E you know, the /100 division "
    "in extract_metrics() needs to be removed - tell me and I'll fix it."
)
