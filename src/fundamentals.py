"""Fundamental-quality screen (market cap, EBITDA margin, ROE, debt/equity)
using yfinance as the data source, per your explicit choice.

IMPORTANT - read before trusting this filter with real capital
-----------------------------------------------------------------
This sandbox has no network route to Yahoo Finance (outbound is proxied
and Yahoo isn't on the allowlist), so none of this has been verified
against live data - only against yfinance's own source code, which tells
me *which* Yahoo modules it requests (`financialData`, `defaultKeyStatistics`,
`assetProfile`, `summaryDetail`) and therefore which field names to expect
in `.info`. It does not tell me whether Yahoo actually has good data for
the specific small/mid-cap Indian names this filter targets - that's the
coverage gap I flagged before building this.

Two things you should verify yourself before trusting this filter, using
`python debug_fundamentals.py <TRADINGSYMBOL> ...` (prints the raw
yfinance `.info` fields this module reads, for real tickers):
1. Coverage - does yfinance actually return marketCap/ebitdaMargins/
   returnOnEquity/debtToEquity for the sub-Rs.5000cr names you care about,
   or come back mostly empty?
2. The debtToEquity unit - Yahoo's `financialData.debtToEquity` has long
   been reported (by other yfinance users, not verified by me here) as a
   percentage-scaled number (e.g. 45.2 meaning a ratio of 0.452) rather
   than a raw ratio. This module divides by 100 on that assumption - if
   your debug run shows otherwise for real NSE tickers, that conversion
   needs to change.

Given both of those are unverified, this filter defaults to disabled, and
any stock with missing data is never silently excluded (see
`require_complete_data` below) - it's flagged as unverified instead.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.config import REPO_ROOT

FUNDAMENTALS_CACHE_DIR = REPO_ROOT / ".cache" / "fundamentals"
DEFAULT_CACHE_TTL_HOURS = 24 * 7  # fundamentals move slowly - weekly refresh is plenty

_EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}


def to_yahoo_symbol(tradingsymbol: str, exchange: str) -> str:
    suffix = _EXCHANGE_SUFFIX.get(exchange, ".NS")
    return f"{tradingsymbol}{suffix}"


def _cache_path(yahoo_symbol: str) -> Path:
    return FUNDAMENTALS_CACHE_DIR / f"{yahoo_symbol}.json"


def _load_cache(yahoo_symbol: str, ttl_hours: float) -> dict | None:
    path = _cache_path(yahoo_symbol)
    if not path.exists():
        return None
    if (time.time() - path.stat().st_mtime) / 3600 > ttl_hours:
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_cache(yahoo_symbol: str, info: dict) -> None:
    FUNDAMENTALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(yahoo_symbol).write_text(json.dumps(info, default=str))


def fetch_raw_info(yahoo_symbol: str, cache_ttl_hours: float = DEFAULT_CACHE_TTL_HOURS) -> tuple[dict, str | None]:
    """Returns (info_dict, error). info_dict is {} on failure - callers
    should treat that as "data unavailable", not "failed every check"."""
    cached = _load_cache(yahoo_symbol, cache_ttl_hours)
    if cached is not None:
        return cached, None

    try:
        import yfinance as yf

        info = yf.Ticker(yahoo_symbol).info or {}
    except Exception as exc:
        return {}, str(exc)

    _save_cache(yahoo_symbol, info)
    return info, None


def extract_metrics(info: dict) -> dict:
    """Pure unit conversion from yfinance's raw `.info` dict to the units
    the screen's thresholds are written in (crores, percent, ratio).
    Any field missing from `info` comes back as None - never guessed."""
    market_cap = info.get("marketCap")
    ebitda_margin = info.get("ebitdaMargins")
    roe = info.get("returnOnEquity")
    debt_to_equity_raw = info.get("debtToEquity")

    return {
        "market_cap_cr": round(market_cap / 1e7, 1) if market_cap is not None else None,
        "ebitda_margin_pct": round(ebitda_margin * 100, 1) if ebitda_margin is not None else None,
        "roe_pct": round(roe * 100, 1) if roe is not None else None,
        # Yahoo's financialData.debtToEquity is reported percentage-scaled
        # (see module docstring) - unverified against live NSE data.
        "debt_to_equity": round(debt_to_equity_raw / 100, 2) if debt_to_equity_raw is not None else None,
    }


def evaluate_fundamentals(metrics: dict, config: dict) -> dict:
    """Pure pass/fail evaluation against configured thresholds. A metric
    that's None (unavailable, not fetched) never counts as a fail on its
    own - it's tracked separately as `data_complete=False` so a stock
    isn't silently dropped just because Yahoo has no data for it, unless
    `require_complete_data` is explicitly turned on."""
    f = config["filters"]["fundamentals"]

    checks = {
        "market_cap_ok": (
            None if metrics["market_cap_cr"] is None else metrics["market_cap_cr"] <= f["max_market_cap_cr"]
        ),
        "ebitda_margin_ok": (
            None
            if metrics["ebitda_margin_pct"] is None
            else f["ebitda_margin_min"] <= metrics["ebitda_margin_pct"] <= f["ebitda_margin_max"]
        ),
        "roe_ok": (None if metrics["roe_pct"] is None else metrics["roe_pct"] >= f["min_roe_pct"]),
        "debt_to_equity_ok": (
            None if metrics["debt_to_equity"] is None else metrics["debt_to_equity"] <= f["max_debt_to_equity"]
        ),
    }

    data_complete = all(v is not None for v in checks.values())
    known_checks = [v for v in checks.values() if v is not None]
    all_known_pass = all(known_checks) if known_checks else False

    if f.get("require_complete_data", False):
        passes = data_complete and all_known_pass
    else:
        passes = all_known_pass and len(known_checks) > 0

    missing = [name.replace("_ok", "") for name, v in checks.items() if v is None]
    note = f"Unverified (missing: {', '.join(missing)})" if missing else ""

    return {**checks, "data_complete": data_complete, "passes_fundamentals": passes, "fundamentals_note": note}


def fetch_and_evaluate(tradingsymbol: str, exchange: str, config: dict) -> dict:
    """I/O + evaluation in one call, for use per-candidate in the scanner."""
    yahoo_symbol = to_yahoo_symbol(tradingsymbol, exchange)
    info, error = fetch_raw_info(yahoo_symbol)
    metrics = extract_metrics(info)
    result = evaluate_fundamentals(metrics, config)
    result.update(metrics)
    result["yahoo_symbol"] = yahoo_symbol
    if error:
        result["fundamentals_note"] = f"Fetch failed: {error}"
        result["passes_fundamentals"] = False
    return result


def annotate_with_fundamentals(result_df, config: dict, progress_callback=None):
    """Second-pass enrichment of a scan result: only fetches fundamentals
    for rows that already passed every technical filter, since this is one
    slow network call per stock and there's no point spending it on names
    the technical screen already rejected. Returns a copy of result_df with
    the fundamental columns added (None/unset for rows that were never
    checked, i.e. didn't pass the technical filters)."""
    import pandas as pd

    out = result_df.copy()
    fundamentals_cols = [
        "market_cap_cr", "ebitda_margin_pct", "roe_pct", "debt_to_equity",
        "market_cap_ok", "ebitda_margin_ok", "roe_ok", "debt_to_equity_ok",
        "data_complete", "passes_fundamentals", "fundamentals_note",
    ]
    for col in fundamentals_cols:
        out[col] = None

    candidates = out[out["passes_all_filters"] == True]  # noqa: E712
    for i, (idx, row) in enumerate(candidates.iterrows()):
        result = fetch_and_evaluate(row["tradingsymbol"], row["exchange"], config)
        for col in fundamentals_cols:
            out.at[idx, col] = result.get(col)
        if progress_callback:
            progress_callback(i + 1, len(candidates))

    return out
