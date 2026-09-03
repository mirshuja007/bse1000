"""Growth & Quality screen: 20%+ YoY sales growth, 40%+ YoY PAT growth (the
"20/40 rule"), low price volatility, and PEG < 1 - a separate, evolving
panel from the market-cap/EBITDA/ROE/D-E "Fundamentals filter", per your
request to keep this one apart so it can be refined independently.

Data sources, and what's actually verified vs assumed
-------------------------------------------------------
- **Sales growth, PAT growth, PEG's P/E input**: yfinance's annual income
  statement and `.info`. Same caveat as src/fundamentals.py: this sandbox
  has no network route to Yahoo Finance, so field names come from reading
  yfinance's own source (`Ticker.income_stmt` is confirmed by yfinance's
  own deprecation warning to index "Net Income"; "Total Revenue" the same
  way), not a live response. Run `debug_growth_financials.py` on a machine
  with real internet before trusting this.
- **Volatility ("low standard deviation")**: computed from the OHLCV data
  this app already pulled from Kite for the technical scan - NOT a new
  data source, so unlike the yfinance-derived numbers above, this one is
  fully verified (it's the same trusted data every indicator in this app
  already relies on).
- **PEG**: computed here as trailing P/E ÷ PAT growth %, not Yahoo's own
  `pegRatio`/`trailingPegRatio` field - that field has its own unverified
  coverage, and computing it ourselves from numbers we already fetch (and
  can show our work on) is more transparent than trusting a second
  unverified number on top of the first.

A stock with missing data is flagged "Unverified (missing: ...)", never
silently excluded, unless require_complete_data is turned on - same
philosophy as src/fundamentals.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT
from src.fundamentals import fetch_raw_info, to_yahoo_symbol

GROWTH_CACHE_DIR = REPO_ROOT / ".cache" / "growth_screen"
DEFAULT_CACHE_TTL_HOURS = 24 * 30  # annual financials update at most quarterly - monthly refresh is plenty


def _cache_path(yahoo_symbol: str) -> Path:
    return GROWTH_CACHE_DIR / f"{yahoo_symbol}.json"


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


def _save_cache(yahoo_symbol: str, data: dict) -> None:
    GROWTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(yahoo_symbol).write_text(json.dumps(data))


def _yoy_growth_pct(income_stmt: pd.DataFrame, row_label: str) -> float | None:
    """% change latest fiscal year vs prior fiscal year for one income
    statement line. yfinance orders columns most-recent-first. Returns None
    off a zero or negative prior-year base - a swing from a loss to a
    profit isn't a meaningful "growth %", and dividing by a near-zero base
    produces a wildly misleading number."""
    if income_stmt is None or row_label not in income_stmt.index:
        return None
    row = income_stmt.loc[row_label].dropna()
    if len(row) < 2:
        return None
    latest, prior = float(row.iloc[0]), float(row.iloc[1])
    if prior <= 0:
        return None
    return round((latest - prior) / prior * 100, 1)


def compute_growth_metrics(income_stmt: pd.DataFrame) -> dict:
    """Pure function - no I/O - so it's cheap to unit test with a synthetic
    income statement DataFrame shaped like yfinance's own."""
    return {
        "sales_growth_pct": _yoy_growth_pct(income_stmt, "Total Revenue"),
        "pat_growth_pct": _yoy_growth_pct(income_stmt, "Net Income"),
    }


def compute_price_volatility(enriched_df: pd.DataFrame | None, lookback_days: int = 252) -> float | None:
    """Annualized volatility (%) of daily returns over the trailing window.
    Computed from data already fetched for the technical scan - this is
    the one metric in this module that ISN'T an unverified yfinance number."""
    if enriched_df is None or len(enriched_df) < 20:
        return None
    closes = enriched_df["close"].tail(lookback_days)
    returns = closes.pct_change().dropna()
    if len(returns) < 10:
        return None
    annualized_pct = returns.std() * (252**0.5) * 100
    return round(float(annualized_pct), 1)


def compute_peg(trailing_pe: float | None, pat_growth_pct: float | None) -> float | None:
    """PEG = trailing P/E / PAT growth %, both already-fetched numbers -
    not Yahoo's own PEG field. Undefined (None) when P/E or growth is
    missing, zero, or negative (a negative PEG from negative growth or a
    negative P/E isn't a meaningful "cheap relative to growth" signal)."""
    if trailing_pe is None or pat_growth_pct is None or trailing_pe <= 0 or pat_growth_pct <= 0:
        return None
    return round(trailing_pe / pat_growth_pct, 2)


def evaluate_growth_screen(metrics: dict, config: dict) -> dict:
    """Pure pass/fail evaluation, same missing-data philosophy as
    src/fundamentals.py's evaluate_fundamentals(): a metric that's None
    never counts as a fail on its own unless require_complete_data is on."""
    g = config["filters"]["growth_quality"]

    checks = {
        "sales_growth_ok": (
            None if metrics["sales_growth_pct"] is None else metrics["sales_growth_pct"] >= g["sales_growth_min_pct"]
        ),
        "pat_growth_ok": (
            None if metrics["pat_growth_pct"] is None else metrics["pat_growth_pct"] >= g["pat_growth_min_pct"]
        ),
        "volatility_ok": (
            None
            if metrics["annualized_volatility_pct"] is None
            else metrics["annualized_volatility_pct"] <= g["max_annualized_volatility_pct"]
        ),
        "peg_ok": (None if metrics["peg"] is None else metrics["peg"] <= g["max_peg"]),
    }

    data_complete = all(v is not None for v in checks.values())
    known_checks = [v for v in checks.values() if v is not None]
    all_known_pass = all(known_checks) if known_checks else False

    if g.get("require_complete_data", False):
        passes = data_complete and all_known_pass
    else:
        passes = all_known_pass and len(known_checks) > 0

    missing = [name.replace("_ok", "") for name, v in checks.items() if v is None]
    note = f"Unverified (missing: {', '.join(missing)})" if missing else ""

    return {**checks, "data_complete": data_complete, "passes_growth_screen": passes, "growth_screen_note": note}


def fetch_and_evaluate_growth(
    tradingsymbol: str, exchange: str, enriched_df: pd.DataFrame | None, config: dict
) -> dict:
    """I/O + evaluation in one call, for use per-candidate in the scanner.
    `enriched_df` is the stock's already-fetched, indicator-enriched OHLCV
    (from the technical scan's enriched_cache) - volatility is computed
    from it directly, no extra network call."""
    yahoo_symbol = to_yahoo_symbol(tradingsymbol, exchange)

    cached = _load_cache(yahoo_symbol, DEFAULT_CACHE_TTL_HOURS)
    error = None
    if cached is not None:
        growth = {"sales_growth_pct": cached.get("sales_growth_pct"), "pat_growth_pct": cached.get("pat_growth_pct")}
        trailing_pe = cached.get("trailing_pe")
    else:
        try:
            import yfinance as yf

            income_stmt = yf.Ticker(yahoo_symbol).income_stmt
        except Exception as exc:
            income_stmt, error = None, str(exc)
        growth = compute_growth_metrics(income_stmt)

        info, info_error = fetch_raw_info(yahoo_symbol)
        trailing_pe = info.get("trailingPE") if info else None
        error = error or info_error

        _save_cache(yahoo_symbol, {**growth, "trailing_pe": trailing_pe})

    metrics = {
        **growth,
        "trailing_pe": trailing_pe,
        "annualized_volatility_pct": compute_price_volatility(enriched_df),
    }
    metrics["peg"] = compute_peg(metrics["trailing_pe"], metrics["pat_growth_pct"])

    result = evaluate_growth_screen(metrics, config)
    result.update(metrics)
    result["yahoo_symbol"] = yahoo_symbol
    if error:
        result["growth_screen_note"] = f"Fetch failed: {error}"
        result["passes_growth_screen"] = False
    return result


def annotate_with_growth_screen(result_df: pd.DataFrame, config: dict, enriched_cache: dict, progress_callback=None):
    """Second-pass enrichment, mirroring fundamentals.annotate_with_fundamentals:
    only evaluates candidates that already passed every technical filter."""
    out = result_df.copy()
    growth_cols = [
        "sales_growth_pct", "pat_growth_pct", "trailing_pe", "annualized_volatility_pct", "peg",
        "sales_growth_ok", "pat_growth_ok", "volatility_ok", "peg_ok",
        "data_complete", "passes_growth_screen", "growth_screen_note",
    ]
    for col in growth_cols:
        out[col] = None

    candidates = out[out["passes_all_filters"] == True]  # noqa: E712
    for i, (idx, row) in enumerate(candidates.iterrows()):
        enriched_df = enriched_cache.get(row["security_code"])
        result = fetch_and_evaluate_growth(row["tradingsymbol"], row["exchange"], enriched_df, config)
        for col in growth_cols:
            out.at[idx, col] = result.get(col)
        if progress_callback:
            progress_callback(i + 1, len(candidates))

    return out
