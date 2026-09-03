"""Automatic, no-click record of every stock the scanner has ever
recommended (i.e. passed all filters), so the results table can tell a
genuinely new idea apart from one that's resurfacing.

This is deliberately separate from src/tracker.py's "Tracked picks": that
module is an opt-in, per-stock decision to follow a position's P&L (you
choose which ones to track). This module logs *every* passing candidate on
*every* scan, automatically, so "have I seen this one before, and does it
still look the same" works even for stocks you never explicitly tracked.

Log entries live in `data/recommendation_history.csv` locally (git-ignored,
like the other generated files - this is your personal scan history, not
sample data for the repo) and, when src/github_store.py is configured, are
mirrored to a dedicated GitHub branch too - the local copy alone doesn't
survive a Streamlit Cloud redeploy, which wipes the container it lives on.
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd

from src import github_store
from src.config import REPO_ROOT

HISTORY_FILE = REPO_ROOT / "data" / "recommendation_history.csv"
GITHUB_DATA_PATH = "data/recommendation_history.csv"

_COLUMNS = [
    "security_code",
    "tradingsymbol",
    "company_name",
    "universe",
    "first_seen_date",
    "first_seen_price",
    "first_seen_conviction_score",
    "first_seen_conviction_tier",
    "first_seen_supertrend_bullish",
    "first_seen_price_above_50",
    "last_seen_date",
    "last_seen_price",
    "times_seen",
]

STATUS_NEW = "New"
STATUS_STILL_LONG = "Repeat - still long"
STATUS_EXIT = "Repeat - exit / trail SL"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[_COLUMNS]


_last_sync_error: str | None = None


def get_last_sync_error() -> str | None:
    """Non-None if the most recent save reached the local file fine but
    failed to sync to GitHub (bad token, network issue, etc.)."""
    return _last_sync_error


def load_recommendation_history() -> pd.DataFrame:
    if github_store.is_configured():
        content = github_store.read_file(GITHUB_DATA_PATH)
        if content is None:
            return pd.DataFrame(columns=_COLUMNS)
        return _normalize_columns(pd.read_csv(io.StringIO(content), dtype={"security_code": str}))

    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=_COLUMNS)
    return _normalize_columns(pd.read_csv(HISTORY_FILE, dtype={"security_code": str}))


def _save_history(df: pd.DataFrame) -> None:
    global _last_sync_error
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HISTORY_FILE, index=False)  # local copy: fast, and a fallback if the GitHub sync below fails

    if github_store.is_configured():
        try:
            github_store.write_file(GITHUB_DATA_PATH, df.to_csv(index=False), "Update recommendation_history.csv")
            _last_sync_error = None
        except Exception as exc:
            _last_sync_error = str(exc)


def _price_above_50(row: dict) -> bool:
    pct = row.get("pct_above_sma50")
    return pct is not None and pct == pct and pct >= 0


def _classify_repeat(row: dict) -> str:
    """A repeat candidate keeps its "long" status only if every signal that
    originally qualified it still holds today: it still passes all filters,
    Supertrend (if computed) is still bullish, and price is still above the
    50DMA. Any one of those failing means the setup that got it recommended
    has changed - flag it for exit/trailing the stop rather than treating
    it as a fresh idea."""
    still_passes = bool(row.get("passes_all_filters"))
    supertrend_ok = row.get("supertrend_bullish") is not False  # None (not computed) doesn't count against it
    above_50_ok = _price_above_50(row)
    return STATUS_STILL_LONG if (still_passes and supertrend_ok and above_50_ok) else STATUS_EXIT


def annotate_with_history(result_df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add first_recommended_date / entry_price / recommendation_status
    columns to a scan result, comparing against history as it stood
    *before* this scan (pass the already-loaded `history` explicitly in
    tests; the app wires it to load_recommendation_history() beforehand).
    Call update_recommendation_history() separately, after annotating, to
    fold today's results into the log for next time."""
    if history is None:
        history = load_recommendation_history()
    hist_by_code = history.set_index("security_code") if not history.empty else None

    out = result_df.copy()
    first_dates, entry_prices, statuses = [], [], []
    for _, row in out.iterrows():
        code = row["security_code"]
        if hist_by_code is not None and code in hist_by_code.index:
            h = hist_by_code.loc[code]
            first_dates.append(h["first_seen_date"])
            entry_prices.append(h["first_seen_price"])
            statuses.append(_classify_repeat(row.to_dict()))
        else:
            first_dates.append(row.get("as_of"))
            entry_prices.append(row.get("close"))
            statuses.append(STATUS_NEW)

    out["first_recommended_date"] = first_dates
    out["entry_price"] = entry_prices
    out["recommendation_status"] = statuses
    return out


def update_recommendation_history(result_df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Upsert today's passing candidates into the history log: first-seen
    stocks get a new row, already-known ones get last_seen_date/price and
    times_seen bumped (first_seen_* fields never change once set)."""
    if history is None:
        history = load_recommendation_history()
    candidates = result_df[result_df["passes_all_filters"] == True]  # noqa: E712
    if candidates.empty:
        return history

    rows = history.to_dict("records")
    index_by_code = {r["security_code"]: i for i, r in enumerate(rows)}

    for _, c in candidates.iterrows():
        code = c["security_code"]
        seen_date = c.get("as_of")
        if code in index_by_code:
            idx = index_by_code[code]
            rows[idx]["last_seen_date"] = seen_date
            rows[idx]["last_seen_price"] = c.get("close")
            rows[idx]["times_seen"] = int(rows[idx].get("times_seen") or 1) + 1
        else:
            rows.append(
                {
                    "security_code": code,
                    "tradingsymbol": c.get("tradingsymbol"),
                    "company_name": c.get("company_name"),
                    "universe": c.get("universe"),
                    "first_seen_date": seen_date,
                    "first_seen_price": c.get("close"),
                    "first_seen_conviction_score": c.get("conviction_score"),
                    "first_seen_conviction_tier": c.get("conviction_tier"),
                    "first_seen_supertrend_bullish": c.get("supertrend_bullish"),
                    "first_seen_price_above_50": _price_above_50(c.to_dict()),
                    "last_seen_date": seen_date,
                    "last_seen_price": c.get("close"),
                    "times_seen": 1,
                }
            )
            index_by_code[code] = len(rows) - 1

    updated = pd.DataFrame(rows, columns=_COLUMNS)
    _save_history(updated)
    return updated
