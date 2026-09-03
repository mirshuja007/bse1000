"""Track scanner recommendations forward in time: did a logged pick reach
its target, hit its stop, run out the 1-15 day holding horizon untouched,
or is the underlying setup itself deteriorating (trend/momentum weakening)
even though price hasn't hit either level yet?

This is deliberately separate from the scanner's own conviction score: the
score is a point-in-time read of a setup, this module is the honest record
of what actually happened to it afterward - the only way to know if the
scoring is any good, and required reading before trusting it with real size.

Log entries are per-pick snapshots taken at the moment you choose to track
one (via the "Track this pick" button in the Stock Detail panel). Stored in
`data/tracked_picks.csv` locally (git-ignored - this is your personal
trading record, not sample data for the repo) and, when src/github_store.py
is configured, mirrored to a dedicated GitHub branch too - the local copy
alone doesn't survive a Streamlit Cloud redeploy, which wipes the container
it lives on."""
from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src import github_store
from src.config import REPO_ROOT
from src.indicators import compute_indicators, latest_snapshot

TRACKED_PICKS_FILE = REPO_ROOT / "data" / "tracked_picks.csv"
GITHUB_DATA_PATH = "data/tracked_picks.csv"

OPEN_STATUS = "OPEN"
CLOSED_STATUSES = {"HIT_TARGET", "HIT_STOPLOSS", "EXPIRED", "CLOSED_MANUAL"}

_COLUMNS = [
    "pick_id",
    "date_logged",
    "security_code",
    "tradingsymbol",
    "exchange",
    "company_name",
    "universe",
    "instrument_token",
    "entry_price",
    "stop_loss",
    "target",
    "supertrend_stop",
    "entry_conviction_score",
    "entry_conviction_tier",
    "entry_rsi",
    "entry_supertrend_bullish",
    "entry_price_above_50",
    "status",
    "exit_date",
    "exit_price",
    "realized_pct",
    "last_checked_date",
    "last_checked_price",
    "unrealized_pct",
    "days_held",
    "signal_notes",
]


def _empty_tracked_picks_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLUMNS)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[_COLUMNS]


_last_sync_error: str | None = None


def get_last_sync_error() -> str | None:
    """Non-None if the most recent save reached the local file fine but
    failed to sync to GitHub (bad token, network issue, etc.) - the app
    surfaces this as a warning so a silent sync failure doesn't get
    mistaken for successful cross-redeploy persistence."""
    return _last_sync_error


def load_tracked_picks() -> pd.DataFrame:
    if github_store.is_configured():
        content = github_store.read_file(GITHUB_DATA_PATH)
        if content is None:
            return _empty_tracked_picks_df()
        return _normalize_columns(pd.read_csv(io.StringIO(content), dtype={"security_code": str}))

    if not TRACKED_PICKS_FILE.exists():
        return _empty_tracked_picks_df()
    return _normalize_columns(pd.read_csv(TRACKED_PICKS_FILE, dtype={"security_code": str}))


def _save_tracked_picks(df: pd.DataFrame) -> None:
    global _last_sync_error
    TRACKED_PICKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRACKED_PICKS_FILE, index=False)  # local copy: fast, and a fallback if the GitHub sync below fails

    if github_store.is_configured():
        try:
            github_store.write_file(GITHUB_DATA_PATH, df.to_csv(index=False), "Update tracked_picks.csv")
            _last_sync_error = None
        except Exception as exc:
            _last_sync_error = str(exc)


def log_pick(scan_row: dict, as_of: date | None = None) -> tuple[bool, str]:
    """Append a new tracked pick from one row of a scan result (as a dict -
    e.g. `result_df.loc[code].to_dict()`). Refuses to log a duplicate while
    an OPEN entry already exists for the same security, since re-tracking
    an already-open recommendation would double-count it in the win-rate
    stats without representing a real second decision."""
    as_of = as_of or date.today()
    existing = load_tracked_picks()
    code = str(scan_row["security_code"])

    already_open = existing[(existing["security_code"] == code) & (existing["status"] == OPEN_STATUS)]
    if len(already_open):
        return False, f"{scan_row.get('company_name', code)} already has an open tracked pick from {already_open.iloc[0]['date_logged']}."

    pct_above_50 = scan_row.get("pct_above_sma50")
    entry_price_above_50 = pct_above_50 is not None and pct_above_50 == pct_above_50 and pct_above_50 >= 0

    row = {
        "pick_id": f"{code}_{as_of.isoformat()}",
        "date_logged": as_of.isoformat(),
        "security_code": code,
        "tradingsymbol": scan_row.get("tradingsymbol"),
        "exchange": scan_row.get("exchange"),
        "company_name": scan_row.get("company_name"),
        "universe": scan_row.get("universe"),
        "instrument_token": scan_row.get("instrument_token"),
        "entry_price": scan_row.get("close"),
        "stop_loss": scan_row.get("stop_loss"),
        "target": scan_row.get("target"),
        "supertrend_stop": scan_row.get("supertrend_stop"),
        "entry_conviction_score": scan_row.get("conviction_score"),
        "entry_conviction_tier": scan_row.get("conviction_tier"),
        "entry_rsi": scan_row.get("rsi"),
        "entry_supertrend_bullish": scan_row.get("supertrend_bullish"),
        "entry_price_above_50": entry_price_above_50,
        "status": OPEN_STATUS,
        "exit_date": None,
        "exit_price": None,
        "realized_pct": None,
        "last_checked_date": as_of.isoformat(),
        "last_checked_price": scan_row.get("close"),
        "unrealized_pct": 0.0,
        "days_held": 0,
        "signal_notes": "",
    }

    new_row_df = pd.DataFrame([row])
    updated = new_row_df if existing.empty else pd.concat([existing, new_row_df], ignore_index=True)
    _save_tracked_picks(updated)
    return True, f"Tracking {scan_row.get('company_name', code)} from {as_of.isoformat()} at close {scan_row.get('close')}."


def _evaluate_pick(pick: dict, bars_since_entry: pd.DataFrame, horizon_days: int) -> dict:
    """Pure function: given a pick and the daily bars strictly *after* the
    day it was logged (ascending by date), decide whether/when it hit its
    target or stop, or ran out the holding horizon untouched.

    Uses each day's high/low against target/stop rather than just the
    latest close, since a stock can spike through a level intraday and
    close back inside it - checking only the close would silently miss
    that. If both target and stop are breached on the same day, this
    conservatively assumes the stop was hit first: daily bars can't tell
    us the intraday sequence, and a risk-management tool should not assume
    the better outcome when it can't actually verify it.
    """
    stop = pick.get("stop_loss")
    target = pick.get("target")
    entry_price = pick.get("entry_price")

    status = OPEN_STATUS
    exit_date = None
    exit_price = None

    for _, bar in bars_since_entry.iterrows():
        hit_stop = stop is not None and stop == stop and bar["low"] <= stop
        hit_target = target is not None and target == target and bar["high"] >= target
        if hit_stop:
            status, exit_price, exit_date = "HIT_STOPLOSS", stop, bar["date"]
            break
        if hit_target:
            status, exit_price, exit_date = "HIT_TARGET", target, bar["date"]
            break

    if status == OPEN_STATUS and len(bars_since_entry) >= horizon_days:
        horizon_bar = bars_since_entry.iloc[horizon_days - 1]
        status, exit_price, exit_date = "EXPIRED", horizon_bar["close"], horizon_bar["date"]

    days_held = len(bars_since_entry) if status == OPEN_STATUS else int((bars_since_entry["date"] <= exit_date).sum())

    result = {
        "status": status,
        "days_held": days_held,
        "last_checked_date": (bars_since_entry["date"].max() if len(bars_since_entry) else pick.get("last_checked_date")),
        "last_checked_price": (bars_since_entry["close"].iloc[-1] if len(bars_since_entry) else pick.get("last_checked_price")),
    }

    if status == OPEN_STATUS:
        result["exit_date"] = None
        result["exit_price"] = None
        result["realized_pct"] = None
        current_price = result["last_checked_price"]
        result["unrealized_pct"] = (
            round((current_price - entry_price) / entry_price * 100, 2)
            if entry_price and current_price == current_price
            else None
        )
    else:
        result["exit_date"] = exit_date.date().isoformat() if hasattr(exit_date, "date") else exit_date
        result["exit_price"] = round(float(exit_price), 2)
        result["realized_pct"] = round((exit_price - entry_price) / entry_price * 100, 2) if entry_price else None
        result["unrealized_pct"] = result["realized_pct"]

    return result


def _signal_health_notes(pick: dict, latest_snap: dict) -> list[str]:
    """Rule-based, plain-English flags for whether the underlying setup is
    deteriorating or improving even though price hasn't hit target/stop yet
    - the "did anything real actually change in this scrip" signal, as
    opposed to just watching the price level."""
    notes = []

    entry_st_bullish = pick.get("entry_supertrend_bullish")
    latest_st_bullish = latest_snap.get("supertrend_bullish")
    if entry_st_bullish and latest_st_bullish is False:
        notes.append("Supertrend has flipped bearish since entry - the trend that triggered this pick has reversed.")

    entry_rsi = pick.get("entry_rsi")
    latest_rsi = latest_snap.get("rsi")
    if entry_rsi is not None and latest_rsi is not None and entry_rsi == entry_rsi and latest_rsi == latest_rsi:
        if entry_rsi >= 55 and latest_rsi < 45:
            notes.append(f"RSI has faded from {entry_rsi:.0f} to {latest_rsi:.0f} - momentum has cooled off materially.")

    entry_above_50 = pick.get("entry_price_above_50")
    latest_above_50 = latest_snap.get("price_above_50")
    if entry_above_50 and latest_above_50 is False:
        notes.append("Price has closed back below its 50DMA - the trend structure that supported this pick is broken.")

    if not notes:
        notes.append("No material deterioration flagged - setup still looks structurally intact.")
    return notes


def update_tracked_picks(kite, config: dict, fetch_bars) -> pd.DataFrame:
    """Re-check every OPEN tracked pick against real market data since it
    was logged: fetch bars, recompute indicators, decide target/stop/expiry,
    and flag any real deterioration in the setup.

    `fetch_bars(kite, instrument_token, history_days) -> pd.DataFrame` is
    injected so this function has no direct Kite/network dependency and can
    be unit tested with a stub.
    """
    picks = load_tracked_picks()
    if picks.empty:
        return picks

    # A freshly-logged pick has every one of these fields as a scalar (a
    # date string, a float, empty text), so pandas infers a narrow dtype per
    # column (often float64, since NaN is the only value seen so far). This
    # loop later assigns mixed types (status strings, ISO dates, notes) into
    # the same cells - cast to object first so those assignments don't warn
    # (and eventually error) about incompatible dtypes.
    mutable_cols = [
        "status", "exit_date", "exit_price", "realized_pct", "last_checked_date",
        "last_checked_price", "unrealized_pct", "days_held", "signal_notes",
    ]
    picks[mutable_cols] = picks[mutable_cols].astype(object)

    horizon_days = config["conviction"]["target_horizon_days"][1]
    open_mask = picks["status"] == OPEN_STATUS

    for idx in picks[open_mask].index:
        pick = picks.loc[idx].to_dict()
        date_logged = datetime.fromisoformat(pick["date_logged"]).date()
        history_days = (date.today() - date_logged).days + config["data"]["history_days"]

        try:
            bars = fetch_bars(kite, int(pick["instrument_token"]), history_days)
        except Exception as exc:
            picks.loc[idx, "signal_notes"] = f"Update failed: {exc}"
            continue

        bars_since_entry = bars[pd.to_datetime(bars["date"]).dt.date > date_logged].reset_index(drop=True)
        if bars_since_entry.empty:
            continue

        evaluation = _evaluate_pick(pick, bars_since_entry, horizon_days)
        for key, value in evaluation.items():
            picks.loc[idx, key] = value

        enriched = compute_indicators(bars, config)
        latest_snap = latest_snapshot(enriched)
        notes = _signal_health_notes(pick, latest_snap)
        picks.loc[idx, "signal_notes"] = " | ".join(notes)

    _save_tracked_picks(picks)
    return picks
