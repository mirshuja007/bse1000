"""Resolve BSE 1000 constituents to tradable Kite instruments.

Strategy
--------
1. Pull Kite's official instrument dumps for the BSE and NSE segments
   (`kite.instruments("BSE")` / `("NSE")`), equity only. These are cached
   locally for `data.cache_dir`-independent reuse (they change rarely).
2. Match each constituent's BSE scrip code (`Symbol` in the source CSV)
   *exactly* against the BSE dump's `exchange_token` - for BSE equities that
   field is the scrip code (confirmed against a live instrument dump: e.g.
   RELIANCE has exchange_token "500325", HDFCBANK "500180"). Note this is
   NOT `tradingsymbol`, which on BSE is a separate mnemonic (e.g.
   "RELIANCE", "ARE&M") unrelated to the scrip code.
3. Because NSE is usually far more liquid for dual-listed names, look up the
   equivalent NSE instrument by comparing Kite's own `name` field between
   the two dumps, using normalized-string equality first and a fuzzy ratio
   as a fallback/confidence score. Kite's BSE `name` field is itself
   truncated around 30 characters for longer company names (the same BSE
   export limitation as the source CSV), so the fuzzy fallback matters even
   here - suffix stripping in `normalize_name` closes most of the gap, but
   not all of it, which is exactly why match_confidence is reported per row.
4. Emit `data/universe_mapping.csv` (git-ignored) with both tokens, the
   chosen exchange, and a match_confidence column so you can eyeball and
   correct any low-confidence rows before the scanner trusts them.
"""
from __future__ import annotations

import difflib
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT
from src.universe import load_universe, normalize_name

INSTRUMENTS_CACHE_DIR = REPO_ROOT / "data"
MAPPING_FILE = REPO_ROOT / "data" / "universe_mapping.csv"
INSTRUMENT_DUMP_TTL_SECONDS = 24 * 3600


def _cached_instruments(kite, exchange: str, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = INSTRUMENTS_CACHE_DIR / f"kite_instruments_{exchange}.csv"
    if (
        not force_refresh
        and cache_path.exists()
        and (time.time() - cache_path.stat().st_mtime) < INSTRUMENT_DUMP_TTL_SECONDS
    ):
        return pd.read_csv(cache_path, dtype={"tradingsymbol": str, "exchange_token": str})

    records = kite.instruments(exchange)
    df = pd.DataFrame.from_records(records)
    df = df[df["instrument_type"] == "EQ"].reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def _build_nse_name_index(nse_df: pd.DataFrame) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(nse_df["name_key"]):
        first_token = name.split(" ", 1)[0] if name else ""
        index[first_token].append(i)
    return index


def _best_nse_match(name_key: str, nse_df: pd.DataFrame, name_index: dict[str, list[int]]):
    if not name_key:
        return None, 0.0

    exact = nse_df.index[nse_df["name_key"] == name_key]
    if len(exact):
        return exact[0], 1.0

    first_token = name_key.split(" ", 1)[0]
    candidates = name_index.get(first_token, [])
    if not candidates:
        return None, 0.0

    best_idx, best_score = None, 0.0
    for idx in candidates:
        score = difflib.SequenceMatcher(None, name_key, nse_df["name_key"].iloc[idx]).ratio()
        if score > best_score:
            best_idx, best_score = idx, score
    return best_idx, best_score


def build_universe_mapping(
    kite,
    universe_path: str | Path | None = None,
    min_fuzzy_confidence: float = 0.82,
    force_refresh_instruments: bool = False,
) -> pd.DataFrame:
    universe = load_universe(universe_path) if universe_path else load_universe()

    bse_df = _cached_instruments(kite, "BSE", force_refresh_instruments)
    nse_df = _cached_instruments(kite, "NSE", force_refresh_instruments)
    bse_df["exchange_token"] = bse_df["exchange_token"].astype(str)
    bse_df["name_key"] = bse_df["name"].apply(normalize_name)
    nse_df["name_key"] = nse_df["name"].apply(normalize_name)
    nse_name_index = _build_nse_name_index(nse_df)

    bse_by_symbol = bse_df.set_index("exchange_token", drop=False)

    rows = []
    for _, u in universe.iterrows():
        row = {
            "company_name_raw": u["company_name_raw"],
            "bse_code": u["bse_code"],
            "sector": u["sector"],
        }

        if u["bse_code"] not in bse_by_symbol.index:
            row.update(
                {
                    "resolved": False,
                    "reason": "bse_code_not_found_in_kite_dump",
                    "exchange": None,
                    "tradingsymbol": None,
                    "instrument_token": None,
                    "match_confidence": 0.0,
                }
            )
            rows.append(row)
            continue

        bse_row = bse_by_symbol.loc[u["bse_code"]]
        if isinstance(bse_row, pd.DataFrame):  # duplicate scrip codes, shouldn't happen normally
            bse_row = bse_row.iloc[0]

        nse_idx, score = _best_nse_match(bse_row["name_key"], nse_df, nse_name_index)

        if nse_idx is not None and score >= min_fuzzy_confidence:
            nse_row = nse_df.iloc[nse_idx]
            row.update(
                {
                    "resolved": True,
                    "reason": "matched_nse_by_name",
                    "exchange": "NSE",
                    "tradingsymbol": nse_row["tradingsymbol"],
                    "instrument_token": int(nse_row["instrument_token"]),
                    "bse_instrument_token": int(bse_row["instrument_token"]),
                    "match_confidence": round(float(score), 3),
                }
            )
        else:
            # Fall back to trading the BSE line itself - still valid, just
            # typically thinner volume.
            row.update(
                {
                    "resolved": True,
                    "reason": "no_confident_nse_match_using_bse",
                    "exchange": "BSE",
                    "tradingsymbol": bse_row["tradingsymbol"],
                    "instrument_token": int(bse_row["instrument_token"]),
                    "bse_instrument_token": int(bse_row["instrument_token"]),
                    "match_confidence": round(float(score), 3),
                }
            )
        rows.append(row)

    mapping = pd.DataFrame(rows)
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(MAPPING_FILE, index=False)
    return mapping


def load_mapping(refresh_with_kite=None) -> pd.DataFrame:
    """Load the cached mapping file, building it first if it doesn't exist
    and a `kite` client was supplied."""
    if not MAPPING_FILE.exists():
        if refresh_with_kite is None:
            raise FileNotFoundError(
                f"{MAPPING_FILE} does not exist yet. Run "
                "`build_universe_mapping(kite)` once (see README) to generate it."
            )
        return build_universe_mapping(refresh_with_kite)
    return pd.read_csv(MAPPING_FILE, dtype={"bse_code": str})
