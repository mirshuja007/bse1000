"""Resolve constituents of the supported universes (BSE 1000, Nifty 500) to
tradable Kite instruments.

BSE 1000 strategy (`build_universe_mapping`)
---------------------------------------------
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

Nifty 500 strategy (`build_nse_mapping`)
-----------------------------------------
NSE's own official constituent list already gives the exact NSE
tradingsymbol per company - no fuzzy name matching is needed at all, just
an exact-match join against `kite.instruments("NSE")`. Emits
`data/nifty500_mapping.csv` (git-ignored), same column shape as the BSE
mapping so the two can be concatenated by `combine_mappings`.
"""
from __future__ import annotations

import difflib
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT
from src.universe import load_nifty500_universe, load_universe, normalize_name

INSTRUMENTS_CACHE_DIR = REPO_ROOT / "data"
MAPPING_FILE = REPO_ROOT / "data" / "universe_mapping.csv"
NSE_MAPPING_FILE = REPO_ROOT / "data" / "nifty500_mapping.csv"
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


def _is_token_prefix(tokens_a: list[str], tokens_b: list[str]) -> bool:
    """True if one token list is a non-empty prefix of the other (at least
    2 tokens of overlap, so a single common word like "TATA" alone doesn't
    count). This is exactly the pattern BSE's ~30-character name truncation
    produces against an NSE dump's full, untruncated name."""
    shorter, longer = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    if len(shorter) < 2:
        return False
    return longer[: len(shorter)] == shorter


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

    tokens = name_key.split(" ")
    best_idx, best_score = None, 0.0
    for idx in candidates:
        candidate_key = nse_df["name_key"].iloc[idx]
        score = difflib.SequenceMatcher(None, name_key, candidate_key).ratio()
        if _is_token_prefix(tokens, candidate_key.split(" ")):
            # Truncation match: one name is a clean prefix of the other.
            # Treat as high-confidence regardless of the length-penalized
            # fuzzy ratio, but keep it a hair below a true exact match.
            score = max(score, 0.95)
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
            "security_code": u["security_code"],
            "sector": u["sector"],
            "universe": "BSE1000",
        }

        if u["security_code"] not in bse_by_symbol.index:
            row.update(
                {
                    "resolved": False,
                    "reason": "security_code_not_found_in_kite_dump",
                    "exchange": None,
                    "tradingsymbol": None,
                    "instrument_token": None,
                    "match_confidence": 0.0,
                }
            )
            rows.append(row)
            continue

        bse_row = bse_by_symbol.loc[u["security_code"]]
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
    return pd.read_csv(MAPPING_FILE, dtype={"security_code": str})


def _resolve_nifty500_universe(universe: pd.DataFrame, nse_df: pd.DataFrame) -> pd.DataFrame:
    """Pure exact-match join of the Nifty 500 universe against an already-
    loaded NSE instrument dump - no Kite calls, no file I/O, so it's cheap
    to unit test directly with synthetic DataFrames."""
    nse_by_symbol = nse_df.set_index("tradingsymbol", drop=False)

    rows = []
    for _, u in universe.iterrows():
        row = {
            "company_name_raw": u["company_name_raw"],
            "security_code": u["security_code"],
            "sector": u["sector"],
            "universe": "NIFTY500",
        }

        if u["security_code"] not in nse_by_symbol.index:
            row.update(
                {
                    "resolved": False,
                    "reason": "symbol_not_found_in_kite_nse_dump",
                    "exchange": None,
                    "tradingsymbol": None,
                    "instrument_token": None,
                    "match_confidence": 0.0,
                }
            )
            rows.append(row)
            continue

        nse_row = nse_by_symbol.loc[u["security_code"]]
        if isinstance(nse_row, pd.DataFrame):  # duplicate tradingsymbols, shouldn't happen normally
            nse_row = nse_row.iloc[0]

        row.update(
            {
                "resolved": True,
                "reason": "exact_nse_symbol_match",
                "exchange": "NSE",
                "tradingsymbol": nse_row["tradingsymbol"],
                "instrument_token": int(nse_row["instrument_token"]),
                "bse_instrument_token": None,
                "match_confidence": 1.0,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_nse_mapping(
    kite,
    universe_path: str | Path | None = None,
    force_refresh_instruments: bool = False,
) -> pd.DataFrame:
    """Resolve the Nifty 500 universe to Kite NSE instruments. Nifty 500's
    `security_code` is already the exact NSE tradingsymbol, so this is a
    plain exact-match join - no fuzzy matching, no confidence score, no
    BSE-vs-NSE exchange choice."""
    universe = load_nifty500_universe(universe_path) if universe_path else load_nifty500_universe()
    nse_df = _cached_instruments(kite, "NSE", force_refresh_instruments)

    mapping = _resolve_nifty500_universe(universe, nse_df)
    NSE_MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(NSE_MAPPING_FILE, index=False)
    return mapping


def load_nse_mapping(refresh_with_kite=None) -> pd.DataFrame:
    """Load the cached Nifty 500 mapping file, building it first if it
    doesn't exist and a `kite` client was supplied."""
    if not NSE_MAPPING_FILE.exists():
        if refresh_with_kite is None:
            raise FileNotFoundError(
                f"{NSE_MAPPING_FILE} does not exist yet. Run "
                "`build_nse_mapping(kite)` once (see README) to generate it."
            )
        return build_nse_mapping(refresh_with_kite)
    return pd.read_csv(NSE_MAPPING_FILE, dtype={"security_code": str})


def combine_mappings(*mappings: pd.DataFrame) -> pd.DataFrame:
    """Concatenate mappings from multiple universes into one scan set.

    A company listed in both BSE 1000 and Nifty 500 will usually resolve to
    the same NSE (tradingsymbol, exchange) pair from both sources - scanning
    it twice would double its weight in sector-strength stats and clutter
    the results table with a duplicate row. Dedupe on that pair, keeping the
    first occurrence's identity but merging the `universe` tags (e.g.
    "BSE1000+NIFTY500") so provenance isn't lost."""
    non_empty = [m for m in mappings if m is not None and not m.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True)

    is_resolved = combined["resolved"] == True  # noqa: E712
    # Group key for merging universe tags: resolved rows group by the actual
    # tradable instrument (so the same NSE symbol reached via both universes
    # merges into one row); each unresolved row is its own group (nothing to
    # merge - there's no shared instrument to key off of) so it isn't
    # accidentally lumped in with unrelated unresolved rows from the other
    # universe.
    group_key = combined["tradingsymbol"].fillna("") + "@" + combined["exchange"].fillna("")
    group_key = group_key.where(is_resolved, "unresolved_" + combined.index.astype(str))

    combined["universe"] = combined.groupby(group_key)["universe"].transform(
        lambda tags: "+".join(dict.fromkeys(tags))
    )
    combined = combined.loc[~group_key.duplicated()].reset_index(drop=True)
    return combined
