"""Load the stock universe(s) this app can scan, from their CSV exports.

Two universes are supported, resolved very differently downstream (see
src/instruments.py):

- BSE 1000 (`load_universe`): the `Constituents` (company name) column is
  truncated to ~30 characters by BSE's own export (e.g. "Aditya Birla
  Fashion and Retai..."). It is kept only as a human-readable label and a
  sanity-check signal - it is NOT used as the primary key for instrument
  resolution. The `Symbol` column (the numeric BSE scrip code) is the
  reliable key and is matched exactly against Kite's BSE instrument dump,
  then fuzzy-matched to an NSE tradingsymbol for liquidity.
- Nifty Total Market (`load_nifty_total_market_universe`): NSE's own
  official constituent list for its broadest equity index - Nifty 500 plus
  ~250 additional smaller-cap names (754 total; verified by set comparison
  against the old Nifty 500 file when this replaced it - every one of the
  500 was a subset). Its `Symbol` column IS already the exact NSE
  tradingsymbol (e.g. "RELIANCE"), so resolution is a plain exact-match
  join against Kite's NSE instrument dump - no fuzzy name matching needed
  at all.

Both loaders normalize to the same `security_code` / `company_name_raw` /
`sector` / `universe` column shape so the rest of the app (scanner,
instrument resolution, app.py) doesn't need to care which universe a row
came from.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT

DEFAULT_UNIVERSE_FILE = REPO_ROOT / "data" / "bse_1000_constituents.csv"
DEFAULT_NIFTY_TOTAL_MARKET_UNIVERSE_FILE = REPO_ROOT / "data" / "nifty_total_market_constituents.csv"

_SUFFIX_WORDS = {
    "LIMITED",
    "LTD",
    "LTD.",
    "LIMITE",  # truncation artifact of "LIMITED"
    "PVT",
    "PRIVATE",
    "CO",
    "COMPANY",
    "INDUSTRIES",
    "INDIA",
    "AND",
    "&",
}


def _merge_initials(tokens: list[str]) -> list[str]:
    """Collapse runs of single-letter tokens into one acronym token, so
    "E.I.D.-PARRY" (which punctuation-stripping splits into "E", "I", "D",
    "PARRY") becomes ["EID", "PARRY"] instead of leaving three stray
    single-letter tokens that never match anything. "&" is excluded since
    it's punctuation, not an initial."""
    merged = []
    i = 0
    while i < len(tokens):
        if len(tokens[i]) == 1 and tokens[i] != "&":
            j = i
            acronym = ""
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j] != "&":
                acronym += tokens[j]
                j += 1
            merged.append(acronym)
            i = j
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def normalize_name(name: str) -> str:
    """Uppercase, strip punctuation and common corporate suffixes, so
    truncated / differently-formatted company names can still be compared.

    Some rows in Kite's live instrument dump (e.g. certain delisted/odd-lot
    entries) have a missing `name`, which pandas reads back as NaN (a
    float) rather than an empty string - guard against that here instead of
    crashing re.sub."""
    if not isinstance(name, str):
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9& ]", " ", name).upper()
    tokens = _merge_initials(cleaned.split())
    tokens = [t for t in tokens if t and t not in _SUFFIX_WORDS]
    return " ".join(tokens)


def load_universe(path: str | Path = DEFAULT_UNIVERSE_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Symbol": str})
    df = df.rename(
        columns={
            "Constituents": "company_name_raw",
            "Symbol": "security_code",
            "Macro-Economic Sector": "sector",
        }
    )
    df["security_code"] = df["security_code"].str.strip()
    df["company_name_raw"] = df["company_name_raw"].str.strip()
    df["sector"] = df["sector"].fillna("Unclassified").str.strip()
    df["name_key"] = df["company_name_raw"].apply(normalize_name)
    df["universe"] = "BSE1000"
    df = df.drop_duplicates(subset="security_code").reset_index(drop=True)
    return df


def load_nifty_total_market_universe(path: str | Path = DEFAULT_NIFTY_TOTAL_MARKET_UNIVERSE_FILE) -> pd.DataFrame:
    """Load NSE's official Nifty Total Market constituent list. Unlike BSE
    1000's `Symbol` (a numeric scrip code), this file's `Symbol` column is
    already the exact NSE tradingsymbol - so `security_code` here doubles
    as the tradingsymbol used for the exact-match join in
    build_nse_mapping()."""
    df = pd.read_csv(path, dtype={"Symbol": str})
    df = df.rename(
        columns={
            "Company Name": "company_name_raw",
            "Symbol": "security_code",
            "Industry": "sector",
        }
    )
    df["security_code"] = df["security_code"].str.strip()
    df["company_name_raw"] = df["company_name_raw"].str.strip()
    df["sector"] = df["sector"].fillna("Unclassified").str.strip()
    df["name_key"] = df["company_name_raw"].apply(normalize_name)
    df["universe"] = "NIFTYTOTALMKT"
    df = df.dropna(subset=["security_code"]).drop_duplicates(subset="security_code").reset_index(drop=True)
    return df
