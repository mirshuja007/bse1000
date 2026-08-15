"""Load the BSE 1000 constituent universe from the CSV export.

Note: the `Constituents` (company name) column in the source file is
truncated to ~30 characters by BSE's own export (e.g. "Aditya Birla Fashion and
Retai..."). It is kept only as a human-readable label and a sanity-check
signal - it is NOT used as the primary key for instrument resolution.
The `Symbol` column (the numeric BSE scrip code) is the reliable key and is
matched exactly against Kite's BSE instrument dump.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT

DEFAULT_UNIVERSE_FILE = REPO_ROOT / "data" / "bse_1000_constituents.csv"

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


def normalize_name(name: str) -> str:
    """Uppercase, strip punctuation and common corporate suffixes, so
    truncated / differently-formatted company names can still be compared."""
    cleaned = re.sub(r"[^A-Za-z0-9& ]", " ", name).upper()
    tokens = [t for t in cleaned.split() if t and t not in _SUFFIX_WORDS]
    return " ".join(tokens)


def load_universe(path: str | Path = DEFAULT_UNIVERSE_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Symbol": str})
    df = df.rename(
        columns={
            "Constituents": "company_name_raw",
            "Symbol": "bse_code",
            "Macro-Economic Sector": "sector",
        }
    )
    df["bse_code"] = df["bse_code"].str.strip()
    df["company_name_raw"] = df["company_name_raw"].str.strip()
    df["sector"] = df["sector"].fillna("Unclassified").str.strip()
    df["name_key"] = df["company_name_raw"].apply(normalize_name)
    df = df.drop_duplicates(subset="bse_code").reset_index(drop=True)
    return df
