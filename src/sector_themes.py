"""Futuristic-sector theme tagging, per "Principle 5" (structural-tailwind
sectors) - a manually-curated list you and I build together over time,
NOT an automatic classifier. NSE's own sector/industry field (already in
your data) is too coarse to map cleanly onto these themes - e.g. "Power"
mixes green energy with coal-fired thermal plants, "Capital Goods" spans
defence, EMS, and general industrials - so a best-effort automatic mapping
would misclassify a meaningful number of stocks. Better to only tag what's
actually been verified.

`data/sector_theme_map.csv` is TRACKED (not git-ignored) - like the
constituent lists, it's curated reference data, not personal runtime
output. It ships with just the schema and zero rows: no stock has been
tagged to a theme yet. Tell me which stocks belong to which theme and
I'll add rows.
"""
from __future__ import annotations

import pandas as pd

from src.config import REPO_ROOT

THEME_MAP_FILE = REPO_ROOT / "data" / "sector_theme_map.csv"

THEMES = [
    "Data Centre / AI / Semiconductors",
    "IT Enabled Services & Telecom",
    "Healthcare",
    "Digital Financial Services",
    "Green Energy",
    "Electronic Manufacturing Services (EMS)",
    "Defence Industry",
]

_COLUMNS = ["security_code", "tradingsymbol", "company_name", "theme"]


def load_theme_map() -> pd.DataFrame:
    if not THEME_MAP_FILE.exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(THEME_MAP_FILE, dtype={"security_code": str})
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[_COLUMNS]


def themes_for(result_df: pd.DataFrame) -> pd.Series:
    """Returns a Series aligned to result_df's index: the theme tag for
    each row's security_code, or "Unclassified" if it hasn't been tagged
    yet - never guessed."""
    theme_map = load_theme_map()
    theme_by_code = theme_map.set_index("security_code")["theme"].to_dict() if not theme_map.empty else {}
    return result_df["security_code"].map(theme_by_code).fillna("Unclassified")
