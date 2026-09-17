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

Per the source material's own stated principle: "structural tailwinds
qualify the sector; stock selection still depends on growth, PEG and risk
filters" - this theme tag is a QUALIFYING filter, never a replacement for
the Growth & Quality / Fundamentals screens. Enabling a theme restriction
narrows the candidate pool; it doesn't relax any other check.

The 8 sectors below (matching "8 sectors & 20 sub-sectors" from the
Portfolio Metrics slide) with their known sub-themes, for reference when
tagging stocks - the sub-theme detail below is NOT a separate filterable
column yet, just documentation of the taxonomy this list was built from:

- Data Centre / AI / Semiconductors: data centres, AI, semiconductors,
  cloud, cybersecurity
- IT Enabled Services & Telecom: e-commerce, drones/eVTOLs, streaming,
  digital ads, optical fiber
- Healthcare: CDMO, obesity, antibiotics, cancer, biotechnology, robotic
  surgery
- Digital Financial Services: exchanges, asset management, stock broking
  & allied, digital lending/payments/wealth platforms
- Green Energy: solar, transformers/T&D, smart meters, wind, nuclear, EV,
  copper, aluminium
- New-Age Consumer: value fashion retail, tech-enabled omnichannel
  eyewear, digital beauty/lifestyle commerce, quick commerce
- Electronic Manufacturing Services (EMS): PCB, mobile manufacturing, RFID
- Defence Industry: anti-drone, space
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
    "New-Age Consumer",
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
