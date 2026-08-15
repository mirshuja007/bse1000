#!/usr/bin/env python3
"""One-off diagnostic: inspect what Kite's real BSE/NSE instrument dumps
look like, so src/instruments.py's matching logic can be corrected against
real data instead of assumptions. Reuses today's cached login session (the
one the Streamlit app already created) - no need to log in again.

Run from the project root:  python debug_instruments.py
"""
from src.auth import load_cached_access_token, AuthError, get_authenticated_kite
from src.config import KiteCredentials
from src.universe import load_universe
from kiteconnect import KiteConnect

creds = KiteCredentials()
cached_token = load_cached_access_token()
if cached_token and creds.api_key:
    kite = KiteConnect(api_key=creds.api_key)
    kite.set_access_token(cached_token)
    print("Using cached session token from today's login.\n")
else:
    kite = get_authenticated_kite()

print("Fetching kite.instruments('BSE') ...")
bse = kite.instruments("BSE")
print(f"Total BSE instrument rows: {len(bse)}")

instrument_types = {}
for row in bse:
    instrument_types[row["instrument_type"]] = instrument_types.get(row["instrument_type"], 0) + 1
print(f"instrument_type counts: {instrument_types}")

bse_eq = [row for row in bse if row["instrument_type"] == "EQ"]
print(f"Rows with instrument_type == 'EQ': {len(bse_eq)}\n")

print("First 8 raw EQ rows (tradingsymbol / name / segment / exchange):")
for row in bse_eq[:8]:
    print(f"  tradingsymbol={row['tradingsymbol']!r:15} name={row['name']!r:35} segment={row.get('segment')!r} exchange={row.get('exchange')!r}")

print()
universe = load_universe()
sample_codes = universe["bse_code"].head(10).tolist()
print(f"First 10 BSE codes from your CSV: {sample_codes}\n")

by_symbol = {row["tradingsymbol"]: row for row in bse_eq}
print("Direct lookup of those codes against tradingsymbol:")
hits = 0
for code in sample_codes:
    match = by_symbol.get(code)
    if match:
        hits += 1
        print(f"  {code} -> MATCH: name={match['name']!r}")
    else:
        print(f"  {code} -> no match")
print(f"\n{hits}/10 matched directly.\n")

print("Searching by company name instead, for a few well-known ones:")
for needle in ["RELIANCE", "HDFC BANK", "ICICI BANK", "TATA STEEL"]:
    found = [row for row in bse_eq if needle in row["name"].upper()][:3]
    print(f"  '{needle}':")
    for row in found:
        print(f"      tradingsymbol={row['tradingsymbol']!r} name={row['name']!r} instrument_type={row['instrument_type']!r}")
    if not found:
        print("      (no rows found containing this name)")

print("\nDone. Paste this whole output back so the matching logic can be fixed.")
