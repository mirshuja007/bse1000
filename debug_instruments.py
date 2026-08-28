#!/usr/bin/env python3
"""One-off diagnostic: inspect what Kite's real BSE/NSE instrument dumps
look like, so src/instruments.py's matching logic can be corrected against
real data instead of assumptions. Reuses today's cached login session (the
one the Streamlit app already created) - no need to log in again.

Run from the project root:  python debug_instruments.py
"""
from src.auth import load_cached_access_token, AuthError, get_authenticated_kite, new_kite_client
from src.config import KiteCredentials
from src.universe import load_universe

creds = KiteCredentials()
cached_token = load_cached_access_token()
if cached_token and creds.api_key:
    kite = new_kite_client(creds.api_key)
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

print("First 8 raw EQ rows (all fields):")
for row in bse_eq[:8]:
    print(f"  {row}")

print()
universe = load_universe()
sample_codes = universe["security_code"].head(10).tolist()
print(f"First 10 BSE codes from your CSV: {sample_codes}\n")

by_symbol = {row["tradingsymbol"]: row for row in bse_eq}
print("Direct lookup of those codes against tradingsymbol (expected to fail, already confirmed):")
hits = 0
for code in sample_codes:
    match = by_symbol.get(code)
    if match:
        hits += 1
        print(f"  {code} -> MATCH: name={match['name']!r}")
    else:
        print(f"  {code} -> no match")
print(f"\n{hits}/10 matched directly.\n")

print("Testing hypothesis: exchange_token == BSE scrip code, for those same 10 codes:")
by_exchange_token = {str(row["exchange_token"]): row for row in bse_eq}
hits2 = 0
for code in sample_codes:
    match = by_exchange_token.get(str(code))
    if match:
        hits2 += 1
        print(f"  {code} -> MATCH via exchange_token: tradingsymbol={match['tradingsymbol']!r} name={match['name']!r}")
    else:
        print(f"  {code} -> no match via exchange_token")
print(f"\n{hits2}/10 matched via exchange_token.\n")

print("Searching by company name instead, for a few well-known ones (showing exchange_token too):")
for needle in ["RELIANCE", "HDFC BANK", "ICICI BANK", "TATA STEEL"]:
    found = [row for row in bse_eq if needle in row["name"].upper()][:3]
    print(f"  '{needle}':")
    for row in found:
        print(f"      tradingsymbol={row['tradingsymbol']!r} exchange_token={row['exchange_token']!r} instrument_token={row['instrument_token']!r} name={row['name']!r}")
    if not found:
        print("      (no rows found containing this name)")

print("\nDone. Paste this whole output back so the matching logic can be fixed.")
