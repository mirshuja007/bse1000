import math

import pandas as pd

from src.instruments import (
    _best_nse_match,
    _build_nse_name_index,
    _is_token_prefix,
    _resolve_nifty_total_market_universe,
    combine_mappings,
)
from src.universe import load_nifty_total_market_universe, normalize_name


def test_load_nifty_total_market_universe_parses_official_nse_csv():
    df = load_nifty_total_market_universe()
    assert len(df) == 754
    assert (df["universe"] == "NIFTYTOTALMKT").all()
    reliance = df[df["company_name_raw"].str.contains("Reliance Industries")]
    assert reliance.iloc[0]["security_code"] == "RELIANCE"


def test_normalize_name_handles_non_string_input():
    # Kite's live instrument dump has some rows with a missing `name`,
    # which pandas reads back as NaN (a float) rather than "".
    assert normalize_name(math.nan) == ""
    assert normalize_name(None) == ""


def test_normalize_name_strips_suffixes_and_punctuation():
    assert normalize_name("AARTI DRUGS LTD.") == "AARTI DRUGS"
    assert normalize_name("ABB INDIA LIMITED") == "ABB"


def test_normalize_name_handles_truncation_artifact():
    # BSE export truncates "...LIMITED" to "...LIMITE" past ~30 chars.
    truncated = normalize_name("ACCELYA SOLUTIONS INDIA LIMITE")
    full = normalize_name("ACCELYA SOLUTIONS INDIA LIMITED")
    assert truncated == full == "ACCELYA SOLUTIONS"


def test_normalize_name_is_order_preserving_and_uppercase():
    assert normalize_name("Aditya Birla Capital Ltd") == "ADITYA BIRLA CAPITAL"


def test_normalize_name_collapses_dotted_initials_into_acronym():
    # Real failure from a live scan: "E.I.D.-PARRY" split into three stray
    # single-letter tokens ("E", "I", "D") that never matched anything,
    # instead of the acronym "EID" a human reads it as.
    assert normalize_name("E.I.D.-PARRY (INDIA) LTD.") == "EID PARRY"
    assert normalize_name("T D POWER SYSTEMS LTD.") == "TD POWER SYSTEMS"
    assert normalize_name("J.K.CEMENT LTD.") == "JK CEMENT"


def test_normalize_name_does_not_merge_ampersand_as_an_initial():
    assert normalize_name("L.G.BALAKRISHNAN & BROS.LTD.") == "LG BALAKRISHNAN BROS"


def test_is_token_prefix_true_for_genuine_prefix():
    assert _is_token_prefix(["POWER", "GRID", "CORP"], ["POWER", "GRID", "CORP", "OF", "INDIA"])
    assert _is_token_prefix(["POWER", "GRID", "CORP", "OF", "INDIA"], ["POWER", "GRID", "CORP"])


def test_is_token_prefix_false_for_single_word_overlap():
    # A single shared word (e.g. "TATA") must not count as a prefix match -
    # that's how "TATA STEEL" would wrongly match "TATA MOTORS".
    assert not _is_token_prefix(["TATA"], ["TATA", "MOTORS"])


def test_is_token_prefix_false_for_divergent_names():
    assert not _is_token_prefix(["TATA", "STEEL"], ["TATA", "MOTORS", "LIMITED"])


def test_best_nse_match_boosts_truncated_bse_name_against_full_nse_name():
    # This is the real-world case that was silently failing: Kite's BSE
    # `name` field is truncated the same way the source CSV's is, so a
    # naive length-sensitive fuzzy ratio scores it below the confidence
    # threshold even though it's unambiguously the same company.
    bse_key = normalize_name("POWER GRID CORPORATION OF INDI")  # truncated, as Kite really returns it
    nse_df = pd.DataFrame(
        {"name_key": [normalize_name("POWER GRID CORPORATION OF INDIA LIMITED"), "SOME UNRELATED CO"]}
    )
    name_index = _build_nse_name_index(nse_df)
    idx, score = _best_nse_match(bse_key, nse_df, name_index)
    assert idx == 0
    assert score >= 0.9


def _fake_nse_dump(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tradingsymbol": symbols,
            "instrument_token": [1000 + i for i in range(len(symbols))],
        }
    )


def test_resolve_nifty_total_market_universe_exact_matches_by_symbol():
    # Nifty Total Market's Symbol column IS the NSE tradingsymbol already -
    # a plain exact-match join, unlike BSE 1000's fuzzy name matching.
    universe = pd.DataFrame(
        {
            "company_name_raw": ["Reliance Industries Ltd.", "Totally Unlisted Ltd."],
            "security_code": ["RELIANCE", "NOTREAL"],
            "sector": ["Oil Gas & Consumable Fuels", "Unclassified"],
        }
    )
    nse_df = _fake_nse_dump(["RELIANCE", "TCS"])
    mapping = _resolve_nifty_total_market_universe(universe, nse_df)

    reliance = mapping.set_index("security_code").loc["RELIANCE"]
    assert reliance["resolved"] == True  # noqa: E712
    assert reliance["exchange"] == "NSE"
    assert reliance["tradingsymbol"] == "RELIANCE"
    assert reliance["match_confidence"] == 1.0

    unlisted = mapping.set_index("security_code").loc["NOTREAL"]
    assert unlisted["resolved"] == False  # noqa: E712
    assert unlisted["reason"] == "symbol_not_found_in_kite_nse_dump"


def test_combine_mappings_merges_universe_tags_for_the_same_instrument():
    # A company in both BSE 1000 and Nifty Total Market that resolves to
    # the same NSE instrument from each should appear once, tagged with both.
    bse_mapping = pd.DataFrame(
        [
            {
                "company_name_raw": "Reliance Industries Ltd", "security_code": "500325",
                "sector": "Energy", "universe": "BSE1000", "resolved": True,
                "reason": "matched_nse_by_name", "exchange": "NSE", "tradingsymbol": "RELIANCE",
                "instrument_token": 1001, "bse_instrument_token": 999, "match_confidence": 0.98,
            },
            {
                "company_name_raw": "Some Bse Only Co", "security_code": "500001",
                "sector": "Misc", "universe": "BSE1000", "resolved": False,
                "reason": "security_code_not_found_in_kite_dump", "exchange": None,
                "tradingsymbol": None, "instrument_token": None, "match_confidence": 0.0,
            },
        ]
    )
    nifty_mapping = pd.DataFrame(
        [
            {
                "company_name_raw": "Reliance Industries Ltd.", "security_code": "RELIANCE",
                "sector": "Oil Gas & Consumable Fuels", "universe": "NIFTYTOTALMKT", "resolved": True,
                "reason": "exact_nse_symbol_match", "exchange": "NSE", "tradingsymbol": "RELIANCE",
                "instrument_token": 1001, "bse_instrument_token": None, "match_confidence": 1.0,
            },
            {
                "company_name_raw": "Nifty Only Co Ltd.", "security_code": "NIFTYONLY",
                "sector": "Services", "universe": "NIFTYTOTALMKT", "resolved": False,
                "reason": "symbol_not_found_in_kite_nse_dump", "exchange": None,
                "tradingsymbol": None, "instrument_token": None, "match_confidence": 0.0,
            },
        ]
    )

    combined = combine_mappings(bse_mapping, nifty_mapping)

    # Only one row for the shared NSE instrument, tagged with both universes.
    reliance_rows = combined[combined["instrument_token"] == 1001]
    assert len(reliance_rows) == 1
    assert reliance_rows.iloc[0]["universe"] == "BSE1000+NIFTYTOTALMKT"

    # Distinct unresolved rows from each universe are kept separate, not
    # merged into each other just because they share an empty tradingsymbol.
    unresolved = combined[combined["resolved"] == False]  # noqa: E712
    assert len(unresolved) == 2
    assert set(unresolved["universe"]) == {"BSE1000", "NIFTYTOTALMKT"}
    assert len(combined) == 3


def test_combine_mappings_passes_through_a_single_universe_unchanged():
    mapping = pd.DataFrame(
        [
            {
                "company_name_raw": "A Ltd", "security_code": "A", "sector": "X",
                "universe": "NIFTYTOTALMKT", "resolved": True, "reason": "exact_nse_symbol_match",
                "exchange": "NSE", "tradingsymbol": "A", "instrument_token": 1, "match_confidence": 1.0,
            },
        ]
    )
    combined = combine_mappings(mapping)
    assert len(combined) == 1
    assert combined.iloc[0]["universe"] == "NIFTYTOTALMKT"
