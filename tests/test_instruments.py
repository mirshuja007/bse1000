import math

import pandas as pd

from src.instruments import _best_nse_match, _build_nse_name_index, _is_token_prefix
from src.universe import normalize_name


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
