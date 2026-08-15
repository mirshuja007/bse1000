import math

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
