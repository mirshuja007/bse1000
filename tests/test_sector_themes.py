"""Offline unit tests for src/sector_themes.py - synthetic data, no
network access."""
import pandas as pd

from src import sector_themes as st


def test_load_theme_map_returns_empty_schema_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "THEME_MAP_FILE", tmp_path / "does_not_exist.csv")
    df = st.load_theme_map()
    assert df.empty
    assert list(df.columns) == st._COLUMNS


def test_themes_for_tags_known_stocks_and_flags_unclassified(tmp_path, monkeypatch):
    theme_file = tmp_path / "sector_theme_map.csv"
    theme_file.write_text(
        "security_code,tradingsymbol,company_name,theme\n"
        "500325,RELIANCE,Reliance Industries Ltd,Green Energy\n"
        "532540,TCS,Tata Consultancy Services Ltd,IT Enabled Services & Telecom\n"
    )
    monkeypatch.setattr(st, "THEME_MAP_FILE", theme_file)

    result_df = pd.DataFrame({"security_code": ["500325", "532540", "999999"]})
    tags = st.themes_for(result_df)

    assert tags.iloc[0] == "Green Energy"
    assert tags.iloc[1] == "IT Enabled Services & Telecom"
    assert tags.iloc[2] == "Unclassified"


def test_themes_for_all_unclassified_when_map_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "THEME_MAP_FILE", tmp_path / "empty.csv")
    result_df = pd.DataFrame({"security_code": ["A1", "B2"]})
    tags = st.themes_for(result_df)
    assert list(tags) == ["Unclassified", "Unclassified"]


def test_themes_constant_matches_the_seven_principle_5_categories():
    assert st.THEMES == [
        "Data Centre / AI / Semiconductors",
        "IT Enabled Services & Telecom",
        "Healthcare",
        "Digital Financial Services",
        "Green Energy",
        "Electronic Manufacturing Services (EMS)",
        "Defence Industry",
    ]
