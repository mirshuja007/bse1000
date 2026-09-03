"""Offline unit tests for src/recommendation_log.py - synthetic data only."""
import pandas as pd

from src import github_store, recommendation_log as rl


def _result_row(**overrides) -> dict:
    row = {
        "security_code": "500325",
        "tradingsymbol": "RELIANCE",
        "company_name": "Reliance Industries Ltd",
        "universe": "BSE1000",
        "as_of": "2024-01-10",
        "close": 100.0,
        "conviction_score": 78.0,
        "conviction_tier": "Very High Conviction",
        "supertrend_bullish": True,
        "pct_above_sma50": 5.0,
        "passes_all_filters": True,
    }
    row.update(overrides)
    return row


def _result_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_annotate_marks_unseen_stock_as_new():
    result_df = _result_df(_result_row())
    annotated = rl.annotate_with_history(result_df, history=pd.DataFrame(columns=rl._COLUMNS))
    row = annotated.iloc[0]
    assert row["recommendation_status"] == rl.STATUS_NEW
    assert row["first_recommended_date"] == "2024-01-10"
    assert row["entry_price"] == 100.0


def test_update_then_annotate_recognizes_a_repeat_with_intact_signal():
    empty_history = pd.DataFrame(columns=rl._COLUMNS)
    day1 = _result_df(_result_row(as_of="2024-01-10", close=100.0))
    history_after_day1 = rl.update_recommendation_history(day1, history=empty_history)

    day2 = _result_df(_result_row(as_of="2024-01-15", close=108.0))  # still bullish, still above 50DMA
    annotated = rl.annotate_with_history(day2, history=history_after_day1)

    row = annotated.iloc[0]
    assert row["recommendation_status"] == rl.STATUS_STILL_LONG
    assert row["first_recommended_date"] == "2024-01-10"
    assert row["entry_price"] == 100.0  # entry price is the ORIGINAL first-seen price, not today's close


def test_update_then_annotate_flags_exit_when_supertrend_flips_bearish():
    empty_history = pd.DataFrame(columns=rl._COLUMNS)
    day1 = _result_df(_result_row(as_of="2024-01-10", close=100.0, supertrend_bullish=True))
    history_after_day1 = rl.update_recommendation_history(day1, history=empty_history)

    day2 = _result_df(_result_row(as_of="2024-01-15", close=90.0, supertrend_bullish=False))
    annotated = rl.annotate_with_history(day2, history=history_after_day1)

    assert annotated.iloc[0]["recommendation_status"] == rl.STATUS_EXIT


def test_update_then_annotate_flags_exit_when_price_falls_below_50dma():
    empty_history = pd.DataFrame(columns=rl._COLUMNS)
    day1 = _result_df(_result_row(as_of="2024-01-10", pct_above_sma50=5.0))
    history_after_day1 = rl.update_recommendation_history(day1, history=empty_history)

    day2 = _result_df(_result_row(as_of="2024-01-15", pct_above_sma50=-2.0))
    annotated = rl.annotate_with_history(day2, history=history_after_day1)

    assert annotated.iloc[0]["recommendation_status"] == rl.STATUS_EXIT


def test_update_then_annotate_flags_exit_when_no_longer_passes_filters():
    empty_history = pd.DataFrame(columns=rl._COLUMNS)
    day1 = _result_df(_result_row(as_of="2024-01-10", passes_all_filters=True))
    history_after_day1 = rl.update_recommendation_history(day1, history=empty_history)

    day2 = _result_df(_result_row(as_of="2024-01-15", passes_all_filters=False))
    annotated = rl.annotate_with_history(day2, history=history_after_day1)

    assert annotated.iloc[0]["recommendation_status"] == rl.STATUS_EXIT


def test_update_recommendation_history_only_logs_passing_candidates():
    empty_history = pd.DataFrame(columns=rl._COLUMNS)
    result_df = _result_df(
        _result_row(security_code="A1", passes_all_filters=True),
        _result_row(security_code="A2", passes_all_filters=False),
    )
    updated = rl.update_recommendation_history(result_df, history=empty_history)
    assert list(updated["security_code"]) == ["A1"]


def test_update_recommendation_history_bumps_times_seen_without_changing_first_seen():
    empty_history = pd.DataFrame(columns=rl._COLUMNS)
    day1 = _result_df(_result_row(as_of="2024-01-10", close=100.0))
    h1 = rl.update_recommendation_history(day1, history=empty_history)
    assert h1.iloc[0]["times_seen"] == 1

    day2 = _result_df(_result_row(as_of="2024-01-15", close=108.0))
    h2 = rl.update_recommendation_history(day2, history=h1)
    row = h2.iloc[0]
    assert row["times_seen"] == 2
    assert row["first_seen_date"] == "2024-01-10"
    assert row["first_seen_price"] == 100.0
    assert row["last_seen_date"] == "2024-01-15"
    assert row["last_seen_price"] == 108.0


def test_save_syncs_to_github_when_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "HISTORY_FILE", tmp_path / "recommendation_history.csv")
    monkeypatch.setattr(github_store, "is_configured", lambda: True)
    written = {}
    monkeypatch.setattr(
        github_store, "write_file", lambda path, content, message: written.update(path=path, content=content)
    )

    empty_history = pd.DataFrame(columns=rl._COLUMNS)
    day1 = _result_df(_result_row(as_of="2024-01-10"))
    rl.update_recommendation_history(day1, history=empty_history)

    assert written["path"] == rl.GITHUB_DATA_PATH
    assert "500325" in written["content"]
    assert rl.get_last_sync_error() is None


def test_load_reads_from_github_when_configured(monkeypatch):
    monkeypatch.setattr(github_store, "is_configured", lambda: True)
    csv_text = "security_code,tradingsymbol,first_seen_date,times_seen\n500325,RELIANCE,2024-01-10,3\n"
    monkeypatch.setattr(github_store, "read_file", lambda path: csv_text)

    history = rl.load_recommendation_history()
    assert len(history) == 1
    assert history.iloc[0]["security_code"] == "500325"
    assert history.iloc[0]["first_seen_date"] == "2024-01-10"


def test_load_returns_empty_when_github_configured_but_no_file_yet(monkeypatch):
    monkeypatch.setattr(github_store, "is_configured", lambda: True)
    monkeypatch.setattr(github_store, "read_file", lambda path: None)
    assert rl.load_recommendation_history().empty
