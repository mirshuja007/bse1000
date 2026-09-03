"""Offline unit tests for src/tracker.py - synthetic data only, no
network/Kite access required."""
import pandas as pd
import pytest

from src import github_store, tracker


def _bars(rows: list[dict]) -> pd.DataFrame:
    """rows: list of {date, high, low, close} - open/volume filled in."""
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(r["date"]),
                "open": r["close"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": 100_000,
            }
            for r in rows
        ]
    )


def base_pick(**overrides) -> dict:
    pick = {
        "entry_price": 100.0,
        "stop_loss": 94.0,
        "target": 112.0,
        "entry_rsi": 65.0,
        "entry_supertrend_bullish": True,
        "entry_price_above_50": True,
    }
    pick.update(overrides)
    return pick


def test_evaluate_pick_hits_target_using_intraday_high():
    pick = base_pick()
    bars = _bars(
        [
            {"date": "2024-01-02", "high": 103, "low": 99, "close": 102},
            {"date": "2024-01-03", "high": 113, "low": 108, "close": 110},  # spikes through target intraday
        ]
    )
    result = tracker._evaluate_pick(pick, bars, horizon_days=15)
    assert result["status"] == "HIT_TARGET"
    assert result["exit_price"] == 112.0
    assert result["exit_date"] == "2024-01-03"
    assert result["days_held"] == 2
    assert result["realized_pct"] == pytest.approx(12.0)


def test_evaluate_pick_hits_stoploss():
    pick = base_pick()
    bars = _bars([{"date": "2024-01-02", "high": 101, "low": 93, "close": 95}])
    result = tracker._evaluate_pick(pick, bars, horizon_days=15)
    assert result["status"] == "HIT_STOPLOSS"
    assert result["exit_price"] == 94.0
    assert result["realized_pct"] == pytest.approx(-6.0)


def test_evaluate_pick_same_day_conflict_assumes_stop_hit_first():
    # A single wild day that spans both levels - daily bars can't tell us
    # which happened first intraday, so the conservative assumption (stop
    # hit) wins rather than assuming the better outcome.
    pick = base_pick()
    bars = _bars([{"date": "2024-01-02", "high": 115, "low": 90, "close": 105}])
    result = tracker._evaluate_pick(pick, bars, horizon_days=15)
    assert result["status"] == "HIT_STOPLOSS"


def test_evaluate_pick_expires_at_horizon_when_neither_level_hit():
    pick = base_pick()
    # 15 quiet days, never touching stop (94) or target (112).
    bars = _bars([{"date": f"2024-01-{2+i:02d}", "high": 103, "low": 99, "close": 101} for i in range(15)])
    result = tracker._evaluate_pick(pick, bars, horizon_days=15)
    assert result["status"] == "EXPIRED"
    assert result["days_held"] == 15
    assert result["exit_price"] == 101.0


def test_evaluate_pick_stays_open_with_unrealized_pct():
    pick = base_pick()
    bars = _bars(
        [
            {"date": "2024-01-02", "high": 103, "low": 99, "close": 102},
            {"date": "2024-01-03", "high": 106, "low": 101, "close": 105},
        ]
    )
    result = tracker._evaluate_pick(pick, bars, horizon_days=15)
    assert result["status"] == "OPEN"
    assert result["days_held"] == 2
    assert result["unrealized_pct"] == pytest.approx(5.0)
    assert result["exit_date"] is None


def test_signal_health_flags_supertrend_flip():
    pick = base_pick(entry_supertrend_bullish=True)
    notes = tracker._signal_health_notes(pick, {"supertrend_bullish": False, "rsi": 60, "price_above_50": True})
    assert any("supertrend" in n.lower() for n in notes)


def test_signal_health_flags_rsi_fade():
    pick = base_pick(entry_rsi=65.0)
    notes = tracker._signal_health_notes(pick, {"supertrend_bullish": True, "rsi": 40.0, "price_above_50": True})
    assert any("rsi" in n.lower() for n in notes)


def test_signal_health_flags_price_below_50dma():
    pick = base_pick(entry_price_above_50=True)
    notes = tracker._signal_health_notes(pick, {"supertrend_bullish": True, "rsi": 60, "price_above_50": False})
    assert any("50dma" in n.lower() for n in notes)


def test_signal_health_reports_clean_when_nothing_deteriorated():
    pick = base_pick()
    notes = tracker._signal_health_notes(pick, {"supertrend_bullish": True, "rsi": 62, "price_above_50": True})
    assert any("no material deterioration" in n.lower() for n in notes)


def _scan_row(**overrides) -> dict:
    row = {
        "security_code": "500325",
        "tradingsymbol": "RELIANCE",
        "exchange": "NSE",
        "company_name": "Reliance Industries Ltd",
        "universe": "BSE1000",
        "instrument_token": 738561,
        "close": 100.0,
        "stop_loss": 94.0,
        "target": 112.0,
        "supertrend_stop": 96.0,
        "conviction_score": 78.0,
        "conviction_tier": "Very High Conviction",
        "rsi": 65.0,
        "supertrend_bullish": True,
        "pct_above_sma50": 5.0,
    }
    row.update(overrides)
    return row


def test_log_pick_writes_a_row_and_refuses_duplicate_while_open(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "TRACKED_PICKS_FILE", tmp_path / "tracked_picks.csv")

    logged, msg = tracker.log_pick(_scan_row())
    assert logged is True
    picks = tracker.load_tracked_picks()
    assert len(picks) == 1
    assert picks.iloc[0]["security_code"] == "500325"
    assert picks.iloc[0]["status"] == "OPEN"

    logged_again, msg2 = tracker.log_pick(_scan_row())
    assert logged_again is False
    assert "already" in msg2.lower()
    assert len(tracker.load_tracked_picks()) == 1


def test_log_pick_allows_retracking_after_previous_pick_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "TRACKED_PICKS_FILE", tmp_path / "tracked_picks.csv")

    tracker.log_pick(_scan_row())
    picks = tracker.load_tracked_picks()
    picks.loc[0, "status"] = "HIT_TARGET"
    tracker._save_tracked_picks(picks)

    logged, _ = tracker.log_pick(_scan_row())
    assert logged is True
    assert len(tracker.load_tracked_picks()) == 2


def test_save_syncs_to_github_when_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "TRACKED_PICKS_FILE", tmp_path / "tracked_picks.csv")
    monkeypatch.setattr(github_store, "is_configured", lambda: True)
    monkeypatch.setattr(github_store, "read_file", lambda path: None)  # no file on GitHub yet
    written = {}
    monkeypatch.setattr(
        github_store, "write_file", lambda path, content, message: written.update(path=path, content=content)
    )

    tracker.log_pick(_scan_row())

    assert written["path"] == tracker.GITHUB_DATA_PATH
    assert "500325" in written["content"]
    assert tracker.get_last_sync_error() is None


def test_load_reads_from_github_when_configured(monkeypatch):
    monkeypatch.setattr(github_store, "is_configured", lambda: True)
    csv_text = "pick_id,date_logged,security_code,status\np1,2024-01-10,500325,OPEN\n"
    monkeypatch.setattr(github_store, "read_file", lambda path: csv_text)

    picks = tracker.load_tracked_picks()
    assert len(picks) == 1
    assert picks.iloc[0]["security_code"] == "500325"
    assert picks.iloc[0]["status"] == "OPEN"


def test_load_returns_empty_when_github_configured_but_no_file_yet(monkeypatch):
    monkeypatch.setattr(github_store, "is_configured", lambda: True)
    monkeypatch.setattr(github_store, "read_file", lambda path: None)
    assert tracker.load_tracked_picks().empty


def test_save_records_sync_error_without_crashing(tmp_path, monkeypatch):
    local_file = tmp_path / "tracked_picks.csv"
    monkeypatch.setattr(tracker, "TRACKED_PICKS_FILE", local_file)
    monkeypatch.setattr(github_store, "is_configured", lambda: True)
    monkeypatch.setattr(github_store, "read_file", lambda path: None)  # no file on GitHub yet

    def failing_write(path, content, message):
        raise RuntimeError("bad token")

    monkeypatch.setattr(github_store, "write_file", failing_write)

    logged, _ = tracker.log_pick(_scan_row())
    assert logged is True  # local save still succeeded despite the GitHub sync failing
    assert "bad token" in tracker.get_last_sync_error()
    assert local_file.exists()
    assert "500325" in local_file.read_text()


