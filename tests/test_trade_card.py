"""Offline unit tests for src/trade_card.py - synthetic data only, no fonts
or filesystem paths asserted (those vary by machine; the fallback to
Pillow's bundled default font is what makes this portable)."""
from datetime import datetime

import pandas as pd
from PIL import Image

from src import trade_card as tc


def make_row(**overrides) -> pd.Series:
    base = {
        "company_name": "Reliance Industries",
        "tradingsymbol": "RELIANCE",
        "exchange": "NSE",
        "sector": "Energy",
        "conviction_tier": "Very High Conviction",
        "conviction_score": 82.4,
        "entry_price": 2500.0,
        "close": 2500.0,
        "stop_loss": 2420.0,
        "target": 2650.0,
        "risk_pct": 3.2,
        "reward_pct": 6.0,
        "rsi": 64.0,
        "adx": 27.0,
        "volume_surge": 2.1,
        "notes": "",
        "explanation": "",
        "passes_all_filters": True,
    }
    base.update(overrides)
    return pd.Series(base)


def test_render_trade_card_returns_correctly_sized_image():
    row = make_row()
    img = tc.render_trade_card(row, rank=1, preset_label="Swing (3-7 days)", as_of=datetime(2026, 9, 13, 9, 30))
    assert isinstance(img, Image.Image)
    assert img.size == (tc.CARD_WIDTH, tc.CARD_HEIGHT)


def test_render_trade_card_handles_missing_optional_fields():
    row = make_row(stop_loss=None, target=None, risk_pct=None, reward_pct=None, rsi=None, adx=None, volume_surge=None)
    img = tc.render_trade_card(row, rank=2, preset_label="Custom (manual)")
    assert img.size == (tc.CARD_WIDTH, tc.CARD_HEIGHT)


def test_image_to_png_bytes_round_trips():
    row = make_row()
    img = tc.render_trade_card(row, rank=1, preset_label="Positional (8-15 days)")
    data = tc.image_to_png_bytes(img)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG file signature
    reopened = Image.open(__import__("io").BytesIO(data))
    assert reopened.size == (tc.CARD_WIDTH, tc.CARD_HEIGHT)


def test_generate_daily_trade_cards_only_uses_passing_candidates_ranked_by_score():
    result_df = pd.DataFrame(
        [
            make_row(tradingsymbol="A", conviction_score=90.0, passes_all_filters=True),
            make_row(tradingsymbol="B", conviction_score=95.0, passes_all_filters=False),  # excluded
            make_row(tradingsymbol="C", conviction_score=70.0, passes_all_filters=True),
            make_row(tradingsymbol="D", conviction_score=85.0, passes_all_filters=True),
        ]
    )
    cards = tc.generate_daily_trade_cards(result_df, preset_label="Swing (3-7 days)", top_n=3)
    symbols = [sym for sym, _ in cards]
    assert symbols == ["A", "D", "C"]  # ranked by score, B excluded despite highest score


def test_generate_daily_trade_cards_respects_top_n():
    result_df = pd.DataFrame(
        [make_row(tradingsymbol=f"S{i}", conviction_score=float(i), passes_all_filters=True) for i in range(10)]
    )
    cards = tc.generate_daily_trade_cards(result_df, preset_label="Custom (manual)", top_n=3)
    assert len(cards) == 3


def test_generate_daily_trade_cards_returns_empty_for_empty_result_df():
    assert tc.generate_daily_trade_cards(pd.DataFrame(), preset_label="Custom (manual)") == []
