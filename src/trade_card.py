"""Trade Card infographic: a single-stock, share-ready image (sized for
WhatsApp) laying out the conviction score, the ATR-based trade plan, and a
fixed disclaimer.

Not a new signal - every number on the card (entry, stop, target,
conviction tier/score, RSI/ADX/volume) is read straight from the same
`result_df` row the main scan already produced. This module only draws
it as an image; `generate_daily_trade_cards` picks the top N candidates
by conviction_score among those passing the active preset's filters -
the same selection rule the email report's top-N CSV attachment already
uses, so the cards and that CSV always agree on "today's top picks."

Font handling: tries common system TrueType fonts (present on most Linux
deployments, including Streamlit Community Cloud, since Pillow/matplotlib
pull them in) and falls back to Pillow's own bundled scalable default
font if none are found - this never hard-fails on a missing font file.
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CARD_WIDTH = 1080
CARD_HEIGHT = 1080

_BG = (13, 20, 33)
_PANEL = (22, 32, 51)
_GOLD = (232, 182, 76)
_WHITE = (240, 242, 246)
_GRAY = (150, 160, 176)
_FOOTER_GRAY = (90, 100, 118)
_GREEN = (62, 207, 126)
_RED = (255, 92, 92)

_FONT_PATHS = {
    False: [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    True: [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
}

DISCLAIMER = (
    "Not financial advice - educational purposes only. Rule-based, algorithmic output, not a "
    "recommendation to buy or sell. Do your own research and consult a SEBI-registered advisor. "
    "Past performance does not indicate future results."
)

TIER_COLORS = {
    "Very High Conviction": _GOLD,
    "High Conviction": _GREEN,
    "Moderate Conviction": (240, 180, 90),
    "Watchlist": _GRAY,
}

_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _FONT_CACHE:
        for path in _FONT_PATHS[bold]:
            try:
                _FONT_CACHE[key] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _FONT_CACHE[key] = ImageFont.load_default(size=size)
    return _FONT_CACHE[key]


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _num(row: pd.Series, *keys):
    """First present, non-null value among `keys` - lets the card fall
    back gracefully across slightly different column names."""
    for key in keys:
        val = row.get(key)
        if val is not None and pd.notna(val):
            return val
    return None


def render_trade_card(row: pd.Series, rank: int, preset_label: str, as_of: datetime | None = None) -> Image.Image:
    """Pure rendering function - takes one already-scored result_df row,
    draws one card, returns a PIL Image. Caller decides whether to show
    it, save it, or attach it to an email."""
    as_of = as_of or datetime.now()
    img = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), _BG)
    draw = ImageDraw.Draw(img)
    pad = 60

    draw.text((pad, 50), "INDIA MOMENTUM SCANNER", font=_font(30, bold=True), fill=_GOLD)
    date_text = as_of.strftime("%d %b %Y")
    date_w = draw.textlength(date_text, font=_font(30, bold=True))
    draw.text((CARD_WIDTH - pad - date_w, 50), date_text, font=_font(30, bold=True), fill=_WHITE)
    draw.line([(pad, 100), (CARD_WIDTH - pad, 100)], fill=_GOLD, width=2)

    rank_label = f"TOP PICK #{rank}"
    draw.text((pad, 130), rank_label, font=_font(26, bold=True), fill=_GRAY)
    preset_w = draw.textlength(preset_label, font=_font(24))
    draw.text((CARD_WIDTH - pad - preset_w, 132), preset_label, font=_font(24), fill=_GRAY)

    company = str(row.get("company_name", ""))[:32]
    draw.text((pad, 175), company, font=_font(56, bold=True), fill=_WHITE)
    sub = f"{row.get('tradingsymbol', '')} - {row.get('exchange', '')} - {row.get('sector', '')}"
    draw.text((pad, 245), sub, font=_font(28), fill=_GRAY)

    tier = str(row.get("conviction_tier", "Watchlist"))
    score = _num(row, "conviction_score")
    tier_color = TIER_COLORS.get(tier, _GRAY)
    badge_text = f"{tier.upper()} - {score:.0f}/100" if score is not None else tier.upper()
    badge_font = _font(28, bold=True)
    badge_w = draw.textlength(badge_text, font=badge_font) + 40
    draw.rounded_rectangle([(pad, 300), (pad + badge_w, 300 + 56)], radius=28, fill=tier_color)
    draw.text((pad + 20, 314), badge_text, font=badge_font, fill=_BG)

    stats_y = 410
    col_w = (CARD_WIDTH - 2 * pad) // 3
    entry_price = _num(row, "entry_price", "close")
    stop_price = _num(row, "stop_loss")
    target_price = _num(row, "target")
    risk_pct = _num(row, "risk_pct")
    reward_pct = _num(row, "reward_pct")
    stats = [
        ("ENTRY", entry_price, None, "", _WHITE),
        ("STOP LOSS", stop_price, risk_pct, "-", _RED),
        ("TARGET", target_price, reward_pct, "+", _GREEN),
    ]
    for i, (label, price, pct, sign, color) in enumerate(stats):
        x = pad + i * col_w
        draw.text((x, stats_y), label, font=_font(24, bold=True), fill=_GRAY)
        price_text = f"Rs.{price:.2f}" if price is not None else "n/a"
        draw.text((x, stats_y + 36), price_text, font=_font(40, bold=True), fill=color)
        if pct is not None:
            draw.text((x, stats_y + 88), f"{sign}{pct:.1f}%", font=_font(26), fill=color)

    if risk_pct and reward_pct:
        rr = reward_pct / risk_pct
        draw.text((pad, stats_y + 140), f"Risk : Reward  ~  1 : {rr:.1f}", font=_font(26, bold=True), fill=_GOLD)

    panel_y = 620
    panel_h = 220
    draw.rounded_rectangle([(pad, panel_y), (CARD_WIDTH - pad, panel_y + panel_h)], radius=20, fill=_PANEL)
    draw.text((pad + 30, panel_y + 24), "WHY THIS SETUP", font=_font(24, bold=True), fill=_GOLD)
    rationale = str(row.get("explanation") or row.get("notes") or "").strip()
    if not rationale:
        parts = []
        rsi = _num(row, "rsi")
        adx = _num(row, "adx")
        vol_surge = _num(row, "volume_surge")
        if rsi is not None:
            parts.append(f"RSI {rsi:.0f}")
        if adx is not None:
            parts.append(f"ADX {adx:.0f}")
        if vol_surge is not None:
            parts.append(f"Volume {vol_surge:.1f}x avg")
        rationale = " - ".join(parts) if parts else "Passed all technical filters for the active preset."
    lines = _wrap_text(draw, rationale, _font(26), CARD_WIDTH - 2 * pad - 60)[:4]
    for i, line in enumerate(lines):
        draw.text((pad + 30, panel_y + 70 + i * 34), line, font=_font(26), fill=_WHITE)

    footer_y = CARD_HEIGHT - 190
    draw.line([(pad, footer_y), (CARD_WIDTH - pad, footer_y)], fill=(60, 70, 90), width=2)
    disclaimer_lines = _wrap_text(draw, DISCLAIMER, _font(20), CARD_WIDTH - 2 * pad)
    for i, line in enumerate(disclaimer_lines):
        draw.text((pad, footer_y + 20 + i * 26), line, font=_font(20), fill=_GRAY)
    stamp = f"Generated {as_of.strftime('%d %b %Y, %H:%M')} - Not SEBI-registered advice"
    draw.text((pad, CARD_HEIGHT - 50), stamp, font=_font(20), fill=_FOOTER_GRAY)

    return img


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_daily_trade_cards(
    result_df: pd.DataFrame, preset_label: str, top_n: int = 3, as_of: datetime | None = None
) -> list[tuple[str, Image.Image]]:
    """Top N candidates by conviction_score among those passing every
    filter for the active preset. Returns [(tradingsymbol, image), ...],
    ranked best first."""
    if result_df is None or result_df.empty:
        return []
    candidates = result_df[result_df["passes_all_filters"] == True]  # noqa: E712
    top = candidates.sort_values("conviction_score", ascending=False).head(top_n)
    return [
        (str(row["tradingsymbol"]), render_trade_card(row, rank=i + 1, preset_label=preset_label, as_of=as_of))
        for i, (_, row) in enumerate(top.iterrows())
    ]
