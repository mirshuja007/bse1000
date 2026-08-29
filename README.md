# BSE 1000 Momentum Scanner

A configurable momentum/breakout scanner for the BSE 1000 universe, built on
Kite Connect. It screens for volume surges, 50/200-DMA breakouts, RSI
strength and other trend/volatility characteristics, then ranks survivors
with a transparent, rule-based **conviction score** aimed at a **1-15
trading-day** swing/momentum horizon.

**Read the "Important - please read before using" section below before
connecting your live account.**

## What this is (and isn't)

- It **is** a real, working technical screener + scoring engine: every
  formula (RSI, ATR, ADX, MACD, Donchian breakout, relative strength, volume
  surge, etc.) is implemented from first principles in `src/indicators.py`
  and unit-tested against synthetic data in `tests/`.
- The **"conviction score"** is a transparent weighted checklist (trend,
  momentum, volume, breakout quality, relative strength, sector strength),
  not a machine-learning prediction and not a guarantee of forward returns.
  Every score shows its full breakdown so you can see exactly why a stock
  ranked where it did, and every weight/threshold is adjustable from the
  sidebar.
- It has **not** been run against live market data yet — this sandbox has
  no route to Zerodha's servers. All logic was verified offline with
  synthetic OHLCV data and passing unit tests (`pytest tests/ -v` → 23/23
  pass). **You need to run it yourself, from a machine with real internet
  access, to see it operate on live data.** Treat the first live run as a
  smoke test: check `data/universe_mapping.csv` and a handful of scan
  results against a chart you trust before relying on it.
- There is **no backtest** in this version — the scoring weights are a
  sensible, standard-practice starting point (trend-following /
  Minervini-/O'Neil-style momentum criteria), not weights fitted to Indian
  market history. Backtesting is the natural next step (see Roadmap).

## Important — please read before using

**Your Kite API key/secret, account password, and TOTP secret were shared
with me in plaintext** (via the uploaded `kite_appaccess.txt`). I did **not**
commit any of them anywhere — they only ever get read from a local `.env`
file that is git-ignored (see `.gitignore`, `.env.example`). Nothing in this
repo contains your credentials; I grepped the full working tree for every
secret value before pushing and found zero matches.

That said, because those credentials passed through a chat upload, I'd
treat them as slightly exposed and recommend, at your convenience:
1. Change your Zerodha login password.
2. Regenerate your Kite Connect API secret at
   [developers.kite.trade](https://developers.kite.trade).
3. Only ever put the live values in your own local `.env` (never in a
   commit, issue, chat, or screenshot).

## Architecture

```
config/scanner_config.yaml   All screening/scoring parameters (edit or override live in the UI)
data/bse_1000_constituents.csv   BSE 1000 universe (company name, BSE scrip code, sector)
data/universe_mapping.csv    Generated: BSE code -> Kite instrument (NSE preferred), git-ignored
data/nifty_500_constituents.csv   Nifty 500 universe, NSE's own official list (company, NSE symbol, sector)
data/nifty500_mapping.csv    Generated: Nifty 500 symbol -> Kite NSE instrument, git-ignored
src/
  config.py        YAML + .env loading, no secrets in YAML ever
  auth.py           Automated Kite login (password+TOTP) with a manual-login fallback
  universe.py       Loads/normalizes both constituent lists
  instruments.py    Resolves each constituent to a tradable Kite instrument (BSE fuzzy match + Nifty 500 exact match)
  data_fetcher.py    Rate-limited historical OHLCV pulls, with local caching
  indicators.py      RSI, ATR, ADX, MACD, DMA breakouts, Donchian, OBV, relative strength...
  scanner.py         Applies configurable filters + sector-strength ranking
  conviction.py      Weighted multi-factor conviction score, transparent breakdown
app.py               Streamlit dashboard (live parameter tuning, charts, watchlist export)
run_scan.py           Headless CLI runner, e.g. for a cron job
tests/                Offline unit tests (no network required)
```

### Two universes: BSE 1000 and Nifty 500

Pick which to scan from the "Universe" section at the top of the sidebar
(BSE 1000 / Nifty 500 / Both, deduped). They resolve very differently:

- **BSE 1000** — your source file lists BSE scrip codes (e.g. `500325`).
  Most names are dual-listed and far more liquid on NSE, so
  `instruments.py`: (1) matches your BSE code exactly against Kite's BSE
  instrument dump (for BSE equities, `tradingsymbol` *is* the scrip code -
  an exact match, not a guess); (2) uses Kite's own company `name` field
  (not your CSV's `Constituents` column, which BSE truncates to ~30
  characters) to find the equivalent NSE listing, via exact then fuzzy name
  matching; (3) falls back to the BSE listing itself if no confident NSE
  match is found, and records a `match_confidence` for every row.

  **After your first run, open `data/universe_mapping.csv` and skim rows
  with `match_confidence < 0.9`** — that's the honest, known limitation of
  automated name matching across ~1000 tickers. You can hand-edit that
  file; the scanner just reads it going forward.

- **Nifty 500** — NSE's own official constituent list (`ind_nifty500list.csv`)
  already gives the exact NSE tradingsymbol per company, so
  `build_nse_mapping()` is a plain exact-match join against Kite's NSE
  instrument dump. No fuzzy matching, no confidence score, no manual review
  needed - every resolvable row is a certain match.

Selecting "Both" scans the union, deduped: a company present in both lists
that resolves to the same NSE instrument is only scored once, tagged
`BSE1000+NIFTY500` in the results table's `universe` column, so provenance
isn't lost.

## Setup

```bash
cd bse1000
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# now edit .env and fill in KITE_API_KEY, KITE_API_SECRET, KITE_USER_ID,
# KITE_PASSWORD, KITE_TOTP_SECRET (and optionally APP_PASSWORD)
```

Run the test suite (no network needed):
```bash
pytest tests/ -v
```

### First run — CLI

```bash
python run_scan.py --top 25
```
This logs in (automated TOTP flow), builds `data/universe_mapping.csv` the
first time, fetches ~400 days of daily history for every resolved stock
(rate-limited to Kite's historical-data limit — budget a few minutes for
~1000 names), scores everything, and writes `outputs/scan_<timestamp>.csv`.

### Dashboard

```bash
streamlit run app.py
```
Click **Run scan**. Use the sidebar to change any filter threshold or
conviction-score weight and re-run instantly. Click any row's ticker to see
its price/50DMA/200DMA/volume/RSI chart and full score breakdown. Export the
current view as CSV for your watchlist.

## How the conviction score works

Six weighted categories (default weights in `config/scanner_config.yaml`,
adjustable live in the sidebar):

| Category | What it measures |
|---|---|
| Trend (20%) | Price vs 50DMA/200DMA, golden-cross regime |
| Momentum (20%) | RSI zone (peaks 60-75, tapers off if extended), MACD histogram, 10-day ROC |
| Volume (15%) | Today's volume vs 20-day average, OBV trend |
| Breakout quality (20%) | Fresh Donchian N-day-high breakout, recent 50DMA breakout, volatility contraction into the move, closes near the day's high |
| Relative strength (15%) | New high vs benchmark (NIFTY 500) relative-strength line, 20-day outperformance |
| Sector strength (10%) | Percentile rank of the stock's sector by median 20-day relative return |

Two penalties pull the score down: being **>12% above the breakout pivot**
(chasing risk) and **RSI > 82** (deep overbought, mean-reversion risk) — both
thresholds adjustable.

Final tiers: **Very High Conviction (≥75)**, **High Conviction (≥60)**,
**Moderate Conviction (≥45)**, **Watchlist (<45)**.

## Telling new recommendations from repeats

The results table now shows **recommendation_status**, **first_recommended_date**,
and **entry_price** for every row, computed automatically - no manual step.
Every stock that passes all filters on any scan is silently logged to
`data/recommendation_history.csv` (git-ignored). Next time it resurfaces:

- **New** - first time this stock has ever passed all filters.
- **Repeat - still long** - seen before, and every signal that qualified it
  then still holds now (still passes filters, Supertrend still bullish,
  price still above its 50DMA). `entry_price` stays the *original*
  first-seen price, not today's close, so you can see how far it's run.
- **Repeat - exit / trail SL** - seen before, but at least one of those
  signals has flipped since (Supertrend turned bearish, price closed back
  below the 50DMA, or it no longer passes the filters that qualified it) -
  the setup that got it recommended has changed, even though this is
  presented in the same conviction-score list.

This is separate from Tracked Picks below: this history is automatic and
covers every candidate the scanner ever surfaces; Tracked Picks is the
opt-in record of positions you've deliberately chosen to follow for P&L.
See `src/recommendation_log.py`.

## Tracking past recommendations

**Done** (was on the roadmap as forward-return tracking): click **📌 Track
this pick** in the Stock Detail panel to log a candidate's entry price,
stop, target, and entry-time signals to `data/tracked_picks.csv`
(git-ignored - it's your personal record, not sample data). The **Tracked
picks** section at the bottom of the app shows every logged pick and a
**🔄 Update tracked positions** button that:

- fetches real daily bars since each open pick was logged and checks each
  day's high/low against target and stop (not just today's close, so an
  intraday spike through a level isn't missed);
- marks a pick `HIT_TARGET`, `HIT_STOPLOSS`, or `EXPIRED` (ran out the
  15-day horizon untouched) accordingly, with the actual exit price/date
  and realized return;
- for picks still `OPEN`, flags real deterioration in the setup itself -
  Supertrend flipping bearish, RSI fading hard, price closing back below
  its 50DMA - so you can see a weakening thesis even before price hits
  either level;
- surfaces a win rate (hit-target vs hit-stop) across everything closed so
  far, which is the actual, non-hallucinated answer to "is this scoring any
  good" - see `src/tracker.py`.

## Roadmap / suggested next steps

Ranked by what would sharpen short-term (1-15 day) alpha the most:

1. **Backtest the scoring model** against BSE 1000 history to validate/tune
   the category weights and penalty thresholds — right now they're
   well-reasoned defaults, not empirically fitted. The tracked-picks log
   above is a live, forward-looking start on this same question.
2. **Delivery % / DII-FII data overlay** (NSE bhavcopy delivery data) —
   a volume surge with high delivery % is a much stronger signal than one
   driven by intraday churn.
3. **Earnings/corporate-action calendar overlay** — flag or exclude names
   with earnings inside your 1-15 day holding window (event risk).
4. **Sector rotation view** — a dedicated screen showing which of the 12
   sectors in your CSV are currently leading/lagging, since sector strength
   already feeds the score.
5. **Alerting** — push a Slack/Telegram/email digest of new Very-High-
   Conviction names each morning (the CLI runner is cron-ready).
6. **Intraday confirmation** — an optional intraday volume-pace check
   (comparing partial-day volume to the same time on average days) so you
   don't have to wait for end-of-day data to catch a breakout starting.

## Disclaimer

This tool ranks stocks by how many well-established technical momentum
characteristics they currently show. It is a decision-support tool, not
investment advice, and carries no guarantee of future performance. Always
apply your own risk management (position sizing, stop-losses, diversification).
