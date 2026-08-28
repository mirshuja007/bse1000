#!/usr/bin/env python3
"""Headless CLI scan runner - useful for cron jobs / scheduled runs.

Usage:
    python run_scan.py                     # full scan, default config
    python run_scan.py --config path.yaml  # custom config
    python run_scan.py --candidates-only    # only rows passing all filters

Writes a timestamped CSV to outputs/ and prints the top conviction names.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.auth import get_authenticated_kite
from src.config import REPO_ROOT, load_config
from src.data_fetcher import fetch_benchmark_history, fetch_universe_history, resolve_benchmark_token
from src.instruments import build_nse_mapping, build_universe_mapping, combine_mappings, load_mapping, load_nse_mapping
from src.scanner import run_scan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--candidates-only", action="store_true")
    parser.add_argument("--refresh-mapping", action="store_true")
    parser.add_argument(
        "--universe",
        choices=["bse1000", "nifty500", "both"],
        default="bse1000",
        help="Which constituent universe(s) to scan (default: bse1000).",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()

    print("Logging in to Kite...")
    kite = get_authenticated_kite()

    print("Resolving universe -> Kite instruments...")
    mappings = []
    if args.universe in ("bse1000", "both"):
        mappings.append(
            build_universe_mapping(kite, force_refresh_instruments=True)
            if args.refresh_mapping
            else load_mapping(refresh_with_kite=kite)
        )
    if args.universe in ("nifty500", "both"):
        mappings.append(
            build_nse_mapping(kite, force_refresh_instruments=True)
            if args.refresh_mapping
            else load_nse_mapping(refresh_with_kite=kite)
        )
    mapping = combine_mappings(*mappings)
    resolved = mapping[mapping["resolved"] == True]  # noqa: E712
    print(f"  {len(resolved)}/{len(mapping)} constituents resolved to a tradable instrument.")
    low_conf = resolved[resolved["match_confidence"] < 0.9]
    if len(low_conf):
        print(f"  {len(low_conf)} rows matched with confidence < 0.9 - review data/universe_mapping.csv")

    print("Fetching benchmark history...")
    bench_token = resolve_benchmark_token(kite)
    benchmark = fetch_benchmark_history(kite, bench_token, config["data"]["history_days"])

    print(f"Fetching OHLCV history for {len(resolved)} stocks (rate-limited, this takes a few minutes)...")

    def progress(done, total):
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total}")

    results = fetch_universe_history(
        kite,
        resolved,
        history_days=config["data"]["history_days"],
        max_requests_per_second=config["data"]["max_requests_per_second"],
        cache_ttl_hours=config["data"]["cache_ttl_hours"],
        progress_callback=progress,
    )

    history = {r.security_code: r.bars for r in results if r.ok}
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"  {len(failed)} stocks failed to fetch (see logs) - continuing with the rest.")

    print("Scoring...")
    result_df, _, skipped = run_scan(history, resolved, config, benchmark)
    if skipped:
        print(f"  {len(skipped)} stocks skipped for insufficient history.")

    if args.candidates_only:
        result_df = result_df[result_df["passes_all_filters"]]

    outputs_dir = REPO_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    out_path = outputs_dir / f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(result_df)} rows to {out_path}")

    print(f"\nTop {args.top} by conviction score:")
    cols = ["company_name", "tradingsymbol", "conviction_score", "conviction_tier", "close", "rsi", "volume_surge"]
    print(result_df[cols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
