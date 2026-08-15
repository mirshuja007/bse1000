"""Pull daily OHLCV history for the resolved universe via Kite Connect.

Handles:
- local on-disk caching (so re-running a scan intraday doesn't re-hit the
  API for bars that can't have changed),
- a simple rate limiter to stay under Kite's historical-data throttle
  (documented at ~3 requests/second),
- retries with backoff on transient errors / rate-limit responses,
- fetching the benchmark index alongside the universe for relative-strength.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT


class RateLimiter:
    """Simple thread-safe token bucket limiting calls to `rate` per second."""

    def __init__(self, rate_per_second: float):
        self.min_interval = 1.0 / rate_per_second
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            sleep_for = self.min_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call = time.monotonic()


@dataclass
class FetchResult:
    bse_code: str
    tradingsymbol: str
    exchange: str
    ok: bool
    bars: pd.DataFrame | None
    error: str | None = None


def _cache_path(cache_dir: Path, instrument_token: int, interval: str) -> Path:
    return cache_dir / f"{instrument_token}_{interval}.parquet"


def _load_cache(cache_dir: Path, instrument_token: int, interval: str, ttl_hours: float):
    path = _cache_path(cache_dir, instrument_token, interval)
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > ttl_hours:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _save_cache(cache_dir: Path, instrument_token: int, interval: str, df: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_cache_path(cache_dir, instrument_token, interval), index=False)


def fetch_one(
    kite,
    instrument_token: int,
    history_days: int,
    interval: str,
    rate_limiter: RateLimiter,
    cache_dir: Path,
    cache_ttl_hours: float,
    max_retries: int = 3,
) -> pd.DataFrame:
    cached = _load_cache(cache_dir, instrument_token, interval, cache_ttl_hours)
    if cached is not None:
        return cached

    to_date = datetime.now().date()
    from_date = to_date - timedelta(days=history_days)

    last_exc = None
    for attempt in range(max_retries):
        rate_limiter.wait()
        try:
            bars = kite.historical_data(instrument_token, from_date, to_date, interval)
            df = pd.DataFrame(bars)
            if df.empty:
                raise ValueError("empty historical_data response")
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.sort_values("date").reset_index(drop=True)
            _save_cache(cache_dir, instrument_token, interval, df)
            return df
        except Exception as exc:  # kiteconnect raises various exception subclasses
            last_exc = exc
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"historical_data failed for token {instrument_token}: {last_exc}")


def fetch_universe_history(
    kite,
    mapping: pd.DataFrame,
    history_days: int,
    interval: str = "day",
    max_requests_per_second: float = 2.5,
    cache_dir: Path | str = REPO_ROOT / ".cache" / "ohlcv",
    cache_ttl_hours: float = 6,
    max_workers: int = 4,
    progress_callback=None,
) -> list[FetchResult]:
    """Fetch OHLCV for every resolved instrument in `mapping`.

    Runs with a small thread pool but a shared rate limiter, so total
    throughput stays bounded regardless of worker count.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_dir = Path(cache_dir)
    rate_limiter = RateLimiter(max_requests_per_second)
    resolved = mapping[mapping["resolved"] == True]  # noqa: E712

    results: list[FetchResult] = []

    def _task(row) -> FetchResult:
        try:
            df = fetch_one(
                kite,
                int(row["instrument_token"]),
                history_days,
                interval,
                rate_limiter,
                cache_dir,
                cache_ttl_hours,
            )
            return FetchResult(row["bse_code"], row["tradingsymbol"], row["exchange"], True, df)
        except Exception as exc:
            return FetchResult(row["bse_code"], row["tradingsymbol"], row["exchange"], False, None, str(exc))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_task, row): row for _, row in resolved.iterrows()}
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if progress_callback:
                progress_callback(done, len(futures))

    return results


def fetch_benchmark_history(
    kite,
    benchmark_token: int,
    history_days: int,
    interval: str = "day",
    cache_dir: Path | str = REPO_ROOT / ".cache" / "ohlcv",
    cache_ttl_hours: float = 6,
) -> pd.DataFrame:
    rate_limiter = RateLimiter(2.5)
    return fetch_one(
        kite, benchmark_token, history_days, interval, rate_limiter, Path(cache_dir), cache_ttl_hours
    )


def resolve_benchmark_token(kite, tradingsymbol: str = "NIFTY 500", exchange: str = "NSE") -> int:
    """Look up the instrument token for the benchmark index (indices live in
    the NSE instrument dump under instrument_type 'EQ' is False; they show up
    as segment 'INDICES')."""
    for row in kite.instruments(exchange):
        if row["tradingsymbol"] == tradingsymbol and row["segment"] == "INDICES":
            return row["instrument_token"]
    raise ValueError(f"Benchmark {exchange}:{tradingsymbol} not found in instrument dump")
