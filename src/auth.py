"""Kite Connect authentication.

Two modes are supported:

1. `login_automated()` - fully non-interactive login using the account
   password + TOTP secret from `.env`. This automates Zerodha's own web
   login flow (POST /api/login -> POST /api/twofa -> exchange the redirect's
   request_token for an access_token). This is the same 3-step flow every
   Kite algo-trading script uses to avoid a manual browser step every
   morning; it only ever touches your own account.

2. `login_manual()` - prints the standard Kite Connect login URL for you to
   open in a browser, and exchanges the request_token you paste back. Use
   this if you'd rather not store your password/TOTP secret at all, or if
   Zerodha changes the web login endpoints and the automated flow breaks.

The resulting access_token is cached to `.cache/session.json` (git-ignored)
and reused for the rest of the trading day instead of logging in on every
run.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyotp
import requests
from kiteconnect import KiteConnect

from src.config import REPO_ROOT, KiteCredentials

SESSION_CACHE = REPO_ROOT / ".cache" / "session.json"
LOGIN_URL = "https://kite.zerodha.com/api/login"
TWOFA_URL = "https://kite.zerodha.com/api/twofa"


class AuthError(RuntimeError):
    pass


def _cache_is_fresh(cache: dict) -> bool:
    """Kite access tokens expire around 06:00-08:00 IST the next day.
    We conservatively treat a cached token as stale once the calendar
    date has changed."""
    generated_on = cache.get("generated_on")
    if not generated_on:
        return False
    return generated_on == datetime.now().strftime("%Y-%m-%d")


def load_cached_access_token() -> str | None:
    if not SESSION_CACHE.exists():
        return None
    try:
        cache = json.loads(SESSION_CACHE.read_text())
    except json.JSONDecodeError:
        return None
    if _cache_is_fresh(cache):
        return cache.get("access_token")
    return None


def _save_access_token(access_token: str) -> None:
    SESSION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_CACHE.write_text(
        json.dumps(
            {
                "access_token": access_token,
                "generated_on": datetime.now().strftime("%Y-%m-%d"),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
    )


def login_automated(creds: KiteCredentials | None = None, timeout: int = 15) -> KiteConnect:
    """Log in end-to-end using password + TOTP secret. Returns an
    authenticated KiteConnect client."""
    creds = creds or KiteCredentials()
    if not creds.is_complete():
        raise AuthError(
            "Missing Kite credentials in environment: " + ", ".join(creds.missing_fields())
        )

    cached_token = load_cached_access_token()
    if cached_token:
        kite = KiteConnect(api_key=creds.api_key)
        kite.set_access_token(cached_token)
        return kite

    session = requests.Session()

    login_resp = session.post(
        LOGIN_URL,
        data={"user_id": creds.user_id, "password": creds.password},
        timeout=timeout,
    )
    login_resp.raise_for_status()
    login_data = login_resp.json()
    if login_data.get("status") != "success":
        raise AuthError(f"Kite login step failed: {login_data}")
    request_id = login_data["data"]["request_id"]

    totp_code = pyotp.TOTP(creds.totp_secret).now()
    twofa_resp = session.post(
        TWOFA_URL,
        data={
            "user_id": creds.user_id,
            "request_id": request_id,
            "twofa_value": totp_code,
            "twofa_type": "totp",
        },
        timeout=timeout,
    )
    twofa_resp.raise_for_status()
    twofa_data = twofa_resp.json()
    if twofa_data.get("status") != "success":
        raise AuthError(
            "Kite TOTP step failed - check KITE_TOTP_SECRET and system clock "
            f"drift. Response: {twofa_data}"
        )

    # Now hit the Connect login redirect using the authenticated session's
    # cookies to obtain a request_token.
    connect_resp = session.get(
        "https://kite.zerodha.com/connect/login",
        params={"api_key": creds.api_key, "v": "3"},
        allow_redirects=False,
        timeout=timeout,
    )
    request_token = None
    for _ in range(5):
        location = connect_resp.headers.get("Location")
        if not location:
            break
        qs = parse_qs(urlparse(location).query)
        if "request_token" in qs:
            request_token = qs["request_token"][0]
            break
        connect_resp = session.get(location, allow_redirects=False, timeout=timeout)

    if not request_token:
        raise AuthError(
            "Could not extract request_token from Kite's login redirect. "
            "Zerodha may have changed their login flow - fall back to "
            "login_manual() instead."
        )

    kite = KiteConnect(api_key=creds.api_key)
    session_data = kite.generate_session(request_token, api_secret=creds.api_secret)
    kite.set_access_token(session_data["access_token"])
    _save_access_token(session_data["access_token"])
    return kite


def get_login_url(creds: KiteCredentials | None = None) -> str:
    """Return the standard Kite Connect login URL to open in a browser -
    the first half of the manual login flow. Used by both the CLI prompt
    and the Streamlit "manual login" panel."""
    creds = creds or KiteCredentials()
    if not creds.api_key:
        raise AuthError("KITE_API_KEY missing from environment")
    return KiteConnect(api_key=creds.api_key).login_url()


def extract_request_token(raw: str) -> str:
    """Accept either a bare request_token or the full redirect URL Kite
    sends you to after login, and return just the token value."""
    raw = raw.strip()
    if "request_token" in raw:
        qs = parse_qs(urlparse(raw).query)
        if "request_token" in qs:
            return qs["request_token"][0]
    return raw


def exchange_request_token(request_token: str, creds: KiteCredentials | None = None) -> KiteConnect:
    """Second half of the manual login flow: trade a request_token (pasted
    from the browser redirect) for an authenticated KiteConnect client.
    This is the piece the Streamlit UI calls directly, since it can't use
    a blocking `input()` prompt like the CLI flow does."""
    creds = creds or KiteCredentials()
    if not creds.api_key or not creds.api_secret:
        raise AuthError("KITE_API_KEY / KITE_API_SECRET missing from environment")

    token = extract_request_token(request_token)
    if not token:
        raise AuthError("Empty request_token")

    kite = KiteConnect(api_key=creds.api_key)
    session_data = kite.generate_session(token, api_secret=creds.api_secret)
    kite.set_access_token(session_data["access_token"])
    _save_access_token(session_data["access_token"])
    return kite


def login_manual(creds: KiteCredentials | None = None) -> KiteConnect:
    """Interactive CLI fallback: prints a login URL, waits for you to paste
    the resulting request_token via a blocking `input()` prompt. Not usable
    from Streamlit - see `get_login_url` / `exchange_request_token` for the
    non-blocking building blocks the GUI uses instead."""
    creds = creds or KiteCredentials()
    if not creds.api_key or not creds.api_secret:
        raise AuthError("KITE_API_KEY / KITE_API_SECRET missing from environment")

    cached_token = load_cached_access_token()
    if cached_token:
        kite = KiteConnect(api_key=creds.api_key)
        kite.set_access_token(cached_token)
        return kite

    print("1. Open this URL in a browser and log in:")
    print(f"   {get_login_url(creds)}")
    print("2. After login you'll be redirected to a URL containing request_token=...")
    request_token = input("Paste the request_token (or the full redirect URL) here: ").strip()

    return exchange_request_token(request_token, creds)


def get_authenticated_kite(prefer_automated: bool = True) -> KiteConnect:
    creds = KiteCredentials()
    if prefer_automated and creds.is_complete():
        try:
            return login_automated(creds)
        except AuthError as exc:
            print(f"Automated login failed ({exc}); falling back to manual login.")
    return login_manual(creds)
