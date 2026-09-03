"""Persistent storage for the app's own generated data (tracked picks,
recommendation history) using the GitHub Contents API, so it survives
Streamlit Cloud redeploys - the local filesystem those write to normally
lives on a throwaway container that gets wiped on every redeploy.

Writes go to a *separate* branch (GITHUB_DATA_BRANCH, default
"data-store") in the same repo, deliberately NOT the branch Streamlit
Cloud deploys from. If it wrote to the deploy branch, every save would
push a new commit there and trigger a fresh redeploy, restarting the app
mid-session - the data branch exists specifically to avoid that loop.

Configuration (env vars, same pattern as KITE_* credentials):
    GITHUB_TOKEN       - a PAT with Contents: read/write on the target repo
    GITHUB_DATA_REPO   - "owner/repo", e.g. "mirshuja007/bse1000"
    GITHUB_DATA_BRANCH - branch to store data in (default: "data-store")

If GITHUB_TOKEN/GITHUB_DATA_REPO aren't set, is_configured() returns False
and every caller in this app falls back to the local (redeploy-unsafe)
file - so the app still works without this configured, just without
persistence across redeploys.
"""
from __future__ import annotations

import base64
import os

import requests

GITHUB_API = "https://api.github.com"
DEFAULT_BRANCH_NAME = "data-store"
_REQUEST_TIMEOUT = 15


def _config() -> tuple[str | None, str | None, str]:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_DATA_REPO")
    branch = os.environ.get("GITHUB_DATA_BRANCH", DEFAULT_BRANCH_NAME)
    return token, repo, branch


def is_configured() -> bool:
    token, repo, _ = _config()
    return bool(token and repo)


def _headers(token: str) -> dict:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def _ensure_branch_exists(token: str, repo: str, branch: str) -> None:
    ref_url = f"{GITHUB_API}/repos/{repo}/git/ref/heads/{branch}"
    resp = requests.get(ref_url, headers=_headers(token), timeout=_REQUEST_TIMEOUT)
    if resp.status_code == 200:
        return  # already exists

    repo_resp = requests.get(f"{GITHUB_API}/repos/{repo}", headers=_headers(token), timeout=_REQUEST_TIMEOUT)
    repo_resp.raise_for_status()
    default_branch = repo_resp.json()["default_branch"]

    default_ref_resp = requests.get(
        f"{GITHUB_API}/repos/{repo}/git/ref/heads/{default_branch}", headers=_headers(token), timeout=_REQUEST_TIMEOUT
    )
    default_ref_resp.raise_for_status()
    base_sha = default_ref_resp.json()["object"]["sha"]

    create_resp = requests.post(
        f"{GITHUB_API}/repos/{repo}/git/refs",
        headers=_headers(token),
        json={"ref": f"refs/heads/{branch}", "sha": base_sha},
        timeout=_REQUEST_TIMEOUT,
    )
    # 422 here means another concurrent call already created it - fine.
    if create_resp.status_code not in (201, 422):
        create_resp.raise_for_status()


def read_file(path: str) -> str | None:
    """Returns the file's text content, or None if it doesn't exist yet
    (a brand-new deployment with nothing saved) or the store isn't
    configured. Raises on genuine errors (bad token, network failure)
    rather than silently swallowing them - a caller falling back to local
    data because of a misconfigured token should be a visible problem,
    not a silent one."""
    token, repo, branch = _config()
    if not is_configured():
        return None

    resp = requests.get(
        f"{GITHUB_API}/repos/{repo}/contents/{path}",
        headers=_headers(token),
        params={"ref": branch},
        timeout=_REQUEST_TIMEOUT,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return base64.b64decode(resp.json()["content"]).decode("utf-8")


def write_file(path: str, content: str, message: str, max_retries: int = 2) -> None:
    """Create or update `path` on the data branch. Retries once on a 409
    (stale SHA - another write landed between our read and write) by
    re-fetching the current SHA and trying again."""
    token, repo, branch = _config()
    if not is_configured():
        raise RuntimeError("GitHub data store not configured (set GITHUB_TOKEN and GITHUB_DATA_REPO)")

    _ensure_branch_exists(token, repo, branch)
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"

    last_exc = None
    for attempt in range(max_retries + 1):
        get_resp = requests.get(url, headers=_headers(token), params={"ref": branch}, timeout=_REQUEST_TIMEOUT)
        sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

        body = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if sha:
            body["sha"] = sha

        put_resp = requests.put(url, headers=_headers(token), json=body, timeout=_REQUEST_TIMEOUT)
        if put_resp.status_code in (200, 201):
            return
        if put_resp.status_code in (409, 422) and attempt < max_retries:
            last_exc = None
            continue
        put_resp.raise_for_status()

    if last_exc:
        raise last_exc
