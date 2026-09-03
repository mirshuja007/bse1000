"""Offline unit tests for src/github_store.py - all HTTP calls mocked, no
real network access."""
import base64

import pytest

from src import github_store as gs


class FakeResponse:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def configured_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    monkeypatch.setenv("GITHUB_DATA_REPO", "someuser/somerepo")
    monkeypatch.setenv("GITHUB_DATA_BRANCH", "data-store")


def test_is_configured_requires_both_token_and_repo(monkeypatch):
    assert gs.is_configured() is True
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert gs.is_configured() is False


def test_is_configured_false_when_env_vars_absent(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_DATA_REPO", raising=False)
    assert gs.is_configured() is False


def test_read_file_returns_none_when_not_configured(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert gs.read_file("data/whatever.csv") is None


def test_read_file_returns_none_on_404(monkeypatch):
    monkeypatch.setattr(gs.requests, "get", lambda *a, **k: FakeResponse(404))
    assert gs.read_file("data/tracked_picks.csv") is None


def test_read_file_decodes_base64_content(monkeypatch):
    text = "a,b\n1,2\n"
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    monkeypatch.setattr(gs.requests, "get", lambda *a, **k: FakeResponse(200, {"content": encoded, "sha": "abc123"}))
    assert gs.read_file("data/tracked_picks.csv") == text


def test_write_file_raises_when_not_configured(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        gs.write_file("data/x.csv", "content", "msg")


def test_write_file_creates_branch_when_missing_then_writes(monkeypatch):
    calls = {"get": [], "post": [], "put": []}

    def fake_get(url, headers=None, params=None, timeout=None):
        calls["get"].append(url)
        if url.endswith("/git/ref/heads/data-store"):
            return FakeResponse(404)  # branch doesn't exist yet
        if url.endswith("/repos/someuser/somerepo"):
            return FakeResponse(200, {"default_branch": "main"})
        if url.endswith("/git/ref/heads/main"):
            return FakeResponse(200, {"object": {"sha": "base-sha-123"}})
        if url.endswith("/contents/data/x.csv"):
            return FakeResponse(404)  # file doesn't exist yet either
        raise AssertionError(f"unexpected GET {url}")

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["post"].append((url, json))
        assert json["ref"] == "refs/heads/data-store"
        assert json["sha"] == "base-sha-123"
        return FakeResponse(201)

    def fake_put(url, headers=None, json=None, timeout=None):
        calls["put"].append((url, json))
        return FakeResponse(201)

    monkeypatch.setattr(gs.requests, "get", fake_get)
    monkeypatch.setattr(gs.requests, "post", fake_post)
    monkeypatch.setattr(gs.requests, "put", fake_put)

    gs.write_file("data/x.csv", "hello", "commit message")

    assert len(calls["post"]) == 1  # branch was created exactly once
    assert len(calls["put"]) == 1
    put_body = calls["put"][0][1]
    assert put_body["message"] == "commit message"
    assert base64.b64decode(put_body["content"]).decode() == "hello"
    assert "sha" not in put_body  # no existing file, so no sha in the update body


def test_write_file_includes_sha_when_updating_existing_file(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None):
        if url.endswith("/git/ref/heads/data-store"):
            return FakeResponse(200)  # branch already exists
        if url.endswith("/contents/data/x.csv"):
            return FakeResponse(200, {"sha": "existing-sha-456"})
        raise AssertionError(f"unexpected GET {url}")

    put_calls = []

    def fake_put(url, headers=None, json=None, timeout=None):
        put_calls.append(json)
        return FakeResponse(200)

    monkeypatch.setattr(gs.requests, "get", fake_get)
    monkeypatch.setattr(gs.requests, "put", fake_put)

    gs.write_file("data/x.csv", "updated content", "update")

    assert put_calls[0]["sha"] == "existing-sha-456"


def test_write_file_retries_once_on_stale_sha_conflict(monkeypatch):
    get_call_count = {"n": 0}

    def fake_get(url, headers=None, params=None, timeout=None):
        if url.endswith("/git/ref/heads/data-store"):
            return FakeResponse(200)
        if url.endswith("/contents/data/x.csv"):
            get_call_count["n"] += 1
            # First read: stale sha. After the conflict, second read: fresh sha.
            sha = "stale-sha" if get_call_count["n"] == 1 else "fresh-sha"
            return FakeResponse(200, {"sha": sha})
        raise AssertionError(f"unexpected GET {url}")

    put_attempts = []

    def fake_put(url, headers=None, json=None, timeout=None):
        put_attempts.append(json["sha"])
        if json["sha"] == "stale-sha":
            return FakeResponse(409)
        return FakeResponse(200)

    monkeypatch.setattr(gs.requests, "get", fake_get)
    monkeypatch.setattr(gs.requests, "put", fake_put)

    gs.write_file("data/x.csv", "content", "msg")

    assert put_attempts == ["stale-sha", "fresh-sha"]
