"""Load app configuration: YAML defaults + environment secrets.

Secrets (Kite API key/secret, user id, password, TOTP secret) are read only
from environment variables (populated via a local `.env` file that is
git-ignored). They are never read from or written to the YAML config, so the
config file stays safe to commit and edit freely.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "scanner_config.yaml"

load_dotenv(REPO_ROOT / ".env")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, overrides: dict) -> dict:
    """Merge `overrides` into `base`, recursing into nested dicts."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class KiteCredentials:
    """Reads Kite Connect credentials strictly from environment variables."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("KITE_API_KEY", "")
        self.api_secret = os.environ.get("KITE_API_SECRET", "")
        self.user_id = os.environ.get("KITE_USER_ID", "")
        self.password = os.environ.get("KITE_PASSWORD", "")
        self.totp_secret = os.environ.get("KITE_TOTP_SECRET", "")

    def is_complete(self) -> bool:
        return all(
            [self.api_key, self.api_secret, self.user_id, self.password, self.totp_secret]
        )

    def missing_fields(self) -> list[str]:
        fields = {
            "KITE_API_KEY": self.api_key,
            "KITE_API_SECRET": self.api_secret,
            "KITE_USER_ID": self.user_id,
            "KITE_PASSWORD": self.password,
            "KITE_TOTP_SECRET": self.totp_secret,
        }
        return [name for name, value in fields.items() if not value]


def app_password() -> str:
    return os.environ.get("APP_PASSWORD", "")
