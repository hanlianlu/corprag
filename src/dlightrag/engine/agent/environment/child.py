# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Explicit functional environment for Bash, fd, and ripgrep children."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

_INHERIT = frozenset({"PATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"})
_SAFE_CA = frozenset({"SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"})
_PROXY = frozenset(
    {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy"}
)
_SECRET_PATTERNS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTHORIZATION",
)
_BLOCK_PREFIXES = (
    "DLIGHTRAG_",
    "POSTGRES_",
    "LIGHTRAG_",
    "LANGFUSE_",
    "OPENAI_",
    "ANTHROPIC_",
)
_BLOCK_EXACT = frozenset(
    {
        "SSH_AUTH_SOCK",
        "GIT_ASKPASS",
        "GIT_TERMINAL_PROMPT",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "NODE_OPTIONS",
        "RIPGREP_CONFIG_PATH",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "PGPASSWORD",
        "DATABASE_URL",
    }
)


def build_child_environment(
    *,
    home: Path,
    tmp: Path,
    parent: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an explicit env: usable PATH, no service secrets, no full inheritance."""
    source = os.environ if parent is None else parent
    env: dict[str, str] = {}
    for name in _INHERIT | _SAFE_CA:
        value = source.get(name)
        if value:
            env[name] = value
    for name in _PROXY:
        value = source.get(name)
        cleaned = _uncredentialed_proxy(value) if value else None
        if cleaned:
            env[name] = cleaned
    env.update(
        {
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "TEMP": str(tmp),
            "TMP": str(tmp),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "TERM": "dumb",
            "NO_COLOR": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return {key: value for key, value in env.items() if not _blocked(key)}


def _uncredentialed_proxy(value: str) -> str | None:
    if "://" not in value:
        return value
    host = value.split("://", 1)[1].split("/", 1)[0]
    if "@" in host:
        return None
    return value


def looks_like_secret_name(name: str) -> bool:
    """Return True when a variable name matches the seeded secret-pattern test."""
    return _blocked(name)


def _blocked(name: str) -> bool:
    if name in _BLOCK_EXACT:
        return True
    if any(name.startswith(prefix) for prefix in _BLOCK_PREFIXES):
        return True
    upper = name.upper()
    return any(pattern in upper for pattern in _SECRET_PATTERNS)


__all__ = ["build_child_environment", "looks_like_secret_name"]
