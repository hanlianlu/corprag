# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit-test fixtures — isolate from .env file.

Prevents real .env values (API keys, provider settings) from leaking
into unit tests that construct DlightragConfig() directly.
"""

from __future__ import annotations

import os

import pytest

from dlightrag.config import DlightragConfig


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent .env file from polluting unit tests."""
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)
    for key in list(os.environ):
        if key.startswith("DLIGHTRAG_"):
            monkeypatch.delenv(key, raising=False)
