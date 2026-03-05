# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures for dlightrag tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dlightrag.config import DlightragConfig, reset_config, set_config


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset the config singleton before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def tmp_working_dir(tmp_path: Path) -> Path:
    """Create a temporary working directory structure."""
    working_dir = tmp_path / "dlightrag_storage"
    (working_dir / "artifacts" / "local").mkdir(parents=True)
    return working_dir


@pytest.fixture
def test_config(tmp_working_dir: Path) -> DlightragConfig:
    """Create a test config with temporary paths.

    Also sets the global singleton so that code calling get_config()
    directly (e.g. /health endpoint) gets the test config.
    """
    cfg = DlightragConfig(  # type: ignore[call-arg]
        working_dir=str(tmp_working_dir),
        openai_api_key=os.getenv("DLIGHTRAG_OPENAI_API_KEY", "test-key-for-unit-tests"),
        # Use JSON storage for unit tests (no PG dependency)
        kv_storage="JsonKVStorage",
        doc_status_storage="JsonDocStatusStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
    )
    set_config(cfg)
    return cfg
