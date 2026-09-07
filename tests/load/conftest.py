# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Opt-in guard for the deterministic RunRuntime load campaign."""

from __future__ import annotations

import os

import asyncpg
import pytest

from tests.integration.pg_conn import PG_CONN_KWARGS


def _enabled() -> bool:
    return os.environ.get("DLIGHTRAG_RUN_LOAD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.fixture(autouse=True)
async def require_opt_in_and_postgres() -> None:
    if not _enabled():
        pytest.skip("set DLIGHTRAG_RUN_LOAD=1 to run the RunRuntime load campaign")
    try:
        connection = await asyncpg.connect(**PG_CONN_KWARGS)
        await connection.fetchval("SELECT 1")
        await connection.close()
    except Exception:
        pytest.fail(
            "RunRuntime load campaign requires reachable isolated PostgreSQL", pytrace=False
        )
