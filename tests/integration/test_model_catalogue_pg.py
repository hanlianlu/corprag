# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL CAS persistence and NOTIFY synchronization for model catalogue overlays."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.model_catalogue import PGModelCatalogueStore
from dlightrag.engine.ai.catalog import catalogue_overlay_revision
from tests.integration.pg_conn import PG_CONN_KWARGS

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_PG_CONN_KWARGS: dict[str, Any] = PG_CONN_KWARGS


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def catalogue_database() -> AsyncIterator[tuple[Any, dict[str, Any]]]:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")
    database = f"dlightrag_catalogue_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()
    kwargs = {**_PG_CONN_KWARGS, "database": database}
    pool = await asyncpg.create_pool(**kwargs, min_size=1, max_size=4)
    try:
        yield pool, kwargs
    finally:
        await pool.close()
        admin = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE "{database}" WITH (FORCE)')
        finally:
            await admin.close()


async def test_writer_initializes_singleton_and_publish_is_compare_and_set(
    catalogue_database: tuple[Any, dict[str, Any]],
) -> None:
    pool, kwargs = catalogue_database
    store = PGModelCatalogueStore(
        initial_revision=catalogue_overlay_revision(()),
        pool=pool,
        connection_kwargs=kwargs,
    )

    await store.initialize(validate_only=False)
    initial = await store.load()
    changed_revision = "sha256:" + "1" * 64

    assert initial.revision == catalogue_overlay_revision(())
    assert initial.overlay == []
    assert await store.publish(
        expected_revision=initial.revision,
        revision=changed_revision,
        overlay=[],
        actor="integration",
    )
    assert not await store.publish(
        expected_revision=initial.revision,
        revision="sha256:" + "2" * 64,
        overlay=[],
        actor="stale",
    )
    assert (await store.load()).revision == changed_revision


async def test_notify_listener_synchronizes_after_publish(
    catalogue_database: tuple[Any, dict[str, Any]],
) -> None:
    pool, kwargs = catalogue_database
    writer = PGModelCatalogueStore(
        initial_revision=catalogue_overlay_revision(()),
        pool=pool,
        connection_kwargs=kwargs,
    )
    listener = PGModelCatalogueStore(
        initial_revision=catalogue_overlay_revision(()),
        pool=pool,
        connection_kwargs=kwargs,
    )
    await writer.initialize(validate_only=False)
    synchronized = asyncio.Event()
    calls = 0

    async def on_change() -> None:
        nonlocal calls
        calls += 1
        synchronized.set()

    await listener.start_listener(on_change)
    await asyncio.wait_for(synchronized.wait(), timeout=5)
    synchronized.clear()
    initial = await writer.load()
    assert await writer.publish(
        expected_revision=initial.revision,
        revision="sha256:" + "3" * 64,
        overlay=[],
        actor="integration",
    )

    await asyncio.wait_for(synchronized.wait(), timeout=5)
    assert calls >= 2
    await listener.aclose()
