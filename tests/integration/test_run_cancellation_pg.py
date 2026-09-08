# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for cross-process cancellation wake (Task 5)."""

from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.engine.runtime.coordinator import RunCoordinator
from dlightrag.engine.runtime.records import (
    PreparedRunEnvelope,
    RunAccessScope,
    RunExecutionOutcome,
)
from tests.integration.pg_conn import PG_CONN_KWARGS

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_ADMIN: dict[str, Any] = PG_CONN_KWARGS
_TEST_DATABASE = "dlightrag_cancellation_test"


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_ADMIN)
        await conn.close()
        return True
    except OSError, asyncpg.PostgresError:
        return False


@pytest.fixture(autouse=True)
async def pool():
    if not await _pg_available():
        pytest.skip("PostgreSQL is not reachable")
    admin = await asyncpg.connect(**_ADMIN)
    try:
        await admin.execute(f'CREATE DATABASE "{_TEST_DATABASE}"')
    except asyncpg.DuplicateDatabaseError:
        pass
    finally:
        await admin.close()
    created = await asyncpg.create_pool(
        **{**_ADMIN, "database": _TEST_DATABASE}, min_size=1, max_size=8
    )
    try:
        yield created
    finally:
        await created.close()
        admin = await asyncpg.connect(**_ADMIN)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{_TEST_DATABASE}" WITH (FORCE)')
        finally:
            await admin.close()


def _request(query: str = "why") -> dict[str, Any]:
    import uuid as _uuid

    return {
        "agent_session_id": str(_uuid.uuid7()),
        "agent_lane_id": "main",
        "query": query,
        "workspaces": ["default"],
    }


async def _create(store: PGRunStore):
    import uuid

    run_id = str(uuid.uuid7())
    prepared = _request()
    return await store.create_run(
        envelope=PreparedRunEnvelope(
            run_kind="answer",
            lane="query",
            submitted_by="owner-a",
            access_scope=RunAccessScope(kind="owner", scope_id="owner-a"),
            submission_key=run_id,
            request_fingerprint="f" * 64,
            payload=prepared,
            accepted_input=prepared,
            retention_seconds=365 * 24 * 60 * 60,
        ),
        run_id=run_id,
    )


async def _claim(pool) -> tuple[PGRunStore, Any]:
    store = PGRunStore(pool=pool)
    await store.initialize()
    creation = await _create(store)
    claimed = await store.claim_next(worker_id="worker-1")
    assert claimed is not None
    return store, creation.run.run_id


async def test_listener_wakes_the_lease_owner_on_notify(pool) -> None:
    store, run_id = await _claim(pool)
    woken: list[str] = []

    async def _on_cancel(owner_id: str, target: str) -> None:
        woken.append(target)

    listener = store.build_cancellation_listener(worker_id="worker-1", on_cancel=_on_cancel)
    await listener.start()
    try:
        import asyncio

        await asyncio.wait_for(listener.ready.wait(), timeout=5.0)
        outcome = await store.request_cancellation(owner_id="owner-a", run_id=run_id)
        assert outcome.outcome == "pending"
        for _ in range(200):
            if woken:
                break
            await asyncio.sleep(0.01)
        assert woken == [run_id]
    finally:
        await listener.aclose()


async def test_second_connection_wake_reaches_a_running_coordinator(pool) -> None:
    store = PGRunStore(pool=pool)
    await store.initialize()
    creation = await _create(store)
    run_id = creation.run.run_id
    cancelled: list[str] = []

    class _BlockedExecutor:
        async def execute(self, session: Any) -> RunExecutionOutcome:
            while True:
                await __import__("asyncio").sleep(60)

    coordinator = RunCoordinator(
        store=store,
        executors={"answer": _BlockedExecutor()},
        query_worker_concurrency=1,
        worker_id="worker-1",
    )

    async def _on_cancel(owner_id: str, target: str) -> None:
        coordinator.cancel_local(owner_id, target)
        cancelled.append(target)

    listener = store.build_cancellation_listener(worker_id="worker-1", on_cancel=_on_cancel)
    await listener.start()
    await coordinator.start()
    try:
        import asyncio

        await asyncio.wait_for(listener.ready.wait(), timeout=5.0)
        for _ in range(200):
            record = await store.get_run(owner_id="owner-a", run_id=run_id)
            if (
                record is not None
                and record.status == "running"
                and record.lease_owner == "worker-1"
            ):
                break
            await asyncio.sleep(0.01)
        # A second "process" cancels: its durable write NOTIFYs the channel in
        # the same transaction, and only the authoritative row wakes the owner.
        other = PGRunStore(pool=pool)
        outcome = await other.request_cancellation(owner_id="owner-a", run_id=run_id)
        assert outcome.outcome == "pending"
        for _ in range(400):
            record = await store.get_run(owner_id="owner-a", run_id=run_id)
            if record is not None and record.status == "cancelled":
                break
            await asyncio.sleep(0.01)
        record = await store.get_run(owner_id="owner-a", run_id=run_id)
        assert record is not None
        assert record.status == "cancelled"
        assert cancelled == [run_id]
        # Exactly one cancelled terminal event and result commit.
        events = await store.read_event_page(owner_id="owner-a", run_id=run_id)
        done = [event for event in events if event.event_type == "done"]
        assert len(done) == 1
    finally:
        await listener.aclose()
        await coordinator.aclose()
