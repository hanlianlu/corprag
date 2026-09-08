# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL Profile Memory records, journal, atomic receipts, undo, and recall."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
from dlightrag_memory import Memory, MemoryOperation, MemoryProvenance, MemoryRecord
from dlightrag_memory._storage.pg_bm25 import index_name
from dlightrag_memory.errors import MemoryWriteRejectedError
from dlightrag_memory.normalize import normalized_body
from dlightrag_memory.postgres import PostgresMemoryStore
from dlightrag_memory.store import operation_change_id, operation_record_id

from dlightrag.adapters.postgres.answer.memory_settings import (
    MEMORY_SETTINGS_DDL,
    PGMemorySettingsStore,
)
from dlightrag.application.memory import MemoryDisabledError, MemoryService
from tests.integration.pg_conn import PG_CONN_KWARGS

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_PG: dict[str, Any] = PG_CONN_KWARGS


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def store() -> AsyncIterator[PostgresMemoryStore]:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")
    db_name = f"dlightrag_mem_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(**{**_PG, "database": db_name}, min_size=1, max_size=4)
    created = PostgresMemoryStore(pool=pool)
    await created.initialize()
    try:
        yield created
    finally:
        await created.aclose()
        await pool.close()
        admin = await asyncpg.connect(**_PG)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


def _provenance(run: str = "run-1") -> MemoryProvenance:
    return MemoryProvenance(
        origin_kind="answer_run", origin_id=run, run_id=run, session_id="session-1"
    )


def _record(
    *, owner: str = "alpha", body: str = "No email.", memory_id: str | None = None
) -> MemoryRecord:
    now = datetime.now(UTC)
    return MemoryRecord(
        owner_id=owner,
        memory_id=memory_id or str(uuid.uuid4()),
        kind="preference",
        body=body,
        provenance=_provenance(),
        created_at=now,
        updated_at=now,
    )


async def _active(memory: Memory, *, owner_id: str = "alpha") -> tuple[MemoryRecord, ...]:
    records, _ = await memory.browse(owner_id=owner_id, limit=100)
    return records


async def _store_active(
    store: PostgresMemoryStore, *, owner_id: str = "alpha"
) -> tuple[MemoryRecord, ...]:
    records, _ = await store.list_active_page(owner_id=owner_id, limit=100)
    return records


async def test_pg_owners_are_isolated(store: PostgresMemoryStore) -> None:
    await store.insert(_record(owner="alpha", body="Alpha only."))
    await store.insert(_record(owner="beta", body="Beta only."))
    assert [row.body for row in await _store_active(store, owner_id="alpha")] == ["Alpha only."]
    assert [row.body for row in await _store_active(store, owner_id="beta")] == ["Beta only."]


async def test_pg_initialization_rejects_legacy_confidence_schema(
    store: PostgresMemoryStore,
) -> None:
    pool = store._operation_pool
    assert pool is not None
    async with pool.acquire() as conn:
        await conn.execute(
            "ALTER TABLE dlightrag_memory_records "
            "ADD COLUMN confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0"
        )
    with pytest.raises(RuntimeError, match="removed confidence"):
        await store.initialize()


async def test_pg_operation_replay_duplicate_cap_and_schema(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    first = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="Use Chinese.",
        provenance=_provenance(),
        idempotency_key="call-1",
        mutation_scope="run-1",
        mutation_limit=1,
    )
    replay = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="Use Chinese.",
        provenance=_provenance(),
        idempotency_key="call-1",
        mutation_scope="run-1",
        mutation_limit=1,
    )
    duplicate = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="  use chinese. ",
        provenance=_provenance(),
        idempotency_key="call-2",
        mutation_scope="run-1",
        mutation_limit=1,
    )
    assert replay == first
    assert duplicate.outcome == "unchanged"
    with pytest.raises(MemoryWriteRejectedError, match="mutation limit"):
        await memory.remember(
            owner_id="alpha",
            kind="fact",
            body="Lives in Gothenburg.",
            provenance=_provenance(),
            idempotency_key="call-3",
            mutation_scope="run-1",
            mutation_limit=1,
        )
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_operations WHERE owner_id = 'alpha'"
            )
            == 2
        )
        assert not await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'dlightrag_memory_records' AND column_name = 'confidence'"
        )


async def test_pg_owner_lock_rechecks_deactivation_before_mutation_commit(
    store: PostgresMemoryStore,
) -> None:
    pool = store._operation_pool
    assert pool is not None
    async with pool.acquire() as conn:
        for statement in MEMORY_SETTINGS_DDL:
            await conn.execute(statement)
    service = MemoryService(
        store,
        settings_store=PGMemorySettingsStore(pool=pool),
        memory_list_cursor_secret=b"memory-pg-list-test",
    )

    async with pool.acquire() as conn, conn.transaction():
        await conn.fetchval("SELECT pg_advisory_xact_lock(hashtext($1))", "alpha")
        await conn.execute(
            "INSERT INTO dlightrag_answer_memory_settings "
            "(owner_id, enabled, epoch) VALUES ($1, FALSE, 1)",
            "alpha",
        )
        pending = asyncio.create_task(
            service.remember(
                owner_id="alpha",
                auth_mode="jwt",
                kind="fact",
                body="Stable.",
                provenance=_provenance(),
                idempotency_key="call-after-disable",
            )
        )
        await asyncio.sleep(0.05)
        assert not pending.done()

    with pytest.raises(MemoryDisabledError):
        await pending
    assert await store.count_active(owner_id="alpha") == 0


async def test_pg_supersede_forget_and_compensating_undo(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    old = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Lives in Beijing.",
        provenance=_provenance(),
        idempotency_key="call-1",
    )
    replacement = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Lives in Gothenburg.",
        provenance=_provenance(),
        idempotency_key="call-2",
        supersedes_id=old.memory_id,
    )
    undone = await memory.undo(
        owner_id="alpha",
        change_id=replacement.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
        idempotency_key="undo-1",
    )
    assert undone.outcome == "changed"
    assert [row.body for row in await _active(memory)] == ["Lives in Beijing."]

    forgotten = await memory.forget(
        owner_id="alpha",
        memory_id=undone.memory_id,
        provenance=_provenance(),
        idempotency_key="forget-1",
    )
    restored = await memory.undo(
        owner_id="alpha",
        change_id=forgotten.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-2"),
        idempotency_key="undo-2",
    )
    assert restored.outcome == "changed"
    assert [row.body for row in await _active(memory)] == ["Lives in Beijing."]


async def test_pg_clear_physically_erases_records_and_operations(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Stable.",
        provenance=_provenance(),
        idempotency_key="call-1",
    )
    assert await memory.clear(owner_id="alpha") == 1
    assert await _active(memory) == ()


async def test_pg_purge_expired_non_active_rows(store: PostgresMemoryStore) -> None:
    old = _record(body="Stale.")
    await store.insert(old)
    await store.supersede(owner_id="alpha", old_id=old.memory_id, new=_record(body="Fresh."))
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "UPDATE dlightrag_memory_records "
            "SET updated_at = NOW() - INTERVAL '400 days' WHERE memory_id = $1",
            uuid.UUID(old.memory_id),
        )
    removed = await store.purge_superseded(older_than=datetime.now(UTC) - timedelta(days=365))
    assert removed == 1
    assert await store.get(owner_id="alpha", memory_id=old.memory_id) is None


async def test_pg_recall_legs_find_only_the_owner(store: PostgresMemoryStore) -> None:
    await store.insert(_record(owner="alpha", body="No email."))
    await store.insert(_record(owner="beta", body="No email."))
    await store.insert(_record(owner="alpha", body="Deploy at midnight."))

    candidates = await store.search_candidates(owner_id="alpha", query="No email", limit=10)
    assert "No email." in {candidate.record.body for candidate in candidates}
    assert all(candidate.record.owner_id == "alpha" for candidate in candidates)


async def test_pg_bm25_indexes_are_provisioned(store: PostgresMemoryStore) -> None:
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        rows = await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'dlightrag_memory_records' "
            "AND indexname LIKE $1",
            f"{index_name('simple')}%",
        )
        assert index_name("simple") in {str(row["indexname"]) for row in rows}


async def _undo(memory: Memory, change_id: str, *, key: str):
    return await memory.undo(
        owner_id="alpha",
        change_id=change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id=f"undo-{key}"),
        idempotency_key=key,
    )


async def _forget_duplicate_rows(memory: Memory, store: PostgresMemoryStore):
    first = _record(body="Prefers tea.")
    second = _record(body="  prefers tea.  ")
    third = _record(body="Likes trains.")
    await store.insert(first)
    await store.insert(second)
    await store.insert(third)
    forgotten = await memory.forget(
        owner_id="alpha",
        body="Prefers tea.",
        provenance=_provenance(),
        idempotency_key="forget-1",
    )
    assert forgotten.outcome == "changed"
    assert len(forgotten.memory_ids) == 2
    return forgotten, (first, second, third)


class _CountingConnection:
    """Connection proxy recording one entry per adapter statement call.

    Each entry is one wire-level statement the backend receives, so the
    recorded list is the backend statement shape of the settlement. The proxy
    deliberately exposes no ``executemany``: any batch-by-client-side-loop
    insert would raise AttributeError and fail the test loudly.
    """

    def __init__(self, conn: Any, calls: list[tuple[Any, ...]]) -> None:
        self._conn = conn
        self._calls = calls

    def _record(self, kind: str, query: str, args: tuple[Any, ...]) -> None:
        self._calls.append((kind, query.split()[0], query, args))

    async def fetch(self, query: str, *args: Any):
        self._record("fetch", query, args)
        return await self._conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any):
        self._record("fetchrow", query, args)
        return await self._conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any):
        self._record("fetchval", query, args)
        return await self._conn.fetchval(query, *args)

    async def execute(self, query: str, *args: Any):
        self._record("execute", query, args)
        return await self._conn.execute(query, *args)

    def transaction(self) -> Any:
        return self._conn.transaction()


class _CountingAcquire:
    def __init__(self, pool: Any, calls: list[tuple[Any, ...]]) -> None:
        self._pool = pool
        self._calls = calls
        self._ctx: Any = None

    async def __aenter__(self) -> _CountingConnection:
        self._ctx = self._pool.acquire()
        conn = await self._ctx.__aenter__()
        return _CountingConnection(conn, self._calls)

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._ctx.__aexit__(*exc)


class _CountingPool:
    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self.calls: list[tuple[Any, ...]] = []

    def acquire(self) -> _CountingAcquire:
        return _CountingAcquire(self._pool, self.calls)


async def test_pg_multi_row_forget_undo_restores_all_rows_in_order(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    forgotten, (first, second, third) = await _forget_duplicate_rows(memory, store)
    assert third.memory_id not in forgotten.memory_ids

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "changed"
    assert undone.memory_ids == (
        operation_record_id("alpha", undone.change_id, index=0),
        operation_record_id("alpha", undone.change_id, index=1),
    )
    restored = [
        record
        for memory_id in undone.memory_ids
        if (record := await store.get(owner_id="alpha", memory_id=memory_id)) is not None
    ]
    assert len(restored) == 2
    assert [record.supersedes_id for record in restored] == list(forgotten.memory_ids)
    assert all(record.status == "active" for record in restored)
    assert all(record.provenance.origin_kind == "undo" for record in restored)
    assert all(record.created_at == undone.created_at for record in restored)
    assert all(record.updated_at == undone.created_at for record in restored)
    assert {record.body for record in restored} == {"Prefers tea.", "  prefers tea.  "}
    first_row = await store.get(owner_id="alpha", memory_id=first.memory_id)
    second_row = await store.get(owner_id="alpha", memory_id=second.memory_id)
    assert first_row is not None and first_row.status == "forgotten"
    assert second_row is not None and second_row.status == "forgotten"
    assert await store.count_active(owner_id="alpha") == 3


async def test_pg_multi_row_forget_undo_late_wrong_state_row_conflicts_cleanly(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    forgotten, (first, second, _third) = await _forget_duplicate_rows(memory, store)

    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "UPDATE dlightrag_memory_records SET status = 'active' "
            "WHERE owner_id = 'alpha' AND memory_id = $1",
            uuid.UUID(second.memory_id),
        )

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 2  # r3 plus the drifted row
    first_row = await store.get(owner_id="alpha", memory_id=first.memory_id)
    assert first_row is not None and first_row.status == "forgotten"
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_records "
                "WHERE owner_id = 'alpha' AND origin_kind = 'undo'"
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT undone_by IS NULL FROM dlightrag_memory_operations "
                "WHERE owner_id = 'alpha' AND change_id = $1",
                uuid.UUID(forgotten.change_id),
            )
            is True
        )


async def test_pg_multi_row_forget_undo_late_missing_row_conflicts_cleanly(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    forgotten, (first, second, _third) = await _forget_duplicate_rows(memory, store)

    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "DELETE FROM dlightrag_memory_records WHERE owner_id = 'alpha' AND memory_id = $1",
            uuid.UUID(second.memory_id),
        )

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 1  # only r3 remains active
    first_row = await store.get(owner_id="alpha", memory_id=first.memory_id)
    assert first_row is not None and first_row.status == "forgotten"
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_records "
                "WHERE owner_id = 'alpha' AND origin_kind = 'undo'"
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT undone_by IS NULL FROM dlightrag_memory_operations "
                "WHERE owner_id = 'alpha' AND change_id = $1",
                uuid.UUID(forgotten.change_id),
            )
            is True
        )


async def test_pg_multi_row_forget_undo_external_duplicate_conflicts_cleanly(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    forgotten, (first, second, _third) = await _forget_duplicate_rows(memory, store)

    external = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="PREFERS TEA.",
        provenance=_provenance(),
        idempotency_key="call-2",
    )
    assert external.outcome == "changed"

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 2  # r3 plus the external duplicate
    first_row = await store.get(owner_id="alpha", memory_id=first.memory_id)
    second_row = await store.get(owner_id="alpha", memory_id=second.memory_id)
    assert first_row is not None and first_row.status == "forgotten"
    assert second_row is not None and second_row.status == "forgotten"
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_records "
                "WHERE owner_id = 'alpha' AND origin_kind = 'undo'"
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT undone_by IS NULL FROM dlightrag_memory_operations "
                "WHERE owner_id = 'alpha' AND change_id = $1",
                uuid.UUID(forgotten.change_id),
            )
            is True
        )

    # The target stays undoable once the external duplicate is gone.
    await memory.forget(
        owner_id="alpha",
        memory_id=external.memory_id,
        provenance=_provenance(),
        idempotency_key="forget-2",
    )
    retry = await _undo(memory, forgotten.change_id, key="undo-2")
    assert retry.outcome == "changed"
    assert await store.count_active(owner_id="alpha") == 3


async def test_pg_multi_row_forget_undo_deterministic_id_collision_rolls_back(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    forgotten, (first, second, _third) = await _forget_duplicate_rows(memory, store)

    undo_change_id = operation_change_id(
        MemoryOperation(
            owner_id="alpha",
            idempotency_key="undo-1",
            action="undo",
            provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
            target_change_id=forgotten.change_id,
        )
    )
    squatter_id = operation_record_id("alpha", undo_change_id, index=0)
    await store.insert(_record(body="Squatter.", memory_id=squatter_id))

    with pytest.raises(ValueError, match="already exists"):
        await _undo(memory, forgotten.change_id, key="undo-1")

    # The transaction rolled back: no restored rows, no undone_by mark, and the
    # squatter and both forgotten targets are untouched.
    squatter = await store.get(owner_id="alpha", memory_id=squatter_id)
    first_row = await store.get(owner_id="alpha", memory_id=first.memory_id)
    second_row = await store.get(owner_id="alpha", memory_id=second.memory_id)
    assert squatter is not None and squatter.body == "Squatter."
    assert first_row is not None and first_row.status == "forgotten"
    assert second_row is not None and second_row.status == "forgotten"
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_records "
                "WHERE owner_id = 'alpha' AND origin_kind = 'undo'"
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT undone_by IS NULL FROM dlightrag_memory_operations "
                "WHERE owner_id = 'alpha' AND change_id = $1",
                uuid.UUID(forgotten.change_id),
            )
            is True
        )
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_operations WHERE owner_id = 'alpha'"
            )
            == 1
        )

    # Clearing the collision leaves the same undo idempotency key settleable.
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "DELETE FROM dlightrag_memory_records WHERE owner_id = 'alpha' AND memory_id = $1",
            uuid.UUID(squatter_id),
        )
    retry = await _undo(memory, forgotten.change_id, key="undo-1")
    assert retry.outcome == "changed"
    assert retry.change_id == undo_change_id
    assert await store.count_active(owner_id="alpha") == 3


async def _rewrite_before_records(
    store: PostgresMemoryStore, forgotten: Any, before_records: list[dict[str, Any]]
) -> None:
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "UPDATE dlightrag_memory_operations SET before_records = $2::jsonb "
            "WHERE owner_id = 'alpha' AND change_id = $1",
            uuid.UUID(forgotten.change_id),
            json.dumps(before_records),
        )


async def _journal_before(store: PostgresMemoryStore, forgotten: Any) -> list[dict[str, Any]]:
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        value = await conn.fetchval(
            "SELECT before_records FROM dlightrag_memory_operations "
            "WHERE owner_id = 'alpha' AND change_id = $1",
            uuid.UUID(forgotten.change_id),
        )
    return json.loads(value) if isinstance(value, str) else value


async def test_pg_multi_row_forget_undo_malformed_journal_owner_conflicts_cleanly(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    forgotten, (first, second, _third) = await _forget_duplicate_rows(memory, store)
    before = await _journal_before(store, forgotten)
    before[1]["owner_id"] = "beta"
    await _rewrite_before_records(store, forgotten, before)

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 1  # only r3 remains active
    assert await store.count_active(owner_id="beta") == 0
    first_row = await store.get(owner_id="alpha", memory_id=first.memory_id)
    second_row = await store.get(owner_id="alpha", memory_id=second.memory_id)
    assert first_row is not None and first_row.status == "forgotten"
    assert second_row is not None and second_row.status == "forgotten"
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_records "
                "WHERE owner_id = 'alpha' AND origin_kind = 'undo'"
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT undone_by IS NULL FROM dlightrag_memory_operations "
                "WHERE owner_id = 'alpha' AND change_id = $1",
                uuid.UUID(forgotten.change_id),
            )
            is True
        )


async def test_pg_multi_row_forget_undo_malformed_journal_duplicate_ids_conflict_cleanly(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    forgotten, (first, second, _third) = await _forget_duplicate_rows(memory, store)
    before = await _journal_before(store, forgotten)
    before[1]["memory_id"] = before[0]["memory_id"]
    await _rewrite_before_records(store, forgotten, before)

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 1  # only r3 remains active
    first_row = await store.get(owner_id="alpha", memory_id=first.memory_id)
    second_row = await store.get(owner_id="alpha", memory_id=second.memory_id)
    assert first_row is not None and first_row.status == "forgotten"
    assert second_row is not None and second_row.status == "forgotten"
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_records "
                "WHERE owner_id = 'alpha' AND origin_kind = 'undo'"
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT undone_by IS NULL FROM dlightrag_memory_operations "
                "WHERE owner_id = 'alpha' AND change_id = $1",
                uuid.UUID(forgotten.change_id),
            )
            is True
        )


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda before: [],
        lambda before: before[:1],
        lambda before: [before[1], before[0]],
        lambda before: [*before, {**before[0], "memory_id": str(uuid.uuid4())}],
    ],
    ids=["empty", "truncated", "reordered", "extra"],
)
async def test_pg_multi_row_forget_undo_malformed_journal_batch_conflicts_cleanly(
    store: PostgresMemoryStore,
    corrupt: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
) -> None:
    memory = Memory(store)
    forgotten, (first, second, _third) = await _forget_duplicate_rows(memory, store)
    original = await _journal_before(store, forgotten)
    await _rewrite_before_records(store, forgotten, corrupt(list(original)))

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 1  # only r3 remains active
    first_row = await store.get(owner_id="alpha", memory_id=first.memory_id)
    second_row = await store.get(owner_id="alpha", memory_id=second.memory_id)
    assert first_row is not None and first_row.status == "forgotten"
    assert second_row is not None and second_row.status == "forgotten"
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_records "
                "WHERE owner_id = 'alpha' AND origin_kind = 'undo'"
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT undone_by IS NULL FROM dlightrag_memory_operations "
                "WHERE owner_id = 'alpha' AND change_id = $1",
                uuid.UUID(forgotten.change_id),
            )
            is True
        )

    # Repairing the journal leaves the same target settleable.
    await _rewrite_before_records(store, forgotten, original)
    retry = await _undo(memory, forgotten.change_id, key="undo-2")
    assert retry.outcome == "changed"
    assert await store.count_active(owner_id="alpha") == 3


@pytest.mark.parametrize(
    "run_id, session_id",
    [("", None), (None, "")],
    ids=["empty-run-id", "empty-session-id"],
)
async def test_pg_multi_row_forget_undo_preserves_exact_provenance(
    store: PostgresMemoryStore, run_id: str | None, session_id: str | None
) -> None:
    memory = Memory(store)
    forgotten, _rows = await _forget_duplicate_rows(memory, store)
    provenance = MemoryProvenance(
        origin_kind="undo", origin_id="undo-1", run_id=run_id, session_id=session_id
    )

    undone = await memory.undo(
        owner_id="alpha",
        change_id=forgotten.change_id,
        provenance=provenance,
        idempotency_key="undo-1",
    )

    assert undone.outcome == "changed"
    restored = [
        record
        for memory_id in undone.memory_ids
        if (record := await store.get(owner_id="alpha", memory_id=memory_id)) is not None
    ]
    assert len(restored) == 2
    assert all(record.provenance == provenance for record in restored)
    assert all(record.provenance.origin_kind == "undo" for record in restored)


async def test_pg_multi_row_forget_undo_db_calls_are_constant(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    counting_pool = _CountingPool(store._operation_pool)
    counting_memory = Memory(PostgresMemoryStore(pool=counting_pool))

    large_calls: list[tuple[Any, ...]] = []
    for size, body in ((2, "Small batch."), (1000, "Bulk preference.")):
        rows = [_record(body=body) for _ in range(size)]
        async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
            await conn.executemany(
                "INSERT INTO dlightrag_memory_records "
                "(owner_id, memory_id, kind, body, normalized_body, origin_kind, origin_id, "
                "status) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                [
                    (
                        record.owner_id,
                        uuid.UUID(record.memory_id),
                        record.kind,
                        record.body,
                        normalized_body(record.body),
                        record.provenance.origin_kind,
                        record.provenance.origin_id,
                        record.status,
                    )
                    for record in rows
                ],
            )
        forgotten = await memory.forget(
            owner_id="alpha",
            body=body,
            provenance=_provenance(),
            idempotency_key=f"forget-{size}",
        )
        assert forgotten.outcome == "changed"
        assert len(forgotten.memory_ids) == size

        counting_pool.calls.clear()
        undone = await counting_memory.undo(
            owner_id="alpha",
            change_id=forgotten.change_id,
            provenance=MemoryProvenance(origin_kind="undo", origin_id=f"undo-{size}"),
            idempotency_key=f"undo-{size}",
        )
        assert undone.outcome == "changed"
        assert len(undone.memory_ids) == size

        calls = list(counting_pool.calls)
        assert [call[0] for call in calls] == [
            "fetchval",  # owner advisory lock
            "fetchrow",  # idempotency replay check
            "fetchrow",  # target operation row
            "fetch",  # set-wise target row fetch/lock
            "fetchval",  # set-wise active normalized-body conflict check
            "fetch",  # one set-wise INSERT ... RETURNING statement
            "execute",  # operation journal insert
            "execute",  # undone_by mark
        ]
        assert [call[:2] for call in calls] == [
            ("fetchval", "SELECT"),
            ("fetchrow", "SELECT"),
            ("fetchrow", "SELECT"),
            ("fetch", "SELECT"),
            ("fetchval", "SELECT"),
            ("fetch", "INSERT"),
            ("execute", "INSERT"),
            ("execute", "UPDATE"),
        ]
        # No client-side batching or per-row insert loop is possible: the
        # proxy exposes no executemany, and exactly one statement inserts into
        # dlightrag_memory_records for any batch size.
        assert not any(call[0] == "executemany" for call in calls)
        record_inserts = [
            call for call in calls if "INSERT INTO dlightrag_memory_records" in call[2]
        ]
        assert len(record_inserts) == 1
        insert = record_inserts[0]
        assert insert[0] == "fetch" and "RETURNING" in insert[2].upper()
        assert "jsonb_array_elements" in insert[2]  # one recordset, not N rows of params
        assert len(json.loads(insert[3][5])) == size  # the $6::jsonb recordset holds the batch
        record_executes = [
            call for call in calls if call[0] == "execute" and "dlightrag_memory_records" in call[2]
        ]
        assert record_executes == []
        if size == 1000:
            large_calls = calls

        # Every restored id maps back to its slot in the forget batch order.
        for index in (0, 1, size - 1):
            restored = await store.get(owner_id="alpha", memory_id=undone.memory_ids[index])
            assert restored is not None
            assert restored.supersedes_id == forgotten.memory_ids[index]
            assert restored.provenance.origin_kind == "undo"

    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_records "
                "WHERE owner_id = 'alpha' AND origin_kind = 'undo' AND status = 'active' "
                "AND body = 'Bulk preference.'"
            )
            == 1000
        )
    assert [call[:2] for call in large_calls] == [
        ("fetchval", "SELECT"),
        ("fetchrow", "SELECT"),
        ("fetchrow", "SELECT"),
        ("fetch", "SELECT"),
        ("fetchval", "SELECT"),
        ("fetch", "INSERT"),
        ("execute", "INSERT"),
        ("execute", "UPDATE"),
    ]


async def test_pg_concurrent_multi_row_undo_has_one_winner(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    forgotten, _rows = await _forget_duplicate_rows(memory, store)

    first, second = await asyncio.gather(
        _undo(memory, forgotten.change_id, key="undo-a"),
        _undo(memory, forgotten.change_id, key="undo-b"),
    )

    assert sorted((first.outcome, second.outcome)) == ["changed", "conflict"]
    assert await store.count_active(owner_id="alpha") == 3  # r3 plus exactly one restored pair
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_records "
                "WHERE owner_id = 'alpha' AND origin_kind = 'undo' AND status = 'active'"
            )
            == 2
        )
        assert (
            await conn.fetchval(
                "SELECT undone_by IS NOT NULL FROM dlightrag_memory_operations "
                "WHERE owner_id = 'alpha' AND change_id = $1",
                uuid.UUID(forgotten.change_id),
            )
            is True
        )

    # A later repeated undo still conflicts via the undone_by mark.
    third = await _undo(memory, forgotten.change_id, key="undo-c")
    assert third.outcome == "conflict"


async def test_pg_list_active_page_traverses_ties_and_over_hundred_rows(
    store: PostgresMemoryStore,
) -> None:
    """Full newest-first traversal: same-timestamp ties, owner isolation, bounds."""
    anchor = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
    records: list[MemoryRecord] = []
    for index in range(60):
        records.append(
            MemoryRecord(
                owner_id="alpha",
                memory_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"tie-a-{index}")),
                kind="preference",
                body=f"Tie A {index}.",
                provenance=_provenance(),
                created_at=anchor,
                updated_at=anchor,
            )
        )
    for index in range(50):
        records.append(
            MemoryRecord(
                owner_id="alpha",
                memory_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"tie-b-{index}")),
                kind="preference",
                body=f"Tie B {index}.",
                provenance=_provenance(),
                created_at=anchor - timedelta(hours=1),
                updated_at=anchor - timedelta(hours=1),
            )
        )
    for record in records:
        await store.insert(record)
    await store.insert(_record(owner="beta", body="Foreign."))

    def _key(record: MemoryRecord) -> tuple[datetime, str]:
        assert record.updated_at is not None
        return (record.updated_at, record.memory_id)

    expected = [_key(record) for record in sorted(records, key=_key, reverse=True)]
    observed: list[tuple[datetime, str]] = []
    after: tuple[datetime, str] | None = None
    while True:
        page, next_after = await store.list_active_page(owner_id="alpha", after=after, limit=40)
        assert len(page) <= 40
        observed.extend(_key(record) for record in page)
        if next_after is None:
            break
        assert page
        after = next_after
    assert observed == expected
    assert len(observed) == 110
    assert len(set(observed)) == 110

    # The exact paged-read index exists and matches the mixed-direction order.
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        indexdef = await conn.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_dlightrag_memory_records_list'"
        )
        assert indexdef is not None
        normalized = " ".join(str(indexdef).split()).lower()
        assert "(owner_id, status, updated_at desc, memory_id desc)" in normalized
