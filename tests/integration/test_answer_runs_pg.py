# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for durable Answer run state on PostgreSQL 18.

Exercises the real contract against a live database: the declared schema and its
constraints, owner-scoped creation and idempotency, queued cancellation, slot-safe
claiming across workers, lease fencing, gap-free event sequences, Session
compare-and-set, terminal transitions, graceful requeue, the crash-recovery bound,
event trimming, retention pruning, and ownership-safe artifact cleanup.

Every test runs inside a throwaway database created and dropped per test, so the
configured administrative database is never mutated. Connection settings come
from the shared integration-test PostgreSQL environment; skipped if unavailable.
"""

import asyncio
import hashlib
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.answer.workspace import PGWorkspaceStore
from dlightrag.adapters.postgres.runtime.run_blob_store import (
    BlobSizeConflict,
    PGRunBlobStore,
    write_blob_content,
    write_complete_blob,
)
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.engine.agent.session.ids import StageIntentId
from dlightrag.engine.runtime import (
    MAX_RECLAIMS_WITHOUT_PROGRESS,
    RUN_ABANDONED_ERROR_KIND,
    HandoffCommit,
    IdempotencyKeyConflict,
    PendingArtifact,
    PendingArtifactReference,
    RunAdmissionLimitExceededError,
    StageTerminalCommit,
    run_request_fingerprint,
)
from dlightrag.engine.runtime.blob_chunks import BLOB_CHUNK_BYTES
from dlightrag.engine.runtime.records import PendingPublication
from tests.conftest import FingerprintingRunStore
from tests.integration.pg_conn import PG_CONN_KWARGS

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_PG_CONN_KWARGS: dict[str, Any] = PG_CONN_KWARGS

_OWNER = "owner-alpha"
_OTHER_OWNER = "owner-beta"
_WORKER = "worker-1"
_ABANDONED_ERROR_MESSAGE = "Run exceeded its reclaim-without-progress bound."

# Verbatim deployed Answer-run tables from baseline main@5c66e5b2. In
# particular, accepted_input_json is already a required column; this fixture
# guards the supported terminal-only rename without inventing an older schema.
_BASELINE_ANSWER_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_runs (
    owner_id            TEXT        NOT NULL,
    run_id              UUID        NOT NULL,
    idempotency_key     TEXT,
    prepared_input_json JSONB,
    accepted_input_json JSONB        NOT NULL DEFAULT '{}'::jsonb,
    request_fingerprint TEXT        NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'queued',
    phase               TEXT,
    stop_reason         TEXT,
    cancel_requested_at TIMESTAMPTZ,
    lease_owner         TEXT,
    lease_expires_at    TIMESTAMPTZ,
    fencing_epoch       BIGINT      NOT NULL DEFAULT 0,
    durable_progress_version       BIGINT  NOT NULL DEFAULT 0,
    last_reclaim_progress_version  BIGINT  NOT NULL DEFAULT 0,
    reclaims_without_progress      INTEGER NOT NULL DEFAULT 0,
    next_event_sequence BIGINT      NOT NULL DEFAULT 1,
    events_trimmed_at   TIMESTAMPTZ,
    result_json         JSONB,
    error_kind          TEXT,
    error_message       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    workspace_epoch     BIGINT,
    PRIMARY KEY (owner_id, run_id),
    CONSTRAINT dlightrag_answer_runs_status_check
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    CONSTRAINT dlightrag_answer_runs_phase_check
        CHECK (phase IS NULL OR phase IN ('routing', 'planning', 'searching', 'researching', 'generating')),
    CONSTRAINT dlightrag_answer_runs_counter_check
        CHECK (fencing_epoch >= 0 AND next_event_sequence >= 1
               AND durable_progress_version >= 0
               AND last_reclaim_progress_version >= 0
               AND reclaims_without_progress >= 0),
    CONSTRAINT dlightrag_answer_runs_lease_check
        CHECK ((lease_owner IS NULL) = (lease_expires_at IS NULL)),
    CONSTRAINT dlightrag_answer_runs_terminal_check
        CHECK ((status IN ('succeeded', 'failed', 'cancelled')) = (finished_at IS NOT NULL)),
    CONSTRAINT dlightrag_answer_runs_result_check
        CHECK (status <> 'succeeded' OR result_json IS NOT NULL),
    CONSTRAINT dlightrag_answer_runs_error_check
        CHECK ((status = 'failed') = (error_kind IS NOT NULL)),
    CONSTRAINT dlightrag_answer_runs_prepared_input_check
        CHECK ((status IN ('queued', 'running')) = (prepared_input_json IS NOT NULL)),
    CONSTRAINT dlightrag_answer_runs_workspace_epoch_check
        CHECK (workspace_epoch IS NULL OR workspace_epoch >= 1)
)
"""
_BASELINE_ANSWER_RUN_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_run_events (
    owner_id       TEXT        NOT NULL,
    run_id         UUID        NOT NULL,
    event_sequence BIGINT      NOT NULL,
    event_type     TEXT        NOT NULL,
    payload        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, event_sequence),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_answer_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_run_events_type_check
        CHECK (event_type IN (
            'progress', 'token', 'reset',
            'tool_start', 'tool_progress', 'tool_end',
            'memory_operation_settled', 'done', 'error'
        )),
    CONSTRAINT dlightrag_answer_run_events_sequence_check
        CHECK (event_sequence >= 1)
)
"""


def _blob_store(store: PGRunStore) -> PGRunBlobStore:
    pool = store._operation_pool  # noqa: SLF001 - bind the separate blob seam in tests
    assert pool is not None
    return PGRunBlobStore(pool=pool)


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    """Provision an isolated throwaway database and yield a pool bound to it."""
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")

    db_name = f"dlightrag_runs_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    created = await asyncpg.create_pool(
        **{**_PG_CONN_KWARGS, "database": db_name}, min_size=1, max_size=8
    )
    try:
        yield created
    finally:
        await created.close()
        admin = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


@pytest.fixture
async def store(pool: Any) -> PGRunStore:
    created = FingerprintingRunStore(pool=pool)
    await created.initialize()
    # Establish the complete operational schema exactly as a real process does.
    from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore

    await PGWebConversationStore(pool=pool, run_store=created).initialize()
    return created


def _request(query: str = "why", **extra: Any) -> dict[str, Any]:
    return {
        "query": query,
        "workspaces": ["alpha"],
        "agent_session_id": "00000000-0000-7000-8000-000000000001",
        "agent_lane_id": "main",
        **extra,
    }


async def _expire_lease(pool: Any, run_id: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs "
            "SET lease_expires_at = NOW() - INTERVAL '1 second' WHERE run_id = $1",
            uuid.UUID(run_id),
        )


async def _backdate_finish(pool: Any, run_id: str, *, days: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs "
            "SET finished_at = NOW() - ($2 * INTERVAL '1 day'), "
            "purge_after = NOW() - ($2 * INTERVAL '1 day') "
            "+ make_interval(secs => retention_seconds::double precision) WHERE run_id = $1",
            uuid.UUID(run_id),
            days,
        )


async def _event_types(pool: Any, run_id: str) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT event_type FROM dlightrag_run_events WHERE run_id = $1 ORDER BY event_sequence",
            uuid.UUID(run_id),
        )
    return [str(row["event_type"]) for row in rows]


async def _prepared_input(pool: Any, run_id: str) -> str | None:
    """Read the raw prepared input column."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT prepared_input_json FROM dlightrag_runs WHERE run_id = $1",
            uuid.UUID(run_id),
        )


async def _write_fetched_blob(pool: Any, run_id: str, content: bytes) -> str:
    """Write complete fetched bytes and their run-owned resource reference."""
    digest = hashlib.sha256(content).hexdigest()
    locator = f"https://example.test/{digest[:12]}".encode()
    async with pool.acquire() as conn, conn.transaction():
        await write_blob_content(
            conn,
            owner_id=_OWNER,
            digest=digest,
            content=content,
        )
        await conn.execute(
            "INSERT INTO dlightrag_answer_resources "
            "(owner_id, run_id, resource_id, kind, safe_name, media_type, capabilities, "
            "ordinal, blob_digest, locator_digest, source_locator) "
            "VALUES ($1, $2, $3, 'fetched_blob', 'page.html', 'text/html', $4::jsonb, "
            "0, $5, $6, $7)",
            _OWNER,
            uuid.UUID(run_id),
            f"fetched-{digest[:16]}",
            '{"resource_kind":"web"}',
            digest,
            hashlib.sha256(locator).hexdigest(),
            locator,
        )
    return digest


async def _claimed(store: PGRunStore, *, worker_id: str = _WORKER) -> Any:
    claim = await store.claim_next(worker_id=worker_id)
    assert claim is not None
    return claim


def _delete_action(value: Any) -> str:
    """``pg_constraint.confdeltype`` is a one-byte ``"char"`` column."""
    return value.decode() if isinstance(value, bytes | bytearray) else str(value)


async def _assert_run_event_parent_guard(store: FingerprintingRunStore, pool: Any) -> None:
    """Exercise the direct-DML event invariants on either schema path."""
    async with pool.acquire() as conn:
        trigger_rows = await conn.fetch(
            "SELECT trigger_name, event_manipulation, action_timing "
            "FROM information_schema.triggers "
            "WHERE trigger_schema = 'public' "
            "AND event_object_table = 'dlightrag_run_events'"
        )
    assert {
        (
            str(row["trigger_name"]),
            str(row["event_manipulation"]),
            str(row["action_timing"]),
        )
        for row in trigger_rows
    } == {
        ("trg_dlightrag_run_events_enforce", "INSERT", "BEFORE"),
        ("trg_dlightrag_run_events_enforce", "UPDATE", "BEFORE"),
    }

    async def remove(run_id: str) -> None:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM dlightrag_runs WHERE run_id = $1", uuid.UUID(run_id))

    # Even a live lease cannot publish a sequence the parent has not allocated.
    creation = await store.create_run(owner_id=_OWNER, request=_request("future sequence"))
    await _claimed(store)
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'progress', '{}'::jsonb)",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
        await conn.execute(
            "UPDATE dlightrag_runs SET next_event_sequence = 2 WHERE run_id = $1",
            uuid.UUID(creation.run.run_id),
        )
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'progress', '[]'::jsonb)",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
    await remove(creation.run.run_id)

    # Nonterminal events require a running parent with a present, live lease.
    creation = await store.create_run(owner_id=_OWNER, request=_request("queued event"))
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs SET next_event_sequence = 2 WHERE run_id = $1",
            uuid.UUID(creation.run.run_id),
        )
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'progress', '{}'::jsonb)",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
    await remove(creation.run.run_id)

    creation = await store.create_run(owner_id=_OWNER, request=_request("unleased event"))
    await _claimed(store)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs SET lease_owner = NULL, "
            "lease_expires_at = NULL, next_event_sequence = 2 WHERE run_id = $1",
            uuid.UUID(creation.run.run_id),
        )
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'progress', '{}'::jsonb)",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
    await remove(creation.run.run_id)

    creation = await store.create_run(owner_id=_OWNER, request=_request("expired event"))
    await _claimed(store)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs SET lease_expires_at = NOW() - INTERVAL '1 second', "
            "next_event_sequence = 2 WHERE run_id = $1",
            uuid.UUID(creation.run.run_id),
        )
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'progress', '{}'::jsonb)",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
    await remove(creation.run.run_id)

    # Terminal events are admitted only after the matching parent transition.
    creation = await store.create_run(owner_id=_OWNER, request=_request("early terminal"))
    await _claimed(store)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs SET next_event_sequence = 2 WHERE run_id = $1",
            uuid.UUID(creation.run.run_id),
        )
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'done', "
                '\'{"status":"succeeded","result":{}}\'::jsonb)',
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
    await remove(creation.run.run_id)

    creation = await store.create_run(owner_id=_OWNER, request=_request("terminal mismatch"))
    claim = await _claimed(store)
    assert (
        await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "complete"},
        )
    ).committed
    async with pool.acquire() as conn:
        # Cleanup and retention DELETEs remain legal; the guard covers only writes.
        assert (
            await conn.execute(
                "DELETE FROM dlightrag_run_events WHERE owner_id = $1 AND run_id = $2",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
        ) == "DELETE 1"
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'done', "
                '\'{"status":"succeeded","result":{"answer":"wrong"}}\'::jsonb)',
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'error', "
                '\'{"kind":"provider_error","message":"boom"}\'::jsonb)',
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
    await remove(creation.run.run_id)

    # UPDATEs pass through the same parent-sequence guard.
    creation = await store.create_run(owner_id=_OWNER, request=_request("event update"))
    claim = await _claimed(store)
    assert (
        await store.append_event(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            phase=None,
            event_type="progress",
            payload={"phase": "working"},
        )
    ) == 1
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "UPDATE dlightrag_run_events SET event_sequence = 2 "
                "WHERE owner_id = $1 AND run_id = $2",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
        assert (
            await conn.execute(
                "DELETE FROM dlightrag_run_events WHERE owner_id = $1 AND run_id = $2",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
        ) == "DELETE 1"
    await remove(creation.run.run_id)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    async def test_migrates_the_deployed_terminal_answer_schema_and_remains_writable(
        self, pool
    ) -> None:
        legacy_run_id = uuid.uuid7()
        accepted_input = {"query": "legacy terminal", "workspaces": ["alpha"]}
        legacy_result = {"answer": "preserved"}
        async with pool.acquire() as conn:
            await conn.execute(_BASELINE_ANSWER_RUNS_DDL)
            await conn.execute(_BASELINE_ANSWER_RUN_EVENTS_DDL)
            await conn.execute(
                "INSERT INTO dlightrag_answer_runs "
                "(owner_id, run_id, idempotency_key, accepted_input_json, "
                "request_fingerprint, status, next_event_sequence, result_json, "
                "started_at, finished_at) "
                "VALUES ($1, $2, 'legacy-terminal', $3::jsonb, 'legacy-fingerprint', "
                "'succeeded', 2, $4::jsonb, NOW(), NOW())",
                _OWNER,
                legacy_run_id,
                json.dumps(accepted_input),
                json.dumps(legacy_result),
            )
            await conn.execute(
                "INSERT INTO dlightrag_answer_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'done', $3::jsonb)",
                _OWNER,
                legacy_run_id,
                json.dumps({"status": "succeeded", "result": legacy_result}),
            )

        migrated = FingerprintingRunStore(pool=pool)
        await migrated.initialize()

        legacy = await migrated.get_run(owner_id=_OWNER, run_id=str(legacy_run_id))
        assert legacy is not None
        assert legacy.status == "succeeded"
        assert legacy.accepted_input == accepted_input
        assert legacy.result == legacy_result
        legacy_events = await migrated.read_event_page(owner_id=_OWNER, run_id=str(legacy_run_id))
        assert [(event.event_type, event.payload) for event in legacy_events] == [
            ("done", {"status": "succeeded", "result": legacy_result})
        ]

        inserted = await migrated.create_run(owner_id=_OWNER, request=_request("after migration"))
        claim = await _claimed(migrated)
        assert claim.run.run_id == inserted.run.run_id
        outcome = await migrated.finish_success(
            owner_id=_OWNER,
            run_id=inserted.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "new"},
        )
        assert outcome.committed is True
        assert (await migrated.get_run(owner_id=_OWNER, run_id=inserted.run.run_id)) is not None
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT to_regclass('dlightrag_answer_runs')") is None
            assert await conn.fetchval("SELECT to_regclass('dlightrag_runs')") == "dlightrag_runs"
            event_checks = {
                str(row["conname"])
                for row in await conn.fetch(
                    "SELECT conname FROM pg_constraint "
                    "WHERE conrelid = 'dlightrag_run_events'::regclass AND contype = 'c'"
                )
            }
        assert "dlightrag_run_events_sequence_check" in event_checks
        assert "dlightrag_answer_run_events_sequence_check" not in event_checks

        reader = PGRunStore(pool=pool)
        await reader.initialize(validate_only=True)
        await _assert_run_event_parent_guard(migrated, pool)

    async def test_fresh_schema_enforces_run_event_parent_contract(self, store, pool) -> None:
        await _assert_run_event_parent_guard(store, pool)

    async def test_current_schema_migration_drops_active_permit(self, store, pool) -> None:
        async with pool.acquire() as conn:
            await conn.execute(
                "ALTER TABLE dlightrag_runs "
                "ADD COLUMN active_permit BOOLEAN NOT NULL DEFAULT FALSE, "
                "ADD CONSTRAINT dlightrag_runs_permit_check "
                "CHECK (NOT active_permit OR (status = 'running' AND lease_owner IS NOT NULL))"
            )
            await conn.execute(
                "DELETE FROM dlightrag_schema_migrations "
                "WHERE scope = 'runs' AND version = 'remove_run_active_permit'"
            )

        migrated = PGRunStore(pool=pool)
        await migrated.initialize()

        async with pool.acquire() as conn:
            assert not await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'dlightrag_runs' AND column_name = 'active_permit')"
            )
            assert not await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_constraint "
                "WHERE conrelid = 'dlightrag_runs'::regclass "
                "AND conname = 'dlightrag_runs_permit_check')"
            )

    async def test_creates_exactly_the_answer_schema_tables(self, store, pool) -> None:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name LIKE 'dlightrag_%'"
            )
        assert {str(row["table_name"]) for row in rows} - {"dlightrag_schema_migrations"} == {
            "dlightrag_runs",
            "dlightrag_run_events",
            "dlightrag_agent_sessions",
            "dlightrag_agent_session_entries",
            "dlightrag_agent_session_registers",
            "dlightrag_answer_run_stages",
            "dlightrag_answer_evidence",
            "dlightrag_answer_resources",
            "dlightrag_blobs",
            "dlightrag_blob_chunks",
            "dlightrag_answer_run_artifacts",
            "dlightrag_answer_artifact_attachments",
            "dlightrag_answer_workspace_inventory",
            "dlightrag_answer_committed_spills",
            "dlightrag_answer_run_routing",
            "dlightrag_answer_child_sessions",
            "dlightrag_agent_controls",
            "dlightrag_answer_memory_settings",
            "dlightrag_corpus_mutation_windows",
        }

    async def test_artifact_reference_constraint_uses_one_output_kind(self, store, pool) -> None:
        async with pool.acquire() as conn:
            definition = await conn.fetchval(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = 'dlightrag_answer_run_artifacts_kind_check'"
            )

        assert definition == (
            "CHECK ((reference_kind = ANY (ARRAY['current_attachment'::text, "
            "'history_attachment'::text, 'fetched_resource'::text, "
            "'published_artifact'::text])))"
        )

    async def test_bounded_operational_scans_use_their_ordered_indexes(
        self,
        store,
        pool,
    ) -> None:
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute("SET LOCAL enable_seqscan = off")
            active_plan = await conn.fetch(
                "EXPLAIN (COSTS OFF) "
                "SELECT created_at, run_id, "
                "prepared_input_json ->> 'context_policy_revision', "
                "prepared_input_json -> 'pinned_models' "
                "FROM dlightrag_runs "
                "WHERE status IN ('queued', 'running') "
                "AND cancel_requested_at IS NULL "
                "AND NOT (status = 'running' AND lease_expires_at < NOW() "
                "         AND reclaims_without_progress >= $1) "
                "ORDER BY created_at, run_id LIMIT 200",
                MAX_RECLAIMS_WITHOUT_PROGRESS,
            )
            cancel_plan = await conn.fetch(
                "EXPLAIN (COSTS OFF) "
                "SELECT owner_id, run_id, created_at "
                "FROM dlightrag_runs "
                "WHERE cancel_requested_at IS NOT NULL "
                "AND status = 'running' AND lease_owner = $1 "
                "AND lease_expires_at > NOW() "
                "ORDER BY created_at, run_id LIMIT 200",
                _WORKER,
            )

        assert "idx_dlightrag_runs_claim" in "\n".join(
            str(row["QUERY PLAN"]) for row in active_plan
        )
        assert "idx_dlightrag_runs_cancel_pending" in "\n".join(
            str(row["QUERY PLAN"]) for row in cancel_plan
        )

    async def test_memory_settings_default_and_roundtrip(self, store, pool) -> None:
        """Enablement defaults on for absent rows and persists across updates."""
        from dlightrag.adapters.postgres.answer.memory_settings import PGMemorySettingsStore

        settings = PGMemorySettingsStore(pool=pool)

        assert (await settings.state(owner_id="alpha")).enabled is True
        disabled = await settings.set_enabled(owner_id="alpha", enabled=False)
        assert disabled.enabled is False
        assert disabled.epoch == 1
        assert (await settings.state(owner_id="beta")).enabled is True
        enabled = await settings.set_enabled(owner_id="alpha", enabled=True)
        assert enabled.enabled is True
        assert enabled.epoch == 1
        bumped = await settings.bump_epoch(owner_id="alpha")
        assert bumped.epoch == 2

    async def test_run_columns_match_the_contract(self, store, pool) -> None:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = $1",
                "dlightrag_runs",
            )
        assert {str(row["column_name"]) for row in rows} == {
            "owner_id",
            "run_id",
            "run_kind",
            "lane",
            "submitted_by",
            "access_scope_kind",
            "submission_key",
            "prepared_input_json",
            "accepted_input_json",
            "request_fingerprint",
            "status",
            "phase",
            "stop_reason",
            "cancel_requested_at",
            "lease_owner",
            "lease_expires_at",
            "fencing_epoch",
            "durable_progress_version",
            "last_reclaim_progress_version",
            "reclaims_without_progress",
            "next_event_sequence",
            "events_trimmed_at",
            "result_json",
            "error_kind",
            "error_message",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
            "agent_workspace_epoch",
            "retention_seconds",
            "purge_after",
            "next_attempt_at",
            "checkpoint_json",
            "handoff_started_at",
            "superseded_by_run_id",
        }

    async def test_foreign_keys_cascade_runs_and_restrict_blobs(self, store, pool) -> None:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.conrelid::regclass::text AS child,
                       c.confrelid::regclass::text AS parent,
                       c.confdeltype AS on_delete
                FROM pg_constraint AS c
                WHERE c.contype = 'f'
                  AND c.conrelid::regclass::text LIKE 'dlightrag_%'
                """
            )
        actions = {
            (str(row["child"]), str(row["parent"])): _delete_action(row["on_delete"])
            for row in rows
        }
        assert actions[("dlightrag_run_events", "dlightrag_runs")] == "c"
        assert actions[("dlightrag_answer_run_artifacts", "dlightrag_runs")] == "c"
        assert actions[("dlightrag_answer_run_artifacts", "dlightrag_blobs")] == "r"

    async def test_rejects_unknown_status_but_accepts_executor_owned_labels(
        self, store, pool
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        run_id = uuid.UUID(creation.run.run_id)
        async with pool.acquire() as conn:
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "UPDATE dlightrag_runs SET status = 'paused' WHERE run_id = $1", run_id
                )
            await conn.execute(
                "UPDATE dlightrag_runs SET phase = 'polishing' WHERE run_id = $1", run_id
            )
        claim = await _claimed(store)
        assert (
            await store.append_event(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                phase=None,
                event_type="thinking",
                payload={},
            )
        ) == 1

    async def test_allows_only_one_terminal_event_per_run(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        assert (
            await store.finish_success(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                result={"answer": "complete"},
            )
        ).committed
        run_id = uuid.UUID(creation.run.run_id)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_runs SET next_event_sequence = 3 WHERE run_id = $1",
                run_id,
            )
            with pytest.raises(asyncpg.exceptions.UniqueViolationError):
                await conn.execute(
                    "INSERT INTO dlightrag_run_events "
                    "(owner_id, run_id, event_sequence, event_type, payload) "
                    "VALUES ($1, $2, 2, 'done', "
                    '\'{"status":"succeeded","result":{"answer":"complete"}}\'::jsonb)',
                    _OWNER,
                    run_id,
                )


# ---------------------------------------------------------------------------
# Creation, idempotency, owner scoping
# ---------------------------------------------------------------------------


class TestCreation:
    async def test_nonterminal_admission_limit_is_atomic_across_submitters(self, pool) -> None:
        first_store = FingerprintingRunStore(pool=pool, query_max_nonterminal_runs=1)
        second_store = FingerprintingRunStore(pool=pool, query_max_nonterminal_runs=1)
        await first_store.initialize()
        results = await asyncio.gather(
            first_store.create_run(owner_id=_OWNER, request=_request("a")),
            second_store.create_run(owner_id=_OTHER_OWNER, request=_request("b")),
            return_exceptions=True,
        )
        assert sum(isinstance(result, RunAdmissionLimitExceededError) for result in results) == 1
        assert sum(not isinstance(result, BaseException) for result in results) == 1

    async def test_creates_queued_run_with_uuid7_identity(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        assert creation.replayed is False
        assert uuid.UUID(creation.run.run_id).version == 7
        assert creation.run.status == "queued"
        assert creation.run.next_event_sequence == 1
        assert creation.run.prepared_input["query"] == "why"

    async def test_run_ids_are_unique_and_time_ordered(self, store) -> None:
        run_ids = [
            uuid.UUID((await store.create_run(owner_id=_OWNER, request=_request())).run.run_id)
            for _ in range(16)
        ]
        assert run_ids == sorted(run_ids)
        assert len(set(run_ids)) == len(run_ids)

    async def test_replays_same_key_and_input(self, store) -> None:
        first = await store.create_run(owner_id=_OWNER, request=_request(), idempotency_key="k1")
        second = await store.create_run(owner_id=_OWNER, request=_request(), idempotency_key="k1")
        assert second.replayed is True
        assert second.run.run_id == first.run.run_id
        lookup = await store.replay_run(
            owner_id=_OWNER,
            idempotency_key="k1",
            idempotency_fingerprint=run_request_fingerprint(_request()),
            run_kind="answer",
        )
        assert lookup is not None
        assert lookup.replayed is True
        assert lookup.run.run_id == first.run.run_id

    async def test_replay_returns_current_status_not_queued(self, store) -> None:
        first = await store.create_run(owner_id=_OWNER, request=_request(), idempotency_key="k1")
        await _claimed(store)
        replay = await store.create_run(owner_id=_OWNER, request=_request(), idempotency_key="k1")
        assert replay.run.run_id == first.run.run_id
        assert replay.run.status == "running"

    async def test_rejects_same_key_with_different_input(self, store) -> None:
        await store.create_run(owner_id=_OWNER, request=_request(), idempotency_key="k1")
        with pytest.raises(IdempotencyKeyConflict):
            await store.create_run(owner_id=_OWNER, request=_request("other"), idempotency_key="k1")

    async def test_replay_normalizes_key_order_but_not_list_order(self, store) -> None:
        first = await store.create_run(
            owner_id=_OWNER,
            request={"query": "why", "workspaces": ["alpha", "beta"]},
            idempotency_key="k1",
        )
        reordered = await store.create_run(
            owner_id=_OWNER,
            request={"workspaces": ["alpha", "beta"], "query": "why"},
            idempotency_key="k1",
        )
        assert reordered.replayed is True
        assert reordered.run.run_id == first.run.run_id
        with pytest.raises(IdempotencyKeyConflict):
            await store.create_run(
                owner_id=_OWNER,
                request={"query": "why", "workspaces": ["beta", "alpha"]},
                idempotency_key="k1",
            )

    async def test_scopes_idempotency_keys_per_owner(self, store) -> None:
        first = await store.create_run(owner_id=_OWNER, request=_request(), idempotency_key="k1")
        second = await store.create_run(
            owner_id=_OTHER_OWNER, request=_request(), idempotency_key="k1"
        )
        assert second.replayed is False
        assert second.run.run_id != first.run.run_id

    async def test_creation_without_key_always_creates_a_new_run(self, store) -> None:
        first = await store.create_run(owner_id=_OWNER, request=_request())
        second = await store.create_run(owner_id=_OWNER, request=_request())
        assert first.run.run_id != second.run.run_id

    async def test_foreign_owner_and_unknown_ids_read_identically(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        assert await store.get_run(owner_id=_OTHER_OWNER, run_id=creation.run.run_id) is None
        assert await store.get_run(owner_id=_OWNER, run_id=str(uuid.uuid7())) is None
        assert await store.get_run(owner_id=_OWNER, run_id="not-a-uuid") is None
        assert await store.read_event_page(owner_id=_OTHER_OWNER, run_id=creation.run.run_id) == ()

    async def test_rejects_empty_owner(self, store) -> None:
        with pytest.raises(ValueError):
            await store.create_run(owner_id="  ", request=_request())


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    async def test_queued_run_cancels_in_one_transaction(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        outcome = await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        assert outcome.outcome == "cancelled"
        assert outcome.run is not None
        assert outcome.run.status == "cancelled"
        assert outcome.run.finished_at is not None
        assert await _event_types(pool, creation.run.run_id) == ["done"]
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert events[0].sequence == 1
        assert events[0].payload == {"status": "cancelled"}

    async def test_running_run_records_pending_request(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        outcome = await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        assert outcome.outcome == "pending"
        assert outcome.run is not None
        assert outcome.run.status == "running"
        assert outcome.run.cancel_requested is True

    async def test_queued_cancellation_racing_a_claim_reports_pending(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        async with pool.acquire() as claiming, claiming.transaction():
            await claiming.execute(
                "UPDATE dlightrag_runs SET status = 'running', lease_owner = 'claiming-worker', "
                "lease_expires_at = NOW() + INTERVAL '30 seconds', "
                "fencing_epoch = fencing_epoch + 1, started_at = NOW(), "
                "updated_at = NOW() "
                "WHERE owner_id = $1 AND run_id = $2",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
            cancellation = asyncio.create_task(
                store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
            )
            await asyncio.sleep(0.05)
            assert not cancellation.done()

        outcome = await asyncio.wait_for(cancellation, timeout=5)
        assert outcome.outcome == "pending"
        assert outcome.run is not None
        assert outcome.run.status == "running"
        assert outcome.run.cancel_requested is True

    async def test_cancelling_a_terminal_run_is_a_no_op(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        repeat = await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        assert repeat.outcome == "already_terminal"
        assert repeat.run is not None
        assert repeat.run.status == "cancelled"
        assert await _event_types(pool, creation.run.run_id) == ["done"]

    async def test_unknown_and_foreign_runs_cancel_identically(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        foreign = await store.request_cancellation(
            owner_id=_OTHER_OWNER, run_id=creation.run.run_id
        )
        unknown = await store.request_cancellation(owner_id=_OWNER, run_id=str(uuid.uuid7()))
        assert foreign == unknown
        assert foreign.outcome == "unknown"

    async def test_success_yields_to_a_cancellation_that_won_the_row(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        outcome = await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "hello"},
        )
        assert outcome.committed is True
        assert outcome.status == "cancelled"
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "cancelled"
        assert record.result is None
        assert await _event_types(pool, creation.run.run_id) == ["done"]

    async def test_failure_yields_to_a_cancellation_that_won_the_row(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        accepted = await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        assert accepted.outcome == "pending"

        outcome, repeated = await asyncio.gather(
            store.finish_failure(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                error_kind="provider_error",
                error_message="provider failed",
                result={"partial": True},
            ),
            store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id),
        )

        assert repeated.outcome in {"pending", "already_terminal"}
        assert outcome.committed is True
        assert outcome.status == "cancelled"
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "cancelled"
        assert record.result is None
        assert record.error_kind is None
        assert record.error_message is None
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(event.event_type, event.payload) for event in events] == [
            ("done", {"status": "cancelled"})
        ]

    async def test_fast_terminal_yields_when_cancellation_commits_first(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        stage_id = StageIntentId.deterministic(
            run_id=creation.run.run_id,
            name="fast:final_generation:2",
        )

        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        outcome = await claim.execution.progress_store.settle_terminal(
            expected_progress_version=0,
            stage_intent_id=stage_id,
            state={"result": {"answer": "withheld"}},
            result={"answer": "withheld"},
        )

        assert isinstance(outcome, StageTerminalCommit)
        assert outcome.status == "cancelled"
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "cancelled"
        assert record.result is None
        assert record.durable_progress_version == 0
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert len(events) == 1
        assert events[0].event_type == "done"
        assert events[0].payload == {"status": "cancelled"}

    async def test_fast_terminal_can_win_when_it_holds_the_row_first(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        stage_id = StageIntentId.deterministic(
            run_id=creation.run.run_id,
            name="fast:final_generation:2",
        )

        outcome = await claim.execution.progress_store.settle_terminal(
            expected_progress_version=0,
            stage_intent_id=stage_id,
            state={"result": {"answer": "winner"}},
            result={"answer": "winner"},
        )
        cancellation = await store.request_cancellation(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
        )

        assert isinstance(outcome, StageTerminalCommit)
        assert outcome.status == "succeeded"
        assert cancellation.outcome == "already_terminal"
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "succeeded"
        assert record.result == {"answer": "winner"}
        assert record.durable_progress_version == 1
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert len(events) == 1
        assert events[0].event_type == "done"
        assert events[0].payload == {
            "status": "succeeded",
            "result": {"answer": "winner"},
        }


# ---------------------------------------------------------------------------
# Claim, lease, fencing
# ---------------------------------------------------------------------------


class TestClaiming:
    async def test_claims_the_oldest_queued_run_first(self, store) -> None:
        first = await store.create_run(owner_id=_OWNER, request=_request("a"))
        second = await store.create_run(owner_id=_OWNER, request=_request("b"))
        claim = await _claimed(store)
        assert claim.run.run_id == first.run.run_id
        assert claim.run.status == "running"
        assert claim.run.lease_owner == _WORKER
        assert claim.run.fencing_epoch == 1
        assert claim.run.started_at is not None
        assert (await _claimed(store, worker_id="worker-2")).run.run_id == second.run.run_id

    async def test_concurrent_workers_never_share_a_row(self, store) -> None:
        await store.create_run(owner_id=_OWNER, request=_request("a"))
        await store.create_run(owner_id=_OWNER, request=_request("b"))
        claims = await asyncio.gather(
            *(store.claim_next(worker_id=f"worker-{index}") for index in range(6))
        )
        claimed = [claim for claim in claims if claim is not None]
        assert len(claimed) == 2
        assert len({claim.run.run_id for claim in claimed}) == 2

    async def test_concurrent_query_claimers_claim_every_eligible_run_once(self, store) -> None:
        for index in range(24):
            await store.create_run(owner_id=_OWNER, request=_request(f"run-{index}"))

        attempts = await asyncio.gather(
            *(store.claim_next(worker_id=f"host-{index}") for index in range(24))
        )
        claimed = [claim for claim in attempts if claim is not None]
        assert len(claimed) == 24
        assert len({claim.run.run_id for claim in claimed}) == 24
        assert await store.claim_next(worker_id="no-work-left") is None

    async def test_returns_none_when_no_row_is_eligible(self, store) -> None:
        assert await store.claim_next(worker_id=_WORKER) is None

    async def test_does_not_claim_a_live_lease(self, store) -> None:
        await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        assert await store.claim_next(worker_id="worker-2") is None

    async def test_reclaims_an_expired_lease_with_a_new_epoch(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        await _expire_lease(pool, creation.run.run_id)
        reclaim = await _claimed(store, worker_id="worker-2")
        assert reclaim.run.run_id == creation.run.run_id
        assert reclaim.run.fencing_epoch == 2
        assert reclaim.run.reclaims_without_progress == 1
        assert reclaim.run.lease_owner == "worker-2"

    async def test_claimed_workspace_handoff_persists_agent_workspace_epoch(
        self, store, pool
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        workspace = PGWorkspaceStore(
            pool=pool,
            owner_id=_OWNER,
            run_id=uuid.UUID(creation.run.run_id),
            worker_id=_WORKER,
            lease_owner=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )

        outcome = await workspace.handoff_epoch(
            expected_epoch=None,
            destination_epoch=claim.run.fencing_epoch,
            inventory=(),
        )

        assert isinstance(outcome, HandoffCommit)
        assert outcome.workspace_epoch == claim.run.fencing_epoch
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.agent_workspace_epoch == claim.run.fencing_epoch

    async def test_reclaim_abandonment_persists_the_exact_public_error(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_runs "
                "SET reclaims_without_progress = $2, lease_expires_at = NOW() - INTERVAL '1 second' "
                "WHERE run_id = $1",
                uuid.UUID(creation.run.run_id),
                MAX_RECLAIMS_WITHOUT_PROGRESS - 1,
            )

        assert await store.claim_next(worker_id="abandoning-worker") is None
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "failed"
        assert record.error_kind == RUN_ABANDONED_ERROR_KIND
        assert record.error_message == _ABANDONED_ERROR_MESSAGE
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(event.event_type, event.payload) for event in events] == [
            (
                "error",
                {
                    "kind": RUN_ABANDONED_ERROR_KIND,
                    "message": _ABANDONED_ERROR_MESSAGE,
                },
            )
        ]

    async def test_skips_cancel_pending_rows(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        assert await store.claim_next(worker_id="worker-2") is None

    async def test_racing_hosts_reclaim_one_expired_lease_once(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        await _expire_lease(pool, creation.run.run_id)
        claims = await asyncio.gather(
            *(store.claim_next(worker_id=f"host-{index}") for index in range(5))
        )
        winners = [claim for claim in claims if claim is not None]
        assert len(winners) == 1
        assert winners[0].run.fencing_epoch == 2
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.lease_owner == winners[0].run.lease_owner
        assert record.reclaims_without_progress == 1


class TestLeaseFencing:
    async def test_heartbeat_renews_and_reports_cancellation(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        renewal = await store.heartbeat(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert renewal.renewed is True
        assert renewal.cancel_requested is False
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        assert (
            await store.heartbeat(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
            )
        ).cancel_requested is True

    async def test_expired_lease_is_never_revived(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await _expire_lease(pool, creation.run.run_id)
        renewal = await store.heartbeat(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert renewal.renewed is False

    async def test_stale_worker_cannot_write_after_reclaim(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        stale = await _claimed(store)
        await _expire_lease(pool, creation.run.run_id)
        fresh = await _claimed(store, worker_id="worker-2")
        stale_args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=stale.run.fencing_epoch,
        )

        assert (await store.heartbeat(**stale_args)).renewed is False
        assert await store.record_phase(**stale_args, phase="searching") is None
        assert (
            await store.append_event(
                **stale_args, phase=None, event_type="token", payload={"text": "stale"}
            )
            is None
        )
        assert (
            await store.append_event(**stale_args, phase=None, event_type="reset", payload={})
            is None
        )
        assert (
            await store.finish_success(**stale_args, result={"answer": "stale"})
        ).committed is False
        assert (
            await store.finish_failure(
                **stale_args, error_kind="provider_error", error_message="stale"
            )
        ).committed is False
        assert await store.release_for_shutdown(**stale_args) == "lease_lost"
        from dlightrag.engine.agent.session.ids import LaneId, SessionId
        from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
        from dlightrag.engine.agent.session.transactions import (
            RegisterExpectation,
            SessionTransaction,
        )

        assert creation.run.prepared_input is not None
        stale_session = SessionId(str(creation.run.prepared_input["agent_session_id"]))
        from dlightrag.adapters.postgres.answer.session_repository import PGAgentSessionRepository

        stale_repository = PGAgentSessionRepository(
            pool=pool,
            owner_id=_OWNER,
            run_id=uuid.UUID(creation.run.run_id),
            worker_id=_WORKER,
            lease_owner=_WORKER,
            fencing_epoch=int(stale_args["fencing_epoch"]),
        )
        head = LaneHead(LaneId.main(), None)
        state = LaneState(LaneId.main())
        stale_append = await stale_repository.transact(
            session_id=stale_session,
            fencing_epoch=int(stale_args["fencing_epoch"]),
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(head), SetRegister(state)],
                expectations=[
                    RegisterExpectation(head.ref, None),
                    RegisterExpectation(state.ref, None),
                ],
            ),
        )
        assert stale_append.__class__.__name__ == "TransactionLeaseLost"

        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "running"
        assert record.lease_owner == "worker-2"
        assert record.fencing_epoch == fresh.run.fencing_epoch
        assert await _event_types(pool, creation.run.run_id) == []

    async def test_same_worker_with_a_stale_epoch_is_fenced_out(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await _expire_lease(pool, creation.run.run_id)
        await _claimed(store)
        assert (
            await store.record_phase(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                phase="planning",
            )
            is None
        )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestEvents:
    async def test_phase_and_progress_event_advance_together(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        sequence = await store.record_phase(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            phase="researching",
        )
        assert sequence == 1
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.phase == "researching"
        assert record.next_event_sequence == 2
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(event.sequence, event.event_type, event.payload) for event in events] == [
            (1, "progress", {"phase": "researching"})
        ]

    async def test_executor_owned_event_labels_round_trip_without_runtime_registration(
        self, store
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        sequence = await store.append_event(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            phase=None,
            event_type="thinking",
            payload={"step": 1},
        )
        assert sequence == 1
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(event.event_type, event.payload) for event in events] == [
            ("thinking", {"step": 1})
        ]

    async def test_sequences_are_gap_free_under_concurrent_appends(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        sequences = await asyncio.gather(
            *(
                store.append_event(
                    **args,
                    phase=None,
                    event_type="token",
                    payload={"text": f"chunk-{index}"},
                )
                for index in range(24)
            )
        )
        assert sorted(sequence for sequence in sequences if sequence is not None) == list(
            range(1, 25)
        )
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [event.sequence for event in events] == list(range(1, 25))

    async def test_replay_resumes_after_a_cursor(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        await store.record_phase(**args, phase="planning")
        await store.append_event(**args, phase=None, event_type="token", payload={"text": "one"})
        await store.append_event(**args, phase=None, event_type="reset", payload={})
        await store.append_event(**args, phase=None, event_type="token", payload={"text": "two"})
        events = await store.read_event_page(
            owner_id=_OWNER, run_id=creation.run.run_id, after_sequence=2
        )
        assert [(event.sequence, event.event_type) for event in events] == [
            (3, "reset"),
            (4, "token"),
        ]

    async def test_appending_an_event_renews_the_lease(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_runs "
                "SET lease_expires_at = NOW() + INTERVAL '1 second' WHERE run_id = $1",
                uuid.UUID(creation.run.run_id),
            )
        await store.append_event(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            phase=None,
            event_type="token",
            payload={"text": "hi"},
        )
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.lease_expires_at is not None
        assert await store.claim_next(worker_id="worker-2") is None


class TestTerminalTransitions:
    async def test_success_stores_the_canonical_result_and_one_done_event(
        self, store, pool
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        result = {"answer": "yes", "references": [{"id": "r1"}]}
        outcome = await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result=result,
            stop_reason="converged",
        )
        assert outcome.committed is True
        assert outcome.status == "succeeded"
        assert outcome.event_sequence == 1
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "succeeded"
        assert record.result == result
        assert record.stop_reason == "converged"
        assert record.lease_owner is None
        assert record.finished_at is not None
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert events[0].event_type == "done"
        assert events[0].payload == {"status": "succeeded", "result": result}
        assert await _event_types(pool, creation.run.run_id) == ["done"]

    async def test_terminal_transitions_are_not_repeatable(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert (
            await store.finish_failure(**args, error_kind="provider_error", error_message="boom")
        ).committed is True
        assert (
            await store.finish_failure(**args, error_kind="provider_error", error_message="boom")
        ).committed is False
        assert (await store.finish_success(**args, result={"answer": "no"})).committed is False
        assert await _event_types(pool, creation.run.run_id) == ["error"]
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "failed"
        assert record.error_kind == "provider_error"
        assert record.error_message == "boom"
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(event.event_type, event.payload) for event in events] == [
            ("error", {"kind": "provider_error", "message": "boom"})
        ]

    async def test_worker_observed_cancellation_commits_cancelled(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        outcome = await store.finish_cancelled(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert outcome.committed is True
        assert outcome.status == "cancelled"
        assert await _event_types(pool, creation.run.run_id) == ["done"]
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert events[0].payload == {"status": "cancelled"}

    async def test_a_fenced_terminal_transition_clears_the_prepared_input(
        self, store, pool
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert await _prepared_input(pool, creation.run.run_id) is not None

        await store.finish_success(**args, result={"answer": "final"})

        assert await _prepared_input(pool, creation.run.run_id) is None
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.durable_progress_version == 0
        assert record.result == {"answer": "final"}

    async def test_an_unleased_finalization_clears_the_prepared_input(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await store.record_phase(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            phase="searching",
        )
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        await _expire_lease(pool, creation.run.run_id)

        assert (await store.sweep_once()).cancelled == 1

        assert await _prepared_input(pool, creation.run.run_id) is None


class TestShutdownAndRecovery:
    async def test_active_requirements_include_only_work_that_may_execute(
        self,
        store,
        pool,
    ) -> None:
        cancelled = await store.create_run(
            owner_id=_OWNER,
            request=_request(context_policy_revision="cancelled", pinned_models=[]),
        )
        await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=cancelled.run.run_id)
        await _expire_lease(pool, cancelled.run.run_id)

        abandoned = await store.create_run(
            owner_id=_OWNER,
            request=_request(context_policy_revision="abandoned", pinned_models=[]),
        )
        await _claimed(store)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_runs "
                "SET reclaims_without_progress = $2, lease_expires_at = NOW() - INTERVAL '1 second' "
                "WHERE run_id = $1",
                uuid.UUID(abandoned.run.run_id),
                MAX_RECLAIMS_WITHOUT_PROGRESS,
            )

        recoverable = await store.create_run(
            owner_id=_OWNER,
            request=_request(context_policy_revision="recoverable", pinned_models=[]),
        )
        await _claimed(store)
        await _expire_lease(pool, recoverable.run.run_id)
        await store.create_run(
            owner_id=_OWNER,
            request=_request(context_policy_revision="queued", pinned_models=[]),
        )

        requirements = [
            requirement async for requirement in store.iter_active_run_requirements(page_size=1)
        ]

        assert {row["prepared_input"]["context_policy_revision"] for row in requirements} == {
            "queued",
            "recoverable",
        }
        assert all(row["run_kind"] == "answer" for row in requirements)
        assert all(row["prepared_input"]["pinned_models"] == [] for row in requirements)

    async def test_cancel_pending_rescan_keyset_pages_only_live_worker_leases(
        self,
        store,
        pool,
    ) -> None:
        first = await store.create_run(owner_id=_OWNER, request=_request())
        first_claim = await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=first.run.run_id)

        second = await store.create_run(owner_id=_OWNER, request=_request())
        second_claim = await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=second.run.run_id)

        other = await store.create_run(owner_id=_OWNER, request=_request())
        other_claim = await store.claim_next(worker_id="other-worker")
        assert other_claim is not None
        await store.request_cancellation(owner_id=_OWNER, run_id=other.run.run_id)

        expired = await store.create_run(owner_id=_OWNER, request=_request())
        expired_claim = await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=expired.run.run_id)
        await _expire_lease(pool, expired.run.run_id)

        pending = [
            item
            async for item in store.iter_cancel_pending(
                worker_id=_WORKER,
                page_size=1,
            )
        ]

        assert {run_id for _, run_id in pending} == {
            first_claim.run.run_id,
            second_claim.run.run_id,
        }
        assert other_claim.run.run_id not in {run_id for _, run_id in pending}
        assert expired_claim.run.run_id not in {run_id for _, run_id in pending}

    async def test_shutdown_finalizes_a_cancel_pending_run(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        release = await store.release_for_shutdown(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert release == "cancelled"
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "cancelled"
        assert await _event_types(pool, creation.run.run_id) == ["done"]
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert events[0].payload == {"status": "cancelled"}

    async def test_sweeper_finalizes_an_unleased_cancellation(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        await _expire_lease(pool, creation.run.run_id)
        sweep = await store.sweep_once()
        assert sweep.cancelled == 1
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "cancelled"
        assert await _event_types(pool, creation.run.run_id) == ["done"]
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert events[0].payload == {"status": "cancelled"}

    async def test_sweeper_abandonment_persists_the_exact_public_error(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_runs "
                "SET reclaims_without_progress = $2, lease_expires_at = NOW() - INTERVAL '1 second' "
                "WHERE run_id = $1",
                uuid.UUID(creation.run.run_id),
                MAX_RECLAIMS_WITHOUT_PROGRESS,
            )

        sweep = await store.sweep_once()

        assert sweep.cancelled == 0
        assert sweep.abandoned == 1
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "failed"
        assert record.error_kind == RUN_ABANDONED_ERROR_KIND
        assert record.error_message == _ABANDONED_ERROR_MESSAGE
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(event.event_type, event.payload) for event in events] == [
            (
                "error",
                {
                    "kind": RUN_ABANDONED_ERROR_KIND,
                    "message": _ABANDONED_ERROR_MESSAGE,
                },
            )
        ]

    async def test_sweeper_leaves_live_leases_alone(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        sweep = await store.sweep_once()
        assert sweep.cancelled == 0
        assert sweep.abandoned == 0
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "running"


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def _reference(
    digest: str, *, resource_id: str = "res-1", ordinal: int = 0
) -> PendingArtifactReference:
    return PendingArtifactReference(
        resource_id=resource_id,
        reference_kind="current_attachment",
        ordinal=ordinal,
        digest=digest,
        filename="report.pdf",
        mime_type="application/pdf",
        transform_locator={"page": 2},
    )


class TestArtifacts:
    async def test_creation_stores_blobs_and_ordered_references(self, store) -> None:
        artifact = PendingArtifact(content=b"%PDF-1.7 payload")
        creation = await store.create_run(
            owner_id=_OWNER,
            request=_request(),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        references = await store.list_run_artifacts(owner_id=_OWNER, run_id=creation.run.run_id)
        assert len(references) == 1
        assert references[0].digest == artifact.digest
        assert references[0].reference_kind == "current_attachment"
        assert references[0].transform_locator == {"page": 2}
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) == (
            artifact.content
        )

    async def test_multichunk_and_empty_blobs_round_trip_exactly(self, store, pool) -> None:
        contents = (b"a" * BLOB_CHUNK_BYTES + b"middle" + b"z" * BLOB_CHUNK_BYTES, b"")
        artifacts = tuple(PendingArtifact(content=content) for content in contents)

        for index, artifact in enumerate(artifacts):
            await store.create_run(
                owner_id=_OWNER,
                request=_request(f"blob-{index}"),
                artifacts=[artifact],
                references=[_reference(artifact.digest, resource_id=f"res-{index}")],
            )
            assert (
                await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest)
                == contents[index]
            )

        async with pool.acquire() as conn:
            counts = [
                await conn.fetchval(
                    "SELECT count(*) FROM dlightrag_blob_chunks"
                    " WHERE owner_id = $1 AND digest = $2",
                    _OWNER,
                    artifact.digest,
                )
                for artifact in artifacts
            ]
        assert counts == [3, 0]

    async def test_concurrent_same_digest_writers_converge_on_one_ordered_blob(
        self, store, pool
    ) -> None:
        content = b"first" + b"m" * (2 * BLOB_CHUNK_BYTES) + b"last"
        artifact = PendingArtifact(content=content)

        creations = await asyncio.gather(
            *(
                store.create_run(
                    owner_id=_OWNER,
                    request=_request(f"concurrent-{index}"),
                    artifacts=[artifact],
                    references=[_reference(artifact.digest, resource_id=f"res-{index}")],
                )
                for index in range(2)
            )
        )

        assert len({creation.run.run_id for creation in creations}) == 2
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) == content
        async with pool.acquire() as conn:
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM dlightrag_blobs WHERE owner_id = $1 AND digest = $2",
                    _OWNER,
                    artifact.digest,
                )
                == 1
            )
            chunk_rows = await conn.fetch(
                "SELECT chunk_index FROM dlightrag_blob_chunks"
                " WHERE owner_id = $1 AND digest = $2 ORDER BY chunk_index",
                _OWNER,
                artifact.digest,
            )
        assert [int(row["chunk_index"]) for row in chunk_rows] == list(range(3))

    async def test_new_blobs_are_acquired_in_canonical_order_across_acceptances(
        self, store
    ) -> None:
        first = PendingArtifact(content=b"new-opposite-order-a")
        second = PendingArtifact(content=b"new-opposite-order-b")

        creations = await asyncio.wait_for(
            asyncio.gather(
                store.create_run(
                    owner_id=_OWNER,
                    request=_request("new-order-left"),
                    artifacts=[first, second],
                    references=[
                        _reference(first.digest, resource_id="left-first", ordinal=0),
                        _reference(second.digest, resource_id="left-second", ordinal=1),
                    ],
                ),
                store.create_run(
                    owner_id=_OWNER,
                    request=_request("new-order-right"),
                    artifacts=[second, first],
                    references=[
                        _reference(second.digest, resource_id="right-second", ordinal=0),
                        _reference(first.digest, resource_id="right-first", ordinal=1),
                    ],
                ),
            ),
            timeout=5,
        )

        assert len({creation.run.run_id for creation in creations}) == 2
        assert await _blob_store(store).read(owner_id=_OWNER, digest=first.digest) == first.content
        assert (
            await _blob_store(store).read(owner_id=_OWNER, digest=second.digest) == second.content
        )
        left_references = await store.list_run_artifacts(
            owner_id=_OWNER, run_id=creations[0].run.run_id
        )
        right_references = await store.list_run_artifacts(
            owner_id=_OWNER, run_id=creations[1].run.run_id
        )
        assert [reference.resource_id for reference in left_references] == [
            "left-first",
            "left-second",
        ]
        assert [reference.resource_id for reference in right_references] == [
            "right-second",
            "right-first",
        ]

    async def test_existing_blobs_can_be_reused_in_opposite_order_without_deadlock(
        self, store, pool
    ) -> None:
        first = PendingArtifact(content=b"opposite-order-a")
        second = PendingArtifact(content=b"opposite-order-b")
        await store.create_run(
            owner_id=_OWNER,
            request=_request("opposite-order-seed"),
            artifacts=[first, second],
            references=[
                _reference(first.digest, resource_id="first", ordinal=0),
                _reference(second.digest, resource_id="second", ordinal=1),
            ],
        )

        async with pool.acquire() as left, pool.acquire() as right:
            async with left.transaction(), right.transaction():
                await write_blob_content(
                    left,
                    owner_id=_OWNER,
                    digest=first.digest,
                    content=first.content,
                )
                await write_blob_content(
                    right,
                    owner_id=_OWNER,
                    digest=second.digest,
                    content=second.content,
                )
                await asyncio.wait_for(
                    asyncio.gather(
                        write_blob_content(
                            left,
                            owner_id=_OWNER,
                            digest=second.digest,
                            content=second.content,
                        ),
                        write_blob_content(
                            right,
                            owner_id=_OWNER,
                            digest=first.digest,
                            content=first.content,
                        ),
                    ),
                    timeout=5,
                )

    async def test_uncommitted_blob_winner_exposes_size_mismatch_after_wait(
        self, store, pool
    ) -> None:
        artifact = PendingArtifact(content=b"uncommitted winner")
        async with (
            pool.acquire() as winner,
            pool.acquire() as contender,
            pool.acquire() as observer,
        ):
            winner_pid = int(await winner.fetchval("SELECT pg_backend_pid()"))
            contender_pid = int(await contender.fetchval("SELECT pg_backend_pid()"))
            winner_tx = winner.transaction()
            contender_tx = contender.transaction()
            contender_write: asyncio.Task[None] | None = None
            contender_consumed = False
            try:
                await winner_tx.start()
                await contender_tx.start()
                await write_blob_content(
                    winner,
                    owner_id=_OWNER,
                    digest=artifact.digest,
                    content=artifact.content,
                )
                contender_write = asyncio.create_task(
                    write_complete_blob(
                        contender,
                        owner_id=_OWNER,
                        digest=artifact.digest,
                        total_bytes=len(artifact.content) + 1,
                        chunks=(artifact.content + b"!",),
                    )
                )
                for _attempt in range(100):
                    blockers = await observer.fetchval("SELECT pg_blocking_pids($1)", contender_pid)
                    if winner_pid in blockers:
                        break
                    await asyncio.sleep(0.01)
                else:
                    pytest.fail("contender never waited on the uncommitted blob winner")
                assert not contender_write.done()
                await winner_tx.commit()
                with pytest.raises(BlobSizeConflict):
                    await asyncio.wait_for(contender_write, timeout=5)
                contender_consumed = True
            finally:
                task_error: BaseException | None = None
                if contender_write is not None and not contender_consumed:
                    if not contender_write.done():
                        contender_write.cancel()
                    try:
                        await contender_write
                    except asyncio.CancelledError, BlobSizeConflict:
                        pass
                    except BaseException as exc:
                        task_error = exc
                rollbacks = []
                if winner.is_in_transaction():
                    rollbacks.append(winner_tx.rollback())
                if contender.is_in_transaction():
                    rollbacks.append(contender_tx.rollback())
                if rollbacks:
                    await asyncio.gather(*rollbacks, return_exceptions=True)
                if task_error is not None:
                    raise task_error

        assert (
            await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest)
            == artifact.content
        )

    async def test_blob_size_collision_rolls_back_acceptance_and_publication(
        self, store, pool
    ) -> None:
        content = b"collision payload"
        artifact = PendingArtifact(content=content)
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO dlightrag_blobs (owner_id, digest, byte_size) VALUES ($1, $2, $3)",
                _OWNER,
                artifact.digest,
                len(content) + 1,
            )

        with pytest.raises(ValueError, match="blob digest collision"):
            await store.create_run(
                owner_id=_OWNER,
                request=_request("collision-acceptance"),
                artifacts=[artifact],
                references=[_reference(artifact.digest)],
            )

        creation = await store.create_run(owner_id=_OWNER, request=_request("publication"))
        claim = await _claimed(store)
        assert claim.run.run_id == creation.run.run_id
        publication = PendingPublication(
            resource_id="published-1",
            reference_kind="published_artifact",
            filename="result.txt",
            mime_type="text/plain",
            content=content,
        )
        with pytest.raises(ValueError, match="blob digest collision"):
            await store.finish_success(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                result={"answer": "not committed"},
                publications=[publication],
            )

        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "running"
        async with pool.acquire() as conn:
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM dlightrag_blob_chunks"
                    " WHERE owner_id = $1 AND digest = $2",
                    _OWNER,
                    artifact.digest,
                )
                == 0
            )
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_answer_run_artifacts") == 0

    async def test_blobs_deduplicate_within_one_owner_only(self, store, pool) -> None:
        artifact = PendingArtifact(content=b"shared bytes")
        first = await store.create_run(
            owner_id=_OWNER,
            request=_request("a"),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        await store.create_run(
            owner_id=_OWNER,
            request=_request("b"),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        await store.create_run(
            owner_id=_OTHER_OWNER,
            request=_request("c"),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        async with pool.acquire() as conn:
            owners = await conn.fetch(
                "SELECT owner_id FROM dlightrag_blobs WHERE digest = $1",
                artifact.digest,
            )
        assert sorted(str(row["owner_id"]) for row in owners) == sorted([_OWNER, _OTHER_OWNER])
        assert await _blob_store(store).read(owner_id="ghost", digest=artifact.digest) is None
        assert first.run.run_id is not None

    async def test_deleting_one_run_keeps_bytes_another_run_still_links(self, store) -> None:
        artifact = PendingArtifact(content=b"shared bytes")
        first = await store.create_run(
            owner_id=_OWNER,
            request=_request("a"),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        second = await store.create_run(
            owner_id=_OWNER,
            request=_request("b"),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        deletion = await store.delete_runs(owner_id=_OWNER, run_ids=[first.run.run_id])
        assert deletion.runs == 1
        assert deletion.artifacts == 0
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) is not None

        final = await store.delete_runs(owner_id=_OWNER, run_ids=[second.run.run_id])
        assert final.runs == 1
        assert final.artifacts == 1
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) is None

    async def test_deleting_a_run_collects_its_fetched_resource_blob(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        digest = await _write_fetched_blob(pool, creation.run.run_id, b"fetched delete bytes")

        deletion = await store.delete_runs(owner_id=_OWNER, run_ids=[creation.run.run_id])

        assert deletion.runs == 1
        assert deletion.artifacts == 1
        assert await _blob_store(store).read(owner_id=_OWNER, digest=digest) is None

    async def test_deletion_is_owner_scoped(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        deletion = await store.delete_runs(owner_id=_OTHER_OWNER, run_ids=[creation.run.run_id])
        assert deletion.runs == 0
        assert await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id) is not None

    async def test_unlinked_session_is_removed_only_after_its_final_run_route(
        self, store, pool
    ) -> None:
        session_id = str(uuid.uuid7())
        first = await store.create_run(
            owner_id=_OWNER,
            request=_request("first", agent_session_id=session_id),
        )
        second = await store.create_run(
            owner_id=_OWNER,
            request=_request("second", agent_session_id=session_id),
        )
        claimed = await _claimed(store)
        from dlightrag.engine.agent.session.ids import LaneId, SessionId
        from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
        from dlightrag.engine.agent.session.transactions import (
            RegisterExpectation,
            SessionTransaction,
        )

        head = LaneHead(LaneId.main(), None)
        state = LaneState(LaneId.main())
        await claimed.execution.session_repository.transact(
            session_id=SessionId(session_id),
            fencing_epoch=claimed.execution.fencing_epoch,
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(head), SetRegister(state)],
                expectations=[
                    RegisterExpectation(head.ref, None),
                    RegisterExpectation(state.ref, None),
                ],
            ),
        )

        assert (await store.delete_runs(owner_id=_OWNER, run_ids=[first.run.run_id])).runs == 1
        async with pool.acquire() as conn:
            assert await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM dlightrag_agent_sessions"
                " WHERE owner_id = $1 AND session_id = $2)",
                _OWNER,
                uuid.UUID(session_id),
            )

        assert (await store.delete_runs(owner_id=_OWNER, run_ids=[second.run.run_id])).runs == 1
        async with pool.acquire() as conn:
            assert not await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM dlightrag_agent_sessions"
                " WHERE owner_id = $1 AND session_id = $2)",
                _OWNER,
                uuid.UUID(session_id),
            )

    async def test_cleanup_yields_to_a_run_still_adopting_the_blob(self, store, pool) -> None:
        """An uncommitted adoption holds the blob's key-share lock, so cleanup skips it."""
        artifact = PendingArtifact(content=b"contended bytes")
        first = await store.create_run(
            owner_id=_OWNER,
            request=_request("a"),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        second = await store.create_run(owner_id=_OWNER, request=_request("b"))

        adopting = await pool.acquire()
        transaction = adopting.transaction()
        await transaction.start()
        try:
            await adopting.execute(
                "INSERT INTO dlightrag_answer_run_artifacts "
                "(owner_id, run_id, resource_id, reference_kind, ordinal, digest, "
                "filename, mime_type) "
                "VALUES ($1, $2, 'res-late', 'fetched_resource', 0, $3, 'late.pdf', "
                "'application/pdf')",
                _OWNER,
                uuid.UUID(second.run.run_id),
                artifact.digest,
            )
            deletion = await asyncio.wait_for(
                store.delete_runs(owner_id=_OWNER, run_ids=[first.run.run_id]), timeout=10
            )
        finally:
            await transaction.commit()
            await pool.release(adopting)

        assert deletion.runs == 1
        assert deletion.artifacts == 0
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) == (
            artifact.content
        )
        references = await store.list_run_artifacts(owner_id=_OWNER, run_id=second.run.run_id)
        assert [reference.resource_id for reference in references] == ["res-late"]

    async def test_a_referenced_blob_cannot_be_deleted_directly(self, store, pool) -> None:
        artifact = PendingArtifact(content=b"still linked")
        await store.create_run(
            owner_id=_OWNER,
            request=_request(),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        async with pool.acquire() as conn:
            with pytest.raises(asyncpg.exceptions.RestrictViolationError):
                await conn.execute(
                    "DELETE FROM dlightrag_blobs WHERE owner_id = $1 AND digest = $2",
                    _OWNER,
                    artifact.digest,
                )


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


class TestRetention:
    async def test_trim_removes_expired_event_logs_and_keeps_the_result(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "kept"},
        )
        assert await store.trim_expired_event_logs() == 0

        await _backdate_finish(pool, creation.run.run_id, days=370)
        assert await store.trim_expired_event_logs() == 1
        assert await store.trim_expired_event_logs() == 0

        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.events_trimmed_at is not None
        assert record.result == {"answer": "kept"}
        assert await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id) == ()

    async def test_prune_deletes_expired_runs_with_events_and_blobs(self, store, pool) -> None:
        artifact = PendingArtifact(content=b"expiring bytes")
        creation = await store.create_run(
            owner_id=_OWNER,
            request=_request(),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        claim = await _claimed(store)
        await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "old"},
        )
        fresh = await store.prune_expired_runs()
        assert fresh.runs == 0
        assert fresh.artifacts == 0

        await _backdate_finish(pool, creation.run.run_id, days=370)
        outcome = await store.prune_expired_runs()
        assert outcome.runs == 1
        assert outcome.artifacts == 1
        assert await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id) is None
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) is None
        async with pool.acquire() as conn:
            remaining = await conn.fetchval(
                "SELECT count(*) FROM dlightrag_run_events WHERE run_id = $1",
                uuid.UUID(creation.run.run_id),
            )
        assert int(remaining) == 0

    async def test_prune_collects_fetched_resource_blobs(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        digest = await _write_fetched_blob(pool, creation.run.run_id, b"fetched retention bytes")
        claim = await _claimed(store)
        await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "old"},
        )
        await _backdate_finish(pool, creation.run.run_id, days=370)

        outcome = await store.prune_expired_runs()

        assert outcome.runs == 1
        assert outcome.artifacts == 1
        assert await _blob_store(store).read(owner_id=_OWNER, digest=digest) is None

    async def test_prune_leaves_unfinished_runs_alone(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_runs "
                "SET created_at = NOW() - INTERVAL '90 days' WHERE run_id = $1",
                uuid.UUID(creation.run.run_id),
            )
        assert (await store.prune_expired_runs()).runs == 0
        assert await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id) is not None


# ---------------------------------------------------------------------------
# Blob-cleanup failure isolation
# ---------------------------------------------------------------------------


async def _force_blob_delete_restrict(pool: Any, *, only_digest: str | None = None) -> None:
    """Make blob deletes raise SQLSTATE 23001, exactly as an adopted blob does.

    A real adopter usually loses to ``FOR UPDATE SKIP LOCKED``, so the surviving
    window is too narrow to schedule. The trigger reproduces the identical
    ``RestrictViolationError`` inside a real transaction, which is what decides
    whether the caller's run deletion survives. ``only_digest`` contends exactly
    one blob so a mixed batch can be observed.
    """
    guard = "IF OLD.digest = $q$" + only_digest + "$q$ THEN" if only_digest else "IF true THEN"
    async with pool.acquire() as conn:
        await conn.execute(
            "CREATE FUNCTION dlightrag_test_restrict() RETURNS trigger "
            "LANGUAGE plpgsql AS $$ BEGIN "
            f"{guard} "
            "RAISE EXCEPTION 'blob adopted concurrently' USING ERRCODE = '23001'; "
            "END IF; RETURN OLD; "
            "END; $$"
        )
        await conn.execute(
            "CREATE TRIGGER dlightrag_test_restrict_trigger "
            "BEFORE DELETE ON dlightrag_blobs "
            "FOR EACH ROW EXECUTE FUNCTION dlightrag_test_restrict()"
        )


async def _succeed_and_expire(store: PGRunStore, pool: Any, run_id: str) -> None:
    claim = await _claimed(store)
    await store.finish_success(
        owner_id=_OWNER,
        run_id=run_id,
        worker_id=_WORKER,
        fencing_epoch=claim.run.fencing_epoch,
        result={"answer": run_id},
    )
    await _backdate_finish(pool, run_id, days=370)


class TestBlobCleanupFailureIsolation:
    async def test_deletion_commits_when_blob_cleanup_hits_restrict(self, store, pool) -> None:
        artifact = PendingArtifact(content=b"contended bytes")
        creation = await store.create_run(
            owner_id=_OWNER,
            request=_request(),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        await _force_blob_delete_restrict(pool)

        deletion = await store.delete_runs(owner_id=_OWNER, run_ids=[creation.run.run_id])

        assert deletion.runs == 1
        assert deletion.artifacts == 0
        assert await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id) is None
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) is not None

    async def test_retention_advances_past_a_contended_head_batch(self, store, pool) -> None:
        run_ids: list[str] = []
        digests: list[str] = []
        for index in range(3):
            artifact = PendingArtifact(content=f"batch bytes {index}".encode())
            creation = await store.create_run(
                owner_id=_OWNER,
                request=_request(f"q{index}"),
                artifacts=[artifact],
                references=[_reference(artifact.digest)],
            )
            await _succeed_and_expire(store, pool, creation.run.run_id)
            run_ids.append(creation.run.run_id)
            digests.append(artifact.digest)
        await _force_blob_delete_restrict(pool)

        outcome = await store.prune_expired_runs()

        assert outcome.runs == 3
        assert outcome.artifacts == 0
        for run_id in run_ids:
            assert await store.get_run(owner_id=_OWNER, run_id=run_id) is None
        assert (await store.prune_expired_runs()).runs == 0
        for digest in digests:
            assert await _blob_store(store).read(owner_id=_OWNER, digest=digest) is not None

    async def test_one_contended_digest_does_not_shield_unrelated_orphans(
        self, store, pool
    ) -> None:
        """A mixed batch must not let one adopted blob keep every other orphan alive."""
        digests: list[str] = []
        run_ids: list[str] = []
        for index in range(3):
            artifact = PendingArtifact(content=f"mixed bytes {index}".encode())
            creation = await store.create_run(
                owner_id=_OWNER,
                request=_request(f"mixed-{index}"),
                artifacts=[artifact],
                references=[_reference(artifact.digest)],
            )
            await _succeed_and_expire(store, pool, creation.run.run_id)
            digests.append(artifact.digest)
            run_ids.append(creation.run.run_id)
        contended = digests[1]
        await _force_blob_delete_restrict(pool, only_digest=contended)

        outcome = await store.prune_expired_runs()

        assert outcome.runs == 3
        assert outcome.artifacts == 2
        for run_id in run_ids:
            assert await store.get_run(owner_id=_OWNER, run_id=run_id) is None
        assert await _blob_store(store).read(owner_id=_OWNER, digest=contended) is not None
        for digest in (digests[0], digests[2]):
            assert await _blob_store(store).read(owner_id=_OWNER, digest=digest) is None


# ---------------------------------------------------------------------------
# Bounded event replay
# ---------------------------------------------------------------------------


class TestEventPaging:
    async def test_paged_replay_crosses_the_boundary_without_gaps(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        appended = 501
        for index in range(appended):
            await store.append_event(
                **args,
                phase=None,
                event_type="token",
                payload={"text": f"chunk-{index}"},
            )
        assert (await store.finish_success(**args, result={"answer": "end"})).committed is True

        pages: list[tuple[int, ...]] = []
        cursor = 0
        while True:
            page = await store.read_event_page(
                owner_id=_OWNER, run_id=creation.run.run_id, after_sequence=cursor
            )
            if not page:
                break
            pages.append(tuple(event.sequence for event in page))
            cursor = page[-1].sequence

        total = appended + 1
        replayed = [sequence for page in pages for sequence in page]
        assert len(pages) >= 2
        assert len(pages[0]) < total
        assert replayed == list(range(1, total + 1))


# ---------------------------------------------------------------------------
# Fetched-resource replay and fenced attachment
# ---------------------------------------------------------------------------


def _fetched(
    digest: str, *, resource_id: str = "res-web", ordinal: int = 0
) -> PendingArtifactReference:
    return PendingArtifactReference(
        resource_id=resource_id,
        reference_kind="fetched_resource",
        ordinal=ordinal,
        digest=digest,
        filename=f"{resource_id}.html",
        mime_type="text/html",
    )


class TestAgentControlsAndChildren:
    async def test_controls_are_ordered_and_consumed_under_the_run_lease(self, store) -> None:
        creation = await store.create_run(
            owner_id=_OWNER,
            request=_request(mode="research"),
        )
        first = await store.enqueue_agent_control(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            kind="steer",
            content="first",
        )
        second = await store.enqueue_agent_control(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            kind="steer",
            content="second",
        )
        claim = await _claimed(store)

        controls = await store.load_pending_agent_controls(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        acknowledged = await store.acknowledge_agent_controls(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            control_sequences=tuple(int(item["control_sequence"]) for item in controls or ()),
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        replay = await store.load_pending_agent_controls(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )

        assert acknowledged
        assert first is not None and first["control_sequence"] == 1
        assert second is not None and second["control_sequence"] == 2
        assert [item["content"] for item in controls or ()] == ["first", "second"]
        assert replay == ()

    async def test_one_spawn_call_persists_multiple_child_lineages(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        parent_id = str(uuid.uuid7())
        children = (str(uuid.uuid7()), str(uuid.uuid7()))
        parent_intent_id = str(uuid.uuid7())
        child_epochs: list[int] = []

        for position, child_id in enumerate(children):
            held = await store.upsert_child_session(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                child_session_id=child_id,
                parent_session_id=parent_id,
                parent_call_id="one-call",
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                parent_intent_id=parent_intent_id,
                objective=f"child {position}",
                context_mode="parent",
                model_role="extract",
                tools=("search_knowledge_base",),
                depth=1,
                context_snapshot={
                    "parent_session_id": parent_id,
                    "parent_entry_id": str(uuid.uuid7()),
                    "depth": 0,
                    "messages": [{"role": "user", "content": "parent"}],
                    "evidence_state": {},
                },
                plan={"schema_version": 1, "tools": ["search_knowledge_base"]},
                budget={"provider_attempt_limit": 2},
                host_state={"evidence": {}},
            )
            assert held
            # Tool execution repeats the roster upsert after the intent-bound
            # precreate. It must not lose the parent intent or lease.
            assert await store.upsert_child_session(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                child_session_id=child_id,
                parent_session_id=parent_id,
                parent_call_id="one-call",
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
            )
            child_epoch = await store.claim_child_session(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                child_session_id=child_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
            )
            assert child_epoch is not None
            child_epochs.append(child_epoch)
            from dlightrag.engine.agent.session.ids import LaneId, SessionId
            from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
            from dlightrag.engine.agent.session.transactions import (
                RegisterExpectation,
                SessionTransaction,
                TransactionCommit,
                TransactionLeaseLost,
            )

            child_store = claim.execution.session_repository.for_child(
                SessionId(child_id),
                fencing_epoch=child_epoch,
            )
            head = LaneHead(LaneId.main(), None)
            state = LaneState(LaneId.main())
            child_commit = await child_store.transact(
                session_id=SessionId(child_id),
                fencing_epoch=child_epoch,
                transaction=SessionTransaction.from_parts(
                    register_writes=[SetRegister(head), SetRegister(state)],
                    expectations=[
                        RegisterExpectation(head.ref, None),
                        RegisterExpectation(state.ref, None),
                    ],
                ),
            )
            assert isinstance(child_commit, TransactionCommit)
            parent_write = await claim.execution.session_repository.transact(
                session_id=SessionId(child_id),
                fencing_epoch=claim.run.fencing_epoch,
                transaction=SessionTransaction.from_parts(
                    register_writes=[SetRegister(state)],
                    expectations=[RegisterExpectation(state.ref, child_commit.commit_sequence)],
                ),
            )
            assert isinstance(parent_write, TransactionLeaseLost)
            assert await store.finish_child_session(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                child_session_id=child_id,
                status="succeeded",
                summary="done",
                usage={"input_tokens": 10 + position},
                outcome={
                    "status": "succeeded",
                    "summary": "done",
                    "handles": [],
                    "usage": {"input_tokens": 10 + position},
                    "child_session_id": child_id,
                    "evidence_state": {"contexts": {}},
                },
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
            )

        roster = await store.list_child_sessions(owner_id=_OWNER, run_id=creation.run.run_id)
        assert {item["child_session_id"] for item in roster} == set(children)
        assert {item["parent_session_id"] for item in roster} == {parent_id}
        assert {item["parent_intent_id"] for item in roster} == {parent_intent_id}
        assert [item["objective"] for item in roster] == ["child 0", "child 1"]
        assert [item["usage"]["input_tokens"] for item in roster] == [10, 11]
        assert child_epochs == [1, 1]
        assert [item["depth"] for item in roster] == [1, 1]
        assert all(item["context_snapshot"]["messages"] for item in roster)
        assert all(item["plan"]["tools"] == ["search_knowledge_base"] for item in roster)
        assert all(item["budget"]["provider_attempt_limit"] == 2 for item in roster)
        assert [item["host_state"]["terminal_outcome"]["status"] for item in roster] == [
            "succeeded",
            "succeeded",
        ]
        assert all(
            item["host_state"]["terminal_outcome"]["evidence_state"] == {"contexts": {}}
            for item in roster
        )

    async def test_child_lease_heartbeat_survives_original_window_and_fences_takeover(
        self, store, pool
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        parent_id = str(uuid.uuid7())
        child_id = str(uuid.uuid7())
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            parent_call_id="lease-call",
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        child_epoch = await store.claim_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert child_epoch == 1

        async with pool.acquire() as conn:
            original_expiry = await conn.fetchval(
                "UPDATE dlightrag_answer_child_sessions"
                " SET lease_expires_at = NOW() + INTERVAL '200 milliseconds'"
                " WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3"
                " RETURNING lease_expires_at",
                _OWNER,
                uuid.UUID(creation.run.run_id),
                uuid.UUID(child_id),
            )
        assert await store.heartbeat_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            child_fencing_epoch=child_epoch,
        )
        await asyncio.sleep(0.3)
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT NOW() > $1", original_expiry)

        from dlightrag.engine.agent.session.ids import LaneId, SessionId
        from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
        from dlightrag.engine.agent.session.transactions import (
            RegisterExpectation,
            SessionTransaction,
            TransactionCommit,
            TransactionLeaseLost,
        )

        child_store = claim.execution.session_repository.for_child(
            SessionId(child_id), fencing_epoch=child_epoch
        )
        head = LaneHead(LaneId.main(), None)
        state = LaneState(LaneId.main())
        commit = await child_store.transact(
            session_id=SessionId(child_id),
            fencing_epoch=child_epoch,
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(head), SetRegister(state)],
                expectations=[
                    RegisterExpectation(head.ref, None),
                    RegisterExpectation(state.ref, None),
                ],
            ),
        )
        assert isinstance(commit, TransactionCommit)

        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_answer_child_sessions"
                " SET lease_expires_at = NOW() - INTERVAL '1 second'"
                " WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3",
                _OWNER,
                uuid.UUID(creation.run.run_id),
                uuid.UUID(child_id),
            )
        next_epoch = await store.claim_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert next_epoch == child_epoch + 1
        assert not await store.heartbeat_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            child_fencing_epoch=child_epoch,
        )
        stale_write = await child_store.transact(
            session_id=SessionId(child_id),
            fencing_epoch=child_epoch,
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(state)],
                expectations=[RegisterExpectation(state.ref, commit.commit_sequence)],
            ),
        )
        assert isinstance(stale_write, TransactionLeaseLost)
