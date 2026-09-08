# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL integration coverage for the Commit 1 promotion control-plane schema.

Compact real-PostgreSQL fixtures prove the durable registry control-plane
fields and the promotion-job table: idempotent enqueue, leased claims,
fenced transitions, legal-state constraints, and the bounded claim indexes.
No worker or trigger exists in this commit, so only the adapter interfaces
drive these rows.
"""

import datetime
from typing import Any

import asyncpg
import pytest

from tests.integration.pg_conn import PG_CONN_KWARGS

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_MAINT_DB = "postgres"
_TEST_DB = "dlightrag_partition_foundation_test"
_WORKSPACE = "pf_promotion_ws"
_WORKSPACE_B = "pf_promotion_ws_b"

_DEFAULT_KWARGS: dict[str, Any] = PG_CONN_KWARGS


def _kwargs(database: str) -> dict[str, Any]:
    return {**_DEFAULT_KWARGS, "database": database}


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_kwargs(_MAINT_DB))
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


async def _ensure_test_database() -> None:
    """Create the dedicated test database if an earlier module has not."""
    conn = await asyncpg.connect(**_kwargs(_MAINT_DB))
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", _TEST_DB)
        if not exists:
            await conn.execute(f"CREATE DATABASE {_TEST_DB}")
    finally:
        await conn.close()
    db = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        for extension in ("vector", "pg_textsearch", "pg_trgm"):
            await db.execute(f"CREATE EXTENSION IF NOT EXISTS {extension}")
    finally:
        await db.close()


@pytest.fixture
async def pool() -> Any:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")
    await _ensure_test_database()
    pool = await asyncpg.create_pool(**_kwargs(_TEST_DB), min_size=1, max_size=2)
    yield pool
    await pool.close()


async def test_registry_control_plane_fields_are_durable_and_constrained(pool: Any) -> None:
    from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry

    registry = PGWorkspaceRegistry(pool=pool)
    await registry.initialize()
    try:
        # Simulate the pre-foundation registry upgrade path: columns may exist
        # while the new checks do not. Replaying the final migration must make
        # the writer-created schema pass the same catalog contract readers use.
        from dlightrag.adapters.postgres.corpus import workspaces

        constraint_names = [name for name, _expression in workspaces._WORKSPACE_CHECK_EXPRESSIONS]
        async with pool.acquire() as conn:
            for name in constraint_names:
                await conn.execute(f"ALTER TABLE dlightrag_workspace_meta DROP CONSTRAINT {name}")
            await conn.execute(
                "DELETE FROM dlightrag_schema_migrations "
                "WHERE scope = 'workspace_registry' "
                "AND version = 'workspace_meta_promotion_constraints'"
            )
        await registry.initialize()
        async with pool.acquire() as conn:
            installed = {
                str(row["conname"])
                for row in await conn.fetch(
                    "SELECT conname FROM pg_catalog.pg_constraint "
                    "WHERE conrelid = 'dlightrag_workspace_meta'::regclass"
                )
            }
        assert set(constraint_names) <= installed

        await registry.upsert(
            workspace=_WORKSPACE,
            display_name="Promotion Workspace",
            embedding_model="pf-it-fake",
        )
        row = await registry.get_row(_WORKSPACE)
        assert row is not None
        assert row["ingested_docs_total"] == 0
        assert row["ingested_chunks_total"] == 0
        assert row["storage_tier"] == "shared"
        assert row["promotion_state"] == "none"

        # Counters are monotonic across repeated additive deltas.
        assert await registry.add_ingested_counts(workspace=_WORKSPACE, docs=3, chunks=41)
        assert await registry.add_ingested_counts(workspace=_WORKSPACE, docs=2, chunks=9)
        row = await registry.get_row(_WORKSPACE)
        assert row is not None
        assert row["ingested_docs_total"] == 5
        assert row["ingested_chunks_total"] == 50

        # Tier + promotion observability transitions, with retry bookkeeping.
        assert await registry.set_promotion_state(workspace=_WORKSPACE, state="pending")
        assert await registry.set_storage_tier(workspace=_WORKSPACE, tier="hot")
        assert not await registry.set_storage_tier(workspace=_WORKSPACE, tier="shared")
        retry_at = datetime.datetime.now(datetime.UTC)
        assert await registry.set_promotion_state(
            workspace=_WORKSPACE,
            state="failed",
            error="cutover invariant mismatch",
            next_retry_at=retry_at,
        )
        row = await registry.get_row(_WORKSPACE)
        assert row is not None
        assert row["storage_tier"] == "hot"
        assert row["promotion_state"] == "failed"
        assert row["promotion_last_error"] == "cutover invariant mismatch"
        assert row["promotion_retry_count"] == 1
        assert row["promotion_next_retry_at"] is not None

        # Write fence: expired requested leases are rejected; valid leases can
        # be acquired, extended, and released only by their owner token.
        assert not await registry.acquire_write_fence(
            workspace=_WORKSPACE,
            owner="stale-worker",
            until=datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=1),
        )
        until = datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5)
        assert await registry.acquire_write_fence(
            workspace=_WORKSPACE, owner="worker-1", until=until
        )
        assert not await registry.acquire_write_fence(
            workspace=_WORKSPACE, owner="worker-2", until=until
        )
        assert await registry.release_write_fence(workspace=_WORKSPACE, owner="worker-1")

        # Legal-state constraints reject impossible registry states outright.
        async with pool.acquire() as conn:
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    UPDATE dlightrag_workspace_meta SET storage_tier = 'promoting'
                    WHERE workspace = $1
                    """,
                    _WORKSPACE,
                )
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    UPDATE dlightrag_workspace_meta
                    SET promotion_next_retry_at = NOW(), promotion_state = 'none'
                    WHERE workspace = $1
                    """,
                    _WORKSPACE,
                )
    finally:
        await registry.delete(_WORKSPACE)


async def test_promotion_jobs_are_idempotent_leased_and_fenced(pool: Any) -> None:
    from dlightrag.adapters.postgres.corpus.promotion_jobs import PGPromotionJobStore

    store = PGPromotionJobStore(pool=pool)
    await store.initialize()
    try:
        assert await store.enqueue(_WORKSPACE_B) is True
        assert await store.enqueue(_WORKSPACE_B) is False  # live job blocks a duplicate

        until = datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5)
        claimed = await store.claim_next(owner="worker-1", lease_until=until)
        assert claimed is not None
        assert claimed["workspace"] == _WORKSPACE_B
        assert claimed["attempt_count"] == 1

        job_id = int(claimed["job_id"])
        generation = int(claimed["lease_generation"])

        # Owner + monotonically increasing generation + lease time form the
        # fencing identity. A different owner cannot finish this attempt.
        assert not await store.mark_done(
            job_id=job_id,
            owner="worker-2",
            lease_generation=generation,
        )
        assert await store.mark_done(
            job_id=job_id,
            owner="worker-1",
            lease_generation=generation,
        )

        # Terminal jobs stay as history; a new live job can be enqueued. Failed
        # jobs remain the one live row and retry that same identity after backoff.
        assert await store.enqueue(_WORKSPACE_B) is True
        claimed = await store.claim_next(owner="worker-1", lease_until=until)
        assert claimed is not None
        retry_at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=1)
        retry_job_id = int(claimed["job_id"])
        retry_generation = int(claimed["lease_generation"])
        assert await store.mark_failed(
            job_id=retry_job_id,
            owner="worker-1",
            lease_generation=retry_generation,
            error="promotion staging verification failed",
            next_retry_at=retry_at,
        )
        assert not await store.enqueue(_WORKSPACE_B)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state, last_error, next_retry_at, attempt_count, lease_generation "
                "FROM dlightrag_promotion_jobs WHERE workspace = $1 AND state = 'failed'",
                _WORKSPACE_B,
            )
            assert row is not None
            assert row["last_error"] == "promotion staging verification failed"
            assert row["attempt_count"] == 1

            # Legal-state constraints reject impossible rows outright.
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    "INSERT INTO dlightrag_promotion_jobs (workspace, state, lease_owner) "
                    "VALUES ('pf_bad_lease', 'promoting', NULL)"
                )
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    "INSERT INTO dlightrag_promotion_jobs "
                    "(workspace, state, last_error, next_retry_at) "
                    "VALUES ('pf_bad_error', 'failed', NULL, NOW())"
                )

        reclaimed = await store.claim_next(owner="worker-1", lease_until=until)
        assert reclaimed is not None
        assert int(reclaimed["job_id"]) == retry_job_id
        assert int(reclaimed["attempt_count"]) == 2
        assert int(reclaimed["lease_generation"]) == retry_generation + 1

        # Even the same process owner cannot let its stale attempt commit after
        # the row has been reclaimed with a newer generation.
        assert not await store.mark_done(
            job_id=retry_job_id,
            owner="worker-1",
            lease_generation=retry_generation,
        )
        assert await store.mark_done(
            job_id=retry_job_id,
            owner="worker-1",
            lease_generation=int(reclaimed["lease_generation"]),
        )
    finally:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM dlightrag_promotion_jobs WHERE workspace = ANY($1::text[])",
                [_WORKSPACE_B],
            )
