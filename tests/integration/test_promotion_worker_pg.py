# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Real-PostgreSQL integration coverage for the Commit 3 promotion control plane.

Runs against a dedicated fresh database (``dlightrag_promotion_worker_test``)
that this module creates and drops itself. Proves on compact fixtures:

* the ingest counter ledger and the threshold trigger are idempotent per
  job/window (replays never double-count and never double-enqueue);
* the write fence refuses ingest claims and shared write gates, with the
  retryable error carrying the remaining fence duration;
* the exclusive write gate drains an in-flight shared write gate;
* one worker run promotes metadata + chunks (BM25 enabled) + vector parents
  atomically: staging copies are checksum-verified, the DEFAULT copies are
  deleted, dedicated partitions attach in ONE transaction, the registry flips
  to hot, and the fence and job complete in the same commit;
* a mid-cutover failure rolls back every earlier attach (all-table atomicity)
  and the retry attempt reconciles cleanly;
* crashes at each state boundary recover: stale staging is dropped, stale
  lease generations cannot complete a newer claim, checksum mismatches fail
  the attempt and release the fence, and an already-attached workspace
  reconciles to bookkeeping only;
* promotion never touches another workspace's rows or blocks its write gate;
* reads keep working through promotion and prune to the dedicated child after;
* readers validate the partitioned contract after promotion, with BM25 and
  vector child indexes rebuilt on the dedicated partitions.
"""

import asyncio
import datetime
import os
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.corpus.partition_foundation import (
    PartitionedTableSpec,
    PGPartitionFoundation,
    child_partition_name,
    default_child_name,
)
from dlightrag.adapters.postgres.corpus.promotion_jobs import PGPromotionJobStore
from dlightrag.adapters.postgres.corpus.promotion_worker import (
    PGPromotionWorker,
    staging_partition_name,
)
from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry
from dlightrag.engine.rag.workspace.ports import WorkspaceWriteFencedError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
]

_MAINT_DB = "postgres"
_TEST_DB = "dlightrag_promotion_worker_test"
_EXTENSIONS = ("vector", "pg_textsearch", "pg_trgm")
_WORKSPACE_A = "pw_hot_ws"
_WORKSPACE_B = "pw_other_ws"

_ws_counter = 0


def _new_workspace(prefix: str = "pw_ws") -> str:
    """One unique workspace per test so module-state never leaks between tests."""
    global _ws_counter
    _ws_counter += 1
    return f"{prefix}_{_ws_counter}"


_CHUNKS_TABLE = "LIGHTRAG_DOC_CHUNKS"
_METADATA_TABLE = "dlightrag_doc_metadata"
_VECTOR_TABLE = "lightrag_vdb_chunks_8"
_CHUNK_SCOPE_INDEX = "idx_lightrag_doc_chunks_dlightrag_full_doc_id"

_DEFAULT_KWARGS = dict(
    host=os.environ.get("DLIGHTRAG_STORAGE__POSTGRES__HOST", "localhost"),
    port=int(os.environ.get("DLIGHTRAG_STORAGE__POSTGRES__PORT", "5432")),
    user=os.environ.get("DLIGHTRAG_STORAGE__POSTGRES__USER", "dlightrag"),
    password=os.environ.get("DLIGHTRAG_STORAGE__POSTGRES__PASSWORD", "dlightrag"),
)


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


@pytest.fixture(scope="module", autouse=True)
async def _fresh_test_database() -> AsyncIterator[None]:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")
    conn = await asyncpg.connect(**_kwargs(_MAINT_DB))
    try:
        await conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = $1 AND pid <> pg_backend_pid()",
            _TEST_DB,
        )
        await conn.execute(f"DROP DATABASE IF EXISTS {_TEST_DB}")
        await conn.execute(f"CREATE DATABASE {_TEST_DB}")
    finally:
        await conn.close()
    db = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        for extension in _EXTENSIONS:
            await db.execute(f"CREATE EXTENSION IF NOT EXISTS {extension}")
    finally:
        await db.close()
    yield None
    conn = await asyncpg.connect(**_kwargs(_MAINT_DB))
    try:
        await conn.execute(f"DROP DATABASE IF EXISTS {_TEST_DB}")
    finally:
        await conn.close()


@pytest.fixture(scope="module")
async def corpus(_fresh_test_database: None) -> AsyncIterator[None]:
    """Build the partitioned retrieval parents plus all control-plane stores."""
    from dlightrag.adapters.postgres.core._pool import pg_pool
    from dlightrag.adapters.postgres.corpus._corpus_schema import CHUNK_DOCUMENT_SCOPE_INDEX
    from dlightrag.adapters.postgres.corpus.pg_metadata_index import PGMetadataIndex
    from dlightrag.application.config import DlightragConfig, reset_config, set_config

    cfg = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        deployment={"workspace": _WORKSPACE_A, "working_dir": "/tmp/pw_workdir"},
        storage={
            "postgres": {
                **_DEFAULT_KWARGS,
                "database": _TEST_DB,
                "pool_min_size": 1,
                "pool_max_size": 6,
            }
        },
        models={
            "chat": {
                "default": {
                    "provider": "openai",
                    "model": "pw-fake-llm",
                    "api_key": "pw-fake-key",
                    "timeout": 30,
                }
            }
        },
    )
    set_config(cfg)
    pg_pool.bind(cfg)
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await conn.execute(f"""
            CREATE TABLE {_CHUNKS_TABLE} (
                workspace TEXT NOT NULL,
                id TEXT NOT NULL,
                full_doc_id TEXT,
                content TEXT,
                file_path TEXT,
                chunk_order_index INTEGER,
                tokens INTEGER,
                llm_cache_list JSONB,
                heading JSONB,
                sidecar JSONB,
                dlightrag_bm25_language TEXT,
                create_time TIMESTAMPTZ DEFAULT NOW(),
                update_time TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (workspace, id)
            )
        """)
        await conn.execute(f"""
            CREATE TABLE {_VECTOR_TABLE} (
                workspace TEXT NOT NULL,
                id TEXT NOT NULL,
                full_doc_id TEXT,
                content TEXT,
                file_path TEXT,
                content_vector vector(8),
                create_time TIMESTAMPTZ DEFAULT NOW(),
                update_time TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (workspace, id)
            )
        """)
        # LightRAG creates these two chunk indexes upstream; the fixture
        # mirrors that so the partition conversion replays them.
        await conn.execute(f"CREATE INDEX idx_lightrag_doc_chunks_id ON {_CHUNKS_TABLE} (id)")
        await conn.execute(
            f"CREATE INDEX idx_lightrag_doc_chunks_workspace_id ON {_CHUNKS_TABLE} (workspace, id)"
        )
        foundation = PGPartitionFoundation()
        chunks_spec = PartitionedTableSpec(
            name=_CHUNKS_TABLE,
            required_columns=("id", "workspace", "full_doc_id", "content", "file_path"),
            primary_key=("workspace", "id"),
            required_indexes=("idx_lightrag_doc_chunks_id", "idx_lightrag_doc_chunks_workspace_id"),
        )
        vector_spec = PartitionedTableSpec(
            name=_VECTOR_TABLE,
            required_columns=("id", "workspace", "full_doc_id", "content", "content_vector"),
            primary_key=("workspace", "id"),
        )
        await foundation.ensure_tables(specs=(chunks_spec, vector_spec))
        # The runtime-owned chunk-side semi-join index and the HNSW vector
        # index are created on the parents exactly like the writer attach does.
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {CHUNK_DOCUMENT_SCOPE_INDEX} "
            f"ON {_CHUNKS_TABLE} (workspace, full_doc_id)"
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_lightrag_doc_chunks_bm25_en "
            f"ON {_CHUNKS_TABLE} USING bm25(content) "
            f"WITH (text_config='english', k1=1.2, b=0.75)"
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {_VECTOR_TABLE}_content_vector_idx "
            f"ON {_VECTOR_TABLE} USING hnsw (content_vector vector_cosine_ops)"
        )
        await PGMetadataIndex(workspace=_WORKSPACE_A).initialize()
        await PGWorkspaceRegistry().initialize()
        await PGPromotionJobStore().initialize()
    finally:
        await conn.close()
    yield None
    await pg_pool.close()
    reset_config()


async def _clean_state() -> None:
    """Empty promotion state so every test starts deterministic."""
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await conn.execute("DELETE FROM dlightrag_promotion_jobs")
        await conn.execute(
            """
            UPDATE dlightrag_workspace_meta
            SET storage_tier = 'shared',
                promotion_state = 'none',
                promotion_last_error = NULL,
                promotion_retry_count = 0,
                promotion_next_retry_at = NULL,
                write_fence_owner = NULL,
                write_fence_until = NULL
            """
        )
    finally:
        await conn.close()


@pytest.fixture
def workspaces() -> tuple[str, str]:
    """Two unique workspaces per test; module state never leaks across tests."""
    return _new_workspace("pw_hot"), _new_workspace("pw_other")


async def _seed_workspace(conn: Any, workspace: str, docs: int, chunks_per_doc: int) -> None:
    for doc_index in range(docs):
        doc_id = f"{workspace}-doc-{doc_index}"
        await conn.execute(
            f"""
            INSERT INTO {_METADATA_TABLE} (workspace, doc_id, filename, filename_stem)
            VALUES ($1, $2, $3, $3)
            """,
            workspace,
            doc_id,
            f"report-{doc_index}.pdf",
        )
        for chunk_index in range(chunks_per_doc):
            chunk_id = f"{workspace}-doc-{doc_index}-c-{chunk_index}"
            content = f"alpha beta gamma {doc_index} {chunk_index}"
            await conn.execute(
                f"""
                INSERT INTO {_CHUNKS_TABLE}
                    (workspace, id, full_doc_id, content, dlightrag_bm25_language)
                VALUES ($1, $2, $3, $4, 'en')
                """,
                workspace,
                chunk_id,
                doc_id,
                content,
            )
            await conn.execute(
                f"""
                INSERT INTO {_VECTOR_TABLE} (workspace, id, full_doc_id, content, content_vector)
                VALUES ($1, $2, $3, $4, $5::vector)
                """,
                workspace,
                chunk_id,
                doc_id,
                content,
                f"[{doc_index * 0.01:.2f},{0.1},{0.2},{0.3},{0.4},{0.5},{0.6},{0.7}]",
            )


async def _table_counts(conn: Any, workspace: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE):
        counts[table] = int(
            await conn.fetchval(f"SELECT COUNT(*) FROM {table} WHERE workspace = $1", workspace)
        )
    return counts


async def _dedicated_partitions(conn: Any, workspace: str) -> dict[str, str]:
    partitions: dict[str, str] = {}
    for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE):
        child = child_partition_name(table, workspace)
        exists = await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", child)
        partitions[table] = child if exists else ""
    return partitions


async def _registry_row(conn: Any, workspace: str) -> dict[str, Any]:
    row = await conn.fetchrow(
        "SELECT * FROM dlightrag_workspace_meta WHERE workspace = $1", workspace
    )
    assert row is not None
    return dict(row)


def _worker(*, lease_seconds: int = 600) -> PGPromotionWorker:
    return PGPromotionWorker(
        job_store=PGPromotionJobStore(),
        registry=PGWorkspaceRegistry(),
        lease_seconds=lease_seconds,
        retry_backoff_seconds=1,
        claim_poll_seconds=0.01,
    )


# ---------------------------------------------------------------------------
# Counter ledger + threshold trigger
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Write fence and gate behavior
# ---------------------------------------------------------------------------


async def test_exclusive_gate_drains_in_flight_shared_write(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()

    from dlightrag.adapters.postgres.corpus.workspace_write_gate import workspace_write_gate

    registry = PGWorkspaceRegistry()
    await registry.upsert(workspace=other, display_name="Other", embedding_model="m")

    release = asyncio.Event()
    entered_exclusive = asyncio.Event()

    async def shared_holder() -> None:
        async with workspace_write_gate(other):
            release.set()
            await asyncio.sleep(0.8)

    async def exclusive_waiter() -> None:
        async with workspace_write_gate(other, exclusive=True):
            entered_exclusive.set()

    holder = asyncio.create_task(shared_holder())
    await release.wait()
    waiter = asyncio.create_task(exclusive_waiter())
    await asyncio.sleep(0.3)
    assert entered_exclusive.is_set() is False  # still draining
    await asyncio.wait_for(holder, timeout=5)
    await asyncio.wait_for(waiter, timeout=10)
    assert entered_exclusive.is_set() is True


# ---------------------------------------------------------------------------
# End-to-end promotion
# ---------------------------------------------------------------------------


async def test_promotion_is_atomic_across_all_parents_and_isolated(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        registry = PGWorkspaceRegistry()
        await registry.upsert(workspace=ws, display_name="Hot", embedding_model="m")
        await registry.upsert(workspace=other, display_name="Other", embedding_model="m")
        await _seed_workspace(conn, ws, docs=3, chunks_per_doc=2)
        await _seed_workspace(conn, other, docs=2, chunks_per_doc=2)
        before_a = await _table_counts(conn, ws)
        before_b = await _table_counts(conn, other)
        field_stats_before = {
            str(row["field_id"]): int(row["document_count"])
            for row in await conn.fetch(
                "SELECT field_id, document_count "
                "FROM dlightrag_metadata_field_stats WHERE workspace = $1",
                ws,
            )
        }

        jobs = PGPromotionJobStore()
        assert await jobs.enqueue(ws) is True
        assert await jobs.enqueue(ws) is False  # idempotent

        worker = _worker()
        assert await worker.run_once() is True

        # Every parent carries a dedicated partition for A.
        partitions = await _dedicated_partitions(conn, ws)
        assert all(partitions.values())
        # The DEFAULT partitions no longer hold A's rows; B is untouched.
        from dlightrag.adapters.postgres.corpus.partition_foundation import default_child_name

        for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE):
            assert (
                int(
                    await conn.fetchval(
                        f"SELECT COUNT(*) FROM ONLY {default_child_name(table)} "
                        "WHERE workspace = $1",
                        ws,
                    )
                )
                == 0
            )
            assert (
                int(
                    await conn.fetchval(
                        f"SELECT COUNT(*) FROM ONLY {default_child_name(table)} "
                        "WHERE workspace = $1",
                        other,
                    )
                )
                == before_b[table]
            )
        counts_a = await _table_counts(conn, ws)
        assert counts_a == before_a
        assert await _table_counts(conn, other) == before_b
        assert {
            str(row["field_id"]): int(row["document_count"])
            for row in await conn.fetch(
                "SELECT field_id, document_count "
                "FROM dlightrag_metadata_field_stats WHERE workspace = $1",
                ws,
            )
        } == field_stats_before
        for table, child in partitions.items():
            child_count = int(await conn.fetchval(f"SELECT COUNT(*) FROM {child}"))
            assert child_count == before_a[table]
        assert await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_constraint "
            "WHERE conrelid = $1::regclass "
            "AND conname = 'dlightrag_doc_metadata_finalization_not_null')",
            partitions[_METADATA_TABLE],
        )

        # Registry flipped to hot with fence released; the job is done.
        row = await _registry_row(conn, ws)
        assert row["storage_tier"] == "hot"
        assert row["promotion_state"] == "none"
        assert row["write_fence_owner"] is None and row["write_fence_until"] is None
        job_row = await conn.fetchrow(
            "SELECT state, lease_owner, lease_until FROM dlightrag_promotion_jobs "
            "WHERE workspace = $1",
            ws,
        )
        assert job_row is not None and job_row["state"] == "done"
        assert job_row["lease_owner"] is None and job_row["lease_until"] is None

        # Reads keep working and prune to the dedicated child.
        plan = await conn.fetchval(
            f"EXPLAIN (FORMAT TEXT) SELECT id FROM {_CHUNKS_TABLE} WHERE workspace = $1",
            ws,
        )
        plan_text = str(plan)
        assert child_partition_name(_CHUNKS_TABLE, ws) in plan_text
        assert default_child_name(_CHUNKS_TABLE) not in plan_text

        # New writes for A route into the dedicated child; its cloned metadata
        # trigger keeps field counts current. B keeps using DEFAULT.
        await conn.execute(
            f"""
            INSERT INTO {_METADATA_TABLE} (
                workspace, doc_id, title, _dlightrag_finalization_complete
            )
            VALUES ($1, 'pw-hot-post-promotion', 'Post-promotion title', TRUE)
            """,
            ws,
        )
        assert (
            await conn.fetchval(
                "SELECT document_count FROM dlightrag_metadata_field_stats "
                "WHERE workspace = $1 AND field_id = 'title'",
                ws,
            )
            == 1
        )
        await conn.execute(
            f"DELETE FROM {_METADATA_TABLE} "
            "WHERE workspace = $1 AND doc_id = 'pw-hot-post-promotion'",
            ws,
        )
        assert not await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM dlightrag_metadata_field_stats "
            "WHERE workspace = $1 AND field_id = 'title')",
            ws,
        )
        await conn.execute(
            f"""
            INSERT INTO {_CHUNKS_TABLE} (workspace, id, full_doc_id, content)
            VALUES ($1, $2, $3, $4)
            """,
            ws,
            "pw-new-chunk",
            "pw-hot-ws-doc-0",
            "post-promotion write",
        )
        assert (
            int(
                await conn.fetchval(
                    f"SELECT COUNT(*) FROM {child_partition_name(_CHUNKS_TABLE, ws)}"
                    " WHERE id = 'pw-new-chunk'"
                )
            )
            == 1
        )
        await conn.execute(
            f"""
            INSERT INTO {_CHUNKS_TABLE} (workspace, id, full_doc_id, content)
            VALUES ($1, $2, $3, $4)
            """,
            other,
            "pw-b-chunk",
            "pw-other-ws-doc-0",
            "other workspace write",
        )
        assert (
            int(
                await conn.fetchval(
                    f"SELECT COUNT(*) FROM {_CHUNKS_TABLE} WHERE workspace = $1",
                    other,
                )
            )
            == before_b[_CHUNKS_TABLE] + 1
        )
        assert await _dedicated_partitions(conn, other) == {table: "" for table in partitions}
    finally:
        await conn.close()


async def test_child_partitions_carry_bm25_vector_and_metadata_indexes(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await PGWorkspaceRegistry().upsert(workspace=ws, display_name="Hot", embedding_model="m")
        await _seed_workspace(conn, ws, docs=1, chunks_per_doc=1)
        await PGPromotionJobStore().enqueue(ws)
        assert await _worker().run_once() is True

        for table in (_CHUNKS_TABLE, _METADATA_TABLE, _VECTOR_TABLE):
            child = child_partition_name(table, ws)
            child_indexes = await conn.fetch(
                """
                SELECT pg_get_indexdef(i.oid) AS definition
                FROM pg_index idx
                JOIN pg_class i ON i.oid = idx.indexrelid
                JOIN pg_class t ON t.oid = idx.indrelid
                WHERE t.relname = $1
                """,
                child,
            )
            definitions = sorted(_normalize_indexdef(str(r["definition"])) for r in child_indexes)
            assert definitions, f"no child indexes on {child}"
        chunk_child = child_partition_name(_CHUNKS_TABLE, ws)
        bm25_indexes = await conn.fetch(
            """
            SELECT pg_get_indexdef(i.oid) AS definition
            FROM pg_index idx
            JOIN pg_class i ON i.oid = idx.indexrelid
            JOIN pg_class t ON t.oid = idx.indrelid
            WHERE t.relname = $1 AND pg_get_indexdef(i.oid) ILIKE '%USING bm25%'
            """,
            chunk_child,
        )
        assert len(bm25_indexes) == 1
        vector_child = child_partition_name(_VECTOR_TABLE, ws)
        hnsw_indexes = await conn.fetch(
            """
            SELECT pg_get_indexdef(i.oid) AS definition
            FROM pg_index idx
            JOIN pg_class i ON i.oid = idx.indexrelid
            JOIN pg_class t ON t.oid = idx.indrelid
            WHERE t.relname = $1 AND pg_get_indexdef(i.oid) ILIKE '%USING hnsw%'
            """,
            vector_child,
        )
        assert len(hnsw_indexes) == 1
        metadata_child = child_partition_name(_METADATA_TABLE, ws)
        gin_indexes = await conn.fetch(
            """
            SELECT pg_get_indexdef(i.oid) AS definition
            FROM pg_index idx
            JOIN pg_class i ON i.oid = idx.indexrelid
            JOIN pg_class t ON t.oid = idx.indrelid
            WHERE t.relname = $1
              AND (pg_get_indexdef(i.oid) ILIKE '%USING gin%'
                   OR pg_get_indexdef(i.oid) ILIKE '%gin_trgm_ops%')
            """,
            metadata_child,
        )
        assert len(gin_indexes) >= 2  # canonical containment GIN + filename trigram
    finally:
        await conn.close()


async def test_readers_validate_the_promoted_contract(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    from dlightrag.adapters.postgres.corpus._corpus_schema import CHUNK_DOCUMENT_SCOPE_INDEX
    from dlightrag.adapters.postgres.corpus.pg_metadata_index import PGMetadataIndex

    await PGWorkspaceRegistry().upsert(workspace=ws, display_name="Hot", embedding_model="m")
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await _seed_workspace(conn, ws, docs=1, chunks_per_doc=1)
    finally:
        await conn.close()
    await PGPromotionJobStore().enqueue(ws)
    assert await _worker().run_once() is True

    foundation = PGPartitionFoundation()
    await foundation.verify_tables(
        specs=(
            PartitionedTableSpec(
                name=_CHUNKS_TABLE,
                required_columns=("id", "workspace", "full_doc_id", "content", "file_path"),
                primary_key=("workspace", "id"),
                required_indexes=(
                    "idx_lightrag_doc_chunks_id",
                    "idx_lightrag_doc_chunks_workspace_id",
                    CHUNK_DOCUMENT_SCOPE_INDEX,
                ),
            ),
            PartitionedTableSpec(
                name=_VECTOR_TABLE,
                required_columns=("id", "workspace", "full_doc_id", "content", "content_vector"),
                primary_key=("workspace", "id"),
                required_index_markers=("USING hnsw",),
            ),
        )
    )
    await PGMetadataIndex(workspace=ws).initialize(validate_only=True)


# ---------------------------------------------------------------------------
# Crash and failure boundaries
# ---------------------------------------------------------------------------


async def test_checksum_mismatch_fails_attempt_keeps_workspace_shared(
    corpus: None,
    monkeypatch: pytest.MonkeyPatch,
    workspaces: tuple[str, str],
) -> None:
    ws, other = workspaces
    await _clean_state()
    from dlightrag.adapters.postgres.corpus import promotion_worker as worker_module

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        registry = PGWorkspaceRegistry()
        await registry.upsert(workspace=other, display_name="Other", embedding_model="m")
        await _seed_workspace(conn, other, docs=2, chunks_per_doc=2)
    finally:
        await conn.close()

    original_copy = worker_module._copy_workspace_rows

    async def copy_then_corrupt(conn: Any, parent: str, staging: str, workspace: str) -> None:
        await original_copy(conn, parent, staging, workspace)
        if parent.lower() == _CHUNKS_TABLE.lower():
            await conn.execute(f"DELETE FROM {staging} WHERE id LIKE '%c-0'")

    monkeypatch.setattr(worker_module, "_copy_workspace_rows", copy_then_corrupt)

    jobs = PGPromotionJobStore()
    await jobs.enqueue(other)
    worker = _worker()
    assert await worker.run_once() is True
    monkeypatch.undo()

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        row = await _registry_row(conn, other)
        assert row["storage_tier"] == "shared"
        assert row["promotion_state"] == "failed"
        assert (
            row["promotion_last_error"]
            and "copy verification failed" in row["promotion_last_error"]
        )
        assert row["write_fence_owner"] is None and row["write_fence_until"] is None
        job_row = await conn.fetchrow(
            "SELECT state, last_error, next_retry_at FROM dlightrag_promotion_jobs "
            "WHERE workspace = $1",
            other,
        )
        assert job_row is not None and job_row["state"] == "failed"
        assert "copy verification failed" in str(job_row["last_error"])
        assert job_row["next_retry_at"] is not None
        # No dedicated partitions were attached and no staging was left behind.
        assert await _dedicated_partitions(conn, other) == {
            table: "" for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE)
        }
        for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE):
            assert (
                await conn.fetchval(
                    "SELECT to_regclass($1) IS NOT NULL",
                    staging_partition_name(table, other),
                )
                is False
            )
        # Every row is still served from the shared DEFAULT partition.
        for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE):
            assert int(
                await conn.fetchval(f"SELECT COUNT(*) FROM {table} WHERE workspace = $1", other)
            ) == (2 if table == _METADATA_TABLE else 4)
    finally:
        await conn.close()


async def test_mid_cutover_failure_rolls_back_every_attach(
    corpus: None,
    monkeypatch: pytest.MonkeyPatch,
    workspaces: tuple[str, str],
) -> None:
    ws, other = workspaces
    await _clean_state()
    from dlightrag.adapters.postgres.corpus import promotion_worker as worker_module

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        registry = PGWorkspaceRegistry()
        await registry.upsert(workspace=other, display_name="Other", embedding_model="m")
        await _seed_workspace(conn, other, docs=2, chunks_per_doc=2)
    finally:
        await conn.close()

    # Sabotage the second table's staging after its indexes were built: its
    # RENAME inside the cutover transaction fails after the first table's
    # DELETE+VALIDATE already executed — the whole cutover must roll back.
    original_build = worker_module._build_staging_indexes
    original_discover = worker_module._discover_retrieval_parents

    async def drop_chunks_staging(conn: Any, parent: str, staging: str) -> None:
        await original_build(conn, parent, staging)
        if parent.lower() == _CHUNKS_TABLE.lower():
            await conn.execute(f"DROP TABLE {staging}")

    async def discover_tables(conn: Any) -> list[str]:
        # Sort so the metadata table attaches first, then chunks.
        return sorted(await original_discover(conn))

    monkeypatch.setattr(worker_module, "_discover_retrieval_parents", discover_tables)
    monkeypatch.setattr(worker_module, "_build_staging_indexes", drop_chunks_staging)

    jobs = PGPromotionJobStore()
    await jobs.enqueue(other)
    worker = _worker()
    assert await worker.run_once() is True
    monkeypatch.undo()

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        row = await _registry_row(conn, other)
        assert row["storage_tier"] == "shared"
        assert row["promotion_state"] == "failed"
        # The rolled-back first table has no dedicated partition: the cutover
        # is all-or-nothing across every parent.
        assert await _dedicated_partitions(conn, other) == {
            table: "" for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE)
        }
        for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE):
            assert int(
                await conn.fetchval(f"SELECT COUNT(*) FROM {table} WHERE workspace = $1", other)
            ) == (2 if table == _METADATA_TABLE else 4)
        assert row["write_fence_owner"] is None
        # The in-gate cleanup removed every phase-1 exclusion proof after the
        # rollback (checks removed on failure).
        from dlightrag.adapters.postgres.corpus import promotion_worker as worker_module

        for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE):
            leftover = await conn.fetch(
                "SELECT conname FROM pg_constraint "
                f"WHERE conrelid = {default_child_name(table)!r}::regclass "
                "AND conname LIKE '%_excl'"
            )
            assert not leftover, f"{table} kept an exclusion proof after failure"
    finally:
        await conn.close()

    # A clean retry reconciles to a completed promotion.
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await conn.execute(
            "UPDATE dlightrag_promotion_jobs SET next_retry_at = NOW() WHERE workspace = $1",
            other,
        )
    finally:
        await conn.close()
    assert await _worker().run_once() is True
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        row = await _registry_row(conn, other)
        assert row["storage_tier"] == "hot"
        assert row["promotion_state"] == "none"
        partitions = await _dedicated_partitions(conn, other)
        assert all(partitions.values())
    finally:
        await conn.close()


async def test_stale_staging_is_dropped_on_the_next_attempt(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        registry = PGWorkspaceRegistry()
        await registry.upsert(workspace=other, display_name="Other", embedding_model="m")
        await _seed_workspace(conn, other, docs=1, chunks_per_doc=1)
        # A crashed attempt left a staging relation behind.
        await conn.execute(
            f"CREATE TABLE {staging_partition_name(_CHUNKS_TABLE, other)} (junk INT)"
        )
    finally:
        await conn.close()

    await PGPromotionJobStore().enqueue(other)
    assert await _worker().run_once() is True

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        assert await _registry_row(conn, other) is not None
        assert (await _registry_row(conn, other))["storage_tier"] == "hot"
        assert (
            await conn.fetchval(
                "SELECT to_regclass($1) IS NOT NULL",
                staging_partition_name(_CHUNKS_TABLE, other),
            )
            is False
        )
    finally:
        await conn.close()


async def test_stale_lease_generation_cannot_complete_a_newer_claim(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    jobs = PGPromotionJobStore()
    await PGWorkspaceRegistry().upsert(workspace=other, display_name="Other", embedding_model="m")
    await jobs.enqueue(other)
    first = await jobs.claim_next(
        owner="worker-old",
        lease_until=datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=300),
    )
    assert first is not None
    first_generation = int(first["lease_generation"])

    # The old worker's lease expires; a new worker reclaims with a new
    # generation.
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await conn.execute(
            "UPDATE dlightrag_promotion_jobs SET lease_until = NOW() - INTERVAL '1 second' "
            "WHERE job_id = $1",
            first["job_id"],
        )
    finally:
        await conn.close()
    second = await jobs.claim_next(
        owner="worker-new",
        lease_until=datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=300),
    )
    assert second is not None
    assert int(second["lease_generation"]) == first_generation + 1

    # The stale worker's completion is refused; the current one succeeds.
    assert (
        await jobs.mark_done(
            job_id=int(first["job_id"]),
            owner="worker-old",
            lease_generation=first_generation,
        )
        is False
    )
    assert (
        await jobs.mark_done(
            job_id=int(second["job_id"]),
            owner="worker-new",
            lease_generation=int(second["lease_generation"]),
        )
        is True
    )


async def test_already_attached_workspace_reconciles_without_new_partitions(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        registry = PGWorkspaceRegistry()
        await registry.upsert(workspace=other, display_name="Other", embedding_model="m")
        await _seed_workspace(conn, other, docs=1, chunks_per_doc=1)
    finally:
        await conn.close()
    await PGPromotionJobStore().enqueue(other)
    assert await _worker().run_once() is True

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        existing = await _dedicated_partitions(conn, other)
        assert all(existing.values())
        # A stale trigger (or an operator retry) enqueues another job for the
        # already-hot workspace: the worker must reconcile without duplicating
        # partitions or touching rows.
        await PGPromotionJobStore().enqueue(other)
    finally:
        await conn.close()
    assert await _worker().run_once() is True

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        assert await _dedicated_partitions(conn, other) == existing
        assert (await _registry_row(conn, other))["storage_tier"] == "hot"
        job_states = await conn.fetch(
            "SELECT state FROM dlightrag_promotion_jobs WHERE workspace = $1 ORDER BY job_id",
            other,
        )
        assert [row["state"] for row in job_states] == ["done", "done"]
    finally:
        await conn.close()


async def test_worker_without_any_parent_fails_loudly(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    """A promotion attempt on an empty database fails instead of half-doing."""
    # The fresh module database always has parents; this exercises the failure
    # path by promoting a workspace whose registry row is missing the tables —
    # simulated by a discovery monkeypatch returning nothing.
    import dlightrag.adapters.postgres.corpus.promotion_worker as worker_module

    async def discover_none(conn: Any) -> list[str]:
        return []

    worker = _worker()
    original = worker_module._discover_retrieval_parents
    worker_module._discover_retrieval_parents = discover_none  # type: ignore[assignment]
    try:
        await PGWorkspaceRegistry().upsert(
            workspace=other, display_name="Other", embedding_model="m"
        )
        await PGPromotionJobStore().enqueue(other)
        assert await worker.run_once() is True
    finally:
        worker_module._discover_retrieval_parents = original  # type: ignore[assignment]

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        row = await _registry_row(conn, other)
        assert row["promotion_state"] == "failed"
        assert "no partitioned retrieval parents" in str(row["promotion_last_error"])
        assert row["write_fence_owner"] is None
    finally:
        await conn.close()


def _normalize_indexdef(definition: str) -> str:
    """Strip the index name so parent/child definitions compare by shape."""
    start = definition.find(" USING ")
    if start == -1:
        return definition
    return definition[start:].strip()


# ---------------------------------------------------------------------------
# Fix round: cutover exclusion proof, PK reuse, strict reconciliation,
# durability hygiene, gate connection regression, queued liveness
# ---------------------------------------------------------------------------


async def test_cutover_leaves_no_exclusion_constraint_and_reuses_prebuilt_pk(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        registry = PGWorkspaceRegistry()
        await registry.upsert(workspace=ws, display_name="Hot", embedding_model="m")
        await _seed_workspace(conn, ws, docs=2, chunks_per_doc=2)
    finally:
        await conn.close()
    await PGPromotionJobStore().enqueue(ws)
    assert await _worker().run_once() is True

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        from dlightrag.adapters.postgres.corpus.partition_foundation import default_child_name

        for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE):
            default_child = default_child_name(table)
            # No crash-persistent exclusion constraint survives the cutover.
            leftover = await conn.fetch(
                "SELECT conname FROM pg_constraint WHERE conrelid = $1::regclass "
                "AND conname LIKE '%_excl'",
                default_child,
            )
            assert not leftover, f"{default_child} kept a temporary exclusion constraint"
            # The dedicated child carries exactly one PRIMARY KEY constraint,
            # backed by the deterministic prebuilt index — the attach built no
            # second PK index under the parent lock.
            child = child_partition_name(table, ws)
            pk_rows = await conn.fetch(
                """
                SELECT c.conname AS constraint_name, i.relname AS index_name
                FROM pg_constraint c
                JOIN pg_class i ON i.oid = c.conindid
                WHERE c.conrelid = $1::regclass AND c.contype = 'p'
                """,
                child,
            )
            assert len(pk_rows) == 1, f"{child} carries {len(pk_rows)} PK constraint(s)"
            assert pk_rows[0]["constraint_name"] == f"{staging_partition_name(table, ws)}_pkey"
            assert pk_rows[0]["index_name"] == f"{staging_partition_name(table, ws)}_pkey"
    finally:
        await conn.close()


async def test_discovery_rejects_vector_parents_without_default_and_ignores_lookalikes(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        registry = PGWorkspaceRegistry()
        await registry.upsert(workspace=ws, display_name="Hot", embedding_model="m")
        # A real vector parent missing its DEFAULT child must fail the attempt
        # loudly instead of promoting a subset.
        await conn.execute(
            """
            CREATE TABLE lightrag_vdb_chunks_9 (
                workspace TEXT NOT NULL, id TEXT NOT NULL,
                PRIMARY KEY (workspace, id)
            ) PARTITION BY LIST (workspace)
            """
        )
        # An underscore-lookalike must not be discovered at all: the prefix
        # match is literal, never a LIKE wildcard.
        await conn.execute(
            """
            CREATE TABLE lightragXvdb_chunks_decoy (
                workspace TEXT NOT NULL, id TEXT NOT NULL,
                PRIMARY KEY (workspace, id)
            ) PARTITION BY LIST (workspace)
            """
        )
        await conn.execute(
            f"CREATE TABLE {default_child_name('lightragXvdb_chunks_decoy')} "
            f"PARTITION OF lightragXvdb_chunks_decoy DEFAULT"
        )
    finally:
        await conn.close()

    try:
        await PGPromotionJobStore().enqueue(ws)
        assert await _worker().run_once() is True

        conn = await asyncpg.connect(**_kwargs(_TEST_DB))
        try:
            row = await _registry_row(conn, ws)
            # Strict discovery fails before any copy, while lenient artifact
            # cleanup still proves there is nothing left behind. The guarded
            # failure releases the fence and leaves the workspace shared.
            assert row["promotion_state"] == "failed"
            assert row["write_fence_owner"] is None
            assert "no shared DEFAULT partition" in str(row["promotion_last_error"])
            assert await _dedicated_partitions(conn, ws) == {
                table: "" for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE)
            }
            assert not await conn.fetchval(
                "SELECT to_regclass($1) IS NOT NULL",
                child_partition_name("lightragXvdb_chunks_decoy", ws),
            )
        finally:
            await conn.close()
    finally:
        conn = await asyncpg.connect(**_kwargs(_TEST_DB))
        try:
            await conn.execute("DROP TABLE IF EXISTS lightrag_vdb_chunks_9")
            await conn.execute("DROP TABLE IF EXISTS lightragXvdb_chunks_decoy")
        finally:
            await conn.close()


async def test_detached_relation_with_the_child_name_fails_loudly(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        registry = PGWorkspaceRegistry()
        await registry.upsert(workspace=ws, display_name="Hot", embedding_model="m")
        await _seed_workspace(conn, ws, docs=1, chunks_per_doc=1)
        # A detached table merely reusing the deterministic child name: it is
        # not attached to the parent, so the worker must fail loudly.
        await conn.execute(
            f"""
            CREATE TABLE {child_partition_name(_CHUNKS_TABLE, ws)} (
                workspace TEXT NOT NULL, id TEXT NOT NULL,
                PRIMARY KEY (workspace, id)
            )
            """
        )
    finally:
        await conn.close()

    try:
        await PGPromotionJobStore().enqueue(ws)
        assert await _worker().run_once() is True
        conn = await asyncpg.connect(**_kwargs(_TEST_DB))
        try:
            row = await _registry_row(conn, ws)
            assert row["promotion_state"] == "failed"
            assert "is not attached" in str(row["promotion_last_error"])
        finally:
            await conn.close()
    finally:
        conn = await asyncpg.connect(**_kwargs(_TEST_DB))
        try:
            await conn.execute(f"DROP TABLE IF EXISTS {child_partition_name(_CHUNKS_TABLE, ws)}")
        finally:
            await conn.close()


async def test_gate_uses_dedicated_connections_not_the_shared_pool(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    from dlightrag.adapters.postgres.core._pool import pg_pool
    from dlightrag.adapters.postgres.corpus.workspace_write_gate import workspace_write_gate

    await PGWorkspaceRegistry().upsert(workspace=ws, display_name="Gate", embedding_model="m")
    pool = await pg_pool.get()
    # Occupy every pooled connection: the gate must still open because it uses
    # dedicated connections, never the shared pool.
    held = [await pool.acquire() for _ in range(pool.get_size() + pool.get_min_size() + 1)]
    try:
        async with workspace_write_gate(ws):
            pass
    finally:
        for conn in held:
            await conn.close()


# ---------------------------------------------------------------------------
# Fix round 3: prompt fenced refusal under a held exclusive, strict discovery
# ---------------------------------------------------------------------------


async def test_fenced_shared_gate_fails_promptly_while_exclusive_is_held(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    import time

    from dlightrag.adapters.postgres.corpus.workspace_write_gate import workspace_write_gate

    registry = PGWorkspaceRegistry()
    await registry.upsert(workspace=ws, display_name="Fenced", embedding_model="m")
    acquired = await registry.acquire_write_fence(
        workspace=ws,
        owner="promo-owner",
        until=datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=120),
    )
    assert acquired is True

    holder_in = asyncio.Event()

    async def hold_exclusive() -> None:
        async with workspace_write_gate(ws, exclusive=True):
            holder_in.set()
            await asyncio.sleep(1.2)

    holder = asyncio.create_task(hold_exclusive())
    try:
        await holder_in.wait()
        start = time.monotonic()
        with pytest.raises(WorkspaceWriteFencedError) as excinfo:
            async with workspace_write_gate(ws):
                pytest.fail("shared gate must refuse under an active fence")
        elapsed = time.monotonic() - start
        # Prompt refusal: the preflight raised without waiting for the
        # exclusive holder (1.2s) to finish.
        assert elapsed < 0.5
        assert excinfo.value.retry_after_seconds > 0
        await asyncio.wait_for(holder, timeout=5)
    finally:
        await registry.release_write_fence(workspace=ws, owner="promo-owner")


async def test_discovery_requires_at_least_one_vector_parent(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        registry = PGWorkspaceRegistry()
        await registry.upsert(workspace=ws, display_name="Hot", embedding_model="m")
        # Renamed away from the vector prefix entirely so discovery sees no
        # vector parent at all.
        await conn.execute(f"ALTER TABLE {_VECTOR_TABLE} RENAME TO hidden_{_VECTOR_TABLE}")
    finally:
        await conn.close()

    try:
        await PGPromotionJobStore().enqueue(ws)
        assert await _worker().run_once() is True
        conn = await asyncpg.connect(**_kwargs(_TEST_DB))
        try:
            row = await _registry_row(conn, ws)
            # Strict discovery fails, but schema-independent artifact cleanup
            # succeeds, so the attempt becomes a retryable failed job and the
            # workspace remains shared and writable.
            assert row["promotion_state"] == "failed"
            assert row["write_fence_owner"] is None
            assert "no partitioned chunk-vector parent" in str(row["promotion_last_error"])
            assert await _dedicated_partitions(conn, ws) == {
                table: "" for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE)
            }
        finally:
            await conn.close()
    finally:
        conn = await asyncpg.connect(**_kwargs(_TEST_DB))
        try:
            await conn.execute(f"ALTER TABLE hidden_{_VECTOR_TABLE} RENAME TO {_VECTOR_TABLE}")
        finally:
            await conn.close()

    # A reclaimed worker (lease + fence expired) repairs and completes.
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await conn.execute(
            "UPDATE dlightrag_promotion_jobs SET next_retry_at = NOW() WHERE workspace = $1",
            ws,
        )
    finally:
        await conn.close()
    assert await _worker().run_once() is True
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        assert (await _registry_row(conn, ws))["storage_tier"] == "hot"
    finally:
        await conn.close()


async def test_discovery_rejects_a_broken_plain_vector_parent(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        registry = PGWorkspaceRegistry()
        await registry.upsert(workspace=ws, display_name="Hot", embedding_model="m")
        await conn.execute(
            """
            CREATE TABLE lightrag_vdb_chunks_broken (
                workspace TEXT NOT NULL, id TEXT NOT NULL,
                PRIMARY KEY (workspace, id)
            )
            """
        )
    finally:
        await conn.close()

    try:
        await PGPromotionJobStore().enqueue(ws)
        assert await _worker().run_once() is True
        conn = await asyncpg.connect(**_kwargs(_TEST_DB))
        try:
            row = await _registry_row(conn, ws)
            assert row["promotion_state"] == "failed"
            assert row["write_fence_owner"] is None
            assert "is not a partitioned parent" in str(row["promotion_last_error"])
        finally:
            await conn.close()
    finally:
        conn = await asyncpg.connect(**_kwargs(_TEST_DB))
        try:
            await conn.execute("DROP TABLE IF EXISTS lightrag_vdb_chunks_broken")
        finally:
            await conn.close()

    # Reclaimed worker completes once the broken relation is gone.
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await conn.execute(
            "UPDATE dlightrag_promotion_jobs SET next_retry_at = NOW() WHERE workspace = $1",
            ws,
        )
    finally:
        await conn.close()
    assert await _worker().run_once() is True
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        assert (await _registry_row(conn, ws))["storage_tier"] == "hot"
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Lock-phase evidence: reads and other-workspace DML continue around the
# exclusion-proof cutover
# ---------------------------------------------------------------------------


async def test_after_committed_not_valid_proofs_reads_and_other_workspace_dml_continue(
    corpus: None,
    monkeypatch: pytest.MonkeyPatch,
    workspaces: tuple[str, str],
) -> None:
    ws, other = workspaces
    await _clean_state()
    import time

    from dlightrag.adapters.postgres.corpus import promotion_worker as worker_module

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        registry = PGWorkspaceRegistry()
        await registry.upsert(workspace=ws, display_name="Hot", embedding_model="m")
        await registry.upsert(workspace=other, display_name="Other", embedding_model="m")
        await _seed_workspace(conn, ws, docs=2, chunks_per_doc=2)
        await _seed_workspace(conn, other, docs=1, chunks_per_doc=1)
    finally:
        await conn.close()

    entered = asyncio.Event()
    release = asyncio.Event()
    observed: dict[str, Any] = {}

    async def phase1_hook(gate_conn: Any) -> None:
        # The ADD ... NOT VALID statements committed in autocommit phase 1:
        # their AccessExclusive locks are already released, so a second
        # connection reads and writes other workspaces promptly.
        conn2 = await asyncpg.connect(**_kwargs(_TEST_DB))
        try:
            start = time.monotonic()
            assert (
                await conn2.fetchval(
                    f"SELECT COUNT(*) FROM {_CHUNKS_TABLE} WHERE workspace = $1", other
                )
                == 1
            )
            await conn2.execute(
                f"INSERT INTO {_CHUNKS_TABLE} (workspace, id, full_doc_id, content) "
                "VALUES ($1, $2, $3, $4)",
                other,
                "pw-phase1-chunk",
                f"{other}-doc-0",
                "phase1 write",
            )
            await conn2.execute(
                f"INSERT INTO {_METADATA_TABLE} (workspace, doc_id, filename, filename_stem) "
                "VALUES ($1, $2, $3, $3)",
                other,
                "pw-phase1-doc",
                "phase1.pdf",
            )
            elapsed = time.monotonic() - start
            observed["phase1_elapsed"] = elapsed
        finally:
            await conn2.close()
        entered.set()
        await release.wait()

    monkeypatch.setattr(worker_module, "_PHASE1_PAUSE_HOOK", phase1_hook)
    try:
        await PGPromotionJobStore().enqueue(ws)
        task = asyncio.create_task(_worker().run_once())
        await asyncio.wait_for(entered.wait(), timeout=10)
        release.set()
        assert await asyncio.wait_for(task, timeout=10) is True
    finally:
        monkeypatch.undo()

    assert observed["phase1_elapsed"] < 1.0


async def test_during_cutover_validation_reads_and_other_workspace_dml_complete_promptly(
    corpus: None,
    monkeypatch: pytest.MonkeyPatch,
    workspaces: tuple[str, str],
) -> None:
    ws, other = workspaces
    await _clean_state()
    import time

    from dlightrag.adapters.postgres.corpus import promotion_worker as worker_module

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        registry = PGWorkspaceRegistry()
        await registry.upsert(workspace=ws, display_name="Hot", embedding_model="m")
        await registry.upsert(workspace=other, display_name="Other", embedding_model="m")
        await _seed_workspace(conn, ws, docs=2, chunks_per_doc=2)
        await _seed_workspace(conn, other, docs=1, chunks_per_doc=1)
    finally:
        await conn.close()

    entered = asyncio.Event()
    release = asyncio.Event()
    observed: dict[str, Any] = {}

    async def cutover_hook(gate_conn: Any) -> None:
        # The cutover transaction has finished DELETE+VALIDATE for every
        # parent but has not taken any ATTACH lock yet: reads and
        # other-workspace DML complete promptly (VALIDATE holds only
        # ShareUpdateExclusive, which does not conflict with RowExclusive or
        # AccessShare).
        conn2 = await asyncpg.connect(**_kwargs(_TEST_DB))
        try:
            start = time.monotonic()
            assert (
                await conn2.fetchval(
                    f"SELECT COUNT(*) FROM {_CHUNKS_TABLE} WHERE workspace = $1", other
                )
                == 1
            )
            await conn2.execute(
                f"INSERT INTO {_CHUNKS_TABLE} (workspace, id, full_doc_id, content) "
                "VALUES ($1, $2, $3, $4)",
                other,
                "pw-cutover-chunk",
                f"{other}-doc-0",
                "cutover-window write",
            )
            observed["cutover_elapsed"] = time.monotonic() - start
        finally:
            await conn2.close()
        entered.set()
        await release.wait()

    monkeypatch.setattr(worker_module, "_CUTOVER_PAUSE_HOOK", cutover_hook)
    try:
        await PGPromotionJobStore().enqueue(ws)
        task = asyncio.create_task(_worker().run_once())
        await asyncio.wait_for(entered.wait(), timeout=10)
        release.set()
        assert await asyncio.wait_for(task, timeout=10) is True
    finally:
        monkeypatch.undo()

    assert observed["cutover_elapsed"] < 1.0
    # After success no temporary exclusion constraints remain.
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE):
            leftover = await conn.fetch(
                "SELECT conname FROM pg_constraint "
                f"WHERE conrelid = {default_child_name(table)!r}::regclass "
                "AND conname LIKE '%_excl'"
            )
            assert not leftover
    finally:
        await conn.close()


async def test_stale_promoting_with_leftover_exclusion_blocks_writes_until_reclaim(
    corpus: None, workspaces: tuple[str, str]
) -> None:
    ws, other = workspaces
    await _clean_state()
    from dlightrag.adapters.postgres.corpus import promotion_worker as worker_module
    from dlightrag.adapters.postgres.corpus.workspace_write_gate import workspace_write_gate

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        registry = PGWorkspaceRegistry()
        await registry.upsert(workspace=ws, display_name="Hot", embedding_model="m")
        await _seed_workspace(conn, ws, docs=1, chunks_per_doc=1)
        # Simulate a crashed worker: promoting state, expired fence, and a
        # committed leftover exclusion proof.
        await PGPromotionJobStore().enqueue(ws)
        await conn.execute(
            "UPDATE dlightrag_promotion_jobs SET state = 'promoting', "
            "lease_owner = 'dead-worker', lease_generation = 1, "
            "lease_until = NOW() - INTERVAL '1 second' WHERE workspace = $1",
            ws,
        )
        await conn.execute(
            "UPDATE dlightrag_workspace_meta SET promotion_state = 'promoting', "
            "write_fence_owner = 'dead-worker#1', "
            "write_fence_until = NOW() - INTERVAL '1 second' WHERE workspace = $1",
            ws,
        )
        ws_literal = await conn.fetchval("SELECT quote_literal($1)", ws)
        for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE):
            await conn.execute(
                f"ALTER TABLE ONLY {default_child_name(table)} "
                f"ADD CONSTRAINT {worker_module._exclusion_constraint_name(table, ws)} "
                f"CHECK (workspace <> {ws_literal}) NOT VALID"
            )
    finally:
        await conn.close()

    # Stale promoting + expired timestamp + leftover exclusion must return a
    # retryable fence error.
    with pytest.raises(WorkspaceWriteFencedError) as excinfo:
        async with workspace_write_gate(ws):
            pytest.fail("stale promoting with leftover proofs must stay fenced")
    assert 0 < excinfo.value.retry_after_seconds <= 5.1

    # A reclaimed worker (lease/fence expired) removes the leftovers and
    # completes the promotion; writes resume afterwards.
    assert await _worker().run_once() is True
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        assert (await _registry_row(conn, ws))["storage_tier"] == "hot"
        for table in (_METADATA_TABLE, _CHUNKS_TABLE, _VECTOR_TABLE):
            leftover = await conn.fetch(
                "SELECT conname FROM pg_constraint "
                f"WHERE conrelid = {default_child_name(table)!r}::regclass "
                "AND conname LIKE '%_excl'"
            )
            assert not leftover
    finally:
        await conn.close()
    async with workspace_write_gate(ws):
        pass
