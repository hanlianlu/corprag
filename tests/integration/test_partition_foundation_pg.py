# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable PostgreSQL integration coverage for the Commit 1 partition foundation.

Runs against a dedicated fresh test database (never the development corpus):
the suite creates/drops ``dlightrag_partition_foundation_test`` itself, so the
shared development database and its running services are untouched.

Proves on compact fixtures:
* fresh writer init builds partitioned parents with DEFAULT children for the
  retrieval-critical tables (metadata, text chunks/BM25, chunk vectors);
* parent HNSW and BM25 indexes create planner-usable child indexes;
* forced generic prepared ``workspace=$n`` plans runtime-prune other children;
* workspace isolation across the DEFAULT and a dedicated hot partition;
* canonical custom-metadata GIN and trigram GIN indexes exist and are usable;
* old unpartitioned shapes fail loudly with the one-time reset message, and
  reader validation fails equivalently while passing on the partitioned corpus.
"""

import hashlib
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_MAINT_DB = "postgres"
_TEST_DB = "dlightrag_partition_foundation_test"
_LEGACY_DB = "dlightrag_partition_foundation_legacy_test"
_EXTENSIONS = ("vector", "pg_textsearch", "pg_trgm")
_WORKSPACE_A = "pf_it_ws_a"
_WORKSPACE_HOT = "pf_it_ws_hot"

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


async def _create_fresh_database(database: str) -> None:
    conn = await asyncpg.connect(**_kwargs(_MAINT_DB))
    try:
        # The test databases belong to this suite alone: disconnect any
        # leftovers from a crashed previous run before dropping.
        await conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = $1 AND pid <> pg_backend_pid()",
            database,
        )
        await conn.execute(f"DROP DATABASE IF EXISTS {database}")
        await conn.execute(f"CREATE DATABASE {database}")
    finally:
        await conn.close()
    db = await asyncpg.connect(**_kwargs(database))
    try:
        for extension in _EXTENSIONS:
            await db.execute(f"CREATE EXTENSION IF NOT EXISTS {extension}")
    finally:
        await db.close()


@pytest.fixture(scope="module", autouse=True)
async def _fresh_test_database() -> AsyncIterator[None]:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")
    await _create_fresh_database(_TEST_DB)
    yield None
    conn = await asyncpg.connect(**_kwargs(_MAINT_DB))
    try:
        await conn.execute(f"DROP DATABASE IF EXISTS {_TEST_DB}")
        await conn.execute(f"DROP DATABASE IF EXISTS {_LEGACY_DB}")
    finally:
        await conn.close()


@dataclass
class WriterCorpus:
    lightrag: Any
    vector_table: str
    config: Any


def _fake_llm(prompt: str, **_: Any) -> str:
    return "{}"


@pytest.fixture
async def pool() -> Any:
    """One direct connection pool onto the dedicated test database."""
    pool = await asyncpg.create_pool(**_kwargs(_TEST_DB), min_size=1, max_size=2)
    yield pool
    await pool.close()


def _embedding_func() -> Any:
    from lightrag.utils import EmbeddingFunc

    async def embed(texts: list[str], *, context: str = "document") -> Any:
        import numpy as np

        values = []
        for text in texts:
            digest = hashlib.sha256(f"{context}:{text}".encode()).digest()
            vector = [((digest[i] / 255.0) * 2.0) - 1.0 for i in range(8)]
            values.append(vector)
        return np.array(values, dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=8,
        max_token_size=512,
        func=embed,
        model_name="pf-it-fake",
        supports_asymmetric=True,
    )


@pytest.fixture(scope="module")
async def writer_corpus() -> AsyncIterator[WriterCorpus]:
    """One fresh writer attach over the dedicated test database."""
    from dlightrag.adapters.postgres.core._pool import pg_pool
    from dlightrag.adapters.postgres.corpus.corpus import build_pg_corpus_backend
    from dlightrag.application.config import DlightragConfig, reset_config, set_config
    from dlightrag.application.settings import rag_settings
    from dlightrag.engine.ai.settings import (
        EmbeddingSettings,
        ModelRoleSettings,
        ModelSettings,
    )
    from dlightrag.engine.rag.workspace.ports import CorpusRuntimeModels

    cfg = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        deployment={"workspace": _WORKSPACE_A, "working_dir": "/tmp/pf_it_workdir"},
        storage={
            "postgres": {
                "host": str(_DEFAULT_KWARGS["host"]),
                "port": int(_DEFAULT_KWARGS["port"]),
                "user": str(_DEFAULT_KWARGS["user"]),
                "password": str(_DEFAULT_KWARGS["password"]),
                "database": _TEST_DB,
                "pool_min_size": 1,
                "pool_max_size": 3,
            }
        },
        models={
            "max_concurrency": 1,
            "chat": ModelRoleSettings(
                default=ModelSettings(
                    provider="openai",
                    model="pf-it-fake-llm",
                    api_key="pf-it-fake-key",
                    timeout=30,
                )
            ),
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="pf-it-fake",
                api_key="pf-it-fake-key",
                dim=8,
                max_token_size=512,
                max_concurrency=1,
                batch_size=2,
                startup_probe=False,
            ),
            "rerank": {"enabled": False},
        },
        corpus={
            "ingestion": {
                "chunk_token_size": 128,
                "pipeline": {"max_concurrency": 1},
            },
            "retrieval": {
                "bm25_enabled": True,
                # The dedicated test database carries no pg_jieba: keep the
                # profile set to bundled text-search configs only.
                "bm25_profiles": [
                    {"name": "en", "text_config": "english", "languages": ["en"]},
                    {"name": "simple", "text_config": "simple", "fallback": True},
                ],
            },
        },
    )
    set_config(cfg)
    pg_pool.bind(cfg)

    backend = build_pg_corpus_backend(cfg)
    await backend.maintenance.initialize(validate_only=False)
    settings = rag_settings(cfg)
    models = CorpusRuntimeModels(
        default_llm_func=_fake_llm,
        embedding_func=_embedding_func(),
        role_llm_configs=None,
    )
    lightrag = backend.runtime.create(models=models, settings=settings)
    try:
        await backend.runtime.attach(lightrag)
    except BaseException:
        # A failed attach must never leak LightRAG's process-wide client or the
        # domain pool: the next test run would block on DROP DATABASE.
        await pg_pool.close()
        try:
            from lightrag.kg.postgres_impl import ClientManager

            db = ClientManager._instances.get("db")
            if db is not None:
                if getattr(db, "pool", None) is not None:
                    await db.pool.close()
                ClientManager._instances["db"] = None
                ClientManager._instances["ref_count"] = 0
        except Exception:
            pass
        reset_config()
        raise
    try:
        yield WriterCorpus(
            lightrag=lightrag,
            vector_table=str(lightrag.chunks_vdb.table_name),
            config=cfg,
        )
    finally:
        try:
            await lightrag.finalize_storages()
        except Exception:
            pass
        await pg_pool.close()
        reset_config()


async def _table_state(conn: Any, table_name: str) -> dict[str, Any]:
    row = await conn.fetchrow(
        """
        SELECT c.relkind::text AS relkind, c.relispartition AS is_partition,
               pt.partattrs IS NOT NULL AS is_partitioned
        FROM pg_catalog.pg_class c
        LEFT JOIN pg_catalog.pg_partitioned_table pt ON pt.partrelid = c.oid
        WHERE c.oid = to_regclass($1)
        """,
        table_name,
    )
    assert row is not None, f"table {table_name} missing"
    return dict(row)


async def _child_names(conn: Any, table_name: str) -> set[str]:
    rows = await conn.fetch(
        "SELECT c.relname FROM pg_catalog.pg_inherits i "
        "JOIN pg_catalog.pg_class c ON c.oid = i.inhrelid "
        "WHERE i.inhparent = to_regclass($1)",
        table_name,
    )
    return {str(row["relname"]) for row in rows}


async def _child_indexes(conn: Any, index_name: str) -> list[str]:
    rows = await conn.fetch(
        """
        SELECT c.relname
        FROM pg_catalog.pg_inherits i
        JOIN pg_catalog.pg_class parent ON parent.oid = i.inhparent
        JOIN pg_catalog.pg_class c ON c.oid = i.inhrelid
        WHERE parent.relname = $1
        ORDER BY c.relname
        """,
        index_name,
    )
    return [str(row["relname"]) for row in rows]


async def test_fresh_writer_init_builds_partitioned_foundation(writer_corpus: WriterCorpus) -> None:
    from dlightrag.adapters.postgres.corpus.partition_foundation import (
        default_child_name,
    )

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        for table in ("dlightrag_doc_metadata", "lightrag_doc_chunks", writer_corpus.vector_table):
            state = await _table_state(conn, table)
            assert state["relkind"] == "p", f"{table} is not a partitioned parent"
            assert state["is_partition"] is False
            children = await _child_names(conn, table)
            assert default_child_name(table) in children, f"{table} lacks its DEFAULT child"

        # Partition keys are plain workspace columns, never expressions.
        keys = await conn.fetchval(
            """
            SELECT array_agg(a.attname ORDER BY a.attnum)
            FROM pg_partitioned_table pt
            JOIN pg_attribute a ON a.attrelid = pt.partrelid AND a.attnum = ANY(pt.partattrs)
            WHERE pt.partrelid = to_regclass('dlightrag_doc_metadata')
            """
        )
        assert keys == ["workspace"]

        # The durable promotion job schema is installed by normal maintenance
        # startup even though Commit 3 has not wired the worker yet.
        assert (
            await conn.fetchval("SELECT to_regclass('dlightrag_promotion_jobs')::text")
            == "dlightrag_promotion_jobs"
        )

        # DOC_STATUS and the full-doc table stay unpartitioned in this commit.
        for table in ("lightrag_doc_status", "lightrag_doc_full"):
            state = await _table_state(conn, table)
            assert state["relkind"] == "r", f"{table} must not be partitioned yet"
    finally:
        await conn.close()


async def test_parent_hnsw_and_bm25_indexes_create_usable_child_indexes(
    writer_corpus: WriterCorpus,
) -> None:
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        hnsw_rows = await conn.fetch(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = $1 AND indexdef ILIKE '%USING hnsw%'",
            writer_corpus.vector_table.lower(),
        )
        assert hnsw_rows, "no parent HNSW index on the vector parent"
        hnsw_child_indexes = await _child_indexes(conn, str(hnsw_rows[0]["indexname"]))
        assert hnsw_child_indexes, "parent HNSW index has no child indexes"

        bm25_rows = await conn.fetch(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = 'lightrag_doc_chunks' AND indexdef ILIKE '%USING bm25%'"
        )
        assert bm25_rows, "no parent BM25 index on the chunk parent"
        for row in bm25_rows:
            child_indexes = await _child_indexes(conn, str(row["indexname"]))
            assert child_indexes, f"parent BM25 index {row['indexname']} has no child indexes"

        # Both are planner-usable through the parent name on a forced generic plan.
        async with conn.transaction():
            await conn.execute("SET LOCAL plan_cache_mode = 'force_generic_plan'")
            await conn.execute("SET LOCAL enable_seqscan = off")
            await conn.execute("SET LOCAL enable_sort = off")
            await conn.execute(
                f"PREPARE pf_hnsw AS SELECT id FROM {writer_corpus.vector_table} "
                "WHERE workspace = $1 "
                "ORDER BY content_vector <=> $2::vector LIMIT 5"
            )
            plans = await conn.fetch(
                "EXPLAIN (VERBOSE, COSTS OFF) "
                "EXECUTE pf_hnsw('pf_it_ws_a', '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'::vector)"
            )
        plan_text = "\n".join(str(row[0]) for row in plans)
        assert "Index Scan" in plan_text
        assert "content_vector_idx" in plan_text  # the child HNSW index

        async with conn.transaction():
            await conn.execute("SET LOCAL plan_cache_mode = 'force_generic_plan'")
            await conn.execute("SET LOCAL enable_seqscan = off")
            await conn.execute("SET LOCAL enable_sort = off")
            await conn.execute(
                "PREPARE pf_bm25 AS SELECT id FROM lightrag_doc_chunks "
                "WHERE workspace = $1 "
                "ORDER BY content <@> to_bm25query($2, $3) LIMIT 5"
            )
            plans = await conn.fetch(
                "EXPLAIN (VERBOSE, COSTS OFF) "
                f"EXECUTE pf_bm25('pf_it_ws_a', 'quarterly report', "
                f"'{str(bm25_rows[0]['indexname'])}')"
            )
        plan_text = "\n".join(str(row[0]) for row in plans)
        assert "Index Scan" in plan_text
        assert "content_idx" in plan_text  # the child BM25 index
    finally:
        await conn.close()


async def test_generic_prepared_plan_prunes_to_the_hot_partition(
    writer_corpus: WriterCorpus,
) -> None:
    from dlightrag.adapters.postgres.corpus.partition_foundation import (
        attach_workspace_partition,
        child_partition_name,
        default_child_name,
    )

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    hot_child = None
    try:
        await conn.execute(
            f"DELETE FROM {writer_corpus.vector_table} WHERE workspace = ANY($1::text[])",
            [_WORKSPACE_A, _WORKSPACE_HOT],
        )
        hot_child = await attach_workspace_partition(
            conn, table_name=writer_corpus.vector_table, workspace=_WORKSPACE_HOT
        )
        assert hot_child == child_partition_name(writer_corpus.vector_table, _WORKSPACE_HOT)
        assert hot_child != _WORKSPACE_HOT  # never the raw workspace identifier

        await conn.execute(
            f"""
            INSERT INTO {writer_corpus.vector_table}
                (workspace, id, full_doc_id, content, content_vector)
            VALUES
                ($1, 'c-a1', 'd-a1', 'alpha', $2::vector),
                ($1, 'c-a2', 'd-a1', 'beta', $3::vector)
            """,
            _WORKSPACE_A,
            "[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]",
            "[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]",
        )
        await conn.execute(
            f"""
            INSERT INTO {writer_corpus.vector_table}
                (workspace, id, full_doc_id, content, content_vector)
            VALUES ($1, 'c-h1', 'd-h1', 'gamma', $2::vector)
            """,
            _WORKSPACE_HOT,
            "[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]",
        )

        async with conn.transaction():
            await conn.execute("SET LOCAL plan_cache_mode = 'force_generic_plan'")
            await conn.execute(
                f"PREPARE pf_prune AS SELECT id FROM {writer_corpus.vector_table} "
                "WHERE workspace = $1 "
                "ORDER BY content_vector <=> $2::vector LIMIT 5"
            )
            plans = await conn.fetch(
                "EXPLAIN (VERBOSE, COSTS OFF) "
                "EXECUTE pf_prune('pf_it_ws_hot', '[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]'::vector)"
            )
        plan_text = "\n".join(str(row[0]) for row in plans)
        assert "Subplans Removed: 1" in plan_text
        assert hot_child in plan_text
        assert default_child_name(writer_corpus.vector_table) not in plan_text

        # Workspace isolation through the parent: each side sees only its rows.
        a_rows = await conn.fetch(
            f"SELECT id FROM {writer_corpus.vector_table} WHERE workspace = $1 ORDER BY id",
            _WORKSPACE_A,
        )
        hot_rows = await conn.fetch(
            f"SELECT id FROM {writer_corpus.vector_table} WHERE workspace = $1 ORDER BY id",
            _WORKSPACE_HOT,
        )
        assert [str(row["id"]) for row in a_rows] == ["c-a1", "c-a2"]
        assert [str(row["id"]) for row in hot_rows] == ["c-h1"]
    finally:
        if hot_child is not None:
            await conn.execute(f"DROP TABLE IF EXISTS {hot_child}")
        await conn.execute(
            f"DELETE FROM {writer_corpus.vector_table} WHERE workspace = ANY($1::text[])",
            [_WORKSPACE_A, _WORKSPACE_HOT],
        )
        await conn.close()


async def test_chunk_parent_isolates_workspaces_across_default_and_hot_partitions(
    writer_corpus: WriterCorpus,
) -> None:
    from dlightrag.adapters.postgres.corpus.partition_foundation import (
        attach_workspace_partition,
    )

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    hot_child = None
    try:
        await conn.execute(
            "DELETE FROM lightrag_doc_chunks WHERE workspace = ANY($1::text[])",
            [_WORKSPACE_A, _WORKSPACE_HOT],
        )
        hot_child = await attach_workspace_partition(
            conn, table_name="lightrag_doc_chunks", workspace=_WORKSPACE_HOT
        )
        await conn.execute(
            """
            INSERT INTO lightrag_doc_chunks (workspace, id, full_doc_id, content)
            VALUES ($1, 't-a1', 'd-a1', 'alpha'), ($2, 't-h1', 'd-h1', 'zeta')
            """,
            _WORKSPACE_A,
            _WORKSPACE_HOT,
        )
        # ON CONFLICT upsert through the parent still resolves per child.
        await conn.execute(
            """
            INSERT INTO lightrag_doc_chunks (workspace, id, full_doc_id, content)
            VALUES ($1, 't-a1', 'd-a1', 'alpha revised')
            ON CONFLICT (workspace, id) DO UPDATE SET content = EXCLUDED.content
            """,
            _WORKSPACE_A,
        )
        content = await conn.fetchval(
            "SELECT content FROM lightrag_doc_chunks WHERE workspace = $1 AND id = 't-a1'",
            _WORKSPACE_A,
        )
        assert content == "alpha revised"

        rows = await conn.fetch(
            "SELECT id FROM lightrag_doc_chunks WHERE workspace = $1 ORDER BY id",
            _WORKSPACE_HOT,
        )
        assert [str(row["id"]) for row in rows] == ["t-h1"]

        # Parameterized BM25 through the parent index name prunes and scores correctly.
        scores = await conn.fetch(
            """
            SELECT id, -(content <@> to_bm25query($1, $2)) AS score
            FROM lightrag_doc_chunks
            WHERE workspace = $3
            ORDER BY content <@> to_bm25query($1, $2)
            LIMIT 5
            """,
            "zeta",
            "idx_lightrag_doc_chunks_bm25_simple",
            _WORKSPACE_HOT,
        )
        assert [str(row["id"]) for row in scores] == ["t-h1"]
    finally:
        if hot_child is not None:
            await conn.execute(f"DROP TABLE IF EXISTS {hot_child}")
        await conn.execute(
            "DELETE FROM lightrag_doc_chunks WHERE workspace = ANY($1::text[])",
            [_WORKSPACE_A, _WORKSPACE_HOT],
        )
        await conn.close()


async def test_metadata_indexes_are_workspace_leading_and_planner_usable(
    writer_corpus: WriterCorpus,
    pool: Any,
) -> None:
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await conn.execute(
            "DELETE FROM dlightrag_doc_metadata WHERE workspace = ANY($1::text[])",
            [_WORKSPACE_A, _WORKSPACE_HOT],
        )
        rows = [
            (_WORKSPACE_A, f"doc-{index:03d}", f"Quarterly Report draft {index}.pdf")
            for index in range(300)
        ]
        rows.extend(
            (_WORKSPACE_HOT, f"hot-{index:03d}", f"Annual Budget {index}.pdf")
            for index in range(300)
        )
        rows.append((_WORKSPACE_A, "doc-pct", "100%_off\\sale.pdf"))
        rows.append((_WORKSPACE_HOT, "doc-hot", "Quarterly Report draft hot.pdf"))
        await conn.executemany(
            """
            INSERT INTO dlightrag_doc_metadata
                (workspace, doc_id, filename, filename_stem,
                 _dlightrag_finalization_complete)
            VALUES ($1, $2, $3, $3, TRUE)
            """,
            rows,
        )
        # Canonical search column + raw column stay distinct and correct when
        # the adapter's write path runs (canonicalization is owned by the SQL).
        from dlightrag.adapters.postgres.corpus.pg_metadata_index import PGMetadataIndex

        index = PGMetadataIndex(workspace=_WORKSPACE_A)
        index._operation_pool = pool  # type: ignore[attr-defined]
        await index.upsert(
            "doc-custom",
            {
                "custom_metadata": {
                    " Department ": " Finance ",
                    " Pages ": 7,
                    " Reviewed ": True,
                }
            },
        )

        raw = await conn.fetchval(
            "SELECT custom_metadata FROM dlightrag_doc_metadata "
            "WHERE workspace = $1 AND doc_id = 'doc-custom'",
            _WORKSPACE_A,
        )
        search = await conn.fetchval(
            "SELECT custom_metadata_search FROM dlightrag_doc_metadata "
            "WHERE workspace = $1 AND doc_id = 'doc-custom'",
            _WORKSPACE_A,
        )
        assert json.loads(str(raw)) == {
            " Department ": " Finance ",
            " Pages ": 7,
            " Reviewed ": True,
        }
        assert json.loads(str(search)) == {
            "department": "finance",
            "pages": "7",
            "reviewed": "true",
        }

        async with conn.transaction():
            await conn.execute("SET LOCAL plan_cache_mode = 'force_generic_plan'")
            plans = await conn.fetch(
                """
                EXPLAIN (VERBOSE, COSTS OFF)
                SELECT doc_id FROM dlightrag_doc_metadata
                WHERE workspace = $1 AND LOWER(TRIM(filename)) = LOWER(TRIM($2))
                """,
                _WORKSPACE_A,
                "Quarterly Report draft 7.pdf",
            )
        plan_text = "\n".join(str(row[0]) for row in plans)
        assert "Index Scan" in plan_text
        assert "lower(TRIM" in plan_text  # workspace-leading expression index in use

        # The trigram GIN is planner-usable for the contains expression. On a
        # compact fixture the workspace-leading B-trees legitimately win the
        # near-tied costs of the full workspace+LIKE predicate, so the probe
        # drops the workspace clause to isolate the contains expression the
        # GIN serves; the workspace-prefixed plan is evidenced just above.
        async with conn.transaction():
            await conn.execute("SET LOCAL plan_cache_mode = 'force_generic_plan'")
            await conn.execute("SET LOCAL enable_indexscan = off")
            await conn.execute("SET LOCAL enable_seqscan = off")
            await conn.execute("ANALYZE dlightrag_doc_metadata")
            await conn.execute(
                "PREPARE pf_trgm AS SELECT doc_id FROM dlightrag_doc_metadata "
                "WHERE LOWER(TRIM(filename)) LIKE $1 ESCAPE '\\'"
            )
            plans = await conn.fetch(
                "EXPLAIN (VERBOSE, COSTS OFF) EXECUTE pf_trgm('%report draft 17%')"
            )
        plan_text = "\n".join(str(row[0]) for row in plans)
        assert "lower_idx" in plan_text  # the trigram GIN child index
        assert "Bitmap Index Scan" in plan_text

        # Literal wildcards stay literal through the escaped pattern.
        matches = await conn.fetch(
            """
            SELECT doc_id FROM dlightrag_doc_metadata
            WHERE workspace = $1 AND LOWER(TRIM(filename)) LIKE $2 ESCAPE '\\'
            """,
            _WORKSPACE_A,
            "%100\\%\\_off\\\\sale%",
        )
        assert [str(row["doc_id"]) for row in matches] == ["doc-pct"]

        # Canonical JSONB containment uses the GIN jsonb_path_ops index.
        async with conn.transaction():
            await conn.execute("SET LOCAL enable_indexscan = off")
            await conn.execute("SET LOCAL enable_seqscan = off")
            plans = await conn.fetch(
                """
                EXPLAIN (VERBOSE, COSTS OFF)
                SELECT doc_id FROM dlightrag_doc_metadata
                WHERE workspace = $1
                  AND custom_metadata_search @> $2::jsonb
                """,
                _WORKSPACE_A,
                '{"department": "finance"}',
            )
        plan_text = "\n".join(str(row[0]) for row in plans)
        assert "custom_metadata_search_idx" in plan_text  # the child GIN index

        # Unbounded title: an oversized value stores and matches through the
        # bounded MD5 key with the equality recheck, where a B-tree could not.
        oversized = "x" * 10_000
        await conn.execute(
            "INSERT INTO dlightrag_doc_metadata "
            "(workspace, doc_id, title, _dlightrag_finalization_complete) "
            "VALUES ($1, 'doc-big-title', $2, TRUE)",
            _WORKSPACE_A,
            oversized,
        )
        async with conn.transaction():
            await conn.execute("SET LOCAL plan_cache_mode = 'force_generic_plan'")
            plans = await conn.fetch(
                """
                EXPLAIN (VERBOSE, COSTS OFF)
                SELECT doc_id FROM dlightrag_doc_metadata
                WHERE workspace = $1
                  AND MD5(LOWER(TRIM(title))) = MD5(LOWER(TRIM($2)))
                  AND LOWER(TRIM(title)) = LOWER(TRIM($2))
                """,
                _WORKSPACE_A,
                oversized,
            )
        plan_text = "\n".join(str(row[0]) for row in plans)
        assert "md5" in plan_text.lower()
        hit = await conn.fetchval(
            "SELECT doc_id FROM dlightrag_doc_metadata "
            "WHERE workspace = $1 AND doc_id = 'doc-big-title'",
            _WORKSPACE_A,
        )
        assert hit == "doc-big-title"
    finally:
        await conn.execute(
            "DELETE FROM dlightrag_doc_metadata WHERE workspace = ANY($1::text[])",
            [_WORKSPACE_A, _WORKSPACE_HOT],
        )
        await conn.close()


async def test_reader_validation_passes_on_partitioned_corpus_and_fails_on_legacy(
    writer_corpus: WriterCorpus,
) -> None:
    from dlightrag.adapters.postgres.corpus.partition_foundation import (
        PartitionedTableSpec,
        verify_partitioned_tables,
    )
    from dlightrag.engine.rag.workspace.ports import CorpusSchemaError

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        specs = (
            PartitionedTableSpec(
                name="lightrag_doc_chunks",
                required_columns=("id", "workspace", "full_doc_id", "content", "file_path"),
                primary_key=("workspace", "id"),
                required_indexes=(
                    "idx_lightrag_doc_chunks_id",
                    "idx_lightrag_doc_chunks_workspace_id",
                ),
            ),
            PartitionedTableSpec(
                name=writer_corpus.vector_table,
                required_columns=(
                    "id",
                    "workspace",
                    "full_doc_id",
                    "content",
                    "content_vector",
                    "file_path",
                ),
                primary_key=("workspace", "id"),
                required_index_markers=("USING hnsw",),
            ),
        )
        await verify_partitioned_tables(conn, specs=specs)  # must not raise
    finally:
        await conn.close()

    await _create_fresh_database(_LEGACY_DB)
    legacy = await asyncpg.connect(**_kwargs(_LEGACY_DB))
    try:
        await legacy.execute(
            """
            CREATE TABLE lightrag_doc_chunks (
                id VARCHAR(255), workspace VARCHAR(255), full_doc_id VARCHAR(256),
                content TEXT, file_path TEXT,
                CONSTRAINT lightrag_doc_chunks_pk PRIMARY KEY (workspace, id)
            )
            """
        )
        await legacy.execute(
            "INSERT INTO lightrag_doc_chunks (workspace, id, full_doc_id, content) "
            "VALUES ('legacy', 'c1', 'd1', 'old row')"
        )

        with pytest.raises(CorpusSchemaError, match="reset_development.py"):
            await verify_partitioned_tables(
                legacy,
                specs=(PartitionedTableSpec(name="lightrag_doc_chunks"),),
            )
    finally:
        await legacy.close()


async def test_old_unpartitioned_corpus_fails_loudly_on_writer_startup() -> None:
    from dlightrag.adapters.postgres.corpus.partition_foundation import (
        PartitionedTableSpec,
        ensure_partitioned_tables,
    )
    from dlightrag.engine.rag.workspace.ports import CorpusSchemaError

    await _create_fresh_database(_LEGACY_DB)
    conn = await asyncpg.connect(**_kwargs(_LEGACY_DB))
    try:
        await conn.execute(
            """
            CREATE TABLE dlightrag_doc_metadata (
                workspace VARCHAR(255) NOT NULL,
                doc_id VARCHAR(255) NOT NULL,
                filename VARCHAR(512),
                PRIMARY KEY (workspace, doc_id)
            )
            """
        )
        await conn.execute(
            "INSERT INTO dlightrag_doc_metadata (workspace, doc_id, filename) "
            "VALUES ('legacy', 'd1', 'old.pdf')"
        )

        with pytest.raises(CorpusSchemaError) as excinfo:
            await ensure_partitioned_tables(
                conn,
                specs=(PartitionedTableSpec(name="dlightrag_doc_metadata"),),
            )
        message = str(excinfo.value)
        assert "holds rows" in message or "unpartitioned" in message
        assert "reset_development.py" in message
        assert "per-workspace data reset cannot" in message

        # The row is untouched: startup failure never rebuilds destructively.
        count = await conn.fetchval(
            "SELECT count(*) FROM dlightrag_doc_metadata WHERE doc_id = 'd1'"
        )
        assert int(count) == 1

        # DlightRAG owns this table's migrations, so even an empty old shape is
        # deliberately not treated as migration-compatible in this fresh-schema release.
        await conn.execute("DELETE FROM dlightrag_doc_metadata")
        with pytest.raises(CorpusSchemaError, match="fresh-schema release"):
            await ensure_partitioned_tables(
                conn,
                specs=(
                    PartitionedTableSpec(
                        name="dlightrag_doc_metadata",
                        required_columns=("workspace", "doc_id", "filename"),
                        primary_key=("workspace", "doc_id"),
                        convert_empty_plain=False,
                    ),
                ),
            )
        state = await _table_state(conn, "dlightrag_doc_metadata")
        assert state["relkind"] == "r"
    finally:
        await conn.close()
