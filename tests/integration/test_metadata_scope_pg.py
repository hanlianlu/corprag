# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable PostgreSQL integration coverage for the metadata runtime cutover.

Runs against a dedicated fresh test database (never the development corpus):
the suite creates/drops ``dlightrag_metadata_scope_test`` itself, so the
shared development database and its running services are untouched.

Proves on compact fixtures:
* the bounded scope preflight stops at ``threshold + 1`` and never exact-counts
  a whole match set (0, at/below-threshold, and sentinel cases), including the
  exact-then-contains filename widening;
* forced generic plans reach the HNSW and BM25 indexes through the database-side
  metadata semi-join with no corpus-scale sequential scan on indexed paths;
* the exact and HNSW vector legs and the BM25 leg fill ``top_k`` when matches
  are reachable within the existing search budgets;
* the one-query graph chunk guard returns positional rows with ``None`` for
  out-of-scope ids;
* the canonical custom JSONB containment predicate rides the GIN index.
"""

import asyncio
import datetime
import hashlib
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope

pytestmark = [
    pytest.mark.integration,
    # One shared event loop for the module: the writer attach creates asyncpg
    # pools whose connections stay bound to the loop they were born on.
    pytest.mark.asyncio(loop_scope="module"),
]

_MAINT_DB = "postgres"
_TEST_DB = "dlightrag_metadata_scope_test"
_EXTENSIONS = ("vector", "pg_textsearch", "pg_trgm")
_WORKSPACE = "ms_it_ws"
_THRESHOLD = 2

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
    finally:
        await conn.close()


def _fake_llm(prompt: str, **_: Any) -> str:
    return "{}"


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
        model_name="ms-it-fake",
        supports_asymmetric=True,
    )


@dataclass
class WriterCorpus:
    lightrag: Any
    stores: Any
    vector_table: str
    settings: Any


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
        deployment={"workspace": _WORKSPACE, "working_dir": "/tmp/ms_it_workdir"},
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
                    model="ms-it-fake-llm",
                    api_key="ms-it-fake-key",
                    timeout=30,
                )
            ),
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="ms-it-fake",
                api_key="ms-it-fake-key",
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
        stores = await backend.runtime.attach(lightrag)
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
            stores=stores,
            vector_table=str(lightrag.chunks_vdb.table_name),
            settings=settings,
        )
    finally:
        try:
            await lightrag.finalize_storages()
        except Exception:
            pass
        await pg_pool.close()
        reset_config()


async def test_finalization_marker_round_trips_and_partial_updates_preserve_true(
    writer_corpus: WriterCorpus,
) -> None:
    from dlightrag.adapters.postgres.corpus.pg_metadata_index import PGMetadataIndex

    index = PGMetadataIndex(workspace="ms_finalization_marker")
    await index.clear()

    await index.upsert(
        "doc-marker",
        {
            "filename": "marker.pdf",
            "_dlightrag_finalization_complete": False,
        },
    )
    first = await index.get("doc-marker")
    assert first is not None
    assert first["_dlightrag_finalization_complete"] is False

    await index.upsert(
        "doc-marker",
        {"_dlightrag_finalization_complete": True},
    )
    complete = await index.get("doc-marker")
    assert complete is not None
    assert complete["_dlightrag_finalization_complete"] is True

    await index.upsert("doc-marker", {"title": "partial update"})
    preserved = await index.get("doc-marker")
    assert preserved is not None
    assert preserved["title"] == "partial update"
    assert preserved["_dlightrag_finalization_complete"] is True

    # Reproduce an already-partitioned pre-marker schema. Foundation validation
    # must allow the append-only migration to add the column before the final
    # schema verifier requires it.
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await conn.execute(
            "DELETE FROM dlightrag_schema_migrations "
            "WHERE scope = 'doc_metadata' AND version IN "
            "('column_finalization_complete', 'product_document_visibility')"
        )
        await conn.execute(
            "ALTER TABLE dlightrag_doc_metadata DROP COLUMN _dlightrag_finalization_complete"
        )
    finally:
        await conn.close()
    await index.initialize()
    migrated = await index.get("doc-marker")
    assert migrated is not None
    assert migrated["_dlightrag_finalization_complete"] is False


async def test_field_schema_stats_follow_writes_deletes_clear_and_workspace_union(
    writer_corpus: WriterCorpus,
) -> None:
    from dlightrag.adapters.postgres.corpus import pg_metadata_index

    first_workspace = "ms_schema_a"
    second_workspace = "ms_schema_b"
    first = pg_metadata_index.PGMetadataIndex(workspace=first_workspace)
    second = pg_metadata_index.PGMetadataIndex(workspace=second_workspace)
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await first.clear()
        await second.clear()
        await first.upsert(
            "doc-a",
            {
                "filename": "report.pdf",
                "filename_stem": "report",
                "file_extension": "pdf",
                "custom_metadata": {"department": "finance"},
                "_dlightrag_finalization_complete": True,
            },
        )
        await first.upsert(
            "doc-b",
            {
                "title": "Quarterly report",
                "author": "Ada",
                "custom_metadata": {"department": "finance", "team": "core"},
                "_dlightrag_finalization_complete": True,
            },
        )
        await second.upsert(
            "doc-c",
            {
                "creation_date": datetime.datetime(2026, 1, 2),
                "custom_metadata": {"jurisdiction": "eu"},
                "_dlightrag_finalization_complete": True,
            },
        )

        counts = {
            str(row["field_id"]): int(row["document_count"])
            for row in await conn.fetch(
                "SELECT field_id, document_count "
                "FROM dlightrag_metadata_field_stats WHERE workspace = $1",
                first_workspace,
            )
        }
        assert counts == {
            "author": 1,
            "department": 2,
            "file_extension": 1,
            "filename": 1,
            "filename_stem": 1,
            "team": 1,
            "title": 1,
        }
        assert await first.get_field_schema(workspaces=(first_workspace, second_workspace)) == {
            "filters": [
                "filename",
                "file_extension",
                "title",
                "author",
                "creation_date_from",
                "creation_date_to",
                "custom",
            ],
            "custom_keys": ["department", "jurisdiction", "team"],
        }
        async with conn.transaction():
            await conn.execute("SET LOCAL enable_seqscan = off")
            plan = "\n".join(
                str(row["QUERY PLAN"])
                for row in await conn.fetch(
                    "EXPLAIN (COSTS OFF) " + pg_metadata_index._FIELD_SCHEMA,
                    [first_workspace, second_workspace],
                )
            )
        assert "dlightrag_metadata_field_stats_pkey" in plan
        assert "dlightrag_doc_metadata" not in plan

        await first.delete("doc-a")
        assert await first.get_field_schema() == {
            "filters": ["title", "author", "custom"],
            "custom_keys": ["department", "team"],
        }
        assert await first.merge_custom_metadata("doc-b", {"custom_metadata": {"region": "north"}})
        assert await first.get_field_schema() == {
            "filters": ["title", "author", "custom"],
            "custom_keys": ["department", "region", "team"],
        }
        await conn.execute(
            "UPDATE dlightrag_doc_metadata SET title = NULL "
            "WHERE workspace = $1 AND doc_id = 'doc-b'",
            first_workspace,
        )
        assert await first.get_field_schema() == {
            "filters": ["author", "custom"],
            "custom_keys": ["department", "region", "team"],
        }

        await first.clear()
        assert await first.get_field_schema() == {"filters": [], "custom_keys": []}
        assert not await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM dlightrag_metadata_field_stats WHERE workspace = $1)",
            first_workspace,
        )

        await asyncio.gather(
            *(
                first.upsert(
                    f"concurrent-{index}",
                    {
                        "custom_metadata": {"shared": index},
                        "_dlightrag_finalization_complete": True,
                    },
                )
                for index in range(12)
            )
        )
        assert (
            await conn.fetchval(
                "SELECT document_count FROM dlightrag_metadata_field_stats "
                "WHERE workspace = $1 AND field_id = 'shared'",
                first_workspace,
            )
            == 12
        )
        await asyncio.gather(*(first.delete(f"concurrent-{index}") for index in range(12)))
        assert not await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM dlightrag_metadata_field_stats WHERE workspace = $1)",
            first_workspace,
        )

        many_keys = {f"key_{index:03d}": index for index in range(130)}
        await second.clear()
        await second.upsert(
            "doc-many",
            {
                "custom_metadata": many_keys,
                "_dlightrag_finalization_complete": True,
            },
        )
        bounded = await second.get_field_schema()
        assert bounded["filters"] == ["custom"]
        assert bounded["custom_keys"] == list(many_keys)[:128]
    finally:
        await first.clear()
        await second.clear()
        await conn.close()


def _scope(*, candidate_count: int, candidate_count_exact: bool = True) -> MetadataScope:
    return MetadataScope(
        filters=MetadataFilter(filename="report.pdf"),
        filename_mode="exact",
        doc_exists=True,
        candidate_count=candidate_count,
        candidate_count_exact=candidate_count_exact,
    )


@pytest.fixture
async def seeded(writer_corpus: WriterCorpus) -> AsyncIterator[None]:
    """Compact corpus: 1 in-scope doc with 5 chunks, 1 out-of-scope doc with 2."""
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await conn.execute("DELETE FROM dlightrag_doc_metadata WHERE workspace = $1", _WORKSPACE)
        await conn.execute("DELETE FROM lightrag_doc_chunks WHERE workspace = $1", _WORKSPACE)
        await conn.execute("DELETE FROM lightrag_doc_status WHERE workspace = $1", _WORKSPACE)
        await conn.execute(
            f"DELETE FROM {writer_corpus.vector_table} WHERE workspace = $1", _WORKSPACE
        )
        await conn.execute(
            """
            INSERT INTO dlightrag_doc_metadata
                (workspace, doc_id, filename, filename_stem, custom_metadata,
                 _dlightrag_finalization_complete)
            VALUES
                ($1, 'doc-in', 'report.pdf', 'report', '{"Team": " Core "}', TRUE),
                ($1, 'doc-out', 'other.pdf', 'other', '{"Team": "Other"}', TRUE)
            """,
            _WORKSPACE,
        )
        await conn.execute(
            """
            UPDATE dlightrag_doc_metadata
            SET custom_metadata_search = dlightrag_canonical_custom_metadata(custom_metadata)
            WHERE workspace = $1
            """,
            _WORKSPACE,
        )
        chunks = [(_WORKSPACE, f"c-in-{i}", "doc-in", f"alpha beta gamma {i}") for i in range(5)]
        chunks.extend(
            [
                (_WORKSPACE, "c-out-0", "doc-out", "zeta eta theta"),
                (_WORKSPACE, "c-out-1", "doc-out", "zeta eta iota"),
            ]
        )
        await conn.executemany(
            """
            INSERT INTO lightrag_doc_chunks
                (workspace, id, full_doc_id, content, dlightrag_bm25_language)
            VALUES ($1, $2, $3, $4, 'en')
            """,
            chunks,
        )
        await conn.executemany(
            f"""
            INSERT INTO {writer_corpus.vector_table}
                (workspace, id, full_doc_id, content, content_vector)
            VALUES ($1, $2, $3, $4, $5::vector)
            """,
            [
                (
                    _WORKSPACE,
                    chunk_id,
                    doc_id,
                    content,
                    f"[{i * 0.01:.2f},{0.1},{0.2},{0.3},{0.4},{0.5},{0.6},{0.7}]",
                )
                for i, (_, chunk_id, doc_id, content) in enumerate(chunks)
            ],
        )
        yield None
    finally:
        await conn.execute("DELETE FROM dlightrag_doc_metadata WHERE workspace = $1", _WORKSPACE)
        await conn.execute("DELETE FROM lightrag_doc_chunks WHERE workspace = $1", _WORKSPACE)
        await conn.execute("DELETE FROM lightrag_doc_status WHERE workspace = $1", _WORKSPACE)
        await conn.execute(
            f"DELETE FROM {writer_corpus.vector_table} WHERE workspace = $1", _WORKSPACE
        )
        await conn.close()


async def _plan_text(
    conn: Any,
    sql: str,
    params: list[Any],
    *,
    generic: bool = False,
    no_seqscan: bool = False,
    no_sort: bool = False,
    no_indexscan: bool = False,
) -> str:
    async with conn.transaction():
        if generic:
            await conn.execute("SET LOCAL plan_cache_mode = 'force_generic_plan'")
        if no_seqscan:
            await conn.execute("SET LOCAL enable_seqscan = off")
        if no_sort:
            await conn.execute("SET LOCAL enable_sort = off")
        if no_indexscan:
            await conn.execute("SET LOCAL enable_indexscan = off")
        name = "ms_plan"
        await conn.execute(f"PREPARE {name} AS {sql}")
        try:
            rows = await conn.fetch(
                f"EXPLAIN (VERBOSE, COSTS OFF) EXECUTE {name}({', '.join(_sql_literals(params))})"
            )
        finally:
            await conn.execute(f"DEALLOCATE {name}")
    return "\n".join(str(row[0]) for row in rows)


def _sql_literals(params: list[Any]) -> list[str]:
    literals = []
    for param in params:
        if isinstance(param, str):
            literals.append("'" + param.replace("'", "''") + "'")
        elif isinstance(param, float):
            literals.append(str(param))
        elif isinstance(param, int):
            literals.append(str(param))
        else:
            raise TypeError(f"unsupported plan literal {param!r}")
    return literals


# ---------------------------------------------------------------------------
# Bounded metadata subset
# ---------------------------------------------------------------------------


async def test_metadata_subset_applies_publication_and_scope_to_only_caller_ids(
    seeded: None,
) -> None:
    from dlightrag.adapters.postgres.corpus.pg_metadata_index import PGMetadataIndex

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await conn.execute(
            "INSERT INTO dlightrag_doc_metadata "
            "(workspace, doc_id, filename, filename_stem, _dlightrag_finalization_complete) "
            "VALUES ($1, 'doc-hidden-report', 'report.pdf', 'report', FALSE)",
            _WORKSPACE,
        )
    finally:
        await conn.close()

    metadata = PGMetadataIndex(workspace=_WORKSPACE)
    assert await metadata.visible_subset(
        ["doc-out", "doc-hidden-report", "doc-in", "doc-missing", "doc-in"],
        scope=_scope(candidate_count=5),
    ) == frozenset({"doc-in"})


# ---------------------------------------------------------------------------
# Bounded scope preflight
# ---------------------------------------------------------------------------


async def test_preflight_is_bounded_and_never_exact_counts_the_whole_match(
    writer_corpus: WriterCorpus,
    seeded: None,
) -> None:
    from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore

    chunks = PGCorpusChunkStore(writer_corpus.lightrag, exact_threshold=_THRESHOLD)

    # 5 chunks match, above threshold: the probe stops at threshold + 1 and
    # reports a non-exact lower bound instead of the exact total 5.
    scope = await chunks.resolve_scope(MetadataFilter(filename="report.pdf"))
    assert scope.doc_exists is True
    assert scope.candidate_count == _THRESHOLD + 1
    assert scope.candidate_count_exact is False
    assert scope.render_candidate_count() == f"{_THRESHOLD + 1}+"

    # A filter with no match reports an inactive scope.
    missing = await chunks.resolve_scope(MetadataFilter(filename="nope.pdf"))
    assert bool(missing) is False
    assert missing.candidate_count == 0
    assert missing.candidate_count_exact is True

    # A matching document with zero chunks is still an active scope.
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await conn.execute(
            "INSERT INTO dlightrag_doc_metadata "
            "(workspace, doc_id, filename, filename_stem, _dlightrag_finalization_complete) "
            "VALUES ($1, 'doc-empty', 'empty.pdf', 'empty', TRUE)",
            _WORKSPACE,
        )
        await conn.execute(
            """
            UPDATE dlightrag_doc_metadata
            SET custom_metadata_search = dlightrag_canonical_custom_metadata(custom_metadata)
            WHERE workspace = $1 AND doc_id = 'doc-empty'
            """,
            _WORKSPACE,
        )
        empty = await chunks.resolve_scope(MetadataFilter(filename="empty.pdf"))
        assert bool(empty) is True
        assert empty.candidate_count == 0
        assert empty.candidate_count_exact is True
    finally:
        await conn.execute(
            "DELETE FROM dlightrag_doc_metadata WHERE workspace = $1 AND doc_id = 'doc-empty'",
            _WORKSPACE,
        )
        await conn.close()


async def test_preflight_widens_exact_to_contains_only_on_exact_miss(
    writer_corpus: WriterCorpus,
    seeded: None,
) -> None:
    from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore

    chunks = PGCorpusChunkStore(writer_corpus.lightrag, exact_threshold=_THRESHOLD)

    # "report" is not the verbatim name; the widened literal-substring clause
    # still matches report.pdf.
    scope = await chunks.resolve_scope(MetadataFilter(filename="repo"))
    assert scope.doc_exists is True
    assert scope.filename_mode == "contains"
    assert scope.candidate_count == _THRESHOLD + 1

    # An exact hit never widens.
    exact = await chunks.resolve_scope(MetadataFilter(filename="report.pdf"))
    assert exact.filename_mode == "exact"


async def test_probe_statement_uses_indexed_metadata_paths(
    writer_corpus: WriterCorpus,
    seeded: None,
) -> None:
    from dlightrag.adapters.postgres.corpus.pg_metadata_scope import build_bounded_scope_probe

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        sql, params = build_bounded_scope_probe(
            _WORKSPACE,
            MetadataFilter(custom={"Team": "core"}),
            filename_mode="exact",
            threshold=_THRESHOLD,
        )
        plan = await _plan_text(conn, sql, params, generic=True, no_seqscan=True)
        # No corpus-scale sequential scan: the EXISTS probe and the capped
        # chunk probe ride workspace-leading indexes.
        assert "Seq Scan" not in plan
        assert "Limit" in plan  # the capped chunk probe
        assert "Index Only Scan" in plan or "Index Scan" in plan

        # The canonical containment predicate is planner-usable through the
        # GIN jsonb_path_ops index. On a compact fixture the workspace-leading
        # B-trees legitimately win the near-tied costs, so the probe drops the
        # workspace clause to isolate the containment expression the GIN
        # serves; the workspace-prefixed indexed plan is evidenced just above.
        gin_plan = await _plan_text(
            conn,
            "SELECT doc_id FROM dlightrag_doc_metadata "
            "WHERE custom_metadata_search @> dlightrag_canonical_custom_metadata($1::jsonb)",
            [json.dumps({"team": "core"})],
            generic=True,
            no_seqscan=True,
            no_indexscan=True,
        )
        assert "Bitmap Index Scan" in gin_plan
        assert "custom_metadata_search_idx" in gin_plan
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Exact/HNSW vector legs
# ---------------------------------------------------------------------------


async def test_exact_vector_leg_fills_top_k_through_the_metadata_semi_join(
    writer_corpus: WriterCorpus,
    seeded: None,
) -> None:
    from dlightrag.adapters.postgres.corpus.corpus_vectors import PGFilteredVectorSearch

    search = PGFilteredVectorSearch(writer_corpus.lightrag.chunks_vdb, exact_threshold=_THRESHOLD)
    rows = await search.search(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        scope=_scope(candidate_count=1),
        top_k=3,
    )

    # Only the in-scope document's chunks are reachable, in distance order.
    assert len(rows) == 3
    assert all(str(row["full_doc_id"]) == "doc-in" for row in rows)


async def test_hnsw_leg_fills_top_k_and_keeps_the_limit_outside_the_filter(
    writer_corpus: WriterCorpus,
    seeded: None,
) -> None:
    from dlightrag.adapters.postgres.corpus.corpus_vectors import PGFilteredVectorSearch

    search = PGFilteredVectorSearch(writer_corpus.lightrag.chunks_vdb, exact_threshold=_THRESHOLD)
    rows = await search.search(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        scope=_scope(candidate_count=_THRESHOLD + 1, candidate_count_exact=False),
        top_k=2,
    )

    assert len(rows) == 2
    assert all(str(row["full_doc_id"]) == "doc-in" for row in rows)


async def test_forced_generic_hnsw_plan_rides_the_index_through_the_semi_join(
    writer_corpus: WriterCorpus,
    seeded: None,
) -> None:
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        sql = (
            f"SELECT id FROM {writer_corpus.vector_table} "
            "WHERE workspace = $2 "
            "AND full_doc_id IN ("
            "SELECT doc_id FROM dlightrag_doc_metadata "
            "WHERE workspace = $3 "
            "AND (LOWER(TRIM(filename)) = LOWER(TRIM($4)) "
            "OR LOWER(TRIM(filename_stem)) = LOWER(TRIM($4)))) "
            "ORDER BY content_vector <=> $1::vector LIMIT $5"
        )
        plan = await _plan_text(
            conn,
            sql,
            ["[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]", _WORKSPACE, _WORKSPACE, "report.pdf", 3],
            generic=True,
            no_seqscan=True,
            no_sort=True,
        )
        assert "Seq Scan" not in plan
        assert "Index Scan" in plan
        assert "content_vector_idx" in plan  # the child HNSW index
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# BM25 leg
# ---------------------------------------------------------------------------


async def test_bm25_leg_fills_top_k_through_the_metadata_semi_join(
    writer_corpus: WriterCorpus,
    seeded: None,
) -> None:
    scope = _scope(candidate_count=3)
    rows = await writer_corpus.stores.bm25.search("alpha beta", scope=scope, top_k=3)

    assert len(rows) == 3
    assert all(str(row["full_doc_id"]) == "doc-in" for row in rows)


async def test_forced_generic_bm25_plan_rides_the_index_through_the_semi_join(
    writer_corpus: WriterCorpus,
    seeded: None,
) -> None:
    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        index_rows = await conn.fetch(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = 'lightrag_doc_chunks' AND indexdef ILIKE '%USING bm25%' "
            "AND indexdef ILIKE '%text_config=simple%'"
        )
        assert index_rows
        index_name = str(index_rows[0]["indexname"])
        sql = (
            "SELECT id FROM lightrag_doc_chunks "
            f"WHERE workspace = $2 AND full_doc_id IN ("
            "SELECT doc_id FROM dlightrag_doc_metadata "
            "WHERE workspace = $3 "
            "AND (LOWER(TRIM(filename)) = LOWER(TRIM($4)) "
            "OR LOWER(TRIM(filename_stem)) = LOWER(TRIM($4)))) "
            f"ORDER BY content <@> to_bm25query($1, '{index_name}') LIMIT $5"
        )
        plan = await _plan_text(
            conn,
            sql,
            ["alpha beta", _WORKSPACE, _WORKSPACE, "report.pdf", 3],
            generic=True,
            no_seqscan=True,
            no_sort=True,
        )
        assert "Seq Scan" not in plan
        assert "Index Scan" in plan
        assert "content_idx" in plan  # the child BM25 index
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# One-query graph chunk guard
# ---------------------------------------------------------------------------


async def test_graph_guard_reads_scoped_chunks_in_one_query(
    writer_corpus: WriterCorpus,
    seeded: None,
) -> None:
    from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore

    chunks = PGCorpusChunkStore(writer_corpus.lightrag, exact_threshold=_THRESHOLD)
    scope = _scope(candidate_count=5)

    rows = await chunks.read_scoped(scope, ["c-out-0", "c-in-1", "c-in-1", "missing"])

    # Positional order with duplicates and None for out-of-scope/missing ids.
    assert rows[0] is None  # doc-out is out of scope
    assert rows[1] is not None and str(rows[1]["id"]) == "c-in-1"
    assert rows[2] is not None and str(rows[2]["id"]) == "c-in-1"
    assert rows[3] is None
    # LightRAG text-chunk JSON decoding matches the storage contract.
    assert isinstance(rows[1]["llm_cache_list"], list)
    assert isinstance(rows[1]["heading"], dict)
    assert isinstance(rows[1]["sidecar"], dict)


async def test_unscoped_product_reads_hide_incomplete_and_metadata_less_rows(
    writer_corpus: WriterCorpus,
    seeded: None,
) -> None:
    """Every direct PostgreSQL read surface applies the same visibility rule."""
    from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore
    from dlightrag.adapters.postgres.corpus.corpus_vectors import PGFilteredVectorSearch
    from dlightrag.adapters.postgres.corpus.file_panel import PGFilePanelStore
    from dlightrag.adapters.postgres.corpus.pg_metadata_index import PGMetadataIndex
    from dlightrag.adapters.postgres.corpus.pg_metadata_search import PGMetadataSearchStore
    from dlightrag.application.corpus_admin import FilePanelPageRequest, MetadataSearchPageRequest
    from dlightrag.engine.rag.corpus.downloads import (
        RedirectDownloadTarget,
        SourceDownloadNotFoundError,
        SourceDownloadService,
    )
    from dlightrag.engine.rag.corpus.visual_assets import VisualAssetResolver

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        # Simulate an interrupted commit and a low-level bypass whose corpus
        # artifacts exist without any metadata row at all.
        await conn.execute(
            "UPDATE dlightrag_doc_metadata "
            "SET _dlightrag_finalization_complete = FALSE, "
            "download_locator = 'https://example.com/hidden.pdf' "
            "WHERE workspace = $1 AND doc_id = 'doc-in'",
            _WORKSPACE,
        )
        await conn.execute(
            "UPDATE dlightrag_doc_metadata "
            "SET download_locator = 'https://example.com/ready.pdf' "
            "WHERE workspace = $1 AND doc_id = 'doc-out'",
            _WORKSPACE,
        )
        await conn.execute(
            "INSERT INTO lightrag_doc_chunks "
            "(workspace, id, full_doc_id, content, dlightrag_bm25_language) "
            "VALUES ($1, 'c-bypass', 'doc-bypass', 'alpha beta bypass', 'en')",
            _WORKSPACE,
        )
        await conn.execute(
            f"INSERT INTO {writer_corpus.vector_table} "
            "(workspace, id, full_doc_id, content, content_vector) "
            "VALUES ($1, 'c-bypass', 'doc-bypass', 'alpha beta bypass', "
            "'[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]'::vector)",
            _WORKSPACE,
        )
        await conn.executemany(
            "INSERT INTO lightrag_doc_status "
            "(workspace, id, status, file_path) VALUES ($1, $2, 'processed', $3)",
            [
                (_WORKSPACE, "doc-in", "/tmp/hidden.pdf"),
                (_WORKSPACE, "doc-out", "/tmp/ready.pdf"),
                (_WORKSPACE, "doc-bypass", "/tmp/bypass.pdf"),
            ],
        )
    finally:
        await conn.close()

    metadata = PGMetadataIndex(workspace=_WORKSPACE)
    assert await metadata.visible_subset(["doc-in", "doc-out", "doc-bypass"]) == {"doc-out"}
    assert await metadata.query(MetadataFilter()) == ["doc-out"]

    metadata_page = await PGMetadataSearchStore().search_metadata_page(
        _WORKSPACE,
        MetadataFilter(),
        page=MetadataSearchPageRequest(limit=10),
    )
    assert metadata_page.document_ids == ("doc-out",)

    vectors = await PGFilteredVectorSearch(writer_corpus.lightrag.chunks_vdb).search(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        scope=None,
        top_k=20,
    )
    assert vectors
    assert {str(row["full_doc_id"]) for row in vectors} == {"doc-out"}

    bm25 = await writer_corpus.stores.bm25.search("alpha beta", scope=None, top_k=20)
    assert {str(row["full_doc_id"]) for row in bm25} <= {"doc-out"}

    chunks = PGCorpusChunkStore(writer_corpus.lightrag)
    graph_rows = await chunks.read_scoped(
        None,
        ["c-in-0", "c-out-0", "c-bypass"],
    )
    assert graph_rows[0] is None
    assert graph_rows[1] is not None and graph_rows[1]["full_doc_id"] == "doc-out"
    assert graph_rows[2] is None

    files = await PGFilePanelStore().list_processed_files(
        _WORKSPACE,
        page=FilePanelPageRequest(limit=10),
    )
    assert tuple(item.doc_id for item in files.items) == ("doc-out",)

    # Stats/schema are trigger-maintained from visible rows only. The hidden
    # transition must decrement counts, and the metadata-less bypass adds none.
    stat_conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        filename_count = await stat_conn.fetchval(
            "SELECT document_count FROM dlightrag_metadata_field_stats "
            "WHERE workspace = $1 AND field_id = 'filename'",
            _WORKSPACE,
        )
    finally:
        await stat_conn.close()
    assert filename_count == 1
    assert "filename" in (await metadata.get_field_schema())["filters"]

    downloads = SourceDownloadService(
        settings=writer_corpus.settings,
        metadata_index=metadata,
        workspace_id=_WORKSPACE,
    )
    with pytest.raises(SourceDownloadNotFoundError):
        await downloads.prepare("doc-in")
    with pytest.raises(SourceDownloadNotFoundError):
        await downloads.prepare("doc-bypass")
    ready = await downloads.prepare("doc-out")
    assert isinstance(ready, RedirectDownloadTarget)
    assert ready.url == "https://example.com/ready.pdf"

    class VisualStores:
        async def context_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
            doc_ids = {
                "image-hidden": "doc-in",
                "image-ready": "doc-out",
                "image-bypass": "doc-bypass",
            }
            return [
                {
                    "chunk_id": chunk_id,
                    "full_doc_id": doc_ids[chunk_id],
                    "image_data": "cmVhZHk=",
                    "image_mime_type": "image/png",
                }
                for chunk_id in chunk_ids
            ]

        async def get_text_chunks(self, chunk_ids: list[str]) -> list[None]:
            return [None for _ in chunk_ids]

        async def get_full_docs(self, doc_ids: list[str]) -> list[None]:
            return [None for _ in doc_ids]

    visuals = VisualAssetResolver(stores=VisualStores(), visibility_lookup=metadata)
    assert await visuals.resolve("image-hidden") is None
    assert await visuals.resolve("image-bypass") is None
    ready_image = await visuals.resolve("image-ready")
    assert ready_image is not None and ready_image.data == b"ready"


# ---------------------------------------------------------------------------
# Canonical custom containment
# ---------------------------------------------------------------------------


async def test_custom_containment_matches_numbers_bools_and_nulls_like_storage(
    writer_corpus: WriterCorpus,
    seeded: None,
) -> None:
    from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore

    conn = await asyncpg.connect(**_kwargs(_TEST_DB))
    try:
        await conn.execute(
            """
            INSERT INTO dlightrag_doc_metadata
                (workspace, doc_id, filename, filename_stem, custom_metadata,
                 _dlightrag_finalization_complete)
            VALUES ($1, 'doc-shapes', 'shapes.pdf', 'shapes', $2::jsonb, TRUE)
            """,
            _WORKSPACE,
            json.dumps(
                {
                    "pages": 7,
                    "reviewed": True,
                    "note": None,
                    "nested": {"leaf": "Text"},
                    "team": " Core ",
                }
            ),
        )
        await conn.execute(
            """
            UPDATE dlightrag_doc_metadata
            SET custom_metadata_search = dlightrag_canonical_custom_metadata(custom_metadata)
            WHERE workspace = $1 AND doc_id = 'doc-shapes'
            """,
            _WORKSPACE,
        )
        chunks = PGCorpusChunkStore(writer_corpus.lightrag, exact_threshold=_THRESHOLD)

        scope = await chunks.resolve_scope(
            MetadataFilter(
                filename="shapes.pdf",
                custom={"pages": 7, "reviewed": True, "note": None, "nested": {"leaf": "Text"}},
            )
        )
        assert scope.doc_exists is True

        mismatch = await chunks.resolve_scope(
            MetadataFilter(filename="shapes.pdf", custom={"pages": "8"})
        )
        assert bool(mismatch) is False
    finally:
        await conn.execute(
            "DELETE FROM dlightrag_doc_metadata WHERE workspace = $1 AND doc_id = 'doc-shapes'",
            _WORKSPACE,
        )
        await conn.close()
