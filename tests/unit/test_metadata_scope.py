# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit contracts for the shared metadata predicate and the bounded scope probe.

Every retrieval leg renders its metadata predicates through
``metadata_match_conditions``, so these tests pin placeholder numbering,
canonicalization, the capped preflight statement, and the guarantee that no
document-id array crosses into Python.
"""

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from dlightrag.adapters.postgres.corpus.corpus_bm25 import build_bm25_sql
from dlightrag.adapters.postgres.corpus.corpus_vectors import PGFilteredVectorSearch
from dlightrag.adapters.postgres.corpus.pg_metadata_index import (
    like_contains_pattern,
    metadata_match_conditions,
)
from dlightrag.adapters.postgres.corpus.pg_metadata_scope import build_bounded_scope_probe
from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope


def _full_filter() -> MetadataFilter:
    return MetadataFilter(
        file_extension="PDF",
        author="Ada",
        title="Quarterly Report",
        filename="Quarterly Report",
        creation_date_from=datetime(2024, 1, 1),
        creation_date_to=datetime(2024, 12, 31, tzinfo=UTC),
        custom={"Department": " Finance ", "pages": 7, "reviewed": True, "note": None},
    )


def _scope(*, candidate_count: int, candidate_count_exact: bool = True) -> MetadataScope:
    return MetadataScope(
        filters=MetadataFilter(filename="x.pdf"),
        filename_mode="exact",
        doc_exists=True,
        candidate_count=candidate_count,
        candidate_count_exact=candidate_count_exact,
    )


# ---------------------------------------------------------------------------
# Shared predicate builder
# ---------------------------------------------------------------------------


def test_conditions_start_at_the_requested_placeholder_index() -> None:
    conditions, params = metadata_match_conditions(
        "finance",
        MetadataFilter(author="Ada"),
        filename_mode="exact",
        start_index=5,
        alias="m",
    )

    assert conditions == [
        "m.workspace = $5",
        "m._dlightrag_finalization_complete IS TRUE",
        "LOWER(TRIM(m.author)) = LOWER(TRIM($6))",
    ]
    assert params == ["finance", "Ada"]


def test_conditions_are_unaliased_by_default_for_direct_table_queries() -> None:
    conditions, params = metadata_match_conditions(
        "finance",
        MetadataFilter(file_extension=".pdf"),
        filename_mode="exact",
    )

    assert conditions == [
        "workspace = $1",
        "_dlightrag_finalization_complete IS TRUE",
        "LOWER(TRIM(file_extension)) = LOWER(TRIM($2))",
    ]
    assert params == ["finance", "pdf"]


def test_all_filter_fields_render_with_consecutive_placeholders() -> None:
    conditions, params = metadata_match_conditions(
        "ws",
        _full_filter(),
        filename_mode="exact",
        start_index=3,
        alias="m",
    )

    assert conditions == [
        "m.workspace = $3",
        "m._dlightrag_finalization_complete IS TRUE",
        "LOWER(TRIM(m.file_extension)) = LOWER(TRIM($4))",
        "LOWER(TRIM(m.author)) = LOWER(TRIM($5))",
        "MD5(LOWER(TRIM(m.title))) = MD5(LOWER(TRIM($6))) "
        "AND LOWER(TRIM(m.title)) = LOWER(TRIM($6))",
        "(LOWER(TRIM(m.filename)) = LOWER(TRIM($7)) "
        "OR LOWER(TRIM(m.filename_stem)) = LOWER(TRIM($7))) ",
        "m.creation_date >= $8",
        "m.creation_date <= $9",
        "m.custom_metadata_search @> dlightrag_canonical_custom_metadata($10::jsonb)",
    ]
    assert params[0] == "ws"
    assert params[4] == "Quarterly Report"
    assert params[5] == datetime(2024, 1, 1)
    assert params[6] == datetime(2024, 12, 31)
    # Custom key/value equalities collapse into one canonical containment
    # object whose keys fold like the write path folds them.
    assert json.loads(params[7]) == {
        "department": " Finance ",
        "pages": 7,
        "reviewed": True,
        "note": None,
    }


def test_custom_containment_uses_the_shared_canonical_sql_function() -> None:
    conditions, params = metadata_match_conditions(
        "ws",
        MetadataFilter(custom={"A": "b"}),
        filename_mode="exact",
    )

    assert conditions[2] == (
        "custom_metadata_search @> dlightrag_canonical_custom_metadata($2::jsonb)"
    )
    assert json.loads(params[1]) == {"a": "b"}
    # No raw per-key custom scan predicate remains on this path.
    assert "->>" not in conditions[2]


def test_empty_custom_dict_emits_no_predicate() -> None:
    conditions, params = metadata_match_conditions(
        "ws",
        MetadataFilter(custom={}),
        filename_mode="exact",
    )

    assert conditions == [
        "workspace = $1",
        "_dlightrag_finalization_complete IS TRUE",
    ]
    assert params == ["ws"]


def test_filename_modes_render_exact_then_contains_clauses() -> None:
    filters = MetadataFilter(filename="100%_off\\sale.pdf")

    exact_conditions, exact_params = metadata_match_conditions("ws", filters, filename_mode="exact")
    contains_conditions, contains_params = metadata_match_conditions(
        "ws", filters, filename_mode="contains"
    )

    assert exact_conditions[2] == (
        "(LOWER(TRIM(filename)) = LOWER(TRIM($2)) OR LOWER(TRIM(filename_stem)) = LOWER(TRIM($2))) "
    )
    assert exact_params[1] == "100%_off\\sale.pdf"
    assert contains_conditions[2] == "LOWER(TRIM(filename)) LIKE LOWER($2) ESCAPE '\\'"
    # Caller wildcards stay literal characters in the widened clause; the
    # database folds the bound pattern under the same collation as the column.
    assert contains_params[1] == "%100\\%\\_off\\\\sale.pdf%"
    assert like_contains_pattern("Report.pdf") == "%Report.pdf%"


def test_invalid_filename_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="filename mode"):
        metadata_match_conditions(
            "ws",
            MetadataFilter(filename="x"),
            filename_mode="regex",
        )


def test_custom_key_collisions_resolve_deterministically_in_caller_order() -> None:
    filters = MetadataFilter(custom={"A": 1, "a": 2})

    conditions, params = metadata_match_conditions("ws", filters, filename_mode="exact")

    # The filter model folds keys once, mirroring the ingest normalization:
    # the later key wins, and the bound containment object carries it.
    assert json.loads(params[1]) == {"a": 2}


# ---------------------------------------------------------------------------
# Bounded scope preflight
# ---------------------------------------------------------------------------


def test_probe_returns_exists_and_capped_chunk_count_in_one_statement() -> None:
    sql, params = build_bounded_scope_probe(
        "ws",
        MetadataFilter(filename="report.pdf"),
        filename_mode="exact",
        threshold=8192,
    )

    assert "EXISTS (SELECT 1 FROM dlightrag_doc_metadata" in sql
    assert "AS doc_exists" in sql
    assert "AS chunk_count" in sql
    assert "count(*)" in sql
    assert "LIMIT $6" in sql
    # The cap binds threshold + 1 last, never an exact corpus-scale COUNT target.
    assert params[5] == 8193
    # The workspace is authenticated on the EXISTS probe, the chunk source,
    # and the inner metadata subquery.
    assert sql.count("workspace = $") == 3
    assert "ANY(" not in sql


def test_probe_placeholder_layout_is_parameter_safe() -> None:
    filters = _full_filter()
    sql, params = build_bounded_scope_probe(
        "ws",
        filters,
        filename_mode="exact",
        threshold=4,
    )

    # EXISTS conditions: $1..$8; chunk workspace $9; inner conditions
    # $10..$17; probe cap $18.
    assert "workspace = $9" in sql
    assert "m.workspace = $10" in sql
    assert "LIMIT $18" in sql
    assert len(params) == 18
    assert params[8] == "ws"  # chunk-side workspace
    assert params[9] == "ws"  # inner metadata workspace
    assert params[17] == 5  # threshold + 1 binds last


def test_probe_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        build_bounded_scope_probe(
            "ws",
            MetadataFilter(filename="x"),
            filename_mode="exact",
            threshold=-1,
        )


async def test_scope_resolver_reports_empty_scope_when_no_document_matches() -> None:
    from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore

    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.fetches: list[tuple[Any, ...]] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def fetchrow(self, *args):  # noqa: ANN002, ANN202
            self.fetches.append(args)
            return {"doc_exists": False, "chunk_count": 0}

    lightrag: Any = type("L", (), {"chunks_vdb": object(), "text_chunks": object()})()
    db = FakeTextChunksDB()
    lightrag.text_chunks = type("T", (), {"db": db, "workspace": "ws"})()
    stores = PGCorpusChunkStore(lightrag, exact_threshold=8192)

    scope = await stores.resolve_scope(MetadataFilter(filename="missing.pdf"))

    assert bool(scope) is False
    assert scope.candidate_count == 0
    assert scope.candidate_count_exact is True
    assert len(db.fetches) == 2  # exact miss widens to contains, which also misses


async def test_scope_resolver_reports_at_threshold_and_sentinel_exactly() -> None:
    from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore

    class FakeTextChunksDB:
        def __init__(self, count: int) -> None:
            self._count = count
            self.fetches: list[tuple[Any, ...]] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def fetchrow(self, *args):  # noqa: ANN002, ANN202
            self.fetches.append(args)
            return {"doc_exists": True, "chunk_count": self._count}

    for count, exact in ((3, True), (4, False)):
        lightrag: Any = type("L", (), {"chunks_vdb": object(), "text_chunks": object()})()
        db = FakeTextChunksDB(count)
        lightrag.text_chunks = type("T", (), {"db": db, "workspace": "ws"})()
        stores = PGCorpusChunkStore(lightrag, exact_threshold=3)

        scope = await stores.resolve_scope(MetadataFilter(file_extension="pdf"))

        assert scope.candidate_count == count
        assert scope.candidate_count_exact is exact
        assert len(db.fetches) == 1  # no filename: never widens


# ---------------------------------------------------------------------------
# Vector and BM25 legs share the identical database-side predicate
# ---------------------------------------------------------------------------


class _FakeVectorDB:
    vector_index_type = "HNSW"

    def __init__(self) -> None:
        self.sql: str | None = None
        self.params: tuple[object, ...] = ()

    async def _run_with_retry(self, operation):
        return await operation(self)

    def transaction(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def execute(self, sql: str) -> None:
        return None

    async def fetch(self, sql: str, *params):
        self.sql = sql
        self.params = params
        return []


def _vector_storage() -> Any:
    storage = type(
        "PGVectorStorage",
        (),
        {
            "table_name": "lightrag_vdb_chunks_test",
            "workspace": "default",
            "cosine_better_than_threshold": 0.3,
            "db": _FakeVectorDB(),
        },
    )()
    return storage


def test_filtered_vector_search_requires_postgres_capabilities() -> None:
    with pytest.raises(RuntimeError, match="cosine_better_than_threshold"):
        PGFilteredVectorSearch(type("IncompleteVectorStorage", (), {})())


async def test_exact_vector_leg_uses_the_metadata_semi_join_without_python_doc_ids() -> None:
    storage = _vector_storage()
    search = PGFilteredVectorSearch(storage, exact_threshold=8192)

    await search.search(
        [0.1, 0.2, 0.3],
        scope=_scope(candidate_count=5),
        top_k=3,
    )

    sql = storage.db.sql or ""
    assert "WITH candidate_rows AS MATERIALIZED" in sql
    assert "v.workspace = $2" in sql
    assert "v.full_doc_id IN (SELECT m.doc_id FROM dlightrag_doc_metadata m" in sql
    assert "m.workspace = $3" in sql
    assert "ORDER BY content_vector <=> $1::vector" in sql
    assert "LIMIT $6" in sql
    # The metadata predicate rides in the database; no doc-id array crosses.
    assert "ANY(" not in sql
    assert [round(float(value), 6) for value in storage.db.params[0].tolist()] == [0.1, 0.2, 0.3]
    assert storage.db.params[1] == "default"
    assert storage.db.params[2] == "default"  # metadata-side workspace
    assert storage.db.params[3] == "x.pdf"
    assert storage.db.params[4] == 0.3
    assert storage.db.params[5] == 3


async def test_hnsw_vector_leg_keeps_the_final_limit_outside_the_filter() -> None:
    storage = _vector_storage()
    search = PGFilteredVectorSearch(storage, exact_threshold=2)

    await search.search(
        [0.1, 0.2, 0.3],
        scope=_scope(candidate_count=9000, candidate_count_exact=False),
        top_k=5,
    )

    sql = storage.db.sql or ""
    assert "WITH nearest_results AS MATERIALIZED" in sql
    assert "full_doc_id IN (SELECT m.doc_id FROM dlightrag_doc_metadata m" in sql
    assert "LIMIT $6" in sql
    assert "FROM nearest_results" in sql
    assert "WHERE score > $5" in sql
    assert "ORDER BY distance + 0" in sql
    assert "ANY(" not in sql
    assert storage.db.params[-2:] == (0.3, 5)


def test_bm25_leg_uses_one_metadata_semi_join_and_keeps_the_index_as_source() -> None:
    from dlightrag.adapters.postgres.corpus.pg_metadata_index import metadata_match_conditions

    conditions, params = metadata_match_conditions(
        "research",
        MetadataFilter(custom={"team": "core"}),
        filename_mode="exact",
        start_index=3,
    )
    sql = build_bm25_sql(
        index_name="idx_lightrag_doc_chunks_bm25_en",
        scoped=True,
        limit=20,
        language="en",
        metadata_conditions=tuple(conditions),
    )

    assert "FROM LIGHTRAG_DOC_CHUNKS" in sql
    assert "WHERE workspace = $2" in sql
    assert "full_doc_id IN (SELECT doc_id FROM dlightrag_doc_metadata" in sql
    assert "workspace = $3" in sql
    assert "custom_metadata_search @> dlightrag_canonical_custom_metadata($4::jsonb)" in sql
    assert "ORDER BY content <@> to_bm25query($1, 'idx_lightrag_doc_chunks_bm25_en')" in sql
    assert "LIMIT $5" in sql
    assert "ANY(" not in sql
    assert params == ["research", json.dumps({"team": "core"})]


async def test_unscoped_vector_leg_ranks_bounded_window_then_checks_visibility() -> None:
    storage = _vector_storage()
    search = PGFilteredVectorSearch(storage)

    await search.search([0.1, 0.2, 0.3], scope=None, top_k=5)

    sql = storage.db.sql or ""
    assert "WITH nearest AS MATERIALIZED" in sql
    assert "LIMIT $3" in sql
    assert "EXISTS (SELECT 1 FROM dlightrag_doc_metadata m" in sql
    assert "m._dlightrag_finalization_complete IS TRUE" in sql
    assert "IN (SELECT" not in sql
    assert storage.db.params[-3:] == (20, 0.3, 5)


def test_unscoped_bm25_ranks_bounded_window_then_checks_visibility() -> None:
    sql = build_bm25_sql(
        index_name="idx_lightrag_doc_chunks_bm25_en",
        scoped=False,
        limit=5,
        language="en",
    )

    assert "WITH ranked AS MATERIALIZED" in sql
    assert "LIMIT $3" in sql
    assert "EXISTS (SELECT 1 FROM dlightrag_doc_metadata m" in sql
    assert "m._dlightrag_finalization_complete IS TRUE" in sql
    assert "IN (SELECT" not in sql
    assert "LIMIT $4" in sql


async def test_graph_chunk_read_without_user_filter_is_visibility_only() -> None:
    from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore

    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.fetch_call: tuple[str, tuple[Any, ...]] | None = None

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            return await operation(self)

        async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
            self.fetch_call = (sql, args)
            return []

    lightrag: Any = type("L", (), {"chunks_vdb": object(), "text_chunks": object()})()
    db = FakeTextChunksDB()
    lightrag.text_chunks = type("T", (), {"db": db, "workspace": "ws"})()
    stores = PGCorpusChunkStore(lightrag)

    assert await stores.read_scoped(None, ["chunk-a", "chunk-b"]) == [None, None]

    assert db.fetch_call is not None
    sql, args = db.fetch_call
    assert "c.id = ANY($2::text[])" in sql
    assert "m._dlightrag_finalization_complete IS TRUE" in sql
    assert args == ("ws", ["chunk-a", "chunk-b"])


async def test_scoped_vector_search_skips_an_empty_scope_entirely() -> None:
    storage = _vector_storage()
    search = PGFilteredVectorSearch(storage)

    scope = MetadataScope(
        filters=MetadataFilter(filename="missing.pdf"),
        filename_mode="exact",
        doc_exists=False,
        candidate_count=0,
        candidate_count_exact=True,
    )
    assert await search.search([0.1], scope=scope, top_k=3) == []
    assert storage.db.sql is None


async def test_retrieval_path_never_binds_a_document_id_array() -> None:
    """The complete doc-id match set must not cross the Python boundary.

    The graph chunk guard still binds the *already-bounded requested chunk ids*;
    nothing on the retrieval path binds a document-id array.
    """
    from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore

    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.fetches: list[tuple[Any, ...]] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def fetchrow(self, *args):  # noqa: ANN002, ANN202
            self.fetches.append(args)
            return {"doc_exists": True, "chunk_count": 2}

    lightrag: Any = type("L", (), {"chunks_vdb": object(), "text_chunks": object()})()
    db = FakeTextChunksDB()
    lightrag.text_chunks = type("T", (), {"db": db, "workspace": "ws"})()
    stores = PGCorpusChunkStore(lightrag)

    await stores.resolve_scope(MetadataFilter(filename="x.pdf"))

    probe_sql = db.fetches[0][0]
    probe_params = db.fetches[0][1:]
    assert "ANY(" not in probe_sql
    assert all(not isinstance(param, list) for param in probe_params)

    storage = _vector_storage()
    await PGFilteredVectorSearch(storage).search(
        [0.1],
        scope=_scope(candidate_count=1),
        top_k=1,
    )
    assert "ANY(" not in (storage.db.sql or "")
    assert all(not isinstance(param, list) for param in storage.db.params)
