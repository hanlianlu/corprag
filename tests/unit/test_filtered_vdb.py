# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for strict metadata in-filtering context."""

from collections.abc import Sequence
from unittest.mock import AsyncMock

from dlightrag.adapters.postgres.corpus.corpus_vectors import PGFilteredVectorSearch
from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope
from dlightrag.engine.rag.retrieval.filtering import (
    FilteredChunkStore,
    FilteredVectorStorage,
    _active_filter,
    metadata_filter_scope,
)


def _scope(
    *, candidate_count: int, candidate_count_exact: bool = True, **overrides: object
) -> MetadataScope:
    return MetadataScope(
        filters=overrides.get("filters", MetadataFilter(filename="x.pdf")),  # type: ignore[arg-type]
        filename_mode=overrides.get("filename_mode", "exact"),  # type: ignore[arg-type]
        doc_exists=overrides.get("doc_exists", True),  # type: ignore[arg-type]
        candidate_count=candidate_count,
        candidate_count_exact=candidate_count_exact,
    )


class _FakeDB:
    vector_index_type = "HNSW"

    def __init__(self) -> None:
        self.sql: str | None = None
        self.params: tuple[object, ...] = ()
        self.local_settings: list[str] = []

    async def _run_with_retry(self, operation):
        return await operation(self)

    def transaction(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def execute(self, sql: str) -> None:
        self.local_settings.append(sql)

    async def fetch(self, sql: str, *params):
        self.sql = sql
        self.params = params
        return []


class _FakePGVectorStorage:
    table_name = "lightrag_vdb_chunks_test"
    workspace = "default"
    cosine_better_than_threshold = 0.3

    def __init__(self) -> None:
        self.db = _FakeDB()


_FakePGVectorStorage.__name__ = "PGVectorStorage"


async def test_empty_scope_is_active_filter() -> None:
    empty = _scope(candidate_count=0, doc_exists=False)
    async with metadata_filter_scope(empty):
        assert _active_filter.get() == empty


async def test_none_scope_has_no_user_metadata_filter() -> None:
    async with metadata_filter_scope(None):
        assert _active_filter.get() is None


async def test_filtered_query_uses_query_embedding_context() -> None:
    storage = type(
        "PGVectorStorage",
        (),
        {
            "table_name": "lightrag_vdb_chunks_test",
            "workspace": "default",
            "cosine_better_than_threshold": 0.3,
            "db": _FakeDB(),
        },
    )()
    embedding_func = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    filtered_search = AsyncMock()
    filtered_search.search.return_value = []
    wrapper = FilteredVectorStorage(
        original=storage,
        embedding_func=embedding_func,
        visibility_lookup=AsyncMock(),
        filtered_search=filtered_search,
    )

    async with metadata_filter_scope(_scope(candidate_count=3)):
        await wrapper.query("question", top_k=5)

    embedding_func.assert_awaited_once_with(["question"], context="query")


async def test_large_candidate_pg_search_places_distance_filter_outside_cte() -> None:
    storage = _FakePGVectorStorage()
    search = PGFilteredVectorSearch(storage, exact_threshold=1)

    await search.search(
        [0.1, 0.2, 0.3],
        scope=_scope(candidate_count=9_000, candidate_count_exact=False),
        top_k=5,
    )

    assert storage.db.local_settings == [
        "SET LOCAL hnsw.iterative_scan = 'relaxed_order'",
        "SET LOCAL hnsw.max_scan_tuples = 20000",
    ]
    assert storage.db.sql is not None
    assert "WITH nearest_results AS MATERIALIZED" in storage.db.sql
    assert "FROM nearest_results" in storage.db.sql
    cte_sql, outer_sql = storage.db.sql.split("FROM nearest_results", maxsplit=1)
    assert "score > $5" not in cte_sql
    assert "score > $5" in outer_sql


class _FakeChunkKV:
    """Mimics PGKVStorage.get_by_ids: caller ordering preserved, None for misses."""

    workspace = "default"

    def __init__(self, rows: dict[str, dict[str, object]]) -> None:
        self._rows = rows
        self.requested: list[list[str]] = []

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, object] | None]:
        self.requested.append(list(ids))
        return [self._rows.get(chunk_id) for chunk_id in ids]


class _FakeScopedReader:
    """Seam double: returns the positional list a scoped database read would."""

    def __init__(self, rows: dict[str, dict[str, object]]) -> None:
        self._rows = rows
        self.calls: list[tuple[MetadataScope | None, list[str]]] = []

    async def read_scoped(
        self,
        scope: MetadataScope | None,
        chunk_ids: list[str],
    ) -> list[dict[str, object] | None]:
        self.calls.append((scope, list(chunk_ids)))
        return [self._rows.get(chunk_id) for chunk_id in chunk_ids]


class _VisibleLookup:
    def __init__(self, visible: set[str], *, scoped_visible: set[str] | None = None) -> None:
        self.visible = visible
        self.scoped_visible = visible if scoped_visible is None else scoped_visible
        self.calls: list[tuple[list[str], MetadataScope | None]] = []

    async def is_visible(self, doc_id: str) -> bool:
        return doc_id in self.visible

    async def visible_subset(
        self,
        doc_ids: Sequence[str],
        *,
        scope: MetadataScope | None = None,
    ) -> frozenset[str]:
        self.calls.append((list(doc_ids), scope))
        eligible = self.visible if scope is None else self.scoped_visible
        return frozenset(eligible.intersection(doc_ids))


def _chunk(chunk_id: str, doc_id: str | None) -> dict[str, object]:
    return {"id": chunk_id, "content": chunk_id, "full_doc_id": doc_id}


async def test_chunk_store_passes_through_outside_product_retrieval() -> None:
    kv = _FakeChunkKV({"c1": _chunk("c1", "doc-1"), "c2": _chunk("c2", "doc-2")})
    reader = _FakeScopedReader({"c1": _chunk("c1", "doc-1")})
    store = FilteredChunkStore(
        original=kv,
        visibility_lookup=_VisibleLookup({"doc-1"}),
        scoped_reader=reader,
    )

    rows = await store.get_by_ids(["c1", "c2"])

    assert rows == [_chunk("c1", "doc-1"), _chunk("c2", "doc-2")]
    assert kv.requested == [["c1", "c2"]]
    assert reader.calls == []


async def test_chunk_store_uses_visibility_reader_without_user_filter() -> None:
    kv = _FakeChunkKV({"c1": _chunk("c1", "doc-1"), "c2": _chunk("c2", "doc-2")})
    reader = _FakeScopedReader({"c1": _chunk("c1", "doc-1")})
    store = FilteredChunkStore(
        original=kv,
        visibility_lookup=_VisibleLookup({"doc-1"}),
        scoped_reader=reader,
    )

    async with metadata_filter_scope(None):
        rows = await store.get_by_ids(["c1", "c2"])

    assert rows[0] is not None and rows[0]["id"] == "c1"
    assert rows[1] is None
    assert reader.calls == [(None, ["c1", "c2"])]
    assert kv.requested == []


async def test_chunk_store_nulls_out_of_scope_rows() -> None:
    reader = _FakeScopedReader({"c1": _chunk("c1", "doc-1")})
    kv = _FakeChunkKV({"c1": _chunk("c1", "doc-1"), "c2": _chunk("c2", "doc-2")})
    store = FilteredChunkStore(
        original=kv,
        visibility_lookup=_VisibleLookup({"doc-1"}),
        scoped_reader=reader,
    )
    scope = _scope(candidate_count=1)

    async with metadata_filter_scope(scope) as stats:
        rows = await store.get_by_ids(["c1", "c2"])

    # Positional alignment is the storage contract both KG legs zip against.
    assert rows[0] is not None and rows[0]["id"] == "c1"
    assert rows[1] is None
    assert stats.kg_chunks_dropped == 1
    assert stats.graph_strategy is True
    assert reader.calls == [(scope, ["c1", "c2"])]


async def test_chunk_store_drops_rows_without_document_attribution() -> None:
    kv = _FakeChunkKV({"c1": _chunk("c1", None)})
    reader = _FakeScopedReader({})
    store = FilteredChunkStore(
        original=kv,
        visibility_lookup=_VisibleLookup(set()),
        scoped_reader=reader,
    )

    async with metadata_filter_scope(_scope(candidate_count=1)):
        rows = await store.get_by_ids(["c1"])

    assert rows == [None]


async def test_chunk_store_still_requests_every_id() -> None:
    """Filtering must not shorten the request: callers zip results against their ids."""
    kv = _FakeChunkKV({"c1": _chunk("c1", "doc-1"), "c2": _chunk("c2", "doc-2")})
    reader = _FakeScopedReader({"c1": _chunk("c1", "doc-1")})
    store = FilteredChunkStore(
        original=kv,
        visibility_lookup=_VisibleLookup({"doc-1"}),
        scoped_reader=reader,
    )

    async with metadata_filter_scope(_scope(candidate_count=1)):
        rows = await store.get_by_ids(["c1", "c2"])

    assert kv.requested == []
    assert reader.calls[0][1] == ["c1", "c2"]
    assert len(rows) == 2


async def test_chunk_store_without_pushdown_postfilters_bounded_requested_ids() -> None:
    kv = _FakeChunkKV(
        {
            "c1": _chunk("c1", "doc-1"),
            "c2": _chunk("c2", "doc-2"),
            "c3": _chunk("c3", None),
        }
    )
    lookup = _VisibleLookup({"doc-1"})
    store = FilteredChunkStore(original=kv, visibility_lookup=lookup)

    async with metadata_filter_scope(None):
        rows = await store.get_by_ids(["c1", "c2", "c3"])

    assert rows == [_chunk("c1", "doc-1"), None, None]
    assert lookup.calls == [(["doc-1", "doc-2"], None)]


async def test_chunk_store_without_pushdown_applies_scope_and_preserves_positions() -> None:
    kv = _FakeChunkKV(
        {
            "c-match": _chunk("c-match", "doc-match"),
            "c-wrong": _chunk("c-wrong", "doc-wrong-metadata"),
            "c-false": _chunk("c-false", "doc-false-marker"),
            "c-missing": _chunk("c-missing", "doc-missing-marker"),
        }
    )
    lookup = _VisibleLookup(
        {"doc-match", "doc-wrong-metadata"},
        scoped_visible={"doc-match"},
    )
    store = FilteredChunkStore(original=kv, visibility_lookup=lookup)
    scope = _scope(candidate_count=1)
    requested = ["missing-chunk", "c-wrong", "c-match", "c-false", "c-missing", "c-match"]

    async with metadata_filter_scope(scope):
        rows = await store.get_by_ids(requested)

    assert rows == [
        None,
        None,
        _chunk("c-match", "doc-match"),
        None,
        None,
        _chunk("c-match", "doc-match"),
    ]
    assert kv.requested == [requested]
    assert lookup.calls == [
        (
            [
                "doc-wrong-metadata",
                "doc-match",
                "doc-false-marker",
                "doc-missing-marker",
            ],
            scope,
        )
    ]


async def test_chunk_store_empty_scope_skips_candidate_read_and_returns_aligned_none() -> None:
    kv = _FakeChunkKV({"c1": _chunk("c1", "doc-1")})
    lookup = _VisibleLookup({"doc-1"})
    store = FilteredChunkStore(original=kv, visibility_lookup=lookup)

    async with metadata_filter_scope(_scope(candidate_count=0, doc_exists=False)):
        rows = await store.get_by_ids(["c1", "missing", "c1"])

    assert rows == [None, None, None]
    assert kv.requested == []
    assert lookup.calls == []


async def test_chunk_store_proxies_unknown_attributes() -> None:
    kv = _FakeChunkKV({})
    store = FilteredChunkStore(original=kv, visibility_lookup=_VisibleLookup(set()))

    assert store.workspace == "default"


async def test_stats_stay_zero_when_no_retrieval_leg_runs() -> None:
    async with metadata_filter_scope(None) as stats:
        assert stats.kg_chunks_dropped == 0
        assert stats.graph_strategy is False
