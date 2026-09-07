# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG storage boundary adapter."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, call

import pytest

from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore
from dlightrag.engine.rag.lightrag.stores import LightRAGStores


class FakeLightRAG:
    def __init__(self) -> None:
        self.chunks_vdb = object()
        self.text_chunks = object()
        self.full_docs = object()
        self.doc_status = object()


def _stores(fake: FakeLightRAG) -> LightRAGStores:
    return LightRAGStores(fake, chunk_store=AsyncMock())


def test_lightrag_stores_validates_required_surfaces() -> None:
    fake = FakeLightRAG()
    stores = _stores(fake)

    assert stores.text_chunks is fake.text_chunks
    assert stores.full_docs is fake.full_docs


async def test_get_full_docs_uses_batch_kv_lookup_and_preserves_alignment() -> None:
    fake = FakeLightRAG()
    fake.full_docs = AsyncMock()
    expected = [{"id": "doc-b"}, None, {"id": "doc-a"}]
    fake.full_docs.get_by_ids.return_value = expected
    stores = _stores(fake)

    result = await stores.get_full_docs(["doc-b", "missing", "doc-a"])

    assert result == expected
    fake.full_docs.get_by_ids.assert_awaited_once_with(["doc-b", "missing", "doc-a"])
    fake.full_docs.get_by_id.assert_not_awaited()


async def test_get_full_docs_empty_input_skips_storage() -> None:
    fake = FakeLightRAG()
    fake.full_docs = AsyncMock()
    stores = _stores(fake)

    assert await stores.get_full_docs([]) == []
    fake.full_docs.get_by_ids.assert_not_awaited()


async def test_status_pages_use_strict_keyset_paging() -> None:
    from lightrag.base import CURSOR_END, CURSOR_START, CursorAfter, DocStatus

    fake = FakeLightRAG()
    fake.doc_status = AsyncMock()
    cursor = CursorAfter("next")
    fake.doc_status.get_docs_by_statuses_page.side_effect = [
        SimpleNamespace(docs={"doc-1": object()}, next_position=cursor),
        SimpleNamespace(docs={"doc-2": object()}, next_position=CURSOR_END),
    ]
    stores = _stores(fake)

    pages = [
        page async for page in stores.iter_doc_status_pages((DocStatus.PROCESSED,), page_size=1)
    ]

    assert [list(page) for page in pages] == [["doc-1"], ["doc-2"]]
    assert fake.doc_status.get_docs_by_statuses_page.await_args_list == [
        call(
            [DocStatus.PROCESSED],
            limit=1,
            position=CURSOR_START,
            strict=True,
        ),
        call(
            [DocStatus.PROCESSED],
            limit=1,
            position=cursor,
            strict=True,
        ),
    ]


async def test_full_doc_statuses_use_strict_batch_hydration() -> None:
    fake = FakeLightRAG()
    fake.doc_status = AsyncMock()
    expected = {"doc-1": object()}
    fake.doc_status.get_full_docs_by_ids.return_value = expected
    stores = _stores(fake)

    result = await stores.get_full_doc_statuses(["doc-1"])

    assert result == expected
    fake.doc_status.get_full_docs_by_ids.assert_awaited_once_with(["doc-1"], strict=True)


async def test_chunk_document_scope_index_is_owned_independently_of_bm25() -> None:
    class FakeDB:
        def __init__(self) -> None:
            self.executed: list[str] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label == "ws chunk_document_scope_index"
            return await operation(self)

        async def execute(self, sql: str) -> None:
            self.executed.append(sql)

    fake = FakeLightRAG()
    db = FakeDB()
    fake.text_chunks = SimpleNamespace(db=db, workspace="ws")

    await PGCorpusChunkStore(fake).ensure_document_scope_index()

    assert db.executed == [
        "CREATE INDEX IF NOT EXISTS idx_lightrag_doc_chunks_dlightrag_full_doc_id "
        "ON LIGHTRAG_DOC_CHUNKS(workspace, full_doc_id)"
    ]


async def test_overwrite_chunk_vectors_requires_matching_dimension() -> None:
    stores = PGCorpusChunkStore(FakeLightRAG())

    with pytest.raises(ValueError, match="vector dimension"):
        await stores.overwrite_chunk_vectors(
            {"chunk-1": [0.1, 0.2]},
            embedding_dim=3,
        )


async def test_overwrite_chunk_vectors_updates_existing_rows_only() -> None:
    class FakeDB:
        def __init__(self) -> None:
            self.executed: list[tuple] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def executemany(self, sql, values) -> None:  # noqa: ANN001
            self.executed.append((sql, values))

    fake = FakeLightRAG()
    db = FakeDB()
    fake.chunks_vdb = SimpleNamespace(table_name="LIGHTRAG_DOC_CHUNKS", db=db, workspace="ws")
    stores = PGCorpusChunkStore(fake)

    await stores.overwrite_chunk_vectors(
        {"doc-1-mm-drawing-000": [0.1, 0.2, 0.3]},
        embedding_dim=3,
    )

    assert len(db.executed) == 1
    sql, values = db.executed[0]
    assert "UPDATE LIGHTRAG_DOC_CHUNKS" in sql
    assert "INSERT" not in sql
    assert values[0][0] == "ws"
    assert values[0][1] == "doc-1-mm-drawing-000"
    assert values[0][2] == [0.1, 0.2, 0.3]


async def test_overwrite_chunk_vectors_respects_batch_record_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDB:
        def __init__(self) -> None:
            self.batches: list[list[tuple]] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def executemany(self, sql, values) -> None:  # noqa: ANN001
            self.batches.append(list(values))

    monkeypatch.setattr(PGCorpusChunkStore, "_VECTOR_WRITE_MAX_RECORDS", 1)
    monkeypatch.setattr(PGCorpusChunkStore, "_VECTOR_WRITE_MAX_BYTES", 16_000_000)

    fake = FakeLightRAG()
    db = FakeDB()
    fake.chunks_vdb = SimpleNamespace(table_name="LIGHTRAG_VDB", db=db, workspace="ws")

    stores = PGCorpusChunkStore(fake)
    await stores.overwrite_chunk_vectors(
        {
            "img-1": [0.1, 0.2, 0.3],
            "img-2": [0.4, 0.5, 0.6],
        },
        embedding_dim=3,
    )

    assert len(db.batches) == 2
    assert [batch[0][1] for batch in db.batches] == ["img-1", "img-2"]


async def test_resolve_scope_probes_bounded_chunk_count_without_document_ids() -> None:
    from dlightrag.engine.rag.retrieval import MetadataFilter

    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.fetches: list[tuple[Any, ...]] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def fetchrow(self, *args):  # noqa: ANN002, ANN202
            self.fetches.append(args)
            return {"doc_exists": True, "chunk_count": 12}

    fake = FakeLightRAG()
    db = FakeTextChunksDB()
    fake.text_chunks = SimpleNamespace(db=db, workspace="ws")
    stores = PGCorpusChunkStore(fake, exact_threshold=8192)

    scope = await stores.resolve_scope(MetadataFilter(filename="x.pdf"))

    assert scope.doc_exists is True
    assert scope.candidate_count == 12
    assert scope.candidate_count_exact is True
    assert scope.filename_mode == "exact"
    args = db.fetches[0]
    sql = args[0]
    params = args[1:]
    assert "EXISTS (SELECT 1 FROM dlightrag_doc_metadata" in sql
    assert "count(*)" in sql
    assert "LIMIT $6" in sql
    # One probe per attempted mode; the exact hit never widens.
    assert len(db.fetches) == 1
    # chunk workspace, then inner metadata predicates, then the probe cap.
    assert params[2] == "ws"
    assert params[3:5] == ("ws", "x.pdf")
    assert params[5] == 8193


async def test_resolve_scope_widens_to_contains_only_on_exact_miss() -> None:
    from dlightrag.engine.rag.retrieval import MetadataFilter

    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.fetches: list[tuple[Any, ...]] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def fetchrow(self, *args):  # noqa: ANN002, ANN202
            self.fetches.append(args)
            if len(self.fetches) == 1:
                return {"doc_exists": False, "chunk_count": 0}
            return {"doc_exists": True, "chunk_count": 4}

    fake = FakeLightRAG()
    db = FakeTextChunksDB()
    fake.text_chunks = SimpleNamespace(db=db, workspace="ws")
    stores = PGCorpusChunkStore(fake, exact_threshold=8192)

    scope = await stores.resolve_scope(MetadataFilter(filename="report"))

    assert scope.doc_exists is True
    assert scope.filename_mode == "contains"
    assert scope.candidate_count == 4
    assert len(db.fetches) == 2
    assert "LIKE LOWER($2) ESCAPE" in db.fetches[1][0]
    assert db.fetches[1][2] == "%report%"


async def test_resolve_scope_reports_the_cap_as_a_non_exact_sentinel() -> None:
    from dlightrag.engine.rag.retrieval import MetadataFilter

    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.fetch_args: tuple | None = None

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def fetchrow(self, *args):  # noqa: ANN002, ANN202
            self.fetch_args = args
            # The probe stopped at threshold + 1: a lower bound, never exact.
            return {"doc_exists": True, "chunk_count": 3}

    fake = FakeLightRAG()
    db = FakeTextChunksDB()
    fake.text_chunks = SimpleNamespace(db=db, workspace="ws")
    stores = PGCorpusChunkStore(fake, exact_threshold=2)

    scope = await stores.resolve_scope(MetadataFilter(file_extension="pdf"))

    assert scope.doc_exists is True
    assert scope.candidate_count == 3
    assert scope.candidate_count_exact is False
    assert scope.render_candidate_count() == "3+"


async def test_read_scoped_chunks_fuses_fetch_and_metadata_guard_in_one_query() -> None:
    from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope

    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.fetch_args: tuple | None = None

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def fetch(self, *args):  # noqa: ANN002, ANN202
            self.fetch_args = args
            return [
                {
                    "id": "c2",
                    "content": "beta",
                    "full_doc_id": "doc-1",
                    "file_path": "x.pdf",
                    "llm_cache_list": '["hit"]',
                    "heading": '{"h": 1}',
                    "sidecar": "{}",
                    "create_time": 10,
                    "update_time": 0,
                }
            ]

    fake = FakeLightRAG()
    db = FakeTextChunksDB()
    fake.text_chunks = SimpleNamespace(db=db, workspace="ws")
    stores = PGCorpusChunkStore(fake)
    scope = MetadataScope(
        filters=MetadataFilter(filename="x.pdf"),
        filename_mode="exact",
        doc_exists=True,
        candidate_count=2,
        candidate_count_exact=True,
    )

    rows = await stores.read_scoped(scope, ["c1", "c2", "c2"])

    assert db.fetch_args is not None
    sql = db.fetch_args[0]
    assert "EXISTS (SELECT 1 FROM dlightrag_doc_metadata m" in sql
    assert "c.id = ANY($2::text[])" in sql
    # Positional order with duplicates and None for missing/out-of-scope ids.
    assert rows[0] is None
    assert rows[1] is not None and rows[1]["id"] == "c2"
    assert rows[2] is not None and rows[2]["id"] == "c2"
    # JSON decoding matches LightRAG's get_by_ids text-chunk contract.
    assert rows[1]["llm_cache_list"] == ["hit"]
    assert rows[1]["heading"] == {"h": 1}
    assert rows[1]["sidecar"] == {}
    assert rows[1]["update_time"] == 10  # 0 folds back to create_time


async def test_read_scoped_chunks_skips_empty_request() -> None:
    from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope

    class FakeTextChunksDB:
        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            raise AssertionError("empty request must not reach storage")

    fake = FakeLightRAG()
    fake.text_chunks = SimpleNamespace(db=FakeTextChunksDB(), workspace="ws")
    stores = PGCorpusChunkStore(fake)
    scope = MetadataScope(
        filters=MetadataFilter(filename="x.pdf"),
        filename_mode="exact",
        doc_exists=True,
        candidate_count=0,
        candidate_count_exact=True,
    )

    assert await stores.read_scoped(scope, []) == []


async def test_read_scoped_chunks_skips_storage_for_empty_scope() -> None:
    from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope

    class FakeTextChunksDB:
        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            raise AssertionError("empty scope must not reach storage")

    fake = FakeLightRAG()
    fake.text_chunks = SimpleNamespace(db=FakeTextChunksDB(), workspace="ws")
    stores = PGCorpusChunkStore(fake)
    scope = MetadataScope(
        filters=MetadataFilter(filename="missing.pdf"),
        filename_mode="exact",
        doc_exists=False,
        candidate_count=0,
        candidate_count_exact=True,
    )

    assert await stores.read_scoped(scope, ["c1", "c2", "c1"]) == [None, None, None]


async def test_context_chunks_by_ids_formats_text_chunks() -> None:
    fake = FakeLightRAG()
    fake.text_chunks = AsyncMock()
    fake.text_chunks.get_by_ids.return_value = [
        {"content": "alpha", "file_path": "/tmp/a.pdf", "full_doc_id": "doc-a"},
        {"content": "beta", "file_path": "/tmp/b.pdf"},
    ]
    stores = _stores(fake)

    result = await stores.context_chunks_by_ids(["c1", "c2"])

    fake.text_chunks.get_by_ids.assert_awaited_once_with(["c1", "c2"])
    assert result == [
        {
            "chunk_id": "c1",
            "content": "alpha",
            "reference_id": "",
            "file_path": "/tmp/a.pdf",
            "full_doc_id": "doc-a",
        },
        {"chunk_id": "c2", "content": "beta", "reference_id": "", "file_path": "/tmp/b.pdf"},
    ]


async def test_fetch_chunk_contents_reads_lightrag_doc_chunks() -> None:
    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.fetch_args: tuple | None = None

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def fetch(self, *args):  # noqa: ANN002, ANN202
            self.fetch_args = args
            return [{"id": "chunk-a", "content": "hello"}]

    fake = FakeLightRAG()
    db = FakeTextChunksDB()
    fake.text_chunks = SimpleNamespace(db=db, workspace="ws")
    stores = PGCorpusChunkStore(fake)

    result = await stores.fetch_chunk_contents(["chunk-a"])

    assert result == [{"id": "chunk-a", "content": "hello"}]
    assert db.fetch_args is not None
    assert "FROM LIGHTRAG_DOC_CHUNKS" in db.fetch_args[0]
    assert db.fetch_args[1] == "ws"
    assert db.fetch_args[2] == ["chunk-a"]


async def test_update_chunk_bm25_languages_uses_batch_update() -> None:
    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.execute_args: tuple | None = None

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def execute(self, *args):  # noqa: ANN002, ANN202
            self.execute_args = args

    fake = FakeLightRAG()
    db = FakeTextChunksDB()
    fake.text_chunks = SimpleNamespace(db=db, workspace="ws")
    stores = PGCorpusChunkStore(fake)

    await stores.update_chunk_bm25_languages({"chunk-a": "en", "chunk-b": "zh"})

    assert db.execute_args is not None
    sql = db.execute_args[0]
    assert "UPDATE LIGHTRAG_DOC_CHUNKS AS chunks" in sql
    assert "FROM UNNEST($2::text[], $3::text[])" in sql
    assert "dlightrag_bm25_language" in sql
    assert db.execute_args[1] == "ws"
    assert db.execute_args[2] == ["chunk-a", "chunk-b"]
    assert db.execute_args[3] == ["en", "zh"]
