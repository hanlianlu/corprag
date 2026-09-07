# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified retrieval orchestration."""

import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope
from dlightrag.engine.rag.retrieval.retriever import UnifiedRetriever


def _scope(
    *, candidate_count: int = 0, candidate_count_exact: bool = True, doc_exists: bool = False
) -> MetadataScope:
    return MetadataScope(
        filters=MetadataFilter(filename="x.pdf"),
        filename_mode="exact",
        doc_exists=doc_exists,
        candidate_count=candidate_count,
        candidate_count_exact=candidate_count_exact,
    )


async def test_unified_retriever_empty_metadata_candidates_short_circuits() -> None:
    stores = AsyncMock()
    stores.resolve_scope.return_value = _scope()
    backend = AsyncMock()
    bm25 = AsyncMock()
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        stores=stores,
    )

    result = await retriever.aretrieve("query", metadata_filter=MetadataFilter(filename="x.pdf"))

    assert result.contexts == {"chunks": [], "entities": [], "relationships": []}
    backend.aretrieve.assert_not_called()
    bm25.search.assert_not_called()


async def test_unified_retriever_llm_empty_candidates_falls_back_unfiltered() -> None:
    from dlightrag.engine.rag.retrieval import RetrievalResult

    stores = AsyncMock()
    stores.resolve_scope.return_value = _scope()
    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={"chunks": [{"chunk_id": "semantic-a"}], "entities": [], "relationships": []}
    )
    bm25 = AsyncMock()
    bm25.search.return_value = []
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        stores=stores,
    )

    result = await retriever.aretrieve(
        "query",
        metadata_filter=MetadataFilter(filename="missing.pdf"),
        metadata_filter_source="llm_inferred",
    )

    assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["semantic-a"]
    backend.aretrieve.assert_awaited_once()
    bm25.search.assert_awaited_once()
    assert bm25.search.await_args.kwargs["scope"] is None


async def test_unified_retriever_llm_filtered_empty_falls_back_unfiltered() -> None:
    from dlightrag.engine.rag.retrieval import RetrievalResult

    stores = AsyncMock()
    stores.resolve_scope.return_value = _scope(candidate_count=12, doc_exists=True)
    backend = AsyncMock()
    backend.aretrieve.side_effect = [
        RetrievalResult(contexts={"chunks": [], "entities": [], "relationships": []}),
        RetrievalResult(
            contexts={"chunks": [{"chunk_id": "semantic-a"}], "entities": [], "relationships": []}
        ),
    ]
    bm25 = AsyncMock()
    bm25.search.side_effect = [[], []]
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        stores=stores,
    )

    result = await retriever.aretrieve(
        "query",
        metadata_filter=MetadataFilter(filename="maybe.pdf"),
        metadata_filter_source="llm_inferred",
    )

    assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["semantic-a"]
    assert result.trace["metadata_filter_relaxed"] is True
    assert result.trace["metadata_match_exists"] is True
    assert result.trace["metadata_candidate_count"] == 12
    assert result.trace["metadata_candidate_count_exact"] is True
    assert result.trace["metadata_candidate_count_lower_bound"] is None
    assert backend.aretrieve.await_count == 2
    assert bm25.search.await_args_list[0].kwargs["scope"].candidate_count == 12
    assert bm25.search.await_args_list[1].kwargs["scope"] is None


async def test_unified_retriever_explicit_filtered_empty_stays_filtered() -> None:
    from dlightrag.engine.rag.retrieval import RetrievalResult

    stores = AsyncMock()
    stores.resolve_scope.return_value = _scope(candidate_count=12, doc_exists=True)
    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={"chunks": [], "entities": [], "relationships": []}
    )
    bm25 = AsyncMock()
    bm25.search.return_value = []
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        stores=stores,
    )

    result = await retriever.aretrieve(
        "query",
        metadata_filter=MetadataFilter(filename="exact.pdf"),
        metadata_filter_source="explicit",
    )

    assert result.contexts["chunks"] == []
    assert result.trace["metadata_filter_relaxed"] is False
    backend.aretrieve.assert_awaited_once()
    bm25.search.assert_awaited_once()


async def test_unified_retriever_fuses_lightrag_and_bm25_chunks() -> None:
    from dlightrag.engine.rag.retrieval import RetrievalResult

    stores = AsyncMock()
    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={
            "chunks": [{"chunk_id": "semantic-a"}, {"chunk_id": "shared"}],
            "entities": [{"entity_name": "E"}],
            "relationships": [],
        }
    )
    bm25 = AsyncMock()
    bm25.search.return_value = [{"chunk_id": "shared"}, {"chunk_id": "bm25-b"}]
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        stores=stores,
        rrf_k=60,
    )

    result = await retriever.aretrieve("query", top_k=3)

    assert [c["chunk_id"] for c in result.contexts["chunks"]] == [
        "shared",
        "semantic-a",
        "bm25-b",
    ]
    assert result.contexts["entities"] == [{"entity_name": "E"}]


async def test_unified_retriever_fuses_visual_leg_as_an_independent_ranking() -> None:
    from dlightrag.engine.rag.retrieval import RetrievalResult

    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={
            "chunks": [{"chunk_id": "semantic-a"}, {"chunk_id": "shared"}],
            "entities": [],
            "relationships": [],
        }
    )
    bm25 = AsyncMock()
    bm25.search.return_value = [{"chunk_id": "bm25-b"}]
    visual = AsyncMock()
    visual.search.return_value = [{"chunk_id": "shared"}, {"chunk_id": "visual-c"}]
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        visual=visual,
        stores=AsyncMock(),
        rrf_k=60,
    )

    result = await retriever.aretrieve("query", top_k=3, query_image_blocks=[{"type": "image_url"}])

    # "shared" wins on two-leg agreement; the visual leg reaches fusion whole,
    # never pre-merged into the semantic ranking.
    assert [c["chunk_id"] for c in result.contexts["chunks"]][0] == "shared"
    assert {c["chunk_id"] for c in result.contexts["chunks"]} == {
        "shared",
        "semantic-a",
        "visual-c",
        "bm25-b",
    }
    assert result.trace["direct_visual_chunk_count"] == 2
    assert result.trace["fused_multi_source_count"] == 1
    assert "query_image_blocks" not in backend.aretrieve.await_args.kwargs


async def test_unified_retriever_skips_visual_leg_without_query_images() -> None:
    from dlightrag.engine.rag.retrieval import RetrievalResult

    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={"chunks": [{"chunk_id": "semantic-a"}], "entities": [], "relationships": []}
    )
    bm25 = AsyncMock()
    bm25.search.return_value = []
    visual = AsyncMock()
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        visual=visual,
        stores=AsyncMock(),
    )

    result = await retriever.aretrieve("query", top_k=3)

    visual.search.assert_not_called()
    assert result.trace["direct_visual_chunk_count"] == 0


async def test_unified_retriever_does_not_cap_fused_chunks_to_candidate_limit() -> None:
    from dlightrag.engine.rag.retrieval import RetrievalResult

    stores = AsyncMock()
    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={
            "chunks": [{"chunk_id": "semantic-a"}, {"chunk_id": "semantic-b"}],
            "entities": [],
            "relationships": [],
        }
    )
    bm25 = AsyncMock()
    bm25.search.return_value = [{"chunk_id": "bm25-a"}, {"chunk_id": "bm25-b"}]
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        stores=stores,
    )

    result = await retriever.aretrieve("query", top_k=2)

    assert bm25.search.await_args.kwargs["top_k"] == 2
    assert [c["chunk_id"] for c in result.contexts["chunks"]] == [
        "semantic-a",
        "bm25-a",
        "semantic-b",
        "bm25-b",
    ]


async def test_unified_retriever_keeps_distinct_chunks_with_same_content_prefix() -> None:
    from dlightrag.engine.rag.retrieval import RetrievalResult

    shared = "The quick brown fox jumps. " + "x" * 173
    stores = AsyncMock()
    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={
            "chunks": [
                {"chunk_id": "semantic-a", "content": shared + " semantic suffix"},
            ],
            "entities": [],
            "relationships": [],
        }
    )
    bm25 = AsyncMock()
    bm25.search.return_value = [
        {"chunk_id": "bm25-b", "content": shared + " bm25 suffix"},
    ]
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        stores=stores,
    )

    result = await retriever.aretrieve("query", top_k=2)

    assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["semantic-a", "bm25-b"]


async def test_unified_retriever_logs_retrieval_mix_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dlightrag.engine.rag.retrieval import RetrievalResult

    stores = AsyncMock()
    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={
            "chunks": [{"chunk_id": "semantic-a"}, {"chunk_id": "shared"}],
            "entities": [],
            "relationships": [],
        }
    )
    bm25 = AsyncMock()
    bm25.search.return_value = [
        {"chunk_id": "shared", "bm25_profile": "en", "score": 2.0},
        {"chunk_id": "bm25-b", "bm25_profile": "en", "score": 1.0},
    ]
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        stores=stores,
        rrf_k=60,
    )

    with caplog.at_level(logging.INFO, logger="dlightrag.engine.rag.retrieval.retriever"):
        result = await retriever.aretrieve("query", bm25_query="keyword query", top_k=3)

    assert "[Retriever] mix" in caplog.text
    assert "bm25_enabled=True" in caplog.text
    assert "bm25_query='keyword query'" in caplog.text
    assert "lightrag_mix_chunks=2" in caplog.text
    assert "semantic_chunks" not in caplog.text
    assert "bm25_chunks=2" in caplog.text
    assert "fused_chunks=3" in caplog.text
    assert "multi_source=1" in caplog.text
    assert "bm25_top=shared:en:2.000,bm25-b:en:1.000" in caplog.text
    assert result.trace["lightrag_mix_chunk_count"] == 2
    assert "semantic_chunk_count" not in result.trace
    assert result.trace["fused_multi_source_count"] == 1


async def test_unified_retriever_lightrag_failure_falls_back_to_bm25() -> None:
    """When LightRAG retrieval raises, BM25 results must still be returned."""

    stores = AsyncMock()
    backend = AsyncMock()
    backend.aretrieve.side_effect = RuntimeError("LightRAG backend down")
    bm25 = AsyncMock()
    bm25.search.return_value = [{"chunk_id": "bm25-a"}, {"chunk_id": "bm25-b"}]
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        stores=stores,
    )

    result = await retriever.aretrieve("query", top_k=5)

    assert result.trace.get("lightrag_error") is True
    assert len(result.contexts["chunks"]) == 2
    assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["bm25-a", "bm25-b"]


async def test_unified_retriever_bm25_failure_continues_without_bm25(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dlightrag.engine.rag.retrieval import RetrievalResult

    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={
            "chunks": [{"chunk_id": "semantic-a"}],
            "entities": [],
            "relationships": [],
        }
    )
    bm25 = AsyncMock()
    bm25.search.side_effect = RuntimeError("BM25 unavailable")
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        stores=AsyncMock(),
    )

    result = await retriever.aretrieve("query", top_k=5)

    assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["semantic-a"]
    assert result.trace["bm25_error_type"] == "RuntimeError"
    assert result.trace["bm25_chunk_count"] == 0
    assert "BM25 retrieval failed; continuing without BM25" in caplog.text


async def test_unified_retriever_raises_semantic_error_when_both_lanes_fail() -> None:
    semantic_error = RuntimeError("semantic unavailable")
    bm25_error = ConnectionError("BM25 unavailable")
    backend = AsyncMock()
    backend.aretrieve.side_effect = semantic_error
    bm25 = AsyncMock()
    bm25.search.side_effect = bm25_error
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        stores=AsyncMock(),
    )

    with pytest.raises(RuntimeError, match="semantic unavailable") as exc_info:
        await retriever.aretrieve("query", top_k=5)

    assert exc_info.value is semantic_error
    assert exc_info.value.__cause__ is bm25_error


async def test_unified_retriever_raises_semantic_error_when_bm25_is_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    semantic_error = RuntimeError("semantic unavailable")
    backend = AsyncMock()
    backend.aretrieve.side_effect = semantic_error
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=None,
        stores=AsyncMock(),
    )

    with caplog.at_level(logging.ERROR, logger="dlightrag.engine.rag.retrieval.retriever"):
        with pytest.raises(RuntimeError, match="semantic unavailable") as exc_info:
            await retriever.aretrieve("query", top_k=5)

    assert exc_info.value is semantic_error
    assert "BM25 is disabled" in caplog.text
    assert "falling back to BM25-only" not in caplog.text


async def test_unified_retriever_traces_kg_chunks_dropped_by_scope() -> None:
    """The KG legs run inside the scope, so their drops must reach the trace."""
    from dlightrag.engine.rag.retrieval import RetrievalResult
    from dlightrag.engine.rag.retrieval.filtering import FilteredChunkStore

    class _ScopedReader:
        async def read_scoped(
            self,
            scope: MetadataScope | None,
            chunk_ids: list[str],
        ) -> list[dict[str, Any] | None]:
            rows: list[dict[str, Any] | None] = [
                {"id": "in", "full_doc_id": "doc-1"},
                {"id": "out", "full_doc_id": "doc-9"},
            ]
            return [rows[0], None]

    chunk_store = FilteredChunkStore(
        original=AsyncMock(
            get_by_ids=AsyncMock(
                return_value=[
                    {"id": "in", "full_doc_id": "doc-1"},
                    {"id": "out", "full_doc_id": "doc-9"},
                ]
            )
        ),
        visibility_lookup=AsyncMock(),
        scoped_reader=_ScopedReader(),
    )

    async def _backend_retrieve(*args: object, **kwargs: object) -> RetrievalResult:
        # Stands in for LightRAG's entity/relation legs resolving chunks by id.
        await chunk_store.get_by_ids(["in", "out"])
        return RetrievalResult(contexts={"chunks": [], "entities": [], "relationships": []})

    stores = AsyncMock()
    stores.resolve_scope.return_value = _scope(candidate_count=5, doc_exists=True)
    backend = AsyncMock()
    backend.aretrieve.side_effect = _backend_retrieve
    bm25 = AsyncMock()
    bm25.search.return_value = []
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        stores=stores,
    )

    result = await retriever.aretrieve(
        "query",
        metadata_filter=MetadataFilter(filename="x.pdf"),
        metadata_filter_source="explicit",
    )

    assert result.trace["metadata_kg_chunks_dropped"] == 1


async def test_unified_retriever_traces_capped_probe_as_a_lower_bound() -> None:
    from dlightrag.engine.rag.retrieval import RetrievalResult

    stores = AsyncMock()
    stores.resolve_scope.return_value = _scope(
        candidate_count=8193, candidate_count_exact=False, doc_exists=True
    )
    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={"chunks": [], "entities": [], "relationships": []}
    )
    bm25 = AsyncMock()
    bm25.search.return_value = []
    retriever = UnifiedRetriever(backend=backend, bm25=bm25, stores=stores)

    result = await retriever.aretrieve(
        "query",
        metadata_filter=MetadataFilter(filename="huge.pdf"),
        metadata_filter_source="explicit",
    )

    # A capped probe is reported as a lower bound, never as an exact total.
    assert result.trace["metadata_match_exists"] is True
    assert result.trace["metadata_candidate_count"] == 8193
    assert result.trace["metadata_candidate_count_exact"] is False
    assert result.trace["metadata_candidate_count_lower_bound"] == "8193+"


async def test_unified_retriever_traces_execution_strategy_and_candidate_shortfall() -> None:
    from dlightrag.engine.rag.retrieval import RetrievalResult
    from dlightrag.engine.rag.retrieval.filtering import current_filter_stats

    stores = AsyncMock()
    stores.resolve_scope.return_value = _scope(candidate_count=30, doc_exists=True)
    backend = AsyncMock()

    async def _backend_retrieve(*args: object, **kwargs: object) -> RetrievalResult:
        stats = current_filter_stats()
        assert stats is not None
        stats.vector_strategy = "exact_vector"
        stats.vector_candidate_shortfall = 3
        return RetrievalResult(
            contexts={"chunks": [{"chunk_id": "semantic-a"}], "entities": [], "relationships": []}
        )

    backend.aretrieve.side_effect = _backend_retrieve
    bm25 = AsyncMock()
    bm25.search.return_value = [{"chunk_id": "bm25-a"}]
    retriever = UnifiedRetriever(backend=backend, bm25=bm25, stores=stores)

    result = await retriever.aretrieve(
        "query",
        metadata_filter=MetadataFilter(filename="x.pdf"),
        metadata_filter_source="explicit",
    )

    assert result.trace["metadata_execution_strategy"] == "exact_vector"
    assert result.trace["metadata_candidate_shortfall"] == {"vector": 3}
