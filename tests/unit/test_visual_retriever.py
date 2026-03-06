# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for VisualRetriever: KG retrieval, visual resolution, reranking, answer generation."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from dlightrag.unifiedrepresent.retriever import VisualRetriever

GRAPH_FIELD_SEP = "<SEP>"

# A tiny valid PNG (1x1 transparent pixel) encoded as base64.
_TINY_PNG_B64 = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
).decode()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(default_mode="mix", top_k=60, chunk_top_k=10)


def _make_lightrag_result() -> dict:
    """Return a mock aquery_data result with entities, relationships, chunks."""
    return {
        "data": {
            "entities": [
                {
                    "entity_name": "E1",
                    "entity_type": "CONCEPT",
                    "description": "First entity",
                    "source_id": "chunk-abc",
                },
            ],
            "relationships": [
                {
                    "src_id": "E1",
                    "tgt_id": "E2",
                    "description": "relates to",
                    "source_id": f"chunk-abc{GRAPH_FIELD_SEP}chunk-def",
                },
            ],
            "chunks": [
                {"chunk_id": "chunk-abc", "content": "text from chunk abc"},
            ],
        },
    }


def _make_visual_data() -> list[dict | None]:
    """Visual data list aligned with chunk_id_list from _make_lightrag_result.

    The lightrag result yields chunk IDs: {chunk-abc, chunk-def}.
    get_by_ids is called with a list of those IDs; we return data for both.
    """
    return [
        {
            "image_data": _TINY_PNG_B64,
            "page_index": 0,
            "doc_id": "doc-1",
            "doc_title": "Test",
            "content": "text abc",
        },
        {
            "image_data": _TINY_PNG_B64,
            "page_index": 1,
            "doc_id": "doc-1",
            "doc_title": "Test",
            "content": "text def",
        },
    ]


def _make_retriever(
    *,
    rerank_model: str | None = None,
    rerank_base_url: str | None = None,
    rerank_api_key: str | None = None,
    vision_model_func: AsyncMock | None = None,
    visual_data: list[dict | None] | None = None,
) -> VisualRetriever:
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(return_value=_make_lightrag_result())

    visual_chunks = MagicMock()
    visual_chunks.get_by_ids = AsyncMock(
        return_value=visual_data if visual_data is not None else _make_visual_data(),
    )

    return VisualRetriever(
        lightrag=lightrag,
        visual_chunks=visual_chunks,
        config=_make_config(),
        vision_model_func=vision_model_func,
        rerank_model=rerank_model,
        rerank_base_url=rerank_base_url,
        rerank_api_key=rerank_api_key,
    )


# ---------------------------------------------------------------------------
# TestFormatKgContext
# ---------------------------------------------------------------------------


class TestFormatKgContext:
    """Test _format_kg_context text formatting."""

    def test_entities_and_relationships(self) -> None:
        retriever = _make_retriever()
        contexts = {
            "entities": [
                {
                    "entity_name": "Python",
                    "entity_type": "LANGUAGE",
                    "description": "A programming language",
                },
            ],
            "relationships": [
                {
                    "src_id": "Python",
                    "tgt_id": "CPython",
                    "description": "default implementation",
                },
            ],
        }
        result = retriever._format_kg_context(contexts)
        assert "## Entities" in result
        assert "**Python** (LANGUAGE): A programming language" in result
        assert "## Relationships" in result
        assert "Python -> CPython: default implementation" in result

    def test_empty_contexts(self) -> None:
        retriever = _make_retriever()
        result = retriever._format_kg_context({"entities": [], "relationships": []})
        assert result == "No knowledge graph context available."

    def test_empty_dict(self) -> None:
        retriever = _make_retriever()
        result = retriever._format_kg_context({})
        assert result == "No knowledge graph context available."


# ---------------------------------------------------------------------------
# TestRetrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    """Test retrieve() — Phases 1-3 without reranking."""

    async def test_returned_structure(self) -> None:
        retriever = _make_retriever()
        result = await retriever.retrieve("What is E1?")

        # Top-level keys
        assert "contexts" in result
        assert "raw" in result

        # Contexts sub-keys
        assert "entities" in result["contexts"]
        assert "relationships" in result["contexts"]
        assert "chunks" in result["contexts"]

        # Raw sub-keys
        assert "sources" in result["raw"]
        assert "media" in result["raw"]

    async def test_entities_passed_through(self) -> None:
        retriever = _make_retriever()
        result = await retriever.retrieve("What is E1?")
        entities = result["contexts"]["entities"]
        assert len(entities) == 1
        assert entities[0]["entity_name"] == "E1"

    async def test_relationships_passed_through(self) -> None:
        retriever = _make_retriever()
        result = await retriever.retrieve("What is E1?")
        rels = result["contexts"]["relationships"]
        assert len(rels) == 1
        assert rels[0]["src_id"] == "E1"
        assert rels[0]["tgt_id"] == "E2"

    async def test_chunk_ids_extracted_from_all_sources(self) -> None:
        """chunk-abc comes from chunks + entities; chunk-def from relationships."""
        retriever = _make_retriever()
        await retriever.retrieve("query")

        # get_by_ids should be called with a list containing both chunk IDs
        call_args = retriever.visual_chunks.get_by_ids.call_args
        chunk_id_list = call_args[0][0]
        assert set(chunk_id_list) == {"chunk-abc", "chunk-def"}

    async def test_resolved_chunks_in_output(self) -> None:
        retriever = _make_retriever()
        result = await retriever.retrieve("query")
        chunks = result["contexts"]["chunks"]
        # Both chunks resolved
        assert len(chunks) == 2
        ref_ids = {c["reference_id"] for c in chunks}
        assert ref_ids == {"chunk-abc", "chunk-def"}

    async def test_media_contains_image_data(self) -> None:
        retriever = _make_retriever()
        result = await retriever.retrieve("query")
        media = result["raw"]["media"]
        assert len(media) == 2
        assert all(m["image_data"] is not None for m in media)

    async def test_sources_deduped_by_doc_id(self) -> None:
        retriever = _make_retriever()
        result = await retriever.retrieve("query")
        sources = result["raw"]["sources"]
        # Both chunks have doc_id="doc-1", so only one source
        assert len(sources) == 1
        assert sources[0]["doc_id"] == "doc-1"
        assert sources[0]["title"] == "Test"

    async def test_none_visual_data_filtered(self) -> None:
        """Visual chunks returning None for some IDs should be filtered out."""
        retriever = _make_retriever(
            visual_data=[
                {
                    "image_data": _TINY_PNG_B64,
                    "page_index": 0,
                    "doc_id": "doc-1",
                    "doc_title": "Test",
                    "content": "text",
                },
                None,  # chunk-def not found
            ]
        )
        result = await retriever.retrieve("query")
        chunks = result["contexts"]["chunks"]
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# TestRetrieveNoRerank
# ---------------------------------------------------------------------------


class TestRetrieveNoRerank:
    """Without rerank_model, resolved should be truncated to chunk_top_k."""

    async def test_truncated_to_chunk_top_k(self) -> None:
        # Create visual data with more items than chunk_top_k=2
        many_chunks = {
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [{"chunk_id": f"c-{i}", "content": f"text {i}"} for i in range(5)],
            },
        }
        visual_data_list = [
            {
                "image_data": f"img{i}",
                "page_index": i,
                "doc_id": "doc-1",
                "doc_title": "Test",
                "content": f"text {i}",
            }
            for i in range(5)
        ]

        lightrag = MagicMock()
        lightrag.aquery_data = AsyncMock(return_value=many_chunks)

        visual_chunks = MagicMock()
        visual_chunks.get_by_ids = AsyncMock(return_value=visual_data_list)

        retriever = VisualRetriever(
            lightrag=lightrag,
            visual_chunks=visual_chunks,
            config=_make_config(),
            vision_model_func=None,
            rerank_model=None,
        )

        result = await retriever.retrieve("query", chunk_top_k=2)
        assert len(result["contexts"]["chunks"]) == 2
        assert len(result["raw"]["media"]) == 2


# ---------------------------------------------------------------------------
# TestAnswer
# ---------------------------------------------------------------------------


class TestAnswer:
    """Test answer() — Phase 4 VLM generation."""

    @patch("PIL.Image.open", return_value=MagicMock())
    async def test_answer_returns_text(self, _mock_open) -> None:
        vision_func = AsyncMock(return_value="The answer is 42.")
        retriever = _make_retriever(vision_model_func=vision_func)

        result = await retriever.answer("What is E1?")

        assert result["answer"] == "The answer is 42."
        assert "contexts" in result
        assert "raw" in result
        vision_func.assert_awaited_once()

    async def test_no_vision_func_returns_none(self) -> None:
        retriever = _make_retriever(vision_model_func=None)
        result = await retriever.answer("query")
        assert result["answer"] is None
        assert "contexts" in result
        assert "raw" in result


# ---------------------------------------------------------------------------
# TestVisualRerank
# ---------------------------------------------------------------------------


class TestVisualRerank:
    """Test _visual_rerank HTTP call and sorting."""

    async def test_reranking_sorts_by_relevance(self) -> None:
        retriever = _make_retriever(
            rerank_model="reranker-v1",
            rerank_base_url="http://localhost:8080",
            rerank_api_key="key-123",
        )

        resolved = {
            "c-0": {"image_data": "img0", "content": "text0"},
            "c-1": {"image_data": "img1", "content": "text1"},
            "c-2": {"image_data": "img2", "content": "text2"},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.80},
                {"index": 1, "relevance_score": 0.60},
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await retriever._visual_rerank("query", resolved, top_k=3)

        # Sorted descending by relevance_score
        keys = list(result.keys())
        assert keys == ["c-2", "c-0", "c-1"]
        assert result["c-2"]["relevance_score"] == 0.95
        assert result["c-0"]["relevance_score"] == 0.80
        assert result["c-1"]["relevance_score"] == 0.60

    async def test_reranking_respects_top_k(self) -> None:
        retriever = _make_retriever(
            rerank_model="reranker-v1",
            rerank_base_url="http://localhost:8080",
        )

        resolved = {
            "c-0": {"image_data": "img0", "content": "text0"},
            "c-1": {"image_data": "img1", "content": "text1"},
            "c-2": {"image_data": "img2", "content": "text2"},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.80},
                {"index": 1, "relevance_score": 0.60},
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await retriever._visual_rerank("query", resolved, top_k=2)

        assert len(result) == 2

    async def test_reranking_failure_returns_unranked(self) -> None:
        retriever = _make_retriever(
            rerank_model="reranker-v1",
            rerank_base_url="http://localhost:8080",
        )

        resolved = {
            "c-0": {"content": "text0"},
            "c-1": {"content": "text1"},
            "c-2": {"content": "text2"},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.HTTPError("connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await retriever._visual_rerank("query", resolved, top_k=2)

        # Fallback: first top_k items by insertion order
        assert len(result) == 2
        assert list(result.keys()) == ["c-0", "c-1"]

    async def test_empty_resolved_returns_empty(self) -> None:
        retriever = _make_retriever(
            rerank_model="reranker-v1",
            rerank_base_url="http://localhost:8080",
        )
        result = await retriever._visual_rerank("query", {}, top_k=5)
        assert result == {}

    async def test_rerank_sends_correct_payload(self) -> None:
        retriever = _make_retriever(
            rerank_model="reranker-v1",
            rerank_base_url="http://localhost:8080/v1",
            rerank_api_key="secret",
        )

        resolved = {
            "c-0": {"image_data": "imgdata", "content": "text0"},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await retriever._visual_rerank("my query", resolved, top_k=5)

        # Verify the POST call
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[0][0] == "http://localhost:8080/v1/rerank"
        payload = call_kwargs[1]["json"]
        assert payload["model"] == "reranker-v1"
        assert payload["query"] == "my query"
        assert len(payload["documents"]) == 1
        assert payload["documents"][0]["type"] == "image_url"
        headers = call_kwargs[1]["headers"]
        assert headers["Authorization"] == "Bearer secret"
