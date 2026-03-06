# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for EntityExtractor (VLM + LightRAG entity extraction pipeline)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from lightrag.utils import compute_mdhash_id

from dlightrag.unifiedrepresent.extractor import EntityExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeLightRAG:
    """Lightweight stand-in for LightRAG with mock storage attributes."""

    def __init__(self) -> None:
        self.llm_response_cache = MagicMock()
        self.text_chunks = MagicMock()
        self.chunk_entity_relation_graph = MagicMock()
        self.entities_vdb = MagicMock()
        self.relationships_vdb = MagicMock()
        self.full_entities = MagicMock()
        self.full_relations = MagicMock()


def _make_lightrag() -> _FakeLightRAG:
    """Create a fake LightRAG instance with all attributes EntityExtractor needs."""
    return _FakeLightRAG()


# ---------------------------------------------------------------------------
# TestEntityExtractorInit
# ---------------------------------------------------------------------------


class TestEntityExtractorInit:
    """Verify constructor stores attributes correctly."""

    def test_stores_lightrag(self) -> None:
        lightrag = _make_lightrag()
        ext = EntityExtractor(lightrag, ["person"], AsyncMock())
        assert ext.lightrag is lightrag

    def test_stores_entity_types(self) -> None:
        entity_types = ["person", "organization", "location"]
        ext = EntityExtractor(_make_lightrag(), entity_types, AsyncMock())
        assert ext.entity_types is entity_types

    def test_stores_vision_model_func(self) -> None:
        vision_fn = AsyncMock()
        ext = EntityExtractor(_make_lightrag(), ["person"], vision_fn)
        assert ext.vision_model_func is vision_fn

    def test_semaphore_default_value(self) -> None:
        ext = EntityExtractor(_make_lightrag(), ["person"], AsyncMock())
        # asyncio.Semaphore stores initial value in _value
        assert isinstance(ext._vlm_semaphore, asyncio.Semaphore)
        assert ext._vlm_semaphore._value == 4  # default max_concurrent_vlm

    def test_semaphore_custom_value(self) -> None:
        ext = EntityExtractor(_make_lightrag(), ["person"], AsyncMock(), max_concurrent_vlm=8)
        assert ext._vlm_semaphore._value == 8


# ---------------------------------------------------------------------------
# TestDescribePage
# ---------------------------------------------------------------------------


class TestDescribePage:
    """Test _describe_page VLM call."""

    async def test_returns_stripped_description(self) -> None:
        vision_fn = AsyncMock(return_value="  Page description text  ")
        ext = EntityExtractor(_make_lightrag(), ["person", "org"], vision_fn)

        result = await ext._describe_page(MagicMock(), page_index=0)

        assert result == "Page description text"

    async def test_prompt_contains_entity_types(self) -> None:
        vision_fn = AsyncMock(return_value="Some description")
        entity_types = ["person", "organization", "location"]
        ext = EntityExtractor(_make_lightrag(), entity_types, vision_fn)

        await ext._describe_page(MagicMock(), page_index=0)

        # The first positional arg to vision_model_func is the prompt
        prompt = vision_fn.call_args[0][0]
        assert "person, organization, location" in prompt

    async def test_empty_response_returns_fallback(self) -> None:
        vision_fn = AsyncMock(return_value="")
        ext = EntityExtractor(_make_lightrag(), ["person"], vision_fn)

        result = await ext._describe_page(MagicMock(), page_index=0)

        assert result == "[Page 1: no content extracted]"

    async def test_none_response_returns_fallback(self) -> None:
        vision_fn = AsyncMock(return_value=None)
        ext = EntityExtractor(_make_lightrag(), ["person"], vision_fn)

        result = await ext._describe_page(MagicMock(), page_index=2)

        assert result == "[Page 3: no content extracted]"

    async def test_whitespace_only_response_returns_fallback(self) -> None:
        vision_fn = AsyncMock(return_value="   \n\t  ")
        ext = EntityExtractor(_make_lightrag(), ["person"], vision_fn)

        result = await ext._describe_page(MagicMock(), page_index=1)

        assert result == "[Page 2: no content extracted]"

    async def test_image_passed_to_vision_func(self) -> None:
        vision_fn = AsyncMock(return_value="desc")
        ext = EntityExtractor(_make_lightrag(), ["person"], vision_fn)
        mock_image = MagicMock()

        await ext._describe_page(mock_image, page_index=0)

        _, kwargs = vision_fn.call_args
        assert kwargs["images"] == [mock_image]


# ---------------------------------------------------------------------------
# TestExtractFromPages
# ---------------------------------------------------------------------------


class TestExtractFromPages:
    """Test extract_from_pages end-to-end flow."""

    async def test_returns_list_with_correct_entries(self) -> None:
        vision_fn = AsyncMock(return_value="Page description")
        lightrag = _make_lightrag()
        ext = EntityExtractor(lightrag, ["person"], vision_fn)
        images = [MagicMock(), MagicMock()]

        with (
            patch(
                "dlightrag.unifiedrepresent.extractor.extract_entities",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "dlightrag.unifiedrepresent.extractor.merge_nodes_and_edges",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await ext.extract_from_pages(images, "doc-123", "/path/file.pdf")

        assert len(result) == 2
        for entry in result:
            assert "chunk_id" in entry
            assert "page_index" in entry
            assert "content" in entry

    async def test_page_indices_are_correct(self) -> None:
        vision_fn = AsyncMock(return_value="Description")
        ext = EntityExtractor(_make_lightrag(), ["person"], vision_fn)
        images = [MagicMock(), MagicMock()]

        with (
            patch(
                "dlightrag.unifiedrepresent.extractor.extract_entities",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "dlightrag.unifiedrepresent.extractor.merge_nodes_and_edges",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await ext.extract_from_pages(images, "doc-1", "/f.pdf")

        assert result[0]["page_index"] == 0
        assert result[1]["page_index"] == 1

    async def test_chunk_id_uses_compute_mdhash_id(self) -> None:
        vision_fn = AsyncMock(return_value="Desc")
        ext = EntityExtractor(_make_lightrag(), ["person"], vision_fn)
        doc_id = "doc-abc"
        images = [MagicMock(), MagicMock()]

        with (
            patch(
                "dlightrag.unifiedrepresent.extractor.extract_entities",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "dlightrag.unifiedrepresent.extractor.merge_nodes_and_edges",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await ext.extract_from_pages(images, doc_id, "/f.pdf")

        expected_id_0 = compute_mdhash_id(f"{doc_id}:page:0", prefix="chunk-")
        expected_id_1 = compute_mdhash_id(f"{doc_id}:page:1", prefix="chunk-")
        assert result[0]["chunk_id"] == expected_id_0
        assert result[1]["chunk_id"] == expected_id_1

    async def test_content_is_vlm_description(self) -> None:
        vision_fn = AsyncMock(side_effect=["First page desc", "Second page desc"])
        ext = EntityExtractor(_make_lightrag(), ["person"], vision_fn)
        images = [MagicMock(), MagicMock()]

        with (
            patch(
                "dlightrag.unifiedrepresent.extractor.extract_entities",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "dlightrag.unifiedrepresent.extractor.merge_nodes_and_edges",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await ext.extract_from_pages(images, "doc-1", "/f.pdf")

        assert result[0]["content"] == "First page desc"
        assert result[1]["content"] == "Second page desc"

    async def test_extract_entities_called_with_correct_chunks(self) -> None:
        vision_fn = AsyncMock(return_value="Page text")
        lightrag = _make_lightrag()
        ext = EntityExtractor(lightrag, ["person"], vision_fn)
        doc_id = "doc-xyz"
        images = [MagicMock()]

        with (
            patch(
                "dlightrag.unifiedrepresent.extractor.extract_entities",
                new_callable=AsyncMock,
                return_value={},
            ) as mock_extract,
            patch(
                "dlightrag.unifiedrepresent.extractor.merge_nodes_and_edges",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await ext.extract_from_pages(images, doc_id, "/path.pdf")

        mock_extract.assert_called_once()
        call_kwargs = mock_extract.call_args[1]

        chunk_id = compute_mdhash_id(f"{doc_id}:page:0", prefix="chunk-")
        chunks = call_kwargs["chunks"]
        assert chunk_id in chunks
        assert chunks[chunk_id]["content"] == "Page text"
        assert chunks[chunk_id]["full_doc_id"] == doc_id
        assert chunks[chunk_id]["chunk_order_index"] == 0
        assert chunks[chunk_id]["file_path"] == "/path.pdf"
        assert call_kwargs["global_config"] is lightrag.__dict__
        assert call_kwargs["llm_response_cache"] is lightrag.llm_response_cache
        assert call_kwargs["text_chunks_storage"] is lightrag.text_chunks

    async def test_merge_nodes_and_edges_called_with_correct_params(self) -> None:
        vision_fn = AsyncMock(return_value="Page text")
        lightrag = _make_lightrag()
        ext = EntityExtractor(lightrag, ["person"], vision_fn)
        doc_id = "doc-merge"
        mock_chunk_results = {"some": "result"}

        with (
            patch(
                "dlightrag.unifiedrepresent.extractor.extract_entities",
                new_callable=AsyncMock,
                return_value=mock_chunk_results,
            ),
            patch(
                "dlightrag.unifiedrepresent.extractor.merge_nodes_and_edges",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_merge,
        ):
            await ext.extract_from_pages([MagicMock()], doc_id, "/merge.pdf")

        mock_merge.assert_called_once()
        call_kwargs = mock_merge.call_args[1]
        assert call_kwargs["chunk_results"] is mock_chunk_results
        assert call_kwargs["knowledge_graph_inst"] is lightrag.chunk_entity_relation_graph
        assert call_kwargs["entity_vdb"] is lightrag.entities_vdb
        assert call_kwargs["relationships_vdb"] is lightrag.relationships_vdb
        assert call_kwargs["global_config"] is lightrag.__dict__
        assert call_kwargs["full_entities_storage"] is lightrag.full_entities
        assert call_kwargs["full_relations_storage"] is lightrag.full_relations
        assert call_kwargs["doc_id"] == doc_id
        assert call_kwargs["llm_response_cache"] is lightrag.llm_response_cache
        assert call_kwargs["file_path"] == "/merge.pdf"

    async def test_empty_images_returns_empty_list(self) -> None:
        vision_fn = AsyncMock()
        ext = EntityExtractor(_make_lightrag(), ["person"], vision_fn)

        with (
            patch(
                "dlightrag.unifiedrepresent.extractor.extract_entities",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "dlightrag.unifiedrepresent.extractor.merge_nodes_and_edges",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await ext.extract_from_pages([], "doc-1", "/f.pdf")

        assert result == []
        vision_fn.assert_not_called()
