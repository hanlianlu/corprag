# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for UnifiedRepresentEngine orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from dlightrag.unifiedrepresent.engine import UnifiedRepresentEngine
from dlightrag.unifiedrepresent.renderer import RenderResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config() -> MagicMock:
    config = MagicMock()
    config.page_render_dpi = 250
    config.embedding_model = "test-model"
    config.embedding_dim = 1024
    config.effective_embedding_provider = "openai"
    config._get_url = MagicMock(return_value="http://localhost:8000/v1")
    config._get_provider_api_key = MagicMock(return_value="test-key")
    config.kg_entity_types = ["Person", "Organization"]
    config.enable_rerank = False
    config.default_mode = "mix"
    config.top_k = 60
    config.chunk_top_k = 10
    return config


def _make_lightrag() -> MagicMock:
    lightrag = MagicMock()
    lightrag.full_docs = MagicMock()
    lightrag.full_docs.upsert = AsyncMock()
    lightrag.text_chunks = MagicMock()
    lightrag.text_chunks.upsert = AsyncMock()
    lightrag.chunks_vdb = MagicMock()
    lightrag.chunks_vdb.upsert = AsyncMock()
    lightrag.chunks_vdb.embedding_func = MagicMock()
    return lightrag


# ---------------------------------------------------------------------------
# TestUnifiedEngineInit
# ---------------------------------------------------------------------------


class TestUnifiedEngineInit:
    """Verify sub-components are created during __init__."""

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    def test_subcomponents_created(
        self,
        mock_renderer_cls: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_extractor_cls: MagicMock,
        mock_retriever_cls: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()
        visual_chunks = MagicMock()
        vision_func = MagicMock()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=visual_chunks,
            config=config,
            vision_model_func=vision_func,
        )

        mock_renderer_cls.assert_called_once_with(dpi=250)
        mock_embedder_cls.assert_called_once_with(
            model="test-model",
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            dim=1024,
        )
        mock_extractor_cls.assert_called_once_with(
            lightrag=lightrag,
            entity_types=["Person", "Organization"],
            vision_model_func=vision_func,
        )
        mock_retriever_cls.assert_called_once()

        assert engine.renderer is mock_renderer_cls.return_value
        assert engine.embedder is mock_embedder_cls.return_value
        assert engine.extractor is mock_extractor_cls.return_value
        assert engine.retriever is mock_retriever_cls.return_value

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    def test_stores_lightrag_and_config(
        self,
        _renderer: MagicMock,
        _embedder: MagicMock,
        _extractor: MagicMock,
        _retriever: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()
        visual_chunks = MagicMock()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=visual_chunks,
            config=config,
        )

        assert engine.lightrag is lightrag
        assert engine.visual_chunks is visual_chunks
        assert engine.config is config
        assert engine.vision_model_func is None


# ---------------------------------------------------------------------------
# TestUpsertWithVisualVectors
# ---------------------------------------------------------------------------


class TestUpsertWithVisualVectors:
    """Test _upsert_with_visual_vectors embedding swap logic."""

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_upsert_swaps_embedding_and_restores(
        self,
        _renderer: MagicMock,
        _embedder: MagicMock,
        _extractor: MagicMock,
        _retriever: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()
        original_func = lightrag.chunks_vdb.embedding_func

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )

        chunks_data = {
            "chunk-001": {"content": "page one text", "full_doc_id": "doc-1"},
            "chunk-002": {"content": "page two text", "full_doc_id": "doc-1"},
        }
        vectors = np.random.rand(2, 1024).astype(np.float32)

        await engine._upsert_with_visual_vectors(chunks_data, vectors)

        lightrag.chunks_vdb.upsert.assert_awaited_once_with(chunks_data)
        # Original embedding func should be restored
        assert lightrag.chunks_vdb.embedding_func is original_func

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_upsert_restores_on_error(
        self,
        _renderer: MagicMock,
        _embedder: MagicMock,
        _extractor: MagicMock,
        _retriever: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()
        original_func = lightrag.chunks_vdb.embedding_func
        lightrag.chunks_vdb.upsert = AsyncMock(side_effect=RuntimeError("boom"))

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )

        chunks_data = {"chunk-001": {"content": "text", "full_doc_id": "d1"}}
        vectors = np.random.rand(1, 1024).astype(np.float32)

        with pytest.raises(RuntimeError, match="boom"):
            await engine._upsert_with_visual_vectors(chunks_data, vectors)

        # Embedding func restored despite error
        assert lightrag.chunks_vdb.embedding_func is original_func

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_upsert_empty_data_skips(
        self,
        _renderer: MagicMock,
        _embedder: MagicMock,
        _extractor: MagicMock,
        _retriever: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )

        await engine._upsert_with_visual_vectors({}, np.array([]))

        lightrag.chunks_vdb.upsert.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestAingest
# ---------------------------------------------------------------------------


class TestAingest:
    """Test aingest() pipeline end-to-end with mocked sub-components."""

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_aingest_two_pages(
        self,
        _renderer_cls: MagicMock,
        _embedder_cls: MagicMock,
        _extractor_cls: MagicMock,
        _retriever_cls: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()
        visual_chunks = MagicMock()
        visual_chunks.upsert = AsyncMock()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=visual_chunks,
            config=config,
        )

        # Mock renderer: returns 2 pages
        img_0 = Image.new("RGB", (100, 100), "white")
        img_1 = Image.new("RGB", (100, 100), "blue")
        render_result = RenderResult(
            pages=[(0, img_0), (1, img_1)],
            metadata={
                "title": "Test Doc",
                "author": "Tester",
                "creation_date": "2025-01-01",
                "page_count": 2,
                "original_format": "pdf",
            },
        )
        engine.renderer.render_file = AsyncMock(return_value=render_result)

        # Mock embedder: returns (2, 1024) vectors
        vectors = np.zeros((2, 1024), dtype=np.float32)
        engine.embedder.embed_pages = AsyncMock(return_value=vectors)

        # Mock extractor: returns page_infos
        page_infos = [
            {
                "chunk_id": "chunk-aaa",
                "page_index": 0,
                "content": "Page one description",
            },
            {
                "chunk_id": "chunk-bbb",
                "page_index": 1,
                "content": "Page two description",
            },
        ]
        engine.extractor.extract_from_pages = AsyncMock(return_value=page_infos)

        # Mock _upsert_with_visual_vectors
        engine._upsert_with_visual_vectors = AsyncMock()

        result = await engine.aingest("/fake/doc.pdf", doc_id="doc-test")

        # Verify return value
        assert result["doc_id"] == "doc-test"
        assert result["page_count"] == 2
        assert result["file_path"] == "/fake/doc.pdf"

        # Verify full_docs.upsert called
        lightrag.full_docs.upsert.assert_awaited_once()
        upsert_arg = lightrag.full_docs.upsert.call_args[0][0]
        assert "doc-test" in upsert_arg

        # Verify text_chunks.upsert called with correct data
        lightrag.text_chunks.upsert.assert_awaited_once()
        tc_arg = lightrag.text_chunks.upsert.call_args[0][0]
        assert "chunk-aaa" in tc_arg
        assert "chunk-bbb" in tc_arg
        assert tc_arg["chunk-aaa"]["source_type"] == "unified_represent"
        assert tc_arg["chunk-bbb"]["chunk_order_index"] == 1

        # Verify visual_chunks.upsert called with image data
        visual_chunks.upsert.assert_awaited_once()
        vc_arg = visual_chunks.upsert.call_args[0][0]
        assert "chunk-aaa" in vc_arg
        assert "image_data" in vc_arg["chunk-aaa"]
        assert vc_arg["chunk-aaa"]["doc_id"] == "doc-test"
        assert vc_arg["chunk-aaa"]["doc_title"] == "Test Doc"

        # Verify _upsert_with_visual_vectors called
        engine._upsert_with_visual_vectors.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestAingestEmptyFile
# ---------------------------------------------------------------------------


class TestAingestEmptyFile:
    """Test aingest raises ValueError when renderer returns 0 pages."""

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_empty_render_raises(
        self,
        _renderer_cls: MagicMock,
        _embedder_cls: MagicMock,
        _extractor_cls: MagicMock,
        _retriever_cls: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )

        # Empty render result
        engine.renderer.render_file = AsyncMock(return_value=RenderResult(pages=[], metadata={}))

        with pytest.raises(ValueError, match="No pages rendered"):
            await engine.aingest("/fake/empty.pdf")


# ---------------------------------------------------------------------------
# TestAretrieve
# ---------------------------------------------------------------------------


class TestAretrieve:
    """Test aretrieve delegates to retriever.retrieve."""

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_delegates_to_retriever(
        self,
        _renderer_cls: MagicMock,
        _embedder_cls: MagicMock,
        _extractor_cls: MagicMock,
        _retriever_cls: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )

        expected = {"chunks": [], "kg_context": ""}
        engine.retriever.retrieve = AsyncMock(return_value=expected)

        result = await engine.aretrieve("test query")

        engine.retriever.retrieve.assert_awaited_once_with(
            query="test query",
            mode="mix",
            top_k=60,
            chunk_top_k=10,
        )
        assert result is expected

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_custom_params_override_defaults(
        self,
        _renderer_cls: MagicMock,
        _embedder_cls: MagicMock,
        _extractor_cls: MagicMock,
        _retriever_cls: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )

        engine.retriever.retrieve = AsyncMock(return_value={})

        await engine.aretrieve("q", mode="local", top_k=5, chunk_top_k=3)

        engine.retriever.retrieve.assert_awaited_once_with(
            query="q",
            mode="local",
            top_k=5,
            chunk_top_k=3,
        )


# ---------------------------------------------------------------------------
# TestAanswer
# ---------------------------------------------------------------------------


class TestAanswer:
    """Test aanswer delegates to retriever.answer."""

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_delegates_to_retriever_answer(
        self,
        _renderer_cls: MagicMock,
        _embedder_cls: MagicMock,
        _extractor_cls: MagicMock,
        _retriever_cls: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )

        expected = {"answer": "test answer", "sources": []}
        engine.retriever.answer = AsyncMock(return_value=expected)

        result = await engine.aanswer("what is X?")

        engine.retriever.answer.assert_awaited_once_with(
            query="what is X?",
            mode="mix",
            top_k=60,
            chunk_top_k=10,
        )
        assert result is expected

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_custom_params_override_defaults(
        self,
        _renderer_cls: MagicMock,
        _embedder_cls: MagicMock,
        _extractor_cls: MagicMock,
        _retriever_cls: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )

        engine.retriever.answer = AsyncMock(return_value={})

        await engine.aanswer("q", mode="global", top_k=20, chunk_top_k=5)

        engine.retriever.answer.assert_awaited_once_with(
            query="q",
            mode="global",
            top_k=20,
            chunk_top_k=5,
        )
