# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for retrieval engine: path extraction, URL generation, result augmentation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from dlightrag.core.retrieval.engine import (
    RetrievalEngine,
    RetrievalResult,
    _extract_rag_relative,
    _to_download_url,
    augment_retrieval_result,
    build_sources_and_media_from_contexts,
)

# ---------------------------------------------------------------------------
# TestExtractRagRelative
# ---------------------------------------------------------------------------


class TestExtractRagRelative:
    """Test _extract_rag_relative path extraction."""

    def test_with_working_dir(self) -> None:
        result = _extract_rag_relative(
            "/abs/dlightrag_storage/sources/local/file.pdf",
            working_dir="/abs/dlightrag_storage",
        )
        assert result == "sources/local/file.pdf"

    def test_with_working_dir_trailing_slash(self) -> None:
        result = _extract_rag_relative(
            "/abs/dlightrag_storage/sources/local/file.pdf",
            working_dir="/abs/dlightrag_storage/",
        )
        assert result == "sources/local/file.pdf"

    def test_fallback_sources_marker(self) -> None:
        result = _extract_rag_relative(
            "/random/prefix/sources/local/file.pdf",
            working_dir=None,
        )
        assert result == "sources/local/file.pdf"

    def test_fallback_artifacts_marker(self) -> None:
        result = _extract_rag_relative(
            "/random/prefix/artifacts/local/report/page1.png",
            working_dir=None,
        )
        assert result == "artifacts/local/report/page1.png"

    def test_no_match(self) -> None:
        result = _extract_rag_relative(
            "/random/path/file.pdf",
            working_dir="/other/dir",
        )
        assert result is None


# ---------------------------------------------------------------------------
# TestToDownloadUrl
# ---------------------------------------------------------------------------


class TestToDownloadUrl:
    """Test _to_download_url URL generation."""

    def test_with_url_transformer(self) -> None:
        def transformer(p: str) -> str:
            return f"https://cdn.example.com/{p}"

        result = _to_download_url("/some/path.pdf", url_transformer=transformer)
        assert result == "https://cdn.example.com//some/path.pdf"

    def test_file_scheme_stripped(self) -> None:
        result = _to_download_url(
            "file:///abs/dlightrag_storage/sources/local/file.pdf",
            working_dir="/abs/dlightrag_storage",
        )
        assert result == "file://sources/local/file.pdf"

    def test_azure_passthrough(self) -> None:
        result = _to_download_url("azure://container/blob.pdf")
        assert result == "azure://container/blob.pdf"

    def test_relative_extraction_no_transformer(self) -> None:
        result = _to_download_url(
            "/abs/dlightrag_storage/sources/local/file.pdf",
            working_dir="/abs/dlightrag_storage",
        )
        assert result == "file://sources/local/file.pdf"

    def test_fallback_full_path(self) -> None:
        result = _to_download_url("/random/path.pdf")
        assert result == "file:///random/path.pdf"


# ---------------------------------------------------------------------------
# TestBuildSourcesAndMediaFromContexts
# ---------------------------------------------------------------------------


class TestBuildSourcesAndMediaFromContexts:
    """Test source/media extraction from chunk contexts."""

    def test_single_source_extracted(self) -> None:
        contexts = [
            {
                "chunk_id": "c1",
                "file_path": "/storage/sources/local/report.pdf",
                "reference_id": "ref-001",
                "content": "Some important text content here",
            },
        ]
        sources, media = build_sources_and_media_from_contexts(contexts)
        assert len(sources) == 1
        assert sources[0]["id"] == "ref-001"
        assert sources[0]["title"] == "report.pdf"
        assert sources[0]["path"] == "/storage/sources/local/report.pdf"
        assert "c1" in sources[0]["chunk_ids"]

    def test_dedup_by_reference_id(self) -> None:
        contexts = [
            {
                "chunk_id": "c1",
                "file_path": "/storage/report.pdf",
                "reference_id": "ref-001",
                "content": "chunk 1",
            },
            {
                "chunk_id": "c2",
                "file_path": "/storage/report.pdf",
                "reference_id": "ref-001",
                "content": "chunk 2",
            },
        ]
        sources, media = build_sources_and_media_from_contexts(contexts)
        assert len(sources) == 1
        assert set(sources[0]["chunk_ids"]) == {"c1", "c2"}

    def test_image_extracted_from_content(self) -> None:
        contexts = [
            {
                "chunk_id": "c1",
                "file_path": "/storage/report.pdf",
                "reference_id": "ref-001",
                "content": "Image Path: /storage/artifacts/img.png\nCaption: A chart showing growth",
            },
        ]
        sources, media = build_sources_and_media_from_contexts(contexts)
        assert len(media) == 1
        assert media[0]["type"] == "image"
        assert media[0]["path"] == "/storage/artifacts/img.png"
        assert media[0]["caption"] == "A chart showing growth"

    def test_no_chunk_id_skipped(self) -> None:
        contexts = [
            {
                "file_path": "/storage/report.pdf",
                "reference_id": "ref-001",
                "content": "text",
            },
        ]
        sources, media = build_sources_and_media_from_contexts(contexts)
        assert len(sources) == 0

    def test_no_file_path_skipped(self) -> None:
        contexts = [
            {
                "chunk_id": "c1",
                "reference_id": "ref-001",
                "content": "text",
            },
        ]
        sources, media = build_sources_and_media_from_contexts(contexts)
        assert len(sources) == 0


# ---------------------------------------------------------------------------
# TestAugmentRetrievalResult
# ---------------------------------------------------------------------------


class TestAugmentRetrievalResult:
    """Test result augmentation with sources/media."""

    async def test_attaches_sources(self) -> None:
        result = RetrievalResult(
            answer=None,
            contexts={
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "file_path": "/storage/sources/local/doc.pdf",
                        "reference_id": "ref-001",
                        "content": "Text content",
                    },
                ],
            },
            raw={},
        )
        augmented = await augment_retrieval_result(result)
        assert "sources" in augmented.raw
        assert len(augmented.raw["sources"]) == 1

    async def test_page_idx_injected(self) -> None:
        lightrag = MagicMock()
        lightrag.text_chunks = MagicMock()
        lightrag.text_chunks.get_by_ids = AsyncMock(
            return_value=[{"page_idx": 2, "content": "text"}]
        )

        result = RetrievalResult(
            answer=None,
            contexts={
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "file_path": "/storage/doc.pdf",
                        "reference_id": "ref-001",
                        "content": "text",
                    },
                ],
            },
            raw={},
        )
        augmented = await augment_retrieval_result(result, lightrag=lightrag)
        chunk = augmented.contexts["chunks"][0]
        assert chunk["page_idx"] == 3  # 0-based 2 -> 1-based 3

    async def test_no_contexts_no_crash(self) -> None:
        result = RetrievalResult(answer=None, contexts={}, raw={})
        augmented = await augment_retrieval_result(result)
        assert augmented is result

    async def test_no_lightrag_no_page_idx(self) -> None:
        result = RetrievalResult(
            answer=None,
            contexts={
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "file_path": "/storage/doc.pdf",
                        "reference_id": "ref-001",
                        "content": "text",
                    },
                ],
            },
            raw={},
        )
        augmented = await augment_retrieval_result(result, lightrag=None)
        assert "page_idx" not in augmented.contexts["chunks"][0]


# ---------------------------------------------------------------------------
# TestRetrievalEngineAretrieve
# ---------------------------------------------------------------------------


class TestRetrievalEngineAretrieve:
    """Test data-only retrieval via composition."""

    def _make_engine(self, config=None) -> tuple[RetrievalEngine, MagicMock]:
        from dlightrag.config import DlightragConfig

        cfg = config or DlightragConfig(openai_api_key="test")  # type: ignore[call-arg]
        mock_rag = MagicMock()
        mock_rag.lightrag = MagicMock()
        mock_rag.lightrag.aquery_data = AsyncMock(
            return_value={"data": {"chunks": [{"id": "c1"}], "entities": [], "relationships": []}}
        )
        mock_rag._process_multimodal_query_content = AsyncMock(return_value="enhanced query")
        engine = RetrievalEngine(rag=mock_rag, config=cfg)
        return engine, mock_rag

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aretrieve_calls_lightrag_aquery_data(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, mock_rag = self._make_engine()
        result = await engine.aretrieve("test query")
        mock_rag.lightrag.aquery_data.assert_awaited_once()
        assert isinstance(result, RetrievalResult)

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aretrieve_with_multimodal_enhances_query(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, mock_rag = self._make_engine()
        await engine.aretrieve("query", multimodal_content=[{"type": "image"}])
        mock_rag._process_multimodal_query_content.assert_awaited_once()

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aretrieve_without_multimodal_no_enhancement(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, mock_rag = self._make_engine()
        await engine.aretrieve("query")
        mock_rag._process_multimodal_query_content.assert_not_awaited()

    async def test_aretrieve_no_lightrag_returns_empty(self) -> None:
        from dlightrag.config import DlightragConfig

        cfg = DlightragConfig(openai_api_key="test")  # type: ignore[call-arg]
        mock_rag = MagicMock()
        mock_rag.lightrag = None
        engine = RetrievalEngine(rag=mock_rag, config=cfg)
        result = await engine.aretrieve("query")
        assert result.answer is None
        assert result.contexts == {}


# ---------------------------------------------------------------------------
# TestRetrievalEngineAanswer
# ---------------------------------------------------------------------------


class TestRetrievalEngineAanswer:
    """Test LLM answer retrieval."""

    def _make_engine(self) -> tuple[RetrievalEngine, MagicMock]:
        from dlightrag.config import DlightragConfig

        cfg = DlightragConfig(openai_api_key="test")  # type: ignore[call-arg]
        mock_rag = MagicMock()
        mock_rag.lightrag = MagicMock()
        mock_rag.lightrag.aquery_llm = AsyncMock(
            return_value={
                "llm_response": {"content": "The answer is 42"},
                "data": {"chunks": [{"id": "c1"}], "entities": [], "relationships": []},
            }
        )
        mock_rag._process_multimodal_query_content = AsyncMock(return_value="enhanced")
        engine = RetrievalEngine(rag=mock_rag, config=cfg)
        return engine, mock_rag

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aanswer_returns_answer(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, _ = self._make_engine()
        result = await engine.aanswer("query")
        assert result.answer == "The answer is 42"

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aanswer_calls_aquery_llm(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, mock_rag = self._make_engine()
        await engine.aanswer("query")
        mock_rag.lightrag.aquery_llm.assert_awaited_once()
