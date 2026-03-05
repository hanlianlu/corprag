# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGService facade (core/service.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.config import DlightragConfig
from dlightrag.core.service import RAGService

# ---------------------------------------------------------------------------
# TestRAGServiceAingest
# ---------------------------------------------------------------------------


class TestRAGServiceAingest:
    """Test ingestion logic -- replace defaults, azure lifecycle."""

    def _make_initialized_service(self, config: DlightragConfig) -> RAGService:
        service = RAGService(config=config)
        service._initialized = True
        service.ingestion = MagicMock()
        service.ingestion.aingest_from_local = AsyncMock(
            return_value=MagicMock(model_dump=MagicMock(return_value={"status": "success"}))
        )
        service.ingestion.aingest_from_azure_blob = AsyncMock(
            return_value=MagicMock(model_dump=MagicMock(return_value={"status": "success"}))
        )
        service.rag = MagicMock()
        service.retrieval = MagicMock()
        return service

    async def test_aingest_not_initialized_raises(self, test_config: DlightragConfig) -> None:
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.aingest(source_type="local", path="/tmp/f.pdf")

    async def test_aingest_replace_default_from_config(self, test_config: DlightragConfig) -> None:
        test_config.ingestion_replace_default = True
        service = self._make_initialized_service(test_config)
        await service.aingest(source_type="local", path="/tmp/file.pdf")
        call_kwargs = service.ingestion.aingest_from_local.call_args
        assert call_kwargs.kwargs["replace"] is True

    async def test_aingest_replace_explicit_overrides_config(
        self, test_config: DlightragConfig
    ) -> None:
        test_config.ingestion_replace_default = True
        service = self._make_initialized_service(test_config)
        await service.aingest(source_type="local", path="/tmp/file.pdf", replace=False)
        call_kwargs = service.ingestion.aingest_from_local.call_args
        assert call_kwargs.kwargs["replace"] is False

    # -- Azure blob lifecycle --

    async def test_aingest_azure_defaults_prefix_when_neither_set(
        self, test_config: DlightragConfig
    ) -> None:
        """When neither blob_path nor prefix provided, prefix defaults to '' (entire container)."""
        service = self._make_initialized_service(test_config)
        mock_source = AsyncMock()
        await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        call_kwargs = service.ingestion.aingest_from_azure_blob.call_args.kwargs
        assert call_kwargs["prefix"] == ""
        assert call_kwargs.get("blob_path") is None

    async def test_aingest_azure_calls_aclose(self, test_config: DlightragConfig) -> None:
        """source.aclose() is called after successful ingestion."""
        service = self._make_initialized_service(test_config)
        mock_source = AsyncMock()
        await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        mock_source.aclose.assert_awaited_once()

    async def test_aingest_azure_calls_aclose_on_error(self, test_config: DlightragConfig) -> None:
        """source.aclose() is called even when ingestion raises."""
        service = self._make_initialized_service(test_config)
        service.ingestion.aingest_from_azure_blob = AsyncMock(
            side_effect=RuntimeError("ingestion failed")
        )
        mock_source = AsyncMock()
        with pytest.raises(RuntimeError, match="ingestion failed"):
            await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        mock_source.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestRAGServiceRerank
# ---------------------------------------------------------------------------


class TestRAGServiceRerank:
    """Test _rerank_chunks with mocked LLM."""

    async def test_rerank_sorts_by_relevance(self, test_config: DlightragConfig) -> None:
        service = RAGService(config=test_config)
        service._initialized = True

        chunks = [
            {"content": "low relevance"},
            {"content": "high relevance"},
        ]

        async def mock_rerank(**kwargs):
            return [{"index": 1}, {"index": 0}]

        with patch("dlightrag.core.service.get_rerank_func", return_value=mock_rerank):
            result = await service._rerank_chunks(chunks, "query")

        assert result[0]["content"] == "high relevance"
        assert result[1]["content"] == "low relevance"

    async def test_rerank_failure_returns_original(self, test_config: DlightragConfig) -> None:
        service = RAGService(config=test_config)
        service._initialized = True

        chunks = [{"content": "a"}, {"content": "b"}]

        async def mock_rerank(**kwargs):
            raise RuntimeError("LLM error")

        with patch("dlightrag.core.service.get_rerank_func", return_value=mock_rerank):
            result = await service._rerank_chunks(chunks, "query")

        assert result == chunks


# ---------------------------------------------------------------------------
# TestRAGServiceClose
# ---------------------------------------------------------------------------


class TestRAGServiceClose:
    """Test cleanup logic."""

    async def test_close_handles_errors(self, test_config: DlightragConfig) -> None:
        service = RAGService(config=test_config)
        service._initialized = True
        service.rag = MagicMock()
        service.rag.finalize_storages = AsyncMock(side_effect=RuntimeError("cleanup failed"))
        service.ingestion = None
        service.retrieval = None

        # Should not raise
        await service.close()


# ---------------------------------------------------------------------------
# TestRAGServiceRetrieve
# ---------------------------------------------------------------------------


class TestRAGServiceRetrieve:
    """Test aretrieve and aanswer delegation to RetrievalEngine."""

    def _make_retrieval_service(self, config: DlightragConfig) -> RAGService:
        service = RAGService(config=config)
        service._initialized = True
        service.retrieval = MagicMock()
        service.retrieval.aretrieve = AsyncMock(return_value=MagicMock())
        service.retrieval.aanswer = AsyncMock(return_value=MagicMock())
        service.rag = MagicMock()
        service.ingestion = MagicMock()
        return service

    async def test_aretrieve_delegates_to_retrieval(self, test_config):
        service = self._make_retrieval_service(test_config)
        await service.aretrieve("test query")
        service.retrieval.aretrieve.assert_awaited_once()

    async def test_aretrieve_passes_multimodal_content(self, test_config):
        service = self._make_retrieval_service(test_config)
        mc = [{"type": "image"}]
        await service.aretrieve("test query", multimodal_content=mc)
        call_kwargs = service.retrieval.aretrieve.call_args.kwargs
        assert call_kwargs["multimodal_content"] == mc

    async def test_aretrieve_not_initialized_raises(self, test_config):
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.aretrieve("query")


# ---------------------------------------------------------------------------
# TestConversationHistoryTruncation
# ---------------------------------------------------------------------------


class TestConversationHistoryTruncation:
    """Test aanswer delegates to RetrievalEngine (truncation now lives there)."""

    def _make_retrieval_service(self, config: DlightragConfig) -> RAGService:
        service = RAGService(config=config)
        service._initialized = True
        service.retrieval = MagicMock()
        service.retrieval.aretrieve = AsyncMock(return_value=MagicMock())
        service.retrieval.aanswer = AsyncMock(return_value=MagicMock())
        service.rag = MagicMock()
        service.ingestion = MagicMock()
        return service

    async def test_aanswer_delegates_to_retrieval(self, test_config):
        service = self._make_retrieval_service(test_config)
        await service.aanswer("query", conversation_history=[{"role": "user", "content": "hi"}])
        service.retrieval.aanswer.assert_awaited_once()
        call_kwargs = service.retrieval.aanswer.call_args.kwargs
        assert "conversation_history" in call_kwargs

    async def test_aanswer_none_history_delegates(self, test_config):
        """None history is passed through to retrieval engine."""
        service = self._make_retrieval_service(test_config)
        await service.aanswer("query", conversation_history=None)
        service.retrieval.aanswer.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestRAGServiceFileManagement
# ---------------------------------------------------------------------------


class TestRAGServiceFileManagement:
    """Test alist_ingested_files and adelete_files delegation."""

    async def test_alist_not_initialized_raises(self, test_config):
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.alist_ingested_files()

    async def test_adelete_not_initialized_raises(self, test_config):
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.adelete_files(filenames=["a.pdf"])

    async def test_alist_delegates_to_ingestion(self, test_config):
        service = RAGService(config=test_config)
        service._initialized = True
        service.ingestion = MagicMock()
        service.ingestion.alist_ingested_files = AsyncMock(return_value=[{"doc_id": "d1"}])
        result = await service.alist_ingested_files()
        assert result == [{"doc_id": "d1"}]
        service.ingestion.alist_ingested_files.assert_awaited_once()

    async def test_adelete_delegates_to_ingestion(self, test_config):
        service = RAGService(config=test_config)
        service._initialized = True
        service.ingestion = MagicMock()
        service.ingestion.adelete_files = AsyncMock(return_value=[{"status": "deleted"}])
        result = await service.adelete_files(filenames=["a.pdf"])
        assert result == [{"status": "deleted"}]
        call_kwargs = service.ingestion.adelete_files.call_args.kwargs
        assert call_kwargs["filenames"] == ["a.pdf"]


# ---------------------------------------------------------------------------
# TestBuildVectorDbKwargs
# ---------------------------------------------------------------------------


class TestBuildVectorDbKwargs:
    """Test _build_vector_db_kwargs passthrough."""

    def test_default_has_cosine_threshold(self, test_config: DlightragConfig) -> None:
        result = RAGService._build_vector_db_kwargs(test_config)
        assert result == {"cosine_better_than_threshold": 0.3}

    def test_passthrough_merges_kwargs(self, test_config: DlightragConfig) -> None:
        test_config.vector_db_kwargs = {
            "index_type": "HNSW_SQ",
            "sq_type": "SQ8",
            "hnsw_m": 32,
        }
        result = RAGService._build_vector_db_kwargs(test_config)
        assert result["cosine_better_than_threshold"] == 0.3
        assert result["index_type"] == "HNSW_SQ"
        assert result["sq_type"] == "SQ8"
        assert result["hnsw_m"] == 32

    def test_passthrough_overrides_default(self, test_config: DlightragConfig) -> None:
        test_config.vector_db_kwargs = {"cosine_better_than_threshold": 0.5}
        result = RAGService._build_vector_db_kwargs(test_config)
        assert result["cosine_better_than_threshold"] == 0.5
