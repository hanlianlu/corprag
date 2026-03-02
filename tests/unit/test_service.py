# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGService facade."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corprag.config import CorpragConfig
from corprag.service import RAGService, _detect_mineru_backend

# ---------------------------------------------------------------------------
# TestRAGServiceInit
# ---------------------------------------------------------------------------


class TestRAGServiceInit:
    """Test RAGService construction and configuration."""

    def test_ensure_initialized_raises(self, test_config: CorpragConfig) -> None:
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            service._ensure_initialized()

    def test_mineru_backend_manual_override(self) -> None:
        result = _detect_mineru_backend("custom-engine")
        assert result == "custom-engine"


# ---------------------------------------------------------------------------
# TestRAGServiceAingest
# ---------------------------------------------------------------------------


class TestRAGServiceAingest:
    """Test ingestion logic — replace defaults, azure lifecycle."""

    def _make_initialized_service(self, config: CorpragConfig) -> RAGService:
        service = RAGService(config=config)
        service._initialized = True
        service.ingestion = MagicMock()
        service.ingestion.aingest_from_local = AsyncMock(
            return_value=MagicMock(model_dump=MagicMock(return_value={"status": "success"}))
        )
        service.ingestion.aingest_from_azure_blob = AsyncMock(
            return_value=MagicMock(model_dump=MagicMock(return_value={"status": "success"}))
        )
        service.rag_text = MagicMock()
        service.rag_vision = MagicMock()
        return service

    async def test_aingest_not_initialized_raises(self, test_config: CorpragConfig) -> None:
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.aingest(source_type="local", path="/tmp/f.pdf")

    async def test_aingest_replace_default_from_config(self, test_config: CorpragConfig) -> None:
        test_config.ingestion_replace_default = True
        service = self._make_initialized_service(test_config)
        await service.aingest(source_type="local", path="/tmp/file.pdf")
        call_kwargs = service.ingestion.aingest_from_local.call_args
        assert call_kwargs.kwargs["replace"] is True

    async def test_aingest_replace_explicit_overrides_config(
        self, test_config: CorpragConfig
    ) -> None:
        test_config.ingestion_replace_default = True
        service = self._make_initialized_service(test_config)
        await service.aingest(source_type="local", path="/tmp/file.pdf", replace=False)
        call_kwargs = service.ingestion.aingest_from_local.call_args
        assert call_kwargs.kwargs["replace"] is False

    # -- Azure blob lifecycle --

    async def test_aingest_azure_defaults_prefix_when_neither_set(
        self, test_config: CorpragConfig
    ) -> None:
        """When neither blob_path nor prefix provided, prefix defaults to '' (entire container)."""
        service = self._make_initialized_service(test_config)
        mock_source = AsyncMock()
        await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        call_kwargs = service.ingestion.aingest_from_azure_blob.call_args.kwargs
        assert call_kwargs["prefix"] == ""
        assert call_kwargs.get("blob_path") is None

    async def test_aingest_azure_calls_aclose(self, test_config: CorpragConfig) -> None:
        """source.aclose() is called after successful ingestion."""
        service = self._make_initialized_service(test_config)
        mock_source = AsyncMock()
        await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        mock_source.aclose.assert_awaited_once()

    async def test_aingest_azure_calls_aclose_on_error(self, test_config: CorpragConfig) -> None:
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

    async def test_rerank_sorts_by_relevance(self, test_config: CorpragConfig) -> None:
        service = RAGService(config=test_config)
        service._initialized = True

        chunks = [
            {"content": "low relevance"},
            {"content": "high relevance"},
        ]

        async def mock_rerank(**kwargs):
            return [{"index": 1}, {"index": 0}]

        with patch("corprag.service.get_rerank_func", return_value=mock_rerank):
            result = await service._rerank_chunks(chunks, "query")

        assert result[0]["content"] == "high relevance"
        assert result[1]["content"] == "low relevance"

    async def test_rerank_failure_returns_original(self, test_config: CorpragConfig) -> None:
        service = RAGService(config=test_config)
        service._initialized = True

        chunks = [{"content": "a"}, {"content": "b"}]

        async def mock_rerank(**kwargs):
            raise RuntimeError("LLM error")

        with patch("corprag.service.get_rerank_func", return_value=mock_rerank):
            result = await service._rerank_chunks(chunks, "query")

        assert result == chunks

    async def test_rerank_empty_chunks(self, test_config: CorpragConfig) -> None:
        service = RAGService(config=test_config)
        service._initialized = True
        result = await service._rerank_chunks([], "query")
        assert result == []


# ---------------------------------------------------------------------------
# TestRAGServiceClose
# ---------------------------------------------------------------------------


class TestRAGServiceClose:
    """Test cleanup logic."""

    async def test_close_handles_errors(self, test_config: CorpragConfig) -> None:
        service = RAGService(config=test_config)
        service._initialized = True
        mock_ingestion = MagicMock()
        mock_ingestion.rag = MagicMock()
        mock_ingestion.rag.finalize_storages = AsyncMock(side_effect=RuntimeError("cleanup failed"))
        service.ingestion = mock_ingestion
        service.rag_text = None
        service.rag_vision = None

        # Should not raise
        await service.close()
