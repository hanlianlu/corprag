# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGService facade."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corprag.config import CorpragConfig
from corprag.service import RAGService, _detect_mineru_backend

# ---------------------------------------------------------------------------
# TestRAGServiceInit
# ---------------------------------------------------------------------------


class TestRAGServiceInit:
    """Test RAGService construction and configuration."""

    def test_init_stores_config(self, test_config: CorpragConfig) -> None:
        service = RAGService(config=test_config)
        assert service.config is test_config
        assert not service._initialized
        assert service.ingestion is None
        assert service.rag_text is None

    def test_ensure_initialized_raises(self, test_config: CorpragConfig) -> None:
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            service._ensure_initialized()


# ---------------------------------------------------------------------------
# TestDetectMineruBackend
# ---------------------------------------------------------------------------


class TestDetectMineruBackend:
    """Test _detect_mineru_backend hardware detection."""

    def test_manual_override(self) -> None:
        result = _detect_mineru_backend("custom-engine")
        assert result == "custom-engine"

    def test_cuda_available(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _detect_mineru_backend()
        assert result == "hybrid-auto-engine"

    def test_apple_silicon(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        with (
            patch.dict(sys.modules, {"torch": mock_torch}),
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            result = _detect_mineru_backend()
        assert result == "pipeline"

    def test_no_torch(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):
            # When module is None, import raises ImportError
            result = _detect_mineru_backend()
        assert result == "pipeline"

    def test_no_gpu(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with (
            patch.dict(sys.modules, {"torch": mock_torch}),
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
        ):
            result = _detect_mineru_backend()
        assert result == "pipeline"


# ---------------------------------------------------------------------------
# TestRAGServiceAingest
# ---------------------------------------------------------------------------


class TestRAGServiceAingest:
    """Test ingestion routing logic."""

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
        service.ingestion.aingest_from_snowflake = AsyncMock(
            return_value=MagicMock(model_dump=MagicMock(return_value={"status": "success"}))
        )
        service.rag_text = MagicMock()
        service.rag_vision = MagicMock()
        return service

    async def test_aingest_local_routing(self, test_config: CorpragConfig) -> None:
        service = self._make_initialized_service(test_config)
        await service.aingest(source_type="local", path="/tmp/file.pdf")
        service.ingestion.aingest_from_local.assert_awaited_once()

    async def test_aingest_azure_routing(self, test_config: CorpragConfig) -> None:
        service = self._make_initialized_service(test_config)
        mock_source = AsyncMock()
        await service.aingest(
            source_type="azure_blob",
            source=mock_source,
            container_name="c",
            blob_path="b/file.pdf",
        )
        service.ingestion.aingest_from_azure_blob.assert_awaited_once()

    async def test_aingest_snowflake_routing(self, test_config: CorpragConfig) -> None:
        service = self._make_initialized_service(test_config)
        await service.aingest(source_type="snowflake", query="SELECT 1")
        service.ingestion.aingest_from_snowflake.assert_awaited_once()

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

    async def test_close_finalizes_storages(self, test_config: CorpragConfig) -> None:
        service = RAGService(config=test_config)
        service._initialized = True
        mock_ingestion = MagicMock()
        mock_ingestion.rag = MagicMock()
        mock_ingestion.rag.finalize_storages = AsyncMock()
        service.ingestion = mock_ingestion

        mock_rag = MagicMock()
        mock_rag.finalize_storages = AsyncMock()
        service.rag_text = mock_rag
        service.rag_vision = mock_rag

        await service.close()

        mock_ingestion.rag.finalize_storages.assert_awaited_once()
        mock_rag.finalize_storages.assert_awaited()

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
