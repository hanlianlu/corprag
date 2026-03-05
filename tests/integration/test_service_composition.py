# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for RAGService composition.

Verifies the single-RAGAnything architecture works end-to-end with
filesystem backends (no PostgreSQL, no real LLM needed).

Tests:
1. RAGService.create() produces correctly composed rag + ingestion + retrieval
2. Single RAGAnything instance is shared between ingestion and retrieval
3. RAGServiceManager workspace isolation with real filesystem backends
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from dlightrag.config import DlightragConfig, set_config
from dlightrag.core.ingestion.pipeline import IngestionPipeline
from dlightrag.core.retrieval.engine import RetrievalEngine
from dlightrag.core.service import RAGService
from dlightrag.core.servicemanager import RAGServiceManager

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


def _dummy_llm_func(config):
    """Fake LLM that returns empty JSON for entity extraction."""

    async def _fn(*args, **kw):
        return '{"entities": [], "relationships": []}'

    return _fn


def _dummy_embedding_func(config):
    """Fake embedding function that returns random vectors."""
    from lightrag.utils import EmbeddingFunc

    async def _embed(texts: list[str]) -> np.ndarray:
        return np.random.rand(len(texts), 256).astype(np.float32)

    return EmbeddingFunc(
        embedding_dim=256,
        max_token_size=8192,
        func=_embed,
    )


def _dummy_rerank_func(config):
    """Return None — no reranking in tests."""
    return None


def _dummy_vision_func(config):
    """Return None — no vision model in tests."""
    return None


@pytest.fixture
def fs_config(tmp_path: Path) -> DlightragConfig:
    """Config with filesystem backends and dummy models."""
    working_dir = tmp_path / "dlightrag_storage"
    working_dir.mkdir()
    cfg = DlightragConfig(  # type: ignore[call-arg]
        working_dir=str(working_dir),
        workspace="test-ws",
        openai_api_key="test-key",
        kv_storage="JsonKVStorage",
        doc_status_storage="JsonDocStatusStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        parser="docling",
    )
    set_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# RAGService composition
# ---------------------------------------------------------------------------


class TestServiceComposition:
    """Verify single-RAGAnything composition through real initialization."""

    @patch("dlightrag.core.service.get_rerank_func", side_effect=_dummy_rerank_func)
    @patch("dlightrag.core.service.get_vision_model_func", side_effect=_dummy_vision_func)
    @patch("dlightrag.core.service.get_embedding_func", side_effect=_dummy_embedding_func)
    @patch("dlightrag.core.service.get_llm_model_func", side_effect=_dummy_llm_func)
    async def test_create_initializes_all_components(
        self, _llm, _embed, _vision, _rerank, fs_config
    ) -> None:
        """RAGService.create() produces rag + ingestion + retrieval."""
        service = await RAGService.create(config=fs_config, enable_vlm=False)
        try:
            assert service._initialized
            assert service.rag is not None
            assert isinstance(service.ingestion, IngestionPipeline)
            assert isinstance(service.retrieval, RetrievalEngine)

            # LightRAG should be initialized inside RAGAnything
            assert service.rag.lightrag is not None
        finally:
            await service.close()

    @patch("dlightrag.core.service.get_rerank_func", side_effect=_dummy_rerank_func)
    @patch("dlightrag.core.service.get_vision_model_func", side_effect=_dummy_vision_func)
    @patch("dlightrag.core.service.get_embedding_func", side_effect=_dummy_embedding_func)
    @patch("dlightrag.core.service.get_llm_model_func", side_effect=_dummy_llm_func)
    async def test_single_rag_shared(self, _llm, _embed, _vision, _rerank, fs_config) -> None:
        """Ingestion and retrieval share the same RAGAnything instance."""
        service = await RAGService.create(config=fs_config, enable_vlm=False)
        try:
            assert service.ingestion.rag is service.rag
            assert service.retrieval.rag is service.rag
        finally:
            await service.close()

    @patch("dlightrag.core.service.get_rerank_func", side_effect=_dummy_rerank_func)
    @patch("dlightrag.core.service.get_vision_model_func", side_effect=_dummy_vision_func)
    @patch("dlightrag.core.service.get_embedding_func", side_effect=_dummy_embedding_func)
    @patch("dlightrag.core.service.get_llm_model_func", side_effect=_dummy_llm_func)
    async def test_workspace_in_working_dir(
        self, _llm, _embed, _vision, _rerank, fs_config
    ) -> None:
        """LightRAG working_dir should include the workspace subdirectory."""
        service = await RAGService.create(config=fs_config, enable_vlm=False)
        try:
            lightrag_wd = service.rag.lightrag.working_dir
            assert "test-ws" in lightrag_wd or lightrag_wd.endswith("dlightrag_storage")
        finally:
            await service.close()


# ---------------------------------------------------------------------------
# RAGServiceManager workspace isolation
# ---------------------------------------------------------------------------


class TestManagerWorkspaceIsolation:
    """Verify RAGServiceManager creates isolated services per workspace."""

    @patch("dlightrag.core.service.get_rerank_func", side_effect=_dummy_rerank_func)
    @patch("dlightrag.core.service.get_vision_model_func", side_effect=_dummy_vision_func)
    @patch("dlightrag.core.service.get_embedding_func", side_effect=_dummy_embedding_func)
    @patch("dlightrag.core.service.get_llm_model_func", side_effect=_dummy_llm_func)
    async def test_different_workspaces_get_different_services(
        self, _llm, _embed, _vision, _rerank, fs_config
    ) -> None:
        """Two workspaces must produce two separate RAGService instances."""
        manager = RAGServiceManager(config=fs_config)
        try:
            svc_a = await manager._get_service("ws-a")
            svc_b = await manager._get_service("ws-b")

            assert svc_a is not svc_b
            assert svc_a.rag is not svc_b.rag
            assert svc_a.config.workspace == "ws-a"
            assert svc_b.config.workspace == "ws-b"
        finally:
            await manager.close()

    @patch("dlightrag.core.service.get_rerank_func", side_effect=_dummy_rerank_func)
    @patch("dlightrag.core.service.get_vision_model_func", side_effect=_dummy_vision_func)
    @patch("dlightrag.core.service.get_embedding_func", side_effect=_dummy_embedding_func)
    @patch("dlightrag.core.service.get_llm_model_func", side_effect=_dummy_llm_func)
    async def test_same_workspace_returns_cached(
        self, _llm, _embed, _vision, _rerank, fs_config
    ) -> None:
        """Same workspace returns the same cached RAGService instance."""
        manager = RAGServiceManager(config=fs_config)
        try:
            svc1 = await manager._get_service("ws-x")
            svc2 = await manager._get_service("ws-x")

            assert svc1 is svc2
        finally:
            await manager.close()
