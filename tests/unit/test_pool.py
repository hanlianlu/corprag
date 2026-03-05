# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for shared RAG service pool: singleton, async lock, exponential backoff."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from dlightrag.pool import (
    RAGServiceUnavailableError,
    close_shared_rag_service,
    close_workspace_services,
    get_rag_error_info,
    get_shared_rag_service,
    get_workspace_service,
    is_rag_service_initialized,
    list_available_workspaces,
    reset_shared_rag_service,
    reset_workspace_pool,
)


@pytest.fixture(autouse=True)
def _reset_pool():
    """Reset pool state before and after each test."""
    reset_shared_rag_service()
    yield
    reset_shared_rag_service()


# ---------------------------------------------------------------------------
# TestGetSharedRagService
# ---------------------------------------------------------------------------


class TestGetSharedRagService:
    """Test singleton with async lock and exponential backoff."""

    async def test_fast_path_returns_cached(self) -> None:
        import dlightrag.pool as pool

        mock_service = AsyncMock()
        pool._shared_rag_service = mock_service
        pool._rag_ready = True

        result = await get_shared_rag_service()
        assert result is mock_service

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_initializes_on_first_call(self, mock_create) -> None:
        mock_service = AsyncMock()
        mock_create.return_value = mock_service

        result = await get_shared_rag_service()

        assert result is mock_service
        mock_create.assert_awaited_once()
        assert is_rag_service_initialized()

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_concurrent_calls_only_init_once(self, mock_create) -> None:
        mock_service = AsyncMock()

        async def slow_create(**kwargs):
            await asyncio.sleep(0.05)
            return mock_service

        mock_create.side_effect = slow_create

        results = await asyncio.gather(
            get_shared_rag_service(),
            get_shared_rag_service(),
            get_shared_rag_service(),
        )

        assert mock_create.await_count == 1
        assert all(r is mock_service for r in results)

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_init_failure_sets_error_state(self, mock_create) -> None:
        mock_create.side_effect = RuntimeError("DB connection failed")

        with pytest.raises(RAGServiceUnavailableError):
            await get_shared_rag_service()

        assert not is_rag_service_initialized()
        error_info = get_rag_error_info()
        assert "RuntimeError" in error_info["last_error"]

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_exponential_backoff_after_failure(self, mock_create) -> None:
        mock_create.side_effect = RuntimeError("fail")

        # First call fails
        with pytest.raises(RAGServiceUnavailableError):
            await get_shared_rag_service()

        # Immediate retry should fail without calling create (within backoff)
        with pytest.raises(RAGServiceUnavailableError):
            await get_shared_rag_service()

        # create should only have been called once
        assert mock_create.await_count == 1

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_backoff_doubles(self, mock_create) -> None:
        import dlightrag.pool as pool

        mock_create.side_effect = RuntimeError("fail")

        # First failure: _rag_retry_after should double from 30 -> 60
        with pytest.raises(RAGServiceUnavailableError):
            await get_shared_rag_service()
        first_retry = pool._rag_retry_after

        # Simulate time passing beyond retry window
        pool._rag_last_error_ts = time.time() - first_retry - 1

        # Second failure: should double again
        with pytest.raises(RAGServiceUnavailableError):
            await get_shared_rag_service()
        second_retry = pool._rag_retry_after

        assert second_retry > first_retry
        # Should cap at 300
        assert second_retry <= 300.0

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_retry_after_backoff_expires(self, mock_create) -> None:
        import dlightrag.pool as pool

        # First call fails
        mock_create.side_effect = RuntimeError("initial fail")
        with pytest.raises(RAGServiceUnavailableError):
            await get_shared_rag_service()

        # Simulate backoff expiry
        pool._rag_last_error_ts = time.time() - 60

        # Now succeed
        mock_service = AsyncMock()
        mock_create.side_effect = None
        mock_create.return_value = mock_service

        result = await get_shared_rag_service()
        assert result is mock_service

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_success_resets_error_state(self, mock_create) -> None:
        import dlightrag.pool as pool

        # First call fails
        mock_create.side_effect = RuntimeError("fail")
        with pytest.raises(RAGServiceUnavailableError):
            await get_shared_rag_service()

        # Wait and succeed
        pool._rag_last_error_ts = time.time() - 60
        mock_service = AsyncMock()
        mock_create.side_effect = None
        mock_create.return_value = mock_service

        await get_shared_rag_service()

        assert pool._rag_last_error is None
        assert pool._rag_retry_after == 30.0


# ---------------------------------------------------------------------------
# TestCloseSharedRagService
# ---------------------------------------------------------------------------


class TestCloseSharedRagService:
    """Test cleanup."""

    async def test_closes_and_resets(self) -> None:
        import dlightrag.pool as pool

        mock_service = AsyncMock()
        pool._shared_rag_service = mock_service
        pool._rag_ready = True

        await close_shared_rag_service()

        mock_service.close.assert_awaited_once()
        assert pool._shared_rag_service is None
        assert not pool._rag_ready

    async def test_close_handles_exception(self) -> None:
        import dlightrag.pool as pool

        mock_service = AsyncMock()
        mock_service.close.side_effect = RuntimeError("cleanup error")
        pool._shared_rag_service = mock_service
        pool._rag_ready = True

        # Should not raise
        await close_shared_rag_service()

        assert pool._shared_rag_service is None
        assert not pool._rag_ready


# ---------------------------------------------------------------------------
# TestWorkspacePool
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_ws_pool():
    """Reset workspace pool state and set test config before each test."""
    from dlightrag.config import DlightragConfig, set_config

    set_config(DlightragConfig(openai_api_key="test"))  # type: ignore[call-arg]
    reset_workspace_pool()
    yield
    reset_workspace_pool()


class TestWorkspacePool:
    """Test workspace-keyed RAGService pool."""

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_creates_service_for_workspace(self, mock_create) -> None:
        mock_service = AsyncMock()
        mock_create.return_value = mock_service

        result = await get_workspace_service("project-a")

        assert result is mock_service
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["config"].workspace == "project-a"

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_caches_service_per_workspace(self, mock_create) -> None:
        mock_service = AsyncMock()
        mock_create.return_value = mock_service

        svc1 = await get_workspace_service("ws-1")
        svc2 = await get_workspace_service("ws-1")

        assert svc1 is svc2
        assert mock_create.await_count == 1

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_different_workspaces_get_different_services(self, mock_create) -> None:
        mock_create.side_effect = [AsyncMock(), AsyncMock()]

        svc1 = await get_workspace_service("ws-a")
        svc2 = await get_workspace_service("ws-b")

        assert svc1 is not svc2
        assert mock_create.await_count == 2

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_concurrent_creates_only_init_once(self, mock_create) -> None:
        mock_service = AsyncMock()

        async def slow_create(**kwargs):
            await asyncio.sleep(0.05)
            return mock_service

        mock_create.side_effect = slow_create

        results = await asyncio.gather(
            get_workspace_service("ws-x"),
            get_workspace_service("ws-x"),
            get_workspace_service("ws-x"),
        )

        assert mock_create.await_count == 1
        assert all(r is mock_service for r in results)

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_close_workspace_services(self, mock_create) -> None:
        import dlightrag.pool as pool

        svc_a = AsyncMock()
        svc_b = AsyncMock()
        pool._workspace_services = {"a": svc_a, "b": svc_b}

        await close_workspace_services()

        svc_a.close.assert_awaited_once()
        svc_b.close.assert_awaited_once()
        assert pool._workspace_services == {}

    async def test_list_workspaces_non_pg_returns_default(self) -> None:
        from dlightrag.config import DlightragConfig, set_config

        cfg = DlightragConfig(  # type: ignore[call-arg]
            kv_storage="JsonKVStorage",
            workspace="myws",
            openai_api_key="test",
        )
        set_config(cfg)

        result = await list_available_workspaces()
        assert result == ["myws"]

    async def test_list_workspaces_filesystem_discovery(self, tmp_path) -> None:
        """Filesystem backends discover workspaces by scanning working_dir subdirectories."""
        from dlightrag.config import DlightragConfig, set_config

        working_dir = tmp_path / "dlightrag_storage"
        working_dir.mkdir()

        # Create workspace directories with LightRAG data files
        ws_a = working_dir / "project-a"
        ws_a.mkdir()
        (ws_a / "kv_store_full_docs.json").write_text("{}")

        ws_b = working_dir / "project-b"
        ws_b.mkdir()
        (ws_b / "vdb_entities.json").write_text("{}")

        # Directory without LightRAG data should NOT be discovered
        ws_empty = working_dir / "empty-dir"
        ws_empty.mkdir()

        cfg = DlightragConfig(  # type: ignore[call-arg]
            working_dir=str(working_dir),
            kv_storage="JsonKVStorage",
            doc_status_storage="JsonDocStatusStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
            openai_api_key="test",
        )
        set_config(cfg)

        result = await list_available_workspaces()
        assert "project-a" in result
        assert "project-b" in result
        assert "empty-dir" not in result
