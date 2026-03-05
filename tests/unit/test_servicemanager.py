# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGServiceManager: workspace pool, routing, health tracking."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.config import DlightragConfig, set_config
from dlightrag.core.servicemanager import RAGServiceManager, RAGServiceUnavailableError


@pytest.fixture()
def test_cfg(tmp_path) -> DlightragConfig:
    cfg = DlightragConfig(
        openai_api_key="test",
        working_dir=str(tmp_path / "dlightrag_storage"),
        kv_storage="JsonKVStorage",
        doc_status_storage="JsonDocStatusStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
    )  # type: ignore[call-arg]
    set_config(cfg)
    return cfg


class TestGetService:
    """Test workspace-keyed RAGService creation and caching."""

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_creates_service_for_workspace(self, mock_create, test_cfg) -> None:
        mock_create.return_value = AsyncMock()
        manager = RAGServiceManager(config=test_cfg)
        svc = await manager._get_service("project-a")
        assert svc is mock_create.return_value
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["config"].workspace == "project-a"

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_caches_per_workspace(self, mock_create, test_cfg) -> None:
        mock_create.return_value = AsyncMock()
        manager = RAGServiceManager(config=test_cfg)
        svc1 = await manager._get_service("ws-1")
        svc2 = await manager._get_service("ws-1")
        assert svc1 is svc2
        assert mock_create.await_count == 1

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_different_workspaces_different_services(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = [AsyncMock(), AsyncMock()]
        manager = RAGServiceManager(config=test_cfg)
        svc1 = await manager._get_service("ws-a")
        svc2 = await manager._get_service("ws-b")
        assert svc1 is not svc2
        assert mock_create.await_count == 2

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_concurrent_creates_once(self, mock_create, test_cfg) -> None:
        mock_service = AsyncMock()

        async def slow_create(**kwargs):
            await asyncio.sleep(0.05)
            return mock_service

        mock_create.side_effect = slow_create
        manager = RAGServiceManager(config=test_cfg)
        results = await asyncio.gather(
            manager._get_service("ws-x"),
            manager._get_service("ws-x"),
            manager._get_service("ws-x"),
        )
        assert mock_create.await_count == 1
        assert all(r is mock_service for r in results)


class TestBackoff:
    """Test exponential backoff on service creation failure."""

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_failure_sets_error_state(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = RuntimeError("DB down")
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws-a")
        assert not manager.is_ready()
        error_info = manager.get_error_info()
        assert "RuntimeError" in error_info["last_error"]

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_backoff_blocks_retry(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = RuntimeError("fail")
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws-a")
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws-a")
        assert mock_create.await_count == 1

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_retry_succeeds_after_backoff(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = RuntimeError("fail")
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws-a")
        manager._last_error_ts = time.time() - 60
        mock_create.side_effect = None
        mock_create.return_value = AsyncMock()
        svc = await manager._get_service("ws-a")
        assert svc is mock_create.return_value

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_success_resets_error_state(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = RuntimeError("fail")
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws-a")
        manager._last_error_ts = time.time() - 60
        mock_create.side_effect = None
        mock_create.return_value = AsyncMock()
        await manager._get_service("ws-a")
        assert manager._last_error is None
        assert manager._retry_after == 30.0


class TestRouting:
    """Test single-workspace vs federated routing."""

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aretrieve_single_workspace(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aretrieve.return_value = MagicMock()
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        await manager.aretrieve("query", workspace="ws-a")
        mock_svc.aretrieve.assert_awaited_once()

    @patch("dlightrag.core.servicemanager.federated_retrieve", new_callable=AsyncMock)
    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aretrieve_multi_workspace_federates(
        self, mock_create, mock_fed, test_cfg
    ) -> None:
        mock_fed.return_value = MagicMock()
        manager = RAGServiceManager(config=test_cfg)
        await manager.aretrieve("query", workspaces=["ws-a", "ws-b"])
        mock_fed.assert_awaited_once()

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aretrieve_default_workspace(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aretrieve.return_value = MagicMock()
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        await manager.aretrieve("query")
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["config"].workspace == test_cfg.workspace

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aanswer_single_workspace(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aanswer.return_value = MagicMock()
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        await manager.aanswer("query", workspace="ws-a")
        mock_svc.aanswer.assert_awaited_once()

    @patch("dlightrag.core.servicemanager.federated_answer", new_callable=AsyncMock)
    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aanswer_multi_workspace_federates(self, mock_create, mock_fed, test_cfg) -> None:
        mock_fed.return_value = MagicMock()
        manager = RAGServiceManager(config=test_cfg)
        await manager.aanswer("query", workspaces=["ws-a", "ws-b"])
        mock_fed.assert_awaited_once()


class TestDelegation:
    """Test write-operation delegation."""

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aingest_delegates(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aingest.return_value = {"status": "ok"}
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        result = await manager.aingest("ws-a", source_type="local", path="/tmp/f.pdf")
        mock_svc.aingest.assert_awaited_once()
        assert result == {"status": "ok"}

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_list_ingested_files_delegates(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.alist_ingested_files.return_value = [{"doc": "d1"}]
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        result = await manager.list_ingested_files("ws-a")
        assert result == [{"doc": "d1"}]

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_delete_files_delegates(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.adelete_files.return_value = [{"status": "deleted"}]
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        result = await manager.delete_files("ws-a", filenames=["a.pdf"])
        assert result == [{"status": "deleted"}]


class TestClose:
    """Test cleanup."""

    async def test_close_all_services(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        svc_a = AsyncMock()
        svc_b = AsyncMock()
        manager._services = {"a": svc_a, "b": svc_b}
        manager._ready = True
        await manager.close()
        svc_a.close.assert_awaited_once()
        svc_b.close.assert_awaited_once()
        assert manager._services == {}
        assert not manager._ready


class TestWorkspaceDiscovery:
    """Test list_workspaces with filesystem backend."""

    async def test_filesystem_discovery(self, tmp_path, test_cfg) -> None:
        working_dir = tmp_path / "dlightrag_storage"
        working_dir.mkdir()
        ws_a = working_dir / "project-a"
        ws_a.mkdir()
        (ws_a / "kv_store_full_docs.json").write_text("{}")
        ws_b = working_dir / "project-b"
        ws_b.mkdir()
        (ws_b / "vdb_entities.json").write_text("{}")
        ws_empty = working_dir / "empty-dir"
        ws_empty.mkdir()

        cfg = DlightragConfig(
            working_dir=str(working_dir),
            kv_storage="JsonKVStorage",
            doc_status_storage="JsonDocStatusStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
            openai_api_key="test",
        )  # type: ignore[call-arg]
        manager = RAGServiceManager(config=cfg)
        result = await manager.list_workspaces()
        assert "project-a" in result
        assert "project-b" in result
        assert "empty-dir" not in result

    async def test_fallback_returns_default(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        result = await manager.list_workspaces()
        assert test_cfg.workspace in result
