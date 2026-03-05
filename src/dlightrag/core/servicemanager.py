# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAGServiceManager — unified multi-workspace RAG coordinator.

Absorbs pool.py workspace management and federation routing into a single
entry point. All API/MCP consumers depend on this class only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig

from dlightrag.core.federation import federated_answer, federated_retrieve
from dlightrag.core.retrieval.engine import RetrievalResult
from dlightrag.core.service import RAGService

logger = logging.getLogger(__name__)

_MAX_RETRY_INTERVAL: float = 300.0


class RAGServiceUnavailableError(Exception):
    """Raised when the RAG service is not ready."""

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or "RAG service is not available"
        super().__init__(self.detail)


class RAGServiceManager:
    """Multi-workspace RAG coordinator.

    Manages a pool of RAGService instances (one per workspace).
    Routes read operations to single workspace or federation.
    """

    def __init__(self, config: DlightragConfig | None = None) -> None:
        from dlightrag.config import get_config

        self._config = config or get_config()
        self._services: dict[str, RAGService] = {}
        self._lock: asyncio.Lock | None = None

        # Health/error tracking
        self._ready: bool = False
        self._last_error: str | None = None
        self._last_error_ts: float | None = None
        self._retry_after: float = 30.0

    @classmethod
    async def create(cls, config: DlightragConfig | None = None) -> RAGServiceManager:
        """Async factory — creates manager and ensures default workspace is ready."""
        manager = cls(config=config)
        # Pre-initialize default workspace
        await manager._get_service(manager._config.workspace)
        manager._ready = True
        return manager

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _get_service(self, workspace: str) -> RAGService:
        """Get or create a RAGService for a specific workspace. Async-safe."""
        if workspace in self._services:
            return self._services[workspace]

        # Check backoff
        if self._last_error_ts is not None:
            if time.time() - self._last_error_ts < self._retry_after:
                raise RAGServiceUnavailableError(detail=self._last_error)

        lock = self._get_lock()
        async with lock:
            # Double-check
            if workspace in self._services:
                return self._services[workspace]

            if self._last_error_ts is not None:
                if time.time() - self._last_error_ts < self._retry_after:
                    raise RAGServiceUnavailableError(detail=self._last_error)

            try:
                ws_config = self._config.model_copy(update={"workspace": workspace})
                svc = await RAGService.create(config=ws_config)
                self._services[workspace] = svc

                # Reset error state on success
                self._last_error = None
                self._last_error_ts = None
                self._retry_after = 30.0

                logger.info("Created RAGService for workspace '%s'", workspace)
                return svc
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                self._last_error = error_msg
                self._last_error_ts = time.time()
                self._retry_after = min(self._retry_after * 2, _MAX_RETRY_INTERVAL)
                logger.error(
                    "RAGService creation failed for '%s': %s. Retry in %ss",
                    workspace,
                    error_msg,
                    self._retry_after,
                )
                raise RAGServiceUnavailableError(detail=error_msg) from e

    # --- Write operations (single workspace) ---

    async def aingest(
        self,
        workspace: str,
        source_type: Literal["local", "azure_blob", "snowflake"],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Ingest documents into a specific workspace."""
        svc = await self._get_service(workspace)
        return await svc.aingest(source_type=source_type, **kwargs)

    async def list_ingested_files(self, workspace: str) -> list[dict[str, Any]]:
        """List ingested files in a specific workspace."""
        svc = await self._get_service(workspace)
        return await svc.alist_ingested_files()

    async def delete_files(self, workspace: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Delete files from a specific workspace."""
        svc = await self._get_service(workspace)
        return await svc.adelete_files(**kwargs)

    # --- Read operations (single or federated) ---

    async def aretrieve(
        self,
        query: str,
        *,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve from one or more workspaces (federated if multiple)."""
        ws_list = workspaces or [workspace or self._config.workspace]
        if len(ws_list) == 1:
            svc = await self._get_service(ws_list[0])
            return await svc.aretrieve(query, **kwargs)
        return await federated_retrieve(query, ws_list, self._get_service, **kwargs)

    async def aanswer(
        self,
        query: str,
        *,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Answer from one or more workspaces (federated if multiple)."""
        ws_list = workspaces or [workspace or self._config.workspace]
        if len(ws_list) == 1:
            svc = await self._get_service(ws_list[0])
            return await svc.aanswer(query, **kwargs)
        return await federated_answer(query, ws_list, self._get_service, **kwargs)

    # --- Management ---

    async def list_workspaces(self) -> list[str]:
        """Discover available workspaces based on storage backend."""
        config = self._config

        if config.kv_storage.startswith("PG"):
            try:
                import asyncpg

                conn = await asyncpg.connect(
                    host=config.postgres_host,
                    port=config.postgres_port,
                    user=config.postgres_user,
                    password=config.postgres_password,
                    database=config.postgres_database,
                )
                try:
                    rows = await conn.fetch(
                        "SELECT DISTINCT workspace FROM dlightrag_file_hashes ORDER BY workspace"
                    )
                    workspaces = [row["workspace"] for row in rows]
                    return workspaces if workspaces else [config.workspace]
                finally:
                    await conn.close()
            except Exception as exc:
                logger.warning("Failed to list workspaces from PG: %s", exc)
                return [config.workspace]

        _fs_backends = {
            "JsonKVStorage",
            "JsonDocStatusStorage",
            "NanoVectorDBStorage",
            "NetworkXStorage",
            "FaissVectorDBStorage",
        }
        if config.kv_storage in _fs_backends or config.vector_storage in _fs_backends:
            return self._discover_filesystem_workspaces()

        cached = list(self._services.keys())
        if config.workspace not in cached:
            cached.append(config.workspace)
        return sorted(cached)

    def _discover_filesystem_workspaces(self) -> list[str]:
        """Scan working_dir for subdirectories containing LightRAG data files."""
        working_dir = self._config.working_dir_path
        if not working_dir.exists():
            return [self._config.workspace]
        workspaces = []
        for entry in working_dir.iterdir():
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if (
                any(entry.glob("kv_store_*.json"))
                or any(entry.glob("vdb_*.json"))
                or any(entry.glob("graph_*.graphml"))
                or any(entry.glob("file_content_hashes.json"))
            ):
                workspaces.append(entry.name)
        return sorted(workspaces) if workspaces else [self._config.workspace]

    async def close(self) -> None:
        """Close all managed RAGService instances."""
        for ws, svc in self._services.items():
            try:
                await svc.close()
            except Exception:
                logger.warning("Failed to close workspace service '%s'", ws, exc_info=True)
        self._services.clear()
        self._ready = False

    # --- Health ---

    def is_ready(self) -> bool:
        """Check if manager is ready (default workspace initialized)."""
        return self._ready

    def get_error_info(self) -> dict[str, str | float | None]:
        """Get error state for health checks."""
        return {
            "last_error": self._last_error,
            "timestamp": self._last_error_ts,
            "retry_after": self._retry_after,
        }


__all__ = [
    "RAGServiceManager",
    "RAGServiceUnavailableError",
]
