# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared RAG Service Pool - Per-process singleton with async lazy initialization.

Provides a shared RAGService instance per process, avoiding the overhead of
creating new instances per task. The service is stateless (connects to PG/storage),
so it can safely be shared across concurrent async tasks within one event loop.

Thread-safety note:
    This module uses ``asyncio.Lock`` and module-level globals. It is safe for
    **multi-process** deployments (e.g. ``gunicorn --workers N``) but is NOT
    thread-safe for **multi-threaded** deployments (``gunicorn --threads M``).
    Deploy with async workers (uvicorn / UvicornWorker) and scale via processes
    or Kubernetes horizontal pod autoscaling, not via threads.

Usage:
    from corprag.pool import get_shared_rag_service

    rag_service = await get_shared_rag_service()
    result = await rag_service.aretrieve(query, ...)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corprag.config import CorpragConfig

from corprag.service import RAGService

logger = logging.getLogger(__name__)

# Per-worker singleton
_shared_rag_service: RAGService | None = None
_rag_init_lock: asyncio.Lock | None = None

# RAG readiness state for health checks and auto-recovery
_rag_ready: bool = False
_rag_last_error: str | None = None
_rag_last_error_ts: float | None = None
_rag_retry_after: float = 30.0
_RAG_MAX_RETRY_INTERVAL: float = 300.0


class RAGServiceUnavailableError(Exception):
    """Raised when the RAG service is not ready."""

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or "RAG service is not available"
        super().__init__(self.detail)


def _get_init_lock() -> asyncio.Lock:
    """Get or create the initialization lock (must be created within event loop).

    # hllyu init lock
    """
    global _rag_init_lock
    if _rag_init_lock is None:
        _rag_init_lock = asyncio.Lock()
    return _rag_init_lock


async def get_shared_rag_service(
    config: CorpragConfig | None = None,
    enable_vlm: bool = True,
    cancel_checker: Callable[[], Awaitable[bool]] | None = None,
    url_transformer: Callable[[str], str] | None = None,
) -> RAGService:
    """Get or initialize the shared RAG service for this worker.

    Async-task-safe via asyncio.Lock (NOT thread-safe). First caller
    initializes; subsequent calls return the cached instance immediately.

    If initialization previously failed, attempts reinitialize with
    exponential backoff (30s -> 60s -> 120s -> 300s max).

    Args:
        config: Optional CorpragConfig. If None, uses get_config() singleton.
        enable_vlm: Whether to enable vision model support.
        cancel_checker: Optional cancellation callback.
        url_transformer: Optional URL generation callback.

    Returns:
        Initialized RAGService instance shared across all tasks.
    """
    global _shared_rag_service, _rag_ready, _rag_last_error, _rag_last_error_ts, _rag_retry_after

    # Fast path: already initialized and ready
    if _shared_rag_service is not None and _rag_ready:
        return _shared_rag_service

    # Check if we should attempt reinitialization (exponential backoff)
    if _rag_last_error_ts is not None:
        time_since_error = time.time() - _rag_last_error_ts
        if time_since_error < _rag_retry_after:
            raise RAGServiceUnavailableError(detail=_rag_last_error)

    # Slow path: need to initialize (with lock to prevent concurrent init)
    lock = _get_init_lock()
    async with lock:
        # Double-check after acquiring lock
        if _shared_rag_service is not None and _rag_ready:
            return _shared_rag_service

        # Check backoff again after acquiring lock
        if _rag_last_error_ts is not None:
            time_since_error = time.time() - _rag_last_error_ts
            if time_since_error < _rag_retry_after:
                raise RAGServiceUnavailableError(detail=_rag_last_error)

        logger.info("Initializing shared RAG service...")

        try:
            _shared_rag_service = await RAGService.create(
                config=config,
                enable_vlm=enable_vlm,
                cancel_checker=cancel_checker,
                url_transformer=url_transformer,
            )

            # Success - mark as ready and reset retry state
            _rag_ready = True
            _rag_last_error = None
            _rag_last_error_ts = None
            _rag_retry_after = 30.0

            logger.info("Shared RAG service initialized successfully")
            return _shared_rag_service

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            _rag_ready = False
            _rag_last_error = error_msg
            _rag_last_error_ts = time.time()
            _rag_retry_after = min(_rag_retry_after * 2, _RAG_MAX_RETRY_INTERVAL)

            logger.error(
                f"RAG service initialization failed: {error_msg}. Next retry in {_rag_retry_after}s"
            )
            raise RAGServiceUnavailableError(detail=error_msg) from e


async def close_shared_rag_service() -> None:
    """Close the shared RAG service during shutdown."""
    global _shared_rag_service, _rag_ready

    if _shared_rag_service is not None:
        logger.info("Closing shared RAG service...")
        try:
            await _shared_rag_service.close()
        except Exception:
            logger.warning("Failed to close shared RAG service", exc_info=True)
        finally:
            _shared_rag_service = None
            _rag_ready = False


def is_rag_service_initialized() -> bool:
    """Check if RAG service has been initialized and is ready."""
    return _shared_rag_service is not None and _rag_ready


def get_rag_error_info() -> dict[str, str | float | None]:
    """Get RAG initialization error info for health checks."""
    return {
        "last_error": _rag_last_error,
        "timestamp": _rag_last_error_ts,
        "retry_after": _rag_retry_after,
    }


def reset_shared_rag_service() -> None:
    """Reset the shared service state. Useful for testing."""
    global _shared_rag_service, _rag_init_lock, _rag_ready
    global _rag_last_error, _rag_last_error_ts, _rag_retry_after
    _shared_rag_service = None
    _rag_init_lock = None
    _rag_ready = False
    _rag_last_error = None
    _rag_last_error_ts = None
    _rag_retry_after = 30.0


__all__ = [
    "RAGServiceUnavailableError",
    "close_shared_rag_service",
    "get_rag_error_info",
    "get_shared_rag_service",
    "is_rag_service_initialized",
    "reset_shared_rag_service",
]
