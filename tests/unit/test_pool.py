# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for shared RAG service pool: singleton, async lock, exponential backoff."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from corprag.pool import (
    RAGServiceUnavailableError,
    close_shared_rag_service,
    get_rag_error_info,
    get_shared_rag_service,
    is_rag_service_initialized,
    reset_shared_rag_service,
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
        import corprag.pool as pool

        mock_service = AsyncMock()
        pool._shared_rag_service = mock_service
        pool._rag_ready = True

        result = await get_shared_rag_service()
        assert result is mock_service

    @patch("corprag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_initializes_on_first_call(self, mock_create) -> None:
        mock_service = AsyncMock()
        mock_create.return_value = mock_service

        result = await get_shared_rag_service()

        assert result is mock_service
        mock_create.assert_awaited_once()
        assert is_rag_service_initialized()

    @patch("corprag.pool.RAGService.create", new_callable=AsyncMock)
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

    @patch("corprag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_init_failure_sets_error_state(self, mock_create) -> None:
        mock_create.side_effect = RuntimeError("DB connection failed")

        with pytest.raises(RAGServiceUnavailableError):
            await get_shared_rag_service()

        assert not is_rag_service_initialized()
        error_info = get_rag_error_info()
        assert "RuntimeError" in error_info["last_error"]

    @patch("corprag.pool.RAGService.create", new_callable=AsyncMock)
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

    @patch("corprag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_backoff_doubles(self, mock_create) -> None:
        import corprag.pool as pool

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

    @patch("corprag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_retry_after_backoff_expires(self, mock_create) -> None:
        import corprag.pool as pool

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

    @patch("corprag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_success_resets_error_state(self, mock_create) -> None:
        import corprag.pool as pool

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
        import corprag.pool as pool

        mock_service = AsyncMock()
        pool._shared_rag_service = mock_service
        pool._rag_ready = True

        await close_shared_rag_service()

        mock_service.close.assert_awaited_once()
        assert pool._shared_rag_service is None
        assert not pool._rag_ready

    async def test_close_handles_exception(self) -> None:
        import corprag.pool as pool

        mock_service = AsyncMock()
        mock_service.close.side_effect = RuntimeError("cleanup error")
        pool._shared_rag_service = mock_service
        pool._rag_ready = True

        # Should not raise
        await close_shared_rag_service()

        assert pool._shared_rag_service is None
        assert not pool._rag_ready


# ---------------------------------------------------------------------------
# TestPoolStateHelpers
# ---------------------------------------------------------------------------


class TestPoolStateHelpers:
    """Test state query functions."""

    def test_is_initialized_true(self) -> None:
        import corprag.pool as pool

        pool._shared_rag_service = AsyncMock()
        pool._rag_ready = True
        assert is_rag_service_initialized()

    def test_is_initialized_false_when_not_ready(self) -> None:
        import corprag.pool as pool

        pool._shared_rag_service = AsyncMock()
        pool._rag_ready = False
        assert not is_rag_service_initialized()

    def test_get_error_info(self) -> None:
        import corprag.pool as pool

        pool._rag_last_error = "RuntimeError: fail"
        pool._rag_last_error_ts = 1234567890.0
        pool._rag_retry_after = 60.0

        info = get_rag_error_info()
        assert info["last_error"] == "RuntimeError: fail"
        assert info["timestamp"] == 1234567890.0
        assert info["retry_after"] == 60.0

    def test_reset_shared_rag_service(self) -> None:
        import corprag.pool as pool

        pool._shared_rag_service = AsyncMock()
        pool._rag_ready = True
        pool._rag_last_error = "error"

        reset_shared_rag_service()

        assert pool._shared_rag_service is None
        assert not pool._rag_ready
        assert pool._rag_last_error is None
        assert pool._rag_retry_after == 30.0
