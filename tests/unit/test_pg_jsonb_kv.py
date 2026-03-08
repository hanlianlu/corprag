# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PGJsonbKVStorage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_storage(namespace: str = "visual_chunks", workspace: str = "test"):
    """Create a PGJsonbKVStorage with mocked pool."""
    from dlightrag.storage.pg_jsonb_kv import PGJsonbKVStorage

    mock_embedding = MagicMock()
    storage = PGJsonbKVStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={},
        embedding_func=mock_embedding,
    )
    # Inject mock pool directly (skip ClientManager)
    storage._pool = MagicMock()
    return storage


class TestUpsertAndGet:
    """Test upsert writes data and get retrieves it."""

    @pytest.mark.asyncio
    async def test_upsert_builds_correct_sql(self) -> None:
        """Upsert calls executemany with INSERT ... ON CONFLICT DO UPDATE."""
        storage = _make_storage()
        mock_conn = AsyncMock()
        storage._pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        storage._pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        await storage.upsert({"chunk1": {"img_b64": "abc", "page": 0}})

        mock_conn.executemany.assert_called_once()
        sql = mock_conn.executemany.call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "dlightrag_kv_store" in sql
        args = mock_conn.executemany.call_args[0][1]
        assert len(args) == 1
        assert args[0][0] == "test"  # workspace
        assert args[0][1] == "visual_chunks"  # namespace
        assert args[0][2] == "chunk1"  # id

    @pytest.mark.asyncio
    async def test_get_by_id_returns_data(self) -> None:
        """get_by_id returns data dict for existing key."""
        storage = _make_storage()
        mock_conn = AsyncMock()
        # asyncpg returns JSONB as already-parsed dict
        mock_conn.fetchrow.return_value = {"data": {"img_b64": "abc"}}
        storage._pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        storage._pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await storage.get_by_id("chunk1")

        assert result == {"img_b64": "abc"}
        # workspace, namespace, id passed as params
        assert mock_conn.fetchrow.call_args[0][1] == "test"
        assert mock_conn.fetchrow.call_args[0][2] == "visual_chunks"

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_for_missing(self) -> None:
        """get_by_id returns None when key not found."""
        storage = _make_storage()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        storage._pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        storage._pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await storage.get_by_id("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_ids_returns_ordered_results(self) -> None:
        """get_by_ids returns results in input order, None for missing."""
        storage = _make_storage()
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {"id": "b", "data": {"page": 1}},
            {"id": "a", "data": {"page": 0}},
        ]
        storage._pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        storage._pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await storage.get_by_ids(["a", "b", "c"])

        assert len(result) == 3
        assert result[0] == {"page": 0}  # "a"
        assert result[1] == {"page": 1}  # "b"
        assert result[2] is None  # "c" not found


class TestDeleteAndFilter:
    """Test delete and filter_keys operations."""

    @pytest.mark.asyncio
    async def test_delete_executes_correct_sql(self) -> None:
        """delete calls DELETE with correct params."""
        storage = _make_storage()
        mock_conn = AsyncMock()
        storage._pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        storage._pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        await storage.delete(["chunk1", "chunk2"])

        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "DELETE" in sql

    @pytest.mark.asyncio
    async def test_filter_keys_returns_nonexistent(self) -> None:
        """filter_keys returns keys that don't exist in storage."""
        storage = _make_storage()
        mock_conn = AsyncMock()
        # Simulate that only "a" exists
        mock_conn.fetch.return_value = [{"id": "a"}]
        storage._pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        storage._pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await storage.filter_keys({"a", "b", "c"})
        assert result == {"b", "c"}


class TestInitialize:
    """Test table creation on initialize."""

    @pytest.mark.asyncio
    async def test_ensure_table_creates_table(self) -> None:
        """_ensure_table executes CREATE TABLE IF NOT EXISTS."""
        storage = _make_storage()
        mock_conn = AsyncMock()
        storage._pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        storage._pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        await storage._ensure_table()

        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS" in sql
        assert "dlightrag_kv_store" in sql
        assert "JSONB" in sql


class TestNamespaceIsolation:
    """Test that namespace isolates data correctly."""

    @pytest.mark.asyncio
    async def test_different_namespaces_use_different_params(self) -> None:
        """Two storages with different namespaces query with their own namespace param."""
        s1 = _make_storage(namespace="visual_chunks")
        s2 = _make_storage(namespace="other_data")

        mock_conn1 = AsyncMock()
        mock_conn1.fetchrow.return_value = None
        s1._pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn1)
        s1._pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_conn2 = AsyncMock()
        mock_conn2.fetchrow.return_value = None
        s2._pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn2)
        s2._pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        await s1.get_by_id("x")
        await s2.get_by_id("x")

        # namespace param differs
        assert mock_conn1.fetchrow.call_args[0][2] == "visual_chunks"
        assert mock_conn2.fetchrow.call_args[0][2] == "other_data"
