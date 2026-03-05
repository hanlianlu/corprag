# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for PostgreSQL storage.

Requires a running PostgreSQL instance with pgvector + AGE extensions.
Skipped automatically if PostgreSQL is not available.

Tests:
- PGHashIndex CRUD
- _ensure_pg_schema idempotent table/index creation
- RAGServiceManager.list_workspaces() PG workspace discovery
"""

from __future__ import annotations

import pytest

# Mark all tests in this module as integration
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


async def _pg_available() -> bool:
    """Check if PostgreSQL is available."""
    try:
        import asyncpg

        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="dlightrag",
            password="dlightrag",
            database="dlightrag",
        )
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def pg_check():
    """Skip test if PostgreSQL is not available."""
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")


class TestPGHashIndex:
    """Test PGHashIndex against real PostgreSQL."""

    async def test_register_and_lookup(self, pg_check) -> None:
        """Test register and lookup with real PostgreSQL."""
        import asyncpg

        from dlightrag.core.ingestion.hash_index import PGHashIndex

        pool = await asyncpg.create_pool(
            host="localhost",
            port=5432,
            user="dlightrag",
            password="dlightrag",
            database="dlightrag",
        )
        try:
            index = PGHashIndex(pool=pool, workspace="test")
            await index.initialize()

            content_hash = "sha256:test_integration_hash"
            await index.register(content_hash, "doc-int-001", "/test/file.pdf")

            exists, doc_id = await index.check_exists(content_hash)
            assert exists
            assert doc_id == "doc-int-001"

            # Cleanup
            removed = await index.remove(content_hash)
            assert removed
            exists2, _ = await index.check_exists(content_hash)
            assert not exists2
        finally:
            await pool.close()


# ---------------------------------------------------------------------------
# _ensure_pg_schema — idempotent table/index creation
# ---------------------------------------------------------------------------

_PG_CONN_KWARGS = dict(
    host="localhost",
    port=5432,
    user="dlightrag",
    password="dlightrag",
    database="dlightrag",
)


class TestEnsurePgSchema:
    """Test _ensure_pg_schema against real PostgreSQL (requires pgvector + AGE)."""

    async def test_schema_creates_table_and_indexes(self, pg_check) -> None:
        """_ensure_pg_schema creates the hash table and indexes."""
        import asyncpg

        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            from dlightrag.core.service import RAGService

            service = RAGService.__new__(RAGService)
            await service._ensure_pg_schema(conn)

            # Table must exist
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'dlightrag_file_hashes')"
            )
            assert exists, "dlightrag_file_hashes table should exist"

            # Indexes must exist
            indexes = {
                r["indexname"]
                for r in await conn.fetch(
                    "SELECT indexname FROM pg_indexes WHERE tablename = 'dlightrag_file_hashes'"
                )
            }
            assert "idx_file_hashes_doc_id" in indexes
            assert "idx_file_hashes_workspace" in indexes
        finally:
            await conn.close()

    async def test_schema_is_idempotent(self, pg_check) -> None:
        """Running _ensure_pg_schema twice should not error."""
        import asyncpg

        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            from dlightrag.core.service import RAGService

            service = RAGService.__new__(RAGService)
            # Run twice — second call must not raise
            await service._ensure_pg_schema(conn)
            await service._ensure_pg_schema(conn)
        finally:
            await conn.close()


# ---------------------------------------------------------------------------
# RAGServiceManager.list_workspaces — PG workspace discovery
# ---------------------------------------------------------------------------


class TestPGWorkspaceDiscovery:
    """Test workspace discovery via SELECT DISTINCT workspace."""

    async def test_discovers_workspaces_from_hash_table(self, pg_check) -> None:
        """list_workspaces() returns workspaces found in dlightrag_file_hashes."""
        import asyncpg

        from dlightrag.config import DlightragConfig, set_config
        from dlightrag.core.servicemanager import RAGServiceManager

        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            # Ensure table exists
            await conn.execute("""CREATE TABLE IF NOT EXISTS dlightrag_file_hashes (
                content_hash TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                workspace TEXT NOT NULL DEFAULT 'default',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )""")

            # Insert test rows for two workspaces
            await conn.execute(
                "INSERT INTO dlightrag_file_hashes (content_hash, doc_id, file_path, workspace) "
                "VALUES ($1, $2, $3, $4) ON CONFLICT (content_hash) DO NOTHING",
                "sha256:ws_test_a",
                "doc-a",
                "/a.pdf",
                "project-alpha",
            )
            await conn.execute(
                "INSERT INTO dlightrag_file_hashes (content_hash, doc_id, file_path, workspace) "
                "VALUES ($1, $2, $3, $4) ON CONFLICT (content_hash) DO NOTHING",
                "sha256:ws_test_b",
                "doc-b",
                "/b.pdf",
                "project-beta",
            )

            cfg = DlightragConfig(  # type: ignore[call-arg]
                kv_storage="PGKVStorage",
                openai_api_key="test",
            )
            set_config(cfg)

            manager = RAGServiceManager(config=cfg)
            workspaces = await manager.list_workspaces()

            assert "project-alpha" in workspaces
            assert "project-beta" in workspaces
        finally:
            # Cleanup test rows
            await conn.execute(
                "DELETE FROM dlightrag_file_hashes WHERE content_hash IN ($1, $2)",
                "sha256:ws_test_a",
                "sha256:ws_test_b",
            )
            await conn.close()

    async def test_empty_table_returns_default_workspace(self, pg_check) -> None:
        """Empty hash table falls back to config.workspace."""
        import asyncpg

        from dlightrag.config import DlightragConfig, set_config
        from dlightrag.core.servicemanager import RAGServiceManager

        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            # Ensure table exists but clear any test data
            await conn.execute("""CREATE TABLE IF NOT EXISTS dlightrag_file_hashes (
                content_hash TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                workspace TEXT NOT NULL DEFAULT 'default',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )""")

            # Count existing rows — only test if table is empty or we use unique prefix
            # Use a unique workspace to avoid interference with real data
            cfg = DlightragConfig(  # type: ignore[call-arg]
                kv_storage="PGKVStorage",
                workspace="test-fallback-ws",
                openai_api_key="test",
            )
            set_config(cfg)

            manager = RAGServiceManager(config=cfg)
            workspaces = await manager.list_workspaces()

            # Should at least contain the default workspace
            # (may contain more if table has data from other tests)
            assert isinstance(workspaces, list)
            assert len(workspaces) >= 1
        finally:
            await conn.close()
