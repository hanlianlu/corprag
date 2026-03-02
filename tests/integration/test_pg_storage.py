# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for PostgreSQL storage.

Requires a running PostgreSQL instance with pgvector + AGE extensions.
Skipped automatically if PostgreSQL is not available.
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
            user="corprag",
            password="corprag",
            database="corprag",
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

        from corprag.ingestion.hash_index import PGHashIndex

        pool = await asyncpg.create_pool(
            host="localhost",
            port=5432,
            user="corprag",
            password="corprag",
            database="corprag",
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
