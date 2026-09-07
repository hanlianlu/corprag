# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PostgreSQL corpus maintenance behavior."""

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.adapters.postgres.corpus.corpus import PGCorpusMaintenanceStore
from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry


class _Tx:
    async def __aenter__(self) -> _Tx:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _Acquire:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    async def __aenter__(self) -> object:
        return self._conn

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _Pool:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


def _maintenance_store(config: MagicMock, conn: object) -> PGCorpusMaintenanceStore:
    return PGCorpusMaintenanceStore(
        config.pg_connection_kwargs(),
        workspace_registry=PGWorkspaceRegistry(pool=_Pool(conn)),
    )


class _Conn:
    def __init__(self) -> None:
        digest = hashlib.sha256(b"research").hexdigest()[:16]
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.fetchvals: list[tuple[str, tuple[object, ...]]] = []
        self.relations = [
            {"relname": f"p_metadata_w_{digest}"},
            {"relname": f"s_metadata_w_{digest}"},
            {"relname": "p_unrelated_w_0000000000000000"},
        ]
        self.closed = False

    def transaction(self) -> _Tx:
        return _Tx()

    async def execute(self, query: str, *args: object) -> str:
        self.executed.append((query, args))
        return "DELETE 1"

    async def fetch(self, query: str) -> list[dict[str, str]]:
        assert "pg_class" in query
        return self.relations

    async def fetchval(self, query: str, *args: object) -> bool:
        self.fetchvals.append((query, args))
        return True

    async def close(self) -> None:
        self.closed = True


@pytest.fixture()
def config() -> MagicMock:
    cfg = MagicMock()
    cfg.pg_connection_kwargs.return_value = {
        "host": "localhost",
        "port": 5432,
        "user": "dlightrag",
        "password": "test",
        "database": "dlightrag",
    }
    return cfg


async def test_workspace_exists_uses_the_operational_registry_point_lookup(
    monkeypatch, config
) -> None:
    conn = _Conn()
    connect = AsyncMock(side_effect=AssertionError("registry operations must not connect directly"))
    monkeypatch.setattr("dlightrag.adapters.postgres.corpus.corpus.asyncpg.connect", connect)

    store = _maintenance_store(config, conn)
    assert await store.workspace_exists("research") is True

    assert len(conn.fetchvals) == 1
    query, args = conn.fetchvals[0]
    assert "SELECT EXISTS" in query
    assert "WHERE workspace = $1" in query
    assert args == ("research",)
    connect.assert_not_awaited()


async def test_clean_orphan_tables_quotes_public_table_identifiers(monkeypatch, config) -> None:
    class Conn:
        def __init__(self) -> None:
            self.executed: list[tuple[str, tuple[object, ...]]] = []
            self.closed = False

        def transaction(self) -> _Tx:
            return _Tx()

        async def fetch(self, query: str) -> list[dict[str, str]]:
            assert "pg_tables" in query
            return [{"tablename": 'dlightrag_bad"name'}]

        async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
            if "information_schema.columns" in query:
                assert "table_schema = 'public'" in query
                assert args == ('dlightrag_bad"name',)
                return {"?column?": 1}
            if "COUNT(*)" in query:
                assert query == (
                    'SELECT COUNT(*) as count FROM public."dlightrag_bad""name" '
                    "WHERE workspace = $1"
                )
                assert args == ("research",)
                return {"count": 1}
            raise AssertionError(query)

        async def fetchval(self, query: str, *args: object) -> str:
            assert query == "SELECT quote_ident($1)"
            assert args == ('dlightrag_bad"name',)
            return '"dlightrag_bad""name"'

        async def execute(self, query: str, *args: object) -> str:
            self.executed.append((query, args))
            return "DELETE 1"

        async def close(self) -> None:
            self.closed = True

    conn = Conn()

    async def fake_connect(**kwargs):
        assert kwargs == config.pg_connection_kwargs.return_value
        return conn

    monkeypatch.setattr("dlightrag.adapters.postgres.corpus.corpus.asyncpg.connect", fake_connect)

    store = PGCorpusMaintenanceStore(config.pg_connection_kwargs())
    cleaned = await store.clean_orphan_rows("research", dry_run=False)

    assert cleaned == 1
    assert conn.executed[0] == (
        'DELETE FROM public."dlightrag_bad""name" WHERE workspace = $1',
        ("research",),
    )
    assert conn.executed[1] == (
        "DELETE FROM dlightrag_promotion_jobs WHERE workspace = $1",
        ("research",),
    )
    assert "UPDATE dlightrag_workspace_meta" in conn.executed[2][0]
    assert conn.executed[2][1] == ("research",)
    assert not any("DELETE FROM dlightrag_workspace_meta" in query for query, _ in conn.executed)
    assert conn.closed is True


async def test_clean_orphan_tables_never_drops_migration_managed_tables(
    monkeypatch, config
) -> None:
    """Reset clears corpus rows but preserves operational identity tables."""

    class Conn:
        def __init__(self) -> None:
            self.executed: list[tuple[str, tuple[object, ...]]] = []
            self.closed = False

        def transaction(self) -> _Tx:
            return _Tx()

        async def fetch(self, query: str) -> list[dict[str, str]]:
            assert "pg_tables" in query
            assert "dlightrag_ingest_jobs" not in query
            return [{"tablename": "dlightrag_doc_metadata"}]

        async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
            if "information_schema.columns" in query:
                return {"?column?": 1}
            if "COUNT(*)" in query:
                return {"count": 1}
            # A "SELECT EXISTS ... has_rows" probe would mean the DROP path is back.
            raise AssertionError(f"unexpected has_rows/DROP probe: {query}")

        async def fetchval(self, query: str, *args: object) -> str:
            assert query == "SELECT quote_ident($1)"
            return "dlightrag_doc_metadata"

        async def execute(self, query: str, *args: object) -> str:
            self.executed.append((query, args))
            return "DELETE 1"

        async def close(self) -> None:
            self.closed = True

    conn = Conn()

    async def fake_connect(**kwargs):
        return conn

    monkeypatch.setattr("dlightrag.adapters.postgres.corpus.corpus.asyncpg.connect", fake_connect)

    store = PGCorpusMaintenanceStore(config.pg_connection_kwargs())
    cleaned = await store.clean_orphan_rows("default", dry_run=False)

    assert cleaned == 1
    assert conn.executed[0] == (
        "DELETE FROM public.dlightrag_doc_metadata WHERE workspace = $1",
        ("default",),
    )
    assert not any("DROP TABLE" in query for query, _ in conn.executed)
    assert not any("DELETE FROM dlightrag_workspace_meta" in query for query, _ in conn.executed)
    assert conn.closed is True


async def test_list_workspace_records_page_delegates_to_the_operational_registry(
    monkeypatch, config
) -> None:
    conn = _Conn()
    connect = AsyncMock(side_effect=AssertionError("registry operations must not connect directly"))
    monkeypatch.setattr("dlightrag.adapters.postgres.corpus.corpus.asyncpg.connect", connect)
    monkeypatch.setattr(
        PGWorkspaceRegistry,
        "list_page",
        AsyncMock(
            return_value=type(
                "Page",
                (),
                {"items": ({"workspace": "finance"},), "has_more": False, "fetched_rows": 1},
            )()
        ),
    )

    store = _maintenance_store(config, conn)
    rows, has_more = await store.list_workspace_records_page(
        after_workspace="default",
        limit=50,
    )

    PGWorkspaceRegistry.list_page.assert_awaited_once_with(  # type: ignore[attr-defined]
        after_workspace="default",
        limit=50,
    )
    assert rows == [{"workspace": "finance"}]
    assert has_more is False
    connect.assert_not_awaited()
