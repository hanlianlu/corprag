# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for DlightRAG-owned PostgreSQL schema migrations."""

from collections.abc import Sequence
from typing import Any

import pytest

from dlightrag.adapters.postgres.core._migrations import (
    ForeignKeyRequirement,
    Migration,
    TableRequirement,
)
from dlightrag.adapters.postgres.core._migrations import (
    apply_migrations as _apply_migrations,
)
from dlightrag.adapters.postgres.core._migrations import (
    verify_migrations as _verify_migrations,
)


class _SchemaError(RuntimeError):
    """Owner-selected schema error used to exercise the shared primitive."""


async def apply_migrations(conn: Any, **kwargs: Any) -> None:
    await _apply_migrations(conn, schema_error=_SchemaError, **kwargs)


async def verify_migrations(conn: Any, **kwargs: Any) -> None:
    await _verify_migrations(conn, schema_error=_SchemaError, **kwargs)


class _Tx:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self) -> None:
        self._conn.transaction_events.append("begin")
        return None

    async def __aexit__(self, exc_type: object, *args: object) -> None:
        self._conn.transaction_events.append("rollback" if exc_type else "commit")
        return None


class _Record:
    def __init__(self, version: str) -> None:
        self._version = version

    def __getitem__(self, key: str) -> str:
        if key != "version":
            raise KeyError(key)
        return self._version


class _Conn:
    def __init__(
        self,
        *,
        row_shape: str = "dict",
        ledger_exists: bool = True,
        catalog: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.applied: set[tuple[str, str]] = set()
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []
        self.transaction_events: list[str] = []
        self.failures: dict[str, int] = {}
        self.row_shape = row_shape
        self.ledger_exists = ledger_exists
        self.catalog = dict(catalog or {})
        self._oids = {name: 900 + index for index, name in enumerate(self.catalog)}
        self._by_oid = {oid: name for name, oid in self._oids.items()}

    def transaction(self) -> _Tx:
        return _Tx(self)

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetches.append((query, args))
        if "dlightrag_schema_migrations" in query:
            return self.ledger_exists
        if "pg_catalog.pg_class" in query:
            return self._oids.get(str(args[0]))
        raise AssertionError(f"unexpected fetchval: {query}")

    async def fetch(self, query: str, *args: Any) -> Sequence[dict[str, Any] | _Record]:
        self.fetches.append((query, args))
        if "pg_catalog" in query:
            return self._catalog_rows(query, self.catalog[self._by_oid[int(args[0])]])
        scope = str(args[0])
        versions = sorted(
            version for applied_scope, version in self.applied if applied_scope == scope
        )
        if self.row_shape == "record":
            return [_Record(version) for version in versions]
        return [{"version": version} for version in versions]

    @staticmethod
    def _catalog_rows(query: str, table: dict[str, Any]) -> list[dict[str, Any]]:
        if "attisdropped" in query:
            return [{"name": name} for name in table.get("columns", ())]
        if "pg_index" in query:
            unique = list(table.get("unique_indexes", ()))
            if "indisunique" in query:
                return [{"name": name} for name in unique]
            return [{"name": name} for name in (*table.get("indexes", ()), *unique)]
        if "contype = 'c'" in query:
            return [{"name": name} for name in table.get("checks", ())]
        if "contype IN ('p', 'u')" in query:
            return [
                {"contype": contype, "columns": list(columns)}
                for contype, columns in table.get("keys", ())
            ]
        if "contype = 'f'" in query:
            return [
                {"columns": list(columns), "referenced": referenced}
                for columns, referenced in table.get("fks", ())
            ]
        raise AssertionError(f"unexpected catalog fetch: {query}")

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        remaining_failures = self.failures.get(query, 0)
        if remaining_failures > 0:
            self.failures[query] = remaining_failures - 1
            raise RuntimeError(f"boom: {query}")
        if query.startswith("INSERT INTO dlightrag_schema_migrations"):
            self.applied.add((str(args[0]), str(args[1])))
            return "INSERT 0 1"
        if query.startswith("SELECT pg_advisory_lock"):
            return "SELECT 1"
        if query.startswith("SELECT pg_advisory_unlock"):
            return "SELECT 1"
        return "OK"


def _example_migrations() -> tuple[Migration, ...]:
    return (
        Migration("create_table", "first", ("CREATE TABLE example (id TEXT)",)),
        Migration("add_name", "second", ("ALTER TABLE example ADD COLUMN name TEXT",)),
    )


def _three_migrations() -> tuple[Migration, ...]:
    return (
        Migration("create_table", "first", ("CREATE TABLE example (id TEXT)",)),
        Migration("add_name", "second", ("ALTER TABLE example ADD COLUMN name TEXT",)),
        Migration("create_index", "third", ("CREATE INDEX example_name_idx ON example (name)",)),
    )


async def test_apply_migrations_skips_versions_already_recorded_for_scope() -> None:
    conn = _Conn()
    migrations = _example_migrations()

    await apply_migrations(conn, scope="example", migrations=migrations)
    await apply_migrations(conn, scope="example", migrations=migrations)

    executed_sql = [query for query, _ in conn.executed]
    assert executed_sql.count("CREATE TABLE example (id TEXT)") == 1
    assert executed_sql.count("ALTER TABLE example ADD COLUMN name TEXT") == 1
    assert conn.applied == {("example", "create_table"), ("example", "add_name")}


async def test_apply_migrations_runs_only_newly_appended_versions() -> None:
    conn = _Conn(row_shape="record")
    initial = (Migration("create_table", "first", ("CREATE TABLE example (id TEXT)",)),)
    appended = initial + (
        Migration("add_name", "second", ("ALTER TABLE example ADD COLUMN name TEXT",)),
    )

    await apply_migrations(conn, scope="example", migrations=initial)
    await apply_migrations(conn, scope="example", migrations=appended)

    executed_sql = [query for query, _ in conn.executed]
    assert executed_sql.count("CREATE TABLE example (id TEXT)") == 1
    assert executed_sql.count("ALTER TABLE example ADD COLUMN name TEXT") == 1
    assert conn.applied == {("example", "create_table"), ("example", "add_name")}


async def test_run_event_guard_is_an_append_only_migration_applied_once_in_order() -> None:
    from dlightrag.adapters.postgres.runtime.run_store import (
        RUN_MIGRATION_SCOPE,
        RUN_MIGRATIONS,
    )

    expected_versions = (
        "run_runtime_v1",
        "child_roster_index",
        "worker_cancel_pending_index",
        "write_model_published_artifact_kind",
        "write_model_root_artifact_attachments",
        "write_model_web_resource_catalog",
        "corpus_mutation_runtime",
        "normalize_run_event_constraints",
    )
    assert tuple(migration.version for migration in RUN_MIGRATIONS) == expected_versions
    assert all(
        "dlightrag_enforce_run_event_constraints" not in statement
        and "trg_dlightrag_run_events_enforce" not in statement
        for migration in RUN_MIGRATIONS[:-1]
        for statement in migration.statements
    )

    conn = _Conn()
    await apply_migrations(
        conn,
        scope=RUN_MIGRATION_SCOPE,
        migrations=RUN_MIGRATIONS[:-1],
    )
    executed_before_append = len(conn.executed)
    await apply_migrations(conn, scope=RUN_MIGRATION_SCOPE, migrations=RUN_MIGRATIONS)
    await apply_migrations(conn, scope=RUN_MIGRATION_SCOPE, migrations=RUN_MIGRATIONS)

    recorded_versions = [
        str(args[1])
        for query, args in conn.executed
        if query.startswith("INSERT INTO dlightrag_schema_migrations")
        and args[0] == RUN_MIGRATION_SCOPE
    ]
    assert recorded_versions == list(expected_versions)
    guard_statements = RUN_MIGRATIONS[-1].statements
    assert len(guard_statements) == 2
    executed_sql = [query for query, _ in conn.executed]
    assert all(
        statement not in executed_sql[:executed_before_append] for statement in guard_statements
    )
    assert all(executed_sql.count(statement) == 1 for statement in guard_statements)


async def test_apply_migrations_does_not_record_failed_versions() -> None:
    conn = _Conn()
    migrations = (
        Migration("create_table", "first", ("CREATE TABLE example (id TEXT)",)),
        Migration("add_name", "second", ("ALTER TABLE example ADD COLUMN name TEXT",)),
    )
    conn.failures["ALTER TABLE example ADD COLUMN name TEXT"] = 1

    with pytest.raises(RuntimeError, match="ALTER TABLE example ADD COLUMN name TEXT"):
        await apply_migrations(conn, scope="example", migrations=migrations)

    assert conn.applied == {("example", "create_table")}
    assert ("example", "add_name") not in conn.applied
    assert conn.transaction_events == ["begin", "commit", "begin", "rollback"]
    assert not any(
        query.startswith("INSERT INTO dlightrag_schema_migrations")
        and args[:2] == ("example", "add_name")
        for query, args in conn.executed
    )


async def test_apply_migrations_isolates_applied_versions_per_scope() -> None:
    conn = _Conn()
    migrations = _example_migrations()

    await apply_migrations(conn, scope="alpha", migrations=migrations)
    await apply_migrations(conn, scope="beta", migrations=migrations)

    assert conn.applied == {
        ("alpha", "create_table"),
        ("alpha", "add_name"),
        ("beta", "create_table"),
        ("beta", "add_name"),
    }
    applied_reads = [
        args[0] for query, args in conn.fetches if "dlightrag_schema_migrations" in query
    ]
    assert applied_reads == ["alpha", "beta"]


async def test_apply_migrations_rejects_gapped_current_ledger_before_running_migrations() -> None:
    conn = _Conn()
    conn.applied.add(("example", "add_name"))
    migrations = _example_migrations()

    with pytest.raises(
        RuntimeError,
        match=(
            r"scope 'example'.*missing current versions: create_table.*"
            r"out-of-order recorded current versions: add_name"
        ),
    ):
        await apply_migrations(conn, scope="example", migrations=migrations)

    executed_sql = [query for query, _ in conn.executed]
    assert "CREATE TABLE example (id TEXT)" not in executed_sql
    assert "ALTER TABLE example ADD COLUMN name TEXT" not in executed_sql
    assert not any(
        query.startswith("INSERT INTO dlightrag_schema_migrations") for query in executed_sql
    )
    assert conn.applied == {("example", "add_name")}
    assert conn.transaction_events == []


@pytest.mark.parametrize("require_applied_prefix", [True, False])
async def test_apply_migrations_rejects_undeclared_ledger_versions(
    require_applied_prefix: bool,
) -> None:
    conn = _Conn()
    conn.applied.update({("example", "create_table"), ("example", "0999")})
    migrations = _example_migrations()

    with pytest.raises(
        RuntimeError,
        match=r"scope 'example'.*undeclared versions: 0999.*reset the development database",
    ):
        await apply_migrations(
            conn,
            scope="example",
            migrations=migrations,
            require_applied_prefix=require_applied_prefix,
        )

    executed_sql = [query for query, _ in conn.executed]
    assert executed_sql.count("CREATE TABLE example (id TEXT)") == 0
    assert executed_sql.count("ALTER TABLE example ADD COLUMN name TEXT") == 0
    assert conn.applied == {("example", "create_table"), ("example", "0999")}


async def test_apply_migrations_releases_lock_when_gap_validation_fails() -> None:
    conn = _Conn()
    conn.applied.add(("example", "add_name"))

    with pytest.raises(RuntimeError, match=r"scope 'example'"):
        await apply_migrations(conn, scope="example", migrations=_example_migrations())

    executed_sql = [query for query, _ in conn.executed]
    assert executed_sql.count("SELECT pg_advisory_lock($1)") == 1
    assert executed_sql.count("SELECT pg_advisory_unlock($1)") == 1
    assert executed_sql[-1] == "SELECT pg_advisory_unlock($1)"


async def test_apply_migrations_can_run_missing_versions_from_non_prefix_ledger() -> None:
    conn = _Conn()
    conn.applied.add(("example", "add_name"))

    await apply_migrations(
        conn,
        scope="example",
        migrations=_three_migrations(),
        require_applied_prefix=False,
    )

    executed_sql = [query for query, _ in conn.executed]
    assert executed_sql.count("CREATE TABLE example (id TEXT)") == 1
    assert executed_sql.count("ALTER TABLE example ADD COLUMN name TEXT") == 0
    assert executed_sql.count("CREATE INDEX example_name_idx ON example (name)") == 1
    assert conn.applied == {
        ("example", "create_table"),
        ("example", "add_name"),
        ("example", "create_index"),
    }
    assert executed_sql.index("CREATE TABLE example (id TEXT)") < executed_sql.index(
        "CREATE INDEX example_name_idx ON example (name)"
    )


async def test_apply_migrations_rejects_duplicate_versions_before_mutating_db() -> None:
    conn = _Conn()
    duplicate = (
        Migration("create_table", "first", ("CREATE TABLE example (id TEXT)",)),
        Migration("create_table", "first again", ("ALTER TABLE example ADD COLUMN name TEXT",)),
    )

    with pytest.raises(ValueError, match="Duplicate schema migration version: create_table"):
        await apply_migrations(conn, scope="example", migrations=duplicate)

    assert conn.executed == []


def _example_table() -> TableRequirement:
    return TableRequirement(
        name="example",
        columns=("id", "name"),
        primary_key=("id",),
        unique=(("name",),),
        foreign_keys=(ForeignKeyRequirement(columns=("id",), references="parent"),),
        checks=("example_name_check",),
        indexes=("example_name_idx",),
        unique_indexes=("example_key_idx",),
    )


def _example_catalog() -> dict[str, dict[str, Any]]:
    """A migrated ``example`` table containing every required object."""
    return {
        "example": {
            "columns": ["id", "name"],
            "indexes": ["example_name_idx"],
            "unique_indexes": ["example_key_idx"],
            "checks": ["example_name_check"],
            "keys": [("p", ["id"]), ("u", ["name"])],
            "fks": [(["id"], "parent")],
        }
    }


async def test_verify_migrations_accepts_a_fully_applied_scope_without_any_ddl() -> None:
    conn = _Conn(catalog=_example_catalog())
    migrations = _example_migrations()
    await apply_migrations(conn, scope="example", migrations=migrations)
    conn.executed.clear()

    await verify_migrations(
        conn, scope="example", migrations=migrations, tables=(_example_table(),)
    )

    assert conn.executed == []


async def test_verify_migrations_rejects_undeclared_ledger_versions() -> None:
    conn = _Conn(row_shape="record", catalog=_example_catalog())
    conn.applied.update({("example", "create_table"), ("example", "add_name"), ("example", "0999")})

    with pytest.raises(RuntimeError, match=r"scope 'example'.*undeclared versions: 0999"):
        await verify_migrations(
            conn, scope="example", migrations=_example_migrations(), tables=(_example_table(),)
        )


async def test_verify_migrations_reports_every_missing_version() -> None:
    conn = _Conn()
    conn.applied.add(("example", "create_table"))

    with pytest.raises(RuntimeError, match=r"scope 'example'.*add_name, create_index"):
        await verify_migrations(conn, scope="example", migrations=_three_migrations(), tables=())

    assert conn.executed == []


async def test_verify_migrations_reports_a_missing_ledger() -> None:
    conn = _Conn(ledger_exists=False)

    with pytest.raises(RuntimeError, match="dlightrag_schema_migrations"):
        await verify_migrations(conn, scope="example", migrations=_example_migrations(), tables=())

    assert conn.executed == []


_DAMAGED_CATALOGS: list[tuple[str, str]] = [
    ("table", "table example"),
    ("column", "column example.name"),
    ("index", "index example_name_idx"),
    ("unique_index", "unique index example_key_idx"),
    ("check", "constraint example_name_check"),
    ("primary_key", "primary key example (id)"),
    ("unique", "unique key example (name)"),
    ("foreign_key", "foreign key example (id) -> parent"),
]


def _damaged_catalog(kind: str) -> dict[str, dict[str, Any]]:
    catalog = _example_catalog()
    if kind == "table":
        return {}
    table = catalog["example"]
    if kind == "column":
        table["columns"] = ["id", "legacy_column"]
    elif kind == "index":
        table["indexes"] = ["example_legacy_idx"]
    elif kind == "unique_index":
        # The name survives a rebuild that dropped uniqueness; only the catalog
        # flag proves the invariant the index is required to enforce.
        table["indexes"] = [*table["indexes"], "example_key_idx"]
        table["unique_indexes"] = []
    elif kind == "check":
        table["checks"] = ["example_legacy_check"]
    elif kind == "primary_key":
        table["keys"] = [("u", ["name"]), ("u", ["legacy_column"])]
    elif kind == "unique":
        table["keys"] = [("p", ["id"]), ("u", ["legacy_column"])]
    elif kind == "foreign_key":
        table["fks"] = []
    return catalog


@pytest.mark.parametrize(("kind", "expected"), _DAMAGED_CATALOGS)
async def test_verify_migrations_rejects_a_fully_recorded_ledger_missing_an_object(
    kind: str, expected: str
) -> None:
    """A ledger row survives a dropped object, so the ledger alone cannot be trusted."""
    conn = _Conn(catalog=_damaged_catalog(kind))
    conn.applied.update({("example", "create_table"), ("example", "add_name")})

    with pytest.raises(_SchemaError) as excinfo:
        await verify_migrations(
            conn,
            scope="example",
            migrations=_example_migrations(),
            tables=(_example_table(),),
        )

    assert expected in str(excinfo.value)
    assert "example" in str(excinfo.value)
    assert conn.executed == []


async def test_web_conversation_migration_creates_only_final_run_links() -> None:
    """The baseline creates only final Web conversation and run-link state."""
    from dlightrag.adapters.postgres.web.web_conversations import WEB_CONVERSATION_MIGRATIONS

    conn = _Conn()
    await apply_migrations(
        conn,
        scope="web_conversations",
        migrations=WEB_CONVERSATION_MIGRATIONS,
    )

    ddl = [query for query, _ in conn.executed if "web_conversation" in query.lower()]
    create_index = next(
        i
        for i, q in enumerate(ddl)
        if "CREATE TABLE IF NOT EXISTS web_conversation_turns" in q and "answer_run_id" in q
    )
    assert "REFERENCES dlightrag_runs (owner_id, run_id)" in ddl[create_index]
    assert "ON DELETE CASCADE" in ddl[create_index]

    # Nothing outside the Web conversation scope is touched.
    executed_sql = "\n".join(query for query, _ in conn.executed)
    for foreign in (
        "dlightrag_doc_metadata",
        "lightrag_doc_chunks",
        "lightrag_graph_nodes",
        "ingest_jobs",
        "dlightrag_checkpoints",
        "init.sql",
    ):
        assert foreign not in executed_sql
    # Every applied version was recorded against the web_conversations scope only.
    assert {scope for scope, _ in conn.applied} == {"web_conversations"}
    assert conn.applied == {
        ("web_conversations", "web_conversations"),
    }
