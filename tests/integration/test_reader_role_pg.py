# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for reader-role PostgreSQL session and schema semantics.

A reader is corpus-read-only, not process-read-only: its DlightRAG domain pool
must accept operational writes while its LightRAG corpus pool must be read-only
in PostgreSQL itself. Reader startup must validate the migrated domain schema
without issuing any DDL and must fail before serving when that schema is absent.

Every test runs inside a throwaway database created and dropped per test, so the
developer's ``dlightrag`` database is never mutated.

Requires PostgreSQL at localhost:5432 (dlightrag/dlightrag); skipped otherwise.
"""

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any, cast

import asyncpg
import pytest

from dlightrag.adapters.postgres.core._migrations import (
    Migration,
    TableRequirement,
)
from dlightrag.adapters.postgres.core._migrations import (
    apply_migrations as _apply_migrations,
)
from dlightrag.adapters.postgres.core._migrations import (
    verify_migrations as _verify_migrations,
)
from dlightrag.adapters.postgres.corpus import pg_metadata_index, workspaces
from dlightrag.adapters.postgres.runtime.run_store import (
    RUN_MIGRATION_SCOPE,
    RUN_MIGRATIONS,
    RUN_SCHEMA_TABLES,
    PGRunStore,
)
from dlightrag.adapters.postgres.web import web_conversations
from dlightrag.application.config import (
    DeploymentSettings,
    DlightragConfig,
    PostgresSettings,
    StorageSettings,
)
from dlightrag.engine.ai.settings import EmbeddingSettings, ModelsSettings
from dlightrag.engine.rag.workspace.ports import CorpusSchemaError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_PG_CONN_KWARGS: dict[str, Any] = dict(
    host="localhost",
    port=5432,
    user="dlightrag",
    password="dlightrag",
    database="dlightrag",
)


async def apply_migrations(conn: Any, **kwargs: Any) -> None:
    await _apply_migrations(conn, schema_error=CorpusSchemaError, **kwargs)


async def verify_migrations(conn: Any, **kwargs: Any) -> None:
    await _verify_migrations(conn, schema_error=CorpusSchemaError, **kwargs)


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _restore_process_env() -> Iterator[None]:
    """Config construction bridges PostgreSQL settings into LightRAG's env API."""
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


@pytest.fixture
async def database() -> AsyncIterator[str]:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")

    db_name = f"dlightrag_reader_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    # The document-metadata migration installs a pg_trgm GIN index; the
    # throwaway databases carry the extension the writer's startup path
    # guarantees on real environments.
    setup = await asyncpg.connect(**{**_PG_CONN_KWARGS, "database": db_name})
    try:
        await setup.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    finally:
        await setup.close()
    try:
        yield db_name
    finally:
        admin = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


def _config(database: str, *, service_role: str) -> DlightragConfig:
    return cast(Any, DlightragConfig)(
        _env_file=None,
        deployment=DeploymentSettings(service_role=cast(Any, service_role)),
        storage=StorageSettings(
            postgres=PostgresSettings(
                host=_PG_CONN_KWARGS["host"],
                port=_PG_CONN_KWARGS["port"],
                user=_PG_CONN_KWARGS["user"],
                password=_PG_CONN_KWARGS["password"],
                database=database,
            )
        ),
        models=ModelsSettings(
            embedding=EmbeddingSettings(
                provider="voyage", model="m", api_key="k", dim=8, startup_probe=False
            )
        ),
    )


async def _pool(config: DlightragConfig, settings: dict[str, str]) -> Any:
    return await asyncpg.create_pool(
        **config.pg_connection_kwargs(),
        min_size=1,
        max_size=2,
        server_settings=settings,
    )


# ---------------------------------------------------------------------------
# Session modes
# ---------------------------------------------------------------------------


async def test_reader_domain_pool_accepts_operational_writes(database: str) -> None:
    config = _config(database, service_role="reader")
    pool = await _pool(config, config.domain_pool_server_settings())
    try:
        async with pool.acquire() as conn:
            assert str(await conn.fetchval("SHOW transaction_read_only")).lower() == "off"
            transaction = conn.transaction()
            await transaction.start()
            try:
                await conn.execute("CREATE TABLE reader_probe (id INT)")
                await conn.execute("INSERT INTO reader_probe VALUES (1)")
                assert await conn.fetchval("SELECT COUNT(*) FROM reader_probe") == 1
            finally:
                await transaction.rollback()
    finally:
        await pool.close()


async def test_reader_corpus_pool_rejects_writes_in_postgres(database: str) -> None:
    config = _config(database, service_role="reader")
    writer_pool = await _pool(
        _config(database, service_role="writer"),
        _config(database, service_role="writer").domain_pool_server_settings(),
    )
    try:
        async with writer_pool.acquire() as conn:
            await conn.execute("CREATE TABLE corpus_probe (id INT)")
    finally:
        await writer_pool.close()

    pool = await _pool(config, config.lightrag_pool_server_settings())
    try:
        async with pool.acquire() as conn:
            assert str(await conn.fetchval("SHOW transaction_read_only")).lower() == "on"
            assert await conn.fetchval("SELECT COUNT(*) FROM corpus_probe") == 0
            with pytest.raises(asyncpg.exceptions.ReadOnlySQLTransactionError):
                await conn.execute("INSERT INTO corpus_probe VALUES (1)")
    finally:
        await pool.close()


async def test_writer_corpus_pool_stays_writable(database: str) -> None:
    config = _config(database, service_role="writer")
    pool = await _pool(config, config.lightrag_pool_server_settings())
    try:
        async with pool.acquire() as conn:
            assert str(await conn.fetchval("SHOW transaction_read_only")).lower() == "off"
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# Validation-only startup
# ---------------------------------------------------------------------------


async def test_reader_startup_fails_before_the_writer_migrates(database: str) -> None:
    config = _config(database, service_role="reader")
    pool = await _pool(config, config.domain_pool_server_settings())
    try:
        store = PGRunStore(pool=pool)
        with pytest.raises(RuntimeError, match="dlightrag_schema_migrations"):
            await store.initialize(validate_only=True)

        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT to_regclass('dlightrag_runs')") is None
    finally:
        await pool.close()


async def test_reader_startup_validates_a_migrated_schema_without_ddl(database: str) -> None:
    writer_config = _config(database, service_role="writer")
    writer_pool = await _pool(writer_config, writer_config.domain_pool_server_settings())
    try:
        await PGRunStore(pool=writer_pool).initialize()
    finally:
        await writer_pool.close()

    reader_config = _config(database, service_role="reader")
    reader_pool = await _pool(reader_config, reader_config.lightrag_pool_server_settings())
    try:
        # A read-only session proves validation issues no DDL and no ledger write.
        await PGRunStore(pool=reader_pool).initialize(validate_only=True)
    finally:
        await reader_pool.close()


async def test_reader_startup_fails_when_a_declared_version_is_missing(database: str) -> None:
    config = _config(database, service_role="writer")
    pool = await _pool(config, config.domain_pool_server_settings())
    try:
        async with pool.acquire() as conn:
            await apply_migrations(
                conn,
                scope=RUN_MIGRATION_SCOPE,
                migrations=RUN_MIGRATIONS[:1],
            )
            await conn.execute(
                "DELETE FROM dlightrag_schema_migrations WHERE scope = $1 AND version = $2",
                RUN_MIGRATION_SCOPE,
                RUN_MIGRATIONS[0].version,
            )
            with pytest.raises(RuntimeError, match=RUN_MIGRATION_SCOPE):
                await verify_migrations(
                    conn,
                    scope=RUN_MIGRATION_SCOPE,
                    migrations=RUN_MIGRATIONS,
                    tables=RUN_SCHEMA_TABLES,
                )
    finally:
        await pool.close()


async def test_metadata_field_stats_migration_backfills_only_published_rows(
    database: str,
) -> None:
    config = _config(database, service_role="writer")
    pool = await _pool(config, config.domain_pool_server_settings())
    pre_stats = tuple(
        migration
        for migration in pg_metadata_index._SCHEMA_MIGRATIONS
        if migration.version not in {"metadata_field_stats", "product_document_visibility"}
    )
    try:
        async with pool.acquire() as conn:
            await apply_migrations(
                conn,
                scope="doc_metadata",
                migrations=pre_stats,
                require_applied_prefix=False,
            )
            await conn.execute(
                "INSERT INTO dlightrag_doc_metadata "
                "(workspace, doc_id, title, custom_metadata, "
                "_dlightrag_finalization_complete) VALUES "
                "('legacy', 'doc-hidden', 'Hidden', '{\"hidden_only\":true}', FALSE), "
                "('legacy', 'doc-ready', 'Report', '{\"department\":\"finance\"}', TRUE)"
            )

            await apply_migrations(
                conn,
                scope="doc_metadata",
                migrations=pg_metadata_index._SCHEMA_MIGRATIONS,
                require_applied_prefix=False,
            )

            rows = await conn.fetch(
                "SELECT field_id, document_count "
                "FROM dlightrag_metadata_field_stats "
                "WHERE workspace = 'legacy' ORDER BY field_id"
            )
            assert [(str(row["field_id"]), int(row["document_count"])) for row in rows] == [
                ("department", 1),
                ("title", 1),
            ]
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# Required schema objects, not just the ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scope:
    """One DlightRAG-owned migration scope and the objects it requires."""

    name: str
    migrations: tuple[Migration, ...]
    tables: tuple[TableRequirement, ...]
    require_applied_prefix: bool = True


_SCOPES = (
    _Scope("workspace_registry", workspaces._SCHEMA_MIGRATIONS, workspaces._SCHEMA_TABLES),
    _Scope(
        "doc_metadata",
        pg_metadata_index._SCHEMA_MIGRATIONS,
        pg_metadata_index._SCHEMA_TABLES,
        require_applied_prefix=False,
    ),
    _Scope(RUN_MIGRATION_SCOPE, RUN_MIGRATIONS, RUN_SCHEMA_TABLES),
    _Scope(
        "web_conversations",
        web_conversations.WEB_CONVERSATION_MIGRATIONS,
        web_conversations.WEB_CONVERSATION_SCHEMA_TABLES,
    ),
)

# Drops an auto-named constraint without building SQL from a catalog value.
_DROP_TURN_UNIQUE_KEY = """
DO $$
DECLARE target text;
BEGIN
    SELECT conname INTO STRICT target
    FROM pg_catalog.pg_constraint
    WHERE conrelid = 'web_conversation_turns'::regclass AND contype = 'u';
    EXECUTE format('ALTER TABLE web_conversation_turns DROP CONSTRAINT %I', target);
END $$;
"""

_DAMAGE_CASES = [
    pytest.param(
        "workspace_registry",
        "DROP TABLE dlightrag_workspace_meta",
        ["table dlightrag_workspace_meta"],
        id="workspace-table",
    ),
    pytest.param(
        "doc_metadata",
        "ALTER TABLE dlightrag_doc_metadata DROP COLUMN title",
        ["column dlightrag_doc_metadata.title"],
        id="doc-column",
    ),
    pytest.param(
        "doc_metadata",
        "DROP INDEX idx_dm_author",
        ["index idx_dm_author"],
        id="doc-index",
    ),
    pytest.param(
        RUN_MIGRATION_SCOPE,
        "DROP TABLE dlightrag_answer_run_artifacts",
        ["table dlightrag_answer_run_artifacts"],
        id="answer-table",
    ),
    pytest.param(
        RUN_MIGRATION_SCOPE,
        "DROP INDEX idx_dlightrag_runs_submission",
        ["index idx_dlightrag_runs_submission"],
        id="run-index",
    ),
    pytest.param(
        RUN_MIGRATION_SCOPE,
        "ALTER TABLE dlightrag_runs DROP CONSTRAINT dlightrag_runs_status_check",
        ["constraint dlightrag_runs_status_check"],
        id="answer-check",
    ),
    pytest.param(
        RUN_MIGRATION_SCOPE,
        "ALTER TABLE dlightrag_blobs DROP CONSTRAINT dlightrag_blobs_pkey CASCADE",
        [
            "primary key dlightrag_blobs (owner_id, digest)",
            "foreign key dlightrag_answer_run_artifacts (owner_id, digest) -> dlightrag_blobs",
        ],
        id="answer-primary-and-foreign-key",
    ),
    pytest.param(
        "web_conversations",
        "DROP TABLE web_conversation_turns",
        ["table web_conversation_turns"],
        id="web-table",
    ),
    pytest.param(
        "web_conversations",
        "ALTER TABLE web_conversation_turns "
        "DROP CONSTRAINT web_conversation_turns_principal_id_answer_run_id_fkey",
        ["foreign key web_conversation_turns (principal_id, answer_run_id) "],
        id="web-run-link",
    ),
    pytest.param(
        "web_conversations",
        "DROP INDEX idx_web_conversation_turns_submission",
        ["index idx_web_conversation_turns_submission"],
        id="web-index",
    ),
    pytest.param(
        "web_conversations",
        _DROP_TURN_UNIQUE_KEY,
        ["unique key web_conversation_turns (principal_id, conversation_id, turn_number)"],
        id="web-unique-key",
    ),
]


async def _migrate_every_scope(conn: Any) -> None:
    for scope in _SCOPES:
        await apply_migrations(
            conn,
            scope=scope.name,
            migrations=scope.migrations,
            require_applied_prefix=scope.require_applied_prefix,
        )


async def _ledger_snapshot(conn: Any) -> list[tuple[str, str]]:
    rows = await conn.fetch("SELECT scope, version FROM dlightrag_schema_migrations")
    return sorted((str(row["scope"]), str(row["version"])) for row in rows)


async def test_writer_migrations_satisfy_every_declared_schema_requirement(database: str) -> None:
    """The writer's DDL and the reader's requirement descriptors must not drift."""
    config = _config(database, service_role="writer")
    pool = await _pool(config, config.domain_pool_server_settings())
    try:
        async with pool.acquire() as conn:
            await _migrate_every_scope(conn)
            # Re-running must stay idempotent for a writer that restarts.
            await _migrate_every_scope(conn)
    finally:
        await pool.close()

    reader_config = _config(database, service_role="reader")
    # A read-only session proves validation issues no DDL and no ledger write.
    reader_pool = await _pool(reader_config, reader_config.lightrag_pool_server_settings())
    try:
        async with reader_pool.acquire() as conn:
            for scope in _SCOPES:
                await verify_migrations(
                    conn,
                    scope=scope.name,
                    migrations=scope.migrations,
                    tables=scope.tables,
                )
    finally:
        await reader_pool.close()


@pytest.mark.parametrize(("scope_name", "damage", "expected"), _DAMAGE_CASES)
async def test_reader_rejects_a_recorded_ledger_missing_a_required_object(
    database: str, scope_name: str, damage: str, expected: list[str]
) -> None:
    scope = next(candidate for candidate in _SCOPES if candidate.name == scope_name)
    writer_config = _config(database, service_role="writer")
    writer_pool = await _pool(writer_config, writer_config.domain_pool_server_settings())
    try:
        async with writer_pool.acquire() as conn:
            await _migrate_every_scope(conn)
            await conn.execute(damage)
            before = await _ledger_snapshot(conn)
    finally:
        await writer_pool.close()

    reader_config = _config(database, service_role="reader")
    reader_pool = await _pool(reader_config, reader_config.lightrag_pool_server_settings())
    try:
        async with reader_pool.acquire() as conn:
            with pytest.raises(CorpusSchemaError) as excinfo:
                await verify_migrations(
                    conn,
                    scope=scope.name,
                    migrations=scope.migrations,
                    tables=scope.tables,
                )
            message = str(excinfo.value)
            assert scope.name in message
            for fragment in expected:
                assert fragment in message
    finally:
        await reader_pool.close()

    verify_config = _config(database, service_role="writer")
    verify_pool = await _pool(verify_config, verify_config.domain_pool_server_settings())
    try:
        async with verify_pool.acquire() as conn:
            assert await _ledger_snapshot(conn) == before
    finally:
        await verify_pool.close()


async def test_reader_rejects_an_undeclared_migration_version(database: str) -> None:
    config = _config(database, service_role="writer")
    pool = await _pool(config, config.domain_pool_server_settings())
    try:
        async with pool.acquire() as conn:
            await _migrate_every_scope(conn)
            await conn.execute(
                "INSERT INTO dlightrag_schema_migrations (scope, version, description) "
                "VALUES ($1, $2, $3)",
                RUN_MIGRATION_SCOPE,
                "9999_from_a_newer_revision",
                "undeclared",
            )

            with pytest.raises(CorpusSchemaError, match="undeclared versions"):
                await verify_migrations(
                    conn,
                    scope=RUN_MIGRATION_SCOPE,
                    migrations=RUN_MIGRATIONS,
                    tables=RUN_SCHEMA_TABLES,
                )
    finally:
        await pool.close()
