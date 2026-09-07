# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for the corpus-read-only reader role."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from lightrag.kg.pgtable_impl import PGTableGraphStorage

from dlightrag.application.config import (
    DeploymentSettings,
    DlightragConfig,
    PostgresSettings,
    StorageSettings,
)
from dlightrag.engine.ai.settings import EmbeddingSettings, ModelRoleSettings, ModelsSettings


def _config(*, service_role: str = "writer", **overrides) -> DlightragConfig:
    postgres = PostgresSettings(
        host=overrides.pop("postgres_host", "localhost"),
        session_settings=overrides.pop("postgres_session_settings", {}),
    )
    assert not overrides
    return cast(Any, DlightragConfig)(
        _env_file=None,
        deployment=DeploymentSettings(service_role=cast(Any, service_role)),
        storage=StorageSettings(postgres=postgres),
        models=ModelsSettings(
            chat=ModelRoleSettings(),
            embedding=EmbeddingSettings(
                provider="voyage", model="m", api_key="k", dim=8, startup_probe=False
            ),
        ),
    )


class TestServiceRoleConfig:
    def test_defaults_to_writer(self) -> None:
        cfg = _config()
        assert cfg.deployment.service_role == "writer"
        assert cfg.is_reader is False

    def test_reader_predicate(self) -> None:
        assert _config(service_role="reader").is_reader is True

    def test_env_spelling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DLIGHTRAG_DEPLOYMENT__SERVICE_ROLE", "reader")
        cfg = cast(Any, DlightragConfig)(
            _env_file=None,
            models=ModelsSettings(
                embedding=EmbeddingSettings(
                    provider="voyage", model="m", api_key="k", dim=8, startup_probe=False
                )
            ),
        )
        assert cfg.is_reader is True

    def test_invalid_role_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _config(service_role="replica")


class TestReaderPoolSessionModes:
    """A reader is corpus-read-only: only the LightRAG pool runs read-only sessions."""

    def test_reader_domain_pool_stays_writable(self) -> None:
        cfg = _config(service_role="reader")
        assert "default_transaction_read_only" not in cfg.domain_pool_server_settings()
        assert "server_settings" not in cfg.pg_connection_kwargs()

    def test_reader_corpus_pool_is_read_only(self) -> None:
        cfg = _config(service_role="reader")
        assert cfg.lightrag_pool_server_settings()["default_transaction_read_only"] == "on"
        assert "default_transaction_read_only=on" in cfg.postgres_server_settings_env_value()

    def test_writer_has_no_read_only_guc_on_either_pool(self) -> None:
        cfg = _config()
        assert "default_transaction_read_only" not in cfg.domain_pool_server_settings()
        assert "default_transaction_read_only" not in cfg.lightrag_pool_server_settings()
        assert "server_settings" not in cfg.pg_connection_kwargs()

    def test_reader_corpus_read_only_cannot_be_overridden_by_session_setting(self) -> None:
        cfg = _config(
            service_role="reader",
            postgres_session_settings={"default_transaction_read_only": "off"},
        )
        # Reader invariant is applied last and wins on the corpus pool.
        assert cfg.lightrag_pool_server_settings()["default_transaction_read_only"] == "on"


class TestPgPoolBinding:
    def test_bind_same_signature_ok(self) -> None:
        from dlightrag.adapters.postgres.core._pool import PGPool

        pool = PGPool()
        pool.bind(_config())
        pool.bind(_config())  # identical signature -> no raise

    def test_bind_incompatible_role_raises(self) -> None:
        from dlightrag.adapters.postgres.core._pool import PGPool

        pool = PGPool()
        pool.bind(_config())
        with pytest.raises(RuntimeError):
            pool.bind(_config(service_role="reader"))

    def test_bind_incompatible_endpoint_raises(self) -> None:
        from dlightrag.adapters.postgres.core._pool import PGPool

        pool = PGPool()
        pool.bind(_config())
        with pytest.raises(RuntimeError):
            pool.bind(_config(postgres_host="other-host"))

    async def test_close_clears_binding(self) -> None:
        from dlightrag.adapters.postgres.core._pool import PGPool

        pool = PGPool()
        pool.bind(_config())
        await pool.close()
        pool.bind(_config(service_role="reader"))  # rebinding after close is allowed


_SERVICE_WRITE_CALLS = [
    ("areset", (), {}),
    ("aupdate_metadata", ("doc-1", {}), {}),
    ("adelete_files", (), {}),
    ("aingest", ("local",), {}),
    ("aingest_source", (None,), {}),
    ("aretry_failed_docs", (), {}),
    ("_upsert_workspace_meta", (), {}),
]


@pytest.mark.parametrize(("method", "args", "kwargs"), _SERVICE_WRITE_CALLS)
async def test_service_write_guards_reject_reader(method, args, kwargs) -> None:
    from dlightrag.application.settings import rag_settings
    from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag

    service = object.__new__(WorkspaceRag)
    service.settings = rag_settings(_config(service_role="reader"))
    with pytest.raises(PermissionError):
        await getattr(service, method)(*args, **kwargs)


# ---------------------------------------------------------------------------
# Validation-only reader startup
# ---------------------------------------------------------------------------


class _SchemaConn:
    """Fake connection answering ledger and catalog reads; records every write.

    Catalog answers echo the scope's own requirement descriptor, so these tests
    prove the no-DDL dispatch and the ledger rules. Whether the catalog queries
    match real PostgreSQL is proven in ``tests/integration/test_reader_role_pg.py``.
    """

    def __init__(
        self,
        applied: set[tuple[str, str]],
        tables: tuple[Any, ...] = (),
        *,
        ledger_exists: bool = True,
    ) -> None:
        self.applied = applied
        self.ledger_exists = ledger_exists
        self.executed: list[str] = []
        self._tables = {table.name: table for table in tables}
        self._oids = {name: 900 + index for index, name in enumerate(self._tables)}
        self._by_oid = {oid: name for name, oid in self._oids.items()}

    def _table_for(self, *args: Any) -> Any:
        """Resolve one catalog row by OID (migration verifier) or name (partition seam)."""
        key = args[0] if args else None
        if key in self._tables:
            return self._tables[key]
        if key in self._by_oid:
            return self._tables[self._by_oid[key]]
        return None

    async def fetchval(self, sql: str, *args: Any) -> Any:
        if "dlightrag_schema_migrations" in sql:
            return self.ledger_exists
        if "pg_inherits" in sql:
            return True  # every declared parent index has a child index
        if "pg_catalog.pg_class" in sql:
            if "SELECT c.relkind" in sql:
                table = self._table_for(*args)
                return "p" if table is not None and table.partitioned_by else "r"
            return self._oids.get(str(args[0]))
        raise AssertionError(f"unexpected fetchval: {sql}")

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        table = self._table_for(*args)
        if "contype = 'p'" in sql:
            return {"name": f"{table.name}_pkey", "columns": list(table.primary_key)}
        if "pg_partitioned_table" in sql:
            return {"columns": list(table.partitioned_by)}
        raise AssertionError(f"unexpected fetchrow: {sql}")

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        if "pg_catalog" not in sql:
            scope = str(args[0])
            return [
                {"version": version}
                for applied_scope, version in sorted(self.applied)
                if applied_scope == scope
            ]
        table = self._table_for(*args)
        if "attisdropped" in sql:
            return [{"name": name} for name in table.columns]
        if "pg_index" in sql:
            if "pg_get_indexdef" in sql:
                return [
                    {"name": name, "definition": f"CREATE INDEX {name} ON {table.name}"}
                    for name in (*table.indexes, *table.unique_indexes)
                ]
            if "indisunique" in sql:
                return [{"name": name} for name in table.unique_indexes]
            return [{"name": name} for name in (*table.indexes, *table.unique_indexes)]
        if "pg_inherits" in sql and "SELECT c.relname" in sql:
            return [{"name": child} for child in table.required_child_partitions]
        if "contype = 'c'" in sql:
            return [{"name": name} for name in table.checks]
        if "contype IN ('p', 'u')" in sql:
            return [
                {"contype": "p", "columns": list(table.primary_key)},
                *({"contype": "u", "columns": list(columns)} for columns in table.unique),
            ]
        if "contype = 'f'" in sql:
            return [
                {"columns": list(key.columns), "referenced": key.references}
                for key in table.foreign_keys
            ]
        raise AssertionError(f"unexpected fetch: {sql}")

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append(sql)
        return "OK"


def _required_domain_scopes() -> list[
    tuple[str, tuple[Any, ...], tuple[Any, ...], Any, type[RuntimeError]]
]:
    """Each domain store with the scope, versions, and tables it validates.

    The Web conversation store validates the durable Answer run schema too: its
    turns carry a foreign key into ``dlightrag_runs``, so a reader whose
    run schema is absent must fail there as well.
    """
    from dlightrag.adapters.postgres.corpus import pg_metadata_index, workspaces
    from dlightrag.adapters.postgres.runtime import run_store
    from dlightrag.adapters.postgres.web import web_conversations
    from dlightrag.application.web_conversations import WebConversationSchemaError
    from dlightrag.engine.rag.workspace.ports import CorpusSchemaError
    from dlightrag.engine.runtime import RunSchemaError

    return [
        (
            "workspace_registry",
            workspaces._SCHEMA_MIGRATIONS,
            workspaces._SCHEMA_TABLES,
            workspaces.PGWorkspaceRegistry,
            CorpusSchemaError,
        ),
        (
            "doc_metadata",
            pg_metadata_index._SCHEMA_MIGRATIONS,
            pg_metadata_index._SCHEMA_TABLES,
            pg_metadata_index.PGMetadataIndex,
            CorpusSchemaError,
        ),
        (
            "runs",
            run_store.RUN_MIGRATIONS,
            run_store.RUN_SCHEMA_TABLES,
            run_store.PGRunStore,
            RunSchemaError,
        ),
        (
            "web_conversations",
            web_conversations.WEB_CONVERSATION_MIGRATIONS,
            web_conversations.WEB_CONVERSATION_SCHEMA_TABLES,
            web_conversations.PGWebConversationStore,
            WebConversationSchemaError,
        ),
    ]


def _prerequisite_versions(scope: str) -> set[tuple[str, str]]:
    """Versions another scope must already carry before this one validates."""
    from dlightrag.adapters.postgres.runtime import run_store

    if scope != "web_conversations":
        return set()
    return {
        (run_store.RUN_MIGRATION_SCOPE, migration.version) for migration in run_store.RUN_MIGRATIONS
    }


def _prerequisite_tables(scope: str) -> tuple[Any, ...]:
    from dlightrag.adapters.postgres.runtime import run_store

    return run_store.RUN_SCHEMA_TABLES if scope == "web_conversations" else ()


@contextmanager
def _domain_pool_routed_to(conn: _SchemaConn) -> Iterator[None]:
    """Route every domain-store operation at ``conn`` without a real pool."""
    from dlightrag.adapters.postgres.core._pool import pg_pool

    async def _run(operation: Any) -> Any:
        return await operation(conn)

    with (
        patch.object(pg_pool, "run", _run),
        patch.object(pg_pool, "run_once", _run),
    ):
        yield


@pytest.mark.parametrize("scope_index", range(4))
async def test_reader_startup_validates_domain_schema_without_ddl(scope_index: int) -> None:
    scope, migrations, tables, store_cls, _schema_error = _required_domain_scopes()[scope_index]
    conn = _SchemaConn(
        {(scope, migration.version) for migration in migrations} | _prerequisite_versions(scope),
        tables + _prerequisite_tables(scope),
    )

    with _domain_pool_routed_to(conn):
        await store_cls().initialize(validate_only=True)

    assert conn.executed == []


@pytest.mark.parametrize("scope_index", range(4))
async def test_reader_startup_fails_on_incompatible_domain_schema(scope_index: int) -> None:
    from dlightrag.engine.runtime import RunSchemaError

    scope, _migrations, tables, store_cls, schema_error = _required_domain_scopes()[scope_index]
    conn = _SchemaConn(set(), tables + _prerequisite_tables(scope))
    expected = "runs" if scope == "web_conversations" else scope
    expected_error = RunSchemaError if scope == "web_conversations" else schema_error

    with _domain_pool_routed_to(conn), pytest.raises(expected_error, match=expected):
        await store_cls().initialize(validate_only=True)

    assert conn.executed == []


@pytest.mark.parametrize("scope_index", range(4))
async def test_reader_startup_fails_when_a_required_table_is_absent(scope_index: int) -> None:
    """Every declared version can be recorded while a required table is gone."""
    scope, migrations, tables, store_cls, schema_error = _required_domain_scopes()[scope_index]
    conn = _SchemaConn(
        {(scope, migration.version) for migration in migrations} | _prerequisite_versions(scope),
        tables[1:] + _prerequisite_tables(scope),
    )

    with (
        _domain_pool_routed_to(conn),
        pytest.raises(schema_error, match=tables[0].name),
    ):
        await store_cls().initialize(validate_only=True)

    assert conn.executed == []


@pytest.mark.parametrize("scope_index", range(4))
async def test_reader_startup_fails_when_the_migration_ledger_is_absent(scope_index: int) -> None:
    from dlightrag.engine.runtime import RunSchemaError

    _scope, _migrations, tables, store_cls, schema_error = _required_domain_scopes()[scope_index]
    conn = _SchemaConn(set(), tables + _prerequisite_tables(_scope), ledger_exists=False)
    expected_error = RunSchemaError if _scope == "web_conversations" else schema_error

    with (
        _domain_pool_routed_to(conn),
        pytest.raises(expected_error, match="dlightrag_schema_migrations"),
    ):
        await store_cls().initialize(validate_only=True)

    assert conn.executed == []


async def test_reader_serves_web_routes() -> None:
    from dlightrag.adapters.http.server import create_app
    from dlightrag.application.config import loading as config_module

    original = config_module._config
    config_module._config = _config(service_role="reader")
    try:
        app = create_app()
    finally:
        config_module._config = original

    assert "/web/api/answer" in set(app.openapi()["paths"])
    assert app.state.web_enabled is True
    assert not hasattr(app.state, "web_conversation_service")


class TestReadOnlyAdapter:
    def test_external_vector_storages_are_never_bound_to_postgres(self) -> None:
        import dlightrag.adapters.postgres.corpus.lightrag_readonly as readonly

        vector = object()
        kv = object()
        lightrag = SimpleNamespace(
            **{name: None for name in readonly.READ_ONLY_STORAGE_ATTRS},
        )
        lightrag.full_docs = kv
        lightrag.chunks_vdb = vector
        lightrag.entities_vdb = vector
        lightrag.relationships_vdb = vector

        active = readonly._active_lightrag_storages(
            lightrag,
            vector_storage="MilvusVectorDBStorage",
        )

        assert active == [kv]
        assert vector not in active

    def test_external_vector_reader_attach_is_rejected_explicitly(self) -> None:
        import dlightrag.adapters.postgres.corpus.lightrag_readonly as readonly

        config = _config(service_role="reader")
        mutate = cast(Any, config.storage.lightrag)
        object.__setattr__(mutate, "vector_storage", "MilvusVectorDBStorage")

        with pytest.raises(ValueError, match="no public nonmutating|supports PGVectorStorage"):
            asyncio.run(
                readonly.attach_lightrag_storages_read_only(
                    SimpleNamespace(),
                    config=config,
                )
            )

    def test_overrides_initdb_without_ddl_bootstrap(self) -> None:
        from lightrag.kg.postgres_impl import PostgreSQLDB

        from dlightrag.adapters.postgres.corpus.lightrag_readonly import ReadOnlyPostgreSQLDB

        # The reader adapter must override initdb so reconnect never re-enters
        # LightRAG's extension/table/graph bootstrap.
        assert ReadOnlyPostgreSQLDB.initdb is not PostgreSQLDB.initdb
        # It inherits LightRAG's reconnect/retry machinery unchanged.
        assert ReadOnlyPostgreSQLDB._ensure_pool is PostgreSQLDB._ensure_pool
        assert ReadOnlyPostgreSQLDB._run_with_retry is PostgreSQLDB._run_with_retry

    async def test_attach_entry_point_initializes_binds_schema_and_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.adapters.postgres.corpus.lightrag_readonly as readonly

        class FakeConn:
            def __init__(self) -> None:
                self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []

            async def fetchval(self, sql: str, *args: Any) -> Any:
                self.fetchval_calls.append((sql, args))
                if sql == "SHOW transaction_read_only":
                    return "on"
                if "SELECT 1 FROM " in sql:
                    return 1
                raise AssertionError(f"unexpected SQL: {sql}")

        class FakeAcquire:
            def __init__(self, conn: FakeConn) -> None:
                self._conn = conn

            async def __aenter__(self) -> FakeConn:
                return self._conn

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        class FakePool:
            def __init__(self, conn: FakeConn) -> None:
                self._conn = conn

            def acquire(self) -> FakeAcquire:
                return FakeAcquire(self._conn)

        class FakeDB:
            def __init__(self, db_config: dict[str, Any]) -> None:
                self.db_config = db_config
                self.workspace = "db-workspace"
                self.pool = FakePool(conn)
                self.initdb = AsyncMock()

        conn = FakeConn()
        set_workspace_calls: list[str | None] = []
        init_pipeline_status = AsyncMock()
        client_manager = SimpleNamespace(
            _lock=asyncio.Lock(),
            _instances={"db": None, "ref_count": 0, "vector_signature": None},
            get_config=lambda *, vector_storage=None: {
                "database": "db",
                "vector_storage": vector_storage,
            },
            _build_vector_signature=lambda config, vector_storage: {
                "database": config["database"],
                "vector_storage": vector_storage,
            },
        )
        client_manager._assert_compatible_vector_signature = lambda signature: None
        monkeypatch.setattr(readonly, "ClientManager", client_manager, raising=False)
        monkeypatch.setattr(readonly, "ReadOnlyPostgreSQLDB", FakeDB, raising=False)
        monkeypatch.setattr(
            readonly,
            "namespace_to_table_name",
            lambda namespace: {"full_docs": "LIGHTRAG_DOC_FULL"}.get(namespace),
            raising=False,
        )
        monkeypatch.setattr(readonly, "get_default_workspace", lambda: None, raising=False)
        monkeypatch.setattr(
            readonly,
            "set_default_workspace",
            lambda workspace=None: set_workspace_calls.append(workspace),
            raising=False,
        )
        monkeypatch.setattr(
            readonly,
            "initialize_pipeline_status",
            init_pipeline_status,
            raising=False,
        )
        monkeypatch.setattr(
            readonly,
            "StoragesStatus",
            SimpleNamespace(INITIALIZED="initialized"),
            raising=False,
        )

        graph = PGTableGraphStorage.__new__(PGTableGraphStorage)
        graph.db = None
        graph.workspace = ""
        full_docs = SimpleNamespace(db=None, workspace=None, namespace="full_docs", table_name=None)
        chunks_vdb = SimpleNamespace(
            db=None,
            workspace=None,
            namespace=None,
            table_name="LIGHTRAG_DOC_CHUNKS",
        )
        llm_cache = SimpleNamespace(
            db=None, workspace="kept-workspace", namespace=None, table_name=None
        )
        lightrag = SimpleNamespace(
            workspace="reader-workspace",
            full_docs=full_docs,
            text_chunks=None,
            full_entities=None,
            full_relations=None,
            entity_chunks=None,
            relation_chunks=None,
            entities_vdb=None,
            relationships_vdb=None,
            chunks_vdb=chunks_vdb,
            chunk_entity_relation_graph=graph,
            llm_response_cache=llm_cache,
            doc_status=None,
        )

        await readonly.attach_lightrag_storages_read_only(
            lightrag, config=_config(service_role="reader")
        )

        db = client_manager._instances["db"]
        assert isinstance(db, FakeDB)
        db.initdb.assert_awaited_once()
        assert client_manager._instances["ref_count"] == 4
        assert client_manager._instances["vector_signature"] == {
            "database": "db",
            "vector_storage": _config(service_role="reader").storage.lightrag.vector_storage,
        }
        assert full_docs.db is db
        assert chunks_vdb.db is db
        assert llm_cache.db is db
        assert full_docs.workspace == "db-workspace"
        assert chunks_vdb.workspace == "db-workspace"
        assert graph.workspace == "db-workspace"
        init_pipeline_status.assert_awaited_once_with(workspace="reader-workspace")
        assert set_workspace_calls == ["reader-workspace"]
        assert lightrag._storages_status == "initialized"
        assert lightrag._owning_loop is asyncio.get_running_loop()
        assert any(sql == "SHOW transaction_read_only" for sql, _ in conn.fetchval_calls)
        probed = " ".join(sql for sql, _ in conn.fetchval_calls if "SELECT 1 FROM " in sql)
        assert "LIGHTRAG_DOC_FULL" in probed
        assert "lightrag_graph_nodes" in probed
        assert "lightrag_graph_edges" in probed

    async def test_attach_entry_point_reuses_db_and_preserves_signature_checks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.adapters.postgres.corpus.lightrag_readonly as readonly

        class FakeConn:
            async def fetchval(self, sql: str, *args: Any) -> Any:
                if sql == "SHOW transaction_read_only":
                    return "on"
                if "SELECT 1 FROM " in sql:
                    return 1
                raise AssertionError(f"unexpected SQL: {sql}")

        class FakeAcquire:
            def __init__(self, conn: FakeConn) -> None:
                self._conn = conn

            async def __aenter__(self) -> FakeConn:
                return self._conn

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        existing_db = SimpleNamespace(
            pool=SimpleNamespace(acquire=lambda: FakeAcquire(FakeConn())), workspace=""
        )
        seen_signatures: list[dict[str, Any]] = []
        client_manager = SimpleNamespace(
            _lock=asyncio.Lock(),
            _instances={"db": existing_db, "ref_count": 7, "vector_signature": {"database": "db"}},
            get_config=lambda *, vector_storage=None: {"database": "db"},
            _build_vector_signature=lambda config, vector_storage: {"database": config["database"]},
            _assert_compatible_vector_signature=lambda signature: seen_signatures.append(signature),
        )
        monkeypatch.setattr(readonly, "ClientManager", client_manager, raising=False)
        monkeypatch.setattr(
            readonly,
            "namespace_to_table_name",
            lambda namespace: "LIGHTRAG_DOC_FULL",
            raising=False,
        )
        monkeypatch.setattr(readonly, "get_default_workspace", lambda: "already-set", raising=False)
        monkeypatch.setattr(readonly, "set_default_workspace", AsyncMock(), raising=False)
        monkeypatch.setattr(
            readonly,
            "initialize_pipeline_status",
            AsyncMock(),
            raising=False,
        )
        monkeypatch.setattr(
            readonly,
            "StoragesStatus",
            SimpleNamespace(INITIALIZED="initialized"),
            raising=False,
        )

        lightrag = SimpleNamespace(
            workspace="reader-workspace",
            full_docs=SimpleNamespace(
                db=None, workspace=None, namespace="full_docs", table_name=None
            ),
            text_chunks=None,
            full_entities=None,
            full_relations=None,
            entity_chunks=None,
            relation_chunks=None,
            entities_vdb=None,
            relationships_vdb=None,
            chunks_vdb=None,
            chunk_entity_relation_graph=None,
            llm_response_cache=None,
            doc_status=None,
        )

        await readonly.attach_lightrag_storages_read_only(
            lightrag, config=_config(service_role="reader")
        )

        assert client_manager._instances["db"] is existing_db
        assert client_manager._instances["ref_count"] == 8
        assert seen_signatures == [{"database": "db"}]

    async def test_attach_entry_point_releases_db_when_schema_verification_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.adapters.postgres.corpus.lightrag_readonly as readonly

        class FakeConn:
            async def fetchval(self, sql: str, *args: Any) -> Any:
                if sql == "SHOW transaction_read_only":
                    return "off"
                raise AssertionError(f"unexpected SQL after read-only failure: {sql}")

        class FakeAcquire:
            async def __aenter__(self) -> FakeConn:
                return FakeConn()

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        existing_db = SimpleNamespace(
            pool=SimpleNamespace(acquire=FakeAcquire),
            workspace="",
        )
        instances = {
            "db": existing_db,
            "ref_count": 7,
            "vector_signature": {"database": "db"},
        }

        async def release_client(db: object) -> None:
            assert db is existing_db
            instances["ref_count"] -= 1

        client_manager = SimpleNamespace(
            _lock=asyncio.Lock(),
            _instances=instances,
            get_config=lambda *, vector_storage=None: {"database": "db"},
            _build_vector_signature=lambda config, vector_storage: {"database": config["database"]},
            _assert_compatible_vector_signature=lambda signature: None,
            release_client=AsyncMock(side_effect=release_client),
        )
        monkeypatch.setattr(readonly, "ClientManager", client_manager, raising=False)

        lightrag = SimpleNamespace(
            workspace="reader-workspace",
            full_docs=SimpleNamespace(
                db=None, workspace=None, namespace="full_docs", table_name="LIGHTRAG_DOC_FULL"
            ),
            text_chunks=None,
            full_entities=None,
            full_relations=None,
            entity_chunks=None,
            relation_chunks=None,
            entities_vdb=None,
            relationships_vdb=None,
            chunks_vdb=None,
            chunk_entity_relation_graph=None,
            llm_response_cache=None,
            doc_status=None,
        )

        with pytest.raises(RuntimeError, match="not read-only"):
            await readonly.attach_lightrag_storages_read_only(
                lightrag,
                config=_config(service_role="reader"),
            )

        assert instances["ref_count"] == 7

    async def test_read_only_db_rollback_finishes_before_propagating_cancellation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.adapters.postgres.corpus.lightrag_readonly as readonly

        close_started = asyncio.Event()
        allow_close = asyncio.Event()

        async def close_pool() -> None:
            close_started.set()
            await allow_close.wait()

        pool = SimpleNamespace(close=AsyncMock(side_effect=close_pool))
        db = SimpleNamespace(pool=pool)
        instances = {"db": db, "ref_count": 3, "vector_signature": {"database": "db"}}

        async def release_client(released_db: object) -> None:
            assert released_db is db
            async with client_manager._lock:
                instances["ref_count"] -= 1
                if instances["ref_count"] == 0:
                    await pool.close()
                    instances["db"] = None
                    instances["vector_signature"] = None

        client_manager = SimpleNamespace(
            _lock=asyncio.Lock(),
            _instances=instances,
            release_client=AsyncMock(side_effect=release_client),
        )
        monkeypatch.setattr(readonly, "ClientManager", client_manager, raising=False)

        rollback = asyncio.create_task(readonly._release_read_only_db(db, reference_count=3))
        await close_started.wait()
        rollback.cancel()
        allow_close.set()

        with pytest.raises(asyncio.CancelledError):
            await rollback

        assert instances == {"db": None, "ref_count": 0, "vector_signature": None}
        pool.close.assert_awaited_once_with()

    async def test_attach_entry_point_signature_mismatch_keeps_refcount(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.adapters.postgres.corpus.lightrag_readonly as readonly

        class FakeConn:
            async def fetchval(self, sql: str, *args: Any) -> Any:
                raise AssertionError(f"schema should not be queried after signature failure: {sql}")

        class FakeAcquire:
            def __init__(self, conn: FakeConn) -> None:
                self._conn = conn

            async def __aenter__(self) -> FakeConn:
                return self._conn

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        existing_db = SimpleNamespace(
            pool=SimpleNamespace(acquire=lambda: FakeAcquire(FakeConn())), workspace=""
        )
        client_manager = SimpleNamespace(
            _lock=asyncio.Lock(),
            _instances={"db": existing_db, "ref_count": 5, "vector_signature": {"database": "db"}},
            get_config=lambda *, vector_storage=None: {"database": "db"},
            _build_vector_signature=lambda config, vector_storage: {"database": config["database"]},
            _assert_compatible_vector_signature=lambda signature: (_ for _ in ()).throw(
                RuntimeError("vector mismatch")
            ),
        )
        monkeypatch.setattr(readonly, "ClientManager", client_manager, raising=False)

        lightrag = SimpleNamespace(
            workspace="reader-workspace",
            full_docs=SimpleNamespace(
                db=None, workspace=None, namespace="full_docs", table_name=None
            ),
            text_chunks=None,
            full_entities=None,
            full_relations=None,
            entity_chunks=None,
            relation_chunks=None,
            entities_vdb=None,
            relationships_vdb=None,
            chunks_vdb=None,
            chunk_entity_relation_graph=None,
            llm_response_cache=None,
            doc_status=None,
        )

        with pytest.raises(RuntimeError, match="vector mismatch"):
            await readonly.attach_lightrag_storages_read_only(
                lightrag,
                config=_config(service_role="reader"),
            )

        assert client_manager._instances["ref_count"] == 5
