# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Read-only LightRAG PostgreSQL attach for corpus-read-only reader processes.

LightRAG's normal ``PostgreSQLDB.initdb()`` bootstraps the vector extension,
tables, and indexes, and ``check_tables()`` runs schema migrations. A reader
owns no corpus schema and must never run that path -- even after a transient
pool reset/reconnect. ``ReadOnlyPostgreSQLDB`` overrides only pool bootstrap: it
creates the asyncpg pool with the pgvector codec, SSL, statement cache, server
settings, and VCHORDRQ session setup, but issues no DDL. All of LightRAG's
query-time retry/reconnect machinery
(``_ensure_pool``/``_reset_pool``/``_run_with_retry``) is inherited unchanged, so
reconnect re-enters this read-only bootstrap and can never fall back to DDL.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import parse_qsl

import asyncpg
from lightrag.kg.pgtable_impl import PGTableGraphStorage
from lightrag.kg.postgres_impl import ClientManager, PostgreSQLDB, namespace_to_table_name
from lightrag.kg.shared_storage import (
    get_default_workspace,
    initialize_pipeline_status,
    set_default_workspace,
)
from lightrag.lightrag import StoragesStatus
from pgvector.asyncpg import register_vector

from dlightrag.adapters.postgres.core.identifiers import pg_qualified_identifier
from dlightrag.adapters.postgres.corpus.lightrag_contract import READ_ONLY_STORAGE_ATTRS
from dlightrag.engine.rag.workspace.ports import CorpusSchemaError

logger = logging.getLogger(__name__)

# PGTableGraphStorage keeps the whole graph in two shared tables scoped by
# (workspace, namespace). LightRAG's namespace_to_table_name() has no entry
# for the graph namespace, so the reader names them here.
GRAPH_TABLES = ("lightrag_graph_nodes", "lightrag_graph_edges")

# Every retrieval leg resolves chunks through this table, so it is the cheapest
# proof that a reader still sees the corpus it attached to.
CORPUS_PROBE_TABLE = namespace_to_table_name("text_chunks") or "LIGHTRAG_DOC_CHUNKS"


def parse_postgres_server_settings(raw: Any) -> dict[str, str]:
    """Parse LightRAG's ``POSTGRES_SERVER_SETTINGS`` query-string format."""
    if raw is None:
        return {}
    return dict(parse_qsl(str(raw), keep_blank_values=False))


class ReadOnlyPostgreSQLDB(PostgreSQLDB):
    """PostgreSQLDB that attaches to an existing schema without any DDL."""

    def _pool_kwargs(self) -> dict[str, Any]:
        pool_kwargs: dict[str, Any] = {
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "host": self.host,
            "port": self.port,
            "min_size": 1,
            "max_size": self.max,
        }
        if self.statement_cache_size is not None:
            pool_kwargs["statement_cache_size"] = int(self.statement_cache_size)

        ssl_context = self._create_ssl_context()
        if ssl_context is not None:
            pool_kwargs["ssl"] = ssl_context
        elif self.ssl_mode:
            ssl_mode = str(self.ssl_mode).lower()
            if ssl_mode in {"require", "prefer"}:
                pool_kwargs["ssl"] = True
            elif ssl_mode == "disable":
                pool_kwargs["ssl"] = False

        server_settings = parse_postgres_server_settings(self.server_settings)
        if server_settings:
            pool_kwargs["server_settings"] = server_settings
        return pool_kwargs

    async def _init_read_only_connection(self, connection: asyncpg.Connection) -> None:
        if self.enable_vector:
            await register_vector(connection)
        if self.enable_vector and self.vector_index_type == "VCHORDRQ":
            await self.configure_vchordrq(connection)

    async def _reset_read_only_connection(self, connection: asyncpg.Connection) -> None:
        reset_query = connection.get_reset_query()
        if reset_query:
            await connection.execute(reset_query)
        if self.enable_vector and self.vector_index_type == "VCHORDRQ":
            await self.configure_vchordrq(connection)

    async def initdb(self) -> None:
        """Create a read-only asyncpg pool without LightRAG bootstrap DDL."""
        attempts = max(1, int(self.connection_retry_attempts))
        backoff = max(0.0, float(self.connection_retry_backoff))
        backoff_max = max(backoff, float(self.connection_retry_backoff_max))

        for attempt in range(1, attempts + 1):
            try:
                self.pool = await asyncpg.create_pool(
                    **self._pool_kwargs(),
                    init=self._init_read_only_connection,
                    reset=self._reset_read_only_connection,
                )
                return
            except self._transient_exceptions as exc:
                self.pool = None
                if attempt >= attempts:
                    raise
                sleep_for = min(backoff * (2 ** (attempt - 1)), backoff_max)
                logger.warning(
                    "PostgreSQL read-only pool transient connection issue on attempt %d/%d: %r",
                    attempt,
                    attempts,
                    exc,
                )
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)


_VECTOR_STORAGE_ATTRS = frozenset({"entities_vdb", "relationships_vdb", "chunks_vdb"})


def _active_lightrag_storages(
    lightrag: Any,
    *,
    vector_storage: str = "PGVectorStorage",
) -> list[Any]:
    """Return only storages owned by the shared PostgreSQL client.

    External vector adapters must never receive ``ReadOnlyPostgreSQLDB``.  The
    public reader contract currently cannot initialize them without mutation,
    so deployment validation rejects reader+Milvus before this attach path.
    """
    return [
        storage
        for name in READ_ONLY_STORAGE_ATTRS
        if not (name in _VECTOR_STORAGE_ATTRS and vector_storage != "PGVectorStorage")
        if (storage := getattr(lightrag, name, None)) is not None
    ]


def _read_only_vector_signature(
    *, vector_storage: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    db_config = ClientManager.get_config(vector_storage=vector_storage)
    signature = ClientManager._build_vector_signature(db_config, vector_storage)
    return db_config, signature


async def _acquire_read_only_db(
    *,
    db_config: dict[str, Any],
    signature: dict[str, Any],
    active_storage_count: int,
) -> ReadOnlyPostgreSQLDB:
    async with ClientManager._lock:
        db = ClientManager._instances["db"]
        if db is None:
            db = ReadOnlyPostgreSQLDB(db_config)
            await db.initdb()
            ClientManager._instances["db"] = db
            ClientManager._instances["ref_count"] = 0
            ClientManager._instances["vector_signature"] = signature
        else:
            ClientManager._assert_compatible_vector_signature(signature)
        ClientManager._instances["ref_count"] += active_storage_count
    return db


async def _release_read_only_db(db: Any, *, reference_count: int) -> None:
    async def _release() -> None:
        close_db = None
        async with ClientManager._lock:
            if db is ClientManager._instances["db"]:
                ClientManager._instances["ref_count"] -= reference_count
                if ClientManager._instances["ref_count"] <= 0:
                    ClientManager._instances["ref_count"] = 0
                    close_db = db
                    ClientManager._instances["db"] = None
                    ClientManager._instances["vector_signature"] = None
            else:
                close_db = db

        if close_db is not None and close_db.pool is not None:
            await close_db.pool.close()

    cleanup_task = asyncio.create_task(_release())
    try:
        await asyncio.shield(cleanup_task)
    except asyncio.CancelledError as cancellation:
        current = asyncio.current_task()
        if current is not None:
            while current.cancelling():
                current.uncancel()
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                if current is not None:
                    while current.cancelling():
                        current.uncancel()
        try:
            cleanup_task.result()
        except Exception:
            logger.warning("Failed to roll back LightRAG read-only DB references", exc_info=True)
        raise cancellation
    except Exception:
        logger.warning("Failed to roll back LightRAG read-only DB references", exc_info=True)


def _bind_storage_db_workspace(
    *,
    storages: list[Any],
    db: Any,
    fallback_workspace: str,
) -> None:
    for storage in storages:
        storage.db = db
        if getattr(db, "workspace", None):
            storage.workspace = db.workspace
        elif not getattr(storage, "workspace", None):
            storage.workspace = fallback_workspace


def _required_tables(storages: list[Any]) -> set[str]:
    tables: set[str] = set()
    for storage in storages:
        if isinstance(storage, PGTableGraphStorage):
            tables.update(GRAPH_TABLES)
            continue
        table_name = getattr(storage, "table_name", None)
        if isinstance(table_name, str) and table_name:
            tables.add(table_name)
            continue
        namespace = getattr(storage, "namespace", None)
        mapped = namespace_to_table_name(namespace) if namespace else None
        if mapped:
            tables.add(mapped)
    return tables


async def _verify_reader_session(conn: asyncpg.Connection) -> None:
    read_only = await conn.fetchval("SHOW transaction_read_only")
    if str(read_only).lower() != "on":
        raise CorpusSchemaError(
            "LightRAG reader pool is not read-only; expected default_transaction_read_only=on"
        )


async def _verify_required_tables(
    conn: asyncpg.Connection,
    *,
    tables: set[str],
) -> None:
    for table in sorted(tables):
        try:
            await conn.fetchval(
                f"SELECT 1 FROM {pg_qualified_identifier(table)} LIMIT 1"  # noqa: S608
            )
        except Exception as exc:
            raise CorpusSchemaError(
                f"LightRAG table {table} is missing or unreadable; initialize it on the writer first"
            ) from exc


async def verify_lightrag_read_only_schema(
    *,
    db: Any,
    storages: list[Any],
) -> None:
    """Verify read-only attach targets exist without any DDL."""
    if db.pool is None:
        raise RuntimeError("LightRAG read-only PostgreSQL pool was not created")

    async with db.pool.acquire() as conn:
        await _verify_reader_session(conn)
        await _verify_required_tables(conn, tables=_required_tables(storages))


async def verify_reader_corpus_session() -> None:
    """Confirm the attached corpus pool is still read-only and still sees the corpus.

    Readiness must fail rather than let a reader answer from a corpus it can no
    longer read, or from a session that lost its read-only default.
    """
    db = ClientManager._instances["db"]
    pool = getattr(db, "pool", None)
    if pool is None:
        raise RuntimeError("LightRAG reader corpus pool is not attached")

    async with pool.acquire() as conn:
        await _verify_reader_session(conn)
        await _verify_required_tables(conn, tables={CORPUS_PROBE_TABLE})


def _bind_default_workspace(lightrag: Any) -> None:
    if get_default_workspace() is None:
        set_default_workspace(lightrag.workspace)


async def attach_lightrag_storages_read_only(
    lightrag: Any,
    *,
    config: Any,
    vector_storage: str | None = None,
) -> None:
    """Attach LightRAG PostgreSQL storages to a read-only pool without DDL."""
    selected_vector = (
        config.storage.lightrag.vector_storage if vector_storage is None else vector_storage
    )
    if selected_vector != "PGVectorStorage":
        raise ValueError(
            "reader storage attach supports PGVectorStorage only; "
            "MilvusVectorDBStorage has no public nonmutating reader lifecycle"
        )
    active_storages = _active_lightrag_storages(
        lightrag,
        vector_storage=selected_vector,
    )
    db_config, signature = _read_only_vector_signature(vector_storage=selected_vector)
    db = await _acquire_read_only_db(
        db_config=db_config,
        signature=signature,
        active_storage_count=len(active_storages),
    )
    try:
        _bind_storage_db_workspace(
            storages=active_storages,
            db=db,
            fallback_workspace=config.deployment.workspace,
        )
        await verify_lightrag_read_only_schema(db=db, storages=active_storages)
        lightrag._owning_loop = asyncio.get_running_loop()
        _bind_default_workspace(lightrag)
        await initialize_pipeline_status(workspace=lightrag.workspace)
        lightrag._storages_status = StoragesStatus.INITIALIZED
    except BaseException:
        await _release_read_only_db(db, reference_count=len(active_storages))
        raise


__all__ = [
    "CORPUS_PROBE_TABLE",
    "ReadOnlyPostgreSQLDB",
    "attach_lightrag_storages_read_only",
    "parse_postgres_server_settings",
    "verify_lightrag_read_only_schema",
    "verify_reader_corpus_session",
]
