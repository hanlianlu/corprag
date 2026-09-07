# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL composition adapter for one LightRAG corpus backend."""

import asyncio
import importlib
import logging
import os
import random
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from lightrag.constants import DEFAULT_COSINE_THRESHOLD

from dlightrag.adapters.postgres.core._errors import is_postgres_unavailable
from dlightrag.adapters.postgres.core._locks import advisory_lock_key
from dlightrag.adapters.postgres.core._pool import pg_pool
from dlightrag.adapters.postgres.core._version import (
    ensure_pgvector_halfvec,
    ensure_postgres_extensions,
    ensure_postgres_major,
)
from dlightrag.adapters.postgres.corpus._corpus_schema import CHUNK_DOCUMENT_SCOPE_INDEX
from dlightrag.adapters.postgres.corpus.corpus_bm25 import (
    create_postgres_bm25,
    required_postgres_extensions,
)
from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore
from dlightrag.adapters.postgres.corpus.corpus_vectors import PGFilteredVectorSearch
from dlightrag.adapters.postgres.corpus.doc_status_lookup import PGDocStatusLookup
from dlightrag.adapters.postgres.corpus.file_panel import PGFilePanelStore
from dlightrag.adapters.postgres.corpus.lightrag_contract import PGLightRAGContractGuard
from dlightrag.adapters.postgres.corpus.lightrag_readonly import (
    attach_lightrag_storages_read_only,
)
from dlightrag.adapters.postgres.corpus.partition_foundation import (
    PartitionedTableSpec,
    PGPartitionFoundation,
)
from dlightrag.adapters.postgres.corpus.pg_metadata_index import PGMetadataIndex
from dlightrag.adapters.postgres.corpus.promotion_jobs import PGPromotionJobStore
from dlightrag.adapters.postgres.corpus.promotion_worker import PGPromotionWorker
from dlightrag.adapters.postgres.corpus.workspace_write_gate import (
    workspace_write_gate as _workspace_write_gate,
)
from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry
from dlightrag.application.config import DlightragConfig
from dlightrag.engine.rag.retrieval.bm25 import profile_languages, profiles_from_config
from dlightrag.engine.rag.workspace.ports import (
    CorpusRuntimeModels,
    CorpusUnavailableError,
    WorkspaceCorpusBackend,
    WorkspaceCorpusStores,
    WorkspaceWriteFencedError,
)
from dlightrag.engine.rag.workspace.settings import RagSettings

logger = logging.getLogger(__name__)

_PG_INIT_LOCK_KEY = 0x446C_6967_6874_0001
_INIT_WAIT_SECONDS = 180.0
_PROCESS_COUNT_ENV_VARS = ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS")
_ORPHAN_TABLES = """SELECT tablename FROM pg_tables
WHERE schemaname = 'public'
  AND (
      tablename LIKE 'lightrag_%'
      OR tablename IN ('dlightrag_doc_metadata', 'dlightrag_metadata_field_stats')
  )
ORDER BY tablename
"""
_HAS_WORKSPACE_COLUMN = """SELECT 1 FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = $1 AND column_name = 'workspace'
"""
_CHUNKS_REQUIRED_COLUMNS = ("id", "workspace", "full_doc_id", "content", "file_path")
_VECTOR_REQUIRED_COLUMNS = (
    "id",
    "workspace",
    "full_doc_id",
    "content",
    "content_vector",
    "file_path",
)
_STORAGE_SELECTIONS = (
    ("KV_STORAGE", "kv_storage"),
    ("VECTOR_STORAGE", "vector_storage"),
    ("GRAPH_STORAGE", "graph_storage"),
    ("DOC_STATUS_STORAGE", "doc_status_storage"),
)


def verify_lightrag_storage_configuration(config: DlightragConfig) -> None:
    """Validate all four deployment-static names through LightRAG's public contract."""
    from lightrag.kg import verify_storage_implementation

    selected = config.storage.lightrag
    for storage_type, field_name in _STORAGE_SELECTIONS:
        verify_storage_implementation(storage_type, str(getattr(selected, field_name)))

    if selected.vector_storage != "MilvusVectorDBStorage":
        return
    if config.is_reader:
        raise ValueError(
            "service_role='reader' cannot use MilvusVectorDBStorage because LightRAG "
            "does not expose a nonmutating external-vector reader attach lifecycle"
        )
    _verify_milvus_dependency()


def _verify_milvus_dependency() -> None:
    """Fail before LightRAG's Milvus module can attempt runtime installation."""
    try:
        importlib.import_module("grpc")
        importlib.import_module("pymilvus")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "MilvusVectorDBStorage requires the optional Milvus client; "
            "install DlightRAG with the 'milvus' extra"
        ) from exc


def lightrag_retrieval_table_specs(
    lightrag: Any,
    *,
    require_chunk_scope_index: bool = True,
) -> tuple[PartitionedTableSpec, ...]:
    """Partition specs for the LightRAG-owned retrieval-critical tables.

    ``LIGHTRAG_DOC_CHUNKS`` (and therefore BM25) plus the dynamic chunk-vector
    table this runtime attached. LightRAG keeps creating these tables; the
    partition seam converts the fresh empty ones after ``initialize_storages``
    and validates them on every later startup. Writers omit DlightRAG's chunk
    scope index only during the initial conversion, create it on the resulting
    parent, and then validate the complete runtime contract; readers always
    require the complete contract without issuing DDL.
    """
    from dlightrag.adapters.postgres.corpus._corpus_schema import LIGHTRAG_CHUNKS_TABLE

    chunks_spec = PartitionedTableSpec(
        name=LIGHTRAG_CHUNKS_TABLE,
        required_columns=_CHUNKS_REQUIRED_COLUMNS,
        primary_key=("workspace", "id"),
        required_indexes=(
            "idx_lightrag_doc_chunks_id",
            "idx_lightrag_doc_chunks_workspace_id",
            *((CHUNK_DOCUMENT_SCOPE_INDEX,) if require_chunk_scope_index else ()),
        ),
    )
    vdb = getattr(lightrag, "chunks_vdb", None)
    vector_table = getattr(vdb, "table_name", None)
    if not vector_table:
        return (chunks_spec,)
    index_type = str(getattr(getattr(vdb, "db", None), "vector_index_type", "") or "").lower()
    # HNSW and HNSW_HALFVEC share the ``hnsw`` access method (halfvec changes
    # only the operator class), so the indexdef marker is the method name.
    marker = {
        "hnsw": "USING hnsw",
        "hnsw_halfvec": "USING hnsw",
        "ivfflat": "USING ivfflat",
        "vchordrq": "USING vchordrq",
    }.get(index_type, "")
    return (
        chunks_spec,
        PartitionedTableSpec(
            name=str(vector_table),
            required_columns=_VECTOR_REQUIRED_COLUMNS,
            primary_key=("workspace", "id"),
            required_index_markers=(marker,) if marker else (),
        ),
    )


def _configured_process_count() -> int:
    for name in _PROCESS_COUNT_ENV_VARS:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            count = int(raw)
        except ValueError:
            continue
        if count > 0:
            return count
    return 1


class PGCorpusCoordination:
    """Own PostgreSQL sessions and locks used by corpus startup."""

    def __init__(
        self,
        *,
        connection_kwargs: Mapping[str, Any],
        workspace: str,
        reader: bool,
        require_halfvec: bool,
        required_extensions: tuple[str, ...],
        lightrag_pool_max_size: int,
        domain_pool_max_size: int,
        acquire_timeout: float,
    ) -> None:
        self._connection_kwargs = dict(connection_kwargs)
        self._workspace = workspace
        self._reader = reader
        self._require_halfvec = require_halfvec
        self._required_extensions = required_extensions
        self._lightrag_pool_max_size = lightrag_pool_max_size
        self._domain_pool_max_size = domain_pool_max_size
        self._acquire_timeout = acquire_timeout

    async def _connect(self) -> Any:
        try:
            return await asyncpg.connect(**self._connection_kwargs)
        except Exception as exc:
            raise CorpusUnavailableError(
                "PostgreSQL is required for DlightRAG startup and could not be reached"
            ) from exc

    async def _log_connection_budget(self, conn: Any) -> None:
        try:
            max_connections = int(await conn.fetchval("SHOW max_connections"))
        except Exception:
            logger.debug("Could not read PostgreSQL max_connections", exc_info=True)
            return

        # The promotion write gate opens dedicated connections bounded by a
        # process-wide semaphore sized from the domain pool max size, so the
        # per-process budget is lightrag + domain + gate.
        gate_max_size = self._domain_pool_max_size
        per_process = self._lightrag_pool_max_size + self._domain_pool_max_size + gate_max_size
        process_count = _configured_process_count()
        estimated = per_process * process_count
        reserved = max(5, max_connections // 10)
        usable = max_connections - reserved
        logger.info(
            "PostgreSQL connection sanity: max_connections=%d usable_after_headroom=%d "
            "configured_pool_connections_per_process=%d "
            "(lightrag=%d, dlightrag=%d, promotion_gate=%d) process_count=%d "
            "estimated_pool_connections=%d",
            max_connections,
            usable,
            per_process,
            self._lightrag_pool_max_size,
            self._domain_pool_max_size,
            gate_max_size,
            process_count,
            estimated,
        )
        if estimated > usable:
            logger.warning(
                "PostgreSQL connection budget is tight: estimated_pool_connections=%d "
                "exceeds usable_after_headroom=%d (max_connections=%d, headroom=%d). "
                "Lower postgres_lightrag_pool_max_size/postgres_pool_max_size/process count "
                "or raise max_connections.",
                estimated,
                usable,
                max_connections,
                reserved,
            )

    async def _wait_for_initializer(self, conn: Any) -> None:
        logger.info("Another worker is initializing, waiting for lock...")
        waited = 0.0
        backoff = 0.1
        while waited < _INIT_WAIT_SECONDS:
            jitter = random.uniform(0, backoff * 0.5)  # noqa: S311 - contention jitter
            await asyncio.sleep(backoff + jitter)
            waited += backoff + jitter
            if await conn.fetchval("SELECT pg_try_advisory_lock($1)", _PG_INIT_LOCK_KEY):
                await conn.execute("SELECT pg_advisory_unlock($1)", _PG_INIT_LOCK_KEY)
                return
            backoff = min(backoff * 1.5, 5.0)
        logger.warning("Lock acquisition timeout after %.0fs", _INIT_WAIT_SECONDS)

    @asynccontextmanager
    async def workspace_initialization(self) -> AsyncIterator[None]:
        conn = await self._connect()
        try:
            await ensure_postgres_major(conn)
            if self._require_halfvec:
                await ensure_pgvector_halfvec(conn)
            await self._log_connection_budget(conn)

            if self._reader:
                logger.info("Initializing RAG pipelines in corpus-read-only reader role")
                yield
                return

            acquired = await conn.fetchval("SELECT pg_try_advisory_lock($1)", _PG_INIT_LOCK_KEY)
            if not acquired:
                await self._wait_for_initializer(conn)
                logger.info("Lock released, connecting to existing storages...")
                yield
                return

            logger.info("Acquired PG advisory lock, initializing RAG pipelines...")
            try:
                await ensure_postgres_extensions(conn, self._required_extensions)
                yield
                logger.info("RAG pipelines initialized successfully")
            finally:
                await conn.execute("SELECT pg_advisory_unlock($1)", _PG_INIT_LOCK_KEY)
        finally:
            await conn.close()

    @asynccontextmanager
    async def pipeline_recovery(self) -> AsyncIterator[None]:
        """Serialize startup native pipeline recovery behind the write gate.

        LightRAG's ``apipeline_process_enqueue_documents`` sweep is a real
        corpus write, so it joins the cross-process per-workspace gate:
        while a promotion fence is active this waits (polling, cancellation
        safe) instead of abandoning pending documents, then holds the shared
        gate for the whole sweep so a promotion cutover drains it.
        """
        from dlightrag.adapters.postgres.corpus.workspace_write_gate import (
            _active_fence_seconds,
            workspace_write_gate,
        )

        while True:
            try:
                remaining = await pg_pool.run(
                    lambda conn: _active_fence_seconds(conn, self._workspace)
                )
            except Exception as exc:
                if is_postgres_unavailable(exc):
                    raise CorpusUnavailableError(
                        "Corpus PostgreSQL session is unavailable"
                    ) from exc
                raise
            if remaining > 0:
                logger.info(
                    "Pipeline recovery for workspace '%s' waiting %.0fs for a "
                    "promotion fence to clear",
                    self._workspace,
                    remaining,
                )
                await asyncio.sleep(min(5.0, max(0.5, remaining)))
                continue
            try:
                async with workspace_write_gate(self._workspace):
                    pool = await pg_pool.get()
                    async with pool.acquire(timeout=self._acquire_timeout) as conn:
                        lock_key = advisory_lock_key("dlightrag_pipeline_recovery", self._workspace)
                        await conn.execute("SELECT pg_advisory_lock($1)", lock_key)
                        try:
                            yield
                        finally:
                            await conn.execute("SELECT pg_advisory_unlock($1)", lock_key)
                return
            except WorkspaceWriteFencedError:
                # A fence landed between the poll and the gate acquisition:
                # wait it out and retry rather than abandoning documents.
                continue
            except Exception as exc:
                if is_postgres_unavailable(exc):
                    raise CorpusUnavailableError(
                        "Corpus PostgreSQL session is unavailable"
                    ) from exc
                raise


class PGCorpusMaintenanceStore:
    """Own catalog and workspace-row maintenance for the PostgreSQL corpus."""

    def __init__(
        self,
        connection_kwargs: Mapping[str, Any],
        *,
        workspace_registry: PGWorkspaceRegistry | None = None,
        promotion_jobs: PGPromotionJobStore | None = None,
    ) -> None:
        self._connection_kwargs = dict(connection_kwargs)
        self._workspace_registry = workspace_registry or PGWorkspaceRegistry()
        self._promotion_jobs = promotion_jobs or PGPromotionJobStore()

    async def initialize(self, *, validate_only: bool = False) -> None:
        # The job table is part of the Commit-1 durable schema foundation even
        # before Commit 3 wires its worker. Readers validate both scopes and
        # remain strictly DDL-free.
        await self._workspace_registry.initialize(validate_only=validate_only)
        await self._promotion_jobs.initialize(validate_only=validate_only)

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[Any]:
        try:
            conn = await asyncpg.connect(**self._connection_kwargs)
        except Exception as exc:
            raise CorpusUnavailableError("Corpus PostgreSQL session is unavailable") from exc
        try:
            yield conn
        finally:
            await conn.close()

    async def clean_orphan_rows(self, workspace: str, *, dry_run: bool) -> int:
        """Clear corpus-owned rows and maintenance counters, never Workspace identity."""
        async with self._connection() as conn:
            async with conn.transaction():
                rows = await conn.fetch(_ORPHAN_TABLES)
                cleaned = 0
                for row in rows:
                    table = str(row["tablename"])
                    if await conn.fetchrow(_HAS_WORKSPACE_COLUMN, table) is None:
                        continue
                    quoted = await conn.fetchval("SELECT quote_ident($1)", table)
                    qualified = f"public.{quoted}"
                    count_row = await conn.fetchrow(
                        f"SELECT COUNT(*) as count FROM {qualified} WHERE workspace = $1",  # noqa: S608
                        workspace,
                    )
                    count = int(count_row["count"]) if count_row else 0
                    if count <= 0:
                        continue
                    if not dry_run:
                        await conn.execute(
                            f"DELETE FROM {qualified} WHERE workspace = $1",  # noqa: S608
                            workspace,
                        )
                    cleaned += 1
                if not dry_run:
                    await conn.execute(
                        "DELETE FROM dlightrag_promotion_jobs WHERE workspace = $1",
                        workspace,
                    )
                    await conn.execute(
                        """UPDATE dlightrag_workspace_meta
                           SET ingested_docs_total = 0,
                               ingested_chunks_total = 0,
                               promotion_state = 'none',
                               promotion_retry_count = 0,
                               promotion_last_error = NULL,
                               promotion_next_retry_at = NULL,
                               updated_at = NOW()
                           WHERE workspace = $1""",
                        workspace,
                    )
                return cleaned

    async def list_workspace_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(await self._workspace_registry.list())

    async def list_workspace_records_page(
        self,
        *,
        after_workspace: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        page = await self._workspace_registry.list_page(
            after_workspace=after_workspace,
            limit=limit,
        )
        return [dict(item) for item in page.items], page.has_more

    async def workspace_exists(self, workspace: str) -> bool:
        return await self._workspace_registry.exists(workspace)

    async def register_workspace(
        self,
        *,
        workspace: str,
        display_name: str,
        embedding_model: str,
    ) -> None:
        await self._workspace_registry.upsert(
            workspace=workspace,
            display_name=display_name,
            embedding_model=embedding_model,
        )

    async def get_workspace_record(self, workspace: str) -> dict[str, Any] | None:
        """Return the full registry row including storage/promotion facts."""
        return await self._workspace_registry.get_row(workspace)

    @asynccontextmanager
    async def workspace_write_gate(self, workspace: str) -> AsyncIterator[None]:
        """Gate one workspace write behind the promotion fence and drain protocol."""
        async with _workspace_write_gate(workspace):
            yield None


class PGCorpusRuntimeBinder:
    """Bind PostgreSQL corpus adapters to initialized LightRAG storage."""

    def __init__(self, config: DlightragConfig) -> None:
        self._config = config

    def create(self, *, models: CorpusRuntimeModels, settings: RagSettings) -> Any:
        """Construct LightRAG after validating its four public storage contracts."""
        config = self._config
        verify_lightrag_storage_configuration(config)

        # Import only after the Milvus preflight. LightRAG's Milvus module may
        # otherwise invoke its package installer during import.
        from lightrag import LightRAG

        vector_kwargs: dict[str, Any] = {
            "cosine_better_than_threshold": DEFAULT_COSINE_THRESHOLD,
            **config.storage.lightrag.vector_db_kwargs,
        }
        return LightRAG(
            working_dir=str(config.working_dir_path),
            llm_model_func=models.default_llm_func,
            embedding_func=models.embedding_func,
            workspace=config.deployment.workspace,
            default_llm_timeout=int(settings.model_roles.default.timeout),
            default_embedding_timeout=int(settings.embedding.timeout),
            **settings.lightrag_pipeline_kwargs(),
            llm_model_max_async=settings.rag_pipeline_max_async,
            embedding_func_max_async=settings.embedding_func_max_async,
            embedding_batch_num=settings.embedding_batch_num,
            vector_storage=config.storage.lightrag.vector_storage,
            graph_storage=config.storage.lightrag.graph_storage,
            kv_storage=config.storage.lightrag.kv_storage,
            doc_status_storage=config.storage.lightrag.doc_status_storage,
            vector_db_storage_cls_kwargs=vector_kwargs,
            role_llm_configs=models.role_llm_configs,
            kg_chunk_pick_method=settings.kg_chunk_pick_method,
            entity_extraction_use_json=settings.entity_extraction_use_json,
            addon_params=settings.addon_params(),
            enable_llm_cache=not settings.read_only,
            enable_llm_cache_for_entity_extract=not settings.read_only,
        )

    async def attach(self, lightrag: Any) -> WorkspaceCorpusStores:
        config = self._config
        guard = PGLightRAGContractGuard(lightrag)
        guard.verify_surface()
        if config.is_reader:
            guard.verify_read_only_attach_contract()
            await attach_lightrag_storages_read_only(lightrag, config=config)
        else:
            await lightrag.initialize_storages()
        if config.storage.lightrag.vector_storage == "PGVectorStorage":
            await guard.verify_all()
        else:
            await guard.verify_all(vector_storage=config.storage.lightrag.vector_storage)
        foundation = PGPartitionFoundation()
        if config.is_reader:
            # Readers never issue DDL: validate the complete partitioned
            # retrieval contract the writer is required to have established.
            await foundation.verify_tables(specs=lightrag_retrieval_table_specs(lightrag))
        else:
            # LightRAG just created its empty upstream tables; convert them to
            # workspace-partitioned parents before deriving DlightRAG indexes.
            # The chunk scope index is added immediately after conversion and
            # the complete contract is then verified below.
            await foundation.ensure_tables(
                specs=lightrag_retrieval_table_specs(
                    lightrag,
                    require_chunk_scope_index=False,
                )
            )
        if not config.is_reader:
            # LightRAG owns the table; DlightRAG adds its derived presentation
            # index only after the writer has established the upstream schema.
            await PGFilePanelStore().ensure_page_index()

        metadata_index = PGMetadataIndex(workspace=config.deployment.workspace)
        await metadata_index.initialize(validate_only=config.is_reader)

        chunks = PGCorpusChunkStore(
            lightrag,
            exact_threshold=config.corpus.retrieval.metadata_filter_exact_vector_threshold,
        )
        if not config.is_reader:
            # Scope preflight needs this chunk-side semi-join index regardless
            # of whether the optional BM25 retrieval leg is enabled.
            await chunks.ensure_document_scope_index()
            await foundation.verify_tables(specs=lightrag_retrieval_table_specs(lightrag))
        filtered_vectors = (
            PGFilteredVectorSearch(
                lightrag.chunks_vdb,
                exact_threshold=config.corpus.retrieval.metadata_filter_exact_vector_threshold,
            )
            if lightrag.chunks_vdb is not None
            and config.storage.lightrag.vector_storage == "PGVectorStorage"
            else None
        )
        if filtered_vectors is not None and not config.is_reader:
            await filtered_vectors.ensure_document_scope_index()

        profiles = (
            profiles_from_config(config.corpus.retrieval.bm25_profiles)
            if config.corpus.retrieval.bm25_enabled
            else ()
        )
        bm25 = await create_postgres_bm25(
            config,
            profiles=profiles or None,
        )
        logger.info(
            "Corpus runtime stores attached (PostgreSQL metadata%s)",
            ", validated" if config.is_reader else "",
        )
        return WorkspaceCorpusStores(
            metadata_index=metadata_index,
            chunks=chunks,
            filtered_vectors=filtered_vectors,
            bm25=bm25,
            bm25_languages=profile_languages(profiles),
            scoped_chunk_reader=chunks,
            doc_status_lookup=PGDocStatusLookup(workspace=config.deployment.workspace),
        )


def apply_lightrag_environment(config: DlightragConfig) -> None:
    """Bridge typed host settings to LightRAG's environment interface."""
    config.apply_lightrag_backend_env(force=True)
    config.apply_lightrag_sidecar_env()
    config.apply_lightrag_runtime_env(force=True)


def build_pg_corpus_backend(config: DlightragConfig) -> WorkspaceCorpusBackend:
    """Translate one root config into one coherent PostgreSQL corpus backend."""
    apply_lightrag_environment(config)
    # pg_trgm backs the literal filename-substring index on the metadata table
    # and is required for every corpus, independent of BM25.
    required_extensions: tuple[str, ...] = ("pg_trgm",)
    if config.corpus.retrieval.bm25_enabled:
        required_extensions = required_extensions + required_postgres_extensions(
            profiles_from_config(config.corpus.retrieval.bm25_profiles)
        )
    connection_kwargs = config.pg_connection_kwargs()
    return WorkspaceCorpusBackend(
        workspace_id=config.deployment.workspace,
        read_only=config.is_reader,
        coordination=PGCorpusCoordination(
            connection_kwargs=connection_kwargs,
            workspace=config.deployment.workspace,
            reader=config.is_reader,
            require_halfvec=(
                config.storage.lightrag.vector_storage == "PGVectorStorage"
                and config.storage.lightrag.vector_index_type == "HNSW_HALFVEC"
            ),
            required_extensions=required_extensions,
            lightrag_pool_max_size=config.storage.postgres.lightrag_pool_max_size,
            domain_pool_max_size=config.storage.postgres.pool_max_size,
            acquire_timeout=config.storage.postgres.acquire_timeout,
        ),
        maintenance=PGCorpusMaintenanceStore(connection_kwargs),
        runtime=PGCorpusRuntimeBinder(config),
        promotion=_build_promotion_worker(config),
    )


def _build_promotion_worker(config: DlightragConfig) -> PGPromotionWorker | None:
    """Build the promotion worker for writer roles; readers never claim DDL.

    The worker runs even while both thresholds are disabled: it reclaims
    expired leases and finishes reconciliations, but nothing enqueues new
    jobs in that configuration.
    """
    if config.is_reader or config.storage.lightrag.vector_storage != "PGVectorStorage":
        return None
    promotion = config.corpus.promotion
    return PGPromotionWorker(
        job_store=PGPromotionJobStore(),
        registry=PGWorkspaceRegistry(),
        lease_seconds=promotion.lease_seconds,
        retry_backoff_seconds=promotion.retry_backoff_seconds,
        claim_poll_seconds=promotion.claim_poll_seconds,
    )


class PGReadinessProbe:
    """Probe only the writable PostgreSQL Operational State authority."""

    def __init__(self, config: DlightragConfig) -> None:
        # Keep the config parameter as the composition contract; corpus role is
        # intentionally irrelevant to control-plane readiness.
        self._service_role = config.deployment.service_role

    async def __call__(self) -> str | None:
        try:
            read_only = await pg_pool.run_once(
                lambda conn: conn.fetchval("SHOW transaction_read_only")
            )
            if str(read_only).lower() != "off":
                raise RuntimeError("domain pool session is read-only")
        except Exception:
            logger.warning("Domain PostgreSQL readiness probe failed", exc_info=True)
            return "DlightRAG domain database session is not writable"

        return None


__all__ = [
    "build_pg_corpus_backend",
    "lightrag_retrieval_table_specs",
    "PGCorpusCoordination",
    "PGCorpusMaintenanceStore",
    "PGCorpusRuntimeBinder",
    "PGReadinessProbe",
    "apply_lightrag_environment",
    "verify_lightrag_storage_configuration",
]
