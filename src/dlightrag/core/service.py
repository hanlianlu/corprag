# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAG Service - High-level facade for ingestion and retrieval.

Slim RAGService that uses a SINGLE RAGAnything instance, composing
IngestionPipeline and RetrievalEngine. No external backend dependencies.
Uses PostgreSQL advisory locks instead of Redis for distributed
initialization coordination.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import platform
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from dlightrag.config import DlightragConfig, get_config

logger = logging.getLogger(__name__)

# Init lock key for PostgreSQL advisory lock (arbitrary 64-bit int)
_PG_INIT_LOCK_KEY = 0x436F727072616700  # "Dlightrag\0" as int


def _detect_mineru_backend(manual_override: str | None = None) -> str:
    """Detect optimal MinerU parsing backend based on hardware."""
    if manual_override:
        logger.info(f"MinerU backend: {manual_override} (manual override)")
        return manual_override

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"MinerU backend: hybrid-auto-engine (CUDA GPU detected: {device_name})")
            return "hybrid-auto-engine"

        if (
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and torch.backends.mps.is_available()
        ):
            logger.info("MinerU backend: Apple Silicon detected")
            return "pipeline"  # Fallback to pipeline for now

    except ImportError:
        logger.debug("torch not available, skipping GPU detection")
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")

    logger.info("MinerU backend: pipeline (fallback, no GPU acceleration detected)")
    return "pipeline"


def _ensure_venv_in_path() -> None:
    """Add venv bin to PATH for MinerU CLI."""
    venv_bin = Path(sys.executable).parent
    current_path = os.environ.get("PATH", "")
    if str(venv_bin) not in current_path.split(":"):
        os.environ["PATH"] = f"{venv_bin}:{current_path}" if current_path else str(venv_bin)


_ensure_venv_in_path()

# Inject custom VLM prompts before importing RAGAnything
from dlightrag.models.prompts import inject_custom_prompts  # noqa: E402

inject_custom_prompts()

from raganything import RAGAnything, RAGAnythingConfig  # noqa: E402

from dlightrag.core.ingestion.pipeline import IngestionPipeline  # noqa: E402
from dlightrag.core.retrieval.engine import (  # noqa: E402
    RetrievalEngine,
    RetrievalResult,
)
from dlightrag.models.llm import (  # noqa: E402
    get_embedding_func,
    get_llm_model_func,
    get_rerank_func,
    get_vision_model_func,
)


class RAGService:
    """High-level RAG service facade.

    Responsibilities:
    - Expose simple API for agents and external modules
    - Coordinate ingestion and retrieval pipelines
    - Manage RAG instances and configuration

    Usage:
        # Production: use async factory method
        rag = await RAGService.create(config=my_config, enable_vlm=True)

        # With callbacks:
        rag = await RAGService.create(
            config=my_config,
            cancel_checker=my_cancel_fn,
            url_transformer=my_url_fn,
        )
    """

    @classmethod
    async def create(
        cls,
        config: DlightragConfig | None = None,
        enable_vlm: bool = True,
        cancel_checker: Callable[[], Awaitable[bool]] | None = None,
        url_transformer: Callable[[str], str] | None = None,
    ) -> RAGService:
        """Async factory method - creates and initializes RAGService."""
        instance = cls(
            config=config,
            enable_vlm=enable_vlm,
            cancel_checker=cancel_checker,
            url_transformer=url_transformer,
        )
        await instance.initialize()
        return instance

    def __init__(
        self,
        config: DlightragConfig | None = None,
        enable_vlm: bool = True,
        cancel_checker: Callable[[], Awaitable[bool]] | None = None,
        url_transformer: Callable[[str], str] | None = None,
    ) -> None:
        """Store configuration only. Use RAGService.create() for full initialization."""
        self.config = config or get_config()
        self.enable_vlm = enable_vlm
        self.enable_rerank = self.config.enable_rerank
        self._initialized: bool = False

        # Callbacks for decoupled integration
        self._cancel_checker = cancel_checker
        self._url_transformer = url_transformer

        # Single RAGAnything instance + composed pipelines (created lazily in initialize())
        self.rag: RAGAnything | None = None
        self.ingestion: IngestionPipeline | None = None
        self.retrieval: RetrievalEngine | None = None

    def _unregister_atexit_cleanup(self, rag_obj: Any) -> None:
        """Prevent double-close logging errors by removing raganything atexit hooks."""
        try:
            atexit.unregister(rag_obj.close)
        except Exception:  # noqa: BLE001
            logger.debug("Unable to unregister atexit hook for %s", rag_obj)

    async def initialize(self) -> None:
        """Initialize LightRAG storages and caches (idempotent).

        Uses PostgreSQL advisory lock for distributed coordination when
        using PG storage backends, or proceeds directly otherwise.
        """
        if self._initialized:
            return

        if self.config.kv_storage.startswith("PG"):
            await self._initialize_with_pg_lock()
        else:
            await self._do_initialize()

        self._initialized = True
        logger.debug("RAGService initialized")

    async def _initialize_with_pg_lock(self) -> None:
        """Initialize with PostgreSQL advisory lock for multi-worker safety."""
        try:
            import asyncpg
        except ImportError:
            logger.warning("asyncpg not available, proceeding without distributed lock")
            await self._do_initialize()
            return

        try:
            conn = await asyncpg.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
                database=self.config.postgres_database,
            )
        except Exception as e:
            logger.warning(f"PostgreSQL unavailable for init lock, proceeding without: {e}")
            await self._do_initialize()
            return

        try:
            acquired = await conn.fetchval("SELECT pg_try_advisory_lock($1)", _PG_INIT_LOCK_KEY)

            if acquired:
                logger.info("Acquired PG advisory lock, initializing RAG pipelines...")
                try:
                    await self._ensure_pg_schema(conn)
                    await self._do_initialize()
                    logger.info("RAG pipelines initialized successfully")
                finally:
                    await conn.execute("SELECT pg_advisory_unlock($1)", _PG_INIT_LOCK_KEY)
            else:
                logger.info("Another worker is initializing, waiting for lock...")
                for _ in range(180):
                    await asyncio.sleep(1)
                    if await conn.fetchval("SELECT pg_try_advisory_lock($1)", _PG_INIT_LOCK_KEY):
                        await conn.execute("SELECT pg_advisory_unlock($1)", _PG_INIT_LOCK_KEY)
                        break
                logger.info("Lock released, connecting to existing storages...")
                await self._do_initialize()
        finally:
            await conn.close()

    async def _ensure_pg_schema(self, conn) -> None:
        """Ensure required PostgreSQL extensions and tables exist (idempotent).

        All statements use IF NOT EXISTS so this is safe to run on every startup.
        Extensions are best-effort (may need superuser -- Docker init.sql handles
        this; remote PG may have them pre-installed). Table creation is critical.
        """
        # Best-effort: extensions need superuser; Docker init.sql handles this,
        # remote PG users with limited privileges must pre-install them.
        for sql in [
            "CREATE EXTENSION IF NOT EXISTS vector",
            "CREATE EXTENSION IF NOT EXISTS age",
            "LOAD 'age'",
            'SET search_path = ag_catalog, "$user", public',
        ]:
            try:
                await conn.execute(sql)
            except Exception as e:
                logger.warning(f"Extension setup skipped (may need superuser): {sql[:50]}... -- {e}")

        # Verify extensions are actually available (regardless of who installed them)
        installed = {r["extname"] for r in await conn.fetch("SELECT extname FROM pg_extension")}
        missing = {"vector", "age"} - installed
        if missing:
            raise RuntimeError(
                f"Required PostgreSQL extensions not installed: {missing}. "
                "Ask your DBA to run: "
                + "; ".join(f"CREATE EXTENSION {ext}" for ext in sorted(missing))
            )

        # Critical: table must exist for hash-based deduplication
        await conn.execute("""CREATE TABLE IF NOT EXISTS dlightrag_file_hashes (
            content_hash TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            workspace TEXT NOT NULL DEFAULT 'default',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )""")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_hashes_doc_id ON dlightrag_file_hashes(doc_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_hashes_workspace ON dlightrag_file_hashes(workspace)"
        )
        logger.info("PostgreSQL schema ensured")

    async def _do_initialize(self) -> None:
        """Create a single RAGAnything object and compose pipelines."""
        config = self.config

        # Detect optimal MinerU backend based on hardware (only for mineru parser)
        mineru_backend = None
        if config.parser == "mineru":
            mineru_backend = _detect_mineru_backend(config.mineru_backend)
        else:
            logger.info(f"Using {config.parser} parser, MinerU backend detection skipped")

        # Configure RAGAnything
        rag_config = RAGAnythingConfig(
            working_dir=str(config.working_dir_path),
            max_concurrent_files=config.max_concurrent_ingestion,
            parser=config.parser,
            parse_method=config.parse_method,
            enable_image_processing=config.enable_image_processing,
            enable_table_processing=config.enable_table_processing,
            enable_equation_processing=config.enable_equation_processing,
            display_content_stats=config.display_content_stats,
            use_full_path=config.use_full_path,
            context_window=config.context_window,
            context_filter_content_types=config.context_filter_types.split(","),
            max_context_tokens=config.max_context_tokens,
        )

        # Get model functions
        llm_func = get_llm_model_func(config)
        vision_func = get_vision_model_func(config) if self.enable_vlm else None
        embedding_func = get_embedding_func(config)
        rerank_func = get_rerank_func(config)

        # LightRAG configuration
        lightrag_kwargs: dict[str, Any] = {
            "workspace": config.workspace,
            "default_llm_timeout": config.llm_request_timeout,
            "chunk_token_size": config.chunk_size,
            "chunk_overlap_token_size": config.chunk_overlap,
            "max_parallel_insert": config.max_parallel_insert,
            "llm_model_max_async": config.max_async,
            "embedding_func_max_async": config.embedding_func_max_async,
            "embedding_batch_num": config.embedding_batch_num,
            "vector_storage": config.vector_storage,
            "graph_storage": config.graph_storage,
            "kv_storage": config.kv_storage,
            "doc_status_storage": config.doc_status_storage,
            "rerank_model_func": rerank_func,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.3,
            },
            "addon_params": {
                "entity_types": config.kg_entity_types,
                "language": "English",
            },
        }

        # ONE RAGAnything with unified llm_func (chat model)
        logger.info("Creating RAGAnything instance...")
        self.rag = RAGAnything(
            None,
            llm_func,
            vision_func,
            embedding_func,
            rag_config,
            lightrag_kwargs,
        )
        self._unregister_atexit_cleanup(self.rag)

        # Initialize LightRAG storages
        if hasattr(self.rag, "_ensure_lightrag_initialized"):
            init_result = await self.rag._ensure_lightrag_initialized()
            if isinstance(init_result, dict) and not init_result.get("success", True):
                error_msg = init_result.get("error", "LightRAG initialization failed")
                logger.error(f"LightRAG initialization failed: {error_msg}")
                raise RuntimeError(f"LightRAG initialization failed: {error_msg}")

        # Auto-detect hash index backend based on storage config
        hash_index = await self._create_hash_index(config)

        # Compose pipelines
        self.ingestion = IngestionPipeline(
            self.rag,
            config=config,
            max_concurrent=config.max_concurrent_ingestion,
            mineru_backend=mineru_backend,
            cancel_checker=self._cancel_checker,
            hash_index=hash_index,
        )

        self.retrieval = RetrievalEngine(rag=self.rag, config=config)
        self.retrieval._url_transformer = self._url_transformer

        logger.info("RAG pipelines initialized successfully")

    async def _create_hash_index(self, config: DlightragConfig) -> Any:
        """Create the appropriate hash index backend based on KV storage config.

        Uses the same backend as the configured KV storage for consistency.
        Falls back to JSON file-based HashIndex if the backend package is unavailable.
        """
        kv = config.kv_storage

        if kv.startswith("PG"):
            try:
                from dlightrag.core.ingestion.hash_index import PGHashIndex

                idx = PGHashIndex(workspace=config.workspace)
                await idx.initialize()
                logger.info("Hash index: PGHashIndex (PostgreSQL via shared pool)")
                return idx
            except ImportError:
                logger.warning("asyncpg not available, falling back to JSON HashIndex")
            except Exception as e:
                logger.warning(f"PGHashIndex creation failed, falling back to JSON: {e}")

        elif kv.startswith("Redis"):
            try:
                from dlightrag.core.ingestion.hash_index import RedisHashIndex

                idx = RedisHashIndex(workspace=config.workspace)
                await idx.initialize()
                logger.info("Hash index: RedisHashIndex (Redis via shared pool)")
                return idx
            except ImportError:
                logger.warning("redis not available, falling back to JSON HashIndex")
            except Exception as e:
                logger.warning(f"RedisHashIndex creation failed, falling back to JSON: {e}")

        elif kv.startswith("Mongo"):
            try:
                from dlightrag.core.ingestion.hash_index import MongoHashIndex

                idx = MongoHashIndex(workspace=config.workspace)
                await idx.initialize()
                logger.info("Hash index: MongoHashIndex (MongoDB via shared client)")
                return idx
            except ImportError:
                logger.warning("motor not available, falling back to JSON HashIndex")
            except Exception as e:
                logger.warning(f"MongoHashIndex creation failed, falling back to JSON: {e}")

        from dlightrag.core.ingestion.hash_index import HashIndex

        logger.info("Hash index: HashIndex (JSON file)")
        return HashIndex(config.working_dir_path, workspace=config.workspace)

    def _ensure_initialized(self) -> None:
        """Raise error if not initialized."""
        if not self._initialized:
            raise RuntimeError(
                "RAGService not initialized. Use 'await RAGService.create()' instead."
            )

    async def close(self) -> None:
        """Clean up storages (best-effort)."""
        if self.rag is not None:
            try:
                await self.rag.finalize_storages()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to finalize storages", exc_info=True)

    # === INGESTION API ===

    async def aingest(
        self,
        source_type: Literal["local", "azure_blob", "snowflake"],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Unified ingestion API.

        Args:
            source_type: "local", "azure_blob", or "snowflake"
            kwargs:
                local: path, replace
                azure_blob: source, container_name, blob_path, prefix, replace
                snowflake: query, table
        """
        self._ensure_initialized()

        if not self.ingestion:
            raise RuntimeError("Ingestion pipeline not initialized")

        ingestion = self.ingestion

        # Get replace default from config if not explicitly set
        def get_replace_value() -> bool:
            replace_arg = kwargs.get("replace")
            if replace_arg is None:
                return self.config.ingestion_replace_default
            return bool(replace_arg)

        if source_type == "local":
            path = Path(kwargs["path"])
            replace = get_replace_value()

            logger.info(f"Ingesting from local path: {path}")

            result = await ingestion.aingest_from_local(
                path=path,
                replace=replace,
            )
            return result.model_dump(exclude_none=True)

        if source_type == "azure_blob":
            container_name = kwargs["container_name"]
            blob_path = kwargs.get("blob_path")
            prefix = kwargs.get("prefix")
            replace = get_replace_value()
            source = kwargs.get("source")

            if source is None:
                from dlightrag.sourcing.azure_blob import AzureBlobDataSource

                source = AzureBlobDataSource(
                    container_name=container_name,
                    connection_string=self.config.blob_connection_string,
                )

            # Default to entire container if neither blob_path nor prefix provided
            if blob_path is None and prefix is None:
                prefix = ""

            try:
                result = await ingestion.aingest_from_azure_blob(
                    source=source,
                    container_name=container_name,
                    blob_path=blob_path,
                    prefix=prefix,
                    replace=replace,
                )
                return result.model_dump(exclude_none=True)
            finally:
                if hasattr(source, "aclose"):
                    await source.aclose()

        # source_type == "snowflake"
        ingestion_result = await ingestion.aingest_from_snowflake(
            query=kwargs["query"],
            table=kwargs.get("table"),
        )
        return ingestion_result.model_dump(exclude_none=True)

    # === RETRIEVAL API ===

    async def aretrieve(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] | None = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        is_reretrieve: bool = False,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve structured data without generating answer.

        Args:
            query: User query
            multimodal_content: Optional multimodal content
            mode: Retrieval mode (local/global/hybrid/naive/mix)
            top_k: Initial vector retrieval count
            chunk_top_k: Final chunk count after round-robin
            is_reretrieve: Whether this is a re-retrieve (disables internal rerank)

        Returns:
            RetrievalResult with contexts and raw retrieval data (no answer).
        """
        self._ensure_initialized()

        if not self.retrieval:
            raise RuntimeError("Retrieval engine not initialized")

        return await self.retrieval.aretrieve(
            query,
            multimodal_content=multimodal_content,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            is_reretrieve=is_reretrieve,
            **kwargs,
        )

    async def aanswer(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] | None = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve contexts and generate an LLM answer.

        Returns the same structured data as aretrieve() but with the
        ``answer`` field populated by the LLM.
        """
        self._ensure_initialized()

        if not self.retrieval:
            raise RuntimeError("Retrieval engine not initialized")

        return await self.retrieval.aanswer(
            query,
            multimodal_content=multimodal_content,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            **kwargs,
        )

    async def _rerank_chunks(
        self,
        chunks: list[dict],
        query: str,
    ) -> list[dict]:
        """Rerank chunks by relevance. Only sorts, never truncates."""
        if not chunks:
            return []

        rerank_func = get_rerank_func(self.config)
        chunk_texts = [c.get("content", "") for c in chunks]

        try:
            rerank_results = await rerank_func(
                query=query,
                documents=chunk_texts,
                domain_knowledge=self.config.domain_knowledge_hints or None,
            )

            reranked = [
                chunks[item["index"]] for item in rerank_results if item["index"] < len(chunks)
            ]

            logger.info(f"[Rerank] Sorted {len(reranked)} chunks by relevance")
            return reranked

        except Exception as e:
            logger.warning(f"[Rerank] Failed: {e}, using original order")
            return chunks

    # === FILE MANAGEMENT API ===

    async def alist_ingested_files(self) -> list[dict[str, Any]]:
        """List all ingested files."""
        self._ensure_initialized()
        if not self.ingestion:
            raise RuntimeError("Ingestion pipeline not initialized")
        return await self.ingestion.alist_ingested_files()

    async def adelete_files(
        self,
        *,
        file_paths: list[str] | None = None,
        filenames: list[str] | None = None,
        delete_source: bool = True,
    ) -> list[dict[str, Any]]:
        """Unified file deletion."""
        self._ensure_initialized()
        if not self.ingestion:
            raise RuntimeError("Ingestion pipeline not initialized")
        return await self.ingestion.adelete_files(
            file_paths=file_paths,
            filenames=filenames,
            delete_source=delete_source,
        )


__all__ = ["RAGService"]
