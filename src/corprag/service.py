# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAG Service - High-level facade for ingestion and retrieval.

Self-contained service with CorpragConfig injection. No external backend
dependencies. Uses PostgreSQL advisory locks instead of Redis for
distributed initialization coordination.
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

from corprag.config import CorpragConfig, get_config

logger = logging.getLogger(__name__)

# Init lock key for PostgreSQL advisory lock (arbitrary 64-bit int)
_PG_INIT_LOCK_KEY = 0x436F727072616700  # "Corprag\0" as int


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
            return "pipeline"

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
from corprag.models.prompts import inject_custom_prompts  # noqa: E402

inject_custom_prompts()

from raganything import RAGAnything, RAGAnythingConfig  # noqa: E402

from corprag.ingestion.pipeline import IngestionPipeline  # noqa: E402
from corprag.models.llm import (  # noqa: E402
    get_embedding_func,
    get_ingestion_llm_model_func,
    get_llm_model_func,
    get_rerank_func,
    get_vision_model_func,
)
from corprag.retrieval.engine import (  # noqa: E402
    EnhancedRAGAnything,
    RetrievalResult,
    augment_retrieval_result,
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
        config: CorpragConfig | None = None,
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
        config: CorpragConfig | None = None,
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

        # Pipelines (created lazily in initialize())
        self.ingestion: IngestionPipeline | None = None
        self.rag_text: EnhancedRAGAnything | None = None
        self.rag_vision: EnhancedRAGAnything | None = None

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
        Extensions are best-effort (may need superuser — Docker init.sql handles
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
                logger.warning(f"Extension setup skipped (may need superuser): {sql[:50]}… — {e}")

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
        await conn.execute("""CREATE TABLE IF NOT EXISTS corprag_file_hashes (
            content_hash TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            workspace TEXT NOT NULL DEFAULT 'default',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )""")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_hashes_doc_id ON corprag_file_hashes(doc_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_hashes_workspace ON corprag_file_hashes(workspace)"
        )
        logger.info("PostgreSQL schema ensured")

    async def _do_initialize(self) -> None:
        """Create RAGAnything objects and initialize storages."""
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
        ingestion_llm_func = get_ingestion_llm_model_func(config)
        vision_func = get_vision_model_func(config) if self.enable_vlm else None
        embedding_func = get_embedding_func(config)
        rerank_func = get_rerank_func(config)

        # LightRAG configuration
        lightrag_kwargs: dict[str, Any] = {
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

        # Create ingestion pipeline
        logger.info("Creating ingestion pipeline with RAGAnything...")
        rag_ingestion = RAGAnything(
            None,
            ingestion_llm_func,
            vision_func,
            embedding_func,
            rag_config,
            lightrag_kwargs,
        )
        self._unregister_atexit_cleanup(rag_ingestion)
        self.ingestion = IngestionPipeline(
            rag_ingestion,
            config=config,
            max_concurrent=config.max_concurrent_ingestion,
            mineru_backend=mineru_backend,
            cancel_checker=self._cancel_checker,
        )

        # Initialize LightRAG storages
        if hasattr(rag_ingestion, "_ensure_lightrag_initialized"):
            init_result = await rag_ingestion._ensure_lightrag_initialized()
            if isinstance(init_result, dict) and not init_result.get("success", True):
                error_msg = init_result.get("error", "LightRAG initialization failed")
                logger.error(f"LightRAG initialization failed: {error_msg}")
                raise RuntimeError(f"LightRAG initialization failed: {error_msg}")

        # Create retrieval pipelines (share LightRAG instance)
        shared_lightrag = getattr(rag_ingestion, "lightrag", None)

        logger.info("Creating text retrieval pipeline...")
        self.rag_text = EnhancedRAGAnything(
            shared_lightrag,
            llm_func,
            None,
            embedding_func,
            rag_config,
        )
        self._unregister_atexit_cleanup(self.rag_text)

        if self.enable_vlm and vision_func:
            logger.info("Creating vision retrieval pipeline...")
            self.rag_vision = EnhancedRAGAnything(
                shared_lightrag,
                llm_func,
                vision_func,
                embedding_func,
                rag_config,
            )
            self._unregister_atexit_cleanup(self.rag_vision)
        else:
            self.rag_vision = self.rag_text
            if self.enable_vlm and not vision_func:
                logger.warning(
                    "enable_vlm=True but no vision model configured; using text-only retrieval."
                )

        logger.info("All RAG pipelines initialized successfully")

    def _ensure_initialized(self) -> None:
        """Raise error if not initialized."""
        if not self._initialized:
            raise RuntimeError(
                "RAGService not initialized. Use 'await RAGService.create()' instead."
            )

    async def close(self) -> None:
        """Clean up storages (best-effort)."""
        if self.ingestion is not None:
            try:
                await self.ingestion.rag.finalize_storages()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to finalize ingestion storages", exc_info=True)

        for rag_obj in (self.rag_text, self.rag_vision):
            if rag_obj is None:
                continue
            try:
                if hasattr(rag_obj, "finalize_storages"):
                    await rag_obj.finalize_storages()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to finalize retrieval storages", exc_info=True)

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
                local: path, replace, sync_hashes
                azure_blob: source, container_name, blob_path, prefix, replace, sync_hashes
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

        sync_hashes = bool(kwargs.get("sync_hashes", False))

        if source_type == "local":
            path = Path(kwargs["path"])
            replace = get_replace_value()

            logger.info(f"Ingesting from local path: {path}")

            result = await ingestion.aingest_from_local(
                path=path,
                replace=replace,
                sync_hashes=sync_hashes,
            )
            return result.model_dump(exclude_none=True)

        if source_type == "azure_blob":
            container_name = kwargs["container_name"]
            blob_path = kwargs.get("blob_path")
            prefix = kwargs.get("prefix")
            replace = get_replace_value()
            source = kwargs.get("source")

            if source is None:
                from corprag.sourcing.azure_blob import AzureBlobDataSource

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
                    sync_hashes=sync_hashes,
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

        if not self.rag_text or not self.rag_vision:
            raise RuntimeError("RAG instances not initialized")

        rag = self.rag_vision if multimodal_content else self.rag_text

        adjusted_top_k = top_k or self.config.top_k
        adjusted_chunk_top_k = chunk_top_k or self.config.chunk_top_k

        if is_reretrieve:
            enable_rerank = False
        else:
            enable_rerank = kwargs.pop("enable_rerank", self.enable_rerank)

        query_kwargs = {
            "top_k": adjusted_top_k,
            "chunk_top_k": adjusted_chunk_top_k,
            "enable_rerank": enable_rerank,
            "max_entity_tokens": kwargs.pop(
                "max_entity_tokens",
                self.config.max_entity_tokens,
            ),
            "max_relation_tokens": kwargs.pop(
                "max_relation_tokens",
                self.config.max_relation_tokens,
            ),
            "max_total_tokens": kwargs.pop(
                "max_total_tokens",
                self.config.max_total_tokens,
            ),
            **kwargs,
        }

        result = await rag.aquery_data_with_multimodal(
            query=query,
            multimodal_content=multimodal_content,
            mode=mode or self.config.default_mode,
            **query_kwargs,
        )

        return augment_retrieval_result(
            result,
            str(self.config.working_dir_path),
            url_transformer=self._url_transformer,
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

        if not self.rag_text or not self.rag_vision:
            raise RuntimeError("RAG instances not initialized")

        rag = self.rag_vision if multimodal_content else self.rag_text

        adjusted_top_k = top_k or self.config.top_k
        adjusted_chunk_top_k = chunk_top_k or self.config.chunk_top_k
        enable_rerank = kwargs.pop("enable_rerank", self.enable_rerank)

        # Truncate conversation history: first by turns, then by token budget
        history = kwargs.pop("conversation_history", None)
        if history:
            max_msgs = self.config.max_conversation_turns * 2
            if len(history) > max_msgs:
                history = history[-max_msgs:]

            token_budget = self.config.max_conversation_tokens
            total = 0
            cutoff = 0
            for i in range(len(history) - 1, -1, -1):
                # ~4 chars per token is a safe approximation
                total += len(history[i].get("content", "")) // 4
                if total > token_budget:
                    cutoff = i + 1
                    break
            if cutoff:
                history = history[cutoff:]

            kwargs["conversation_history"] = history

        query_kwargs = {
            "top_k": adjusted_top_k,
            "chunk_top_k": adjusted_chunk_top_k,
            "enable_rerank": enable_rerank,
            "max_entity_tokens": kwargs.pop(
                "max_entity_tokens",
                self.config.max_entity_tokens,
            ),
            "max_relation_tokens": kwargs.pop(
                "max_relation_tokens",
                self.config.max_relation_tokens,
            ),
            "max_total_tokens": kwargs.pop(
                "max_total_tokens",
                self.config.max_total_tokens,
            ),
            **kwargs,
        }

        result = await rag.aquery_llm_with_multimodal(
            query=query,
            multimodal_content=multimodal_content,
            mode=mode or self.config.default_mode,
            **query_kwargs,
        )

        return augment_retrieval_result(
            result,
            str(self.config.working_dir_path),
            url_transformer=self._url_transformer,
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

    def list_ingested_files(self) -> list[dict[str, Any]]:
        """List all ingested files."""
        self._ensure_initialized()
        if not self.ingestion:
            raise RuntimeError("Ingestion pipeline not initialized")
        return self.ingestion.list_ingested_files()

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
