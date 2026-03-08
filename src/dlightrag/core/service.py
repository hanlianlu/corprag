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
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from dlightrag.core.ingestion.hash_index import HashIndexProtocol

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

# RAGAnything is only needed for caption mode — make import optional
# so unified mode doesn't require raganything dependencies.
try:
    from raganything import RAGAnything, RAGAnythingConfig  # noqa: E402

    _HAS_RAGANYTHING = True
except ImportError:
    _HAS_RAGANYTHING = False

from lightrag.utils import EmbeddingFunc  # noqa: E402

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

        # Caption mode (Mode 1): RAGAnything + composed pipelines
        self.rag: Any = None  # RAGAnything (caption mode)
        self.ingestion: IngestionPipeline | None = None
        self.retrieval: RetrievalEngine | None = None

        # Unified mode (Mode 2): direct LightRAG + UnifiedRepresentEngine
        self.unified: Any = None  # UnifiedRepresentEngine
        self._lightrag: Any = None  # Direct LightRAG reference (unified mode)
        self._visual_chunks: Any = None  # Visual chunks KV store (unified mode)

    @property
    def lightrag(self) -> Any:
        """Return the underlying LightRAG instance regardless of mode.

        - Unified mode: ``self._lightrag`` (created directly)
        - Caption mode: ``self.rag.lightrag`` (via RAGAnything)
        """
        return self._lightrag or getattr(self.rag, "lightrag", None)

    @staticmethod
    def _build_vector_db_kwargs(config: DlightragConfig) -> dict[str, Any]:
        """Build vector_db_storage_cls_kwargs from config.vector_db_kwargs passthrough."""
        kwargs: dict[str, Any] = {"cosine_better_than_threshold": 0.3}
        kwargs.update(config.vector_db_kwargs)
        return kwargs

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
                logger.warning(
                    f"Extension setup skipped (may need superuser): {sql[:50]}... -- {e}"
                )

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
        """Create RAG backend and compose pipelines based on rag_mode."""
        config = self.config

        if config.rag_mode == "unified":
            await self._do_initialize_unified()
            return

        # --- Caption mode (existing code, unchanged) ---
        if not _HAS_RAGANYTHING:
            raise ImportError(
                "raganything is required for caption mode. Install it with: pip install raganything"
            )

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
            "default_embedding_timeout": config.embedding_request_timeout,
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
            "vector_db_storage_cls_kwargs": self._build_vector_db_kwargs(config),
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

        logger.info("RAG pipelines initialized successfully (caption mode)")

    async def _do_initialize_unified(self) -> None:
        """Initialize unified representational RAG mode (Mode 2).

        Creates LightRAG directly (no RAGAnything), sets up visual_chunks
        KV store, and creates UnifiedRepresentEngine.
        """
        import dataclasses

        from lightrag import LightRAG

        from dlightrag.unifiedrepresent.engine import UnifiedRepresentEngine

        config = self.config
        logger.info("Initializing unified representational RAG mode...")

        # Get model functions
        llm_func = get_llm_model_func(config)
        vision_func = get_vision_model_func(config) if self.enable_vlm else None

        # Use httpx_text_embed instead of LightRAG's openai_embed for text
        # embedding.  openai_embed uses encoding_format:"base64" and the openai
        # Python client — both fail with Xinference VL models.
        # partial() with plain strings is deepcopy-safe (LightRAG's __post_init__
        # does asdict→deepcopy on embedding_func).
        from functools import partial

        from dlightrag.unifiedrepresent.embedder import (
            OpenAICompatProvider,
            VisualEmbedder,
            VoyageProvider,
            httpx_text_embed,
        )

        emb_provider = config.effective_embedding_provider
        emb_base_url = config._get_url(f"{emb_provider}_base_url") or ""
        emb_api_key = config._get_provider_api_key(emb_provider) or ""

        if emb_provider == "voyage":
            embed_provider = VoyageProvider()
        else:
            embed_provider = OpenAICompatProvider()

        embedding_func = EmbeddingFunc(
            embedding_dim=config.embedding_dim,
            max_token_size=8192,
            func=partial(
                httpx_text_embed,
                model=config.embedding_model,
                base_url=emb_base_url,
                api_key=emb_api_key,
                provider=embed_provider,
            ),
        )

        # VisualEmbedder for image embedding (reuses persistent httpx client)
        visual_embedder = VisualEmbedder(
            model=config.embedding_model,
            base_url=emb_base_url,
            api_key=emb_api_key,
            dim=config.embedding_dim,
            batch_size=config.embedding_func_max_async,
            provider=embed_provider,
        )

        # LightRAG configuration (same storage backends as caption mode)
        # Do NOT pass rerank_model_func — we handle reranking ourselves
        lightrag = LightRAG(
            working_dir=str(config.working_dir_path),
            llm_model_func=llm_func,
            embedding_func=embedding_func,
            workspace=config.workspace,
            default_llm_timeout=config.llm_request_timeout,
            default_embedding_timeout=config.embedding_request_timeout,
            chunk_token_size=config.chunk_size,
            chunk_overlap_token_size=config.chunk_overlap,
            max_parallel_insert=config.max_parallel_insert,
            llm_model_max_async=config.max_async,
            embedding_func_max_async=config.embedding_func_max_async,
            embedding_batch_num=config.embedding_batch_num,
            vector_storage=config.vector_storage,
            graph_storage=config.graph_storage,
            kv_storage=config.kv_storage,
            doc_status_storage=config.doc_status_storage,
            vector_db_storage_cls_kwargs=self._build_vector_db_kwargs(config),
            addon_params={
                "entity_types": config.kg_entity_types,
                "language": "English",
            },
        )
        await lightrag.initialize_storages()
        self._lightrag = lightrag

        # Create visual_chunks KV store
        # PGKVStorage hardcodes namespace handlers and breaks on custom namespaces
        # like "visual_chunks". Use our PGJsonbKVStorage (generic JSONB table) instead.
        kv_cls = lightrag.key_string_value_json_storage_cls
        if config.kv_storage.startswith("PG"):
            from dlightrag.storage.pg_jsonb_kv import PGJsonbKVStorage

            kv_cls = PGJsonbKVStorage
        visual_chunks = kv_cls(
            namespace="visual_chunks",
            workspace=config.workspace,
            global_config=dataclasses.asdict(lightrag),
            embedding_func=embedding_func,
        )
        await visual_chunks.initialize()
        self._visual_chunks = visual_chunks

        # Create engine (pass pre-built embedder to avoid creating a duplicate)
        self.unified = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=visual_chunks,
            config=config,
            vision_model_func=vision_func,
            visual_embedder=visual_embedder,
        )

        # Create hash index for deduplication (same backend as caption mode)
        self._hash_index = await self._create_hash_index(config)

        logger.info("Unified representational RAG mode initialized")

    async def _create_hash_index(self, config: DlightragConfig) -> HashIndexProtocol:
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
        """Clean up storages and worker pools (best-effort)."""
        # Shutdown LightRAG worker pools first — they hold background asyncio
        # tasks that block asyncio.run() from exiting.
        await self._shutdown_worker_pools()

        if self.rag is not None:
            try:
                await self.rag.finalize_storages()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to finalize storages", exc_info=True)

        # Unified mode cleanup
        if self.unified is not None:
            try:
                await self.unified.aclose()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to close unified engine", exc_info=True)
        if self._visual_chunks is not None:
            try:
                await self._visual_chunks.finalize()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to finalize visual_chunks", exc_info=True)
        if self._lightrag is not None:
            try:
                await self._lightrag.finalize_storages()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to finalize LightRAG storages", exc_info=True)

    async def _shutdown_worker_pools(self) -> None:
        """Shutdown LightRAG's EmbeddingFunc/LLM worker pools.

        LightRAG wraps embedding_func and llm_model_func with
        priority_limit_async_func_call which creates background asyncio
        tasks.  finalize_storages() does NOT shut these down, so they
        block asyncio.run() from exiting.  The wrapped functions expose
        a ``.shutdown()`` coroutine we can call explicitly.
        """
        lr = self.lightrag
        if lr is None:
            return

        for attr in ("embedding_func", "llm_model_func"):
            try:
                obj = getattr(lr, attr, None)
                # EmbeddingFunc stores the wrapped func in .func
                func = getattr(obj, "func", obj)
                shutdown = getattr(func, "shutdown", None)
                if shutdown is not None:
                    await shutdown()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to shutdown %s worker pool", attr, exc_info=True)

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

        # Unified mode
        if self.unified is not None:
            if source_type == "local":
                path = Path(kwargs["path"])
                replace = kwargs.get("replace", self.config.ingestion_replace_default)

                if path.is_file():
                    return await self._unified_ingest_single_local(path, replace)

                if path.is_dir():
                    return await self._unified_ingest_local_dir(path, replace)

                raise FileNotFoundError(f"Path not found: {path}")

            if source_type == "azure_blob":
                return await self._unified_ingest_azure_blob(**kwargs)

            # source_type == "snowflake"
            return await self._unified_ingest_snowflake(**kwargs)

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

    async def _unified_ingest_single_local(self, path: Path, replace: bool) -> dict[str, Any]:
        """Unified mode: ingest a single local file with dedup."""
        should_skip, content_hash, reason = await self._hash_index.should_skip_file(path, replace)
        if should_skip:
            logger.info("Skipped (dedup): %s - %s", path.name, reason)
            return {"status": "skipped", "reason": reason, "file_path": str(path)}

        result = await self.unified.aingest(file_path=str(path))

        if content_hash and result.get("doc_id"):
            await self._hash_index.register(content_hash, result["doc_id"], str(path))

        return result

    async def _unified_ingest_local_dir(self, path: Path, replace: bool) -> dict[str, Any]:
        """Unified mode: recursively ingest all supported files in a directory."""
        from dlightrag.unifiedrepresent.renderer import (
            _IMAGE_EXTENSIONS,
            _OFFICE_EXTENSIONS,
        )

        supported = {".pdf"} | _IMAGE_EXTENSIONS | _OFFICE_EXTENSIONS

        files = sorted(f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in supported)

        if not files:
            logger.warning("No supported files found in %s", path)
            return {
                "status": "success",
                "source_type": "local",
                "folder": str(path),
                "processed": 0,
                "skipped": 0,
                "total_files": 0,
            }

        results: list[dict[str, Any]] = []
        skipped = 0
        for file_path in files:
            result = await self._unified_ingest_single_local(file_path, replace)
            if result.get("status") == "skipped":
                skipped += 1
            results.append(result)

        processed = len(results) - skipped
        logger.info(
            "Directory ingestion complete: %s (%d processed, %d skipped)",
            path,
            processed,
            skipped,
        )
        return {
            "status": "success",
            "source_type": "local",
            "folder": str(path),
            "results": results,
            "processed": processed,
            "skipped": skipped,
            "total_files": len(files),
        }

    async def _unified_ingest_azure_blob(self, **kwargs: Any) -> dict[str, Any]:
        """Unified mode: download blob(s) to temp → visual pipeline."""
        from dlightrag.sourcing.azure_blob import AzureBlobDataSource

        container_name: str = kwargs["container_name"]
        blob_path: str | None = kwargs.get("blob_path")
        prefix: str | None = kwargs.get("prefix")
        source = kwargs.get("source")

        if source is None:
            source = AzureBlobDataSource(
                container_name=container_name,
                connection_string=self.config.blob_connection_string,
            )

        replace: bool = kwargs.get("replace", self.config.ingestion_replace_default)

        if blob_path is None and prefix is None:
            prefix = ""

        try:
            if blob_path:
                return await self._unified_ingest_single_blob(source, blob_path, replace)

            # Batch: list + ingest each
            blob_ids = await source.alist_documents(prefix=prefix)
            results: list[dict[str, Any]] = []
            for bid in blob_ids:
                result = await self._unified_ingest_single_blob(source, bid, replace)
                results.append(result)

            return {
                "status": "success",
                "source_type": "azure_blob",
                "container": container_name,
                "prefix": prefix,
                "results": results,
                "processed": len(results),
            }
        finally:
            if hasattr(source, "aclose"):
                await source.aclose()

    async def _unified_ingest_single_blob(
        self, source: Any, blob_path: str, replace: bool = False
    ) -> dict[str, Any]:
        """Download one blob to temp dir, ingest via unified engine, cleanup."""
        import shutil
        import uuid

        tmpdir = self.config.temp_dir / uuid.uuid4().hex[:12]
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            target = tmpdir / Path(blob_path).name
            content = await source.aload_document(blob_path)
            await asyncio.to_thread(target.write_bytes, content)

            # Content-aware deduplication
            should_skip, content_hash, reason = await self._hash_index.should_skip_file(
                target, replace
            )
            if should_skip:
                logger.info("Skipped (dedup): %s - %s", blob_path, reason)
                return {"status": "skipped", "reason": reason, "blob_path": blob_path}

            logger.info("Downloaded blob to temp: %s", target)
            result = await self.unified.aingest(file_path=str(target))

            # Register with original blob path as file_path
            if content_hash and result.get("doc_id"):
                await self._hash_index.register(
                    content_hash, result["doc_id"], f"azure://{blob_path}"
                )

            return result
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    async def _unified_ingest_snowflake(self, **kwargs: Any) -> dict[str, Any]:
        """Unified mode: Snowflake text → LightRAG ainsert (no visual pipeline)."""
        from dlightrag.sourcing.snowflake import SnowflakeDataSource

        config = self.config
        query: str = kwargs["query"]
        table: str | None = kwargs.get("table")
        source_label = table or "query"

        source = await asyncio.to_thread(
            SnowflakeDataSource,
            account=config.snowflake_account or "",
            user=config.snowflake_user or "",
            password=config.snowflake_password or "",
            warehouse=config.snowflake_warehouse,
            database=config.snowflake_database,
            schema=config.snowflake_schema,
        )

        try:
            await asyncio.to_thread(source.execute_query, query, source_label)

            texts: list[str] = []
            for doc_id in source.list_documents():
                raw = source.load_document(doc_id)
                texts.append(raw.decode("utf-8", errors="ignore"))

            if not texts:
                logger.warning("Snowflake query returned no results")
                return {"status": "success", "source_type": "snowflake", "processed": 0}

            combined = "\n\n".join(texts)
            await self._lightrag.ainsert(combined)
            logger.info("Inserted %d Snowflake rows via LightRAG ainsert", len(texts))

            return {
                "status": "success",
                "source_type": "snowflake",
                "processed": len(texts),
                "source_label": source_label,
            }
        finally:
            await asyncio.to_thread(source.close)

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

        if self.unified is not None:
            result = await self.unified.aretrieve(
                query=query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k
            )
            return RetrievalResult(
                answer=None,
                contexts=result.get("contexts", {}),
                raw=result.get("raw", {}),
            )

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

        if self.unified is not None:
            result = await self.unified.aanswer(
                query=query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k
            )
            return RetrievalResult(
                answer=result.get("answer"),
                contexts=result.get("contexts", {}),
                raw=result.get("raw", {}),
            )

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

    async def aanswer_stream(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] | None = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], AsyncIterator[str]]:
        """Streaming answer: retrieve contexts, then stream LLM tokens.

        Returns (contexts, raw, token_iterator).
        """
        self._ensure_initialized()

        if self.unified is not None:
            return await self.unified.aanswer_stream(
                query=query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k
            )

        if not self.retrieval:
            raise RuntimeError("Retrieval engine not initialized")

        return await self.retrieval.aanswer_stream(
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
        if self.unified is not None:
            return await self._hash_index.list_all()
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
        if self.unified is not None:
            return await self._unified_delete_files(file_paths=file_paths, filenames=filenames)
        if not self.ingestion:
            raise RuntimeError("Ingestion pipeline not initialized")
        return await self.ingestion.adelete_files(
            file_paths=file_paths,
            filenames=filenames,
            delete_source=delete_source,
        )

    async def _unified_delete_files(
        self,
        *,
        file_paths: list[str] | None = None,
        filenames: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Delete files in unified mode: hash lookup -> adelete_by_doc_id -> visual_chunks."""
        from dlightrag.core.ingestion.cleanup import collect_deletion_context

        if not file_paths and not filenames:
            return []

        results: list[dict[str, Any]] = []
        identifiers: list[str] = []
        if file_paths:
            identifiers.extend(file_paths)
        if filenames:
            identifiers.extend(filenames)

        lightrag = self._lightrag

        for identifier in identifiers:
            ctx = await collect_deletion_context(
                identifier=identifier,
                hash_index=self._hash_index,
                lightrag=lightrag,
            )

            deletion_result: dict[str, Any] = {
                "identifier": identifier,
                "doc_ids_found": list(ctx.doc_ids),
                "sources_used": ctx.sources_used,
                "cleanup_results": {},
                "status": "deleted",
            }

            if not ctx.doc_ids:
                deletion_result["status"] = "not_found"
                deletion_result["file_path"] = identifier
                deletion_result["doc_id"] = None
                results.append(deletion_result)
                continue

            for doc_id in ctx.doc_ids:
                # Phase 1: Clean visual_chunks (unified-specific)
                if self.unified:
                    try:
                        vc_result = await self.unified.adelete_doc(doc_id)
                        deletion_result["cleanup_results"]["visual_chunks"] = vc_result
                    except Exception as exc:
                        logger.warning("visual_chunks cleanup failed for %s: %s", doc_id, exc)
                        deletion_result["cleanup_results"]["visual_chunks"] = f"error: {exc}"

                # Phase 2: LightRAG cleanup (chunks_vdb, text_chunks, full_docs, KG)
                if lightrag and hasattr(lightrag, "adelete_by_doc_id"):
                    try:
                        result = await lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
                        deletion_result["cleanup_results"][f"lightrag_{doc_id}"] = (
                            result.status if hasattr(result, "status") else "completed"
                        )
                    except Exception as exc:
                        logger.warning("LightRAG deletion failed for %s: %s", doc_id, exc)
                        deletion_result["cleanup_results"][f"lightrag_{doc_id}"] = f"error: {exc}"

            # Phase 3: Remove from hash index
            for content_hash in ctx.content_hashes:
                if await self._hash_index.remove(content_hash):
                    deletion_result["cleanup_results"]["hash_index"] = "removed"

            deletion_result["file_path"] = list(ctx.file_paths)[0] if ctx.file_paths else identifier
            deletion_result["doc_id"] = list(ctx.doc_ids)[0] if ctx.doc_ids else None
            results.append(deletion_result)

        return results


__all__ = ["RAGService"]
