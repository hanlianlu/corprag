# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One-workspace storage-neutral RAG capability."""

import asyncio
import logging
import uuid
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from inspect import isawaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from lightrag.constants import PARSED_DIR_NAME

from dlightrag.engine.ai.embedding import MultimodalEmbedder, create_embedding_model
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.telemetry import Telemetry
from dlightrag.engine.rag.corpus.contracts import (
    DocStatusLookup,
    IngestDocument,
    SourceType,
    VisualAssetSize,
)
from dlightrag.engine.rag.corpus.ingestion.document_embedding import (
    build_document_embedder,
    resolve_direct_image_embedding_enabled,
)
from dlightrag.engine.rag.corpus.ingestion.engine import (
    PreparedIngestFile,
    UnifiedIngestionEngine,
)
from dlightrag.engine.rag.corpus.ingestion.errors import RetryOutcomeUncertainError
from dlightrag.engine.rag.corpus.ingestion.paths import (
    iter_ingestable_files,
    lightrag_archived_source_path,
    remote_ingest_batch_root,
    remote_parser_input_path,
    retained_remote_source_path,
    stage_input_file,
    workspace_input_root,
)
from dlightrag.engine.rag.corpus.metadata_index import MetadataIndexProtocol
from dlightrag.engine.rag.corpus.sources.base import AsyncDataSource, SourceDocument
from dlightrag.engine.rag.corpus.sources.source_contract import (
    SourceDownloadContractError,
    local_source_uri,
    safe_source_filename,
    validate_download_uri,
    validate_source_uri,
)
from dlightrag.engine.rag.corpus.visual_assets import ThumbnailCache, VisualAssetResolver
from dlightrag.engine.rag.lightrag.models import LightRagChatModels, build_lightrag_embedding
from dlightrag.engine.rag.retrieval import MetadataFilter, RetrievalResult
from dlightrag.engine.rag.retrieval.metadata_fields import (
    INGEST_FINALIZATION_COMPLETE_FIELD,
)
from dlightrag.engine.rag.retrieval.rerank import (
    build_product_reranker,
    rerank_consumes_images,
)
from dlightrag.engine.rag.retrieval.visibility import ingest_finalization_complete
from dlightrag.engine.rag.workspace.lifecycle import (
    defer_cancellation,
    shutdown_lightrag_worker_pools,
)
from dlightrag.engine.rag.workspace.ports import CorpusRuntimeModels, WorkspaceCorpusBackend
from dlightrag.engine.rag.workspace.settings import RagSettings
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id

if TYPE_CHECKING:
    from dlightrag.engine.rag.corpus.ingestion.document_embedding import RobustDocumentEmbedder
    from dlightrag.engine.rag.lightrag.stores import LightRAGStores
    from dlightrag.engine.rag.retrieval.lightrag_backend import LightRAGMixBackend
    from dlightrag.engine.rag.retrieval.ports import BM25Search, RetrievalBackend
    from dlightrag.engine.rag.retrieval.provenance import ProvenanceCache
    from dlightrag.engine.rag.retrieval.retriever import UnifiedRetriever
    from dlightrag.engine.rag.retrieval.visual import PreparedVisualQuery, VisualEmbeddingDomain

logger = logging.getLogger(__name__)


def _ingest_documents(value: Any | None) -> list[IngestDocument] | None:
    if value is None:
        return None
    return [
        document
        if isinstance(document, IngestDocument)
        else IngestDocument.model_validate(document)
        for document in value
    ]


def _source_document_from_manifest(document: IngestDocument, *, key: str) -> SourceDocument:
    return SourceDocument(
        key=key,
        source_uri=document.source_uri,
        download_uri=document.download_uri,
        display_filename=document.filename,
        title=document.title,
        author=document.author,
        metadata=document.metadata,
    )


_REMOTE_INGEST_BATCH_SIZE = 64
_REMOTE_DOWNLOAD_CONCURRENCY = 8

RemoteIngestProgressCallback = Callable[["RemoteIngestWindowProgress"], Awaitable[None]]


@dataclass(frozen=True)
class RemoteIngestWindowProgress:
    """Progress emitted after one remote object ingest window finishes."""

    source_type: str
    batch_index: int
    total_delta: int
    processed_delta: int
    failed_delta: int
    errors: tuple[str, ...] = ()
    # Successfully committed chunks in this window; feeds the workspace's
    # monotonic ingested-chunk counter behind the promotion trigger.
    chunk_delta: int = 0


@dataclass(frozen=True)
class _RemoteDownloadFailure:
    """Sanitized per-document failure returned without exposing source exceptions."""

    error: str


def _safe_remote_source_id(document: SourceDocument) -> str:
    return safe_source_filename(document.display_filename or document.key)


def _normalized_retry_status(value: object) -> str:
    raw_status = (
        value.get("status") if isinstance(value, Mapping) else getattr(value, "status", None)
    )
    return str(getattr(raw_status, "value", raw_status) or "").strip().lower()


def _retry_display_filename(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("source filename is invalid")
    raw_basename = value.replace("\\", "/").rsplit("/", 1)[-1]
    if not raw_basename or raw_basename in {".", ".."}:
        raise ValueError("source filename is invalid")
    filename = safe_source_filename(value)
    if filename in {".", ".."}:
        raise ValueError("source filename is invalid")
    return filename


def _download_locator_kind(locator: str | None, *, retained: bool = False) -> str:
    if retained:
        return "local"
    if locator is None:
        return "none"
    for scheme in ("s3", "azure", "https"):
        if locator.startswith(f"{scheme}://"):
            return scheme
    return "unsupported"


def _log_download_locator_outcome(*, outcome: str, locator_kind: str, source_filename: str) -> None:
    logger.info(
        "source_download_locator_outcome",
        extra={
            "outcome": outcome,
            "locator_kind": locator_kind,
            "source_filename": source_filename,
        },
    )


# Identity and storage plumbing: never part of a caller-facing answer or payload.
_INTERNAL_FIELDS = frozenset(
    {
        "workspace",
        "doc_id",
        "download_locator",
        INGEST_FINALIZATION_COMPLETE_FIELD,
    }
)


class WorkspaceRag:
    """Own one canonical workspace's retrieval and ingestion lifecycle.

    Usage:
        rag = await WorkspaceRag.acreate(
            workspace_id="research",
            settings=settings,
            backend=backend,
            scheduler=scheduler,
            telemetry=telemetry,
        )
    """

    @classmethod
    async def acreate(
        cls,
        *,
        workspace_id: str,
        settings: RagSettings,
        backend: WorkspaceCorpusBackend,
        scheduler: ModelScheduler,
        telemetry: Telemetry,
        rerank_supports_vision: bool | None = None,
    ) -> WorkspaceRag:
        """Create and initialize one workspace capability."""
        instance = cls(
            workspace_id=workspace_id,
            settings=settings,
            backend=backend,
            scheduler=scheduler,
            telemetry=telemetry,
            rerank_supports_vision=rerank_supports_vision,
        )
        try:
            await instance.initialize()
        except BaseException:
            try:
                await instance.aclose()
            except BaseException:  # noqa: BLE001 - preserve the initialization failure
                logger.warning("Failed to close partially initialized WorkspaceRag", exc_info=True)
            raise
        return instance

    def __init__(
        self,
        *,
        workspace_id: str,
        settings: RagSettings,
        backend: WorkspaceCorpusBackend,
        scheduler: ModelScheduler,
        telemetry: Telemetry,
        rerank_supports_vision: bool | None = None,
    ) -> None:
        """Store immutable settings and collaborators without side effects."""
        self.workspace_id = require_canonical_workspace_id(workspace_id)
        if backend.workspace_id != self.workspace_id:
            raise ValueError(
                f"backend workspace {backend.workspace_id!r} does not match {self.workspace_id!r}"
            )
        if backend.read_only != settings.read_only:
            raise ValueError("backend reader role does not match RagSettings")
        self.settings = settings
        self._model_scheduler = scheduler
        self.telemetry = telemetry
        self._initialized: bool = False
        self._pipeline_recovery_task: asyncio.Task[None] | None = None
        self._corpus_backend: WorkspaceCorpusBackend = backend

        self._rerank_supports_vision = rerank_supports_vision
        self._rerank_consumes_images: bool = True

        # Direct LightRAG runtime and DlightRAG orchestration.
        self._lightrag: Any = None  # Direct LightRAG reference
        self._metadata_index: MetadataIndexProtocol | None = None
        self._doc_status_lookup: DocStatusLookup | None = None
        self._table_schema: dict[str, Any] | None = None  # Cached metadata table schema
        self._lightrag_stores: LightRAGStores | None = None
        self._ingestion_engine: UnifiedIngestionEngine | None = None
        self._bm25: BM25Search | None = None
        self._retrieval_orchestrator: UnifiedRetriever | None = None
        self._chat_models: LightRagChatModels | None = None
        self._multimodal_embedder: MultimodalEmbedder | None = None
        self._document_embedder: RobustDocumentEmbedder | None = None
        self._rerank_func: Any = None
        self._direct_image_embedding_enabled = False
        self._visual_asset_resolver: VisualAssetResolver | None = None

        # Retrieval backend (satisfies RetrievalBackend Protocol).
        # Explicitly wired by the unified LightRAG initialization path.
        self._backend: RetrievalBackend | None = None

    @property
    def lightrag(self) -> Any:
        """Return the underlying LightRAG instance for the unified runtime."""
        return self._lightrag

    @property
    def backend(self) -> WorkspaceCorpusBackend:
        """Return the coherent backend bundle bound to this workspace."""
        return self._corpus_backend

    def _require_writer(self, operation: str) -> None:
        if self.settings.read_only:
            raise PermissionError(
                f"{operation} is not available on a reader instance; "
                "route corpus writes to a writer instance."
            )

    @staticmethod
    def _build_retrieval_backend(
        settings: RagSettings,
        *,
        lightrag: Any,
        stores: Any,
    ) -> LightRAGMixBackend:
        """Build the LightRAG retrieval backend from immutable RAG settings."""
        from dlightrag.engine.rag.retrieval.lightrag_backend import LightRAGMixBackend

        return LightRAGMixBackend(
            lightrag=lightrag,
            stores=stores,
            max_entity_tokens=settings.max_entity_tokens,
            max_relation_tokens=settings.max_relation_tokens,
            max_total_tokens=settings.max_total_tokens,
        )

    async def initialize(self) -> None:
        """Initialize LightRAG storages and caches (idempotent)."""
        if self._initialized:
            return

        async with self.backend.coordination.workspace_initialization():
            await self._do_initialize()

        self._initialized = True
        logger.debug("WorkspaceRag initialized")

    async def _do_initialize(self) -> None:
        """Create one LightRAG-backed unified pipeline."""
        from lightrag.parser.routing import validate_parser_routing_config

        from dlightrag.engine.rag.lightrag.patches import apply as apply_lightrag_patches

        validate_parser_routing_config(self.settings.parser_rules)
        apply_lightrag_patches(
            docling_active=self.settings.docling_active,
            docling_code_formula_preset=self.settings.docling_code_formula_preset,
        )
        await self._do_initialize_unified()

    async def _do_initialize_unified(self) -> None:
        """Initialize the direct LightRAG multimodal runtime.

        Wires storage-neutral retrieval and ingestion around the backend runtime.
        """
        settings = self.settings
        logger.info("Initializing unified representational RAG mode...")

        # Build one service-owned model bundle for LightRAG's default and role calls.
        chat_models = await LightRagChatModels.acreate(
            settings.model_roles,
            scheduler=self._model_scheduler,
            telemetry=self.telemetry,
        )
        self._chat_models = chat_models
        default_func_lr = chat_models.default_func
        resolved_rerank = settings.rerank
        rerank_func = build_product_reranker(
            resolved_rerank,
            scoring_settings=settings.rerank_scoring_model,
            scheduler=self._model_scheduler,
            supports_vision=self._rerank_supports_vision,
            telemetry=self.telemetry,
        )
        self._rerank_func = rerank_func
        self._rerank_consumes_images = rerank_consumes_images(
            resolved_rerank,
            supports_vision=self._rerank_supports_vision,
        )
        role_overrides = chat_models.role_configs
        if role_overrides is not None:
            logger.info("LightRAG role overrides: %s", sorted(role_overrides.keys()))

        resolved_embedding = settings.embedding
        multimodal_embedder = create_embedding_model(
            resolved_embedding,
            scheduler=self._model_scheduler,
            telemetry=self.telemetry,
        )
        self._multimodal_embedder = multimodal_embedder
        embedding_func = build_lightrag_embedding(resolved_embedding, multimodal_embedder)
        self._direct_image_embedding_enabled = await resolve_direct_image_embedding_enabled(
            multimodal_embedder,
            startup_probe=resolved_embedding.startup_probe,
            require_image_support=resolved_embedding.input_modality == "multimodal",
        )
        document_embedder = build_document_embedder(
            settings,
            multimodal_embedder,
            image_enabled=self._direct_image_embedding_enabled,
        )
        self._document_embedder = document_embedder

        lightrag = self.backend.runtime.create(
            models=CorpusRuntimeModels(
                default_llm_func=default_func_lr,
                embedding_func=embedding_func,
                role_llm_configs=role_overrides,
            ),
            settings=settings,
        )
        self._lightrag = lightrag
        corpus_stores = await self.backend.runtime.attach(lightrag)
        logger.info(
            "LightRAG storages %s",
            "attached (read-only)" if settings.read_only else "initialized",
        )

        # Wrap chunks_vdb for metadata in-filtering
        if lightrag.chunks_vdb is not None:
            from dlightrag.engine.rag.retrieval.filtering import FilteredVectorStorage

            filtered_vdb = FilteredVectorStorage(
                original=lightrag.chunks_vdb,
                embedding_func=embedding_func,
                visibility_lookup=corpus_stores.metadata_index,
                filtered_search=corpus_stores.filtered_vectors,
            )
            lightrag.chunks_vdb = filtered_vdb  # type: ignore[assignment]

        # Wrap text_chunks so the same scope reaches the entity/relation legs,
        # which resolve chunks by id and never pass through chunks_vdb. Under
        # an active scope the wrapper replaces the KV round trip with the
        # storage's one-query scoped chunk read.
        if lightrag.text_chunks is not None:
            from dlightrag.engine.rag.retrieval.filtering import FilteredChunkStore

            lightrag.text_chunks = FilteredChunkStore(  # type: ignore[assignment]
                original=lightrag.text_chunks,
                visibility_lookup=corpus_stores.metadata_index,
                scoped_reader=corpus_stores.scoped_chunk_reader,
            )

        from dlightrag.engine.rag.lightrag.stores import LightRAGStores

        self._lightrag_stores = LightRAGStores(
            lightrag,
            chunk_store=corpus_stores.chunks,
        )

        self._visual_asset_resolver = VisualAssetResolver(
            stores=self._lightrag_stores,
            visibility_lookup=corpus_stores.metadata_index,
            thumb_cache=ThumbnailCache(max_size=settings.thumb_cache_size),
        )

        self._backend = self._build_retrieval_backend(
            settings,
            lightrag=lightrag,
            stores=self._lightrag_stores,
        )

        self._metadata_index = corpus_stores.metadata_index
        self._doc_status_lookup = corpus_stores.doc_status_lookup
        from dlightrag.engine.rag.retrieval.language import BM25LanguageClassifier

        bm25_language_classifier = (
            BM25LanguageClassifier(corpus_stores.bm25_languages)
            if corpus_stores.bm25 is not None
            else None
        )
        self._ingestion_engine = (
            None
            if settings.read_only
            else UnifiedIngestionEngine(
                lightrag=lightrag,
                stores=self._lightrag_stores,
                metadata_index=self._metadata_index,
                document_embedder=document_embedder,
                workspace=self.workspace_id,
                parser_rules=settings.parser_rules,
                chunk_options=dict(settings.chunk_options),
                bm25_language_classifier=bm25_language_classifier,
                telemetry=self.telemetry,
            )
        )

        self._bm25 = corpus_stores.bm25

        from dlightrag.engine.rag.retrieval.retriever import UnifiedRetriever
        from dlightrag.engine.rag.retrieval.visual import DirectVisualRetriever

        self._retrieval_orchestrator = UnifiedRetriever(
            backend=self._backend,
            bm25=self._bm25,
            visual=(
                DirectVisualRetriever(
                    embedder=multimodal_embedder,
                    stores=self._lightrag_stores,
                    top_k=settings.direct_visual_top_k,
                )
                if self._direct_image_embedding_enabled
                else None
            ),
            stores=self._lightrag_stores,
            rrf_k=settings.rrf_k,
        )

        logger.info("LightRAG main runtime path ready")

        if not settings.read_only:
            self._pipeline_recovery_task = asyncio.create_task(self._resume_lightrag_pipeline())

    async def _resume_lightrag_pipeline(self) -> None:
        """Run LightRAG's native sweep for pending and interrupted documents."""
        try:
            async with self.backend.coordination.pipeline_recovery():
                async with self.telemetry.observe(
                    "ingest_pipeline",
                    as_type="chain",
                    metadata={"trigger": "startup_recovery"},
                ):
                    await self._lightrag.apipeline_process_enqueue_documents()
                logger.info("LightRAG startup pipeline recovery complete")
        except Exception:
            logger.warning("LightRAG startup pipeline recovery failed", exc_info=True)

    def _ensure_initialized(self) -> None:
        """Raise error if not initialized."""
        if not self._initialized:
            raise RuntimeError(
                "WorkspaceRag not initialized. Use 'await WorkspaceRag.acreate()' instead."
            )

    # -- Graph verification ----------------------------------------------------

    async def aclose(self) -> None:
        """Clean up storages and worker pools (best-effort)."""
        cancellation: asyncio.CancelledError | None = None
        if self._pipeline_recovery_task is not None:
            self._pipeline_recovery_task.cancel()
            try:
                await self._pipeline_recovery_task
            except asyncio.CancelledError as exc:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    cancellation = defer_cancellation(cancellation, exc)
            except Exception:
                logger.warning("Pipeline recovery task raised during shutdown", exc_info=True)
            self._pipeline_recovery_task = None
        # Shutdown LightRAG worker pools first — they hold background asyncio
        # tasks that block asyncio.run() from exiting.
        try:
            await self._shutdown_worker_pools()
        except asyncio.CancelledError as exc:
            cancellation = defer_cancellation(cancellation, exc)
        except Exception:
            logger.warning("Failed to shutdown LightRAG worker pools", exc_info=True)

        if self._rerank_func is not None and hasattr(self._rerank_func, "aclose"):
            try:
                await self._rerank_func.aclose()
            except asyncio.CancelledError as exc:
                cancellation = defer_cancellation(cancellation, exc)
            except Exception:
                logger.warning("Failed to close rerank function", exc_info=True)

        if self._multimodal_embedder is not None:
            try:
                await self._multimodal_embedder.aclose()
            except asyncio.CancelledError as exc:
                cancellation = defer_cancellation(cancellation, exc)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to close multimodal embedder", exc_info=True)
        if self._chat_models is not None:
            try:
                await self._chat_models.aclose()
            except asyncio.CancelledError as exc:
                cancellation = defer_cancellation(cancellation, exc)
            except Exception:
                logger.warning("Failed to close LightRAG chat models", exc_info=True)
            self._chat_models = None
        if self._lightrag is not None:
            try:
                await self._lightrag.finalize_storages()
            except asyncio.CancelledError as exc:
                cancellation = defer_cancellation(cancellation, exc)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to finalize LightRAG storages", exc_info=True)
        if cancellation is not None:
            raise cancellation

    async def areset(
        self,
        *,
        keep_files: bool = False,
        dry_run: bool = False,
        preserve_run_sources_after: str | None = None,
    ) -> dict[str, Any]:
        """Completely remove this workspace -- all data, graph schemas, and files.

        Delegates to the dedicated five-phase RAG reset module.
        """
        self._require_writer("workspace reset")
        from dlightrag.engine.rag.corpus.reset import areset

        result = await areset(
            workspace_id=self.workspace_id,
            input_root=self.settings.input_root,
            lightrag=self.lightrag,
            metadata_index=self._metadata_index,
            maintenance=self.backend.maintenance,
            keep_files=keep_files,
            dry_run=dry_run,
            preserve_run_sources_after=preserve_run_sources_after,
        )
        if not dry_run:
            self._initialized = False
        return result

    async def _shutdown_worker_pools(self) -> None:
        """Shutdown LightRAG priority-queue worker pools."""
        await shutdown_lightrag_worker_pools(self.lightrag)

    async def _upsert_workspace_meta(self, *, display_name: str | None = None) -> None:
        """Persist this workspace in DlightRAG's PostgreSQL registry."""
        self._require_writer("workspace registration")
        await self.backend.maintenance.register_workspace(
            workspace=self.workspace_id,
            display_name=display_name or self.workspace_id,
            embedding_model=self.settings.embedding.model,
        )

    async def aregister_workspace(self, *, display_name: str | None = None) -> None:
        """Persist this initialized workspace so it is discoverable by managers."""
        self._ensure_initialized()
        await self._upsert_workspace_meta(display_name=display_name)

    # === INGESTION API ===

    def _resolve_replace(self, explicit: Any) -> bool:
        if explicit is None:
            return self.settings.ingestion_replace_default
        return bool(explicit)

    def _remote_replacement_locators(
        self,
        *,
        primary_locator: str,
        source_type: str,
        source_uri: str,
        key: str,
    ) -> tuple[str, ...]:
        """Return every locator representation remote ingestion can persist."""
        retained_locator = retained_remote_source_path(
            input_root=self._workspace_input_root(),
            source_type=source_type,
            source_uri=source_uri,
            key=key,
        )
        return tuple(
            dict.fromkeys(
                (
                    primary_locator,
                    str(retained_locator),
                    str(lightrag_archived_source_path(retained_locator)),
                )
            )
        )

    async def _replacement_owners_for_locators(
        self,
        *,
        download_locators: tuple[str, ...],
        source_uri: str,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]]:
        """Resolve exact locator owners for engine-side locked revalidation."""
        if self._metadata_index is None:
            return (), ()
        ownership_by_id: dict[str, str] = {}
        for locator in dict.fromkeys(download_locators):
            owner_ids = await self._metadata_index.find_by_download_locator(locator)
            if any(not isinstance(owner_id, str) or not owner_id for owner_id in owner_ids):
                raise RuntimeError("replacement document ownership is invalid")
            for owner_id in owner_ids:
                prior_locator = ownership_by_id.setdefault(owner_id, locator)
                if prior_locator != locator:
                    raise RuntimeError("replacement document ownership is inconsistent")
        owner_ids = tuple(ownership_by_id)
        return owner_ids, tuple(
            (owner_id, ownership_by_id[owner_id], source_uri) for owner_id in owner_ids
        )

    async def _retry_replacement_owners(
        self,
        *,
        download_locators: tuple[str, ...],
        source_uri: str,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]]:
        """Reconstruct retry retirement intent for engine-side locked revalidation."""
        if self._metadata_index is None:
            raise RetryOutcomeUncertainError("retry replacement ownership is unavailable")
        try:
            return await self._replacement_owners_for_locators(
                download_locators=download_locators,
                source_uri=source_uri,
            )
        except Exception as exc:
            raise RetryOutcomeUncertainError("retry replacement ownership lookup failed") from exc

    async def _aingest_local_file(
        self,
        file_path: Path,
        *,
        replace: bool,
        source_root: Path | None = None,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        track_id: str | None = None,
    ) -> dict[str, Any]:
        """Ingest one local file through the unified LightRAG path."""
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")

        file_path = await asyncio.to_thread(
            stage_input_file,
            input_root=self._workspace_input_root(),
            file_path=file_path,
            relative_to=source_root,
        )
        source_uri = local_source_uri(
            self.workspace_id,
            file_path.relative_to(self._workspace_input_root()),
        )
        result = await self._ingestion_engine.aingest_file(
            file_path,
            source_uri=source_uri,
            download_locator=str(file_path),
            source_uri_explicit=False,
            download_locator_explicit=False,
            replace=replace,
            title=title,
            author=author,
            metadata=metadata,
            track_id=track_id,
        )
        return result

    async def _aingest_local_files(
        self,
        file_paths: list[Path],
        *,
        replace: bool,
        source_root: Path | None = None,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        track_id: str | None = None,
    ) -> dict[str, Any]:
        """Ingest local files through one LightRAG staged batch."""
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")
        if not file_paths:
            return {"processed": 0, "errors": [], "results": []}

        staged_paths = [
            await asyncio.to_thread(
                stage_input_file,
                input_root=self._workspace_input_root(),
                file_path=file_path,
                relative_to=source_root,
            )
            for file_path in file_paths
        ]
        prepared_items = [
            PreparedIngestFile(
                parser_path=staged,
                source_uri=local_source_uri(
                    self.workspace_id,
                    staged.relative_to(self._workspace_input_root()),
                ),
                download_locator=str(staged),
                source_uri_explicit=False,
                download_locator_explicit=False,
            )
            for staged in staged_paths
        ]
        result = await self._ingestion_engine.aingest_files(
            prepared_items,
            replace=replace,
            title=title,
            author=author,
            metadata=metadata,
            track_id=track_id,
        )
        return result

    async def _aingest_local_manifest(
        self,
        documents: list[IngestDocument],
        *,
        replace: bool,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        track_id: str | None = None,
    ) -> dict[str, Any]:
        """Ingest explicitly listed local files with per-document metadata."""
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")
        prepared_items: list[PreparedIngestFile] = []
        for document in documents:
            if document.path is None:
                raise ValueError("local manifest documents require a path")
            file_path = Path(document.path)
            relative_to = self._local_manifest_relative_root(file_path)
            staged = await asyncio.to_thread(
                stage_input_file,
                input_root=self._workspace_input_root(),
                file_path=file_path,
                relative_to=relative_to,
            )
            prepared_items.append(
                PreparedIngestFile(
                    parser_path=staged,
                    source_uri=document.source_uri
                    or local_source_uri(
                        self.workspace_id,
                        staged.relative_to(self._workspace_input_root()),
                    ),
                    download_locator=str(staged),
                    display_filename=document.filename,
                    title=document.title,
                    author=document.author,
                    metadata=document.metadata,
                    source_uri_explicit=document.source_uri is not None,
                    download_locator_explicit=False,
                    display_filename_explicit=document.filename is not None,
                )
            )
        result = await self._ingestion_engine.aingest_files(
            prepared_items,
            replace=replace,
            title=title,
            author=author,
            metadata=metadata,
            track_id=track_id,
        )
        return result

    def _workspace_input_root(self) -> Path:
        return workspace_input_root(self.settings.input_root, self.workspace_id)

    def _local_manifest_relative_root(self, file_path: Path) -> Path | None:
        input_root = self._workspace_input_root()
        try:
            file_path.resolve().relative_to(input_root.resolve())
        except ValueError:
            return None
        return input_root

    async def _download_remote_to_prepared_item(
        self,
        *,
        source: AsyncDataSource,
        document: SourceDocument,
        source_uri: str,
        download_locator: str,
        batch_root: Path,
        retain_source_file: bool,
    ) -> PreparedIngestFile:
        key = document.display_filename or document.key
        if retain_source_file:
            parser_path = Path(download_locator)
        else:
            parser_path = remote_parser_input_path(
                batch_root=batch_root,
                source_uri=source_uri,
                key=key,
            )
        parser_path.parent.mkdir(parents=True, exist_ok=True)
        await source.amaterialize_document(document, parser_path)
        return PreparedIngestFile(
            parser_path=parser_path,
            source_uri=source_uri,
            download_locator=download_locator,
            display_filename=document.display_filename or Path(key).name,
            title=document.title,
            author=document.author,
            metadata=document.metadata,
        )

    async def _aingest_remote_documents(
        self,
        *,
        source: AsyncDataSource,
        source_type: str,
        documents: Iterable[SourceDocument] | AsyncIterable[SourceDocument],
        source_uri_for_key: Callable[[str], str],
        download_uri_for_key: Callable[[str], str | None] | None,
        replace: bool,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        progress_callback: RemoteIngestProgressCallback | None = None,
        resume_from_window: int = 0,
        retain_source_file: bool | None = None,
        track_id: str | None = None,
    ) -> dict[str, Any]:
        """Download remote objects into ephemeral parser batches and ingest them."""
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")
        from dlightrag.engine.ai.concurrency import bounded_map

        resume_from_window = max(0, int(resume_from_window))
        processed = 0
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        saw_documents = False
        retain_source_files = (
            self.settings.retain_remote_source_files
            if retain_source_file is None
            else bool(retain_source_file)
        )

        async for batch_index, window in _aiter_chunks(documents, _REMOTE_INGEST_BATCH_SIZE):
            saw_documents = True
            if batch_index < resume_from_window:
                continue

            batch_root = remote_ingest_batch_root(
                input_root=self._workspace_input_root(),
                source_type=source_type,
                batch_id=f"{batch_index:04d}-{uuid.uuid4().hex}",
            )
            prepared_items: list[PreparedIngestFile] = []
            window_errors: list[str] = []
            remote_primary_locators: dict[int, str] = {}

            async def _download(
                document: SourceDocument,
                *,
                current_batch_root: Path = batch_root,
                current_primary_locators: dict[int, str] = remote_primary_locators,
            ) -> PreparedIngestFile | _RemoteDownloadFailure:
                safe_source_id = _safe_remote_source_id(document)
                try:
                    source_uri = validate_source_uri(
                        document.source_uri or source_uri_for_key(document.key)
                    )
                except asyncio.CancelledError:
                    raise
                except TypeError, ValueError:
                    return _RemoteDownloadFailure(f"{safe_source_id}: source_uri is invalid")
                except Exception:  # noqa: BLE001
                    return _RemoteDownloadFailure(f"{safe_source_id}: source_uri resolution failed")

                try:
                    download_uri = document.download_uri
                    if download_uri is None and download_uri_for_key is not None:
                        download_uri = download_uri_for_key(document.key)

                    if retain_source_files:
                        download_locator = str(
                            retained_remote_source_path(
                                input_root=self._workspace_input_root(),
                                source_type=source_type,
                                source_uri=source_uri,
                                key=document.display_filename or document.key,
                            )
                        )
                        _log_download_locator_outcome(
                            outcome="accepted",
                            locator_kind=_download_locator_kind(None, retained=True),
                            source_filename=safe_source_id,
                        )
                    else:
                        if download_uri is None:
                            _log_download_locator_outcome(
                                outcome="missing",
                                locator_kind=_download_locator_kind(None),
                                source_filename=safe_source_id,
                            )
                            raise SourceDownloadContractError(
                                "retain_source_file=false requires a durable download_uri "
                                f"for source {safe_source_id}; "
                                "provide download_uri/download_uri_for_key or enable "
                                "retain_source_file"
                            )
                        try:
                            download_locator = validate_download_uri(download_uri)
                        except ValueError:
                            _log_download_locator_outcome(
                                outcome="unsupported",
                                locator_kind=_download_locator_kind(download_uri),
                                source_filename=safe_source_id,
                            )
                            raise SourceDownloadContractError(
                                f"invalid durable download_uri for source {safe_source_id}; "
                                "provide a supported durable URI or enable retain_source_file"
                            ) from None
                        _log_download_locator_outcome(
                            outcome="accepted",
                            locator_kind=_download_locator_kind(download_locator),
                            source_filename=safe_source_id,
                        )

                    if retain_source_files and download_uri is not None:
                        try:
                            current_primary_locators[id(document)] = validate_download_uri(
                                download_uri
                            )
                        except ValueError:
                            pass
                    else:
                        current_primary_locators[id(document)] = download_locator

                    return await self._download_remote_to_prepared_item(
                        source=source,
                        document=document,
                        source_uri=source_uri,
                        download_locator=download_locator,
                        batch_root=current_batch_root,
                        retain_source_file=retain_source_files,
                    )
                except SourceDownloadContractError as exc:
                    return _RemoteDownloadFailure(str(exc))
                except Exception:  # noqa: BLE001
                    return _RemoteDownloadFailure(
                        f"{safe_source_id}: remote materialization failed"
                    )

            try:
                download_results = await bounded_map(
                    window,
                    _download,
                    max_concurrent=_REMOTE_DOWNLOAD_CONCURRENCY,
                    task_name=f"{source_type}-download",
                )

                prepared_documents: list[SourceDocument] = []
                for document, downloaded in zip(window, download_results, strict=True):
                    if isinstance(downloaded, _RemoteDownloadFailure):
                        error = downloaded.error
                        errors.append(error)
                        window_errors.append(error)
                        continue
                    if isinstance(downloaded, Exception):
                        error = f"{_safe_remote_source_id(document)}: remote materialization failed"
                        errors.append(error)
                        window_errors.append(error)
                        continue
                    prepared_items.append(downloaded)
                    prepared_documents.append(document)

                if not prepared_items:
                    if progress_callback is not None:
                        await progress_callback(
                            RemoteIngestWindowProgress(
                                source_type=source_type,
                                batch_index=batch_index,
                                total_delta=len(window),
                                processed_delta=0,
                                failed_delta=len(window),
                                errors=tuple(window_errors),
                            )
                        )
                    continue

                if replace:
                    replacement_items: list[PreparedIngestFile] = []
                    for document, prepared_item in zip(
                        prepared_documents, prepared_items, strict=True
                    ):
                        replacement_locators = self._remote_replacement_locators(
                            primary_locator=remote_primary_locators.get(
                                id(document), prepared_item.download_locator
                            ),
                            source_type=source_type,
                            source_uri=prepared_item.source_uri,
                            key=document.display_filename or document.key,
                        )
                        (
                            replacement_doc_ids,
                            replacement_ownership,
                        ) = await self._replacement_owners_for_locators(
                            download_locators=replacement_locators,
                            source_uri=prepared_item.source_uri,
                        )
                        replacement_items.append(
                            dataclass_replace(
                                prepared_item,
                                replacement_doc_ids=replacement_doc_ids,
                                replacement_ownership=replacement_ownership,
                            )
                        )
                    prepared_items = replacement_items

                batch_result = await self._ingestion_engine.aingest_files(
                    prepared_items,
                    replace=replace,
                    title=title,
                    author=author,
                    metadata=metadata,
                    track_id=track_id,
                )

                processed += int(batch_result.get("processed") or 0)
                results.extend(batch_result.get("results") or [])
                batch_errors = [str(error) for error in batch_result.get("errors") or []]
                errors.extend(batch_errors)
                if progress_callback is not None:
                    await progress_callback(
                        RemoteIngestWindowProgress(
                            source_type=source_type,
                            batch_index=batch_index,
                            total_delta=len(window),
                            processed_delta=int(batch_result.get("processed") or 0),
                            failed_delta=len(batch_errors)
                            + max(0, len(window) - len(prepared_items)),
                            errors=tuple([*window_errors, *batch_errors]),
                            chunk_delta=sum(
                                len(item.get("chunks") or [])
                                for item in batch_result.get("results") or []
                                if isinstance(item, dict)
                            ),
                        )
                    )
            finally:
                if not retain_source_files:
                    await asyncio.to_thread(_remove_remote_parser_sources, prepared_items)
                await asyncio.to_thread(
                    _remove_empty_parents,
                    batch_root,
                    self._workspace_input_root(),
                )

        if not saw_documents:
            return {"processed": 0, "errors": [], "results": []}
        return {"processed": processed, "errors": errors, "results": results}

    @staticmethod
    def _single_file_result(batch_result: dict[str, Any]) -> dict[str, Any]:
        results = batch_result.get("results")
        if isinstance(results, list) and results:
            return results[0]
        return batch_result

    @staticmethod
    def _default_source_uri_for_key(source_type: str, key: str) -> str:
        if not source_type or "://" in source_type:
            raise ValueError("source_type must be a non-empty URI scheme name")
        return f"{source_type}://{key.lstrip('/')}"

    async def aingest_source(
        self,
        source: AsyncDataSource,
        *,
        source_type: str = "source",
        documents: Iterable[SourceDocument] | AsyncIterable[SourceDocument] | None = None,
        prefix: str | None = None,
        source_uri_for_key: Callable[[str], str] | None = None,
        download_uri_for_key: Callable[[str], str | None] | None = None,
        replace: bool | None = None,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        retain_source_file: bool | None = None,
        _progress_callback: RemoteIngestProgressCallback | None = None,
        _resume_from_window: int = 0,
        _track_id: str | None = None,
    ) -> dict[str, Any]:
        """Ingest documents from a caller-provided async data source.

        SDK connectors expose stable document ids and write bytes into the
        destination path DlightRAG provides. DlightRAG handles temporary parser
        files, metadata provenance, replace semantics, and cleanup.
        """
        self._require_writer("ingestion")
        self._ensure_initialized()
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")

        resolved_documents = (
            documents
            if documents is not None
            else cast(
                Iterable[SourceDocument] | AsyncIterable[SourceDocument],
                source.aiter_documents(prefix=prefix),
            )
        )
        uri_for_key = source_uri_for_key or (
            lambda key: self._default_source_uri_for_key(source_type, key)
        )
        close = getattr(source, "aclose", None)
        try:
            return await self._aingest_remote_documents(
                source=source,
                source_type=source_type,
                documents=resolved_documents,
                source_uri_for_key=uri_for_key,
                download_uri_for_key=download_uri_for_key,
                replace=self._resolve_replace(replace),
                title=title,
                author=author,
                metadata=metadata,
                progress_callback=_progress_callback,
                resume_from_window=_resume_from_window,
                retain_source_file=retain_source_file,
                track_id=_track_id,
            )
        finally:
            if close is not None:
                result = close()
                if isawaitable(result):
                    _ = await result

    async def _aingest_url(self, *, replace: bool, **kwargs: Any) -> dict[str, Any]:
        documents = _ingest_documents(kwargs.get("documents"))
        if documents is not None:
            from dlightrag.engine.rag.corpus.sources.url import URLDataSource

            source = URLDataSource(
                documents=[
                    _source_document_from_manifest(document, key=cast(str, document.url))
                    for document in documents
                ],
                max_download_bytes=self.settings.url_ingest_max_bytes,
                allow_private_hosts=self.settings.url_ingest_private_host_allowlist,
            )
            result = await self.aingest_source(
                source,
                source_type="url",
                source_uri_for_key=source.source_uri_for_key,
                download_uri_for_key=source.download_uri_for_key,
                replace=replace,
                title=kwargs.get("title"),
                author=kwargs.get("author"),
                metadata=kwargs.get("metadata"),
                retain_source_file=kwargs.get("retain_source_file"),
                _progress_callback=kwargs.get("_progress_callback"),
                _resume_from_window=int(kwargs.get("_resume_from_window") or 0),
                _track_id=kwargs.get("_track_id"),
            )
            if len(documents) == 1:
                return self._single_file_result(result)
            return result

        raw_urls = kwargs.get("urls")
        urls = list(raw_urls) if isinstance(raw_urls, list) else []
        if kwargs.get("url"):
            urls = [str(kwargs["url"])]

        from dlightrag.engine.rag.corpus.sources.url import URLDataSource

        source_kwargs: dict[str, Any] = {"urls": urls}
        if kwargs.get("filename") is not None:
            source_kwargs["filename"] = kwargs["filename"]
        if kwargs.get("source_uri") is not None:
            source_kwargs["source_uri"] = kwargs["source_uri"]
        if kwargs.get("source_uris") is not None:
            source_kwargs["source_uris"] = kwargs["source_uris"]
        if kwargs.get("download_uri") is not None:
            source_kwargs["download_uri"] = kwargs["download_uri"]
        if kwargs.get("download_uris") is not None:
            source_kwargs["download_uris"] = kwargs["download_uris"]
        source = URLDataSource(
            **source_kwargs,
            max_download_bytes=self.settings.url_ingest_max_bytes,
            allow_private_hosts=self.settings.url_ingest_private_host_allowlist,
        )
        result = await self.aingest_source(
            source,
            source_type="url",
            source_uri_for_key=source.source_uri_for_key,
            download_uri_for_key=source.download_uri_for_key,
            replace=replace,
            title=kwargs.get("title"),
            author=kwargs.get("author"),
            metadata=kwargs.get("metadata"),
            retain_source_file=kwargs.get("retain_source_file"),
            _progress_callback=kwargs.get("_progress_callback"),
            _resume_from_window=int(kwargs.get("_resume_from_window") or 0),
            _track_id=kwargs.get("_track_id"),
        )
        if len(urls) == 1:
            return self._single_file_result(result)
        return result

    async def _aingest_object_store(
        self,
        *,
        source_type: str,
        source: Any,
        locator_for_key: Callable[[str], str],
        single_key: str | None,
        replace: bool,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Shared manifest/single-key/prefix routing for blob-style sources."""
        common_kwargs = {
            "source_type": source_type,
            "source_uri_for_key": locator_for_key,
            "download_uri_for_key": locator_for_key,
            "replace": replace,
            "title": kwargs.get("title"),
            "author": kwargs.get("author"),
            "metadata": kwargs.get("metadata"),
            "retain_source_file": kwargs.get("retain_source_file"),
            "_progress_callback": kwargs.get("_progress_callback"),
            "_resume_from_window": int(kwargs.get("_resume_from_window") or 0),
            "_track_id": kwargs.get("_track_id"),
        }
        documents = _ingest_documents(kwargs.get("documents"))
        if documents is not None:
            return await self.aingest_source(
                source,
                documents=[
                    _source_document_from_manifest(document, key=cast(str, document.key))
                    for document in documents
                ],
                **common_kwargs,
            )
        if single_key:
            return self._single_file_result(
                await self.aingest_source(
                    source,
                    documents=[SourceDocument(key=single_key)],
                    **common_kwargs,
                )
            )
        prefix = "" if kwargs.get("prefix") is None else str(kwargs.get("prefix"))
        return await self.aingest_source(source, prefix=prefix, **common_kwargs)

    async def _aingest_azure_blob(self, *, replace: bool, **kwargs: Any) -> dict[str, Any]:
        container_name = kwargs.get("container_name")
        source = kwargs.get("source")
        if source is None:
            if not container_name:
                raise ValueError("'container_name' is required for azure_blob source_type")
            from dlightrag.engine.rag.corpus.sources.azure_blob import AzureBlobDataSource

            source = AzureBlobDataSource(
                connection_string=self.settings.blob_connection_string,
                container_name=container_name,
            )
        blob_path = kwargs.get("blob_path")
        return await self._aingest_object_store(
            source_type="azure_blob",
            source=source,
            locator_for_key=lambda key: f"azure://{container_name}/{key}",
            single_key=str(blob_path) if blob_path else None,
            replace=replace,
            kwargs=kwargs,
        )

    async def _aingest_s3(self, *, replace: bool, **kwargs: Any) -> dict[str, Any]:
        bucket = kwargs.get("bucket")
        source = kwargs.get("source")
        if source is None:
            if not bucket:
                raise ValueError("'bucket' is required for s3 source_type")
            from dlightrag.engine.rag.corpus.sources.aws_s3 import S3DataSource

            source = S3DataSource(
                bucket=str(bucket), region=kwargs.get("s3_region") or self.settings.s3_region
            )
        # The MCP contract accepts either name for the single-object case.
        key = kwargs.get("s3_key") or kwargs.get("blob_path")
        return await self._aingest_object_store(
            source_type="s3",
            source=source,
            locator_for_key=lambda key: f"s3://{bucket}/{key}",
            single_key=str(key) if key else None,
            replace=replace,
            kwargs=kwargs,
        )

    async def aingest(
        self,
        source_type: SourceType,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Unified ingestion API.

        Args:
            source_type: "local", "azure_blob", "s3", or "url"
            kwargs:
                local: path, replace
                azure_blob: source, container_name, blob_path, prefix, replace
                s3: bucket, key, prefix, replace
                url: url or urls, optional filename, replace
        """
        self._require_writer("ingestion")
        self._ensure_initialized()
        replace = self._resolve_replace(kwargs.pop("replace", None))
        track_id = kwargs.pop("_track_id", None)

        if self._ingestion_engine is not None and source_type == "local":
            documents = _ingest_documents(kwargs.get("documents"))
            if documents is not None:
                return await self._aingest_local_manifest(
                    documents,
                    replace=replace,
                    title=kwargs.get("title"),
                    author=kwargs.get("author"),
                    metadata=kwargs.get("metadata"),
                    track_id=track_id,
                )
            path_str = kwargs.get("path")
            if not path_str:
                raise ValueError("'path' is required for local source_type")
            local_path = Path(path_str)
            file_paths = await asyncio.to_thread(iter_ingestable_files, local_path)
            common_kwargs = {
                "replace": replace,
                "title": kwargs.get("title"),
                "author": kwargs.get("author"),
                "metadata": kwargs.get("metadata"),
                "track_id": track_id,
            }
            if local_path.is_file():
                return await self._aingest_local_file(local_path, **common_kwargs)

            return await self._aingest_local_files(
                file_paths,
                source_root=local_path,
                **common_kwargs,
            )

        kwargs["_track_id"] = track_id
        if self._ingestion_engine is not None and source_type == "azure_blob":
            return await self._aingest_azure_blob(replace=replace, **kwargs)

        if self._ingestion_engine is not None and source_type == "s3":
            return await self._aingest_s3(replace=replace, **kwargs)

        if self._ingestion_engine is not None and source_type == "url":
            return await self._aingest_url(replace=replace, **kwargs)

        raise RuntimeError("Ingestion engine not initialized")

    # === RETRIEVAL API ===

    @property
    def visual_embedding_domain(self) -> VisualEmbeddingDomain | None:
        """Return the enabled direct-visual vector compatibility domain."""
        orchestrator = self._retrieval_orchestrator
        return orchestrator.visual_embedding_domain if orchestrator is not None else None

    async def prepare_visual_query(
        self, query_image_blocks: list[dict[str, Any]]
    ) -> PreparedVisualQuery | None:
        """Prepare query-image vectors without searching this workspace."""
        self._ensure_initialized()
        orchestrator = self._retrieval_orchestrator
        if orchestrator is None:
            raise RuntimeError("Retrieval orchestrator not initialized")
        return await orchestrator.prepare_visual_query(query_image_blocks)

    async def aretrieve(
        self,
        query: str,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        filters: MetadataFilter | None = None,
        *,
        filter_source: str | None = None,
        bm25_query: str | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve structured data without generating answer.

        The unified retriever applies metadata scope inside every retrieval leg,
        fuses the resulting candidates, and this service hydrates provenance,
        reranks, enriches metadata, and assigns citation identities.

        Args:
            filters: Optional MetadataFilter for structured metadata queries.
            filter_source: Whether filters are explicit or LLM-inferred.
        """
        self._ensure_initialized()
        if self._retrieval_orchestrator is None:
            raise RuntimeError("Retrieval orchestrator not initialized")

        kg_result = await self._retrieval_orchestrator.aretrieve(
            query,
            metadata_filter=filters,
            metadata_filter_source=filter_source,
            bm25_query=(bm25_query or "").strip() or None,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            **kwargs,
        )

        # --- Step 2.5: Hydrate provenance for the complete fused candidate set ---
        stores = self._lightrag_stores
        provenance_cache: ProvenanceCache | None = None
        if stores is not None:
            from dlightrag.engine.rag.retrieval.provenance import (
                ProvenanceCache,
                hydrate_lightrag_chunk_provenance,
            )

            chunks_to_hydrate = kg_result.contexts.get("chunks", [])
            if chunks_to_hydrate:
                # Defer image bytes past rerank truncation for a text-only
                # reranker; multimodal rerankers need them with the candidates.
                provenance_cache = ProvenanceCache()
                await hydrate_lightrag_chunk_provenance(
                    stores,
                    chunks_to_hydrate,
                    include_image_data=self._rerank_consumes_images,
                    cache=provenance_cache,
                )

        await self._rerank_retrieval_chunks(
            query,
            kg_result,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
        )

        # Text-reranker path: image bytes were deferred above, so hydrate them for
        # the surviving chunks now (already-hydrated chunks skip the read).
        if stores is not None and not self._rerank_consumes_images:
            survivors = kg_result.contexts.get("chunks", [])
            if survivors:
                from dlightrag.engine.rag.retrieval.provenance import (
                    hydrate_lightrag_chunk_provenance,
                )

                await hydrate_lightrag_chunk_provenance(stores, survivors, cache=provenance_cache)

        # --- Step 3: Enrich chunks with document metadata ---
        await self._enrich_chunks_with_metadata(kg_result)
        kg_result.trace["metadata_enriched"] = True

        # --- Step 4: Canonicalize reference_id across all merged chunks ---
        # Required so fused chunks get stable doc-level IDs and become citable
        # as [ref-chunk_idx].
        from dlightrag.engine.rag.retrieval.references import (
            canonicalize_reference_ids,
            tag_context_workspace,
        )

        kg_result.contexts["chunks"] = canonicalize_reference_ids(
            kg_result.contexts.get("chunks", [])
        )
        tag_context_workspace(kg_result.contexts, self.workspace_id)
        kg_result.trace["workspace"] = self.workspace_id

        return kg_result

    async def _rerank_retrieval_chunks(
        self,
        query: str,
        result: RetrievalResult,
        *,
        top_k: int | None,
        chunk_top_k: int | None,
    ) -> None:
        chunks = result.contexts.get("chunks", [])
        result.trace.setdefault("reranked_chunk_count", 0)
        if not chunks:
            logger.info(
                "[Rerank] skipped: strategy=%s enabled=%s chunks=0 reason=no_chunks",
                self.settings.rerank.strategy,
                self.settings.rerank.enabled,
            )
            return

        limit = chunk_top_k or top_k or len(chunks)
        from dlightrag.engine.rag.retrieval.rerank_fallback import rerank_with_fallback

        outcome = await rerank_with_fallback(
            query=query,
            chunks=chunks,
            top_k=limit,
            rerank_func=self._rerank_func,
        )
        result.contexts["chunks"] = outcome.chunks
        result.trace["reranked_chunk_count"] = len(outcome.chunks)
        if self._rerank_func is None:
            logger.info(
                "[Rerank] skipped: strategy=%s enabled=%s chunks=%d limit=%d reason=no_function",
                self.settings.rerank.strategy,
                self.settings.rerank.enabled,
                len(chunks),
                limit,
            )
            return
        if outcome.error_type:
            result.trace["rerank_error_type"] = outcome.error_type
            if outcome.failed_batch is not None:
                result.trace["rerank_failed_batch"] = outcome.failed_batch
            return
        top_score = outcome.chunks[0].get("rerank_score") if outcome.chunks else None
        logger.info(
            "[Rerank] strategy=%s model=%s input_chunks=%d limit=%d output_chunks=%d top_score=%s",
            self.settings.rerank.strategy,
            self.settings.rerank.model,
            len(chunks),
            limit,
            result.trace["reranked_chunk_count"],
            top_score,
        )

    async def aget_visual_asset(
        self, chunk_id: str, *, size: VisualAssetSize = "full"
    ) -> Any | None:
        """Resolve a chunk image asset for API/Web image routes."""
        self._ensure_initialized()
        if self._visual_asset_resolver is None:
            return None
        if size == "thumb":
            return await self._visual_asset_resolver.resolve_thumbnail(
                chunk_id,
                max_px=self.settings.thumb_max_px,
            )
        if size == "full":
            return await self._visual_asset_resolver.resolve(chunk_id)
        raise ValueError("size must be 'full' or 'thumb'")

    async def _enrich_chunks_with_metadata(self, result: RetrievalResult) -> None:
        """Inject document metadata into chunk contexts for LLM consumption.

        Looks up each chunk's LightRAG full_doc_id in the metadata index and merges
        any non-empty fields into the chunk's metadata dict. Fields are
        dynamic: whatever the metadata index returns is included, minus
        internal/system fields that add no value to the LLM context.
        """
        _SKIP = _INTERNAL_FIELDS | frozenset(
            {
                "source_uri",
                "file_extension",
                "filename",
                "filename_stem",
                "ingested_at",
                "custom_metadata",
            }
        )

        chunks = result.contexts.get("chunks", [])
        if not chunks:
            return

        # Collect unique LightRAG full_doc_id values. Retrieval adapters must
        # propagate this identity; file_path lookup is for deletion/cleanup, not
        # answer-time enrichment.
        doc_meta: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            doc_id = chunk.get("full_doc_id") or chunk.get("_full_doc_id")
            if doc_id:
                doc_meta.setdefault(str(doc_id), {})

        idx = self._metadata_index
        if idx is None or not doc_meta:
            return

        try:
            fetched = await idx.get_many(list(doc_meta))
        except Exception:
            fetched = {}  # enrichment is best-effort

        for doc_id, meta in fetched.items():
            fetched_meta = {
                k: v for k, v in meta.items() if k not in _SKIP and v is not None and v != ""
            }
            custom = meta.get("custom_metadata")
            if isinstance(custom, dict):
                # User metadata is the reason the column exists: the model sees it.
                fetched_meta.update(custom)
            source_uri = meta.get("source_uri")
            download_locator = meta.get("download_locator")
            if isinstance(source_uri, str) and source_uri:
                fetched_meta["source_uri"] = source_uri
            if isinstance(download_locator, str) and download_locator:
                fetched_meta["source_download_locator"] = download_locator
            display_name = meta.get("filename")
            if isinstance(display_name, str) and display_name:
                fetched_meta["source_file_name"] = display_name
            if fetched_meta:
                doc_meta[doc_id] = fetched_meta

        # Inject into each chunk's metadata field
        for chunk in chunks:
            doc_id = chunk.get("full_doc_id") or chunk.get("_full_doc_id")
            fetched_meta = doc_meta.get(str(doc_id)) if doc_id else None
            if fetched_meta:
                existing = chunk.get("metadata") or {}
                existing.update(fetched_meta)
                chunk["metadata"] = existing

    # === METADATA API ===

    async def aget_metadata(self, doc_id: str) -> dict[str, Any]:
        """Get document metadata by ID."""
        result = await self._metadata_index.get(doc_id)  # type: ignore[union-attr]
        if not ingest_finalization_complete(result):
            return {}
        return {key: value for key, value in result.items() if key not in _INTERNAL_FIELDS}

    async def aupdate_metadata(
        self,
        doc_id: str,
        data: dict[str, Any],
    ) -> None:
        """Update (merge) document metadata."""
        self._require_writer("metadata update")
        from dlightrag.engine.rag.retrieval.metadata_fields import normalize_user_metadata

        if self._metadata_index is None:
            raise RuntimeError("Metadata index not initialized")
        normalized = normalize_user_metadata(data)
        if not await self._metadata_index.is_visible(doc_id):
            raise KeyError(doc_id)
        updated = await self._metadata_index.merge_custom_metadata(
            doc_id,
            {
                **normalized.system,
                "custom_metadata": normalized.custom_metadata,
            },
        )
        if not updated:
            raise KeyError(doc_id)

    async def asearch_metadata(self, filters: MetadataFilter) -> list[str]:
        """Search metadata by filters, return matching doc_ids."""
        if self._metadata_index is None or filters is None:
            return []
        return await self._metadata_index.query(filters)

    # === FILE MANAGEMENT API ===

    async def _iter_failed_doc_pages(self) -> AsyncIterator[list[dict[str, Any]]]:
        """Yield full failed-document presentation rows one bounded page at a time."""
        if self._lightrag_stores is None:
            return

        from lightrag.base import DocStatus

        async for scheduled in self._lightrag_stores.iter_doc_status_pages((DocStatus.FAILED,)):
            doc_ids = list(scheduled)
            full_rows = await self._lightrag_stores.get_full_doc_statuses(doc_ids)
            missing = set(doc_ids).difference(full_rows)
            if missing:
                raise RuntimeError(
                    f"document-status rows disappeared during failed listing: {sorted(missing)}"
                )
            yield [
                {
                    "doc_id": doc_id,
                    "file_path": full_rows[doc_id].file_path or "",
                    "error": full_rows[doc_id].error_msg or full_rows[doc_id].content_summary,
                    "updated_at": str(full_rows[doc_id].updated_at),
                }
                for doc_id in doc_ids
            ]

    async def aretryable_document_ids(self) -> tuple[str, ...]:
        """Snapshot FAILED plus PROCESSED-but-unfinalized Product Documents."""
        self._require_writer("failed-document retry")
        self._ensure_initialized()
        if self._lightrag_stores is None or self._metadata_index is None:
            raise RuntimeError("retry document stores are unavailable")
        from lightrag.base import DocStatus

        cohort: list[str] = []
        async for rows in self._lightrag_stores.iter_doc_status_pages(
            (DocStatus.FAILED, DocStatus.PROCESSED)
        ):
            full = await self._lightrag_stores.get_full_doc_statuses(list(rows))
            for doc_id, row in full.items():
                status = _normalized_retry_status(row)
                if status == "failed":
                    cohort.append(doc_id)
                    continue
                metadata = await self._metadata_index.get(doc_id)
                if not ingest_finalization_complete(metadata):
                    cohort.append(doc_id)
        return tuple(dict.fromkeys(cohort))

    async def aretry_failed_docs(
        self,
        *,
        cohort_doc_ids: Sequence[str] | None = None,
        cohort_callback: Callable[[tuple[str, ...]], Awaitable[None]] | None = None,
        outcome_callback: Callable[[str, str, dict[str, Any]], Awaitable[None]] | None = None,
        track_id: str | None = None,
    ) -> dict[str, Any]:
        """Retry one frozen FAILED cohort and report compact per-document outcomes."""
        self._require_writer("failed-document retry")
        self._ensure_initialized()

        if cohort_doc_ids is None:
            entries = [entry async for page in self._iter_failed_doc_pages() for entry in page]
        else:
            entries = await self._retry_cohort_entries(cohort_doc_ids)
        entries = list({str(entry.get("doc_id")): entry for entry in entries}.values())
        cohort = tuple(str(entry["doc_id"]) for entry in entries)
        if cohort_callback is not None:
            # The durable coordinator seals the complete cohort before the
            # first destructive retry starts.
            await cohort_callback(cohort)

        succeeded: list[dict[str, Any]] = []
        still_failed: list[dict[str, Any]] = []
        succeeded_count = 0
        failed_count = 0
        details_truncated = False

        async def record(doc_id: str, outcome: str, detail: dict[str, Any]) -> None:
            nonlocal details_truncated, failed_count, succeeded_count
            if outcome_callback is not None:
                await outcome_callback(doc_id, outcome, detail)
            if outcome == "succeeded":
                succeeded_count += 1
            else:
                failed_count += 1
            target = succeeded if outcome == "succeeded" else still_failed
            if len(target) < 500:
                target.append(detail)
            else:
                details_truncated = True

        for entry in entries:
            doc_id = str(entry["doc_id"])
            file_path = str(entry.get("file_path") or "")
            recovered_processed = str(entry.get("status") or "") == "processed"

            async def preflight_failure(
                reason: str,
                *,
                processed: bool = recovered_processed,
                expected_doc_id: str = doc_id,
            ) -> None:
                if processed:
                    # A processed pending item may already be a complete commit,
                    # or may only need idempotent application finalization. Never
                    # freeze the opposite ledger outcome because its source
                    # preflight is temporarily unavailable.
                    raise RetryOutcomeUncertainError(reason)
                await record(
                    expected_doc_id,
                    "failed",
                    {"doc_id": expected_doc_id, "reason": reason},
                )

            # A pending durable item with PROCESSED LightRAG status may have
            # crashed before required DlightRAG finalization. A durable complete
            # marker proves success without requiring the source to still exist;
            # incomplete/legacy markers re-enter the same-ID finalization seam.
            if self._metadata_index is None:
                await preflight_failure("source metadata unavailable")
                continue

            try:
                metadata = await self._metadata_index.get(doc_id)
            except Exception as exc:
                logger.warning("Failed to load retry metadata for doc_id=%s", doc_id)
                if recovered_processed:
                    raise RetryOutcomeUncertainError(
                        "retry finalization metadata read failed"
                    ) from exc
                await preflight_failure("source metadata unavailable")
                continue

            if not isinstance(metadata, Mapping):
                await preflight_failure("source metadata incomplete")
                continue
            if recovered_processed and ingest_finalization_complete(metadata):
                await record(
                    doc_id,
                    "succeeded",
                    {"doc_id": doc_id, "file_path": file_path, "replacement_count": 1},
                )
                continue
            source_uri = metadata.get("source_uri")
            download_locator = metadata.get("download_locator")
            stored_filename = metadata.get("filename")
            if (
                not isinstance(source_uri, str)
                or not source_uri
                or not isinstance(download_locator, str)
                or not download_locator
                or not isinstance(stored_filename, str)
                or not stored_filename
            ):
                await preflight_failure("source metadata incomplete")
                continue

            try:
                display_filename = _retry_display_filename(stored_filename)
                self._validate_retry_source_contract(source_uri, download_locator)
            except (OSError, TypeError, ValueError) as exc:
                if recovered_processed:
                    raise RetryOutcomeUncertainError(
                        "retry finalization source preflight failed"
                    ) from exc
                await preflight_failure("source metadata invalid")
                continue

            retry_metadata = self._allowed_retry_metadata(metadata)
            try:
                result = await self._aingest_download_locator(
                    source_uri,
                    download_locator,
                    display_filename,
                    retry_metadata,
                    track_id=track_id,
                )
                processed = result.get("processed")
                if result.get("errors") or (isinstance(processed, int | float) and processed < 1):
                    await record(
                        doc_id,
                        "failed",
                        {
                            "doc_id": doc_id,
                            **({"file_path": file_path} if file_path else {}),
                            "reason": "retry ingestion failed",
                        },
                    )
                    continue
            except RetryOutcomeUncertainError:
                raise
            except Exception:
                logger.warning("Retry failed for doc_id=%s", doc_id)
                outcome = await self._retry_doc_authoritative_outcome(doc_id)
                if outcome == "succeeded":
                    # Corpus commit won a late finalization/outcome race. Keep
                    # durable totals aligned with the authoritative status.
                    await record(
                        doc_id,
                        "succeeded",
                        {"doc_id": doc_id, "file_path": file_path, "replacement_count": 1},
                    )
                    continue
                if outcome is None:
                    raise RetryOutcomeUncertainError(
                        "retry document status is not yet authoritative"
                    ) from None
                await record(
                    doc_id,
                    "failed",
                    {
                        "doc_id": doc_id,
                        **({"file_path": file_path} if file_path else {}),
                        "reason": "retry ingestion failed",
                    },
                )
                continue

            if self._retry_result_identity(result) != doc_id:
                # The parser path fixes the canonical ID before ingestion. An
                # identity-invalid result is not an ambiguous finalization
                # error and must never be reconciled from PROCESSED to success.
                # A reported mismatch may name a pre-existing unrelated doc.
                logger.warning("Retry returned invalid identity for doc_id=%s", doc_id)
                await record(
                    doc_id,
                    "failed",
                    {
                        "doc_id": doc_id,
                        **({"file_path": file_path} if file_path else {}),
                        "reason": "retry ingestion failed",
                    },
                )
                continue

            await record(
                doc_id,
                "succeeded",
                {"doc_id": doc_id, "file_path": file_path, "replacement_count": 1},
            )

        retried = len(entries)
        if retried == 0:
            return {"retried": 0, "succeeded": 0, "failed": 0, "results": []}
        return {
            "retried": retried,
            "succeeded": succeeded_count,
            "failed": failed_count,
            "succeeded_docs": succeeded,
            "failed_docs": still_failed,
            "details_truncated": details_truncated,
        }

    async def _retry_doc_authoritative_outcome(self, doc_id: str) -> str | None:
        """Return a durable outcome only for an authoritative terminal status."""
        if self._lightrag_stores is None:
            raise RetryOutcomeUncertainError("retry document status store is unavailable")
        try:
            row = await self._lightrag_stores.get_doc_status(doc_id)
        except Exception as exc:
            raise RetryOutcomeUncertainError("retry document status read failed") from exc
        status = _normalized_retry_status(row)
        if status == "processed":
            if self._metadata_index is None:
                raise RetryOutcomeUncertainError("retry finalization metadata is unavailable")
            try:
                metadata = await self._metadata_index.get(doc_id)
            except Exception as exc:
                raise RetryOutcomeUncertainError("retry finalization metadata read failed") from exc
            return "succeeded" if ingest_finalization_complete(metadata) else None
        if status == "failed":
            return "failed"
        return None

    async def _retry_cohort_entries(self, cohort_doc_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Project named durable items, including already-committed success."""
        doc_ids = tuple(dict.fromkeys(str(doc_id) for doc_id in cohort_doc_ids if doc_id))
        if not doc_ids:
            return []
        if self._lightrag_stores is None:
            raise RetryOutcomeUncertainError("retry document status store is unavailable")
        try:
            rows = await self._lightrag_stores.get_full_doc_statuses(list(doc_ids))
        except Exception as exc:
            raise RetryOutcomeUncertainError("retry cohort status read failed") from exc
        missing = set(doc_ids).difference(rows)
        if missing:
            raise RetryOutcomeUncertainError("retry cohort status rows are incomplete")
        entries: list[dict[str, Any]] = []
        for doc_id in doc_ids:
            row = rows.get(doc_id)
            status = _normalized_retry_status(row)
            entries.append(
                {
                    "doc_id": doc_id,
                    "file_path": str(getattr(row, "file_path", "") or ""),
                    "status": status,
                }
            )
        return entries

    @staticmethod
    def _allowed_retry_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
        allowed: dict[str, Any] = {}
        for key in ("title", "author", "creation_date"):
            value = metadata.get(key)
            if value is not None:
                allowed[key] = value
        custom = metadata.get("custom_metadata")
        if isinstance(custom, Mapping):
            allowed["custom_metadata"] = dict(custom)
        return allowed

    @staticmethod
    def _retry_result_identity(result: Mapping[str, Any]) -> str | None:
        direct = result.get("doc_id")
        nested = result.get("results")
        direct_id = direct if isinstance(direct, str) and direct else None
        nested_ids = (
            tuple(
                item["doc_id"]
                for item in nested
                if isinstance(item, Mapping)
                and isinstance(item.get("doc_id"), str)
                and item.get("doc_id")
            )
            if isinstance(nested, list)
            else ()
        )
        if direct_id is not None and not nested_ids:
            accepted = direct_id
        elif direct_id is None and len(nested_ids) == 1:
            accepted = nested_ids[0]
        else:
            accepted = None
        return accepted

    def _validate_retry_source_contract(
        self,
        source_uri: str,
        download_locator: str,
    ) -> tuple[SourceType, dict[str, Any]]:
        from dlightrag.engine.rag.corpus.sources.uri import parse_remote_uri

        stable_source_uri = validate_source_uri(source_uri)
        if not download_locator or "\x00" in download_locator:
            raise ValueError("download locator is invalid")
        source_type, parts = parse_remote_uri(download_locator)
        if source_type == "local":
            self._retry_local_source_path(download_locator)
        else:
            validate_download_uri(download_locator)
        if not stable_source_uri:
            raise ValueError("source identity is invalid")
        return source_type, parts

    def _retry_local_source_path(self, download_locator: str) -> Path:
        """Resolve a local source that LightRAG may have moved under __parsed__."""
        original = Path(download_locator)
        input_root = self._workspace_input_root().resolve()
        try:
            resolved_original = original.resolve()
            resolved_original.relative_to(input_root)
        except ValueError:
            raise FileNotFoundError("download locator is unavailable") from None
        if resolved_original.is_file():
            return resolved_original

        candidates = (
            original.parent / PARSED_DIR_NAME / original.name,
            input_root / PARSED_DIR_NAME / original.name,
        )
        for candidate in dict.fromkeys(candidates):
            resolved = candidate.resolve()
            if resolved.is_relative_to(input_root) and resolved.is_file():
                return resolved
        raise FileNotFoundError("download locator is unavailable")

    async def _aingest_download_locator(
        self,
        source_uri: str,
        download_locator: str,
        display_filename: str,
        retry_metadata: Mapping[str, Any] | None = None,
        *,
        track_id: str | None = None,
    ) -> dict[str, Any]:
        """Materialize one validated locator while preserving source provenance."""
        source_type, parts = self._validate_retry_source_contract(source_uri, download_locator)
        stable_source_uri = validate_source_uri(source_uri)
        if source_type == "local":
            ownership_locators = (download_locator,)
        else:
            ownership_locators = self._remote_replacement_locators(
                primary_locator=download_locator,
                source_type=source_type,
                source_uri=stable_source_uri,
                key=display_filename,
            )
        replacement_doc_ids, replacement_ownership = await self._retry_replacement_owners(
            download_locators=ownership_locators,
            source_uri=stable_source_uri,
        )

        raw_title = retry_metadata.get("title") if retry_metadata else None
        raw_author = retry_metadata.get("author") if retry_metadata else None
        title = raw_title if isinstance(raw_title, str) else None
        author = raw_author if isinstance(raw_author, str) else None
        user_metadata: dict[str, Any] = {}
        custom_metadata = retry_metadata.get("custom_metadata") if retry_metadata else None
        if isinstance(custom_metadata, Mapping):
            user_metadata.update(
                (key, value) for key, value in custom_metadata.items() if isinstance(key, str)
            )
        if retry_metadata and retry_metadata.get("creation_date") is not None:
            user_metadata["creation_date"] = retry_metadata["creation_date"]

        retry_kwargs: dict[str, Any] = {
            "source_uri": stable_source_uri,
            "download_locator": download_locator,
            "display_filename": display_filename,
            "title": title,
            "author": author,
            "metadata": user_metadata,
            "replacement_doc_ids": replacement_doc_ids,
            "replacement_ownership": replacement_ownership,
        }
        if source_type == "local":
            return await self._aingest_local_retry_locator(**retry_kwargs, track_id=track_id)
        return await self._aingest_remote_retry_locator(
            source_type=source_type,
            parts=parts,
            track_id=track_id,
            **retry_kwargs,
        )

    async def _aingest_remote_retry_locator(
        self,
        *,
        source_type: str,
        parts: Mapping[str, Any],
        source_uri: str,
        download_locator: str,
        display_filename: str,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        replacement_doc_ids: tuple[str, ...] = (),
        replacement_ownership: tuple[tuple[str, str, str], ...] = (),
        track_id: str | None = None,
    ) -> dict[str, Any]:
        """Download one remote retry locator and replay it in the same-ID seam.

        Replays with ``replace=False``; the adapter and transient parser
        source are cleaned up either way.
        """
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")
        common_fields: dict[str, Any] = {
            "filename": display_filename,
            "source_uri": source_uri,
            "title": title,
            "author": author,
            "metadata": metadata,
        }
        if source_type == "url":
            from dlightrag.engine.rag.corpus.sources.url import URLDataSource

            document = IngestDocument(
                url=download_locator, download_uri=download_locator, **common_fields
            )
            source_document = _source_document_from_manifest(document, key=cast(str, document.url))
            source: AsyncDataSource = URLDataSource(
                documents=[source_document],
                max_download_bytes=self.settings.url_ingest_max_bytes,
                allow_private_hosts=self.settings.url_ingest_private_host_allowlist,
            )
        else:
            object_key = str(parts["blob_path"] if source_type == "azure_blob" else parts["key"])
            document = IngestDocument(key=object_key, **common_fields)
            source_document = _source_document_from_manifest(document, key=object_key)
            if source_type == "s3":
                from dlightrag.engine.rag.corpus.sources.aws_s3 import S3DataSource

                source = S3DataSource(
                    bucket=str(parts["bucket"]),
                    region=self.settings.s3_region,
                )
            else:
                from dlightrag.engine.rag.corpus.sources.azure_blob import AzureBlobDataSource

                source = AzureBlobDataSource(
                    connection_string=self.settings.blob_connection_string,
                    container_name=str(parts["container_name"]),
                )

        try:
            parser_path = (
                self._workspace_input_root()
                / remote_parser_input_path(
                    batch_root=Path(), source_uri=source_uri, key=display_filename
                ).name
            )
            parser_path.parent.mkdir(parents=True, exist_ok=True)
            prepared = PreparedIngestFile(
                parser_path=parser_path,
                source_uri=source_uri,
                download_locator=download_locator,
                display_filename=display_filename,
                title=title,
                author=author,
                metadata=metadata,
                replacement_doc_ids=replacement_doc_ids,
                replacement_ownership=replacement_ownership,
            )
            try:
                await source.amaterialize_document(source_document, parser_path)
                result = await self._ingestion_engine.aingest_files(
                    [prepared], replace=False, track_id=track_id
                )
                return self._single_file_result(result)
            finally:
                await asyncio.to_thread(_remove_remote_parser_sources, [prepared])
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                result = close()
                if isawaitable(result):
                    _ = await result

    async def _aingest_local_retry_locator(
        self,
        *,
        source_uri: str,
        download_locator: str,
        display_filename: str,
        title: str | None = None,
        author: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        replacement_doc_ids: tuple[str, ...] = (),
        replacement_ownership: tuple[tuple[str, str, str], ...] = (),
        track_id: str | None = None,
    ) -> dict[str, Any]:
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")

        source_path = self._retry_local_source_path(download_locator)
        item = PreparedIngestFile(
            parser_path=source_path,
            source_uri=source_uri,
            download_locator=download_locator,
            display_filename=display_filename,
            title=title,
            author=author,
            metadata=metadata,
            replacement_doc_ids=replacement_doc_ids,
            replacement_ownership=replacement_ownership,
        )
        result = await self._ingestion_engine.aingest_files(
            [item], replace=False, track_id=track_id
        )
        return self._single_file_result(result)

    async def adelete_files(
        self,
        *,
        file_paths: list[str] | None = None,
        filenames: list[str] | None = None,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        """Unified file deletion — DB records and physical files."""
        if not dry_run:
            self._require_writer("file deletion")
        self._ensure_initialized()
        from dlightrag.engine.rag.corpus.ingestion.cleanup import (
            cascade_delete,
            collect_deletion_context,
            remove_deleted_files,
        )

        identifiers = [*(file_paths or []), *(filenames or [])]
        results: list[dict[str, Any]] = []
        for identifier in identifiers:
            ctx = await collect_deletion_context(
                identifier=identifier,
                metadata_index=self._metadata_index,
                doc_status_lookup=self._doc_status_lookup,
            )
            if dry_run:
                results.append(
                    {
                        "identifier": identifier,
                        "status": "would_delete" if ctx.doc_ids else "not_found",
                        "dry_run": True,
                        "docs_deleted": 0,
                        "errors": [],
                        "matched_doc_ids": sorted(ctx.doc_ids),
                        "matched_file_paths": sorted(ctx.file_paths),
                        "sources_used": list(ctx.sources_used),
                    }
                )
                continue

            stats = await cascade_delete(
                ctx=ctx,
                lightrag=self._lightrag,
                metadata_index=self._metadata_index,
            )
            outcomes = [
                dict(item) for item in stats.get("outcomes") or () if isinstance(item, Mapping)
            ]
            outcome_states = {str(item.get("status") or "") for item in outcomes}
            if not ctx.doc_ids:
                status = "not_found"
            elif "waiting_for_repair" in outcome_states:
                status = "waiting_for_repair"
            elif "failed" in outcome_states or "rejected" in outcome_states:
                status = "rejected"
            else:
                try:
                    removed = remove_deleted_files(
                        ctx.file_paths,
                        str(self.settings.input_root / self.workspace_id),
                    )
                    stats["files_removed"] = removed
                    status = "deleted"
                except OSError:
                    stats.setdefault("errors", []).append("Physical source cleanup is ambiguous")
                    status = "waiting_for_repair"

            results.append({"identifier": identifier, "status": status, **stats})
        return results


async def _aiter_chunks[T](
    items: Iterable[T] | AsyncIterable[T],
    size: int,
) -> AsyncIterator[tuple[int, list[T]]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    batch_index = 0
    window: list[T] = []
    async for item in _aiter_items(items):
        window.append(item)
        if len(window) >= size:
            yield batch_index, window
            batch_index += 1
            window = []
    if window:
        yield batch_index, window


async def _aiter_items[T](items: Iterable[T] | AsyncIterable[T]) -> AsyncIterator[T]:
    if isinstance(items, AsyncIterable):
        async for item in items:
            yield item
        return
    for item in items:
        yield item


def _remove_empty_parents(path: Path, stop: Path) -> None:
    stop = stop.resolve()
    current = path.resolve()
    while current != stop and current.is_relative_to(stop):
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _remove_remote_parser_sources(items: list[PreparedIngestFile]) -> None:
    for item in items:
        parser_path = item.parser_path
        for candidate in (
            parser_path,
            parser_path.parent / PARSED_DIR_NAME / parser_path.name,
        ):
            try:
                if candidate.exists() and candidate.is_file():
                    candidate.unlink()
            except OSError:
                logger.debug("Failed to remove remote parser source: %s", candidate, exc_info=True)


__all__ = ["WorkspaceRag"]
