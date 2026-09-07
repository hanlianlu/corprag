# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral corpus backend composition interfaces."""

import math
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag.engine.dependencies import TransientDependencyError
from dlightrag.engine.rag.corpus.contracts import DocStatusLookup
from dlightrag.engine.rag.corpus.metadata_index import MetadataIndexProtocol
from dlightrag.engine.rag.retrieval.ports import (
    BM25Search,
    CorpusChunkStore,
    FilteredVectorSearch,
    ScopedChunkReader,
)
from dlightrag.engine.rag.workspace.settings import RagSettings


class CorpusSchemaError(RuntimeError):
    """The deployed corpus schema is incompatible with this software revision."""


class CorpusUnavailableError(TransientDependencyError):
    """The configured corpus backend cannot currently be reached."""

    def __init__(self, detail: str | None = None) -> None:
        super().__init__("corpus_storage", detail or "Corpus storage is temporarily unavailable")


class WorkspaceWriteFencedError(RuntimeError):
    """A workspace write is refused while a promotion write fence is active.

    Retryable: the caller should surface the remaining fence duration and try
    again after it elapses.
    """

    def __init__(self, *, workspace: str, retry_after_seconds: float) -> None:
        self.workspace = workspace
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Workspace '{workspace}' is being promoted to dedicated storage; "
            f"retry after {int(math.ceil(retry_after_seconds))} seconds"
        )


class CorpusCoordination(Protocol):
    """Serialize workspace initialization and startup pipeline recovery."""

    def workspace_initialization(self) -> AbstractAsyncContextManager[None]: ...

    def pipeline_recovery(self) -> AbstractAsyncContextManager[None]: ...


class CorpusMaintenanceStore(Protocol):
    """Own storage-neutral workspace catalog maintenance operations."""

    async def initialize(self, *, validate_only: bool = False) -> None: ...

    async def clean_orphan_rows(self, workspace: str, *, dry_run: bool) -> int: ...

    async def list_workspace_records(self) -> tuple[dict[str, Any], ...]: ...

    async def list_workspace_records_page(
        self,
        *,
        after_workspace: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], bool]: ...

    async def workspace_exists(self, workspace: str) -> bool: ...

    async def register_workspace(
        self,
        *,
        workspace: str,
        display_name: str,
        embedding_model: str,
    ) -> None: ...

    async def get_workspace_record(self, workspace: str) -> dict[str, Any] | None:
        """Return the full registry row including storage/promotion facts."""
        ...

    def workspace_write_gate(self, workspace: str) -> AbstractAsyncContextManager[None]:
        """Gate one workspace write behind the promotion fence and drain protocol.

        Raises ``WorkspaceWriteFencedError`` when the workspace write fence is
        active. Corpus Mutation Runs use this same cross-process gate.
        """
        ...


class PromotionWorker(Protocol):
    """Background worker that drives hot-workspace promotion jobs."""

    def start(self) -> None: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkspaceCorpusStores:
    """Retrieval and ingestion stores attached to one LightRAG runtime."""

    metadata_index: MetadataIndexProtocol
    chunks: CorpusChunkStore
    filtered_vectors: FilteredVectorSearch | None
    bm25: BM25Search | None
    doc_status_lookup: DocStatusLookup
    bm25_languages: tuple[str, ...] = ()
    scoped_chunk_reader: ScopedChunkReader | None = None


@dataclass(frozen=True, slots=True)
class CorpusRuntimeModels:
    """Model callbacks required to construct one LightRAG runtime."""

    default_llm_func: Any
    embedding_func: Any
    role_llm_configs: Any


class CorpusRuntimeBinder(Protocol):
    """Construct and attach one backend-specific LightRAG runtime."""

    def create(self, *, models: CorpusRuntimeModels, settings: RagSettings) -> Any: ...

    async def attach(self, lightrag: Any) -> WorkspaceCorpusStores: ...


@dataclass(frozen=True, slots=True)
class WorkspaceCorpusBackend:
    """Coherent backend capabilities bound to one workspace."""

    workspace_id: str
    read_only: bool
    coordination: CorpusCoordination
    maintenance: CorpusMaintenanceStore
    runtime: CorpusRuntimeBinder
    promotion: PromotionWorker | None = None


__all__ = [
    "CorpusCoordination",
    "CorpusMaintenanceStore",
    "CorpusRuntimeBinder",
    "CorpusRuntimeModels",
    "CorpusSchemaError",
    "CorpusUnavailableError",
    "PromotionWorker",
    "WorkspaceCorpusBackend",
    "WorkspaceCorpusStores",
    "WorkspaceWriteFencedError",
]
