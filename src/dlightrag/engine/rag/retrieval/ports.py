# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral retrieval interfaces."""

from typing import Any, Protocol

from dlightrag.engine.rag.retrieval.models import ContextRow, MetadataFilter, MetadataScope
from dlightrag.engine.rag.retrieval.results import RetrievalResult


class RetrievalBackend(Protocol):
    async def aretrieve(
        self,
        query: str,
        *,
        mode: str = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult: ...


class BM25Search(Protocol):
    async def search(
        self,
        query: str,
        *,
        scope: MetadataScope | None,
        top_k: int | None = None,
    ) -> list[ContextRow]: ...


class BM25ProfileSearch(Protocol):
    async def search_profile(
        self,
        query: str,
        *,
        profile_name: str,
        language: str | None,
        scope: MetadataScope | None,
        limit: int,
    ) -> list[ContextRow]: ...


class MetadataScopeStore(Protocol):
    """Resolve one normalized filter into scope facts without materializing ids."""

    async def resolve_scope(self, filters: MetadataFilter) -> MetadataScope: ...


class ScopedChunkReader(Protocol):
    """Read graph-referenced chunks under an active metadata scope.

    Returns one entry per requested id in positional order (duplicates
    preserved, ``None`` for missing or out-of-scope rows) so callers keep
    zipping results against the ids they asked for.
    """

    async def read_scoped(
        self,
        scope: MetadataScope | None,
        chunk_ids: list[str],
    ) -> list[dict[str, Any] | None]: ...


class CorpusChunkStore(MetadataScopeStore, Protocol):
    async def overwrite_chunk_vectors(
        self,
        vectors: dict[str, list[float]],
        *,
        embedding_dim: int,
    ) -> None: ...

    async def fetch_chunk_contents(self, chunk_ids: list[str]) -> list[dict[str, Any]]: ...

    async def update_chunk_bm25_languages(self, labels: dict[str, str]) -> None: ...


class FilteredVectorSearch(Protocol):
    async def search(
        self,
        embedding: list[float],
        *,
        scope: MetadataScope | None,
        top_k: int,
    ) -> list[dict[str, Any]]: ...

    async def ensure_document_scope_index(self) -> None: ...


__all__ = [
    "BM25Search",
    "BM25ProfileSearch",
    "CorpusChunkStore",
    "FilteredVectorSearch",
    "MetadataScopeStore",
    "RetrievalBackend",
    "ScopedChunkReader",
]
