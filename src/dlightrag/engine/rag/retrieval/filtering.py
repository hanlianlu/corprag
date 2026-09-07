# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Filtered storage wrappers for metadata-aware in-filtering.

Wraps the two LightRAG collaborators a document scope has to reach: ``chunks_vdb``
for the vector leg, and the ``text_chunks`` KV store for the knowledge-graph legs.
Uses contextvars for async-safe per-request state — concurrent requests don't
interfere, and ingest/delete paths run outside the retrieval scope so LightRAG's
mutation reads pass through unchanged.

Metadata filtering is a hard adapter-level in-filter constraint, not a
post-filter hint.
"""

import contextvars
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from dlightrag.engine.rag.retrieval import MetadataScope
from dlightrag.engine.rag.retrieval.ports import FilteredVectorSearch, ScopedChunkReader
from dlightrag.engine.rag.retrieval.visibility import (
    VisibleDocumentLookup,
    bounded_visibility_candidate_limit,
    retain_visible_chunks,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MetadataFilterStats:
    """Per-request facts the scoped legs report back to the retrieval trace."""

    kg_chunks_dropped: int = 0
    vector_strategy: str | None = None
    bm25_strategy: bool = False
    graph_strategy: bool = False
    vector_candidate_shortfall: int | None = None
    bm25_candidate_shortfall: int | None = None
    visibility_strategy: str | None = None
    visibility_dropped: int = 0
    visibility_shortfall: int | None = None


# Per-request filter state (async-safe: each coroutine gets its own value)
_active_filter: contextvars.ContextVar[MetadataScope | None] = contextvars.ContextVar(
    "_active_filter", default=None
)
_active_stats: contextvars.ContextVar[MetadataFilterStats | None] = contextvars.ContextVar(
    "_active_stats", default=None
)


def current_filter_stats() -> MetadataFilterStats | None:
    """Stats sink for the active scope, or None outside a retrieval request."""
    return _active_stats.get()


@asynccontextmanager
async def metadata_filter_scope(
    scope: MetadataScope | None,
) -> AsyncIterator[MetadataFilterStats]:
    """Activate visibility and an optional metadata filter for one retrieval.

    Within this context, chunk queries enforce the publication barrier and, when
    ``scope`` is present, restrict results to that document scope. The yielded
    stats record the strategies used by retrieval legs that actually run.
    """
    stats = MetadataFilterStats()
    filter_token = _active_filter.set(scope)
    stats_token = _active_stats.set(stats)
    try:
        yield stats
    finally:
        _active_filter.reset(filter_token)
        _active_stats.reset(stats_token)


class FilteredVectorStorage:
    """Wrap LightRAG chunk queries with the always-on visibility barrier.

    Native adapters push visibility (and an optional user metadata scope) into
    storage. Other adapters over-fetch one bounded neighbor window and ask the
    visibility lookup only about document ids present in that window.
    """

    def __init__(
        self,
        original: Any,
        embedding_func: Callable[..., Any],
        *,
        visibility_lookup: VisibleDocumentLookup,
        filtered_search: FilteredVectorSearch | None,
    ) -> None:
        self._original = original
        self._embedding_func = embedding_func
        self._visibility_lookup = visibility_lookup
        self._filtered_search = filtered_search

    async def query(
        self, query: str | Any, top_k: int, query_embedding: list[float] | None = None
    ) -> list[dict[str, Any]]:
        """Query under visibility plus the optional request metadata scope."""
        scope = _active_filter.get()
        if scope is not None and not scope:
            return []

        if self._filtered_search is not None:
            if query_embedding is None:
                if isinstance(query, str):
                    embeddings = await self._embedding_func([query], context="query")
                    emb = embeddings[0]
                    query_embedding = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                else:
                    query_embedding = query
            if query_embedding is None:
                raise RuntimeError("Filtered vector search requires a query embedding")
            rows = await self._filtered_search.search(
                query_embedding,
                scope=scope,
                top_k=top_k,
            )
            stats = _active_stats.get()
            if stats is not None:
                stats.visibility_strategy = "pushdown" if scope is not None else "bounded_pushdown"
                shortfall = max(0, int(top_k) - len(rows))
                if shortfall:
                    stats.visibility_shortfall = shortfall
            return rows

        overfetch = bounded_visibility_candidate_limit(top_k)
        candidates = await self._original.query(query, overfetch, query_embedding)
        rows = await retain_visible_chunks(
            candidates or [],
            lookup=self._visibility_lookup,
            requested_k=top_k,
            scope=scope,
        )
        stats = _active_stats.get()
        if stats is not None:
            stats.visibility_strategy = "postfilter"
            if len(rows) < int(top_k):
                # Only a short result proves every non-retained bounded
                # candidate was dropped rather than merely truncated.
                stats.visibility_dropped += max(0, len(candidates or []) - len(rows))
            shortfall = max(0, int(top_k) - len(rows))
            if shortfall:
                stats.visibility_shortfall = shortfall
        return rows

    async def ensure_doc_scope_index(self) -> None:
        if self._filtered_search is not None:
            await self._filtered_search.ensure_document_scope_index()

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to original (table_name, workspace, etc.)."""
        return getattr(self._original, name)


class FilteredChunkStore:
    """Wraps LightRAG's text_chunks KV store so a document scope reaches the KG legs.

    LightRAG's entity and relation legs never vector-search for their chunks: they
    read the chunk ids baked into graph nodes at ingest time and resolve them by
    primary key, so the chunks_vdb in-filter cannot see them. Under an active
    scope this wrapper replaces the KV ``get_by_ids`` round trip with one scoped
    chunk read that fuses the chunk fetch and the metadata guard in the database,
    returning the same positional list with ``None`` for missing or out-of-scope
    ids — the storage's own contract every caller already zips against.

    Filtering the vector lookup that *selects* those ids is not an option: LightRAG
    reads a short result from chunks_vdb.get_vectors_by_ids as storage corruption
    and falls back to an unfiltered ranking method.
    """

    def __init__(
        self,
        original: Any,
        *,
        visibility_lookup: VisibleDocumentLookup,
        scoped_reader: ScopedChunkReader | None = None,
    ) -> None:
        self._original = original
        self._visibility_lookup = visibility_lookup
        self._scoped_reader = scoped_reader

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        stats = _active_stats.get()
        if stats is None:
            # This wrapper is installed on LightRAG's own KV collaborator. Its
            # ingest, rollback, and delete paths are outside product retrieval
            # and must retain the upstream storage contract unchanged.
            return await self._original.get_by_ids(ids)

        scope = _active_filter.get()
        rows: list[dict[str, Any] | None]
        if scope is not None and not scope:
            rows = [None] * len(ids)
        elif self._scoped_reader is not None:
            rows = await self._scoped_reader.read_scoped(scope, list(ids))
        else:
            candidates = await self._original.get_by_ids(ids)
            doc_ids = list(
                dict.fromkeys(
                    doc_id
                    for row in candidates
                    if isinstance(row, dict)
                    and isinstance((doc_id := row.get("full_doc_id")), str)
                    and doc_id
                )
            )
            visible = (
                await self._visibility_lookup.visible_subset(doc_ids, scope=scope)
                if doc_ids
                else frozenset()
            )
            rows = [
                row
                if isinstance(row, dict)
                and isinstance((doc_id := row.get("full_doc_id")), str)
                and doc_id in visible
                else None
                for row in candidates
            ]
        dropped = sum(1 for row in rows if row is None)
        if dropped:
            stats.kg_chunks_dropped += dropped
            logger.info(
                "Visibility/metadata scope returned no chunk for %d of %d graph-referenced id(s)",
                dropped,
                len(ids),
            )
        stats.graph_strategy = True
        return rows

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to original (global_config, embedding_func, etc.)."""
        return getattr(self._original, name)
