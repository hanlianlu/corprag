# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified retrieval orchestration for LightRAG mix + DlightRAG BM25 and visual legs."""

import asyncio
import logging
from collections import Counter
from typing import Any

from dlightrag.engine.rag.retrieval import (
    ContextRow,
    MetadataFilter,
    MetadataScope,
    RetrievalResult,
    format_bm25_top,
    rrf_fuse,
)
from dlightrag.engine.rag.retrieval.filtering import metadata_filter_scope
from dlightrag.engine.rag.retrieval.ports import BM25Search, MetadataScopeStore, RetrievalBackend
from dlightrag.engine.rag.retrieval.visual import (
    DirectVisualRetriever,
    PreparedVisualQuery,
    VisualEmbeddingDomain,
)

logger = logging.getLogger(__name__)


def _chunk_ids(chunks: list[ContextRow]) -> set[str]:
    return {
        str(chunk_id) for chunk in chunks if (chunk_id := chunk.get("chunk_id") or chunk.get("id"))
    }


def _multi_source_count(rankings: list[list[ContextRow]]) -> int:
    """Chunks more than one leg retrieved — the agreement RRF rewards."""
    hits: Counter[str] = Counter()
    for ranking in rankings:
        hits.update(_chunk_ids(ranking))
    return sum(1 for count in hits.values() if count > 1)


def _scope_trace(scope: MetadataScope | None) -> dict[str, Any]:
    """The one trace vocabulary for a resolved metadata scope."""
    if scope is None:
        return {
            "metadata_match_exists": None,
            "metadata_candidate_count": None,
            "metadata_candidate_count_exact": None,
            "metadata_candidate_count_lower_bound": None,
        }
    return {
        "metadata_match_exists": scope.doc_exists,
        "metadata_candidate_count": scope.candidate_count,
        "metadata_candidate_count_exact": scope.candidate_count_exact,
        # A capped probe is a lower bound; it is never rendered as an exact total.
        "metadata_candidate_count_lower_bound": (
            scope.render_candidate_count() if not scope.candidate_count_exact else None
        ),
    }


class UnifiedRetriever:
    """Run retrieval-wide metadata filtering, the retrieval legs, and fusion.

    The legs stay independent until one RRF pass: pre-merging any pair would
    collapse its two votes into one and distort the ranks the survivors carry
    into fusion.
    """

    def __init__(
        self,
        *,
        backend: RetrievalBackend,
        bm25: BM25Search | None,
        stores: MetadataScopeStore,
        visual: DirectVisualRetriever | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._backend = backend
        self._bm25 = bm25
        self._visual = visual
        self._stores = stores
        self._rrf_k = rrf_k

    @property
    def visual_embedding_domain(self) -> VisualEmbeddingDomain | None:
        """Expose the enabled direct-visual compatibility domain."""
        return self._visual.embedding_domain if self._visual is not None else None

    async def prepare_visual_query(
        self, query_image_blocks: list[dict[str, Any]]
    ) -> PreparedVisualQuery | None:
        """Prepare query images without touching this workspace's vector store."""
        if self._visual is None:
            return None
        return await self._visual.prepare(query_image_blocks)

    async def aretrieve(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None = None,
        metadata_filter_source: str | None = None,
        bm25_query: str | None = None,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        prepared_visual_query: PreparedVisualQuery | None = None,
        query_image_blocks: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        scope = await self._resolve_candidates(metadata_filter)
        trace: dict[str, Any] = {
            "metadata_filter_source": metadata_filter_source,
            **_scope_trace(scope),
            "metadata_filter_relaxed": False,
        }
        if scope is not None and not scope:
            if metadata_filter_source == "llm_inferred":
                scope = None
                trace["metadata_filter_relaxed"] = True
            else:
                trace["strict_filter_empty"] = True
                return RetrievalResult(trace=trace)

        chunk_candidate_limit = chunk_top_k or top_k
        lexical_query = bm25_query or query
        async with metadata_filter_scope(scope) as filter_stats:
            lightrag_task = asyncio.create_task(
                self._backend.aretrieve(
                    query,
                    mode="mix",
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    **kwargs,
                )
            )
            bm25_task = (
                asyncio.create_task(
                    self._bm25.search(
                        lexical_query,
                        scope=scope,
                        top_k=chunk_candidate_limit,
                    )
                )
                if self._bm25 is not None
                else None
            )
            visual_task: asyncio.Task[list[ContextRow]] | None = None
            if self._visual is not None:
                if prepared_visual_query is not None:
                    visual_task = asyncio.create_task(
                        self._visual.search_prepared(prepared_visual_query)
                    )
                elif query_image_blocks:
                    # Convenience path below the typed prepared boundary for
                    # direct engine callers. Application retrieval never uses it.
                    visual_task = asyncio.create_task(self._visual.search(query_image_blocks))

            lightrag_error: Exception | None = None
            try:
                lightrag_result = await lightrag_task
            except Exception as exc:
                lightrag_error = exc
                if bm25_task is None:
                    logger.error(
                        "LightRAG retrieval failed and BM25 is disabled",
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        "LightRAG retrieval failed; falling back to BM25-only",
                        exc_info=True,
                    )
                lightrag_result = RetrievalResult(
                    trace={
                        "lightrag_error": True,
                        "lightrag_error_type": type(exc).__name__,
                    },
                )
            bm25_error: Exception | None = None
            try:
                bm25_chunks = await bm25_task if bm25_task is not None else []
            except Exception as exc:
                bm25_error = exc
                bm25_chunks = []
                logger.warning("BM25 retrieval failed; continuing without BM25", exc_info=True)
            visual_chunks = await visual_task if visual_task is not None else []
            if lightrag_error is not None:
                if bm25_task is None:
                    raise lightrag_error
                if bm25_error is not None:
                    raise lightrag_error from bm25_error

        trace.update(getattr(lightrag_result, "trace", {}) or {})
        trace["bm25_enabled"] = self._bm25 is not None
        trace["bm25_query"] = lexical_query if self._bm25 is not None else None
        trace["metadata_kg_chunks_dropped"] = filter_stats.kg_chunks_dropped
        trace["visibility_strategy"] = filter_stats.visibility_strategy
        trace["visibility_dropped"] = filter_stats.visibility_dropped
        trace["visibility_shortfall"] = filter_stats.visibility_shortfall
        strategies = [
            strategy
            for strategy in (
                filter_stats.vector_strategy,
                "bm25" if filter_stats.bm25_strategy else None,
                "scoped_graph" if filter_stats.graph_strategy else None,
            )
            if strategy
        ]
        trace["metadata_execution_strategy"] = "+".join(strategies) if strategies else None
        shortfall: dict[str, int] = {}
        if filter_stats.vector_candidate_shortfall is not None:
            shortfall["vector"] = filter_stats.vector_candidate_shortfall
        if filter_stats.bm25_candidate_shortfall is not None:
            shortfall["bm25"] = filter_stats.bm25_candidate_shortfall
        trace["metadata_candidate_shortfall"] = shortfall or None
        if bm25_error is not None:
            trace["bm25_error_type"] = type(bm25_error).__name__
        trace["bm25_chunk_count"] = len(bm25_chunks)
        trace["direct_visual_chunk_count"] = len(visual_chunks)
        lightrag_mix_chunks = lightrag_result.contexts.get("chunks", [])
        trace["lightrag_mix_chunk_count"] = len(lightrag_mix_chunks)
        rankings = [
            ranking for ranking in (lightrag_mix_chunks, visual_chunks, bm25_chunks) if ranking
        ]
        fused = rrf_fuse(rankings, k=self._rrf_k)
        lightrag_result.contexts["chunks"] = fused
        trace["fused_chunk_count"] = len(fused)
        if scope is not None and metadata_filter_source == "llm_inferred" and not fused:
            relaxed = await self.aretrieve(
                query,
                metadata_filter=None,
                metadata_filter_source=None,
                bm25_query=bm25_query,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                prepared_visual_query=prepared_visual_query,
                query_image_blocks=query_image_blocks,
                **kwargs,
            )
            relaxed.trace["metadata_filter_source"] = metadata_filter_source
            relaxed.trace.update(_scope_trace(scope))
            relaxed.trace["metadata_filter_relaxed"] = True
            return relaxed
        trace["fused_multi_source_count"] = _multi_source_count(rankings)
        logger.info(
            "[Retriever] mix: bm25_enabled=%s bm25_query=%r filter_source=%s "
            "metadata_scope=%s filter_relaxed=%s kg_chunks_dropped=%d "
            "execution_strategy=%s candidate_shortfall=%s "
            "lightrag_mix_chunks=%d visual_chunks=%d bm25_chunks=%d fused_chunks=%d "
            "multi_source=%d bm25_top=%s",
            self._bm25 is not None,
            lexical_query if self._bm25 is not None else None,
            metadata_filter_source,
            f"{scope.render_candidate_count()}chunk" if scope is not None else "all",
            trace.get("metadata_filter_relaxed", False),
            filter_stats.kg_chunks_dropped,
            trace.get("metadata_execution_strategy"),
            trace.get("metadata_candidate_shortfall"),
            trace["lightrag_mix_chunk_count"],
            trace["direct_visual_chunk_count"],
            trace["bm25_chunk_count"],
            trace["fused_chunk_count"],
            trace["fused_multi_source_count"],
            format_bm25_top(bm25_chunks),
        )
        lightrag_result.trace = trace
        return lightrag_result

    async def _resolve_candidates(
        self, metadata_filter: MetadataFilter | None
    ) -> MetadataScope | None:
        """Resolve filter facts without materializing document ids across the store seam."""
        if metadata_filter is None or metadata_filter.is_empty():
            return None
        scope = await self._stores.resolve_scope(metadata_filter)
        logger.info(
            "[MetadataPath] filters matched_any=%s candidate_chunks=%s filename_mode=%s",
            scope.doc_exists,
            scope.render_candidate_count(),
            scope.filename_mode,
        )
        return scope
