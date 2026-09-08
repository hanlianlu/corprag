# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared answer finalization for citation-aware transports."""

from dataclasses import dataclass, field

from dlightrag.engine.answer.citations.contracts import SourceReference
from dlightrag.engine.rag.retrieval import ContextRow, RetrievalContexts

from .indexer import CitationIndexer, build_citation_index
from .processor import CitationProcessor
from .source_builder import build_sources


@dataclass
class FinalizedAnswer:
    """Citation-processed answer plus projected sources."""

    answer: str
    sources: list[SourceReference]
    cited_chunks: dict[str, list[str]] = field(default_factory=dict)
    flat_contexts: list[ContextRow] = field(default_factory=list)
    all_sources: list[SourceReference] = field(default_factory=list)


def flatten_context_chunks(contexts: RetrievalContexts) -> list[ContextRow]:
    """Flatten retrieval context lists into citation processor input."""
    flat_contexts: list[ContextRow] = []
    for items in contexts.values():
        if isinstance(items, list):
            flat_contexts.extend(item for item in items if isinstance(item, dict))
    return flat_contexts


def finalize_answer(
    answer_text: str,
    contexts: RetrievalContexts,
    *,
    indexer: CitationIndexer | None = None,
    image_url_prefix: str | None = "/images",
    default_workspace: str | None = None,
) -> FinalizedAnswer:
    """Clean citation markers and return only cited sources.

    Raw ``contexts`` are used both to validate citation markers and construct
    internal sources. Transport adapters project separate public payloads.
    """
    flat_contexts = flatten_context_chunks(contexts)
    enriched_contexts = flat_contexts
    if flat_contexts:
        if indexer is None:
            indexer, enriched_contexts = build_citation_index(flat_contexts)
        else:
            enriched_contexts = indexer.inject_chunk_idx(flat_contexts)
    enriched_chunks = [ctx for ctx in enriched_contexts if ctx.get("chunk_id")]
    all_sources = build_sources(
        contexts,
        image_url_prefix=image_url_prefix,
        default_workspace=default_workspace,
        indexer=indexer,
        enriched_chunks=enriched_chunks,
    )

    if not answer_text or not flat_contexts:
        return FinalizedAnswer(
            answer=answer_text,
            sources=[],
            cited_chunks={},
            flat_contexts=enriched_contexts,
            all_sources=all_sources,
        )

    result = CitationProcessor(
        contexts=flat_contexts,
        available_sources=all_sources,
        indexer=indexer,
        enriched_contexts=enriched_contexts,
    ).process(answer_text)
    return FinalizedAnswer(
        answer=result.answer,
        sources=result.sources,
        cited_chunks=result.cited_chunks,
        flat_contexts=enriched_contexts,
        all_sources=all_sources,
    )


__all__ = ["FinalizedAnswer", "finalize_answer", "flatten_context_chunks"]
