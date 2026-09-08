"""Citation processor -- validates inline markers and projects cited sources."""

import logging
from dataclasses import dataclass, field
from typing import Any

from dlightrag.engine.answer.citations.contracts import SourceReference

from .indexer import CitationIndexer
from .parser import (
    claimless_chunk_ids,
    clean_invalid_citations,
    extract_cited_chunks,
    strip_generated_references_section,
)
from .source_builder import build_sources_from_chunks

logger = logging.getLogger(__name__)


@dataclass
class CitationResult:
    """Output of citation processing."""

    answer: str
    sources: list[SourceReference]
    cited_chunks: dict[str, list[str]] = field(default_factory=dict)


class CitationProcessor:
    """Process raw LLM answer + retrieval contexts into enriched citations."""

    def __init__(
        self,
        contexts: list[dict[str, Any]],
        available_sources: list[SourceReference],
        *,
        indexer: CitationIndexer | None = None,
        enriched_contexts: list[dict[str, Any]] | None = None,
    ) -> None:
        self._contexts = contexts
        self._available_sources = available_sources
        self._indexer = indexer or CitationIndexer()
        if indexer is None:
            self._indexer.build_index(contexts)
        self._enriched_contexts = (
            enriched_contexts
            if enriched_contexts is not None
            else self._indexer.inject_chunk_idx(contexts)
        )
        self._claimless_chunks = claimless_chunk_ids(contexts)

    def process(self, answer_text: str) -> CitationResult:
        """Clean citations, extract chunks, build source references."""
        answer_body = strip_generated_references_section(answer_text)
        cleaned = clean_invalid_citations(
            self._indexer, answer_body, claimless_chunks=self._claimless_chunks
        )
        cited_chunks = extract_cited_chunks(self._indexer, cleaned)
        sources = self._filter_cited_sources(cited_chunks)

        return CitationResult(
            answer=cleaned,
            sources=sources,
            cited_chunks=cited_chunks,
        )

    def _filter_cited_sources(self, cited_chunks: dict[str, list[str]]) -> list[SourceReference]:
        """Build SourceReference list with ChunkSnippet details."""
        if not cited_chunks or not self._available_sources:
            return []

        sources = build_sources_from_chunks(
            self._contexts,
            indexer=self._indexer,
            enriched_chunks=self._enriched_contexts,
            cited_chunks=cited_chunks,
            source_catalog=self._available_sources,
        )
        missing = set(cited_chunks) - {source.id for source in sources}
        for ref_id in sorted(missing):
            logger.warning("Cited ref_id=%s not in available_sources or contexts", ref_id)
        return sources
