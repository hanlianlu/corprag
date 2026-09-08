"""Citation pattern matching and extraction.

Supports two formats:
- [ref]     — doc-level citations (LightRAG document format)
- [ref-idx] — chunk-level citations (DlightRAG granular format)
"""

import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from dlightrag.engine.answer.citations.contracts import CITATION_PATTERN, DOC_CITATION_PATTERN

from .indexer import CitationIndexer

logger = logging.getLogger(__name__)


def extract_cited_chunks(indexer: CitationIndexer, answer_text: str) -> dict[str, list[str]]:
    """Extract cited chunk_ids grouped by ref_id.

    Handles both [n] (all chunks for ref) and [n-m] (specific chunk).
    """
    positions: list[tuple[int, str, int | None]] = []
    for m in CITATION_PATTERN.finditer(answer_text):
        positions.append((m.start(), m.group(1), int(m.group(2))))
    for m in DOC_CITATION_PATTERN.finditer(answer_text):
        positions.append((m.start(), m.group(1), None))

    result: defaultdict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for _, ref_id, chunk_idx in sorted(positions, key=lambda item: item[0]):
        if chunk_idx is not None:
            chunk_id = indexer.get_chunk_id(ref_id, chunk_idx)
            if chunk_id is None:
                logger.debug("Invalid citation [%s-%d]: no chunk found", ref_id, chunk_idx)
                continue
            if (ref_id, chunk_id) in seen:
                continue
            seen.add((ref_id, chunk_id))
            result[ref_id].append(chunk_id)
            continue

        max_idx = indexer.get_max_chunk_idx(ref_id)
        if max_idx == 0:
            logger.debug("Invalid citation [%s]: no chunks found", ref_id)
        for idx in range(1, max_idx + 1):
            chunk_id = indexer.get_chunk_id(ref_id, idx)
            if chunk_id and (ref_id, chunk_id) not in seen:
                seen.add((ref_id, chunk_id))
                result[ref_id].append(chunk_id)

    return result


_HEADING_LINE_RE = re.compile(r"^#{1,6}[ \t]")


def claimless_chunk_ids(contexts: Iterable[dict[str, Any]]) -> frozenset[str]:
    """Chunk ids whose text is nothing but Markdown headings.

    Such an excerpt states no fact, so a claim can never be drawn from it. A
    heading-only chunk that carries an image is excluded -- the image is evidence.
    """
    claimless: set[str] = set()
    for ctx in contexts:
        chunk_id = ctx.get("chunk_id")
        if not chunk_id or ctx.get("image_data"):
            continue
        lines = [ln.strip() for ln in str(ctx.get("content") or "").split("\n") if ln.strip()]
        if lines and all(_HEADING_LINE_RE.match(ln) for ln in lines):
            claimless.add(str(chunk_id))
    return frozenset(claimless)


def clean_invalid_citations(
    indexer: CitationIndexer,
    answer_text: str,
    *,
    claimless_chunks: frozenset[str] = frozenset(),
) -> str:
    """Remove citations that reference non-existent chunks/docs.

    A marker resolving to a claimless excerpt is degraded to its document
    marker: the claim is supported by the document, just not by that excerpt.
    """

    def _replace_chunk(m: re.Match) -> str:
        ref_id = m.group(1)
        chunk_idx = int(m.group(2))
        chunk_id = indexer.get_chunk_id(ref_id, chunk_idx)
        if chunk_id is None:
            logger.debug("Removing invalid citation [%s-%d]", ref_id, chunk_idx)
            return ""
        if chunk_id in claimless_chunks:
            logger.debug("Degrading claimless citation [%s-%d] to [%s]", ref_id, chunk_idx, ref_id)
            return f"[{ref_id}]"
        return m.group(0)

    def _replace_doc(m: re.Match) -> str:
        ref_id = m.group(1)
        if indexer.get_max_chunk_idx(ref_id) > 0:
            return m.group(0)
        logger.debug("Removing invalid citation [%s]", ref_id)
        return ""

    text = CITATION_PATTERN.sub(_replace_chunk, answer_text)
    text = DOC_CITATION_PATTERN.sub(_replace_doc, text)
    return text


# Matches generated reference-section headings at a line boundary:
# # References, ## References, **References**, References:, References
_REFERENCES_HEADING_RE = re.compile(r"(?im)^\s{0,3}(?:#{1,6}\s*|\*{2})?references(?:\*{2})?[:\s]*$")


def strip_generated_references_section(answer_text: str) -> str:
    """Strip a model-generated trailing References section from answer text.

    DlightRAG builds cited sources deterministically from validated inline
    markers. A model-generated ``### References`` tail is therefore protocol
    noise, not a trusted data source.
    """
    match = None
    for candidate in _REFERENCES_HEADING_RE.finditer(answer_text):
        match = candidate
    if match is None:
        return answer_text
    return answer_text[: match.start()].rstrip()
