# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Inject page_idx metadata into LightRAG text_chunks after ingestion.

RAGAnything's separate_content() merges text blocks into one string,
discarding per-block page_idx. After LightRAG chunks this merged text,
we map each chunk back to its source page and upsert page_idx into KV store.
"""

from __future__ import annotations

import bisect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_page_boundary_map(
    content_list: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Build character-offset → page_idx map from text blocks.

    Mirrors separate_content()'s merging logic: non-empty text blocks
    joined with "\\n\\n".

    Returns:
        Sorted list of (char_offset, page_idx) tuples.
    """
    boundaries: list[tuple[int, int]] = []
    offset = 0
    first_text = True

    for item in content_list:
        if item.get("type", "text") != "text":
            continue

        text = item.get("text", "")
        if not text.strip():
            continue

        page_idx = item.get("page_idx")

        if not first_text:
            offset += 2  # "\n\n" separator

        if page_idx is not None:
            boundaries.append((offset, page_idx))

        offset += len(text)
        first_text = False

    return boundaries


def find_page_for_offset(
    boundaries: list[tuple[int, int]],
    offset: int,
) -> int | None:
    """Binary search boundaries for the page_idx at a character offset.

    Returns the page_idx of the last boundary whose offset <= the query offset,
    or None if boundaries is empty.
    """
    if not boundaries:
        return None

    # bisect on the offset component
    offsets = [b[0] for b in boundaries]
    idx = bisect.bisect_right(offsets, offset) - 1

    if idx < 0:
        # offset is before the first boundary — use first page
        return boundaries[0][1]

    return boundaries[idx][1]


def reconstruct_merged_text(content_list: list[dict[str, Any]]) -> str:
    """Reconstruct the merged text string from content_list.

    Replicates separate_content()'s text extraction logic exactly.
    """
    parts: list[str] = []
    for item in content_list:
        if item.get("type", "text") != "text":
            continue
        text = item.get("text", "")
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


async def inject_page_idx_to_chunks(
    lightrag: Any,
    doc_id: str,
    content_list: list[dict[str, Any]],
) -> int:
    """Inject page_idx into text_chunks KV store after LightRAG insertion.

    Args:
        lightrag: LightRAG instance (needs .doc_status and .text_chunks)
        doc_id: Document ID from insertion
        content_list: Original content_list with page_idx on text blocks

    Returns:
        Number of chunks updated.
    """
    if not lightrag:
        return 0

    if not hasattr(lightrag, "text_chunks") or not hasattr(lightrag, "doc_status"):
        return 0

    # Step 1: Build page boundary map
    boundaries = build_page_boundary_map(content_list)
    if not boundaries:
        logger.debug(f"No page boundaries found for doc {doc_id}, skipping")
        return 0

    # Step 2: Reconstruct merged text
    merged_text = reconstruct_merged_text(content_list)
    if not merged_text:
        return 0

    # Step 3: Get chunk IDs from doc_status
    doc_info = await lightrag.doc_status.get_by_id(doc_id)
    if not doc_info:
        logger.debug(f"No doc_status entry for {doc_id}")
        return 0

    chunk_ids = doc_info.get("chunks_list", [])
    if not chunk_ids:
        logger.debug(f"No chunks_list in doc_status for {doc_id}")
        return 0

    # Step 4: Get chunk data
    chunks_data = await lightrag.text_chunks.get_by_ids(chunk_ids)

    # Step 5: Sort by chunk_order_index for sequential search
    indexed: list[tuple[str, dict[str, Any]]] = []
    for cid, cdata in zip(chunk_ids, chunks_data, strict=True):
        if cdata is not None:
            indexed.append((cid, cdata))
    indexed.sort(key=lambda x: x[1].get("chunk_order_index", 0))

    # Step 6: Map each chunk to a page via position in merged text
    update_data: dict[str, dict[str, Any]] = {}
    search_pos = 0

    for chunk_id, chunk_data in indexed:
        content = chunk_data.get("content", "")
        if not content:
            continue

        # Find chunk's start position in merged text (sequential search)
        pos = merged_text.find(content, search_pos)
        if pos == -1:
            # Fallback: search from beginning (handles overlap edge cases)
            pos = merged_text.find(content)
        if pos == -1:
            continue

        search_pos = pos  # Advance for next chunk

        page_idx = find_page_for_offset(boundaries, pos)
        if page_idx is not None:
            # Read-merge-write: preserve existing fields, add page_idx
            updated = {k: v for k, v in chunk_data.items() if k != "_id"}
            updated["page_idx"] = page_idx
            update_data[chunk_id] = updated

    # Step 7: Batch upsert
    if update_data:
        await lightrag.text_chunks.upsert(update_data)

    return len(update_data)


__all__ = [
    "build_page_boundary_map",
    "find_page_for_offset",
    "reconstruct_merged_text",
    "inject_page_idx_to_chunks",
]
