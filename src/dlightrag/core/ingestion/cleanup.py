# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG deletion helpers.

Provides doc_id lookup for the deletion pipeline. The actual data cleanup
across all storage layers is handled by LightRAG's adelete_by_doc_id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlightrag.core.ingestion.hash_index import HashIndexProtocol

logger = logging.getLogger(__name__)


@dataclass
class DeletionContext:
    """Aggregated deletion context from available data sources."""

    identifier: str  # Original filename/path requested for deletion
    doc_ids: set[str] = field(default_factory=set)
    content_hashes: set[str] = field(default_factory=set)
    file_paths: set[str] = field(default_factory=set)
    sources_used: list[str] = field(default_factory=list)  # For audit trail


async def collect_deletion_context(
    identifier: str,
    hash_index: HashIndexProtocol | None,
    lightrag: Any = None,
) -> DeletionContext:
    """Find doc_id(s) for a file using hash index and LightRAG doc_status.

    Strategy order:
    1. Hash index - authoritative source for content_hash -> doc_id
    2. LightRAG doc_status - storage-agnostic lookup by file_path
       (works with PG, JSON, Redis, etc. — whatever backend is configured)
    """
    ctx = DeletionContext(identifier=identifier)
    basename = Path(identifier).name
    stem = Path(identifier).stem

    # Strategy 1: Hash index lookup (fastest, authoritative)
    if hash_index:
        doc_id, content_hash, file_path = hash_index.find_by_name(basename)
        if not doc_id:
            doc_id, content_hash, file_path = hash_index.find_by_path(identifier)

        if doc_id:
            ctx.doc_ids.add(doc_id)
            ctx.sources_used.append("hash_index")
        if content_hash:
            ctx.content_hashes.add(content_hash)
        if file_path:
            ctx.file_paths.add(file_path)

    # Strategy 2: LightRAG doc_status lookup (storage-agnostic)
    if lightrag and hasattr(lightrag, "doc_status"):
        doc_status = lightrag.doc_status
        try:
            # Try exact path match first
            result = await doc_status.get_doc_by_file_path(identifier)

            # If identifier is a basename, also try matching stored paths
            if result is None and not Path(identifier).is_absolute():
                # Query all processed docs and match by basename/stem
                from lightrag.base import DocStatus

                all_docs = await doc_status.get_docs_by_status(DocStatus.PROCESSED)
                for d_id, doc_info in all_docs.items():
                    fp = getattr(doc_info, "file_path", "") or ""
                    stored_name = Path(fp).name
                    stored_stem = Path(fp).stem

                    if stored_name == basename or stored_stem == stem:
                        result = {"id": d_id, "file_path": fp}
                        ctx.doc_ids.add(d_id)
                        ctx.file_paths.add(fp)

            if result and "doc_status" not in ctx.sources_used:
                # get_doc_by_file_path returns dict with id as key or 'id' field
                if isinstance(result, dict):
                    d_id = result.get("id") or result.get("doc_id")
                    fp = result.get("file_path", "")
                    if d_id:
                        ctx.doc_ids.add(d_id)
                    if fp:
                        ctx.file_paths.add(fp)
                    ctx.sources_used.append("doc_status")

        except Exception as e:
            logger.warning(f"LightRAG doc_status lookup failed for {identifier}: {e}")

    logger.info(
        f"Deletion context for {identifier}: "
        f"doc_ids={len(ctx.doc_ids)}, "
        f"file_paths={len(ctx.file_paths)}, sources={ctx.sources_used}"
    )

    return ctx


__all__ = ["DeletionContext", "collect_deletion_context"]
