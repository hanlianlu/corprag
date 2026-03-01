# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG deletion helpers.

Provides doc_id lookup for the deletion pipeline. The actual data cleanup
across all storage layers is handled by LightRAG's adelete_by_doc_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

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
    rag_working_dir: Path,
    hash_index: Any,
    lightrag: Any = None,
) -> DeletionContext:
    """Find doc_id(s) for a file using hash index and KV doc_status fallback.

    Strategy order:
    1. Hash index - authoritative source for content_hash -> doc_id
    2. KV store doc_status - filename matching (catches additional doc_ids)
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

    # Strategy 2: KV store doc_status lookup (JSON fallback for non-PG backends)
    doc_status_path = rag_working_dir / "kv_store_doc_status.json"
    if doc_status_path.exists():
        try:
            data = json.loads(doc_status_path.read_text())
            for d_id, status in data.items():
                fp = status.get("file_path", "")
                stored_name = Path(fp).name
                stored_stem = Path(fp).stem

                if stored_name == basename or stored_stem == stem:
                    ctx.doc_ids.add(d_id)
                    ctx.file_paths.add(fp)
                    if "doc_status" not in ctx.sources_used:
                        ctx.sources_used.append("doc_status")
        except Exception as e:
            logger.warning(f"Could not read doc_status for {identifier}: {e}")

    logger.info(
        f"Deletion context for {identifier}: "
        f"doc_ids={len(ctx.doc_ids)}, "
        f"file_paths={len(ctx.file_paths)}, sources={ctx.sources_used}"
    )

    return ctx


__all__ = ["DeletionContext", "collect_deletion_context"]
