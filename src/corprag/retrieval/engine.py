# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval pipeline for RAG queries.

Provides EnhancedRAGAnything (data-only retrieval) and augment_retrieval_result
(source/media extraction with optional URL transformation).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Literal

from lightrag import QueryParam
from raganything import RAGAnything

from corprag.utils.content_filters import filter_content_for_snippet

logger = logging.getLogger(__name__)


def _extract_rag_relative(file_path: str, working_dir: str | None = None) -> str | None:
    """Extract relative path within the RAG working directory.

    E.g., "/abs/path/corprag_storage/sources/local/file.pdf" -> "sources/local/file.pdf"
    """
    if working_dir:
        wd = working_dir.rstrip("/")
        idx = file_path.find(wd)
        if idx != -1:
            return file_path[idx + len(wd):].lstrip("/")

    # Fallback: look for sources/ marker
    for marker in ("sources/", "artifacts/"):
        idx = file_path.find(marker)
        if idx != -1:
            return file_path[idx:]

    return None


def _to_download_url(
    path: str,
    url_transformer: Callable[[str], str] | None = None,
    working_dir: str | None = None,
) -> str:
    """Convert a file/azure path into a URL.

    If url_transformer is provided (e.g., for signed URLs),
    it handles URL generation. Otherwise, returns a file:// relative URL.
    """
    if url_transformer:
        return url_transformer(path)

    # Strip file:// scheme if present
    inner_path = path
    if inner_path.startswith("file://"):
        inner_path = inner_path[7:]

    # Azure blob paths pass through unchanged
    if path.startswith("azure://"):
        return path

    # Extract relative path for portability
    relative = _extract_rag_relative(inner_path, working_dir)
    return f"file://{relative}" if relative else f"file://{inner_path}"


@dataclass
class RetrievalResult:
    """Wrapper for RAG query results."""

    answer: str | None = field(default=None)
    contexts: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _load_kv_store_page_indices(rag_working_dir: str) -> dict[str, int | None]:
    """Load page_idx mapping from kv_store_text_chunks.json.

    Converts 0-based page_idx to 1-based for display (PDF pages are 1-indexed).
    """
    kv_path = Path(rag_working_dir) / "kv_store_text_chunks.json"

    if not kv_path.exists():
        return {}

    try:
        with open(kv_path) as f:
            kv_store = json.load(f)
        return {
            chunk_id: (chunk_data["page_idx"] + 1)
            if chunk_data.get("page_idx") is not None
            else None
            for chunk_id, chunk_data in kv_store.items()
        }
    except Exception as e:
        logger.warning(f"Failed to load kv_store for page_idx: {e}")
        return {}


def build_sources_and_media_from_contexts(
    contexts: list[dict[str, Any]],
    url_transformer: Callable[[str], str] | None = None,
    rag_working_dir: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build sources/media lists from chunk contexts.

    Sources are keyed by document reference_id (one entry per document).
    Media are keyed by image path hash (allows multiple images per document).
    """
    sources: dict[str, dict[str, Any]] = {}
    media: dict[str, dict[str, Any]] = {}

    for ctx in contexts:
        if "chunk_id" not in ctx:
            continue

        file_path = ctx.get("file_path")
        ref_id = ctx.get("reference_id")
        chunk_id = ctx.get("chunk_id")

        if not file_path or not ref_id:
            continue

        ref_id = str(ref_id)

        if ref_id not in sources:
            content = ctx.get("content") or ""
            snippet = filter_content_for_snippet(content, max_chars=100)
            sources[ref_id] = {
                "id": ref_id,
                "type": "file",
                "title": Path(str(file_path)).name,
                "path": str(file_path),
                "url": _to_download_url(
                    str(file_path),
                    url_transformer=url_transformer,
                    working_dir=rag_working_dir,
                ),
                "snippet": snippet,
                "chunk_ids": [],
            }

        # Collect chunk ID for this source
        if chunk_id and chunk_id not in sources[ref_id]["chunk_ids"]:
            sources[ref_id]["chunk_ids"].append(chunk_id)

        # Extract images from chunk content
        content = ctx.get("content", "")
        match_img = re.search(r"Image Path:\s*(.*)", content)
        if match_img:
            img_path = match_img.group(1).strip()
            match_caption = re.search(r"[Cc]aption:\s*(.*)", content)

            media_key = hashlib.md5(img_path.encode()).hexdigest()[:12]

            media[media_key] = {
                "id": media_key,
                "type": "image",
                "reference_id": ref_id,
                "source_chunk_id": chunk_id,
                "title": Path(str(file_path)).name,
                "path": img_path,
                "url": _to_download_url(
                    img_path,
                    url_transformer=url_transformer,
                    working_dir=rag_working_dir,
                ),
                "caption": match_caption.group(1).strip()
                if match_caption is not None
                else None,
            }

    return list(sources.values()), list(media.values())


def augment_retrieval_result(
    result: RetrievalResult,
    rag_working_dir: str | None = None,
    url_transformer: Callable[[str], str] | None = None,
) -> RetrievalResult:
    """Attach sources/media (download URLs) derived from contexts into result.raw.

    Sources are keyed by document reference_id (one entry per document).
    Media are keyed by image path hash (allows multiple images per document).

    Args:
        result: The retrieval result to augment
        rag_working_dir: Base directory for RAG storage
        url_transformer: Optional callback for URL generation (e.g., signed URLs)
    """
    chunk_contexts = result.contexts.get("chunks", []) if result.contexts else []

    # Load page_idx mapping from kv_store once for all chunks
    page_idx_map: dict[str, int | None] = {}
    if rag_working_dir:
        page_idx_map = _load_kv_store_page_indices(rag_working_dir)

    for ctx in chunk_contexts:
        chunk_id = ctx.get("chunk_id")
        if chunk_id and chunk_id in page_idx_map and page_idx_map[chunk_id] is not None:
            ctx["page_idx"] = page_idx_map[chunk_id]

    sources, media = build_sources_and_media_from_contexts(
        chunk_contexts,
        url_transformer=url_transformer,
        rag_working_dir=rag_working_dir,
    )

    if sources or media:
        raw = dict(result.raw or {})
        if sources:
            raw["sources"] = sources
        if media:
            raw["media"] = media
        result.raw = raw

    return result


class EnhancedRAGAnything(RAGAnything):
    """Extended RAGAnything with data-only retrieval.

    ONLY FOR RETRIEVAL - NOT FOR INGESTION.
    Unifies results to RetrievalResult(answer, contexts, raw).
    """

    async def aquery_data_with_multimodal(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        *,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix",
        **kwargs: Any,
    ) -> RetrievalResult:
        """Multimodal query that captures retrieval data without answer generation."""
        # Handle empty index gracefully
        if not getattr(self, "lightrag", None):
            self.logger.warning("LightRAG not initialized - no documents ingested yet")
            return RetrievalResult(
                answer=None,
                contexts={},
                raw={"warning": "No documents ingested yet"},
            )

        # 1) Enhance query with multimodal content if provided
        enhanced_query = query
        if multimodal_content:
            enhanced_query = await self._process_multimodal_query_content(
                query,
                multimodal_content,
            )
            self.logger.info(
                "Enhanced query with multimodal content: %d chars",
                len(enhanced_query),
            )

        # 2) Get structured retrieval data
        query_param = QueryParam(mode=mode, **kwargs)  # type: ignore
        retrieval_data = await self.lightrag.aquery_data(  # type: ignore[func-returns-value]
            enhanced_query,
            param=query_param,
        )

        # 3) Extract data only as contexts
        contexts = retrieval_data.get("data", {})

        # raw keeps the complete original data structure
        raw: dict[str, Any] = retrieval_data if isinstance(retrieval_data, dict) else {}

        return RetrievalResult(
            answer=None,
            contexts=contexts,
            raw=raw,
        )


__all__ = [
    "EnhancedRAGAnything",
    "RetrievalResult",
    "augment_retrieval_result",
    "build_sources_and_media_from_contexts",
]
