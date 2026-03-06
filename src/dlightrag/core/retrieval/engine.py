# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval engine — composition-based wrapper over RAGAnything.

Provides structured retrieval (RetrievalResult) by composing a RAGAnything
instance rather than inheriting from it. Calls LightRAG's aquery_data/aquery_llm
for structured data and RAGAnything's multimodal query enhancement.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from lightrag import QueryParam

from dlightrag.utils.content_filters import filter_content_for_snippet

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Wrapper for RAG query results."""

    answer: str | None = field(default=None)
    contexts: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class RetrievalEngine:
    """Structured retrieval over a RAGAnything instance (composition, not inheritance).

    Uses:
    - rag.lightrag.aquery_data() for data-only retrieval
    - rag.lightrag.aquery_llm() for retrieval + LLM answer
    - rag._process_multimodal_query_content() for multimodal query enhancement
    """

    def __init__(self, rag: Any, config: Any) -> None:
        self.rag = rag
        self.config = config
        self._url_transformer: Callable[[str], str] | None = None

    async def aretrieve(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        *,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] | None = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        is_reretrieve: bool = False,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve structured data without generating answer."""
        lightrag = getattr(self.rag, "lightrag", None)
        if lightrag is None:
            logger.warning("LightRAG not initialized - no documents ingested yet")
            return RetrievalResult(
                answer=None, contexts={}, raw={"warning": "No documents ingested yet"}
            )

        enhanced_query = query
        if multimodal_content and hasattr(self.rag, "_process_multimodal_query_content"):
            enhanced_query = await self.rag._process_multimodal_query_content(
                query, multimodal_content
            )

        adjusted_top_k = top_k or self.config.top_k
        adjusted_chunk_top_k = chunk_top_k or self.config.chunk_top_k

        if is_reretrieve:
            enable_rerank = False
        else:
            enable_rerank = kwargs.pop("enable_rerank", self.config.enable_rerank)

        query_kwargs = {
            "top_k": adjusted_top_k,
            "chunk_top_k": adjusted_chunk_top_k,
            "enable_rerank": enable_rerank,
            "max_entity_tokens": kwargs.pop("max_entity_tokens", self.config.max_entity_tokens),
            "max_relation_tokens": kwargs.pop(
                "max_relation_tokens", self.config.max_relation_tokens
            ),
            "max_total_tokens": kwargs.pop("max_total_tokens", self.config.max_total_tokens),
            **kwargs,
        }

        query_param = QueryParam(mode=mode or self.config.default_mode, **query_kwargs)
        retrieval_data = await lightrag.aquery_data(enhanced_query, param=query_param)

        contexts = retrieval_data.get("data", {})
        raw: dict[str, Any] = retrieval_data if isinstance(retrieval_data, dict) else {}

        result = RetrievalResult(answer=None, contexts=contexts, raw=raw)

        return await augment_retrieval_result(
            result,
            str(self.config.working_dir_path),
            url_transformer=getattr(self, "_url_transformer", None),
            lightrag=lightrag,
        )

    async def aanswer(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        *,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] | None = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve contexts and generate an LLM answer."""
        lightrag = getattr(self.rag, "lightrag", None)
        if lightrag is None:
            logger.warning("LightRAG not initialized - no documents ingested yet")
            return RetrievalResult(
                answer=None, contexts={}, raw={"warning": "No documents ingested yet"}
            )

        enhanced_query = query
        if multimodal_content and hasattr(self.rag, "_process_multimodal_query_content"):
            enhanced_query = await self.rag._process_multimodal_query_content(
                query, multimodal_content
            )

        adjusted_top_k = top_k or self.config.top_k
        adjusted_chunk_top_k = chunk_top_k or self.config.chunk_top_k
        enable_rerank = kwargs.pop("enable_rerank", self.config.enable_rerank)

        # Truncate conversation history
        history = kwargs.pop("conversation_history", None)
        if history:
            max_msgs = self.config.max_conversation_turns * 2
            if len(history) > max_msgs:
                history = history[-max_msgs:]
            token_budget = self.config.max_conversation_tokens
            total = 0
            cutoff = 0
            for i in range(len(history) - 1, -1, -1):
                total += len(history[i].get("content", "")) // 4
                if total > token_budget:
                    cutoff = i + 1
                    break
            if cutoff:
                history = history[cutoff:]
            kwargs["conversation_history"] = history

        query_kwargs = {
            "top_k": adjusted_top_k,
            "chunk_top_k": adjusted_chunk_top_k,
            "enable_rerank": enable_rerank,
            "max_entity_tokens": kwargs.pop("max_entity_tokens", self.config.max_entity_tokens),
            "max_relation_tokens": kwargs.pop(
                "max_relation_tokens", self.config.max_relation_tokens
            ),
            "max_total_tokens": kwargs.pop("max_total_tokens", self.config.max_total_tokens),
            **kwargs,
        }

        query_param = QueryParam(mode=mode or self.config.default_mode, **query_kwargs)
        result_data = await lightrag.aquery_llm(enhanced_query, param=query_param)

        llm_response = result_data.get("llm_response", {})
        answer = llm_response.get("content")
        contexts = result_data.get("data", {})
        raw: dict[str, Any] = result_data if isinstance(result_data, dict) else {}

        result = RetrievalResult(answer=answer, contexts=contexts, raw=raw)

        return await augment_retrieval_result(
            result,
            str(self.config.working_dir_path),
            url_transformer=getattr(self, "_url_transformer", None),
            lightrag=lightrag,
        )

    async def aanswer_stream(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        *,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] | None = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], AsyncIterator[str]]:
        """Two-phase streaming: retrieve contexts, then stream LLM answer.

        Returns:
            (contexts, raw, token_iterator) — contexts/raw are complete,
            token_iterator yields LLM chunks as they arrive.
        """
        lightrag = getattr(self.rag, "lightrag", None)
        if lightrag is None:
            raise RuntimeError("LightRAG not initialized - no documents ingested yet")

        enhanced_query = query
        if multimodal_content and hasattr(self.rag, "_process_multimodal_query_content"):
            enhanced_query = await self.rag._process_multimodal_query_content(
                query, multimodal_content
            )

        adjusted_top_k = top_k or self.config.top_k
        adjusted_chunk_top_k = chunk_top_k or self.config.chunk_top_k
        enable_rerank = kwargs.pop("enable_rerank", self.config.enable_rerank)

        # Truncate conversation history (same logic as aanswer)
        history = kwargs.pop("conversation_history", None)
        if history:
            max_msgs = self.config.max_conversation_turns * 2
            if len(history) > max_msgs:
                history = history[-max_msgs:]
            token_budget = self.config.max_conversation_tokens
            total = 0
            cutoff = 0
            for i in range(len(history) - 1, -1, -1):
                total += len(history[i].get("content", "")) // 4
                if total > token_budget:
                    cutoff = i + 1
                    break
            if cutoff:
                history = history[cutoff:]
            kwargs["conversation_history"] = history

        query_kwargs = {
            "top_k": adjusted_top_k,
            "chunk_top_k": adjusted_chunk_top_k,
            "enable_rerank": enable_rerank,
            "max_entity_tokens": kwargs.pop("max_entity_tokens", self.config.max_entity_tokens),
            "max_relation_tokens": kwargs.pop(
                "max_relation_tokens", self.config.max_relation_tokens
            ),
            "max_total_tokens": kwargs.pop("max_total_tokens", self.config.max_total_tokens),
            **kwargs,
        }

        # Phase 1: Retrieve contexts (non-streaming)
        query_param = QueryParam(mode=mode or self.config.default_mode, **query_kwargs)
        retrieval_data = await lightrag.aquery_data(enhanced_query, param=query_param)

        contexts = retrieval_data.get("data", {})
        raw: dict[str, Any] = retrieval_data if isinstance(retrieval_data, dict) else {}

        # Augment with sources/media
        temp_result = RetrievalResult(answer=None, contexts=contexts, raw=raw)
        augmented = await augment_retrieval_result(
            temp_result,
            str(self.config.working_dir_path),
            url_transformer=getattr(self, "_url_transformer", None),
            lightrag=lightrag,
        )

        # Phase 2: Stream LLM answer
        stream_param = QueryParam(
            mode=mode or self.config.default_mode, stream=True, **query_kwargs
        )
        token_iter = await lightrag.aquery(enhanced_query, param=stream_param)

        return augmented.contexts, augmented.raw, token_iter


# --- Augmentation utilities (moved from old retrieval/engine.py) ---


def _extract_rag_relative(file_path: str, working_dir: str | None = None) -> str | None:
    """Extract relative path within the RAG working directory.

    E.g., "/abs/path/dlightrag_storage/sources/local/file.pdf" -> "sources/local/file.pdf"
    """
    if working_dir:
        wd = working_dir.rstrip("/")
        idx = file_path.find(wd)
        if idx != -1:
            return file_path[idx + len(wd) :].lstrip("/")

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
                "caption": match_caption.group(1).strip() if match_caption is not None else None,
            }

    return list(sources.values()), list(media.values())


async def augment_retrieval_result(
    result: RetrievalResult,
    rag_working_dir: str | None = None,
    url_transformer: Callable[[str], str] | None = None,
    lightrag: Any = None,
) -> RetrievalResult:
    """Attach sources/media (download URLs) derived from contexts into result.raw.

    Sources are keyed by document reference_id (one entry per document).
    Media are keyed by image path hash (allows multiple images per document).

    Args:
        result: The retrieval result to augment
        rag_working_dir: Base directory for RAG storage
        url_transformer: Optional callback for URL generation (e.g., signed URLs)
        lightrag: LightRAG instance for storage-agnostic KV lookups
    """
    chunk_contexts = result.contexts.get("chunks", []) if result.contexts else []

    # Look up page_idx from KV store via storage-agnostic API
    if lightrag and hasattr(lightrag, "text_chunks") and chunk_contexts:
        chunk_ids = [ctx.get("chunk_id") for ctx in chunk_contexts if ctx.get("chunk_id")]
        if chunk_ids:
            try:
                chunk_data_list = await lightrag.text_chunks.get_by_ids(chunk_ids)
                page_idx_map: dict[str, int | None] = {}
                for cid, cdata in zip(chunk_ids, chunk_data_list, strict=True):
                    if cdata and cdata.get("page_idx") is not None:
                        page_idx_map[cid] = cdata["page_idx"] + 1  # 0-based -> 1-based
                for ctx in chunk_contexts:
                    cid = ctx.get("chunk_id")
                    if cid and cid in page_idx_map:
                        ctx["page_idx"] = page_idx_map[cid]
            except Exception as e:
                logger.warning(f"Failed to load page_idx from KV store: {e}")

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


__all__ = [
    "RetrievalEngine",
    "RetrievalResult",
    "augment_retrieval_result",
    "build_sources_and_media_from_contexts",
]
