# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Federated retrieval across multiple workspaces.

Orchestrates parallel queries to multiple RAGService instances (one per
workspace) and merges results via round-robin interleaving to ensure
fair representation from each workspace.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from dlightrag.retrieval.engine import RetrievalResult

logger = logging.getLogger(__name__)

# Type alias for RBAC hook: given requested workspaces, return accessible subset
WorkspaceFilter = Callable[[list[str]], Awaitable[list[str]]]


def merge_results(
    results: list[RetrievalResult],
    workspaces: list[str],
    chunk_top_k: int | None = None,
) -> RetrievalResult:
    """Merge multiple RetrievalResults via round-robin interleaving.

    Each chunk/entity/relation is tagged with ``_workspace`` to identify
    its source. Results are interleaved: ws_a[0], ws_b[0], ws_a[1], ws_b[1]...
    then truncated to ``chunk_top_k``.

    Args:
        results: One RetrievalResult per workspace (same order as workspaces).
        workspaces: Workspace names corresponding to each result.
        chunk_top_k: Maximum number of chunks in merged output. None = no limit.
    """
    # Tag and collect chunks from each workspace
    per_ws_chunks: list[list[dict[str, Any]]] = []
    for result, ws in zip(results, workspaces, strict=True):
        chunks = result.contexts.get("chunks", [])
        tagged = []
        for chunk in chunks:
            c = dict(chunk)
            c["_workspace"] = ws
            tagged.append(c)
        per_ws_chunks.append(tagged)

    # Round-robin interleave
    merged_chunks: list[dict[str, Any]] = []
    max_len = max((len(cs) for cs in per_ws_chunks), default=0)
    for i in range(max_len):
        for ws_chunks in per_ws_chunks:
            if i < len(ws_chunks):
                merged_chunks.append(ws_chunks[i])

    # Truncate
    if chunk_top_k is not None:
        merged_chunks = merged_chunks[:chunk_top_k]

    # Merge sources with workspace tag
    merged_sources: list[dict[str, Any]] = []
    for result, ws in zip(results, workspaces, strict=True):
        for source in result.raw.get("sources", []):
            s = dict(source)
            s["_workspace"] = ws
            merged_sources.append(s)

    # Merge media with workspace tag
    merged_media: list[dict[str, Any]] = []
    for result, ws in zip(results, workspaces, strict=True):
        for media in result.raw.get("media", []):
            m = dict(media)
            m["_workspace"] = ws
            merged_media.append(m)

    # Merge entities/relations (round-robin, same as chunks)
    merged_entities = _round_robin_merge_key(results, workspaces, "entities")
    merged_relations = _round_robin_merge_key(results, workspaces, "relationships")

    # Build merged answer (concatenate non-None answers)
    answers = [r.answer for r in results if r.answer]
    merged_answer = "\n\n---\n\n".join(answers) if answers else None

    return RetrievalResult(
        answer=merged_answer,
        contexts={
            "chunks": merged_chunks,
            "entities": merged_entities,
            "relationships": merged_relations,
        },
        raw={
            "sources": merged_sources,
            "media": merged_media,
            "workspaces": workspaces,
        },
    )


def _round_robin_merge_key(
    results: list[RetrievalResult],
    workspaces: list[str],
    key: str,
) -> list[dict[str, Any]]:
    """Round-robin merge a specific context key across results."""
    per_ws: list[list[dict[str, Any]]] = []
    for result, ws in zip(results, workspaces, strict=True):
        items = result.contexts.get(key, [])
        tagged = []
        for item in items:
            d = dict(item)
            d["_workspace"] = ws
            tagged.append(d)
        per_ws.append(tagged)

    merged: list[dict[str, Any]] = []
    max_len = max((len(items) for items in per_ws), default=0)
    for i in range(max_len):
        for ws_items in per_ws:
            if i < len(ws_items):
                merged.append(ws_items[i])
    return merged


async def federated_retrieve(
    query: str,
    workspaces: list[str],
    get_service: Callable[[str], Awaitable[Any]],
    *,
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix",
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    workspace_filter: WorkspaceFilter | None = None,
    **kwargs: Any,
) -> RetrievalResult:
    """Execute federated retrieval across multiple workspaces.

    Args:
        query: The search query.
        workspaces: List of workspace names to search.
        get_service: Async callable that returns a RAGService for a workspace name.
        mode: LightRAG query mode.
        top_k: Per-workspace top_k for vector search.
        chunk_top_k: Final merged chunk count limit.
        workspace_filter: Optional RBAC filter — given requested workspaces,
            returns the accessible subset. Default: all workspaces accessible.
        **kwargs: Additional kwargs passed to each RAGService.aretrieve().
    """
    # Apply RBAC filter if provided
    if workspace_filter is not None:
        workspaces = await workspace_filter(workspaces)

    if not workspaces:
        return RetrievalResult(
            answer=None,
            contexts={"chunks": [], "entities": [], "relationships": []},
            raw={"sources": [], "media": [], "workspaces": []},
        )

    # Single workspace — no federation overhead
    if len(workspaces) == 1:
        svc = await get_service(workspaces[0])
        result = await svc.aretrieve(
            query=query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k, **kwargs
        )
        # Tag chunks with workspace
        for chunk in result.contexts.get("chunks", []):
            chunk["_workspace"] = workspaces[0]
        for source in result.raw.get("sources", []):
            source["_workspace"] = workspaces[0]
        result.raw["workspaces"] = workspaces
        return result

    # Parallel queries
    async def _query_workspace(ws: str) -> RetrievalResult | Exception:
        try:
            svc = await get_service(ws)
            return await svc.aretrieve(
                query=query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k, **kwargs
            )
        except Exception as exc:
            logger.warning("Federated query failed for workspace '%s': %s", ws, exc)
            return exc

    raw_results = await asyncio.gather(*[_query_workspace(ws) for ws in workspaces])

    # Filter out failed workspaces
    successful_results: list[RetrievalResult] = []
    successful_workspaces: list[str] = []
    for ws, result in zip(workspaces, raw_results, strict=True):
        if isinstance(result, Exception):
            continue
        successful_results.append(result)
        successful_workspaces.append(ws)

    if not successful_results:
        return RetrievalResult(
            answer=None,
            contexts={"chunks": [], "entities": [], "relationships": []},
            raw={
                "sources": [],
                "media": [],
                "workspaces": [],
                "errors": [str(r) for r in raw_results],
            },
        )

    return merge_results(successful_results, successful_workspaces, chunk_top_k=chunk_top_k)


async def federated_answer(
    query: str,
    workspaces: list[str],
    get_service: Callable[[str], Awaitable[Any]],
    *,
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix",
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    workspace_filter: WorkspaceFilter | None = None,
    **kwargs: Any,
) -> RetrievalResult:
    """Execute federated answer (retrieve + LLM) across multiple workspaces.

    Same as federated_retrieve but calls aanswer() on each service.
    """
    if workspace_filter is not None:
        workspaces = await workspace_filter(workspaces)

    if not workspaces:
        return RetrievalResult(
            answer=None,
            contexts={"chunks": [], "entities": [], "relationships": []},
            raw={"sources": [], "media": [], "workspaces": []},
        )

    if len(workspaces) == 1:
        svc = await get_service(workspaces[0])
        result = await svc.aanswer(
            query=query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k, **kwargs
        )
        for chunk in result.contexts.get("chunks", []):
            chunk["_workspace"] = workspaces[0]
        for source in result.raw.get("sources", []):
            source["_workspace"] = workspaces[0]
        result.raw["workspaces"] = workspaces
        return result

    async def _query_workspace(ws: str) -> RetrievalResult | Exception:
        try:
            svc = await get_service(ws)
            return await svc.aanswer(
                query=query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k, **kwargs
            )
        except Exception as exc:
            logger.warning("Federated answer failed for workspace '%s': %s", ws, exc)
            return exc

    raw_results = await asyncio.gather(*[_query_workspace(ws) for ws in workspaces])

    successful_results: list[RetrievalResult] = []
    successful_workspaces: list[str] = []
    for ws, result in zip(workspaces, raw_results, strict=True):
        if isinstance(result, Exception):
            continue
        successful_results.append(result)
        successful_workspaces.append(ws)

    if not successful_results:
        return RetrievalResult(
            answer=None,
            contexts={"chunks": [], "entities": [], "relationships": []},
            raw={
                "sources": [],
                "media": [],
                "workspaces": [],
                "errors": [str(r) for r in raw_results],
            },
        )

    return merge_results(successful_results, successful_workspaces, chunk_top_k=chunk_top_k)
