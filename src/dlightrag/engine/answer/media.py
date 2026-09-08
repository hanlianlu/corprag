# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Derive Evidence Images from cited visual source chunks."""

from collections import Counter
from typing import Any
from urllib.parse import urlparse

from dlightrag.engine.answer.citations.contracts import SourceReference
from dlightrag.engine.answer.citations.sources import can_project_workspace_visual
from dlightrag.engine.answer.citations.utils import context_chunk_key
from dlightrag.engine.rag.retrieval import RetrievalContexts


def evidence_images_from_sources(
    sources: list[SourceReference],
    *,
    contexts: RetrievalContexts | None = None,
    visual_workspaces: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return cited visual assets in a transport-neutral shape."""
    sent = _answer_image_sent_by_chunk(contexts)
    chunk_counts = Counter(
        chunk.chunk_id for source in sources for chunk in (source.chunks or []) if chunk.chunk_id
    )
    seen: set[str] = set()
    images: list[dict[str, Any]] = []
    for source in sources:
        if not can_project_workspace_visual(source.workspace, visual_workspaces):
            continue
        source_id = source.id
        base_label = source.title or source_id
        for chunk in source.chunks or []:
            chunk_id = chunk.chunk_id
            image_id = context_chunk_key(chunk_id, workspace=source.workspace)
            if not chunk_id or image_id in seen:
                continue
            url = _public_render_url(chunk.image_url)
            thumbnail_url = _public_render_url(chunk.thumbnail_url) or url
            if not url and not thumbnail_url:
                continue
            url = url or thumbnail_url
            thumbnail_url = thumbnail_url or url
            chunk_idx = chunk.chunk_idx
            source_ref = f"{source_id}-{chunk_idx}" if chunk_idx else source_id
            page_number = getattr(chunk, "page_number", None)
            label = f"{base_label} · Page {page_number}" if page_number else base_label
            images.append(
                {
                    "id": image_id if chunk_counts[chunk_id] > 1 else chunk_id,
                    "chunk_id": chunk_id,
                    "source_ref": source_ref,
                    "url": url,
                    "thumbnail_url": thumbnail_url,
                    "label": label,
                    "answer_image_sent": sent.get(image_id, sent.get(chunk_id, True)),
                }
            )
            seen.add(image_id)
    return images


def _answer_image_sent_by_chunk(contexts: RetrievalContexts | None) -> dict[str, bool]:
    if not contexts:
        return {}
    return {
        context_chunk_key(
            chunk.get("chunk_id") or chunk.get("id"),
            workspace=chunk.get("_workspace"),
        ): chunk.get("_answer_image_sent") is not False
        for chunk in contexts.get("chunks", [])
        if chunk.get("chunk_id") or chunk.get("id")
    }


def _public_render_url(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    if candidate.startswith("/") and not candidate.startswith("//"):
        return candidate
    return candidate if urlparse(candidate).scheme in {"http", "https"} else None


__all__ = ["evidence_images_from_sources"]
