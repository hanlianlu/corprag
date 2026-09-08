# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical durable source snapshots for Web conversation answers."""

from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.engine.answer.citations.contracts import ChunkSnippet, SourceReference


class _StoredChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    chunk_idx: int | None = None
    page_number: int | None = None
    content: str
    highlight_phrases: list[str] | None = None
    has_visual: bool = False


class _StoredSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str | None = None
    type: str | None = None
    source_uri: str
    workspace: str
    document_id: str | None = None
    cited_chunk_ids: list[str] | None = None
    chunks: list[_StoredChunk] = Field(default_factory=list)


class _StoredAnswerSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[_StoredSource] = Field(default_factory=list)


def dump_answer_snapshot(sources: list[SourceReference]) -> dict[str, Any]:
    """Serialize source facts without response URLs or duplicated media blocks."""
    snapshot = _StoredAnswerSnapshot(
        sources=[
            _StoredSource(
                id=source.id,
                title=source.title,
                type=source.type,
                source_uri=source.source_uri,
                workspace=source.workspace,
                document_id=source.document_id,
                cited_chunk_ids=source.cited_chunk_ids,
                chunks=[
                    _StoredChunk(
                        chunk_id=chunk.chunk_id,
                        chunk_idx=chunk.chunk_idx,
                        page_number=chunk.page_number,
                        content=chunk.content,
                        highlight_phrases=chunk.highlight_phrases,
                        has_visual=bool(chunk.image_url or chunk.thumbnail_url),
                    )
                    for chunk in source.chunks or []
                ],
            )
            for source in sources
        ]
    )
    return snapshot.model_dump(mode="json")


def load_answer_snapshot(
    value: Any,
    *,
    image_url_prefix: str = "/web/api/images",
) -> list[SourceReference]:
    """Restore internal sources and derive managed image routes for this adapter."""
    snapshot = _StoredAnswerSnapshot.model_validate(value)
    sources: list[SourceReference] = []
    for stored in snapshot.sources:
        chunks: list[ChunkSnippet] = []
        for chunk in stored.chunks:
            image_url = None
            thumbnail_url = None
            if chunk.has_visual:
                base = (
                    f"{image_url_prefix.rstrip('/')}/"
                    f"{quote(stored.workspace, safe='')}/"
                    f"{quote(chunk.chunk_id, safe='')}"
                )
                image_url = f"{base}?size=full"
                thumbnail_url = f"{base}?size=thumb"
            chunks.append(
                ChunkSnippet(
                    chunk_id=chunk.chunk_id,
                    chunk_idx=chunk.chunk_idx,
                    page_number=chunk.page_number,
                    content=chunk.content,
                    image_url=image_url,
                    thumbnail_url=thumbnail_url,
                    highlight_phrases=chunk.highlight_phrases,
                )
            )
        sources.append(
            SourceReference(
                id=stored.id,
                title=stored.title,
                type=stored.type,
                source_uri=stored.source_uri,
                workspace=stored.workspace,
                document_id=stored.document_id,
                download_locator="",
                cited_chunk_ids=stored.cited_chunk_ids,
                chunks=chunks,
            )
        )
    return sources


__all__ = ["dump_answer_snapshot", "load_answer_snapshot"]
