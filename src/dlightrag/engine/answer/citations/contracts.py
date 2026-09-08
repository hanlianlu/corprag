"""Caller-facing citation patterns and data models."""

import re
from typing import Protocol

from pydantic import BaseModel, Field

_CHUNK_INDEX = r"[0-9]{1,9}"
_NUMERIC_REF = r"[0-9]+"
_ATTACHMENT_REF = r"att-[0-9]+"
_GENERIC_CHUNK_REF = rf"(?!att-{_CHUNK_INDEX}\])\w+"
_CHUNK_REFERENCE_ID = rf"(?:{_ATTACHMENT_REF}|{_NUMERIC_REF}|{_GENERIC_CHUNK_REF})"
_CITATION_TRAILING_BOUNDARY = r"(?![A-Za-z0-9_-])"

CITATION_PATTERN = re.compile(
    rf"\[({_CHUNK_REFERENCE_ID})-({_CHUNK_INDEX})\]{_CITATION_TRAILING_BOUNDARY}"
)
DOC_CITATION_PATTERN = re.compile(
    rf"\[((?:{_NUMERIC_REF}|{_ATTACHMENT_REF}))\]{_CITATION_TRAILING_BOUNDARY}"
)


class ChunkSnippet(BaseModel):
    """Individual chunk reference with optional semantic highlights."""

    chunk_id: str
    chunk_idx: int | None = None
    page_number: int | None = None
    content: str
    image_url: str | None = None
    thumbnail_url: str | None = None
    highlight_phrases: list[str] | None = None

    model_config = {"extra": "forbid"}


class SourceReference(BaseModel):
    """Internal document source with durable download-routing metadata."""

    id: str
    title: str | None = None
    type: str | None = None
    source_uri: str
    workspace: str = Field(exclude=True, repr=False)
    document_id: str | None = Field(default=None, exclude=True, repr=False)
    download_locator: str = Field(exclude=True, repr=False)
    cited_chunk_ids: list[str] | None = None
    chunks: list[ChunkSnippet] | None = None

    model_config = {"extra": "forbid"}


class SourceReferencePayload(BaseModel):
    """Public source payload with an adapter-projected download URL."""

    id: str
    title: str | None = None
    type: str | None = None
    source_uri: str
    download_url: str | None = None
    cited_chunk_ids: list[str] | None = None
    chunks: list[ChunkSnippet] | None = None

    model_config = {"extra": "forbid"}


class HighlightSource(Protocol):
    """Structural source contract used by semantic highlight enrichment."""

    id: str
    chunks: list[ChunkSnippet] | None
