# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Data models for multi-path retrieval."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from dlightrag.engine.rag.retrieval.metadata_fields import canonical_metadata_key

ContextRow = dict[str, Any]


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    """Operation-neutral retrieval controls shared by Answer and Retrieval.

    Public requests remain flat. This value object only keeps Engine call seams
    from growing parallel scalar arguments.
    """

    top_k: int | None = None
    chunk_top_k: int | None = None
    federated_rerank: bool = False


class MetadataFilter(BaseModel):
    """Structured filter for document metadata queries."""

    # An unknown name would otherwise be dropped, turning a typo into a filter
    # that matches every document rather than an error.
    model_config = ConfigDict(extra="forbid")

    filename: str | None = None
    file_extension: str | None = None
    title: str | None = None
    author: str | None = None
    creation_date_from: datetime | None = None
    creation_date_to: datetime | None = None
    custom: dict[str, Any] | None = None

    @field_validator(
        "filename",
        "file_extension",
        "title",
        "author",
        mode="before",
    )
    @classmethod
    def _strip_text_filter(cls, value: Any) -> Any:
        """Normalize user/LLM filter strings without adding fuzzy semantics."""
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("file_extension")
    @classmethod
    def _normalize_file_extension(cls, value: str | None) -> str | None:
        # Only the leading dot: case and padding are folded by the comparison.
        return value.lstrip(".") if value is not None else None

    @field_validator("creation_date_from", "creation_date_to")
    @classmethod
    def _as_utc(cls, value: datetime | None) -> datetime | None:
        """Store one instant regardless of how the caller wrote it.

        Aware timestamps are converted to UTC; a bare timestamp is read as UTC
        rather than as the server's local time. The normalized value is naive
        UTC so a storage session cannot reinterpret it.
        """
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    @field_validator("custom", mode="before")
    @classmethod
    def _canonicalize_custom_keys(cls, value: Any) -> Any:
        """Fold custom keys once at the filter boundary.

        Ingest stores custom metadata under keys folded with the same
        ``canonical_metadata_key`` contract, so a filter and the rows it can
        match never drift apart. Colliding folds resolve to the last key in
        caller order, mirroring how the ingest normalization collapses them.
        """
        if not isinstance(value, dict):
            return value
        return {canonical_metadata_key(str(key)): item for key, item in value.items()}

    def is_empty(self) -> bool:
        """Return True if no filter criteria are set."""
        # An empty `custom` dict carries no criterion, and treating it as one
        # would drop every condition and return the whole workspace.
        return not any(self.model_dump().values())


@dataclass(frozen=True, slots=True)
class MetadataScope:
    """Normalized metadata predicate facts plus a bounded candidate probe.

    The scope carries the filter facts and the selected filename mode, not any
    materialized document-id set: matching documents may number in the millions,
    and shipping that set through Python on every vector, BM25, and graph
    request would not scale. Storage adapters translate these facts into their
    own predicates; ``candidate_count`` bounds what the exact vector scan has to
    brute-force and is derived from a capped database probe, so when
    ``candidate_count_exact`` is False it is a lower bound (the probe stopped
    at its cap), never an exact total.
    """

    filters: MetadataFilter
    filename_mode: str
    doc_exists: bool
    candidate_count: int
    candidate_count_exact: bool

    def __bool__(self) -> bool:
        """True when the metadata predicate matched at least one document.

        A matching document with zero chunks is still an active scope: the
        filter must stay applied rather than falling back to the whole corpus.
        """
        return self.doc_exists

    def render_candidate_count(self) -> str:
        """Human/trace rendering that never misstates a bounded probe as exact."""
        if self.candidate_count_exact:
            return str(self.candidate_count)
        return f"{self.candidate_count}+"
