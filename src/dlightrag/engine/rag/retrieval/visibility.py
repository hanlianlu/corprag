# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral Product Document visibility contracts and bounded filtering."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeGuard

from dlightrag.engine.rag.retrieval.metadata_fields import (
    INGEST_FINALIZATION_COMPLETE_FIELD,
)
from dlightrag.engine.rag.retrieval.models import MetadataScope

MAX_VISIBILITY_CANDIDATES = 20_000


class VisibleDocumentLookup(Protocol):
    """Resolve strict document eligibility for caller-supplied identities.

    Implementations must treat a missing row, NULL, and false as unpublished.
    When a metadata scope is supplied, its predicate is an additional required
    match. This port deliberately has no operation that lists the corpus.
    """

    async def is_visible(self, doc_id: str) -> bool: ...

    async def visible_subset(
        self,
        doc_ids: Sequence[str],
        *,
        scope: MetadataScope | None = None,
    ) -> frozenset[str]:
        """Return publication-and-scope matches from one caller-bounded id set."""
        ...


def ingest_finalization_complete(metadata: object) -> TypeGuard[Mapping[str, Any]]:
    """Return true only for the exact DlightRAG publication marker value."""
    return (
        isinstance(metadata, Mapping) and metadata.get(INGEST_FINALIZATION_COMPLETE_FIELD) is True
    )


def bounded_visibility_candidate_limit(requested_k: int) -> int:
    """Bound post-filter over-fetch to the product retrieval safety ceiling."""
    limit = max(0, int(requested_k))
    return min(max(limit * 4, limit), MAX_VISIBILITY_CANDIDATES)


async def retain_visible_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    lookup: VisibleDocumentLookup,
    requested_k: int,
    scope: MetadataScope | None = None,
) -> list[dict[str, Any]]:
    """Drop rows outside publication or optional metadata scope from candidates."""
    limit = max(0, int(requested_k))
    if limit == 0 or not chunks or (scope is not None and not scope):
        return []
    bounded_chunks = chunks[: bounded_visibility_candidate_limit(limit)]
    doc_ids = list(
        dict.fromkeys(
            doc_id
            for chunk in bounded_chunks
            if isinstance((doc_id := chunk.get("full_doc_id")), str) and doc_id
        )
    )
    visible = await lookup.visible_subset(doc_ids, scope=scope) if doc_ids else frozenset()
    retained: list[dict[str, Any]] = []
    for chunk in bounded_chunks:
        doc_id = chunk.get("full_doc_id")
        if not isinstance(doc_id, str) or doc_id not in visible:
            continue
        retained.append(dict(chunk))
        if len(retained) >= limit:
            break
    return retained


__all__ = [
    "MAX_VISIBILITY_CANDIDATES",
    "VisibleDocumentLookup",
    "bounded_visibility_candidate_limit",
    "ingest_finalization_complete",
    "retain_visible_chunks",
]
