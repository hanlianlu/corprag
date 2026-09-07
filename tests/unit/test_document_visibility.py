# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral Product Document visibility barrier contracts."""

from collections.abc import Mapping, Sequence
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope
from dlightrag.engine.rag.retrieval.filtering import (
    FilteredVectorStorage,
    metadata_filter_scope,
)
from dlightrag.engine.rag.retrieval.metadata_fields import (
    INGEST_FINALIZATION_COMPLETE_FIELD,
)
from dlightrag.engine.rag.retrieval.visibility import (
    bounded_visibility_candidate_limit,
    ingest_finalization_complete,
    retain_visible_chunks,
)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({INGEST_FINALIZATION_COMPLETE_FIELD: True}, True),
        ({INGEST_FINALIZATION_COMPLETE_FIELD: False}, False),
        ({INGEST_FINALIZATION_COMPLETE_FIELD: None}, False),
        ({}, False),
        (None, False),
        ({INGEST_FINALIZATION_COMPLETE_FIELD: 1}, False),
    ],
)
def test_visibility_marker_is_exactly_true(metadata: object, expected: bool) -> None:
    assert ingest_finalization_complete(metadata) is expected


async def test_retain_visible_chunks_drops_missing_and_unattributed_rows() -> None:
    lookup = AsyncMock()
    lookup.visible_subset.return_value = frozenset({"doc-ready"})
    candidates = [
        {"id": "hidden", "full_doc_id": "doc-hidden"},
        {"id": "missing-metadata", "full_doc_id": "doc-bypass"},
        {"id": "unattributed"},
        {"id": "ready", "full_doc_id": "doc-ready"},
    ]

    rows = await retain_visible_chunks(candidates, lookup=lookup, requested_k=3)

    assert rows == [{"id": "ready", "full_doc_id": "doc-ready"}]
    lookup.visible_subset.assert_awaited_once_with(
        ["doc-hidden", "doc-bypass", "doc-ready"],
        scope=None,
    )


def _scope(*, doc_exists: bool = True) -> MetadataScope:
    return MetadataScope(
        filters=MetadataFilter(filename="report.pdf"),
        filename_mode="exact",
        doc_exists=doc_exists,
        candidate_count=1 if doc_exists else 0,
        candidate_count_exact=True,
    )


class _BoundedMetadataLookup:
    """Apply the test's publication-and-exact-filename predicate in one batch."""

    def __init__(self, metadata: Mapping[str, Mapping[str, Any]]) -> None:
        self.metadata = metadata
        self.calls: list[tuple[list[str], MetadataScope | None]] = []

    async def is_visible(self, doc_id: str) -> bool:
        return ingest_finalization_complete(self.metadata.get(doc_id))

    async def visible_subset(
        self,
        doc_ids: Sequence[str],
        *,
        scope: MetadataScope | None = None,
    ) -> frozenset[str]:
        self.calls.append((list(doc_ids), scope))
        matched: set[str] = set()
        for doc_id in doc_ids:
            metadata = self.metadata.get(doc_id)
            if not ingest_finalization_complete(metadata):
                continue
            if scope is not None and metadata.get("filename") != scope.filters.filename:
                continue
            matched.add(doc_id)
        return frozenset(matched)


async def test_retain_visible_chunks_bounds_the_metadata_lookup_input() -> None:
    lookup = AsyncMock()
    lookup.visible_subset.return_value = frozenset()
    candidates = [{"id": f"chunk-{index}", "full_doc_id": f"doc-{index}"} for index in range(9)]

    await retain_visible_chunks(candidates, lookup=lookup, requested_k=2)

    lookup.visible_subset.assert_awaited_once_with(
        [f"doc-{index}" for index in range(8)],
        scope=None,
    )


class _NonPushdownVectorStore:
    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self.candidates = candidates
        self.calls: list[int] = []

    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        del query, query_embedding
        self.calls.append(top_k)
        return self.candidates[:top_k]


async def test_non_pushdown_vector_visibility_is_bounded_and_may_lose_recall() -> None:
    original = _NonPushdownVectorStore(
        [
            {"id": "hidden-1", "full_doc_id": "doc-hidden"},
            {"id": "unknown"},
            {"id": "ready", "full_doc_id": "doc-ready"},
            {"id": "hidden-2", "full_doc_id": "doc-hidden"},
        ]
    )
    lookup = AsyncMock()
    lookup.visible_subset.return_value = frozenset({"doc-ready"})
    wrapper = FilteredVectorStorage(
        original=original,
        embedding_func=AsyncMock(),
        visibility_lookup=lookup,
        filtered_search=None,
    )

    async with metadata_filter_scope(None) as stats:
        rows = await wrapper.query("query", top_k=2, query_embedding=[0.1])

    assert rows == [{"id": "ready", "full_doc_id": "doc-ready"}]
    assert original.calls == [bounded_visibility_candidate_limit(2)]
    lookup.visible_subset.assert_awaited_once_with(
        ["doc-hidden", "doc-ready"],
        scope=None,
    )
    assert stats.visibility_strategy == "postfilter"
    assert stats.visibility_dropped == 3
    assert stats.visibility_shortfall == 1


async def test_non_pushdown_vector_applies_scope_to_one_bounded_metadata_lookup() -> None:
    scope = _scope()
    original = _NonPushdownVectorStore(
        [
            {"id": "wrong", "full_doc_id": "doc-wrong-metadata"},
            {"id": "false", "full_doc_id": "doc-false-marker"},
            {"id": "missing", "full_doc_id": "doc-missing-marker"},
            {"id": "match", "full_doc_id": "doc-match"},
            {"id": "unattributed"},
        ]
    )
    lookup = _BoundedMetadataLookup(
        {
            "doc-wrong-metadata": {
                "filename": "other.pdf",
                INGEST_FINALIZATION_COMPLETE_FIELD: True,
            },
            "doc-false-marker": {
                "filename": "report.pdf",
                INGEST_FINALIZATION_COMPLETE_FIELD: False,
            },
            "doc-missing-marker": {"filename": "report.pdf"},
            "doc-match": {
                "filename": "report.pdf",
                INGEST_FINALIZATION_COMPLETE_FIELD: True,
            },
        }
    )
    wrapper = FilteredVectorStorage(
        original=original,
        embedding_func=AsyncMock(),
        visibility_lookup=lookup,
        filtered_search=None,
    )

    async with metadata_filter_scope(scope):
        rows = await wrapper.query("query", top_k=2, query_embedding=[0.1])

    assert rows == [{"id": "match", "full_doc_id": "doc-match"}]
    assert original.calls == [bounded_visibility_candidate_limit(2)]
    assert lookup.calls == [
        (
            [
                "doc-wrong-metadata",
                "doc-false-marker",
                "doc-missing-marker",
                "doc-match",
            ],
            scope,
        )
    ]


async def test_non_pushdown_vector_empty_scope_skips_backend_and_lookup() -> None:
    original = _NonPushdownVectorStore([{"id": "candidate", "full_doc_id": "doc-candidate"}])
    lookup = _BoundedMetadataLookup({})
    wrapper = FilteredVectorStorage(
        original=original,
        embedding_func=AsyncMock(),
        visibility_lookup=lookup,
        filtered_search=None,
    )

    async with metadata_filter_scope(_scope(doc_exists=False)):
        rows = await wrapper.query("query", top_k=2, query_embedding=[0.1])

    assert rows == []
    assert original.calls == []
    assert lookup.calls == []


def test_visibility_overfetch_has_a_fixed_upper_bound() -> None:
    assert bounded_visibility_candidate_limit(10) == 40
    assert bounded_visibility_candidate_limit(10_000) == 20_000
    assert bounded_visibility_candidate_limit(100_000) == 20_000
