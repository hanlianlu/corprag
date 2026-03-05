# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for deletion context collection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from dlightrag.core.ingestion.cleanup import collect_deletion_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hash_index(
    find_by_name_result=(None, None, None),
    find_by_path_result=(None, None, None),
):
    """Create a mock hash index with configurable lookup results."""
    index = MagicMock()
    index.find_by_name.return_value = find_by_name_result
    index.find_by_path.return_value = find_by_path_result
    return index


def _make_lightrag(docs: dict[str, str] | None = None):
    """Create a mock lightrag with doc_status API.

    Args:
        docs: mapping of doc_id -> file_path for processed docs
    """
    if docs is None:
        return None

    doc_status = MagicMock()

    # get_doc_by_file_path: exact match on file_path
    async def _get_by_path(fp: str):
        for d_id, stored_fp in docs.items():
            if stored_fp == fp:
                return {"id": d_id, "file_path": stored_fp}
        return None

    doc_status.get_doc_by_file_path = AsyncMock(side_effect=_get_by_path)

    # get_docs_by_status: return all docs as SimpleNamespace objects
    async def _get_by_status(_status):
        return {d_id: SimpleNamespace(file_path=fp) for d_id, fp in docs.items()}

    doc_status.get_docs_by_status = AsyncMock(side_effect=_get_by_status)

    lightrag = MagicMock()
    lightrag.doc_status = doc_status
    return lightrag


# ---------------------------------------------------------------------------
# TestCollectDeletionContext
# ---------------------------------------------------------------------------


class TestCollectDeletionContext:
    """Test multi-strategy doc_id lookup for deletion."""

    async def test_hash_index_finds_by_name(self) -> None:
        index = _make_hash_index(find_by_name_result=("doc-001", "sha256:abc", "/path/file.pdf"))
        ctx = await collect_deletion_context(
            identifier="file.pdf",
            hash_index=index,
        )
        assert "doc-001" in ctx.doc_ids
        assert "sha256:abc" in ctx.content_hashes
        assert "/path/file.pdf" in ctx.file_paths
        assert "hash_index" in ctx.sources_used

    async def test_hash_index_finds_by_path(self) -> None:
        index = _make_hash_index(
            find_by_name_result=(None, None, None),
            find_by_path_result=("doc-002", "sha256:def", "/full/path/doc.pdf"),
        )
        ctx = await collect_deletion_context(
            identifier="/full/path/doc.pdf",
            hash_index=index,
        )
        assert "doc-002" in ctx.doc_ids
        assert "hash_index" in ctx.sources_used

    async def test_lightrag_doc_status_fallback(self) -> None:
        index = _make_hash_index()  # Returns nothing
        lightrag = _make_lightrag({"doc-003": "/storage/sources/local/report.pdf"})
        ctx = await collect_deletion_context(
            identifier="report.pdf",
            hash_index=index,
            lightrag=lightrag,
        )
        assert "doc-003" in ctx.doc_ids
        assert "doc_status" in ctx.sources_used

    async def test_both_strategies_merge(self) -> None:
        index = _make_hash_index(find_by_name_result=("doc-001", "sha256:abc", "/path/file.pdf"))
        lightrag = _make_lightrag({"doc-002": "/storage/sources/local/file.pdf"})
        ctx = await collect_deletion_context(
            identifier="file.pdf",
            hash_index=index,
            lightrag=lightrag,
        )
        assert ctx.doc_ids == {"doc-001", "doc-002"}

    async def test_no_matches(self) -> None:
        index = _make_hash_index()
        ctx = await collect_deletion_context(
            identifier="nonexistent.pdf",
            hash_index=index,
        )
        assert ctx.doc_ids == set()
        assert ctx.file_paths == set()

    async def test_stem_match_via_doc_status(self) -> None:
        index = _make_hash_index()
        lightrag = _make_lightrag({"doc-004": "/storage/report.pdf"})
        # Query with different extension but same stem
        ctx = await collect_deletion_context(
            identifier="report.xlsx",
            hash_index=index,
            lightrag=lightrag,
        )
        assert "doc-004" in ctx.doc_ids

    async def test_doc_status_exception_handled(self) -> None:
        index = _make_hash_index(find_by_name_result=("doc-001", "sha256:abc", "/path/file.pdf"))
        lightrag = MagicMock()
        lightrag.doc_status = MagicMock()
        lightrag.doc_status.get_doc_by_file_path = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )
        ctx = await collect_deletion_context(
            identifier="file.pdf",
            hash_index=index,
            lightrag=lightrag,
        )
        # Hash index result should still be present despite doc_status failure
        assert "doc-001" in ctx.doc_ids

    async def test_no_lightrag(self) -> None:
        index = _make_hash_index(find_by_name_result=("doc-001", None, None))
        ctx = await collect_deletion_context(
            identifier="file.pdf",
            hash_index=index,
            lightrag=None,
        )
        assert "doc-001" in ctx.doc_ids

    async def test_no_hash_index_uses_doc_status(self) -> None:
        lightrag = _make_lightrag({"doc-005": "/storage/data.pdf"})
        ctx = await collect_deletion_context(
            identifier="data.pdf",
            hash_index=None,
            lightrag=lightrag,
        )
        assert "doc-005" in ctx.doc_ids
        assert "doc_status" in ctx.sources_used
