# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for deletion context collection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from corprag.ingestion.cleanup import collect_deletion_context

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


def _write_kv_doc_status(working_dir: Path, data: dict) -> None:
    """Write a kv_store_doc_status.json file."""
    kv_path = working_dir / "kv_store_doc_status.json"
    kv_path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# TestCollectDeletionContext
# ---------------------------------------------------------------------------


class TestCollectDeletionContext:
    """Test multi-strategy doc_id lookup for deletion."""

    async def test_hash_index_finds_by_name(self, tmp_path: Path) -> None:
        index = _make_hash_index(find_by_name_result=("doc-001", "sha256:abc", "/path/file.pdf"))
        ctx = await collect_deletion_context(
            identifier="file.pdf",
            rag_working_dir=tmp_path,
            hash_index=index,
        )
        assert "doc-001" in ctx.doc_ids
        assert "sha256:abc" in ctx.content_hashes
        assert "/path/file.pdf" in ctx.file_paths
        assert "hash_index" in ctx.sources_used

    async def test_hash_index_finds_by_path(self, tmp_path: Path) -> None:
        index = _make_hash_index(
            find_by_name_result=(None, None, None),
            find_by_path_result=("doc-002", "sha256:def", "/full/path/doc.pdf"),
        )
        ctx = await collect_deletion_context(
            identifier="/full/path/doc.pdf",
            rag_working_dir=tmp_path,
            hash_index=index,
        )
        assert "doc-002" in ctx.doc_ids
        assert "hash_index" in ctx.sources_used

    async def test_kv_doc_status_json_fallback(self, tmp_path: Path) -> None:
        index = _make_hash_index()  # Returns nothing
        _write_kv_doc_status(
            tmp_path,
            {
                "doc-003": {"file_path": "/storage/sources/local/report.pdf"},
            },
        )
        ctx = await collect_deletion_context(
            identifier="report.pdf",
            rag_working_dir=tmp_path,
            hash_index=index,
        )
        assert "doc-003" in ctx.doc_ids
        assert "doc_status" in ctx.sources_used

    async def test_both_strategies_merge(self, tmp_path: Path) -> None:
        index = _make_hash_index(find_by_name_result=("doc-001", "sha256:abc", "/path/file.pdf"))
        _write_kv_doc_status(
            tmp_path,
            {
                "doc-002": {"file_path": "/storage/sources/local/file.pdf"},
            },
        )
        ctx = await collect_deletion_context(
            identifier="file.pdf",
            rag_working_dir=tmp_path,
            hash_index=index,
        )
        assert ctx.doc_ids == {"doc-001", "doc-002"}

    async def test_no_matches(self, tmp_path: Path) -> None:
        index = _make_hash_index()
        ctx = await collect_deletion_context(
            identifier="nonexistent.pdf",
            rag_working_dir=tmp_path,
            hash_index=index,
        )
        assert ctx.doc_ids == set()
        assert ctx.file_paths == set()

    async def test_stem_match_in_kv_store(self, tmp_path: Path) -> None:
        index = _make_hash_index()
        _write_kv_doc_status(
            tmp_path,
            {
                "doc-004": {"file_path": "/storage/report.pdf"},
            },
        )
        # Query with different extension but same stem
        ctx = await collect_deletion_context(
            identifier="report.xlsx",
            rag_working_dir=tmp_path,
            hash_index=index,
        )
        assert "doc-004" in ctx.doc_ids

    async def test_corrupt_kv_store(self, tmp_path: Path) -> None:
        index = _make_hash_index(find_by_name_result=("doc-001", "sha256:abc", "/path/file.pdf"))
        kv_path = tmp_path / "kv_store_doc_status.json"
        kv_path.write_text("not valid json {{{")
        ctx = await collect_deletion_context(
            identifier="file.pdf",
            rag_working_dir=tmp_path,
            hash_index=index,
        )
        # Hash index result should still be present
        assert "doc-001" in ctx.doc_ids

    async def test_no_kv_store_file(self, tmp_path: Path) -> None:
        index = _make_hash_index(find_by_name_result=("doc-001", None, None))
        ctx = await collect_deletion_context(
            identifier="file.pdf",
            rag_working_dir=tmp_path,
            hash_index=index,
        )
        assert "doc-001" in ctx.doc_ids

    async def test_no_hash_index(self, tmp_path: Path) -> None:
        _write_kv_doc_status(
            tmp_path,
            {
                "doc-005": {"file_path": "/storage/data.pdf"},
            },
        )
        ctx = await collect_deletion_context(
            identifier="data.pdf",
            rag_working_dir=tmp_path,
            hash_index=None,
        )
        assert "doc-005" in ctx.doc_ids
        assert "doc_status" in ctx.sources_used
