# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for deletion context collection."""

from unittest.mock import AsyncMock, MagicMock

from dlightrag.engine.rag.corpus.contracts import DocStatusMatch
from dlightrag.engine.rag.corpus.ingestion.cleanup import (
    DeletionContext,
    cascade_delete,
    collect_deletion_context,
    remove_deleted_files,
)


def _make_status_lookup(docs: dict[str, str]) -> MagicMock:
    """Create an indexed deletion lookup fake."""

    async def _resolve(*, file_paths, doc_ids):
        paths = set(file_paths)
        ids = set(doc_ids)
        return tuple(
            DocStatusMatch(doc_id=doc_id, file_path=file_path)
            for doc_id, file_path in sorted(docs.items())
            if doc_id in ids or file_path in paths
        )

    lookup = MagicMock()
    lookup.resolve_deletion_matches = AsyncMock(side_effect=_resolve)
    return lookup


class TestCollectDeletionContext:
    """Test multi-strategy doc_id lookup for deletion."""

    async def test_doc_status_exact_path(self) -> None:
        lookup = _make_status_lookup({"doc-003": "/storage/docs/report.pdf"})
        ctx = await collect_deletion_context(
            identifier="/storage/docs/report.pdf",
            metadata_index=None,
            doc_status_lookup=lookup,
        )
        assert "doc-003" in ctx.doc_ids
        assert "/storage/docs/report.pdf" in ctx.file_paths
        assert "doc_status" in ctx.sources_used

    async def test_doc_status_basename(self) -> None:
        lookup = _make_status_lookup({"doc-004": "report.pdf"})
        ctx = await collect_deletion_context(
            identifier="report.pdf",
            metadata_index=None,
            doc_status_lookup=lookup,
        )
        assert "doc-004" in ctx.doc_ids
        assert "doc_status" in ctx.sources_used

    async def test_bare_stem_does_not_match_another_extension(self) -> None:
        lookup = _make_status_lookup({"doc-005": "report.pdf"})
        ctx = await collect_deletion_context(
            identifier="report.xlsx",
            metadata_index=None,
            doc_status_lookup=lookup,
        )
        assert ctx.doc_ids == set()

    async def test_path_identifier_never_expands_to_same_basename_documents(self) -> None:
        metadata_index = MagicMock()
        metadata_index.find_by_download_locator = AsyncMock(return_value=[])
        metadata_index.find_by_filename = AsyncMock(return_value=["doc-wrong"])
        lookup = _make_status_lookup(
            {
                "doc-right": "/inputs/a/report.pdf",
                "doc-wrong": "/inputs/b/report.pdf",
            }
        )

        ctx = await collect_deletion_context(
            identifier="/inputs/a/report.pdf",
            metadata_index=metadata_index,
            doc_status_lookup=lookup,
        )

        assert ctx.doc_ids == {"doc-right"}
        metadata_index.find_by_filename.assert_not_awaited()

    async def test_path_identifier_does_not_fallback_to_bare_status_path(self) -> None:
        metadata_index = MagicMock()
        metadata_index.find_by_download_locator = AsyncMock(return_value=[])
        metadata_index.find_by_filename = AsyncMock(return_value=["doc-wrong"])
        lookup = _make_status_lookup({"doc-wrong": "report.pdf"})

        ctx = await collect_deletion_context(
            identifier="/inputs/a/report.pdf",
            metadata_index=metadata_index,
            doc_status_lookup=lookup,
        )

        assert ctx.doc_ids == set()
        metadata_index.find_by_filename.assert_not_awaited()

    async def test_metadata_status_path_expands_exact_duplicate_receipts(self) -> None:
        metadata_index = MagicMock()
        metadata_index.find_by_download_locator = AsyncMock(return_value=["doc-primary"])
        metadata_index.find_by_filename = AsyncMock(return_value=[])
        lookup = _make_status_lookup(
            {
                "doc-primary": "/staged/report__abc.pdf",
                "doc-duplicate": "/staged/report__abc.pdf",
            }
        )

        ctx = await collect_deletion_context(
            identifier="s3://bucket/report.pdf",
            metadata_index=metadata_index,
            doc_status_lookup=lookup,
        )

        assert ctx.doc_ids == {"doc-primary", "doc-duplicate"}
        assert ctx.file_paths == {"/staged/report__abc.pdf"}

    async def test_metadata_index_fallback(self) -> None:
        metadata_index = MagicMock()
        metadata_index.find_by_download_locator = AsyncMock(return_value=[])
        metadata_index.find_by_filename = AsyncMock(return_value=["doc-006"])
        ctx = await collect_deletion_context(
            identifier="report.pdf",
            metadata_index=metadata_index,
            doc_status_lookup=None,
        )
        assert ctx.doc_ids == {"doc-006"}
        assert ctx.sources_used == ["metadata_index"]

    async def test_metadata_index_exact_remote_path_precedes_filename(self) -> None:
        lookup = _make_status_lookup(
            {"doc-remote": "/inputs/default/__remote_ingest__/s3/b1/report__abc.pdf"}
        )
        metadata_index = MagicMock()
        metadata_index.find_by_download_locator = AsyncMock(return_value=["doc-remote"])
        metadata_index.find_by_filename = AsyncMock(return_value=["doc-wrong"])

        ctx = await collect_deletion_context(
            identifier="s3://bucket/team-a/report.pdf",
            metadata_index=metadata_index,
            doc_status_lookup=lookup,
        )

        assert ctx.doc_ids == {"doc-remote"}
        assert ctx.file_paths == {"/inputs/default/__remote_ingest__/s3/b1/report__abc.pdf"}
        metadata_index.find_by_download_locator.assert_awaited_once_with(
            "s3://bucket/team-a/report.pdf"
        )
        metadata_index.find_by_filename.assert_not_awaited()

    async def test_doc_status_exception_falls_back_to_metadata(self) -> None:
        lookup = MagicMock()
        lookup.resolve_deletion_matches = AsyncMock(side_effect=RuntimeError("connection lost"))
        metadata_index = MagicMock()
        metadata_index.find_by_download_locator = AsyncMock(return_value=[])
        metadata_index.find_by_filename = AsyncMock(return_value=["doc-007"])

        ctx = await collect_deletion_context(
            identifier="file.pdf",
            metadata_index=metadata_index,
            doc_status_lookup=lookup,
        )

        assert ctx.doc_ids == {"doc-007"}
        assert ctx.sources_used == ["metadata_index"]

    async def test_no_matches(self) -> None:
        ctx = await collect_deletion_context(
            identifier="nonexistent.pdf",
            metadata_index=None,
            doc_status_lookup=None,
        )
        assert ctx.doc_ids == set()
        assert ctx.file_paths == set()
        assert ctx.sources_used == []


class TestRemoveDeletedFiles:
    """Test physical source cleanup boundaries."""

    def test_remote_uri_does_not_delete_local_same_basename(self, tmp_path) -> None:
        workspace_input = tmp_path / "inputs" / "default"
        workspace_input.mkdir(parents=True)
        local_copy = workspace_input / "report.pdf"
        local_copy.write_bytes(b"local")

        removed = remove_deleted_files(
            {
                "azure://container/team-a/report.pdf",
                "s3://bucket/team-b/report.pdf",
                "https://api.bynder.com/docs/report.pdf",
            },
            str(workspace_input),
        )

        assert removed == 0
        assert local_copy.read_bytes() == b"local"

    def test_full_filename_parser_artifacts_removed_when_source_is_missing(self, tmp_path) -> None:
        workspace_input = tmp_path / "inputs" / "default"
        parsed_root = workspace_input / "__parsed__"
        artifact_dirs = [
            parsed_root / "report.pdf.parsed",
            parsed_root / "report.pdf.docling_raw",
            parsed_root / "report.pdf.mineru_raw",
        ]
        for artifact_dir in artifact_dirs:
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "artifact.txt").write_text("stale")

        removed = remove_deleted_files({"report.pdf"}, str(workspace_input))

        assert removed == len(artifact_dirs)
        assert all(not artifact_dir.exists() for artifact_dir in artifact_dirs)

    def test_nested_remote_parser_path_removes_artifacts_not_remote_source(self, tmp_path) -> None:
        workspace_input = tmp_path / "inputs" / "default"
        batch_root = workspace_input / "__remote_ingest__" / "s3" / "batch-1"
        parsed_root = batch_root / "__parsed__"
        artifact_dir = parsed_root / "report__abc.parsed"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "blocks.jsonl").write_text("{}\n")
        archived_source = parsed_root / "report__abc.pdf"
        archived_source.write_bytes(b"%PDF")
        direct_source = batch_root / "report__abc.pdf"
        direct_source.write_bytes(b"%PDF")

        removed = remove_deleted_files({str(direct_source)}, str(workspace_input))

        assert removed == 3
        assert not direct_source.exists()
        assert not archived_source.exists()
        assert not artifact_dir.exists()


def _make_ctx(doc_ids: set[str] | None = None) -> DeletionContext:
    """Build a minimal DeletionContext for cascade_delete tests."""
    return DeletionContext(identifier="test.pdf", doc_ids=doc_ids or set())


def _make_lightrag_for_delete() -> MagicMock:
    lr = MagicMock()
    lr.adelete_by_doc_id = AsyncMock(return_value={"status": "success"})
    return lr


def _make_metadata_index(page_count: int = 0) -> MagicMock:
    mi = MagicMock()
    mi.get = AsyncMock(return_value={"page_count": page_count})
    mi.upsert = AsyncMock(return_value=None)
    mi.delete = AsyncMock(return_value=None)
    return mi


class TestCascadeDelete:
    """Test cascade_delete with per-layer fault isolation."""

    async def test_cascade_delete_lightrag_and_metadata_called(self) -> None:
        ctx = _make_ctx(doc_ids={"doc-1"})
        lr = _make_lightrag_for_delete()
        mi = _make_metadata_index()

        stats = await cascade_delete(
            ctx=ctx,
            lightrag=lr,
            metadata_index=mi,
        )

        mi.upsert.assert_awaited_once_with("doc-1", {"_dlightrag_finalization_complete": False})
        lr.adelete_by_doc_id.assert_awaited_once_with("doc-1", delete_llm_cache=True)
        mi.delete.assert_awaited_once_with("doc-1")
        assert stats["docs_deleted"] == 1
        assert stats["errors"] == []

    async def test_cascade_delete_lightrag_failure_preserves_hidden_metadata(self) -> None:
        ctx = _make_ctx(doc_ids={"doc-2"})
        lr = _make_lightrag_for_delete()
        lr.adelete_by_doc_id = AsyncMock(side_effect=RuntimeError("connection lost"))
        mi = _make_metadata_index()

        stats = await cascade_delete(
            ctx=ctx,
            lightrag=lr,
            metadata_index=mi,
        )

        assert any("ambiguous" in e for e in stats["errors"])
        assert stats["docs_deleted"] == 0
        mi.delete.assert_not_awaited()

    async def test_cascade_delete_empty_context(self) -> None:
        ctx = _make_ctx(doc_ids=set())
        lr = _make_lightrag_for_delete()

        stats = await cascade_delete(ctx=ctx, lightrag=lr)

        lr.adelete_by_doc_id.assert_not_called()
        assert stats["docs_deleted"] == 0
        assert stats["errors"] == []

    async def test_cascade_delete_skips_none_stores(self) -> None:
        ctx = _make_ctx(doc_ids={"doc-3"})
        lr = _make_lightrag_for_delete()

        stats = await cascade_delete(ctx=ctx, lightrag=lr)

        lr.adelete_by_doc_id.assert_awaited_once_with("doc-3", delete_llm_cache=True)
        assert stats["docs_deleted"] == 1
        assert stats["errors"] == []
