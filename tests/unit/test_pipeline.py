# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for IngestionPipeline core logic."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.config import DlightragConfig
from dlightrag.ingestion.pipeline import (
    IngestionCancelledError,
    IngestionPipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(
    test_config: DlightragConfig,
    *,
    cancel_checker=None,
    mineru_backend=None,
) -> IngestionPipeline:
    """Build an IngestionPipeline with a fully-mocked RAGAnything."""
    rag = MagicMock()
    rag.parse_document = AsyncMock(
        return_value=(
            [{"type": "text", "text": "hello world"}],
            "doc-001",
        )
    )
    rag.insert_content_list = AsyncMock()
    rag._ensure_lightrag_initialized = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.lightrag = MagicMock()

    pipeline = IngestionPipeline(
        rag,
        config=test_config,
        max_concurrent=2,
        cancel_checker=cancel_checker,
        mineru_backend=mineru_backend,
    )
    # Stub out converter to never trigger Excel conversion
    pipeline.converter = MagicMock()
    pipeline.converter.should_convert.return_value = False
    return pipeline


# ---------------------------------------------------------------------------
# TestIngestionPipelineHelpers
# ---------------------------------------------------------------------------


class TestIngestionPipelineHelpers:
    """Pure function / synchronous helper tests."""

    def test_extract_relative_source_path_standard(self, test_config: DlightragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        result = pipeline._extract_relative_source_path(
            "/abs/path/dlightrag_storage/sources/local/file.pdf"
        )
        assert result == "local/file.pdf"

    def test_extract_relative_source_path_no_marker(self, test_config: DlightragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        result = pipeline._extract_relative_source_path("/some/random/path/file.pdf")
        assert result is None

    def test_extract_relative_source_path_azure_blobs(self, test_config: DlightragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        result = pipeline._extract_relative_source_path(
            "/abs/dlightrag_storage/sources/azure_blobs/c1/sub/file.pdf"
        )
        assert result == "azure_blobs/c1/sub/file.pdf"

    def test_resolve_source_file_absolute_exists(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config)
        f = tmp_path / "test_file.txt"
        f.write_text("data")
        assert pipeline._resolve_source_file(str(f)) == f

    def test_resolve_source_file_absolute_not_found(self, test_config: DlightragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        assert pipeline._resolve_source_file("/nonexistent/file.pdf") is None

    def test_resolve_source_file_basename_glob_fallback(self, test_config: DlightragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        # Create file nested inside sources dir
        nested = test_config.sources_dir / "local" / "report.pdf"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text("pdf content")
        result = pipeline._resolve_source_file("report.pdf")
        assert result is not None
        assert result.name == "report.pdf"

    def test_get_storage_dir_creates_path(self, test_config: DlightragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        d = pipeline._get_storage_dir(test_config.sources_dir, "custom_type", "sub1", "sub2")
        assert d.exists()
        assert d.is_dir()
        assert "custom_type" in str(d)

    async def test_acopy_to_sources_local_new_file(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config)
        src = tmp_path / "new_doc.pdf"
        src.write_text("pdf bytes")
        dest = await pipeline._acopy_to_sources_local(src)
        assert dest.exists()
        assert dest.name == "new_doc.pdf"
        assert dest.read_text() == "pdf bytes"

    async def test_acopy_to_sources_local_conflict_resolution(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        from datetime import date

        pipeline = _make_pipeline(test_config)
        # Pre-create file in sources/local/
        existing = test_config.sources_dir / "local" / "report.pdf"
        existing.write_text("existing")

        # New file with same name but different content
        src = tmp_path / "report.pdf"
        src.write_text("new version")
        dest = await pipeline._acopy_to_sources_local(src)
        date_str = date.today().strftime("%Y_%m_%d")
        assert dest.name == f"report_{date_str}.pdf"
        assert dest.read_text() == "new version"

    async def test_acopy_to_sources_local_already_in_place(
        self, test_config: DlightragConfig
    ) -> None:
        pipeline = _make_pipeline(test_config)
        # File already in sources/local
        existing = test_config.sources_dir / "local" / "report.pdf"
        existing.write_text("content")
        dest = await pipeline._acopy_to_sources_local(existing)
        assert dest == existing


# ---------------------------------------------------------------------------
# TestIngestSingleFileWithPolicy
# ---------------------------------------------------------------------------


class TestIngestSingleFileWithPolicy:
    """Core parse -> filter -> insert pipeline."""

    async def test_successful_ingest(self, test_config: DlightragConfig, tmp_path: Path) -> None:
        pipeline = _make_pipeline(test_config)
        file_path = tmp_path / "doc.pdf"
        file_path.write_text("content")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = await pipeline._ingest_single_file_with_policy(
            file_path, artifacts, content_hash="sha256:abc123"
        )

        assert result.status == "success"
        assert result.processed == 1
        assert result.doc_id == "doc-001"
        pipeline.rag.insert_content_list.assert_awaited_once()
        # Hash should be registered
        assert pipeline._hash_index.lookup("sha256:abc123") is not None

    async def test_all_content_filtered_by_policy(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config)
        # Return only discarded blocks
        pipeline.rag.parse_document.return_value = (
            [{"type": "discarded", "text": "noise"}],
            "doc-002",
        )
        file_path = tmp_path / "noisy.pdf"
        file_path.write_text("content")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = await pipeline._ingest_single_file_with_policy(file_path, artifacts)

        assert result.status == "success"
        assert result.processed == 1
        assert result.stats is not None
        assert result.stats.indexed == 0
        pipeline.rag.insert_content_list.assert_not_awaited()

    async def test_parse_document_error(self, test_config: DlightragConfig, tmp_path: Path) -> None:
        pipeline = _make_pipeline(test_config)
        pipeline.rag.parse_document.side_effect = RuntimeError("parse failed")
        file_path = tmp_path / "bad.pdf"
        file_path.write_text("content")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = await pipeline._ingest_single_file_with_policy(file_path, artifacts)

        assert result.status == "error"
        assert "parse failed" in result.error

    async def test_no_hash_registration_when_hash_is_none(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config)
        file_path = tmp_path / "doc.pdf"
        file_path.write_text("content")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = await pipeline._ingest_single_file_with_policy(
            file_path, artifacts, content_hash=None
        )

        assert result.status == "success"
        # Hash index should remain empty since no hash was provided
        assert await pipeline._hash_index.list_all() == []

    async def test_mineru_backend_passed_to_parse_kwargs(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config, mineru_backend="hybrid-auto-engine")
        file_path = tmp_path / "doc.pdf"
        file_path.write_text("content")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        await pipeline._ingest_single_file_with_policy(file_path, artifacts)

        call_kwargs = pipeline.rag.parse_document.call_args
        assert call_kwargs.kwargs.get("backend") == "hybrid-auto-engine"


# ---------------------------------------------------------------------------
# TestAingestFromLocal
# ---------------------------------------------------------------------------


class TestAingestFromLocal:
    """Local ingestion workflow: dedup, copy, process."""

    async def test_single_file_new(self, test_config: DlightragConfig, tmp_path: Path) -> None:
        pipeline = _make_pipeline(test_config)
        src = tmp_path / "new.pdf"
        src.write_text("content")

        result = await pipeline.aingest_from_local(src)

        assert result.source_type == "local"
        assert result.total_files == 1
        assert result.processed == 1
        assert result.skipped == 0

    async def test_single_file_duplicate_skipped(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config)
        src = tmp_path / "dup.pdf"
        src.write_text("content")

        # Ingest once
        await pipeline.aingest_from_local(src)

        # Second ingest should skip
        result = await pipeline.aingest_from_local(src)

        assert result.processed == 0
        assert result.skipped == 1

    async def test_path_not_found(self, test_config: DlightragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        with pytest.raises(FileNotFoundError, match="Path not found"):
            await pipeline.aingest_from_local(Path("/nonexistent/file.pdf"))

    async def test_directory_multiple_files(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config)
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.pdf").write_text("aaa")
        (d / "b.pdf").write_text("bbb")
        (d / "c.pdf").write_text("ccc")

        result = await pipeline.aingest_from_local(d)

        assert result.total_files == 3
        assert result.processed == 3
        assert result.skipped == 0

    async def test_directory_empty(self, test_config: DlightragConfig, tmp_path: Path) -> None:
        pipeline = _make_pipeline(test_config)
        d = tmp_path / "empty_dir"
        d.mkdir()

        result = await pipeline.aingest_from_local(d)

        assert result.processed == 0
        assert result.total_files == 0


# ---------------------------------------------------------------------------
# TestCheckCancelled
# ---------------------------------------------------------------------------


class TestCheckCancelled:
    """Test cancellation detection."""

    async def test_no_checker_no_error(self, test_config: DlightragConfig) -> None:
        pipeline = _make_pipeline(test_config, cancel_checker=None)
        # Should not raise
        await pipeline._check_cancelled()

    async def test_checker_returns_false(self, test_config: DlightragConfig) -> None:
        checker = AsyncMock(return_value=False)
        pipeline = _make_pipeline(test_config, cancel_checker=checker)
        await pipeline._check_cancelled()

    async def test_checker_returns_true(self, test_config: DlightragConfig) -> None:
        checker = AsyncMock(return_value=True)
        pipeline = _make_pipeline(test_config, cancel_checker=checker)
        with pytest.raises(IngestionCancelledError, match="cancelled by caller"):
            await pipeline._check_cancelled()

    async def test_asyncio_cancellation(self, test_config: DlightragConfig) -> None:
        pipeline = _make_pipeline(test_config)

        async def run():
            task = asyncio.current_task()
            task.cancel()
            await asyncio.sleep(0)  # Yield to allow cancellation
            await pipeline._check_cancelled()

        with pytest.raises(asyncio.CancelledError):
            await run()


# ---------------------------------------------------------------------------
# TestTempDirAndSourceUri
# ---------------------------------------------------------------------------


class TestTempDirAndSourceUri:
    """Tests for temp dir creation and source_uri flow."""

    @pytest.mark.asyncio
    async def test_create_temp_dir_under_working_dir(self, test_config):
        """Temp dirs are created under working_dir/.tmp/."""
        pipeline = _make_pipeline(test_config)
        tmpdir = pipeline._create_temp_dir()
        assert tmpdir.exists()
        assert ".tmp" in tmpdir.parts
        # Cleanup
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_source_uri_passed_to_insert_content_list(self, test_config):
        """source_uri (not parse_path) is passed to insert_content_list."""
        pipeline = _make_pipeline(test_config)
        test_file = test_config.working_dir_path / "test.txt"
        test_file.write_text("hello")

        await pipeline._ingest_single_file_with_policy(
            file_path=test_file,
            artifacts_dir=test_config.artifacts_dir,
            source_uri="/original/path/test.txt",
        )

        # insert_content_list should receive source_uri, not parse_path
        call_args = pipeline.rag.insert_content_list.call_args
        assert call_args is not None
        # Check both positional and keyword args
        file_path_arg = call_args.kwargs.get("file_path") or call_args[1].get("file_path")
        assert file_path_arg == "/original/path/test.txt"

    @pytest.mark.asyncio
    async def test_source_uri_stored_in_hash_index(self, test_config):
        """source_uri is stored in hash_index, not parse_path."""
        pipeline = _make_pipeline(test_config)
        test_file = test_config.working_dir_path / "test.txt"
        test_file.write_text("hello")

        await pipeline._ingest_single_file_with_policy(
            file_path=test_file,
            artifacts_dir=test_config.artifacts_dir,
            content_hash="abc123",
            source_uri="/original/path/test.txt",
        )

        entries = await pipeline._hash_index.list_all()
        assert len(entries) == 1
        assert entries[0]["file_path"] == "/original/path/test.txt"

    @pytest.mark.asyncio
    async def test_prepare_for_parsing_non_excel(self, test_config):
        """Non-Excel files pass through unchanged."""
        pipeline = _make_pipeline(test_config)
        test_file = test_config.working_dir_path / "test.pdf"
        test_file.write_text("pdf content")
        tmpdir = pipeline._create_temp_dir()
        result = await pipeline._prepare_for_parsing(test_file, tmpdir)
        assert result == test_file
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
