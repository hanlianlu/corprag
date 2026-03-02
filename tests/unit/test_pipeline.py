# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for IngestionPipeline core logic."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from corprag.config import CorpragConfig
from corprag.ingestion.pipeline import (
    IngestionCancelledError,
    IngestionPipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(
    test_config: CorpragConfig,
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

    def test_extract_relative_source_path_standard(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        result = pipeline._extract_relative_source_path(
            "/abs/path/corprag_storage/sources/local/file.pdf"
        )
        assert result == "local/file.pdf"

    def test_extract_relative_source_path_no_marker(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        result = pipeline._extract_relative_source_path("/some/random/path/file.pdf")
        assert result is None

    def test_extract_relative_source_path_azure_blobs(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        result = pipeline._extract_relative_source_path(
            "/abs/corprag_storage/sources/azure_blobs/c1/sub/file.pdf"
        )
        assert result == "azure_blobs/c1/sub/file.pdf"

    def test_resolve_source_file_absolute_exists(
        self, test_config: CorpragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config)
        f = tmp_path / "test_file.txt"
        f.write_text("data")
        assert pipeline._resolve_source_file(str(f)) == f

    def test_resolve_source_file_absolute_not_found(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        assert pipeline._resolve_source_file("/nonexistent/file.pdf") is None

    def test_resolve_source_file_basename_glob_fallback(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        # Create file nested inside sources dir
        nested = test_config.sources_dir / "local" / "report.pdf"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text("pdf content")
        result = pipeline._resolve_source_file("report.pdf")
        assert result is not None
        assert result.name == "report.pdf"

    def test_get_storage_dir_creates_path(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        d = pipeline._get_storage_dir(test_config.sources_dir, "custom_type", "sub1", "sub2")
        assert d.exists()
        assert d.is_dir()
        assert "custom_type" in str(d)

    def test_copy_to_sources_local_new_file(
        self, test_config: CorpragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config)
        src = tmp_path / "new_doc.pdf"
        src.write_text("pdf bytes")
        dest = pipeline._copy_to_sources_local(src)
        assert dest.exists()
        assert dest.name == "new_doc.pdf"
        assert dest.read_text() == "pdf bytes"

    def test_copy_to_sources_local_conflict_resolution(
        self, test_config: CorpragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config)
        # Pre-create file in sources/local/
        existing = test_config.sources_dir / "local" / "report.pdf"
        existing.write_text("existing")

        # New file with same name but different content
        src = tmp_path / "report.pdf"
        src.write_text("new version")
        dest = pipeline._copy_to_sources_local(src)
        assert dest.name == "report_1.pdf"
        assert dest.read_text() == "new version"

    def test_copy_to_sources_local_already_in_place(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        # File already in sources/local
        existing = test_config.sources_dir / "local" / "report.pdf"
        existing.write_text("content")
        dest = pipeline._copy_to_sources_local(existing)
        assert dest == existing


# ---------------------------------------------------------------------------
# TestFindAllByBasename
# ---------------------------------------------------------------------------


class TestFindAllByBasename:
    """Tests for _find_all_by_basename KV store lookup."""

    def _write_kv_store(self, config: CorpragConfig, data: dict) -> None:
        kv_path = config.working_dir_path / "kv_store_doc_status.json"
        kv_path.write_text(json.dumps(data))

    def test_exact_match(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        self._write_kv_store(
            test_config,
            {
                "doc-001": {"file_path": "/storage/sources/local/report.pdf"},
            },
        )
        matches = pipeline._find_all_by_basename("report.pdf")
        assert len(matches) == 1
        assert matches[0]["doc_id"] == "doc-001"

    def test_stem_match_fallback(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        self._write_kv_store(
            test_config,
            {
                "doc-001": {"file_path": "/storage/sources/local/report.pdf"},
            },
        )
        # Different extension but same stem
        matches = pipeline._find_all_by_basename("report.xlsx")
        assert len(matches) == 1

    def test_exact_before_stem(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        self._write_kv_store(
            test_config,
            {
                "doc-exact": {"file_path": "/storage/report.pdf"},
                "doc-stem": {"file_path": "/storage/report.docx"},
            },
        )
        matches = pipeline._find_all_by_basename("report.pdf")
        assert matches[0]["doc_id"] == "doc-exact"

    def test_no_kv_file(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        matches = pipeline._find_all_by_basename("anything.pdf")
        assert matches == []

    def test_corrupt_kv_file(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        kv_path = test_config.working_dir_path / "kv_store_doc_status.json"
        kv_path.write_text("not valid json {{{")
        matches = pipeline._find_all_by_basename("report.pdf")
        assert matches == []


# ---------------------------------------------------------------------------
# TestIngestSingleFileWithPolicy
# ---------------------------------------------------------------------------


class TestIngestSingleFileWithPolicy:
    """Core parse -> filter -> insert pipeline."""

    async def test_successful_ingest(self, test_config: CorpragConfig, tmp_path: Path) -> None:
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
        self, test_config: CorpragConfig, tmp_path: Path
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

    async def test_parse_document_error(self, test_config: CorpragConfig, tmp_path: Path) -> None:
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
        self, test_config: CorpragConfig, tmp_path: Path
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
        assert pipeline._hash_index.list_all() == []

    async def test_mineru_backend_passed_to_parse_kwargs(
        self, test_config: CorpragConfig, tmp_path: Path
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

    async def test_single_file_new(self, test_config: CorpragConfig, tmp_path: Path) -> None:
        pipeline = _make_pipeline(test_config)
        src = tmp_path / "new.pdf"
        src.write_text("content")

        result = await pipeline.aingest_from_local(src)

        assert result.source_type == "local"
        assert result.total_files == 1
        assert result.processed == 1
        assert result.skipped == 0

    async def test_single_file_duplicate_skipped(
        self, test_config: CorpragConfig, tmp_path: Path
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

    async def test_path_not_found(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        with pytest.raises(FileNotFoundError, match="Path not found"):
            await pipeline.aingest_from_local(Path("/nonexistent/file.pdf"))

    async def test_directory_multiple_files(
        self, test_config: CorpragConfig, tmp_path: Path
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

    async def test_directory_empty(self, test_config: CorpragConfig, tmp_path: Path) -> None:
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

    async def test_no_checker_no_error(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config, cancel_checker=None)
        # Should not raise
        await pipeline._check_cancelled()

    async def test_checker_returns_false(self, test_config: CorpragConfig) -> None:
        checker = AsyncMock(return_value=False)
        pipeline = _make_pipeline(test_config, cancel_checker=checker)
        await pipeline._check_cancelled()

    async def test_checker_returns_true(self, test_config: CorpragConfig) -> None:
        checker = AsyncMock(return_value=True)
        pipeline = _make_pipeline(test_config, cancel_checker=checker)
        with pytest.raises(IngestionCancelledError, match="cancelled by caller"):
            await pipeline._check_cancelled()

    async def test_asyncio_cancellation(self, test_config: CorpragConfig) -> None:
        pipeline = _make_pipeline(test_config)

        async def run():
            task = asyncio.current_task()
            task.cancel()
            await asyncio.sleep(0)  # Yield to allow cancellation
            await pipeline._check_cancelled()

        with pytest.raises(asyncio.CancelledError):
            await run()
