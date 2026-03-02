# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for retrieval engine: path extraction, URL generation, result augmentation."""

from __future__ import annotations

import json
from pathlib import Path

from corprag.retrieval.engine import (
    RetrievalResult,
    _extract_rag_relative,
    _load_kv_store_page_indices,
    _to_download_url,
    augment_retrieval_result,
    build_sources_and_media_from_contexts,
)

# ---------------------------------------------------------------------------
# TestExtractRagRelative
# ---------------------------------------------------------------------------


class TestExtractRagRelative:
    """Test _extract_rag_relative path extraction."""

    def test_with_working_dir(self) -> None:
        result = _extract_rag_relative(
            "/abs/corprag_storage/sources/local/file.pdf",
            working_dir="/abs/corprag_storage",
        )
        assert result == "sources/local/file.pdf"

    def test_with_working_dir_trailing_slash(self) -> None:
        result = _extract_rag_relative(
            "/abs/corprag_storage/sources/local/file.pdf",
            working_dir="/abs/corprag_storage/",
        )
        assert result == "sources/local/file.pdf"

    def test_fallback_sources_marker(self) -> None:
        result = _extract_rag_relative(
            "/random/prefix/sources/local/file.pdf",
            working_dir=None,
        )
        assert result == "sources/local/file.pdf"

    def test_fallback_artifacts_marker(self) -> None:
        result = _extract_rag_relative(
            "/random/prefix/artifacts/local/report/page1.png",
            working_dir=None,
        )
        assert result == "artifacts/local/report/page1.png"

    def test_no_match(self) -> None:
        result = _extract_rag_relative(
            "/random/path/file.pdf",
            working_dir="/other/dir",
        )
        assert result is None


# ---------------------------------------------------------------------------
# TestToDownloadUrl
# ---------------------------------------------------------------------------


class TestToDownloadUrl:
    """Test _to_download_url URL generation."""

    def test_with_url_transformer(self) -> None:
        def transformer(p: str) -> str:
            return f"https://cdn.example.com/{p}"

        result = _to_download_url("/some/path.pdf", url_transformer=transformer)
        assert result == "https://cdn.example.com//some/path.pdf"

    def test_file_scheme_stripped(self) -> None:
        result = _to_download_url(
            "file:///abs/corprag_storage/sources/local/file.pdf",
            working_dir="/abs/corprag_storage",
        )
        assert result == "file://sources/local/file.pdf"

    def test_azure_passthrough(self) -> None:
        result = _to_download_url("azure://container/blob.pdf")
        assert result == "azure://container/blob.pdf"

    def test_relative_extraction_no_transformer(self) -> None:
        result = _to_download_url(
            "/abs/corprag_storage/sources/local/file.pdf",
            working_dir="/abs/corprag_storage",
        )
        assert result == "file://sources/local/file.pdf"

    def test_fallback_full_path(self) -> None:
        result = _to_download_url("/random/path.pdf")
        assert result == "file:///random/path.pdf"


# ---------------------------------------------------------------------------
# TestLoadKvStorePageIndices
# ---------------------------------------------------------------------------


class TestLoadKvStorePageIndices:
    """Test KV store page index loading."""

    def test_loads_and_converts_to_1_based(self, tmp_path: Path) -> None:
        kv_path = tmp_path / "kv_store_text_chunks.json"
        kv_path.write_text(
            json.dumps(
                {
                    "chunk-001": {"page_idx": 0},
                    "chunk-002": {"page_idx": 4},
                }
            )
        )
        result = _load_kv_store_page_indices(str(tmp_path))
        assert result["chunk-001"] == 1
        assert result["chunk-002"] == 5

    def test_none_page_idx(self, tmp_path: Path) -> None:
        kv_path = tmp_path / "kv_store_text_chunks.json"
        kv_path.write_text(
            json.dumps(
                {
                    "chunk-001": {"page_idx": None},
                }
            )
        )
        result = _load_kv_store_page_indices(str(tmp_path))
        assert result["chunk-001"] is None

    def test_file_not_found(self, tmp_path: Path) -> None:
        result = _load_kv_store_page_indices(str(tmp_path))
        assert result == {}

    def test_corrupt_json(self, tmp_path: Path) -> None:
        kv_path = tmp_path / "kv_store_text_chunks.json"
        kv_path.write_text("not valid json {{{")
        result = _load_kv_store_page_indices(str(tmp_path))
        assert result == {}


# ---------------------------------------------------------------------------
# TestBuildSourcesAndMediaFromContexts
# ---------------------------------------------------------------------------


class TestBuildSourcesAndMediaFromContexts:
    """Test source/media extraction from chunk contexts."""

    def test_single_source_extracted(self) -> None:
        contexts = [
            {
                "chunk_id": "c1",
                "file_path": "/storage/sources/local/report.pdf",
                "reference_id": "ref-001",
                "content": "Some important text content here",
            },
        ]
        sources, media = build_sources_and_media_from_contexts(contexts)
        assert len(sources) == 1
        assert sources[0]["id"] == "ref-001"
        assert sources[0]["title"] == "report.pdf"
        assert sources[0]["path"] == "/storage/sources/local/report.pdf"
        assert "c1" in sources[0]["chunk_ids"]

    def test_dedup_by_reference_id(self) -> None:
        contexts = [
            {
                "chunk_id": "c1",
                "file_path": "/storage/report.pdf",
                "reference_id": "ref-001",
                "content": "chunk 1",
            },
            {
                "chunk_id": "c2",
                "file_path": "/storage/report.pdf",
                "reference_id": "ref-001",
                "content": "chunk 2",
            },
        ]
        sources, media = build_sources_and_media_from_contexts(contexts)
        assert len(sources) == 1
        assert set(sources[0]["chunk_ids"]) == {"c1", "c2"}

    def test_image_extracted_from_content(self) -> None:
        contexts = [
            {
                "chunk_id": "c1",
                "file_path": "/storage/report.pdf",
                "reference_id": "ref-001",
                "content": "Image Path: /storage/artifacts/img.png\nCaption: A chart showing growth",
            },
        ]
        sources, media = build_sources_and_media_from_contexts(contexts)
        assert len(media) == 1
        assert media[0]["type"] == "image"
        assert media[0]["path"] == "/storage/artifacts/img.png"
        assert media[0]["caption"] == "A chart showing growth"

    def test_no_chunk_id_skipped(self) -> None:
        contexts = [
            {
                "file_path": "/storage/report.pdf",
                "reference_id": "ref-001",
                "content": "text",
            },
        ]
        sources, media = build_sources_and_media_from_contexts(contexts)
        assert len(sources) == 0

    def test_no_file_path_skipped(self) -> None:
        contexts = [
            {
                "chunk_id": "c1",
                "reference_id": "ref-001",
                "content": "text",
            },
        ]
        sources, media = build_sources_and_media_from_contexts(contexts)
        assert len(sources) == 0


# ---------------------------------------------------------------------------
# TestAugmentRetrievalResult
# ---------------------------------------------------------------------------


class TestAugmentRetrievalResult:
    """Test result augmentation with sources/media."""

    def test_attaches_sources(self) -> None:
        result = RetrievalResult(
            answer=None,
            contexts={
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "file_path": "/storage/sources/local/doc.pdf",
                        "reference_id": "ref-001",
                        "content": "Text content",
                    },
                ],
            },
            raw={},
        )
        augmented = augment_retrieval_result(result)
        assert "sources" in augmented.raw
        assert len(augmented.raw["sources"]) == 1

    def test_page_idx_injected(self, tmp_path: Path) -> None:
        kv_path = tmp_path / "kv_store_text_chunks.json"
        kv_path.write_text(json.dumps({"c1": {"page_idx": 2}}))

        result = RetrievalResult(
            answer=None,
            contexts={
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "file_path": "/storage/doc.pdf",
                        "reference_id": "ref-001",
                        "content": "text",
                    },
                ],
            },
            raw={},
        )
        augmented = augment_retrieval_result(result, rag_working_dir=str(tmp_path))
        chunk = augmented.contexts["chunks"][0]
        assert chunk["page_idx"] == 3  # 0-based 2 -> 1-based 3

    def test_no_contexts_no_crash(self) -> None:
        result = RetrievalResult(answer=None, contexts={}, raw={})
        augmented = augment_retrieval_result(result)
        assert augmented is result
