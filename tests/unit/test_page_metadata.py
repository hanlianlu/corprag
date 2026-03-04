# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for page_idx metadata injection into text_chunks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from corprag.ingestion.page_metadata import (
    build_page_boundary_map,
    find_page_for_offset,
    inject_page_idx_to_chunks,
    reconstruct_merged_text,
)

# ---------------------------------------------------------------------------
# TestBuildPageBoundaryMap
# ---------------------------------------------------------------------------


class TestBuildPageBoundaryMap:
    """Test page boundary extraction from content_list."""

    def test_single_page(self) -> None:
        content_list = [
            {"type": "text", "text": "Hello world", "page_idx": 0},
        ]
        result = build_page_boundary_map(content_list)
        assert result == [(0, 0)]

    def test_multi_page(self) -> None:
        content_list = [
            {"type": "text", "text": "Page one", "page_idx": 0},
            {"type": "text", "text": "Page two", "page_idx": 1},
            {"type": "text", "text": "Page three", "page_idx": 2},
        ]
        result = build_page_boundary_map(content_list)
        # "Page one" starts at 0, len=8
        # "\n\n" separator = 2
        # "Page two" starts at 10, len=8
        # "\n\n" separator = 2
        # "Page three" starts at 20
        assert result == [(0, 0), (10, 1), (20, 2)]

    def test_skips_multimodal_items(self) -> None:
        content_list = [
            {"type": "text", "text": "Before image", "page_idx": 0},
            {"type": "image", "img_path": "/img.png", "page_idx": 0},
            {"type": "text", "text": "After image", "page_idx": 1},
        ]
        result = build_page_boundary_map(content_list)
        assert len(result) == 2
        assert result[0] == (0, 0)
        # "Before image" = 12 chars + 2 separator = offset 14
        assert result[1] == (14, 1)

    def test_skips_empty_text(self) -> None:
        content_list = [
            {"type": "text", "text": "Real text", "page_idx": 0},
            {"type": "text", "text": "   ", "page_idx": 0},  # whitespace-only
            {"type": "text", "text": "More text", "page_idx": 1},
        ]
        result = build_page_boundary_map(content_list)
        assert len(result) == 2
        # "Real text" = 9 chars + 2 = offset 11
        assert result == [(0, 0), (11, 1)]

    def test_missing_page_idx(self) -> None:
        content_list = [
            {"type": "text", "text": "Has page", "page_idx": 0},
            {"type": "text", "text": "No page field"},
            {"type": "text", "text": "Has page too", "page_idx": 2},
        ]
        result = build_page_boundary_map(content_list)
        # Block without page_idx is skipped in boundaries but still advances offset
        assert len(result) == 2
        assert result[0] == (0, 0)
        # "Has page"=8 + 2 + "No page field"=13 + 2 = offset 25
        assert result[1] == (25, 2)

    def test_empty_content_list(self) -> None:
        assert build_page_boundary_map([]) == []

    def test_no_text_blocks(self) -> None:
        content_list = [
            {"type": "image", "img_path": "/img.png", "page_idx": 0},
        ]
        assert build_page_boundary_map(content_list) == []

    def test_same_page_multiple_blocks(self) -> None:
        content_list = [
            {"type": "text", "text": "Paragraph 1", "page_idx": 3},
            {"type": "text", "text": "Paragraph 2", "page_idx": 3},
        ]
        result = build_page_boundary_map(content_list)
        assert result[0][1] == 3
        assert result[1][1] == 3


# ---------------------------------------------------------------------------
# TestFindPageForOffset
# ---------------------------------------------------------------------------


class TestFindPageForOffset:
    """Test binary search for page lookup."""

    def test_exact_boundary(self) -> None:
        boundaries = [(0, 0), (100, 1), (200, 2)]
        assert find_page_for_offset(boundaries, 100) == 1

    def test_between_boundaries(self) -> None:
        boundaries = [(0, 0), (100, 1), (200, 2)]
        assert find_page_for_offset(boundaries, 150) == 1

    def test_before_first_boundary(self) -> None:
        boundaries = [(10, 1), (100, 2)]
        # Offset before first boundary returns first page
        assert find_page_for_offset(boundaries, 5) == 1

    def test_after_last_boundary(self) -> None:
        boundaries = [(0, 0), (100, 1)]
        assert find_page_for_offset(boundaries, 500) == 1

    def test_empty_boundaries(self) -> None:
        assert find_page_for_offset([], 50) is None

    def test_single_boundary(self) -> None:
        boundaries = [(0, 5)]
        assert find_page_for_offset(boundaries, 0) == 5
        assert find_page_for_offset(boundaries, 999) == 5


# ---------------------------------------------------------------------------
# TestReconstructMergedText
# ---------------------------------------------------------------------------


class TestReconstructMergedText:
    """Test merged text reconstruction."""

    def test_basic_merge(self) -> None:
        content_list = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        assert reconstruct_merged_text(content_list) == "Hello\n\nWorld"

    def test_skips_multimodal(self) -> None:
        content_list = [
            {"type": "text", "text": "Before"},
            {"type": "image", "img_path": "/img.png"},
            {"type": "text", "text": "After"},
        ]
        assert reconstruct_merged_text(content_list) == "Before\n\nAfter"

    def test_skips_empty(self) -> None:
        content_list = [
            {"type": "text", "text": "A"},
            {"type": "text", "text": "   "},
            {"type": "text", "text": "B"},
        ]
        assert reconstruct_merged_text(content_list) == "A\n\nB"

    def test_empty_list(self) -> None:
        assert reconstruct_merged_text([]) == ""


# ---------------------------------------------------------------------------
# TestInjectPageIdxToChunks
# ---------------------------------------------------------------------------


def _make_lightrag(
    chunks_list: list[str],
    chunks_data: list[dict | None],
) -> MagicMock:
    """Create a mock lightrag with doc_status and text_chunks."""
    lightrag = MagicMock()

    # doc_status.get_by_id returns doc info with chunks_list
    lightrag.doc_status = MagicMock()
    lightrag.doc_status.get_by_id = AsyncMock(return_value={"chunks_list": chunks_list})

    # text_chunks.get_by_ids returns chunk data
    lightrag.text_chunks = MagicMock()
    lightrag.text_chunks.get_by_ids = AsyncMock(return_value=chunks_data)
    lightrag.text_chunks.upsert = AsyncMock()

    return lightrag


class TestInjectPageIdxToChunks:
    """Test end-to-end page_idx injection."""

    async def test_basic_injection(self) -> None:
        content_list = [
            {"type": "text", "text": "Page zero content here", "page_idx": 0},
            {"type": "text", "text": "Page one content here", "page_idx": 1},
        ]
        # Merged text: "Page zero content here\n\nPage one content here"
        # Chunk 0 matches "Page zero content here" → page 0
        # Chunk 1 matches "Page one content here" → page 1
        chunks_list = ["chunk-0", "chunk-1"]
        chunks_data = [
            {
                "content": "Page zero content here",
                "chunk_order_index": 0,
                "tokens": 5,
                "full_doc_id": "doc-1",
            },
            {
                "content": "Page one content here",
                "chunk_order_index": 1,
                "tokens": 5,
                "full_doc_id": "doc-1",
            },
        ]
        lightrag = _make_lightrag(chunks_list, chunks_data)

        updated = await inject_page_idx_to_chunks(lightrag, "doc-1", content_list)

        assert updated == 2
        lightrag.text_chunks.upsert.assert_called_once()
        upsert_arg = lightrag.text_chunks.upsert.call_args[0][0]
        assert upsert_arg["chunk-0"]["page_idx"] == 0
        assert upsert_arg["chunk-1"]["page_idx"] == 1

    async def test_chunk_spanning_page_boundary(self) -> None:
        content_list = [
            {"type": "text", "text": "Short", "page_idx": 0},
            {"type": "text", "text": "Also short", "page_idx": 1},
        ]
        # Merged: "Short\n\nAlso short"
        # A chunk that starts in page 0 region
        chunks_list = ["chunk-0"]
        chunks_data = [
            {
                "content": "Short\n\nAlso short",
                "chunk_order_index": 0,
                "tokens": 5,
                "full_doc_id": "doc-1",
            },
        ]
        lightrag = _make_lightrag(chunks_list, chunks_data)

        updated = await inject_page_idx_to_chunks(lightrag, "doc-1", content_list)

        assert updated == 1
        upsert_arg = lightrag.text_chunks.upsert.call_args[0][0]
        # Chunk starts at offset 0 → page 0
        assert upsert_arg["chunk-0"]["page_idx"] == 0

    async def test_no_page_idx_in_content(self) -> None:
        content_list = [
            {"type": "text", "text": "No page info"},
        ]
        lightrag = _make_lightrag([], [])

        updated = await inject_page_idx_to_chunks(lightrag, "doc-1", content_list)

        assert updated == 0
        lightrag.text_chunks.upsert.assert_not_called()

    async def test_no_lightrag(self) -> None:
        content_list = [{"type": "text", "text": "Hello", "page_idx": 0}]
        assert await inject_page_idx_to_chunks(None, "doc-1", content_list) == 0

    async def test_lightrag_without_text_chunks(self) -> None:
        lightrag = MagicMock(spec=[])  # No attributes
        content_list = [{"type": "text", "text": "Hello", "page_idx": 0}]
        assert await inject_page_idx_to_chunks(lightrag, "doc-1", content_list) == 0

    async def test_no_chunks_list_in_doc_status(self) -> None:
        content_list = [{"type": "text", "text": "Hello", "page_idx": 0}]
        lightrag = MagicMock()
        lightrag.doc_status = MagicMock()
        lightrag.doc_status.get_by_id = AsyncMock(return_value={"chunks_list": []})
        lightrag.text_chunks = MagicMock()

        updated = await inject_page_idx_to_chunks(lightrag, "doc-1", content_list)

        assert updated == 0

    async def test_doc_status_returns_none(self) -> None:
        content_list = [{"type": "text", "text": "Hello", "page_idx": 0}]
        lightrag = MagicMock()
        lightrag.doc_status = MagicMock()
        lightrag.doc_status.get_by_id = AsyncMock(return_value=None)
        lightrag.text_chunks = MagicMock()

        updated = await inject_page_idx_to_chunks(lightrag, "doc-1", content_list)

        assert updated == 0

    async def test_preserves_existing_fields(self) -> None:
        content_list = [{"type": "text", "text": "Content", "page_idx": 5}]
        chunks_data = [
            {
                "_id": "chunk-0",
                "content": "Content",
                "chunk_order_index": 0,
                "tokens": 1,
                "full_doc_id": "doc-1",
                "file_path": "/path/file.pdf",
                "llm_cache_list": [],
            },
        ]
        lightrag = _make_lightrag(["chunk-0"], chunks_data)

        await inject_page_idx_to_chunks(lightrag, "doc-1", content_list)

        upsert_arg = lightrag.text_chunks.upsert.call_args[0][0]
        chunk = upsert_arg["chunk-0"]
        # page_idx added
        assert chunk["page_idx"] == 5
        # Existing fields preserved
        assert chunk["content"] == "Content"
        assert chunk["full_doc_id"] == "doc-1"
        assert chunk["file_path"] == "/path/file.pdf"
        # _id field stripped (it's the key, not a stored field)
        assert "_id" not in chunk

    async def test_chunk_not_found_in_merged_text(self) -> None:
        content_list = [{"type": "text", "text": "Original text", "page_idx": 0}]
        chunks_data = [
            {
                "content": "Completely different content",
                "chunk_order_index": 0,
                "tokens": 3,
                "full_doc_id": "doc-1",
            },
        ]
        lightrag = _make_lightrag(["chunk-0"], chunks_data)

        updated = await inject_page_idx_to_chunks(lightrag, "doc-1", content_list)

        # Chunk content not found in merged text → skipped
        assert updated == 0

    async def test_only_multimodal_content(self) -> None:
        content_list = [
            {"type": "image", "img_path": "/img.png", "page_idx": 0},
            {"type": "table", "table_body": "<table>", "page_idx": 1},
        ]
        lightrag = _make_lightrag([], [])

        updated = await inject_page_idx_to_chunks(lightrag, "doc-1", content_list)

        assert updated == 0
