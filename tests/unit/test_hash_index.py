# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for HashIndex (JSON-based)."""

from __future__ import annotations

from pathlib import Path

from corprag.ingestion.hash_index import HashIndex, compute_file_hash


class TestComputeFileHash:
    """Test file hash computation."""

    def test_computes_sha256(self, tmp_path: Path) -> None:
        """Test SHA256 hash computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash_value = compute_file_hash(test_file)
        assert hash_value.startswith("sha256:")
        assert len(hash_value) == 71  # "sha256:" + 64 hex chars

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        """Test that identical content produces identical hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("same content")
        file2.write_text("same content")

        assert compute_file_hash(file1) == compute_file_hash(file2)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Test that different content produces different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content A")
        file2.write_text("content B")

        assert compute_file_hash(file1) != compute_file_hash(file2)


class TestHashIndex:
    """Test JSON-based hash index."""

    async def test_register_and_lookup(self, tmp_path: Path) -> None:
        """Test registering and looking up hashes."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir)

        content_hash = "sha256:abc123"
        await index.register(content_hash, "doc-001", "/path/to/file.pdf")

        entry = index.lookup(content_hash)
        assert entry is not None
        assert entry["doc_id"] == "doc-001"
        assert entry["file_path"] == "/path/to/file.pdf"

    def test_lookup_missing(self, tmp_path: Path) -> None:
        """Test lookup for non-existent hash."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir)
        assert index.lookup("sha256:nonexistent") is None

    async def test_remove(self, tmp_path: Path) -> None:
        """Test removing a hash entry."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir)
        content_hash = "sha256:abc123"

        await index.register(content_hash, "doc-001", "/path/to/file.pdf")
        assert index.lookup(content_hash) is not None

        result = await index.remove(content_hash)
        assert result is True
        assert index.lookup(content_hash) is None

    async def test_remove_missing(self, tmp_path: Path) -> None:
        """Test removing non-existent hash."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir)
        assert await index.remove("sha256:nonexistent") is False

    async def test_list_all(self, tmp_path: Path) -> None:
        """Test listing all entries."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir)
        await index.register("sha256:aaa", "doc-001", "/path/a.pdf")
        await index.register("sha256:bbb", "doc-002", "/path/b.pdf")

        entries = await index.list_all()
        assert len(entries) == 2

    async def test_invalidate(self, tmp_path: Path) -> None:
        """Test cache invalidation."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir)
        await index.register("sha256:aaa", "doc-001", "/path/a.pdf")

        # Invalidate should clear cache without error
        index.invalidate()

        # Lookup should still work after invalidation
        entry = index.lookup("sha256:aaa")
        assert entry is not None

    async def test_should_skip_file_new(self, tmp_path: Path) -> None:
        """Test should_skip_file for a new file."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        test_file = tmp_path / "new_file.txt"
        test_file.write_text("new content")

        index = HashIndex(tmp_path, sources_dir)
        should_skip, content_hash, reason = await index.should_skip_file(test_file, replace=False)

        assert not should_skip
        assert content_hash is not None
        assert content_hash.startswith("sha256:")

    async def test_should_skip_file_duplicate(self, tmp_path: Path) -> None:
        """Test should_skip_file for a duplicate file."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        test_file = tmp_path / "file.txt"
        test_file.write_text("same content")

        index = HashIndex(tmp_path, sources_dir)

        # Register the hash first
        content_hash = compute_file_hash(test_file)
        await index.register(content_hash, "doc-001", str(test_file))

        # Now check — should skip
        should_skip, _, reason = await index.should_skip_file(test_file, replace=False)
        assert should_skip
        assert reason is not None
