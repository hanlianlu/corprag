# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for HashIndex (JSON-based)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dlightrag.ingestion.hash_index import HashIndex, compute_file_hash


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


class TestContentAwarePdfHash:
    """Test PDF content-aware hashing."""

    def test_pdf_same_content_different_metadata_same_hash(self, tmp_path: Path) -> None:
        """Two PDFs with same text but different CreationDate produce same hash."""
        _create_pdf_with_metadata(tmp_path / "pdf1.pdf", "Hello World", "2026-01-01")
        _create_pdf_with_metadata(tmp_path / "pdf2.pdf", "Hello World", "2026-06-15")

        assert compute_file_hash(tmp_path / "pdf1.pdf") == compute_file_hash(tmp_path / "pdf2.pdf")

    def test_pdf_different_content_different_hash(self, tmp_path: Path) -> None:
        """PDFs with different text produce different hashes."""
        _create_pdf_with_metadata(tmp_path / "pdf1.pdf", "Content Alpha", "2026-01-01")
        _create_pdf_with_metadata(tmp_path / "pdf2.pdf", "Content Beta", "2026-01-01")

        assert compute_file_hash(tmp_path / "pdf1.pdf") != compute_file_hash(tmp_path / "pdf2.pdf")

    def test_pdf_no_text_falls_back_to_byte_hash(self, tmp_path: Path) -> None:
        """Scanned/image-only PDF falls back to byte hash without error."""
        pdf_bytes = _minimal_pdf_no_text()
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(pdf_bytes)

        result = compute_file_hash(pdf_path)
        assert result.startswith("sha256:")
        assert len(result) == 71

    def test_pdf_pypdfium2_unavailable_falls_back(self, tmp_path: Path, monkeypatch: Any) -> None:
        """When pypdfium2 import fails, falls back to byte hash."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pypdfium2":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(_minimal_pdf_no_text())

        result = compute_file_hash(pdf_path)
        assert result.startswith("sha256:")


class TestContentAwareOfficeHash:
    """Test Office (ZIP-based) content-aware hashing."""

    def test_docx_same_content_different_metadata_same_hash(self, tmp_path: Path) -> None:
        """Two DOCX files with same document.xml but different docProps produce same hash."""
        _create_docx(tmp_path / "a.docx", "<w:body>Hello</w:body>", "2026-01-01T00:00:00Z")
        _create_docx(tmp_path / "b.docx", "<w:body>Hello</w:body>", "2026-06-15T12:00:00Z")

        assert compute_file_hash(tmp_path / "a.docx") == compute_file_hash(tmp_path / "b.docx")

    def test_docx_different_content_different_hash(self, tmp_path: Path) -> None:
        """DOCX files with different document.xml produce different hashes."""
        _create_docx(tmp_path / "a.docx", "<w:body>Alpha</w:body>", "2026-01-01T00:00:00Z")
        _create_docx(tmp_path / "b.docx", "<w:body>Beta</w:body>", "2026-01-01T00:00:00Z")

        assert compute_file_hash(tmp_path / "a.docx") != compute_file_hash(tmp_path / "b.docx")

    def test_corrupted_zip_falls_back_to_byte_hash(self, tmp_path: Path) -> None:
        """Corrupted DOCX (invalid ZIP) falls back to byte hash."""
        bad_docx = tmp_path / "bad.docx"
        bad_docx.write_bytes(b"this is not a zip file")

        result = compute_file_hash(bad_docx)
        assert result.startswith("sha256:")
        assert len(result) == 71

    def test_non_office_extension_uses_byte_hash(self, tmp_path: Path) -> None:
        """Non-PDF, non-Office files use raw byte hash (unchanged behavior)."""
        txt = tmp_path / "readme.txt"
        txt.write_text("hello world")

        result = compute_file_hash(txt)
        assert result.startswith("sha256:")

        from dlightrag.ingestion.hash_index import _hash_file_bytes

        assert result == _hash_file_bytes(txt)


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

    async def test_clear(self, tmp_path: Path) -> None:
        """Test clear removes all entries and deletes the JSON file."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir)
        await index.register("sha256:aaa", "doc-001", "/path/a.pdf")
        await index.register("sha256:bbb", "doc-002", "/path/b.pdf")

        index_path = index._get_index_path()
        assert index_path.exists()

        await index.clear()

        assert not index_path.exists()
        assert index.check_exists("sha256:aaa") == (False, None)
        assert index.check_exists("sha256:bbb") == (False, None)

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


class TestHashIndexWorkspace:
    """Test JSON HashIndex workspace isolation."""

    async def test_workspace_creates_subdirectory(self, tmp_path: Path) -> None:
        """HashIndex with workspace stores files in workspace subdirectory."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir, workspace="project-a")
        await index.register("sha256:abc", "doc-1", "/path/a.pdf")

        # Hash file should be in workspace subdirectory
        hash_file = tmp_path / "project-a" / "file_content_hashes.json"
        assert hash_file.exists()

    async def test_different_workspaces_isolated(self, tmp_path: Path) -> None:
        """Different workspaces have separate hash indexes."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index_a = HashIndex(tmp_path, sources_dir, workspace="ws-a")
        index_b = HashIndex(tmp_path, sources_dir, workspace="ws-b")

        await index_a.register("sha256:same", "doc-a", "/path/a.pdf")

        # ws-b should NOT see ws-a's hash
        exists, _ = index_b.check_exists("sha256:same")
        assert not exists

    async def test_default_workspace_uses_subdirectory(self, tmp_path: Path) -> None:
        """Default workspace also uses a subdirectory for consistency."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        index = HashIndex(tmp_path, sources_dir, workspace="default")
        await index.register("sha256:abc", "doc-1", "/path/a.pdf")

        hash_file = tmp_path / "default" / "file_content_hashes.json"
        assert hash_file.exists()


def _create_pdf_with_metadata(path: Path, text: str, date_str: str) -> None:
    """Create a minimal PDF with given text and CreationDate metadata."""
    content = f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET".encode("latin-1")

    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>>endobj\n"
        b"4 0 obj<</Length " + str(len(content)).encode() + b">>\n"
        b"stream\n" + content + b"\nendstream\nendobj\n"
        b"5 0 obj<</CreationDate(D:"
        + date_str.replace("-", "").encode()
        + b"000000+00'00')>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000306 00000 n \n"
        b"0000000000 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R/Info 5 0 R>>\n"
        b"startxref\n0\n%%EOF"
    )
    Path(path).write_bytes(pdf)


def _minimal_pdf_no_text() -> bytes:
    """Return bytes for a minimal valid PDF with no extractable text."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n0\n%%EOF"
    )


def _create_docx(path: Path, body_xml: str, modified_time: str) -> None:
    """Create a minimal DOCX (ZIP) with given content and metadata timestamp."""
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", f'<?xml version="1.0"?>{body_xml}')
        zf.writestr(
            "docProps/core.xml",
            f'<?xml version="1.0"?><cp:coreProperties>'
            f"<dcterms:modified>{modified_time}</dcterms:modified>"
            f"</cp:coreProperties>",
        )
        zf.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships/>')
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')


class FakeRedis:
    """Minimal async Redis mock using a dict.

    Simulates decode_responses=True (LightRAG default), returning str not bytes.
    """

    def __init__(self):
        self._data: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, field: str, value: str) -> int:
        self._data.setdefault(key, {})[field] = value
        return 1

    async def hget(self, key: str, field: str) -> str | None:
        return self._data.get(key, {}).get(field)

    async def hdel(self, key: str, *fields: str) -> int:
        bucket = self._data.get(key, {})
        count = 0
        for f in fields:
            if f in bucket:
                del bucket[f]
                count += 1
        return count

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._data.get(key, {}))

    async def hkeys(self, key: str) -> list[str]:
        return list(self._data.get(key, {}).keys())

    async def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self._data:
                del self._data[k]
                count += 1
        return count


class TestRedisHashIndex:
    """Tests for RedisHashIndex using a mock Redis client."""

    @pytest.fixture
    def redis_index(self):
        from dlightrag.ingestion.hash_index import RedisHashIndex

        idx = RedisHashIndex(workspace="test", sources_dir=None)
        # Mock the Redis client with a dict-based fake
        idx._redis = FakeRedis()
        return idx

    async def test_register_and_check_exists(self, redis_index):
        await redis_index.register("sha256:aaa", "doc-001", "/path/a.pdf")
        exists, doc_id = await redis_index._async_check_exists("sha256:aaa")
        assert exists is True
        assert doc_id == "doc-001"

    async def test_check_exists_missing(self, redis_index):
        exists, doc_id = await redis_index._async_check_exists("sha256:missing")
        assert exists is False
        assert doc_id is None

    async def test_remove(self, redis_index):
        await redis_index.register("sha256:aaa", "doc-001", "/path/a.pdf")
        removed = await redis_index.remove("sha256:aaa")
        assert removed is True
        exists, _ = await redis_index._async_check_exists("sha256:aaa")
        assert exists is False

    async def test_clear(self, redis_index):
        await redis_index.register("sha256:aaa", "doc-001", "/a.pdf")
        await redis_index.register("sha256:bbb", "doc-002", "/b.pdf")
        await redis_index.clear()
        entries = await redis_index.list_all()
        assert len(entries) == 0

    async def test_list_all(self, redis_index):
        await redis_index.register("sha256:aaa", "doc-001", "/a.pdf")
        entries = await redis_index.list_all()
        assert len(entries) == 1
        assert entries[0]["doc_id"] == "doc-001"


class FakeDeleteResult:
    def __init__(self, count: int):
        self.deleted_count = count


class FakeMongoCollection:
    """Minimal async MongoDB collection mock."""

    def __init__(self):
        self._docs: dict[str, dict] = {}

    async def find_one(self, filter: dict) -> dict | None:
        _id = filter.get("_id")
        workspace = filter.get("workspace")
        doc = self._docs.get(_id)
        if doc and doc.get("workspace") == workspace:
            return doc
        return None

    async def update_one(self, filter: dict, update: dict, upsert: bool = False):
        _id = filter["_id"]
        if _id in self._docs:
            self._docs[_id].update(update.get("$set", {}))
        elif upsert:
            self._docs[_id] = {"_id": _id, **update.get("$set", {})}

    async def delete_one(self, filter: dict) -> FakeDeleteResult:
        _id = filter.get("_id")
        workspace = filter.get("workspace")
        doc = self._docs.get(_id)
        if doc and doc.get("workspace") == workspace:
            del self._docs[_id]
            return FakeDeleteResult(1)
        return FakeDeleteResult(0)

    async def delete_many(self, filter: dict) -> FakeDeleteResult:
        workspace = filter.get("workspace")
        to_del = [k for k, v in self._docs.items() if v.get("workspace") == workspace]
        for k in to_del:
            del self._docs[k]
        return FakeDeleteResult(len(to_del))

    def find(self, filter: dict):
        workspace = filter.get("workspace")
        docs = [v for v in self._docs.values() if v.get("workspace") == workspace]
        return FakeAsyncCursor(docs)

    async def create_index(self, field: str):
        pass


class FakeAsyncCursor:
    def __init__(self, docs: list):
        self._docs = docs
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._docs):
            raise StopAsyncIteration
        doc = self._docs[self._idx]
        self._idx += 1
        return doc


class TestMongoHashIndex:
    """Tests for MongoHashIndex using a mock collection."""

    @pytest.fixture
    def mongo_index(self):
        from dlightrag.ingestion.hash_index import MongoHashIndex

        idx = MongoHashIndex(workspace="test", sources_dir=None)
        idx._collection = FakeMongoCollection()
        return idx

    async def test_register_and_check_exists(self, mongo_index):
        await mongo_index.register("sha256:aaa", "doc-001", "/path/a.pdf")
        exists, doc_id = await mongo_index._async_check_exists("sha256:aaa")
        assert exists is True
        assert doc_id == "doc-001"

    async def test_check_exists_missing(self, mongo_index):
        exists, doc_id = await mongo_index._async_check_exists("sha256:missing")
        assert exists is False

    async def test_remove(self, mongo_index):
        await mongo_index.register("sha256:aaa", "doc-001", "/path/a.pdf")
        removed = await mongo_index.remove("sha256:aaa")
        assert removed is True

    async def test_clear(self, mongo_index):
        await mongo_index.register("sha256:aaa", "doc-001", "/a.pdf")
        await mongo_index.register("sha256:bbb", "doc-002", "/b.pdf")
        await mongo_index.clear()
        entries = await mongo_index.list_all()
        assert len(entries) == 0

    async def test_list_all(self, mongo_index):
        await mongo_index.register("sha256:aaa", "doc-001", "/a.pdf")
        entries = await mongo_index.list_all()
        assert len(entries) == 1
        assert entries[0]["doc_id"] == "doc-001"


class TestHashIndexFactory:
    """Test that _create_hash_index selects the right backend."""

    async def test_pg_backend_selection(self, tmp_path):
        """PG config should select PGHashIndex (or fall back to JSON without PG)."""
        from unittest.mock import MagicMock

        from dlightrag.service import RAGService

        config = MagicMock()
        config.kv_storage = "PGKVStorage"
        config.workspace = "test"
        config.sources_dir = None
        config.working_dir_path = tmp_path

        service = RAGService.__new__(RAGService)
        result = await service._create_hash_index(config)
        # PGHashIndex if PG is available, HashIndex fallback otherwise
        assert type(result).__name__ in ("PGHashIndex", "HashIndex")

    async def test_json_backend_selection(self, tmp_path):
        from unittest.mock import MagicMock

        from dlightrag.service import RAGService

        config = MagicMock()
        config.kv_storage = "JsonKVStorage"
        config.workspace = "test"
        config.sources_dir = None
        config.working_dir_path = tmp_path

        service = RAGService.__new__(RAGService)
        result = await service._create_hash_index(config)
        assert type(result).__name__ == "HashIndex"

    async def test_redis_fallback_to_json(self, tmp_path):
        """Redis config without redis package should fall back to JSON."""
        from unittest.mock import MagicMock

        from dlightrag.service import RAGService

        config = MagicMock()
        config.kv_storage = "RedisKVStorage"
        config.workspace = "test"
        config.sources_dir = None
        config.working_dir_path = tmp_path

        service = RAGService.__new__(RAGService)
        result = await service._create_hash_index(config)
        # Will be HashIndex if redis not installed, RedisHashIndex if installed
        assert type(result).__name__ in ("HashIndex", "RedisHashIndex")

    async def test_mongo_fallback_to_json(self, tmp_path):
        """Mongo config without motor package should fall back to JSON."""
        from unittest.mock import MagicMock

        from dlightrag.service import RAGService

        config = MagicMock()
        config.kv_storage = "MongoKVStorage"
        config.workspace = "test"
        config.sources_dir = None
        config.working_dir_path = tmp_path

        service = RAGService.__new__(RAGService)
        result = await service._create_hash_index(config)
        assert type(result).__name__ in ("HashIndex", "MongoHashIndex")
