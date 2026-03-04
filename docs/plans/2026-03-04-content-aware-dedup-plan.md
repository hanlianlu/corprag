# Content-Aware File Deduplication Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `compute_file_hash` ignore volatile metadata in PDF and Office files so that identical content produces identical hashes regardless of download timestamps.

**Architecture:** Replace the single `compute_file_hash` function with a dispatcher that selects a content-extraction strategy by file extension: pypdfium2 text extraction for PDFs, ZIP content-file filtering for Office formats, raw byte hash for everything else. Each strategy falls back to raw byte hash on failure.

**Tech Stack:** pypdfium2 (transitive dep, already installed), zipfile (stdlib), hashlib (stdlib)

---

### Task 1: Add content-aware hash helpers to hash_index.py

**Files:**
- Modify: `src/dlightrag/ingestion/hash_index.py:44-54`

**Step 1: Write the three helper functions and update compute_file_hash**

Add above the existing `compute_file_hash` (line 44):

```python
# --- Content-aware hash extensions ---
# File extensions that use content-aware hashing
_PDF_EXTENSIONS = {".pdf"}
_OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}

# Office ZIP content paths to include (skip docProps/*, _rels/*, [Content_Types].xml)
_OFFICE_CONTENT_PREFIXES: dict[str, tuple[str, ...]] = {
    ".docx": ("word/document.xml", "word/media/"),
    ".pptx": ("ppt/slides/", "ppt/media/"),
    ".xlsx": ("xl/worksheets/", "xl/sharedStrings.xml"),
}


def _hash_pdf_content(file_path: Path) -> str | None:
    """Extract text from PDF pages via pypdfium2 and hash it.

    Returns sha256 hash string, or None to signal fallback to byte hash
    (when no text is extractable or pypdfium2 is unavailable).
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.debug("pypdfium2 not available, falling back to byte hash for %s", file_path.name)
        return None

    try:
        doc = pdfium.PdfDocument(str(file_path))
        sha256_hash = hashlib.sha256()
        has_text = False

        for i in range(len(doc)):
            page = doc[i]
            textpage = page.get_textpage()
            text = textpage.get_text_bounded()
            if text.strip():
                has_text = True
            sha256_hash.update(text.encode("utf-8"))

        if not has_text:
            logger.debug("No text extracted from %s, falling back to byte hash", file_path.name)
            return None

        return f"sha256:{sha256_hash.hexdigest()}"
    except Exception as exc:
        logger.debug("PDF content extraction failed for %s: %s", file_path.name, exc)
        return None


def _hash_office_content(file_path: Path) -> str | None:
    """Hash only content entries from Office ZIP archive, skipping metadata.

    Returns sha256 hash string, or None to signal fallback to byte hash.
    """
    import zipfile

    ext = file_path.suffix.lower()
    prefixes = _OFFICE_CONTENT_PREFIXES.get(ext)
    if not prefixes:
        return None

    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            sha256_hash = hashlib.sha256()
            has_content = False

            # Sort for deterministic ordering
            for name in sorted(zf.namelist()):
                if any(name.startswith(p) or name == p for p in prefixes):
                    sha256_hash.update(name.encode("utf-8"))
                    sha256_hash.update(zf.read(name))
                    has_content = True

            if not has_content:
                logger.debug("No content entries found in %s, falling back to byte hash", file_path.name)
                return None

            return f"sha256:{sha256_hash.hexdigest()}"
    except Exception as exc:
        logger.debug("Office content extraction failed for %s: %s", file_path.name, exc)
        return None
```

Then replace the existing `compute_file_hash` (lines 44-54) with:

```python
def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of file content.

    For PDF and Office formats, hashes only the actual content (text, media)
    and skips volatile metadata (timestamps, IDs) that change on every
    download/export. Falls back to raw byte hash if content extraction fails.

    Returns:
        Hash string in format "sha256:<hex_digest>"
    """
    suffix = file_path.suffix.lower()

    # PDF: extract text via pypdfium2
    if suffix in _PDF_EXTENSIONS:
        result = _hash_pdf_content(file_path)
        if result is not None:
            return result

    # Office: hash ZIP content entries only
    if suffix in _OFFICE_EXTENSIONS:
        result = _hash_office_content(file_path)
        if result is not None:
            return result

    # Fallback: raw byte hash
    return _hash_file_bytes(file_path, chunk_size)


def _hash_file_bytes(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of raw file bytes (original behavior)."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return f"sha256:{sha256_hash.hexdigest()}"
```

**Step 2: Run existing tests to verify no regression**

Run: `uv run python -m pytest tests/unit/test_hash_index.py -v`
Expected: All 11 tests PASS (existing .txt files use byte hash fallback — same behavior)

**Step 3: Commit**

```bash
git add src/dlightrag/ingestion/hash_index.py
git commit -m "feat: content-aware file hashing for PDF and Office dedup"
```

---

### Task 2: Add tests for PDF content-aware hashing

**Files:**
- Modify: `tests/unit/test_hash_index.py`

**Step 1: Write PDF dedup tests**

Add after `TestComputeFileHash` class (after line 39):

```python
class TestContentAwarePdfHash:
    """Test PDF content-aware hashing."""

    def test_pdf_same_content_different_metadata_same_hash(self, tmp_path: Path) -> None:
        """Two PDFs with same text but different CreationDate produce same hash."""
        import pypdfium2 as pdfium

        # Create two minimal PDFs with different metadata using pypdfium2
        for i, name in enumerate(["a.pdf", "b.pdf"]):
            doc = pdfium.PdfDocument.new()
            page = doc.new_page(200, 200)
            # Both PDFs have identical text content
            doc.save(str(tmp_path / name))

        # Both should hash to same value (empty text -> fallback to byte hash,
        # so we need to test with real PDFs that have text)
        # Instead, create PDFs with raw bytes that differ only in metadata
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
        # Create a minimal valid PDF with no extractable text
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


def _create_pdf_with_metadata(path: Path, text: str, date_str: str) -> None:
    """Create a minimal PDF with given text and CreationDate metadata."""
    # Minimal valid PDF with one page of text and a custom CreationDate
    content_stream = text.encode("latin-1")
    stream_length = len(content_stream) + len(b"BT /F1 12 Tf 100 700 Td () Tj ET") - 2 + len(content_stream)

    # Build the content stream properly
    content = f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET".encode("latin-1")

    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>>endobj\n"
        b"4 0 obj<</Length " + str(len(content)).encode() + b">>\n"
        b"stream\n" + content + b"\nendstream\nendobj\n"
        b"5 0 obj<</CreationDate(D:" + date_str.replace("-", "").encode() + b"000000+00'00')>>endobj\n"
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
```

Also add `from typing import Any` to the imports at top of test file (for monkeypatch type hint).

**Step 2: Run tests**

Run: `uv run python -m pytest tests/unit/test_hash_index.py -v`
Expected: All tests PASS including new PDF tests

**Step 3: Commit**

```bash
git add tests/unit/test_hash_index.py
git commit -m "test: add PDF content-aware hash tests"
```

---

### Task 3: Add tests for Office content-aware hashing

**Files:**
- Modify: `tests/unit/test_hash_index.py`

**Step 1: Write Office dedup tests**

Add after `TestContentAwarePdfHash`:

```python
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

        # Verify it matches raw byte hash
        from dlightrag.ingestion.hash_index import _hash_file_bytes
        assert result == _hash_file_bytes(txt)


def _create_docx(path: Path, body_xml: str, modified_time: str) -> None:
    """Create a minimal DOCX (ZIP) with given content and metadata timestamp."""
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        # Content file (hashed)
        zf.writestr("word/document.xml", f'<?xml version="1.0"?>{body_xml}')
        # Metadata file (skipped by content-aware hash)
        zf.writestr(
            "docProps/core.xml",
            f'<?xml version="1.0"?><cp:coreProperties>'
            f"<dcterms:modified>{modified_time}</dcterms:modified>"
            f"</cp:coreProperties>",
        )
        # Rels file (skipped)
        zf.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships/>')
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
```

**Step 2: Run all tests**

Run: `uv run python -m pytest tests/unit/test_hash_index.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/unit/test_hash_index.py
git commit -m "test: add Office content-aware hash tests"
```

---

### Task 4: Verify with real duplicate PDFs

**Step 1: Run verification against the actual duplicate files**

```bash
uv run python -c "
from pathlib import Path
from dlightrag.ingestion.hash_index import compute_file_hash

sources = Path('corprag_storage/sources/local')
for f in sorted(sources.glob('Project-Management*')):
    print(f'{compute_file_hash(f)[:30]}...  {f.name}')
"
```

Expected: All 6 files produce the **same** hash.

**Step 2: Run full test suite to confirm no regressions**

Run: `uv run python -m pytest tests/unit/ -v`
Expected: All tests PASS

**Step 3: Final commit (if any fixups needed)**
