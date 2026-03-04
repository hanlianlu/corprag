# Content-Aware File Deduplication

## Problem

PDF files downloaded multiple times have identical visible content but different binary bytes due to embedded metadata (CreationDate, ModDate, /ID trailer field). The current `compute_file_hash` hashes raw bytes, so these files produce different SHA256 hashes and bypass deduplication. Same issue applies to Office formats (DOCX/PPTX/XLSX) which embed timestamps in `docProps/core.xml`.

## Solution

Modify `compute_file_hash` in `src/dlightrag/ingestion/hash_index.py` to extract and hash only content — skipping volatile metadata — for PDF and Office formats.

### PDF (`.pdf`) — via pypdfium2 (already in transitive deps)

- Use `pypdfium2` to open the PDF and extract text from each page
- Concatenate all page text, SHA256 hash the result
- **Fallback**: if no text is extractable (scanned images, corrupted PDF, or pypdfium2 import fails), fall back to raw byte hash

### Office (`.docx`, `.pptx`, `.xlsx`) — via zipfile (stdlib)

These are ZIP archives. Hash only content files, skip metadata:

| Format | Hash these files | Skip these |
|--------|-----------------|------------|
| DOCX | `word/document.xml`, `word/media/*` | `docProps/*`, `_rels/*`, `[Content_Types].xml` |
| PPTX | `ppt/slides/*.xml`, `ppt/media/*` | `docProps/*`, `_rels/*`, `[Content_Types].xml` |
| XLSX | `xl/worksheets/*.xml`, `xl/sharedStrings.xml` | `docProps/*`, `_rels/*`, `[Content_Types].xml` |

- Read matching entries in sorted order (deterministic), feed bytes into SHA256
- **Fallback**: if ZIP extraction fails, fall back to raw byte hash

### Other formats

No change — full file byte hash as before.

## Hash format

Unchanged: `sha256:<hex_digest>`. Same content with different metadata now produces the same hash.

## Backward compatibility

- Existing hash index entries remain valid (old hashes won't collide with new ones)
- Old and new hashes coexist; old files won't be deduped against new uploads automatically
- Users can run `sync_hashes` to recompute hashes with the new logic

## Files to change

1. `src/dlightrag/ingestion/hash_index.py` — `compute_file_hash` function
2. `tests/unit/test_hash_index.py` — add tests for content-aware hashing

## Test plan

- PDF: two files with same content but different CreationDate -> same hash
- PDF: scanned/image-only PDF -> falls back to byte hash without error
- PDF: pypdfium2 unavailable -> falls back to byte hash
- DOCX: two files with same content but different docProps/core.xml timestamps -> same hash
- DOCX: corrupted ZIP -> falls back to byte hash
- Other formats (`.txt`, `.png`): unchanged behavior
