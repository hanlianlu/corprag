# Eliminate sources/ Directory — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the redundant `sources/` copy from the ingestion pipeline, record true original source URIs in metadata, and use temp directories for intermediate files.

**Architecture:** Local files are read directly from their original paths. Remote files are downloaded to `{working_dir}/.tmp/` and cleaned up after parsing. All metadata (`hash_index`, LightRAG `doc_status`) records the true original source URI instead of the local copy path.

**Tech Stack:** Python 3.12, asyncio, tempfile, shutil, pytest, pytest-asyncio

**Design doc:** `docs/plans/2026-03-05-eliminate-sources-design.md`

---

### Task 1: Add `derive_source_type()` helper and `temp_dir` config property

Add a shared module-level helper to derive source type from a file path / URI (replacing the fragile "look for `/sources/` in path" logic), and add `temp_dir` to config.

**Files:**
- Modify: `src/dlightrag/ingestion/hash_index.py:319-340,487-515,658-679,798-818`
- Modify: `src/dlightrag/config.py:305-311`
- Test: `tests/unit/test_hash_index.py`
- Test: `tests/unit/test_config.py`

**Step 1: Write the failing tests**

In `tests/unit/test_hash_index.py`, add a new test class at the end of the file:

```python
class TestDeriveSourceType:
    """Tests for derive_source_type() module-level helper."""

    def test_azure_uri(self):
        from dlightrag.ingestion.hash_index import derive_source_type
        assert derive_source_type("azure://container/path/file.pdf") == "azure_blobs"

    def test_snowflake_uri(self):
        from dlightrag.ingestion.hash_index import derive_source_type
        assert derive_source_type("snowflake://source_label") == "snowflake"

    def test_local_absolute_path(self):
        from dlightrag.ingestion.hash_index import derive_source_type
        assert derive_source_type("/Users/me/docs/report.pdf") == "local"

    def test_local_relative_path(self):
        from dlightrag.ingestion.hash_index import derive_source_type
        assert derive_source_type("report.pdf") == "local"

    def test_legacy_sources_path(self):
        """Backward compat: old entries with sources/ still work."""
        from dlightrag.ingestion.hash_index import derive_source_type
        assert derive_source_type("/abs/path/sources/local/file.pdf") == "local"
        assert derive_source_type("/abs/path/sources/azure_blobs/c/file.pdf") == "azure_blobs"

    def test_empty_string(self):
        from dlightrag.ingestion.hash_index import derive_source_type
        assert derive_source_type("") == "unknown"
```

In `tests/unit/test_config.py`, add a test for `temp_dir`:

```python
def test_temp_dir_property(test_config):
    assert test_config.temp_dir == test_config.working_dir_path / ".tmp"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_hash_index.py::TestDeriveSourceType -v`
Expected: FAIL with `ImportError: cannot import name 'derive_source_type'`

Run: `python -m pytest tests/unit/test_config.py::test_temp_dir_property -v`
Expected: FAIL with `AttributeError: ... has no attribute 'temp_dir'`

**Step 3: Implement**

In `src/dlightrag/ingestion/hash_index.py`, add after the `SUPPORTED_EXTENSIONS` block (~line 43):

```python
def derive_source_type(file_path: str) -> str:
    """Derive source type from a file path or URI.

    Returns "azure_blobs", "snowflake", "local", or "unknown".
    Handles both new-style URIs (azure://...) and legacy sources/ paths.
    """
    if not file_path:
        return "unknown"
    if file_path.startswith("azure://"):
        return "azure_blobs"
    if file_path.startswith("snowflake://"):
        return "snowflake"
    # Legacy: /abs/path/sources/{source_type}/file.pdf
    parts = Path(file_path).parts
    idx = next((i for i, p in enumerate(parts) if p == "sources"), -1)
    if idx >= 0 and len(parts) > idx + 1:
        return parts[idx + 1]
    # Default for local absolute/relative paths
    if file_path.startswith("/") or file_path.startswith(".") or "://" not in file_path:
        return "local"
    return "unknown"
```

Add it to `__all__` at the end of the file.

Now update all four `list_all()` methods in HashIndex, PGHashIndex, RedisHashIndex, MongoHashIndex:
Replace the inline source_type derivation block (the 5-line `source_type = "unknown" ... sources_idx ...` pattern) with a single call: `source_type = derive_source_type(file_path)` (or `derive_source_type(fp)`).

In `src/dlightrag/config.py`, add after `sources_dir` property (line 307):

```python
@property
def temp_dir(self) -> Path:
    return self.working_dir_path / ".tmp"
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_hash_index.py::TestDeriveSourceType tests/unit/test_config.py::test_temp_dir_property -v`
Expected: ALL PASS

Also run full test suite to ensure no regressions:
Run: `python -m pytest tests/unit/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/dlightrag/ingestion/hash_index.py src/dlightrag/config.py tests/unit/test_hash_index.py tests/unit/test_config.py
git commit -m "feat: add derive_source_type helper and temp_dir config property"
```

---

### Task 2: Delete `sources_dir`, `sync_existing`, and `sync_hashes` from entire chain

Cleanly remove the entire `sync_existing` → `sync_hashes` → `sources_dir` chain. These have zero value without a `sources/` directory.

**The full chain to delete:**
```
api/server.py    sync_hashes field in IngestRequest
    ↓
service.py       sync_hashes kwarg passing
    ↓
pipeline.py      sync_hashes param in aingest_from_local / aingest_from_azure_blob
    ↓
hash_index.py    sync_existing() method in all 4 backends + sources_dir param
    ↓
cli.py           --sync-hashes flag
```

**Files:**
- Modify: `src/dlightrag/ingestion/hash_index.py` (all 4 backends)
- Modify: `src/dlightrag/ingestion/pipeline.py:96-100,366,387-388,529,548-549`
- Modify: `src/dlightrag/service.py:389-439,477-478,495,506,536`
- Modify: `src/dlightrag/api/server.py:89,131`
- Modify: `scripts/cli.py:122,139,149,322`
- Test: `tests/unit/test_hash_index.py`
- Test: `tests/unit/test_pipeline.py`
- Test: `tests/unit/test_cli.py:154`
- Test: `tests/unit/test_api_server.py`

**Step 1: Write tests that verify the clean removal**

In `tests/unit/test_hash_index.py`, add:

```python
class TestSyncExistingRemoved:
    """Verify sync_existing is fully removed from all backends."""

    def test_json_hash_index_no_sync_existing(self, tmp_path):
        index = HashIndex(tmp_path)
        assert not hasattr(index, "sync_existing")

    def test_json_hash_index_no_sources_dir(self, tmp_path):
        """HashIndex constructor no longer accepts sources_dir."""
        index = HashIndex(tmp_path)
        assert not hasattr(index, "_sources_dir")
```

In `tests/unit/test_pipeline.py`, verify `sync_hashes` param is gone:

```python
class TestSyncHashesRemoved:
    @pytest.mark.asyncio
    async def test_aingest_from_local_no_sync_hashes_param(self, test_config):
        """aingest_from_local no longer accepts sync_hashes."""
        import inspect
        pipeline = _make_pipeline(test_config)
        sig = inspect.signature(pipeline.aingest_from_local)
        assert "sync_hashes" not in sig.parameters
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_hash_index.py::TestSyncExistingRemoved -v`
Expected: FAIL (sync_existing still exists)

**Step 3: Implement**

**hash_index.py — all 4 backends:**

**HashIndex (JSON):**
- Change `__init__` signature: `(self, working_dir: Path, sources_dir: Path, workspace: str = "default")` → `(self, working_dir: Path, workspace: str = "default")`
- Delete `self._sources_dir = sources_dir` (line 177)
- Delete entire `sync_existing()` method (lines 272-302)

**PGHashIndex:**
- Remove `sources_dir: Path | None = None` from `__init__` (line 356)
- Delete `self._sources_dir = sources_dir` (line 361)
- Delete entire `sync_existing()` method (lines 517-548)

**RedisHashIndex:**
- Remove `sources_dir: Path | None = None` from `__init__` (line 563)
- Delete `self._sources_dir = sources_dir` (line 565)
- Delete entire `sync_existing()` method (lines 681-712)

**MongoHashIndex:**
- Remove `sources_dir: Path | None = None` from `__init__` (line 724)
- Delete `self._sources_dir = sources_dir` (line 726)
- Delete entire `sync_existing()` method (lines 820-848)

**pipeline.py:**
- `__init__` (line 97-100): Change `HashIndex(self.config.working_dir_path, self.config.sources_dir)` → `HashIndex(self.config.working_dir_path)`
- `aingest_from_local()`: Remove `sync_hashes: bool = False` param (line 366), remove docstring line about sync_hashes (line 380), delete `if sync_hashes: await self._hash_index.sync_existing()` block (lines 387-388)
- `aingest_from_azure_blob()`: Remove `sync_hashes: bool = False` param (line 529), remove docstring line (line 542), delete sync_hashes block (lines 548-549)

**service.py:**
- `aingest()`: Remove `sync_hashes` from docstring (lines 477-478), delete `sync_hashes = bool(kwargs.get("sync_hashes", False))` (line 495), remove `sync_hashes=sync_hashes` from `aingest_from_local` call (line 506), remove from `aingest_from_azure_blob` call (line 536)
- `_create_hash_index()` (lines 389-439):
  - Line 401: `PGHashIndex(workspace=config.workspace, sources_dir=config.sources_dir)` → `PGHashIndex(workspace=config.workspace)`
  - Line 414: `RedisHashIndex(workspace=config.workspace, sources_dir=config.sources_dir)` → `RedisHashIndex(workspace=config.workspace)`
  - Line 427: `MongoHashIndex(workspace=config.workspace, sources_dir=config.sources_dir)` → `MongoHashIndex(workspace=config.workspace)`
  - Line 439: `HashIndex(config.working_dir_path, config.sources_dir, workspace=config.workspace)` → `HashIndex(config.working_dir_path, workspace=config.workspace)`

**api/server.py:**
- Remove `sync_hashes: bool = False` from `IngestRequest` model (line 89)
- Remove `kwargs["sync_hashes"] = body.sync_hashes` (line 131)

**scripts/cli.py:**
- Delete `--sync-hashes` argument definition (around line 322)
- Remove `if args.sync_hashes:` block (line 122)
- Remove `kwargs["sync_hashes"] = args.sync_hashes` lines (lines 139, 149)

**Update tests:**
- `tests/unit/test_cli.py`: Remove or update `test_snowflake_rejects_sync_hashes` (line 154) — the param no longer exists
- `tests/unit/test_hash_index.py`: Remove any tests that reference `sync_existing` or `sources_dir` param. Update `HashIndex(...)` constructor calls to drop `sources_dir`.
- `tests/unit/test_api_server.py`: Remove any `sync_hashes` references in request bodies
- `tests/unit/test_pipeline.py`: Remove any `sync_hashes` references

**Step 4: Run tests**

Run: `python -m pytest tests/unit/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/dlightrag/ingestion/hash_index.py src/dlightrag/ingestion/pipeline.py src/dlightrag/service.py src/dlightrag/api/server.py scripts/cli.py tests/unit/
git commit -m "refactor: delete sync_existing, sources_dir, sync_hashes from entire chain"
```

---

### Task 3: Pipeline — add `source_uri` parameter and temp infrastructure

Add `source_uri` to `_ingest_single_file_with_policy()` so metadata records the true original source. Add temp directory helpers.

**Files:**
- Modify: `src/dlightrag/ingestion/pipeline.py:128-137,155-176,269-355`
- Test: `tests/unit/test_pipeline.py`

**Step 1: Write the failing tests**

In `tests/unit/test_pipeline.py`, add tests:

```python
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
        call_kwargs = pipeline.rag.insert_content_list.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("file_path") == "/original/path/test.txt" or \
               call_kwargs[1].get("file_path") == "/original/path/test.txt"

    @pytest.mark.asyncio
    async def test_source_uri_passed_to_hash_index(self, test_config):
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

        # Check hash_index.register was called with source_uri
        entries = await pipeline._hash_index.list_all()
        if entries:
            assert entries[0]["file_path"] == "/original/path/test.txt"

    @pytest.mark.asyncio
    async def test_prepare_for_parsing_non_excel(self, test_config):
        """Non-Excel files pass through unchanged."""
        pipeline = _make_pipeline(test_config)
        test_file = test_config.working_dir_path / "test.pdf"
        test_file.write_text("pdf content")
        tmpdir = pipeline._create_temp_dir()
        result = await pipeline._prepare_for_parsing(test_file, tmpdir)
        assert result == test_file  # unchanged
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_pipeline.py::TestTempDirAndSourceUri -v`
Expected: FAIL — methods don't exist yet

**Step 3: Implement**

In `src/dlightrag/ingestion/pipeline.py`:

**Add `_create_temp_dir()` method** (after `_get_storage_dir`, around line 137):

```python
def _create_temp_dir(self) -> Path:
    """Create a unique temp directory under working_dir/.tmp/."""
    import uuid
    tmpdir = self.config.temp_dir / uuid.uuid4().hex[:12]
    tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir
```

**Add `_prepare_for_parsing()` method** (after `_create_temp_dir`):

```python
async def _prepare_for_parsing(self, file_path: Path, tmpdir: Path) -> Path:
    """Prepare file for parsing. Converts Excel to PDF if needed, otherwise returns original."""
    return await self._maybe_convert_excel_to_pdf(file_path, tmpdir)
```

**Modify `_ingest_single_file_with_policy()`** (lines 269-355):

Add `source_uri: str | None = None` parameter. Default to `str(file_path)` if None (backward compat during transition).

Change line 311-315:
```python
# OLD: file_path=str(file_path)
# NEW: file_path=source_uri or str(file_path)
await self.rag.insert_content_list(
    content_list=result.index_stream,
    file_path=source_uri or str(file_path),
    doc_id=doc_id,
)
```

Change line 338:
```python
# OLD: await self._hash_index.register(content_hash, doc_id, str(file_path))
# NEW:
await self._hash_index.register(content_hash, doc_id, source_uri or str(file_path))
```

Change line 344 (IngestionResult.source_path):
```python
source_path=source_uri or str(file_path),
```

**Step 4: Run tests**

Run: `python -m pytest tests/unit/test_pipeline.py -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/dlightrag/ingestion/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: add source_uri param and temp dir infrastructure to pipeline"
```

---

### Task 4: Local ingestion — eliminate sources/ copy

Remove `_acopy_to_sources_local()`. Rewrite `aingest_from_local()` to read directly from the original path, using the unified temp pattern for Excel conversion.

**Files:**
- Modify: `src/dlightrag/ingestion/pipeline.py:178-211,361-516`
- Test: `tests/unit/test_pipeline.py`

**Step 1: Write the failing tests**

In `tests/unit/test_pipeline.py`, update or add tests in `TestAingestFromLocal`:

```python
class TestAingestFromLocal:

    @pytest.mark.asyncio
    async def test_single_file_records_original_path(self, test_config):
        """Ingested file metadata records the original path, not a sources/ copy."""
        pipeline = _make_pipeline(test_config)
        test_file = test_config.working_dir_path / "outside" / "report.pdf"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("test content")

        result = await pipeline.aingest_from_local(test_file)

        assert result.status == "success"
        assert result.processed == 1
        # source_path should be the original, not sources/local/
        assert "sources" not in (result.source_path or "")
        assert str(test_file.resolve()) in (result.source_path or "")

    @pytest.mark.asyncio
    async def test_no_copy_to_sources_dir(self, test_config):
        """Files are NOT copied to sources/ anymore."""
        pipeline = _make_pipeline(test_config)
        test_file = test_config.working_dir_path / "outside" / "report.pdf"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("test content")

        await pipeline.aingest_from_local(test_file)

        sources_local = test_config.working_dir_path / "sources" / "local"
        if sources_local.exists():
            assert not list(sources_local.iterdir()), "No files should be in sources/local/"

    @pytest.mark.asyncio
    async def test_temp_cleaned_up_after_ingestion(self, test_config):
        """Temp dir is cleaned up after ingestion completes."""
        pipeline = _make_pipeline(test_config)
        test_file = test_config.working_dir_path / "outside" / "report.pdf"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("test content")

        await pipeline.aingest_from_local(test_file)

        tmp_dir = test_config.temp_dir
        if tmp_dir.exists():
            # Should have no subdirectories (all cleaned up)
            assert not list(tmp_dir.iterdir()), "Temp dirs should be cleaned up"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_pipeline.py::TestAingestFromLocal -v`
Expected: FAIL — still copies to sources/

**Step 3: Implement**

In `src/dlightrag/ingestion/pipeline.py`:

**Delete `_acopy_to_sources_local()`** (lines 178-211).

**Rewrite `aingest_from_local()`** (lines 361-516):

For single file (lines 392-428), replace:
```python
# OLD:
# source_path = await self._acopy_to_sources_local(path)
# result = await self._ingest_single_file_with_policy(source_path, artifacts_dir, ...)

# NEW:
source_uri = str(path.resolve())
tmpdir = self._create_temp_dir()
try:
    working_path = await self._prepare_for_parsing(path, tmpdir)
    result = await self._ingest_single_file_with_policy(
        working_path, artifacts_dir, content_hash=content_hash, source_uri=source_uri
    )
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
```

For directory (lines 430-516), replace the `_acopy_to_sources_local` call in the loop:
```python
# OLD:
# copied_path = await self._acopy_to_sources_local(file_path)
# files_to_process.append((copied_path, content_hash or ""))

# NEW:
files_to_process.append((file_path, content_hash or "", str(file_path.resolve())))
```

And update the batch processing to wrap each file in a temp dir:
```python
async def _ingest_local_single(fp: Path, ch: str, uri: str) -> IngestionResult:
    tmpdir = self._create_temp_dir()
    try:
        working = await self._prepare_for_parsing(fp, tmpdir)
        return await self._ingest_single_file_with_policy(
            working, artifacts_dir, content_hash=ch if ch else None, source_uri=uri
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

tasks = [_ingest_local_single(fp, ch, uri) for fp, ch, uri in files_to_process]
```

**Step 4: Run tests**

Run: `python -m pytest tests/unit/test_pipeline.py -x -q`
Expected: ALL PASS

Run: `python -m pytest tests/unit/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/dlightrag/ingestion/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: local ingestion reads from original path, no sources/ copy"
```

---

### Task 5: Azure ingestion — temp downloads

Replace the permanent `sources/azure_blobs/` download with temp directory downloads. Build `azure://` source URIs.

**Files:**
- Modify: `src/dlightrag/ingestion/pipeline.py:213-230,522-733`
- Test: `tests/unit/test_pipeline.py`

**Step 1: Write the failing tests**

```python
class TestAingestFromAzureBlob:

    @pytest.mark.asyncio
    async def test_azure_source_uri_format(self, test_config):
        """Azure blobs record azure://container/path as source_uri."""
        pipeline = _make_pipeline(test_config)
        mock_source = AsyncMock()
        mock_source.aload_document = AsyncMock(return_value=b"blob content")

        result = await pipeline.aingest_from_azure_blob(
            source=mock_source,
            container_name="mycontainer",
            blob_path="data/report.pdf",
        )

        assert result.status == "success"
        # Verify source_uri format
        if pipeline.rag.insert_content_list.called:
            call_kwargs = pipeline.rag.insert_content_list.call_args
            file_path_arg = call_kwargs.kwargs.get("file_path", "")
            assert file_path_arg == "azure://mycontainer/data/report.pdf"

    @pytest.mark.asyncio
    async def test_azure_no_permanent_download(self, test_config):
        """Azure blobs are NOT permanently stored in sources/."""
        pipeline = _make_pipeline(test_config)
        mock_source = AsyncMock()
        mock_source.aload_document = AsyncMock(return_value=b"blob content")

        await pipeline.aingest_from_azure_blob(
            source=mock_source,
            container_name="mycontainer",
            blob_path="data/report.pdf",
        )

        sources_azure = test_config.working_dir_path / "sources" / "azure_blobs"
        if sources_azure.exists():
            assert not list(sources_azure.rglob("*")), "No files in sources/azure_blobs/"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_pipeline.py::TestAingestFromAzureBlob -v`
Expected: FAIL

**Step 3: Implement**

**Delete `_download_blob_to_storage_async()`** (lines 213-230).

**Add `_download_blob_to_temp()`**:
```python
async def _download_blob_to_temp(
    self,
    source: AsyncDataSource,
    blob_path: str,
    tmpdir: Path,
) -> Path:
    """Download Azure Blob to temp directory, auto-converting Excel to PDF."""
    target_path = tmpdir / Path(blob_path).name
    content = await source.aload_document(blob_path)
    await asyncio.to_thread(target_path.write_bytes, content)
    logger.info("Downloaded blob to temp: %s", target_path)
    return await self._prepare_for_parsing(target_path, tmpdir)
```

**Rewrite `aingest_from_azure_blob()`** (lines 522-733):

For single blob:
```python
source_uri = f"azure://{container_name}/{blob_path}"
tmpdir = self._create_temp_dir()
try:
    temp_file = await self._download_blob_to_temp(source, blob_path, tmpdir)
    # Dedup check after download
    should_skip, content_hash, reason = await self._hash_index.should_skip_file(temp_file, replace)
    if should_skip:
        return IngestionResult(status="success", processed=0, skipped=1, ...)
    result = await self._ingest_single_file_with_policy(
        temp_file, artifacts_dir, content_hash=content_hash, source_uri=source_uri
    )
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
```

For prefix batch: same pattern — each blob gets its own temp dir in the concurrent processing phase.

**Step 4: Run tests**

Run: `python -m pytest tests/unit/test_pipeline.py -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/dlightrag/ingestion/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: azure ingestion uses temp downloads with azure:// source URIs"
```

---

### Task 6: Deletion cleanup and stale code removal

Simplify `adelete_files()` — remove source file deletion (we don't own originals). Remove stale helper methods and `sources_dir` from config.

**Files:**
- Modify: `src/dlightrag/ingestion/pipeline.py` (deletion + helpers)
- Modify: `src/dlightrag/config.py:305-307`
- Modify: `src/dlightrag/ingestion/cleanup.py` (if references sources)
- Test: `tests/unit/test_pipeline.py`
- Test: `tests/unit/test_config.py`

**Step 1: Write the failing tests**

```python
class TestDeleteFilesNoSourceCleanup:

    @pytest.mark.asyncio
    async def test_delete_does_not_touch_original_file(self, test_config):
        """Deletion should NOT delete the user's original source file."""
        pipeline = _make_pipeline(test_config)
        # Create a file that simulates the "original"
        original = test_config.working_dir_path / "outside" / "report.pdf"
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_text("original content")

        # Mock deletion context to simulate a found doc_id referencing the original
        pipeline._hash_index = MagicMock()
        # ... setup mock to simulate a found doc_id

        # After deletion, original file should still exist
        assert original.exists()
```

Also verify `sources_dir` property is removed from config:
```python
def test_config_no_sources_dir(test_config):
    assert not hasattr(test_config, 'sources_dir')
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_pipeline.py::TestDeleteFilesNoSourceCleanup -v`

**Step 3: Implement**

**In `adelete_files()` (lines 862-992):**

Remove Phase 3b entirely (lines 948-984) — the block that deletes source files from `sources/` and artifacts using `_resolve_source_file` and `_extract_relative_source_path`.

Replace with simplified artifact cleanup that searches by filename stem:
```python
# Phase 3b: Delete artifacts (best-effort glob search)
if delete_source:
    artifacts_dir = self.config.artifacts_dir
    for fp in ctx.file_paths:
        stem = Path(fp).stem
        for match in artifacts_dir.rglob(f"{stem}*"):
            if match.is_dir():
                shutil.rmtree(match, ignore_errors=True)
                logger.info(f"Deleted artifacts directory: {match}")
```

Note: The `delete_source` parameter is kept for backward compat but its meaning changes from "delete the source copy" to "delete artifacts". Consider renaming to `delete_artifacts` in a future PR.

**Delete stale helpers:**
- `_get_source_dir()` (lines 139-145)
- `_extract_relative_source_path()` (lines 232-241)
- `_resolve_source_file()` (lines 243-263)

**In `config.py`:**
- Remove `sources_dir` property (lines 306-307)

**Update tests:**
- Remove tests for `_acopy_to_sources_local`, `_extract_relative_source_path`, `_resolve_source_file` from `TestIngestionPipelineHelpers`
- Remove `test_config.sources_dir` references from remaining tests
- Update `test_config.py` to not assert `sources_dir`

**Step 4: Run tests**

Run: `python -m pytest tests/unit/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/dlightrag/ingestion/pipeline.py src/dlightrag/config.py src/dlightrag/ingestion/cleanup.py tests/unit/
git commit -m "refactor: remove sources/ dir, simplify deletion, drop stale helpers"
```

---

### Post-Implementation Notes

**Migration:** Existing `sources/` directories are NOT auto-deleted. Users can run `rm -rf {working_dir}/sources/` manually after confirming no issues.

**Backward compat:** Existing hash_index entries with `sources/` paths still work for:
- Dedup (hash-based, path is metadata)
- `list_all()` (legacy path detection via `derive_source_type`)
- `find_by_path/name` (string matching, works regardless)

**Known limitation:** Artifact cleanup in deletion uses glob search by filename stem, which may match unrelated files with similar names. This is a pre-existing limitation (the old code also had issues with RAGAnything's `{stem}_{hash}` directory naming). Can be improved in a follow-up by storing artifact paths in hash_index.
