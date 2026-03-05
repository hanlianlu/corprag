# Test Cleanup & Supplement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove ~25 low-value/fragile tests, add ~20 high-value missing tests targeting the biggest coverage gaps.

**Architecture:** Two phases — Phase 1 deletes low-value tests (net negative lines), Phase 2 adds high-value tests focused on hash_index, pipeline, and service layers. Each task is one test file. No production code changes.

**Tech Stack:** pytest, pytest-asyncio, unittest.mock

**Run command:** `uv run python -m pytest tests/unit/ -v`

---

### Task 1: Clean up test_hash_index.py — remove low-value tests

**Files:**
- Modify: `tests/unit/test_hash_index.py`

**What to remove:**

1. `TestComputeFileHash.test_different_content_different_hash` — tests SHA256 collision resistance, not app logic
2. All `len(result) == 71` assertions (3 occurrences) — magic number; `startswith("sha256:")` already validates format
3. `TestContentAwareOfficeHash.test_non_office_extension_uses_byte_hash` — couples to private `_hash_file_bytes`, trivially obvious from suffix dispatch
4. `TestHashIndex.test_lookup_missing` — tests `dict.get()` returns None
5. `TestHashIndex.test_remove_missing` — tests deleting non-existent key returns False
6. `TestHashIndexWorkspace.test_default_workspace_uses_subdirectory` — duplicate of `test_workspace_creates_subdirectory` with different string
7. `TestHashIndexFactory` — all 4 tests. They use `RAGService.__new__()`, accept any result (`in ("PGHashIndex", "HashIndex")`), and can never fail. Move meaningful factory testing to test_service.py Task 5.

**Step 1: Remove the tests listed above**

Delete entire `test_different_content_different_hash` method.
Delete entire `test_non_office_extension_uses_byte_hash` method (and its `from dlightrag.ingestion.hash_index import _hash_file_bytes` import).
Delete entire `test_lookup_missing` method.
Delete entire `test_remove_missing` method.
Delete entire `test_default_workspace_uses_subdirectory` method.
Delete entire `TestHashIndexFactory` class (all 4 tests).
Remove `len(result) == 71` lines from `test_computes_sha256`, `test_pdf_no_text_falls_back_to_byte_hash`, `test_corrupted_zip_falls_back_to_byte_hash`.

**Step 2: Run tests**

```bash
uv run python -m pytest tests/unit/test_hash_index.py -v
```

Expected: All remaining tests pass. Count should drop by ~10 tests.

**Step 3: Commit**

```bash
git add tests/unit/test_hash_index.py
git commit -m "test: remove low-value hash_index tests"
```

---

### Task 2: Add high-value hash_index tests

**Files:**
- Modify: `tests/unit/test_hash_index.py`

**Tests to add (all in `TestHashIndex` class unless noted):**

```python
# --- In TestHashIndex ---

async def test_should_skip_file_replace_bypasses_dedup(self, tmp_path: Path) -> None:
    """replace=True returns (False, hash, None) even for known duplicate."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("content")
    index = HashIndex(tmp_path)
    content_hash = compute_file_hash(test_file)
    await index.register(content_hash, "doc-001", str(test_file))

    should_skip, returned_hash, reason = await index.should_skip_file(test_file, replace=True)
    assert not should_skip
    assert returned_hash == content_hash
    assert reason is None

async def test_find_by_name_found(self, tmp_path: Path) -> None:
    """find_by_name returns (doc_id, hash, path) for matching filename."""
    index = HashIndex(tmp_path)
    await index.register("sha256:abc", "doc-001", "/deep/path/report.pdf")
    doc_id, h, path = index.find_by_name("report.pdf")
    assert doc_id == "doc-001"
    assert h == "sha256:abc"
    assert path == "/deep/path/report.pdf"

async def test_find_by_name_not_found(self, tmp_path: Path) -> None:
    """find_by_name returns (None, None, None) when no match."""
    index = HashIndex(tmp_path)
    await index.register("sha256:abc", "doc-001", "/path/other.pdf")
    assert index.find_by_name("missing.pdf") == (None, None, None)

async def test_find_by_path_found(self, tmp_path: Path) -> None:
    """find_by_path matches exact path string."""
    index = HashIndex(tmp_path)
    await index.register("sha256:abc", "doc-001", "/exact/path/file.pdf")
    doc_id, h, path = index.find_by_path("/exact/path/file.pdf")
    assert doc_id == "doc-001"

async def test_register_overwrites_existing(self, tmp_path: Path) -> None:
    """Registering same hash with different doc_id overwrites."""
    index = HashIndex(tmp_path)
    await index.register("sha256:abc", "doc-001", "/path/a.pdf")
    await index.register("sha256:abc", "doc-002", "/path/b.pdf")
    entry = index.lookup("sha256:abc")
    assert entry["doc_id"] == "doc-002"
    assert entry["file_path"] == "/path/b.pdf"

async def test_persistence_across_instances(self, tmp_path: Path) -> None:
    """Data survives creating a new HashIndex on the same directory."""
    index1 = HashIndex(tmp_path)
    await index1.register("sha256:abc", "doc-001", "/path/a.pdf")

    index2 = HashIndex(tmp_path)
    entry = index2.lookup("sha256:abc")
    assert entry is not None
    assert entry["doc_id"] == "doc-001"

async def test_corrupted_json_recovers(self, tmp_path: Path) -> None:
    """Corrupted index file recovers gracefully to empty index."""
    index = HashIndex(tmp_path)
    # Write garbage to the index file
    index._get_index_path().parent.mkdir(parents=True, exist_ok=True)
    index._get_index_path().write_text("NOT VALID JSON {{{")
    index.invalidate()

    # Should not raise, returns None
    assert index.lookup("sha256:anything") is None
    # Should be able to register new entries
    await index.register("sha256:new", "doc-new", "/new.pdf")
    assert index.lookup("sha256:new") is not None

async def test_list_all_returns_correct_structure(self, tmp_path: Path) -> None:
    """list_all returns dicts with all expected fields."""
    index = HashIndex(tmp_path)
    await index.register("sha256:aaa", "doc-001", "/Users/me/report.pdf")
    entries = await index.list_all()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["file_path"] == "/Users/me/report.pdf"
    assert entry["doc_id"] == "doc-001"
    assert entry["source_type"] == "local"
    assert entry["file_name"] == "report.pdf"
    assert entry["content_hash"] == "sha256:aaa"
    assert "created_at" in entry
```

```python
# --- In TestDeriveSourceType, add: ---

def test_unknown_uri_scheme(self):
    from dlightrag.ingestion.hash_index import derive_source_type
    assert derive_source_type("s3://bucket/file.pdf") == "unknown"

def test_dot_relative_path(self):
    from dlightrag.ingestion.hash_index import derive_source_type
    assert derive_source_type("./data/report.pdf") == "local"
```

**Step 1: Add all tests above**

**Step 2: Run tests**

```bash
uv run python -m pytest tests/unit/test_hash_index.py -v
```

Expected: All pass.

**Step 3: Commit**

```bash
git add tests/unit/test_hash_index.py
git commit -m "test: add high-value hash_index tests (find, persistence, recovery, list_all structure)"
```

---

### Task 3: Clean up test_pipeline.py — fix fragile tests

**Files:**
- Modify: `tests/unit/test_pipeline.py`

**What to change:**

1. **Delete** `test_get_storage_dir_creates_path` — tests stdlib Path joining
2. **Delete** `test_create_temp_dir_under_working_dir` — tests uuid + mkdir
3. **Delete** `test_prepare_for_parsing_non_excel` — tautological with stub
4. **Delete** one of `test_no_checker_no_error` / `test_checker_returns_false` — keep `test_checker_returns_false` only
5. **Fix** `test_no_copy_to_sources_dir`, `test_temp_cleaned_up_after_ingestion`, `test_azure_temp_cleaned_up`, `test_azure_no_permanent_download` — remove conditional `if .exists():` guards, make assertions unconditional:

```python
# Replace: if sources_local.exists(): assert not list(...)
# With:    assert not sources_local.exists() or not list(sources_local.iterdir())

# Replace: if tmp_dir.exists(): assert not list(...)
# With:    assert not tmp_dir.exists() or not list(tmp_dir.iterdir())
```

6. **Fix** `test_azure_source_uri_format` — remove `if pipeline.rag.insert_content_list.called:` guard, make assertion unconditional:

```python
pipeline.rag.insert_content_list.assert_awaited_once()
call_kwargs = pipeline.rag.insert_content_list.call_args
file_path_arg = call_kwargs.kwargs.get("file_path", "")
assert file_path_arg == "azure://mycontainer/data/report.pdf"
```

**Step 1: Apply all changes**

**Step 2: Run tests**

```bash
uv run python -m pytest tests/unit/test_pipeline.py -v
```

Expected: All pass. Count drops by ~4 deleted tests.

**Step 3: Commit**

```bash
git add tests/unit/test_pipeline.py
git commit -m "test: remove low-value pipeline tests, fix fragile conditional assertions"
```

---

### Task 4: Add high-value pipeline tests

**Files:**
- Modify: `tests/unit/test_pipeline.py`

**Tests to add:**

```python
class TestAingestFromLocalEdgeCases:
    """Additional local ingestion edge cases."""

    async def test_replace_mode_bypasses_dedup(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """replace=True re-ingests even when hash exists."""
        pipeline = _make_pipeline(test_config)
        src = tmp_path / "doc.pdf"
        src.write_text("content")

        # First ingest
        await pipeline.aingest_from_local(src)
        # Second ingest with replace — should NOT skip
        result = await pipeline.aingest_from_local(src, replace=True)
        assert result.processed == 1
        assert result.skipped == 0

    async def test_source_uri_uses_azure_scheme(
        self, test_config: DlightragConfig
    ) -> None:
        """Azure blob_path ingestion records azure:// source_uri."""
        pipeline = _make_pipeline(test_config)
        mock_source = AsyncMock()
        mock_source.aload_document = AsyncMock(return_value=b"blob content")

        await pipeline.aingest_from_azure_blob(
            source=mock_source,
            container_name="mybucket",
            blob_path="folder/doc.pdf",
        )

        pipeline.rag.insert_content_list.assert_awaited_once()
        call_kwargs = pipeline.rag.insert_content_list.call_args
        assert call_kwargs.kwargs.get("file_path") == "azure://mybucket/folder/doc.pdf"

    async def test_azure_both_blob_path_and_prefix_raises(
        self, test_config: DlightragConfig
    ) -> None:
        """Providing both blob_path and prefix raises ValueError."""
        pipeline = _make_pipeline(test_config)
        mock_source = AsyncMock()

        with pytest.raises(ValueError, match="both"):
            await pipeline.aingest_from_azure_blob(
                source=mock_source,
                container_name="c",
                blob_path="a.pdf",
                prefix="data/",
            )

    async def test_directory_partial_failure(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """One file failing does not prevent others from processing."""
        pipeline = _make_pipeline(test_config)
        d = tmp_path / "docs"
        d.mkdir()
        (d / "good.pdf").write_text("ok")
        (d / "bad.pdf").write_text("fail")

        call_count = 0
        original_parse = pipeline.rag.parse_document

        async def maybe_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            file_path = str(kwargs.get("file_path", args[0] if args else ""))
            if "bad" in file_path:
                raise RuntimeError("parse failed")
            return await original_parse(*args, **kwargs)

        pipeline.rag.parse_document = AsyncMock(side_effect=maybe_fail)
        result = await pipeline.aingest_from_local(d)

        assert result.total_files == 2
        # At least one succeeded
        assert result.processed >= 1
```

**Step 1: Add all tests above**

Import `AsyncMock` is already at the top. If `test_config` fixture is used (from conftest), that's fine.

**Step 2: Run tests**

```bash
uv run python -m pytest tests/unit/test_pipeline.py -v
```

Expected: All pass.

**Step 3: Commit**

```bash
git add tests/unit/test_pipeline.py
git commit -m "test: add pipeline tests for replace mode, azure URI, partial failure"
```

---

### Task 5: Add high-value service tests (aretrieve, aanswer, close)

**Files:**
- Modify: `tests/unit/test_service.py`

**What to remove first:**
1. `test_ensure_initialized_raises` — redundant with `test_aingest_not_initialized_raises`
2. `test_mineru_backend_manual_override` — tests identity function
3. `test_rerank_empty_chunks` — tests one-liner early return

**Tests to add:**

```python
class TestRAGServiceRetrieve:
    """Test aretrieve and aanswer dispatch logic."""

    def _make_retrieval_service(self, config: DlightragConfig) -> RAGService:
        service = RAGService(config=config)
        service._initialized = True
        service.enable_rerank = False

        rag_text = MagicMock()
        rag_text.aquery_data_with_multimodal = AsyncMock(return_value={
            "answer": "", "chunks": [], "entities": [], "relationships": [],
            "sources": [], "media": [],
        })
        rag_text.lightrag = MagicMock()

        rag_vision = MagicMock()
        rag_vision.aquery_data_with_multimodal = AsyncMock(return_value={
            "answer": "", "chunks": [], "entities": [], "relationships": [],
            "sources": [], "media": [],
        })
        rag_vision.lightrag = MagicMock()

        service.rag_text = rag_text
        service.rag_vision = rag_vision
        service.ingestion = MagicMock()
        return service

    @patch("dlightrag.service.augment_retrieval_result", new_callable=AsyncMock,
           return_value={"answer": "", "chunks": [], "sources": [], "media": []})
    async def test_aretrieve_uses_text_rag_by_default(self, mock_augment, test_config):
        service = self._make_retrieval_service(test_config)
        await service.aretrieve("test query")
        service.rag_text.aquery_data_with_multimodal.assert_awaited_once()
        service.rag_vision.aquery_data_with_multimodal.assert_not_awaited()

    @patch("dlightrag.service.augment_retrieval_result", new_callable=AsyncMock,
           return_value={"answer": "", "chunks": [], "sources": [], "media": []})
    async def test_aretrieve_uses_vision_rag_with_multimodal(self, mock_augment, test_config):
        service = self._make_retrieval_service(test_config)
        await service.aretrieve("test query", multimodal_content=[{"type": "image"}])
        service.rag_vision.aquery_data_with_multimodal.assert_awaited_once()
        service.rag_text.aquery_data_with_multimodal.assert_not_awaited()

    async def test_aretrieve_not_initialized_raises(self, test_config):
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.aretrieve("query")


class TestConversationHistoryTruncation:
    """Test aanswer conversation history truncation logic."""

    def _make_answer_service(self, config: DlightragConfig) -> RAGService:
        service = RAGService(config=config)
        service._initialized = True
        service.enable_rerank = False

        rag = MagicMock()
        rag.aquery_llm_with_multimodal = AsyncMock(return_value={
            "answer": "response", "chunks": [], "entities": [],
            "relationships": [], "sources": [], "media": [],
        })
        rag.lightrag = MagicMock()

        service.rag_text = rag
        service.rag_vision = rag
        service.ingestion = MagicMock()
        return service

    @patch("dlightrag.service.augment_retrieval_result", new_callable=AsyncMock,
           return_value={"answer": "ok", "chunks": [], "sources": [], "media": []})
    async def test_history_truncated_by_turns(self, mock_augment, test_config):
        """History exceeding max_conversation_turns*2 is truncated from front."""
        test_config.max_conversation_turns = 2  # max 4 messages
        service = self._make_answer_service(test_config)

        history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        await service.aanswer("query", conversation_history=history)

        call_kwargs = service.rag_text.aquery_llm_with_multimodal.call_args.kwargs
        passed_history = call_kwargs.get("conversation_history", [])
        assert len(passed_history) <= 4

    @patch("dlightrag.service.augment_retrieval_result", new_callable=AsyncMock,
           return_value={"answer": "ok", "chunks": [], "sources": [], "media": []})
    async def test_none_history_passes_through(self, mock_augment, test_config):
        """None history does not add conversation_history kwarg."""
        service = self._make_answer_service(test_config)
        await service.aanswer("query", conversation_history=None)

        call_kwargs = service.rag_text.aquery_llm_with_multimodal.call_args.kwargs
        assert "conversation_history" not in call_kwargs


class TestRAGServiceFileManagement:
    """Test alist_ingested_files and adelete_files delegation."""

    async def test_alist_not_initialized_raises(self, test_config):
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.alist_ingested_files()

    async def test_adelete_not_initialized_raises(self, test_config):
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.adelete_files(filenames=["a.pdf"])

    async def test_alist_delegates_to_ingestion(self, test_config):
        service = RAGService(config=test_config)
        service._initialized = True
        service.ingestion = MagicMock()
        service.ingestion.alist_ingested_files = AsyncMock(return_value=[{"doc_id": "d1"}])
        result = await service.alist_ingested_files()
        assert result == [{"doc_id": "d1"}]
        service.ingestion.alist_ingested_files.assert_awaited_once()

    async def test_adelete_delegates_to_ingestion(self, test_config):
        service = RAGService(config=test_config)
        service._initialized = True
        service.ingestion = MagicMock()
        service.ingestion.adelete_files = AsyncMock(return_value=[{"status": "deleted"}])
        result = await service.adelete_files(filenames=["a.pdf"])
        assert result == [{"status": "deleted"}]
        call_kwargs = service.ingestion.adelete_files.call_args.kwargs
        assert call_kwargs["filenames"] == ["a.pdf"]
```

**Step 1: Remove the 3 low-value tests, add all tests above**

**Step 2: Run tests**

```bash
uv run python -m pytest tests/unit/test_service.py -v
```

Expected: All pass.

**Step 3: Commit**

```bash
git add tests/unit/test_service.py
git commit -m "test: add aretrieve/aanswer dispatch, history truncation, file management tests"
```

---

### Task 6: Clean up remaining low-value tests across other files

**Files:**
- Modify: `tests/unit/test_llm_providers.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_api_server.py`
- Modify: `tests/unit/test_pool.py`
- Modify: `tests/unit/test_schemas.py`

**What to change:**

**test_llm_providers.py:**
- In all 7 `test_*_returns_partial` tests: remove the `func.func.__module__` assertions (tests upstream module structure). Keep `isinstance(func, partial)` and `func.keywords["model"]` assertions where present.

**test_config.py:**
- Delete `test_defaults` — asserts 12+ exact field values; any default change breaks it. The individual property tests + `test_env_override_*` tests provide sufficient coverage.
- Delete `test_temp_dir_property` — one-liner property.

**test_api_server.py:**
- Merge 4 `test_*_requires_auth` into a single parametrized test:

```python
@pytest.mark.parametrize("method,path", [
    ("POST", "/ingest"),
    ("POST", "/retrieve"),
    ("POST", "/answer"),
    ("DELETE", "/files"),
])
async def test_endpoint_requires_auth(self, method, path, client):
    """All protected endpoints return 401 without auth."""
    resp = await client.request(method, path, json={})
    assert resp.status_code == 401
```

Delete the 4 individual tests.

**test_pool.py:**
- Delete `test_is_initialized_true`, `test_is_initialized_false_when_not_ready`, `test_get_error_info`, `test_reset_shared_rag_service` — all test trivial global variable get/set.

**test_schemas.py:**
- Delete `test_valid` (RankedChunk) — pydantic construction guarantee.
- Delete `test_boundary_scores` — pydantic constraint testing.
- Delete `test_empty_chunks_allowed` — trivial.

**Step 1: Apply all changes**

**Step 2: Run full test suite**

```bash
uv run python -m pytest tests/unit/ -v
```

Expected: All pass. Net reduction of ~15-20 tests across these files.

**Step 3: Commit**

```bash
git add tests/unit/test_llm_providers.py tests/unit/test_config.py tests/unit/test_api_server.py tests/unit/test_pool.py tests/unit/test_schemas.py
git commit -m "test: clean up low-value tests across providers, config, api, pool, schemas"
```

---

## Summary

| Task | File | Removed | Added | Net |
|------|------|---------|-------|-----|
| 1 | test_hash_index.py | ~10 | 0 | -10 |
| 2 | test_hash_index.py | 0 | ~10 | +10 |
| 3 | test_pipeline.py | ~4 | 0 | -4 |
| 4 | test_pipeline.py | 0 | ~4 | +4 |
| 5 | test_service.py | ~3 | ~10 | +7 |
| 6 | multiple | ~18 | 1 | -17 |
| **Total** | | **~35** | **~25** | **-10** |

Net effect: fewer tests, better coverage. Removes ~35 low-value tests, adds ~25 high-value tests covering: find_by_name/path, persistence, corruption recovery, list_all structure, replace mode, azure URI, partial failure, aretrieve/aanswer dispatch, conversation truncation, file management delegation.
