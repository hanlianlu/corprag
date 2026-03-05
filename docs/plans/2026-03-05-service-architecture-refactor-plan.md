# Service Architecture Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor RAGService layer: single RAGAnything instance, composition-based RetrievalEngine, RAGServiceManager coordinator, `core/` package structure.

**Architecture:** Create `core/` package containing servicemanager (multi-workspace pool + routing), service (single workspace lifecycle), federation (parallel query orchestration), retrieval engine (composition over inheritance), and ingestion pipeline. API/MCP servers simplified to depend only on RAGServiceManager.

**Tech Stack:** Python 3.12, pytest, pytest-asyncio, FastAPI, MCP SDK, RAGAnything, LightRAG

---

### Task 1: Create core/ package scaffold and RetrievalEngine

Create the `core/` package structure and implement `RetrievalEngine` using composition instead of `EnhancedRAGAnything` inheritance.

**Files:**
- Create: `src/dlightrag/core/__init__.py`
- Create: `src/dlightrag/core/retrieval/__init__.py`
- Create: `src/dlightrag/core/retrieval/engine.py`
- Test: `tests/unit/test_retrieval_engine_v2.py`

**Step 1: Write failing tests for RetrievalEngine**

```python
# tests/unit/test_retrieval_engine_v2.py
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for composition-based RetrievalEngine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.core.retrieval.engine import RetrievalEngine, RetrievalResult


class TestRetrievalEngineAretrieve:
    """Test data-only retrieval via composition."""

    def _make_engine(self, config=None) -> tuple[RetrievalEngine, MagicMock]:
        from dlightrag.config import DlightragConfig

        cfg = config or DlightragConfig(openai_api_key="test")  # type: ignore[call-arg]
        mock_rag = MagicMock()
        mock_rag.lightrag = MagicMock()
        mock_rag.lightrag.aquery_data = AsyncMock(
            return_value={"data": {"chunks": [{"id": "c1"}], "entities": [], "relationships": []}}
        )
        mock_rag._process_multimodal_query_content = AsyncMock(return_value="enhanced query")
        engine = RetrievalEngine(rag=mock_rag, config=cfg)
        return engine, mock_rag

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aretrieve_calls_lightrag_aquery_data(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, mock_rag = self._make_engine()
        result = await engine.aretrieve("test query")
        mock_rag.lightrag.aquery_data.assert_awaited_once()
        assert isinstance(result, RetrievalResult)

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aretrieve_with_multimodal_enhances_query(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, mock_rag = self._make_engine()
        await engine.aretrieve("query", multimodal_content=[{"type": "image"}])
        mock_rag._process_multimodal_query_content.assert_awaited_once()

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aretrieve_without_multimodal_no_enhancement(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, mock_rag = self._make_engine()
        await engine.aretrieve("query")
        mock_rag._process_multimodal_query_content.assert_not_awaited()

    async def test_aretrieve_no_lightrag_returns_empty(self) -> None:
        from dlightrag.config import DlightragConfig

        cfg = DlightragConfig(openai_api_key="test")  # type: ignore[call-arg]
        mock_rag = MagicMock()
        mock_rag.lightrag = None
        engine = RetrievalEngine(rag=mock_rag, config=cfg)
        result = await engine.aretrieve("query")
        assert result.answer is None
        assert result.contexts == {}


class TestRetrievalEngineAanswer:
    """Test LLM answer retrieval."""

    def _make_engine(self) -> tuple[RetrievalEngine, MagicMock]:
        from dlightrag.config import DlightragConfig

        cfg = DlightragConfig(openai_api_key="test")  # type: ignore[call-arg]
        mock_rag = MagicMock()
        mock_rag.lightrag = MagicMock()
        mock_rag.lightrag.aquery_llm = AsyncMock(
            return_value={
                "llm_response": {"content": "The answer is 42"},
                "data": {"chunks": [{"id": "c1"}], "entities": [], "relationships": []},
            }
        )
        mock_rag._process_multimodal_query_content = AsyncMock(return_value="enhanced")
        engine = RetrievalEngine(rag=mock_rag, config=cfg)
        return engine, mock_rag

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aanswer_returns_answer(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, _ = self._make_engine()
        result = await engine.aanswer("query")
        assert result.answer == "The answer is 42"

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aanswer_calls_aquery_llm(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, mock_rag = self._make_engine()
        await engine.aanswer("query")
        mock_rag.lightrag.aquery_llm.assert_awaited_once()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_retrieval_engine_v2.py -v`
Expected: FAIL (module not found)

**Step 3: Create core/ package and RetrievalEngine**

```python
# src/dlightrag/core/__init__.py
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Core business logic for DlightRAG."""
```

```python
# src/dlightrag/core/retrieval/__init__.py
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval engine for RAG queries."""

from dlightrag.core.retrieval.engine import RetrievalEngine, RetrievalResult

__all__ = ["RetrievalEngine", "RetrievalResult"]
```

```python
# src/dlightrag/core/retrieval/engine.py
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval engine — composition-based wrapper over RAGAnything.

Provides structured retrieval (RetrievalResult) by composing a RAGAnything
instance rather than inheriting from it. Calls LightRAG's aquery_data/aquery_llm
for structured data and RAGAnything's multimodal query enhancement.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from lightrag import QueryParam

from dlightrag.utils.content_filters import filter_content_for_snippet

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Wrapper for RAG query results."""

    answer: str | None = field(default=None)
    contexts: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class RetrievalEngine:
    """Structured retrieval over a RAGAnything instance (composition, not inheritance).

    Uses:
    - rag.lightrag.aquery_data() for data-only retrieval
    - rag.lightrag.aquery_llm() for retrieval + LLM answer
    - rag._process_multimodal_query_content() for multimodal query enhancement
    """

    def __init__(self, rag: Any, config: Any) -> None:
        self.rag = rag
        self.config = config

    async def aretrieve(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        *,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] | None = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        is_reretrieve: bool = False,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve structured data without generating answer."""
        lightrag = getattr(self.rag, "lightrag", None)
        if lightrag is None:
            logger.warning("LightRAG not initialized - no documents ingested yet")
            return RetrievalResult(
                answer=None, contexts={}, raw={"warning": "No documents ingested yet"}
            )

        enhanced_query = query
        if multimodal_content and hasattr(self.rag, "_process_multimodal_query_content"):
            enhanced_query = await self.rag._process_multimodal_query_content(
                query, multimodal_content
            )

        adjusted_top_k = top_k or self.config.top_k
        adjusted_chunk_top_k = chunk_top_k or self.config.chunk_top_k

        if is_reretrieve:
            enable_rerank = False
        else:
            enable_rerank = kwargs.pop("enable_rerank", self.config.enable_rerank)

        query_kwargs = {
            "top_k": adjusted_top_k,
            "chunk_top_k": adjusted_chunk_top_k,
            "enable_rerank": enable_rerank,
            "max_entity_tokens": kwargs.pop("max_entity_tokens", self.config.max_entity_tokens),
            "max_relation_tokens": kwargs.pop("max_relation_tokens", self.config.max_relation_tokens),
            "max_total_tokens": kwargs.pop("max_total_tokens", self.config.max_total_tokens),
            **kwargs,
        }

        query_param = QueryParam(mode=mode or self.config.default_mode, **query_kwargs)
        retrieval_data = await lightrag.aquery_data(enhanced_query, param=query_param)

        contexts = retrieval_data.get("data", {})
        raw: dict[str, Any] = retrieval_data if isinstance(retrieval_data, dict) else {}

        result = RetrievalResult(answer=None, contexts=contexts, raw=raw)

        return await augment_retrieval_result(
            result,
            str(self.config.working_dir_path),
            url_transformer=getattr(self, "_url_transformer", None),
            lightrag=lightrag,
        )

    async def aanswer(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        *,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] | None = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve contexts and generate an LLM answer."""
        lightrag = getattr(self.rag, "lightrag", None)
        if lightrag is None:
            logger.warning("LightRAG not initialized - no documents ingested yet")
            return RetrievalResult(
                answer=None, contexts={}, raw={"warning": "No documents ingested yet"}
            )

        enhanced_query = query
        if multimodal_content and hasattr(self.rag, "_process_multimodal_query_content"):
            enhanced_query = await self.rag._process_multimodal_query_content(
                query, multimodal_content
            )

        adjusted_top_k = top_k or self.config.top_k
        adjusted_chunk_top_k = chunk_top_k or self.config.chunk_top_k
        enable_rerank = kwargs.pop("enable_rerank", self.config.enable_rerank)

        # Truncate conversation history
        history = kwargs.pop("conversation_history", None)
        if history:
            max_msgs = self.config.max_conversation_turns * 2
            if len(history) > max_msgs:
                history = history[-max_msgs:]
            token_budget = self.config.max_conversation_tokens
            total = 0
            cutoff = 0
            for i in range(len(history) - 1, -1, -1):
                total += len(history[i].get("content", "")) // 4
                if total > token_budget:
                    cutoff = i + 1
                    break
            if cutoff:
                history = history[cutoff:]
            kwargs["conversation_history"] = history

        query_kwargs = {
            "top_k": adjusted_top_k,
            "chunk_top_k": adjusted_chunk_top_k,
            "enable_rerank": enable_rerank,
            "max_entity_tokens": kwargs.pop("max_entity_tokens", self.config.max_entity_tokens),
            "max_relation_tokens": kwargs.pop("max_relation_tokens", self.config.max_relation_tokens),
            "max_total_tokens": kwargs.pop("max_total_tokens", self.config.max_total_tokens),
            **kwargs,
        }

        query_param = QueryParam(mode=mode or self.config.default_mode, **query_kwargs)
        result_data = await lightrag.aquery_llm(enhanced_query, param=query_param)

        llm_response = result_data.get("llm_response", {})
        answer = llm_response.get("content")
        contexts = result_data.get("data", {})
        raw: dict[str, Any] = result_data if isinstance(result_data, dict) else {}

        result = RetrievalResult(answer=answer, contexts=contexts, raw=raw)

        return await augment_retrieval_result(
            result,
            str(self.config.working_dir_path),
            url_transformer=getattr(self, "_url_transformer", None),
            lightrag=lightrag,
        )


# --- Keep augmentation utilities (moved from old engine.py) ---

import hashlib
import re
from pathlib import Path


def _extract_rag_relative(file_path: str, working_dir: str | None = None) -> str | None:
    if working_dir:
        wd = working_dir.rstrip("/")
        idx = file_path.find(wd)
        if idx != -1:
            return file_path[idx + len(wd) :].lstrip("/")
    for marker in ("sources/", "artifacts/"):
        idx = file_path.find(marker)
        if idx != -1:
            return file_path[idx:]
    return None


def _to_download_url(
    path: str,
    url_transformer: Callable[[str], str] | None = None,
    working_dir: str | None = None,
) -> str:
    if url_transformer:
        return url_transformer(path)
    inner_path = path
    if inner_path.startswith("file://"):
        inner_path = inner_path[7:]
    if path.startswith("azure://"):
        return path
    relative = _extract_rag_relative(inner_path, working_dir)
    return f"file://{relative}" if relative else f"file://{inner_path}"


def build_sources_and_media_from_contexts(
    contexts: list[dict[str, Any]],
    url_transformer: Callable[[str], str] | None = None,
    rag_working_dir: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources: dict[str, dict[str, Any]] = {}
    media: dict[str, dict[str, Any]] = {}
    for ctx in contexts:
        if "chunk_id" not in ctx:
            continue
        file_path = ctx.get("file_path")
        ref_id = ctx.get("reference_id")
        chunk_id = ctx.get("chunk_id")
        if not file_path or not ref_id:
            continue
        ref_id = str(ref_id)
        if ref_id not in sources:
            content = ctx.get("content") or ""
            snippet = filter_content_for_snippet(content, max_chars=100)
            sources[ref_id] = {
                "id": ref_id, "type": "file",
                "title": Path(str(file_path)).name,
                "path": str(file_path),
                "url": _to_download_url(str(file_path), url_transformer=url_transformer, working_dir=rag_working_dir),
                "snippet": snippet, "chunk_ids": [],
            }
        if chunk_id and chunk_id not in sources[ref_id]["chunk_ids"]:
            sources[ref_id]["chunk_ids"].append(chunk_id)
        content = ctx.get("content", "")
        match_img = re.search(r"Image Path:\s*(.*)", content)
        if match_img:
            img_path = match_img.group(1).strip()
            match_caption = re.search(r"[Cc]aption:\s*(.*)", content)
            media_key = hashlib.md5(img_path.encode()).hexdigest()[:12]
            media[media_key] = {
                "id": media_key, "type": "image",
                "reference_id": ref_id, "source_chunk_id": chunk_id,
                "title": Path(str(file_path)).name,
                "path": img_path,
                "url": _to_download_url(img_path, url_transformer=url_transformer, working_dir=rag_working_dir),
                "caption": match_caption.group(1).strip() if match_caption is not None else None,
            }
    return list(sources.values()), list(media.values())


async def augment_retrieval_result(
    result: RetrievalResult,
    rag_working_dir: str | None = None,
    url_transformer: Callable[[str], str] | None = None,
    lightrag: Any = None,
) -> RetrievalResult:
    chunk_contexts = result.contexts.get("chunks", []) if result.contexts else []
    if lightrag and hasattr(lightrag, "text_chunks") and chunk_contexts:
        chunk_ids = [ctx.get("chunk_id") for ctx in chunk_contexts if ctx.get("chunk_id")]
        if chunk_ids:
            try:
                chunk_data_list = await lightrag.text_chunks.get_by_ids(chunk_ids)
                page_idx_map: dict[str, int | None] = {}
                for cid, cdata in zip(chunk_ids, chunk_data_list, strict=True):
                    if cdata and cdata.get("page_idx") is not None:
                        page_idx_map[cid] = cdata["page_idx"] + 1
                for ctx in chunk_contexts:
                    cid = ctx.get("chunk_id")
                    if cid and cid in page_idx_map:
                        ctx["page_idx"] = page_idx_map[cid]
            except Exception as e:
                logger.warning(f"Failed to load page_idx from KV store: {e}")
    sources, media = build_sources_and_media_from_contexts(
        chunk_contexts, url_transformer=url_transformer, rag_working_dir=rag_working_dir,
    )
    if sources or media:
        raw = dict(result.raw or {})
        if sources:
            raw["sources"] = sources
        if media:
            raw["media"] = media
        result.raw = raw
    return result
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_retrieval_engine_v2.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/dlightrag/core/ tests/unit/test_retrieval_engine_v2.py
git commit -m "feat: add core/ package with composition-based RetrievalEngine"
```

---

### Task 2: Move federation.py to core/

Move federation logic from `retrieval/federation.py` to `core/federation.py`. Update imports.

**Files:**
- Create: `src/dlightrag/core/federation.py` (copy from `src/dlightrag/retrieval/federation.py`)
- Modify: `src/dlightrag/core/federation.py` — update import of RetrievalResult to `core.retrieval.engine`
- Test: `tests/unit/test_federation.py` — update imports

**Step 1: Copy federation.py to core/, update RetrievalResult import**

In `src/dlightrag/core/federation.py`, change line 16:
```python
# Before:
from dlightrag.retrieval.engine import RetrievalResult
# After:
from dlightrag.core.retrieval.engine import RetrievalResult
```

**Step 2: Update test_federation.py imports**

```python
# Before:
from dlightrag.retrieval.engine import RetrievalResult
from dlightrag.retrieval.federation import federated_answer, federated_retrieve, merge_results

# After:
from dlightrag.core.retrieval.engine import RetrievalResult
from dlightrag.core.federation import federated_answer, federated_retrieve, merge_results
```

**Step 3: Run tests**

Run: `uv run pytest tests/unit/test_federation.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/dlightrag/core/federation.py tests/unit/test_federation.py
git commit -m "refactor: move federation.py to core/"
```

---

### Task 3: Move ingestion/ to core/ingestion/

Move ingestion package into `core/`.

**Files:**
- Create: `src/dlightrag/core/ingestion/` (copy from `src/dlightrag/ingestion/`)
- Modify: internal imports if any reference `dlightrag.ingestion` → `dlightrag.core.ingestion`
- Test: `tests/unit/test_pipeline.py`, `tests/unit/test_hash_index.py`, `tests/unit/test_ingestion_policy.py`

**Step 1: Copy ingestion/ to core/ingestion/**

Copy all files. Update any internal imports within `core/ingestion/` that reference `dlightrag.ingestion` to `dlightrag.core.ingestion`.

**Step 2: Update test imports**

In `test_pipeline.py`, `test_hash_index.py`, `test_ingestion_policy.py`: change `from dlightrag.ingestion` to `from dlightrag.core.ingestion`.

**Step 3: Run tests**

Run: `uv run pytest tests/unit/test_pipeline.py tests/unit/test_hash_index.py tests/unit/test_ingestion_policy.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/dlightrag/core/ingestion/ tests/unit/
git commit -m "refactor: move ingestion/ to core/ingestion/"
```

---

### Task 4: Refactor RAGService into core/service.py

Slim down RAGService: single RAGAnything instance, compose IngestionEngine + RetrievalEngine.

**Files:**
- Create: `src/dlightrag/core/service.py`
- Test: `tests/unit/test_service.py` — update to test new structure

**Step 1: Update test_service.py for new structure**

Key changes in tests:
- Import from `dlightrag.core.service`
- Replace `service.rag_text` / `service.rag_vision` references with `service.retrieval`
- Replace `rag_text.aquery_data_with_multimodal` with `service.retrieval.aretrieve`

**Step 2: Write core/service.py**

Key changes from current `service.py`:
- Create ONE `RAGAnything` instance with `llm_model_func` (chat model, unified)
- Create `IngestionPipeline` composing that RAGAnything
- Create `RetrievalEngine` composing that RAGAnything
- Remove `rag_text` / `rag_vision` split
- Pass `url_transformer` to `RetrievalEngine` instead of storing on service
- Keep PG lock, schema init, hash index logic

```python
# Key structural change in _do_initialize():
# ONE RAGAnything instance
self.rag = RAGAnything(
    None,           # lightrag created internally
    llm_func,       # unified LLM (chat_model)
    vision_func,
    embedding_func,
    rag_config,
    lightrag_kwargs,
)

# Compose pipelines
self.ingestion = IngestionPipeline(self.rag, config=config, ...)
self.retrieval = RetrievalEngine(rag=self.rag, config=config)
self.retrieval._url_transformer = self._url_transformer
```

**Step 3: Run tests**

Run: `uv run pytest tests/unit/test_service.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/dlightrag/core/service.py tests/unit/test_service.py
git commit -m "refactor: slim RAGService with single RAGAnything + composition"
```

---

### Task 5: Create RAGServiceManager (core/servicemanager.py)

Absorb pool.py workspace management + federation routing.

**Files:**
- Create: `src/dlightrag/core/servicemanager.py`
- Rename: `tests/unit/test_pool.py` → `tests/unit/test_servicemanager.py`

**Step 1: Write test_servicemanager.py**

Adapt existing `test_pool.py` tests:
- `TestWorkspacePool` → `TestRAGServiceManager`
- Replace `get_workspace_service` calls with `manager._get_service()`
- Add tests for `manager.aretrieve()` routing (single ws → direct, multi ws → federation)
- Add test for `manager.aanswer()` routing
- Keep shared service tests (backoff, singleton) adapted for manager

**Step 2: Write core/servicemanager.py**

```python
class RAGServiceManager:
    """Multi-workspace RAG coordinator."""

    def __init__(self, config: DlightragConfig | None = None):
        from dlightrag.config import get_config
        self._config = config or get_config()
        self._services: dict[str, RAGService] = {}
        self._lock: asyncio.Lock | None = None

    @classmethod
    async def create(cls, config=None) -> RAGServiceManager:
        return cls(config=config)

    async def _get_service(self, workspace: str) -> RAGService:
        # Async-safe lazy creation with lock (from pool.py logic)
        ...

    async def aretrieve(self, query, *, workspace=None, workspaces=None, **kwargs):
        ws_list = workspaces or [workspace or self._config.workspace]
        if len(ws_list) == 1:
            svc = await self._get_service(ws_list[0])
            return await svc.aretrieve(query, **kwargs)
        return await federated_retrieve(query, ws_list, self._get_service, **kwargs)

    async def aanswer(self, query, *, workspace=None, workspaces=None, **kwargs):
        ws_list = workspaces or [workspace or self._config.workspace]
        if len(ws_list) == 1:
            svc = await self._get_service(ws_list[0])
            return await svc.aanswer(query, **kwargs)
        return await federated_answer(query, ws_list, self._get_service, **kwargs)

    async def aingest(self, workspace, source_type, **kwargs):
        svc = await self._get_service(workspace)
        return await svc.aingest(source_type=source_type, **kwargs)

    async def list_workspaces(self) -> list[str]:
        # workspace discovery logic from pool.py
        ...

    async def list_ingested_files(self, workspace: str) -> list[dict]:
        svc = await self._get_service(workspace)
        return await svc.alist_ingested_files()

    async def delete_files(self, workspace: str, **kwargs) -> list[dict]:
        svc = await self._get_service(workspace)
        return await svc.adelete_files(**kwargs)

    async def close(self):
        for svc in self._services.values():
            try: await svc.close()
            except Exception: pass
        self._services.clear()
```

**Step 3: Run tests**

Run: `uv run pytest tests/unit/test_servicemanager.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/dlightrag/core/servicemanager.py tests/unit/test_servicemanager.py
git commit -m "feat: add RAGServiceManager as unified multi-workspace coordinator"
```

---

### Task 6: Update API server to use RAGServiceManager

**Files:**
- Modify: `src/dlightrag/api/server.py`
- Modify: `tests/unit/test_api_server.py`

**Step 1: Update api/server.py**

Key changes:
- Replace imports: remove `pool`, `federation` imports, add `RAGServiceManager`
- Create manager in lifespan, store in `app.state`
- All endpoints delegate to manager methods
- Remove `_resolve_workspace` helper (manager handles defaults)

```python
# New imports
from dlightrag.core.servicemanager import RAGServiceManager, RAGServiceUnavailableError

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.manager = await RAGServiceManager.create()
    yield
    await app.state.manager.close()

# Endpoints simplified
@app.post("/retrieve")
async def retrieve(body: RetrieveRequest):
    manager: RAGServiceManager = app.state.manager
    result = await manager.aretrieve(
        query=body.query, workspaces=body.workspaces,
        mode=body.mode, top_k=body.top_k, chunk_top_k=body.chunk_top_k,
    )
    return {"answer": result.answer, "contexts": result.contexts, "raw": result.raw}
```

**Step 2: Update test_api_server.py**

- Patch `RAGServiceManager` instead of `get_workspace_service`
- Mock `app.state.manager` with a mock manager

**Step 3: Run tests**

Run: `uv run pytest tests/unit/test_api_server.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/dlightrag/api/server.py tests/unit/test_api_server.py
git commit -m "refactor: API server uses RAGServiceManager"
```

---

### Task 7: Update MCP server to use RAGServiceManager

**Files:**
- Modify: `src/dlightrag/mcp/server.py`

**Step 1: Update mcp/server.py**

Same pattern as API server:
- Replace `pool`/`federation` imports with `RAGServiceManager`
- Store manager as module-level singleton (MCP server lifecycle differs from FastAPI)
- All tool handlers delegate to manager

**Step 2: Run full test suite**

Run: `uv run pytest tests/unit/ -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/dlightrag/mcp/server.py
git commit -m "refactor: MCP server uses RAGServiceManager"
```

---

### Task 8: Update scripts (cli.py, reset.py)

**Files:**
- Modify: `scripts/cli.py`
- Modify: `scripts/reset.py`
- Modify: `tests/unit/test_cli.py`

**Step 1: Update cli.py**

- `_run_ingest()`: use `RAGServiceManager` instead of `RAGService.create()`
- Import from `dlightrag.core.servicemanager`

**Step 2: Update reset.py**

- Import `RAGService` from `dlightrag.core.service`
- Adjust `_STORAGE_ATTRS` access if needed

**Step 3: Run tests**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add scripts/ tests/unit/test_cli.py
git commit -m "refactor: scripts use core/ imports"
```

---

### Task 9: Update __init__.py, delete old files, clean up

**Files:**
- Modify: `src/dlightrag/__init__.py` — update lazy imports
- Modify: `src/dlightrag/core/__init__.py` — add public re-exports
- Delete: `src/dlightrag/pool.py`
- Delete: `src/dlightrag/service.py`
- Delete: `src/dlightrag/retrieval/` (old package)
- Delete: `src/dlightrag/ingestion/` (old package)
- Delete: `tests/unit/test_pool.py` (replaced by test_servicemanager.py)
- Modify: `tests/unit/test_retrieval_engine.py` — update imports to core/

**Step 1: Update __init__.py lazy imports**

```python
def _lazy_imports():
    from dlightrag.core.retrieval.engine import RetrievalResult
    from dlightrag.core.service import RAGService
    from dlightrag.core.servicemanager import RAGServiceManager
    return RAGService, RAGServiceManager, RetrievalResult

def __getattr__(name: str):
    if name in ("RAGService", "RAGServiceManager", "RetrievalResult"):
        RAGService, RAGServiceManager, RetrievalResult = _lazy_imports()
        return {"RAGService": RAGService, "RAGServiceManager": RAGServiceManager, "RetrievalResult": RetrievalResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Step 2: Update test_retrieval_engine.py imports**

Change `from dlightrag.retrieval.engine import ...` to `from dlightrag.core.retrieval.engine import ...`

**Step 3: Delete old files**

```bash
rm src/dlightrag/pool.py
rm src/dlightrag/service.py
rm -rf src/dlightrag/retrieval/
rm -rf src/dlightrag/ingestion/
rm tests/unit/test_pool.py
```

**Step 4: Run full test suite**

Run: `uv run pytest tests/unit/ -v`
Expected: PASS

**Step 5: Run linter**

Run: `uv run ruff check src/ tests/ scripts/`
Expected: No errors

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove old modules, complete core/ migration"
```

---

### Task 10: Update documentation and config

**Files:**
- Modify: `README.md` — update architecture diagram, import examples
- Modify: `.env.example` — add TODO comment on DLIGHTRAG_INGESTION_MODEL

**Step 1: Update .env.example**

Add comment to ingestion model line:
```bash
# TODO: ingestion_model currently shares LightRAG's llm_model_func with chat_model.
# Independent ingestion LLM awaits LightRAG per-call override support.
# DLIGHTRAG_INGESTION_MODEL=gpt-4.1-mini
```

**Step 2: Update README.md architecture section**

Update the architecture diagram to reflect:
- `RAGServiceManager` as the entry point
- `core/` package structure
- Composition-based `RetrievalEngine`

**Step 3: Commit**

```bash
git add README.md .env.example
git commit -m "docs: update architecture docs for core/ refactor"
```

---

### Task 11: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/unit/ -v --tb=short`
Expected: ALL PASS

**Step 2: Run linter + formatter**

Run: `uv run ruff check src/ tests/ scripts/ && uv run ruff format --check src/ tests/ scripts/`
Expected: Clean

**Step 3: Verify imports work**

Run: `uv run python -c "from dlightrag import RAGService, RAGServiceManager, RetrievalResult; print('OK')"`
Expected: `OK`
