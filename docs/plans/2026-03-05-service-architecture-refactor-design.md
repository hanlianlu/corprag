# RAGService Architecture Refactor Design

## Goal

Refactor the RAGService layer to eliminate redundant RAGAnything instances,
replace inheritance with composition, and provide a clean multi-workspace
entry point (RAGServiceManager) for API/MCP/CLI consumers.

## Problems with Current Design

1. **3 RAGAnything instances per workspace** — 1 for ingestion + 2 EnhancedRAGAnything
   for retrieval (text/vision), all sharing the same LightRAG. Redundant config,
   processors, parser on each instance.

2. **EnhancedRAGAnything inherits RAGAnything** — inherits full ingestion/batch/query
   capability but only uses 2 methods + `_process_multimodal_query_content`. Heavy
   inheritance to access internal state.

3. **federation.py misplaced in retrieval/** — it orchestrates RAGService instances
   (higher layer) but lives inside retrieval/ (lower layer).

4. **API/MCP must juggle pool + federation** — two separate imports, manual
   single-vs-multi routing in every handler.

5. **LLM role confusion** — ingestion_llm_func set on LightRAG but retrieval
   EnhancedRAGAnything gets a different llm_func that only affects
   `_process_multimodal_query_content`, not LightRAG's answer generation.

## Architecture

```
src/dlightrag/
├── config.py                          # Configuration (unchanged)
├── models/
│   ├── llm.py                         # LLM/embedding/vision/rerank factories (unchanged)
│   └── prompts.py                     # Custom VLM prompts (unchanged)
├── core/
│   ├── __init__.py
│   ├── servicemanager.py              # RAGServiceManager
│   ├── service.py                     # RAGService (slimmed)
│   ├── federation.py                  # Parallel query orchestration + merge
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pipeline.py                # IngestionEngine (mostly unchanged)
│   │   └── hash_index.py             # Hash index backends (unchanged)
│   └── retrieval/
│       ├── __init__.py
│       └── engine.py                  # RetrievalEngine (replaces EnhancedRAGAnything)
├── api/
│   └── server.py                      # REST API (simplified, depends on RAGServiceManager)
├── mcp/
│   └── server.py                      # MCP server (simplified, depends on RAGServiceManager)
├── sourcing/                          # Data sources (unchanged)
└── utils/                             # Utilities (unchanged)
```

### Layer Responsibilities

| Layer | Module | Responsibility | Does NOT |
|-------|--------|----------------|----------|
| Protocol | `api/`, `mcp/` | HTTP/MCP protocol translation | Business logic |
| Coordination | `core/servicemanager.py` | Workspace pool, routing (single vs multi) | Merge logic |
| Orchestration | `core/federation.py` | Parallel queries, round-robin merge | Instance lifecycle |
| Business | `core/service.py` | Single workspace lifecycle, PG init | Multi-workspace awareness |
| Engine | `core/retrieval/engine.py` | Structured query, source augmentation | Ingestion |
| Engine | `core/ingestion/pipeline.py` | Document parsing, KG insertion | Retrieval |

### Component Details

#### RAGServiceManager (`core/servicemanager.py`)

Multi-workspace coordinator. Single entry point for all consumers.

```python
class RAGServiceManager:
    _services: dict[str, RAGService]
    _config: DlightragConfig

    @classmethod
    async def create(cls, config=None) -> RAGServiceManager

    # Write operations (single workspace)
    async def aingest(self, workspace: str, source_type, **kwargs) -> dict
    async def list_ingested_files(self, workspace: str) -> list[dict]
    async def delete_files(self, workspace: str, **kwargs) -> list[dict]

    # Read operations (single or federated)
    async def aretrieve(self, query, *, workspace=None, workspaces=None, **kwargs) -> RetrievalResult
    async def aanswer(self, query, *, workspace=None, workspaces=None, **kwargs) -> RetrievalResult

    # Management
    async def list_workspaces(self) -> list[str]
    async def close(self)

    # Internal
    async def _get_service(self, workspace: str) -> RAGService
```

Routing logic:
- `workspace` param for single-workspace ops
- `workspaces` param triggers federation (parallel query + merge)
- Default workspace from `config.workspace`
- Absorbs pool.py instance caching + health check / error info

#### RAGService (`core/service.py`)

Single workspace RAG unit. Creates ONE RAGAnything instance.

```python
class RAGService:
    rag: RAGAnything              # single instance, holds LightRAG
    ingestion: IngestionEngine    # composes self.rag
    retrieval: RetrievalEngine    # composes self.rag
    config: DlightragConfig

    @classmethod
    async def create(cls, config=None, **kwargs) -> RAGService

    async def aretrieve(self, query, **kwargs) -> RetrievalResult
    async def aanswer(self, query, **kwargs) -> RetrievalResult
    async def aingest(self, source_type, **kwargs) -> dict
    async def close(self)
```

Key change: no more `rag_text` / `rag_vision` split. RetrievalEngine handles
multimodal internally.

#### RetrievalEngine (`core/retrieval/engine.py`)

Composition over inheritance. Wraps RAGAnything for structured retrieval.

```python
class RetrievalEngine:
    def __init__(self, rag: RAGAnything, config: DlightragConfig):
        self.rag = rag       # composition, not inheritance
        self.config = config

    async def aretrieve(self, query, ...) -> RetrievalResult:
        # 1. Multimodal query enhancement via rag._process_multimodal_query_content()
        # 2. Structured data via rag.lightrag.aquery_data()
        # 3. Source/media augmentation via augment_retrieval_result()

    async def aanswer(self, query, ...) -> RetrievalResult:
        # Same but via rag.lightrag.aquery_llm() (includes LLM answer)
```

Replaces EnhancedRAGAnything. `RetrievalResult`, `augment_retrieval_result()`,
and `build_sources_and_media_from_contexts()` remain as-is.

### LLM Roles

| Role | Config | Used By | Independently Configurable |
|------|--------|---------|---------------------------|
| **LLM** | `chat_model` + `llm_provider` | LightRAG: entity extraction + answer generation | Primary LLM |
| **Vision LLM** | `vision_model` + `vision_provider` | RAGAnything: image/multimodal processing | Yes, fully independent |
| **Ingestion LLM** | `ingestion_model` | TODO: awaiting LightRAG per-call override support | Config retained, not effective yet |

Single `llm_model_func` bound to LightRAG at construction time.
Vision model function is separate and used at RAGAnything level only.

### Deleted Modules

- `src/dlightrag/pool.py` — logic absorbed into `core/servicemanager.py`
- `src/dlightrag/service.py` — moved to `core/service.py`
- `src/dlightrag/retrieval/` — moved to `core/retrieval/`
- `src/dlightrag/ingestion/` — moved to `core/ingestion/`
- `EnhancedRAGAnything` class — replaced by `RetrievalEngine`

### Import Path Changes

```python
# Before
from dlightrag.service import RAGService
from dlightrag.pool import get_workspace_service, get_shared_rag_service
from dlightrag.retrieval.federation import federated_retrieve
from dlightrag.retrieval.engine import EnhancedRAGAnything, RetrievalResult

# After
from dlightrag.core.servicemanager import RAGServiceManager
from dlightrag.core.service import RAGService
from dlightrag.core.federation import federated_retrieve
from dlightrag.core.retrieval.engine import RetrievalEngine, RetrievalResult
```

## Affected Files

| File | Change |
|------|--------|
| `api/server.py` | Depend on RAGServiceManager only, remove pool/federation imports |
| `mcp/server.py` | Same as above |
| `scripts/cli.py` | Update imports, use `manager.aingest(workspace=...)` |
| `scripts/reset.py` | Update imports |
| `.env.example` | Add TODO comment on `DLIGHTRAG_INGESTION_MODEL` |
| `README.md` | Update architecture diagram, configuration docs |
| `pyproject.toml` | Entry points unchanged (dlightrag-api, dlightrag-mcp) |
| `Dockerfile` | Unchanged (entry points unchanged) |
| `docker-compose.yml` | Unchanged |
| `tests/unit/test_api_server.py` | Update imports |
| `tests/unit/test_federation.py` | Update imports, federation.py path change |
| `tests/unit/test_pool.py` | Rename/refactor to test_servicemanager.py |
| `tests/unit/test_service.py` | New: test slimmed RAGService |
| `tests/unit/test_retrieval_engine.py` | New: test RetrievalEngine (replaces EnhancedRAGAnything tests) |

## Testing Strategy

- Unit tests for RetrievalEngine: mock RAGAnything, verify it calls
  `lightrag.aquery_data()` / `aquery_llm()` correctly
- Unit tests for RAGServiceManager: mock RAGService, verify routing
  (single ws → direct, multi ws → federation)
- Existing federation tests: update imports, logic unchanged
- Existing API/MCP tests: update imports, verify handler simplification
- Run full test suite after each task to catch regressions
