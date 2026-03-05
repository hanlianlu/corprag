# DlightRAG

[![PyPI](https://img.shields.io/pypi/v/dlightrag)](https://pypi.org/project/dlightrag/)
[![CI](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml/badge.svg)](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml)

Multimodal RAG package built upon [LightRAG](https://github.com/HKUDS/LightRAG) with additional enhancements as the production ready unified service.

## Features

- 🌐 **Flexible data sourcing** -- Ingest from local filesystem, Azure Blob Storage, or Snowflake tables
- 🗂️ **Multimodal ingestion with granular enhancements** -- PDF, Word, Excel, PowerPoint, images, and more via parsing engine
- 🔭 **Knowledge graph + Vector semantic** -- Ingestion and Retrieval with LightRAG paradigm 
- ↕️ **Reranking** -- LLM-based listwise OR Reranker from Cohere, Jina, Aliyun, Azure Cohere; Point any backend at a custom endpoint (Xinference, Ollama etc.)
- ✨ **Retrieval enrichment** -- Enhanced answer and retrieval formation for better citation and reference
- 🔗 **Cross-workspace federation** -- Query across multiple workspaces in a single request with round-robin result merging and RBAC-ready interface
- 🔍 **Content-aware deduplication** -- Files are hashed by content, preventing duplicate ingestion when only metadata changes
- 🔌 **Three interfaces** -- Python SDK, REST API, and MCP server

## Quick Start

### Option A: Python SDK

```bash
uv add dlightrag        # or: pip install dlightrag
```

```python
import asyncio
from dlightrag import RAGService, DlightragConfig

async def main():
    # Minimal config example -- just needs an OpenAI API key
    config = DlightragConfig(openai_api_key="sk-...")

    # Initialize (connects to PostgreSQL, sets up RAG engine)
    service = await RAGService.create(config=config)

    # Ingest documents
    result = await service.aingest(source_type="local", path="./docs")
    print(f"Ingested {result['processed']} documents")

    # Retrieve (structured contexts + sources, no LLM answer)
    result = await service.aretrieve(query="What are the key findings?")
    print(result.contexts)

    # Answer (LLM-generated answer + structured contexts + sources)
    result = await service.aanswer(query="What are the key findings?")
    print(result.answer)

    await service.close()

asyncio.run(main())
```

> **Note:** The SDK requires a running PostgreSQL instance with pgvector + AGE extensions, or use JSON fallback for development (see [Configuration](#configuration)).


### Option B: Self-Hosted Server (Docker)

```bash
git clone https://github.com/hanlianlu/dlightrag.git
cd dlightrag
cp .env.example .env
# Edit .env -- at minimum set DLIGHTRAG_OPENAI_API_KEY
docker compose up
```

Everything is included: PostgreSQL (pgvector + AGE), REST API (`:8100`), and MCP server (`:8101`).

```bash
# Health check
curl http://localhost:8100/health

# Ingest documents (into default workspace)
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "/app/dlightrag_storage/sources"}'

# Ingest into a specific workspace
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "/data/project-a", "workspace": "project-a"}'

# Retrieve (contexts + sources, no LLM answer)
curl -X POST http://localhost:8100/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?"}'

# Cross-workspace retrieval
curl -X POST http://localhost:8100/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "workspaces": ["project-a", "project-b"]}'

# Answer (LLM-generated answer + contexts + sources)
curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?"}'

# List available workspaces
curl http://localhost:8100/workspaces
```


### Option C: MCP Server (for AI Agents)

```bash
uv tool install dlightrag   # or: pip install dlightrag
dlightrag-mcp --env-file /path/to/.env
```

Create a `.env` with `DLIGHTRAG_*` variables — see [`.env.example`](.env.example) for a full template.

Example MCP client configuration (works with Claude Desktop, VS Code, Cursor, or any MCP-compatible agent):

```json
{
  "mcpServers": {
    "dlightrag": {
      "command": "uvx",
      "args": ["dlightrag-mcp", "--env-file", "/absolute/path/to/.env"]
    }
  }
}
```

Available MCP tools: `retrieve`, `answer`, `ingest`, `list_files`, `delete_files`, `list_workspaces`. All tools support workspace isolation — `ingest`/`list_files`/`delete_files` accept optional `workspace`, while `retrieve`/`answer` accept optional `workspaces` list for cross-workspace federated search.

> **Note:** Like the SDK, the MCP server requires PostgreSQL with pgvector + AGE, or JSON fallback storage (see [Configuration](#configuration)). Use `--env-file` to point to your `.env` with `DLIGHTRAG_*` variables (API keys, database, etc.).


## Local Development

```bash
git clone https://github.com/hanlianlu/dlightrag.git
cd dlightrag

# Configure environment
cp .env.example .env

# Edit .env -- at minimum set DLIGHTRAG_OPENAI_API_KEY

# Install dependencies
uv sync

# Start PostgreSQL (pick one)
docker compose up postgres -d        # via Docker

# starts all services including PostgreSQL, API, and MCP
docker compose up  -d
```


### Testing

```bash
uv run pytest tests/unit                    # unit tests (no external services)
uv run pytest tests/integration             # integration tests (requires PostgreSQL)
uv run pytest                               # all tests
uv run pytest --cov-report=html             # + HTML report → htmlcov/index.html
```


### Linting

```bash
uv run ruff check src/ tests/ scripts/              # lint check
uv run ruff format --check src/ tests/ scripts/     # format check

uv run ruff check --fix src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/
```

> **Tip:** To skip PostgreSQL entirely during development, set these in your `.env`:
> ```
> DLIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage
> DLIGHTRAG_GRAPH_STORAGE=NetworkXStorage
> DLIGHTRAG_KV_STORAGE=JsonKVStorage
> DLIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage
> ```

> **Note:** Excel-to-PDF conversion requires [LibreOffice](https://www.libreoffice.org/) (`libreoffice` on PATH). If not installed, Excel files are ingested as-is without conversion. The Docker image includes LibreOffice.


## Configuration

All configuration is via `DLIGHTRAG_` environment variables, a `.env` file, or constructor arguments.

**Priority order** (highest to lowest):
1. Constructor args -- `DlightragConfig(openai_api_key="sk-...")`
2. Environment variables -- `DLIGHTRAG_OPENAI_API_KEY=sk-...`
3. `.env` file
4. Defaults

### LLM Provider

| Variable | Default | Description |
|---|---|---|
| `DLIGHTRAG_LLM_PROVIDER` | `openai` | `openai`, `azure_openai`, `anthropic`, `google_gemini`, `qwen`, `minimax`, `ollama`, `xinference`, `openrouter` |
| `DLIGHTRAG_EMBEDDING_PROVIDER` | (follows `llm_provider`) | Override embedding provider (e.g., `openai` when using Anthropic) |
| `DLIGHTRAG_VISION_PROVIDER` | (follows `llm_provider`) | Override vision provider |
| `DLIGHTRAG_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model |

Each provider has its own API key. Model names are unified across providers.

See [.env.example](.env.example) for all provider-specific variables.

### Storage Backends

| Variable | Default | Options |
|---|---|---|
| `DLIGHTRAG_VECTOR_STORAGE` | `PGVectorStorage` | PGVectorStorage, MilvusVectorDBStorage, NanoVectorDBStorage, ... |
| `DLIGHTRAG_GRAPH_STORAGE` | `PGGraphStorage` | PGGraphStorage, Neo4JStorage, NetworkXStorage, ... |
| `DLIGHTRAG_KV_STORAGE` | `PGKVStorage` | PGKVStorage, JsonKVStorage, RedisKVStorage, ... |
| `DLIGHTRAG_DOC_STATUS_STORAGE` | `PGDocStatusStorage` | PGDocStatusStorage, JsonDocStatusStorage, ... |

See [.env.example](.env.example) for all available configuration options.

### Workspaces

Workspaces provide data isolation — each workspace has its own knowledge graph, vector store, and document index. Isolation works across all storage backend combinations:

| Backend type | Isolation mechanism |
|---|---|
| PostgreSQL (PG*) | `workspace` column / graph name in same database |
| Neo4j / Memgraph | Label prefix via `NEO4J_WORKSPACE` / `MEMGRAPH_WORKSPACE` env var |
| Milvus / Qdrant | Collection prefix via LightRAG constructor `workspace` param |
| MongoDB / Redis | Collection scope via `MONGODB_WORKSPACE` / `REDIS_WORKSPACE` env var |
| JSON / Nano / NetworkX / Faiss | Subdirectory under `working_dir/<workspace>/` |

| Variable | Default | Description |
|---|---|---|
| `DLIGHTRAG_WORKSPACE` | `default` | Default workspace name |

DlightRAG automatically bridges `DLIGHTRAG_WORKSPACE` to the backend-specific env var (e.g. `POSTGRES_WORKSPACE`, `NEO4J_WORKSPACE`) and passes it via LightRAG's constructor — no manual env var setup needed.

**Usage in endpoints:**
- Write operations (`/ingest`, `/files` DELETE) accept an optional `workspace` parameter
- Read operations (`/retrieve`, `/answer`) accept an optional `workspaces` list for cross-workspace federated search (round-robin result merging)
- `GET /workspaces` discovers available workspaces (PG: queries database, filesystem backends: scans `working_dir` subdirectories)
- When omitted, the default workspace is used

### Reranking

Five backends are available. The `cohere`, `jina`, and `aliyun` backends use LightRAG's built-in rerank functions and can target any API-compatible service via `RERANK_BASE_URL`.

| Variable | Default | Description |
|---|---|---|
| `DLIGHTRAG_RERANK_BACKEND` | `llm` | `llm`, `cohere`, `jina`, `aliyun`, `azure_cohere` |
| `DLIGHTRAG_RERANK_MODEL` | (backend default) | Model name sent to the endpoint |
| `DLIGHTRAG_RERANK_BASE_URL` | (provider default) | Custom endpoint URL for any compatible service |
| `DLIGHTRAG_RERANK_API_KEY` | — | Generic API key (falls back to provider-specific keys) |

**Backend defaults** (used when `RERANK_MODEL` / `RERANK_API_KEY` are not set):

| Backend | Default model | Provider-specific key |
|---|---|---|
| `llm` | (follows `INGESTION_MODEL`) | (follows `LLM_PROVIDER` credentials) |
| `cohere` | `rerank-v4.0-pro` | `DLIGHTRAG_COHERE_API_KEY` |
| `jina` | `jina-reranker-v3` | `DLIGHTRAG_JINA_API_KEY` |
| `aliyun` | `qwen3-rerank` | `DLIGHTRAG_ALIYUN_RERANK_API_KEY` |
| `azure_cohere` | `Cohere-rerank-v4.0-pro` | `DLIGHTRAG_AZURE_COHERE_API_KEY` + `DLIGHTRAG_AZURE_COHERE_ENDPOINT` |

**Examples:**

```bash
# Cohere (direct)
DLIGHTRAG_RERANK_BACKEND=cohere
DLIGHTRAG_COHERE_API_KEY=your-key

# Local reranker via Xinference / LiteLLM / any Cohere-compatible endpoint
DLIGHTRAG_RERANK_BACKEND=cohere
DLIGHTRAG_RERANK_MODEL=bge-reranker-v2-m3
DLIGHTRAG_RERANK_BASE_URL=http://localhost:9997/v1/rerank

# LLM-based listwise reranker (default -- no extra config needed)
DLIGHTRAG_RERANK_BACKEND=llm
```

See [.env.example](.env.example) for all reranking options.


## REST API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Ingest documents from local, Azure Blob, or Snowflake. Optional `workspace` param. |
| `POST` | `/retrieve` | Retrieve contexts and sources (no LLM answer). Optional `workspaces` list for cross-workspace search. |
| `POST` | `/answer` | LLM-generated answer with contexts and sources. Optional `workspaces` list for cross-workspace search. |
| `GET` | `/files` | List ingested documents. Optional `?workspace=` query param. |
| `DELETE` | `/files` | Delete documents. Optional `workspace` param. |
| `GET` | `/workspaces` | List all available workspaces. |
| `GET` | `/health` | Health check with storage status. |

Set `DLIGHTRAG_API_AUTH_TOKEN` to enable bearer token authentication.


## Architecture

```
┌──────────────────────────────────────────────────────┐
│   Python SDK  ·  REST API (:8100)  ·  MCP (:8101)    │
└─────────────────────────┬────────────────────────────┘
                          │  workspace(s) param
               ┌──────────▼──────────┐
               │  RAGServiceManager  │  lazy per-workspace cache + federation
               └──────────┬──────────┘
                          │
                 ┌────────▼────────┐
                 │   RAGService    │  one per workspace
                 └────┬───┬───┬───┘
                      │   │   │
          ┌───────────▼┐  │  ┌▼──────────────┐
          │ Ingestion  │  │  │  Retrieval    │
          │ Pipeline   │  │  │  Engine       │
          │            │  │  │               │
          │ local      │  │  │  retrieve()   │
          │ azure blob │  │  │  answer()     │
          │ snowflake  │  │  └───────┬───────┘
          └──────┬─────┘  │         │
                 │  ┌─────▼─────┐   │
                 └──│RAGAnything│───┘  single shared instance
                    │ (LightRAG)│
                    └─────┬─────┘
                          │
            ┌─────────────┴──────────────────┐
            │                                │
  ┌─────────▼─────────┐      ┌──────────────▼───────┐
  │   LLM Providers   │      │      Storage         │
  │  Chat · Embed     │      │  PostgreSQL          │
  │  Vision · Rerank  │      │  (pgvector + AGE)    │
  │                   │      │                      │
  │  OpenAI · Azure   │      │  Neo4j · Milvus      │
  │  Anthropic ·      │      │  Redis · JSON · ...  │
  │  Gemini · Qwen    │      └──────────────────────┘
  │  Ollama · ...     │
  └───────────────────┘
```


## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

Built by HanlianLyu. Contributions welcome! Please open issues or pull requests on GitHub.
