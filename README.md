# corprag

[![PyPI](https://img.shields.io/pypi/v/corprag)](https://pypi.org/project/corprag/)
[![CI](https://github.com/hanlianlu/corprag/actions/workflows/ci.yml/badge.svg)](https://github.com/hanlianlu/corprag/actions/workflows/ci.yml)

Multimodal RAG service built upon [RAGAnything](https://github.com/HKUDS/RAG-Anything) + [LightRAG](https://github.com/HKUDS/LightRAG) with PostgreSQL and additional enhancements as the unified service.

## Features

- 🗂️ **Multimodal ingestion** -- PDF, Word, Excel, PowerPoint, images, and more via parsing engine
- 🔭 **Knowledge graph + vector search** -- Dual retrieval with Apache AGE (graph) and pgvector (vector) in a single PostgreSQL instance
- ♻️ **Content deduplication** -- SHA-256 hash index prevents re-ingesting unchanged documents
- 🎛️ **Multiple retrieval modes** -- local, global, hybrid, naive, and mix modes for flexible retrieval strategies
- 🤖 **Multi-provider LLM** -- OpenAI, Azure OpenAI, Anthropic, Google Gemini, Qwen, MiniMax, Ollama, OpenRouter
- ↕️ **LLM reranking** -- Optional reranking with any LLM provider, or specialized Cohere, or Azure Cohere
- 🛠️ **Three interfaces** -- Python SDK, REST API, and MCP server
- 🔌 **Pluggable storage** -- Default PostgreSQL, also supports Neo4j, Milvus, Redis, MongoDB, JSON (via LightRAG)
- 🌐 **Flexible data sourcing** -- Ingest from local filesystem, Azure Blob Storage, or Snowflake tables

## Quick Start

### Option A: Python SDK

```bash
pip install corprag   # requires Python 3.12
```

```python
import asyncio
from corprag import RAGService, CorpragConfig

async def main():
    # Minimal config -- just needs an LLM API key
    config = CorpragConfig(openai_api_key="sk-...")

    # Initialize (connects to PostgreSQL, sets up RAG engine)
    service = await RAGService.create(config=config)

    # Ingest documents
    result = await service.aingest(source_type="local", path="./docs")
    print(f"Ingested {result['ingested']} documents")

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
git clone https://github.com/hanlianlu/corprag.git
cd corprag
cp .env.example .env
# Edit .env -- at minimum set CORPRAG_OPENAI_API_KEY
docker compose up
```

Everything is included: PostgreSQL (pgvector + AGE), REST API (`:8100`), and MCP server (`:8101`).

```bash
# Health check
curl http://localhost:8100/health

# Ingest documents
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "/app/corprag_storage/sources"}'

# Retrieve (contexts + sources, no LLM answer)
curl -X POST http://localhost:8100/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?"}'

# Answer (LLM-generated answer + contexts + sources)
curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?"}'
```

### Option C: MCP Server (for AI Agents)

```bash
pip install corprag
corprag-mcp  # stdio mode
```

Add to Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "corprag": {
      "command": "corprag-mcp"
    }
  }
}
```

Available MCP tools: `retrieve`, `answer`, `ingest`, `list_files`, `delete_files`.

> **Note:** Like the SDK, the MCP server requires PostgreSQL with pgvector + AGE, or JSON fallback storage (see [Configuration](#configuration)).

## Local Development

```bash
git clone https://github.com/hanlianlu/corprag.git
cd corprag

# Configure environment
cp .env.example .env
# Edit .env -- at minimum set CORPRAG_OPENAI_API_KEY

# Install dependencies
uv sync

# Start PostgreSQL (pick one)
docker compose up postgres -d        # via Docker
# or use an existing PostgreSQL with pgvector + AGE extensions
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
> CORPRAG_VECTOR_STORAGE=NanoVectorDBStorage
> CORPRAG_GRAPH_STORAGE=NetworkXStorage
> CORPRAG_KV_STORAGE=JsonKVStorage
> CORPRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage
> ```

> **Note:** Excel-to-PDF conversion requires [LibreOffice](https://www.libreoffice.org/) (`libreoffice` on PATH). If not installed, Excel files are ingested as-is without conversion. The Docker image includes LibreOffice.

## Configuration

All configuration is via `CORPRAG_` environment variables, a `.env` file, or constructor arguments.

**Priority order** (highest to lowest):
1. Constructor args -- `CorpragConfig(openai_api_key="sk-...")`
2. Environment variables -- `CORPRAG_OPENAI_API_KEY=sk-...`
3. `.env` file
4. Defaults

### LLM Provider

| Variable | Default | Description |
|---|---|---|
| `CORPRAG_LLM_PROVIDER` | `openai` | `openai`, `azure_openai`, `anthropic`, `google_gemini`, `qwen`, `minimax`, `ollama`, `openrouter` |
| `CORPRAG_EMBEDDING_PROVIDER` | (follows `llm_provider`) | Override embedding provider (e.g., `openai` when using Anthropic) |
| `CORPRAG_VISION_PROVIDER` | (follows `llm_provider`) | Override vision provider |
| `CORPRAG_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model |

Each provider has its own API key. Model names are unified across providers.

See [.env.example](.env.example) for all provider-specific variables.

### Storage Backends

| Variable | Default | Options |
|---|---|---|
| `CORPRAG_VECTOR_STORAGE` | `PGVectorStorage` | PGVectorStorage, MilvusVectorDBStorage, NanoVectorDBStorage, ... |
| `CORPRAG_GRAPH_STORAGE` | `PGGraphStorage` | PGGraphStorage, Neo4JStorage, NetworkXStorage, ... |
| `CORPRAG_KV_STORAGE` | `PGKVStorage` | PGKVStorage, JsonKVStorage, RedisKVStorage, ... |
| `CORPRAG_DOC_STATUS_STORAGE` | `PGDocStatusStorage` | PGDocStatusStorage, JsonDocStatusStorage, ... |

See [.env.example](.env.example) for all available configuration options.

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Ingest documents from local, Azure Blob, or Snowflake |
| `POST` | `/retrieve` | Retrieve contexts and sources (no LLM answer) |
| `POST` | `/answer` | LLM-generated answer with contexts and sources |
| `GET` | `/files` | List ingested documents |
| `DELETE` | `/files` | Delete documents |
| `GET` | `/health` | Health check with storage status |

Set `CORPRAG_API_AUTH_TOKEN` to enable bearer token authentication.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│   Python SDK  ·  REST API (:8100)  ·  MCP (:8101)    │
└─────────────────────────┬────────────────────────────┘
                          │  CorpragConfig
                 ┌────────▼────────┐
                 │   RAGService    │
                 └────┬───────┬────┘
                      │       │
          ┌───────────▼─┐  ┌──▼─────────────┐
          │  Ingestion  │  │   Retrieval    │
          │  Pipeline   │  │ (RAGAnything   │
          │             │  │  + LightRAG)   │
          │  local      │  │                │
          │  azure blob │  │  retrieve()    │
          │  snowflake  │  │  answer()      │
          └──────┬──────┘  └───────┬────────┘
                 └─────────┬───────┘
                           │
             ┌─────────────┴──────────────────┐
             │                                │
  ┌──────────▼────────┐      ┌────────────────▼─────┐
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
