# corprag

Multimodal RAG service built on [RAGAnything](https://github.com/HKUDS/RAG-Anything) + [LightRAG](https://github.com/HKUDS/LightRAG) with PostgreSQL as the unified backend.

## Features

- **Multimodal ingestion** -- PDF, Word, Excel, PowerPoint, images, and more via MinerU / Docling parsing
- **Knowledge graph + vector search** -- Dual retrieval with Apache AGE (graph) and pgvector (vector) in a single PostgreSQL instance
- **Content deduplication** -- SHA-256 hash index prevents re-ingesting unchanged documents
- **Multiple retrieval modes** -- local, global, hybrid, naive, and mix modes via LightRAG
- **LLM reranking** -- Optional reranking with OpenAI, Cohere, or Azure Cohere
- **Three interfaces** -- Python SDK, REST API, and MCP server
- **Pluggable storage** -- Default PostgreSQL, also supports Neo4j, Milvus, Redis, MongoDB, JSON (via LightRAG)
- **Flexible data sourcing** -- Ingest from local filesystem, Azure Blob Storage, or Snowflake tables

## Quick Start

### Option A: Python SDK

```bash
# Install from GitHub
uv add "corprag @ git+https://github.com/hanlianlu/corprag.git"
# or with pip:
pip install "corprag @ git+https://github.com/hanlianlu/corprag.git"
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

    # Query
    answer = await service.aretrieve(query="What are the key findings?")
    print(answer.answer)

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

Everything is included: PostgreSQL (pgvector + AGE), corprag REST API server.

```bash
# Health check
curl http://localhost:8100/health

# Ingest documents
curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "/app/corprag_storage/sources"}'

# Query
curl -X POST http://localhost:8100/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?"}'
```

### Option C: MCP Server (for AI Agents)

```bash
uv add "corprag @ git+https://github.com/hanlianlu/corprag.git"
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

Available MCP tools: `retrieve`, `ingest`, `list_files`, `delete_files`.

## Local Development

```bash
git clone https://github.com/hanlianlu/corprag.git
cd corprag

# Configure environment
cp .env.example .env
# Edit .env -- at minimum set CORPRAG_OPENAI_API_KEY

# Install dependencies
uv sync          # or: pip install -e ".[dev]"

# Start PostgreSQL (pick one)
docker compose up postgres -d        # via Docker
# or use an existing PostgreSQL with pgvector + AGE extensions

# Run tests
pytest tests/unit
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

### Required

| Variable | Description |
|---|---|
| `CORPRAG_OPENAI_API_KEY` | OpenAI API key (or use Azure below) |

### LLM Provider

| Variable | Default | Description |
|---|---|---|
| `CORPRAG_LLM_PROVIDER` | `openai` | `openai` or `azure_openai` |
| `CORPRAG_OPENAI_CHAT_MODEL` | `gpt-4.1-mini` | Chat model |
| `CORPRAG_OPENAI_INGESTION_MODEL` | `gpt-4.1-mini` | Ingestion model |
| `CORPRAG_OPENAI_VISION_MODEL` | `gpt-4.1-mini` | Vision model |
| `CORPRAG_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model |

### PostgreSQL

| Variable | Default | Description |
|---|---|---|
| `CORPRAG_POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `CORPRAG_POSTGRES_PORT` | `5432` | PostgreSQL port |
| `CORPRAG_POSTGRES_USER` | `rag` | PostgreSQL user |
| `CORPRAG_POSTGRES_PASSWORD` | `rag` | PostgreSQL password |
| `CORPRAG_POSTGRES_DATABASE` | `corprag` | Database name |

### Storage Backends

| Variable | Default | Options |
|---|---|---|
| `CORPRAG_VECTOR_STORAGE` | `PGVectorStorage` | PGVectorStorage, MilvusVectorDBStorage, NanoVectorDBStorage, ... |
| `CORPRAG_GRAPH_STORAGE` | `PGGraphStorage` | PGGraphStorage, Neo4JStorage, NetworkXStorage, ... |
| `CORPRAG_KV_STORAGE` | `PGKVStorage` | PGKVStorage, JsonKVStorage, RedisKVStorage, ... |
| `CORPRAG_DOC_STATUS_STORAGE` | `PGDocStatusStorage` | PGDocStatusStorage, JsonDocStatusStorage, ... |

### Development without PostgreSQL

For local development without PostgreSQL, use JSON-based storage:

```python
config = CorpragConfig(
    openai_api_key="sk-...",
    kv_storage="JsonKVStorage",
    vector_storage="NanoVectorDBStorage",
    graph_storage="NetworkXStorage",
    doc_status_storage="JsonDocStatusStorage",
)
```

See [.env.example](.env.example) for all available configuration options.

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Ingest documents from local, Azure Blob, or Snowflake |
| `POST` | `/query` | Query the knowledge base |
| `GET` | `/files` | List ingested documents |
| `DELETE` | `/files` | Delete documents |
| `GET` | `/health` | Health check with storage status |

Set `CORPRAG_API_AUTH_TOKEN` to enable bearer token authentication.

## Architecture

```
CorpragConfig (pydantic-settings)
       |
   RAGService  ←── cancel_checker / url_transformer callbacks
    /       \
IngestionPipeline    EnhancedRAGAnything (retrieval)
    |                       |
DataSources              LightRAG + RAGAnything
(local/azure/snowflake)     |
                     PostgreSQL (pgvector + AGE)
                     or any LightRAG backend
```

**Dual interface:**
- REST API (`corprag-api`) -- for bulk ingestion, batch queries, ops
- MCP server (`corprag-mcp`) -- for AI agent integration (Claude, DeerFlow, etc.)

## Docker Reference

The included `Dockerfile` and `docker-compose.yml` provide:

- **corprag-api** service: REST API server on port 8100
- **postgres** service: PostgreSQL with pgvector + AGE (`gzdaniel/postgres-for-rag` image)
- Persistent volumes for data and PostgreSQL storage
- Health checks for PostgreSQL readiness

```bash
# Build and start
docker compose up -d

# View logs
docker compose logs -f corprag-api

# Stop
docker compose down
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

Maintained by HanlianLyu and hllyu
