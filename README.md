# DlightRAG

[![PyPI](https://img.shields.io/pypi/v/dlightrag)](https://pypi.org/project/dlightrag/)
[![CI](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml/badge.svg)](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml)

Dual-mode multimodal RAG built on [LightRAG](https://github.com/HKUDS/LightRAG) — knowledge graph + vector retrieval as a modern and unified production-ready service.

## Features

- **Dual multimodal RAG modes** — caption mode (parse → caption → embed) for pipeline based mulitimodal paradigm; unified mode (render → multimodal embed) for more modern multimodal paradigm
- **Knowledge graph + vector retrieval** — fusional search with LightRAG's foundation
- **Multimodal ingestion** — PDF, Word, Excel, PowerPoint, images, etc.
- **Reranking** — generic LLM-based listwise; Specialized rerankers support from Cohere, Jina, Aliyun, Azure Cohere; Support any additional backend via custom endpoint
- **Cross-workspace federation** — query across workspaces with round-robin merging
- **Content-aware dedup** — files hashed by content, preventing duplicate ingestion
- **Flexible sourcing** — local filesystem, Azure Blob Storage, Snowflake
- **Three interfaces** — Python SDK, REST API, MCP server


## Quick Start

### Python SDK

```bash
uv add dlightrag        # or: pip install dlightrag
```

```python
import asyncio
from dlightrag import RAGService, DlightragConfig

async def main():
    config = DlightragConfig(openai_api_key="sk-...")
    service = await RAGService.create(config=config)

    await service.aingest(source_type="local", path="./docs")

    result = await service.aretrieve(query="What are the key findings?")
    print(result.contexts)

    result = await service.aanswer(query="What are the key findings?")
    print(result.answer)

    await service.close()

asyncio.run(main())
```

> Requires PostgreSQL with pgvector + AGE, or JSON fallback for development (see [Configuration](#configuration)).

### Docker (Self-Hosted)

```bash
git clone https://github.com/hanlianlu/dlightrag.git && cd dlightrag
cp .env.example .env    # edit .env — at minimum set DLIGHTRAG_OPENAI_API_KEY
docker compose up
```

Includes PostgreSQL (pgvector + AGE), REST API (`:8100`), and MCP server (`:8101`).

> **Local models (Ollama, Xinference, etc.):** use `host.docker.internal` instead of `localhost` in base URL settings.

```bash
curl http://localhost:8100/health

curl -X POST http://localhost:8100/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "path": "/app/dlightrag_storage/sources"}'

curl -X POST http://localhost:8100/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?"}'

curl -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "stream": true}'
```

### MCP Server (for AI Agents)

```bash
uv tool install dlightrag   # or: pip install dlightrag
dlightrag-mcp --env-file /path/to/.env
```

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

Tools: `retrieve`, `answer`, `ingest`, `list_files`, `delete_files`, `list_workspaces` — all with workspace isolation.


## Configuration

All settings via `DLIGHTRAG_` env vars, `.env` file, or constructor args. See [`.env.example`](.env.example) for the full reference.

**Priority:** constructor args > env vars > `.env` file > defaults

### RAG Mode

The first decision — determines your ingestion pipeline, model requirements, and retrieval behavior.

| Mode | Pipeline | Best for |
|------|----------|----------|
| `caption` (default) | Document parsing → VLM captioning → text embedding → KG | Text-heavy documents, structured elements |
| `unified` | Page rendering → multimodal embedding → VLM entity extraction → KG | Visually rich documents (charts, diagrams, complex layouts) |

**Model usage by stage:**

| Stage | Caption | Unified |
|-------|---------|---------|
| Image captioning | `VISION_MODEL` ¹ | `VISION_MODEL` |
| Table / equation captioning | `CHAT_MODEL` | — |
| Entity extraction | `CHAT_MODEL` | `CHAT_MODEL` |
| Embedding | `EMBEDDING_MODEL` | `EMBEDDING_MODEL` (multimodal) |
| Rerank | `RERANK_*` via LightRAG | `VISION_MODEL` ² or `RERANK_*` API |
| Answer generation | `CHAT_MODEL` | `VISION_MODEL` (sees page images) |

¹ Falls back to `CHAT_MODEL` if vision model not configured.
² When `RERANK_BACKEND=llm` (pointwise VLM scoring).

For unified mode, set `DLIGHTRAG_RAG_MODE=unified` and point embedding/vision at multimodal models:

```bash
DLIGHTRAG_RAG_MODE=unified
DLIGHTRAG_EMBEDDING_MODEL=Qwen3-VL-Embedding    # must be multimodal
DLIGHTRAG_EMBEDDING_DIM=4096
DLIGHTRAG_VISION_MODEL=qwen3-vl-32b
```

> **Limitations:** Snowflake is text-only (no visual embedding). A workspace is locked to one mode after first ingestion. Page images ~3-7 MB/page at 250 DPI.

### Providers

| Variable | Default | Description |
|---|---|---|
| `DLIGHTRAG_LLM_PROVIDER` | `openai` | `openai`, `azure_openai`, `anthropic`, `google_gemini`, `qwen`, `minimax`, `xinference`, `openrouter` |
| `DLIGHTRAG_EMBEDDING_PROVIDER` | (follows `llm_provider`) | Override embedding provider |
| `DLIGHTRAG_VISION_PROVIDER` | (follows `llm_provider`) | Override vision provider |
| `DLIGHTRAG_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model |

Each provider uses its own API key. For **Ollama**, use `openai` provider with `DLIGHTRAG_OPENAI_BASE_URL` pointing to Ollama.

### Storage Backends

| Variable | Default | Options |
|---|---|---|
| `DLIGHTRAG_VECTOR_STORAGE` | `PGVectorStorage` | PGVectorStorage, MilvusVectorDBStorage, NanoVectorDBStorage, ... |
| `DLIGHTRAG_GRAPH_STORAGE` | `PGGraphStorage` | PGGraphStorage, Neo4JStorage, NetworkXStorage, ... |
| `DLIGHTRAG_KV_STORAGE` | `PGKVStorage` | PGKVStorage, JsonKVStorage, RedisKVStorage, ... |
| `DLIGHTRAG_DOC_STATUS_STORAGE` | `PGDocStatusStorage` | PGDocStatusStorage, JsonDocStatusStorage, ... |

### Workspaces

Each workspace has its own knowledge graph, vector store, and document index. `DLIGHTRAG_WORKSPACE` (default: `default`) is automatically bridged to backend-specific env vars — no manual setup needed.

| Backend type | Isolation mechanism |
|---|---|
| PostgreSQL (PG*) | `workspace` column / graph name in same database |
| Neo4j / Memgraph | Label prefix |
| Milvus / Qdrant | Collection prefix |
| MongoDB / Redis | Collection scope |
| JSON / Nano / NetworkX / Faiss | Subdirectory under `working_dir/<workspace>/` |

### Reranking

| Variable | Default | Description |
|---|---|---|
| `DLIGHTRAG_RERANK_BACKEND` | `llm` | `llm`, `cohere`, `jina`, `aliyun`, `azure_cohere` |
| `DLIGHTRAG_RERANK_MODEL` | (backend default) | Model name sent to the endpoint |
| `DLIGHTRAG_RERANK_BASE_URL` | (provider default) | Custom endpoint URL for any compatible service |
| `DLIGHTRAG_RERANK_API_KEY` | — | API key (falls back to provider-specific keys) |

| Backend | Default model | Key |
|---|---|---|
| `llm` | (follows `CHAT_MODEL`) | (follows `LLM_PROVIDER`) |
| `cohere` | `rerank-v4.0-pro` | `DLIGHTRAG_COHERE_API_KEY` |
| `jina` | `jina-reranker-v3` | `DLIGHTRAG_JINA_API_KEY` |
| `aliyun` | `qwen3-rerank` | `DLIGHTRAG_ALIYUN_RERANK_API_KEY` |
| `azure_cohere` | `Cohere-rerank-v4.0-pro` | `DLIGHTRAG_AZURE_COHERE_API_KEY` |

Point any backend at a local reranker (Xinference, LiteLLM, etc.) via `RERANK_BASE_URL` + `RERANK_MODEL`.


## REST API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Ingest from local, Azure Blob, or Snowflake |
| `POST` | `/retrieve` | Contexts + sources (no LLM answer) |
| `POST` | `/answer` | LLM answer + contexts + sources (`stream: true` for SSE) |
| `GET` | `/files` | List ingested documents |
| `DELETE` | `/files` | Delete documents |
| `GET` | `/workspaces` | List available workspaces |
| `GET` | `/health` | Health check with storage status |

All write endpoints accept optional `workspace`; read endpoints accept `workspaces` list for cross-workspace federated search. Set `DLIGHTRAG_API_AUTH_TOKEN` to enable bearer auth.

### SSE Streaming

Set `"stream": true` to receive Server-Sent Events:

| Event type | Payload | Description |
|---|---|---|
| `context` | `{type, data, raw}` | Contexts and sources (sent first) |
| `token` | `{type, content}` | LLM answer token |
| `done` | `{type}` | Stream complete |
| `error` | `{type, message}` | Error mid-stream |


## Development

```bash
git clone https://github.com/hanlianlu/dlightrag.git && cd dlightrag
cp .env.example .env && uv sync
docker compose up -d                # PostgreSQL + API + MCP
docker compose up postgres -d       # PostgreSQL only
```

```bash
uv run pytest tests/unit            # unit tests (no external services)
uv run pytest tests/integration     # integration tests (requires PostgreSQL)
uv run ruff check src/ tests/ scripts/ && uv run ruff format --check src/ tests/ scripts/
```

> **Skip PostgreSQL** for development:
> ```
> DLIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage
> DLIGHTRAG_GRAPH_STORAGE=NetworkXStorage
> DLIGHTRAG_KV_STORAGE=JsonKVStorage
> DLIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage
> ```


## Architecture

<p align="center">
  <img src="docs/architecture.png" alt="DlightRAG Architecture" width="800" />
</p>

<sub>Source: <a href="docs/architecture.drawio">docs/architecture.drawio</a></sub>


## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

Built by HanlianLyu. Contributions welcome!
