# DlightRAG

[![PyPI](https://img.shields.io/pypi/v/dlightrag)](https://pypi.org/project/dlightrag/)
[![CI](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml/badge.svg)](https://github.com/hanlianlu/dlightrag/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hanlianlu/DlightRAG)

DlightRAG is a production multimodal RAG service built on LightRAG. It combines
knowledge-graph and vector retrieval with metadata filtering, BM25, visual
retrieval, reranking, citations, highlights, and durable agentic answers. The
same runtime is available through Web, REST, MCP, and an in-process Python API.

**Runtime:** Python ≥3.14.7 · PostgreSQL 18 ecosystem · Apache-2.0

LightRAG storage defaults are exactly `PGKVStorage`, `PGVectorStorage`,
`PGTableGraphStorage`, and `PGDocStatusStorage`. Writer deployments may
explicitly replace only the vector leg with `MilvusVectorDBStorage` (including
Milvus-compatible Zilliz endpoints) by installing `dlightrag[milvus]`; reader
processes currently remain PostgreSQL-vector-only.

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="DlightRAG system context shown as one black box between browser users, REST and MCP clients, trusted embedding applications, an optional Web identity boundary, external AI, parser, corpus source and Research systems, PostgreSQL, and a shared corpus artifact root" width="1180" />
</p>

LightRAG supplies graph and vector retrieval. DlightRAG owns product policy,
multimodal alignment, durable ingestion and answers, security, storage adapters,
and public interfaces. Fast and Research answers share one durable conversation
tree; Research adds a per-run workspace, tools, memory, and child agents.
See [Architecture](docs/architecture.md) for module and storage ownership.

## Deployment Paths

| Path | PostgreSQL | Parser | Security |
|---|---|---|---|
| Local Docker | Compose PG18 | Self-hosted MinerU by default | Loopback, `auth_mode: none` |
| Native API | Compose or external PG18 | Any reachable MinerU or Docling | Local or explicit auth |
| Shared service | Managed or self-hosted PG18 | Independently operated parser | `simple` or `jwt` |
| Enterprise | Managed PG18 | Independently operated parser | JWKS plus claim access control |

The parser runs outside the DlightRAG app container. The checked-in Docker
configuration uses self-hosted MinerU at
`http://host.docker.internal:8210`. Docling and MinerU cloud remain supported.

## Quick Start

Install [Docker + Compose](https://docs.docker.com/get-docker/),
[`uv`](https://docs.astral.sh/uv/), `git`, and `make`.

### Interactive setup

```bash
git clone https://github.com/hanlianlu/dlightrag.git
cd dlightrag
uv run prerequisite_setup.py
```

The wizard configures models, parser, secrets, and the local stack. It defaults
to self-hosted MinerU and is safe to rerun.

### Manual setup

```bash
git clone https://github.com/hanlianlu/dlightrag.git
cd dlightrag
cp .env.example .env
mkdir -p "${HOME}/.dlightrag/skills"
```

The last command prepares the default read-only operator Skills bind source;
when `COMPOSE_GLOBAL_SKILLS_DIR` selects another host path, create that directory
instead. Add the keys required by your `config.yaml` model blocks:

```bash
DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=...
DLIGHTRAG_MODELS__EMBEDDING__API_KEY=...
DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY=...
DLIGHTRAG_MODELS__CHAT__ROLES__KEYWORD__API_KEY=...
DLIGHTRAG_MODELS__CHAT__ROLES__QUERY__API_KEY=...
DLIGHTRAG_MODELS__CHAT__ROLES__VLM__API_KEY=...
DLIGHTRAG_MODELS__RERANK__API_KEY=...
```

Install and start MinerU, then start DlightRAG:

```bash
make mineru-install
make mineru-service-install  # installs and starts the background service
curl http://127.0.0.1:8210/health

docker compose up -d
docker compose ps
```

Open <http://localhost:8100/web/>. The stack publishes:

| Service | Address |
|---|---|
| REST API and Web | `http://127.0.0.1:8100` |
| MCP streamable HTTP | `http://127.0.0.1:8101` |
| PostgreSQL | `127.0.0.1:5432` |

Use `make mineru-api` when the platform cannot install a background user
service. To use Docling, replace the MinerU block in `config.yaml`; a commented
example is included there. Parser changes affect only new parses.

Configuration fields and parser operations are documented in
[Configuration](docs/configuration.md) and [Operations](docs/operations.md).

### Native API

Run PostgreSQL in Docker and the API on the host:

```bash
docker compose up -d postgres
uv sync
DLIGHTRAG_CORPUS__SIDECARS__MINERU__LOCAL_ENDPOINT=http://127.0.0.1:8210 \
  uv run dlightrag-api
```

The checked-in config is Docker-first, so a native process overrides the parser
host alias with loopback. Native managed inputs live under
`./dlightrag_storage/inputs/<workspace>`.

## Use DlightRAG

### Web

The Web UI supports workspace and file management, durable Fast and Research
conversations, answer attachments, citations, source highlights, child-agent
status, and typed Answer Artifacts. Research publishes only workspace roots it
explicitly attaches; prose links alone never publish files. English, Chinese,
and automatic browser language modes are available under Settings.

### REST

Ingestion, Retrieval, and Answer create durable Runs and return `202 Accepted`.
Corpus mutations require a stable idempotency key.

```bash
INGEST_RUN=$(curl -sS -X POST http://localhost:8100/runs/corpus/ingest \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: ingest-report-1" \
  -d '{"source_type":"local","path":"report.pdf"}' | jq -r .run_id)
curl "http://localhost:8100/runs/$INGEST_RUN"

RETRIEVAL_RUN=$(curl -sS -X POST http://localhost:8100/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the key findings?"}' | jq -r .run_id)
curl "http://localhost:8100/runs/$RETRIEVAL_RUN"

RUN=$(curl -sS -X POST http://localhost:8100/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the key findings?"}' | jq -r .run_id)
curl -N "http://localhost:8100/runs/$RUN/events"
curl "http://localhost:8100/runs/$RUN"
```

See [Interfaces](docs/interfaces.md) for requests, responses, pagination, SSE,
attachments, citations, and all transport contracts.

### MCP

For a local stdio client:

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

The Compose stack also exposes streamable HTTP on port 8101. MCP supports
durable Retrieval and Answer Runs, steering, follow-up/fork, child status,
corpus administration, and capability discovery. The authoritative tool list
is in [Interfaces](docs/interfaces.md#mcp-server).

### Python

```bash
uv add dlightrag
```

Create an application with `create_application(config)`, use
`application.corpus_mutations` for durable corpus writes,
`application.retrieval` for durable Retrieval, and `application.answers` for
durable Answers, then call
`application.aclose()`. Complete typed examples are in
[Interfaces](docs/interfaces.md#in-process-application).

## Core Concepts

| Concept | Meaning | Reference |
|---|---|---|
| Workspace | Isolation unit for indexed data, metadata, files, and queries | [Domain language](docs/domain-language.md) |
| Ingestion | One durable contract for local files, uploads, object storage, URLs, and SDK sources | [Interfaces](docs/interfaces.md#ingestion) |
| Retrieval | One durable Query-lane Run returning LightRAG mix plus metadata, BM25, visual fusion, and rerank evidence | [Retrieval and Answer](docs/retrieval-answer.md) |
| Run | Common durable lifecycle for Retrieval, Answer, and Corpus Mutation across REST, MCP, Web, Python, and evaluation | [RunRuntime](docs/durable-answer-runs.md) |
| Answer Run | A Query-lane Run that resolves Fast or Research and generates an Answer | [Retrieval and Answer](docs/retrieval-answer.md#answer-orchestration) |
| Resource | Request-local attachment read deterministically or inspected visually on demand | [Retrieval and Answer](docs/retrieval-answer.md#answer-attachments-and-resources) |
| Published Artifact | Owner-visible Research output authorized by a settled root attachment and validated at publication | [Domain language](docs/domain-language.md) |
| Source | Durable provenance and download contract for an ingested document | [Interfaces](docs/interfaces.md#sources) |

## Security

Loopback development can use `access.auth_mode: none`. Shared deployments should
use a bearer token or externally issued JWT; JWKS and claim-based workspace/action
rules are supported. DlightRAG does not issue tokens or replace an ingress WAF,
rate limiter, TLS terminator, or identity provider. See
[Security](docs/security.md).

## Development

```bash
uv sync
npm --prefix frontend ci
make hooks
make ci          # lint, security, format, types, architecture, frontend, unit
make ci-full     # plus integration tests
make ci-e2e      # plus E2E smoke
```

Use [Operations](docs/operations.md) for reset, rebuild, parser, Langfuse, and
maintenance runbooks. RAGAS evaluation is documented in
[Evaluation](docs/evaluation.md).

## Documentation

| Document | Owns |
|---|---|
| [Architecture](docs/architecture.md) | Runtime ownership, flows, storage topology, layering |
| [Domain Language](docs/domain-language.md) | Canonical product vocabulary |
| [Configuration](docs/configuration.md) | Configuration precedence, fields, defaults, examples |
| [Interfaces](docs/interfaces.md) | Python, REST, MCP, and Web contracts |
| [Retrieval and Answer](docs/retrieval-answer.md) | Retrieval, fusion, rerank, packing, citations, highlights |
| [RunRuntime and Durable Execution](docs/durable-answer-runs.md) | Common Query and Corpus Mutation state machine, leases, events, recovery, retention |
| [RunRuntime and Scaling Target](docs/run-runtime-and-scaling-target.md) | Accepted workload model, lane bounds, lifecycle guarantees, and ownership boundary |
| [RunRuntime Slice 6 Validation](docs/validation/run-runtime-slice-6.md) | Failure matrix, reproducible commands, 10k control-plane evidence, accepted bounds, limitations |
| [Security](docs/security.md) | Authentication, authorization, ingress and content boundaries |
| [PostgreSQL](docs/postgresql.md) | PostgreSQL requirements, schema ownership, tuning |
| [Operations](docs/operations.md) | Executable runbooks and recovery workflows |
| [Evaluation](docs/evaluation.md) | RAGAS workflow |
| [Web Theme Design](docs/web-theme-design.md) | Web appearance and interaction decisions |

Plans, ADRs, and research notes under `docs/` are historical design evidence,
not required reading for operating DlightRAG.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Built by HanlianLyu. Contributions welcome.
