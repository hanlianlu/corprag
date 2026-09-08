# Configuration

This document owns configuration precedence, public settings, defaults, and
examples. Runtime design belongs in [Architecture](architecture.md), public
payloads in [Interfaces](interfaces.md), security policy in
[Security](security.md), and executable procedures in [Operations](operations.md).

Root [`config.yaml`](../config.yaml) is the canonical home for non-secret
application behavior. Advanced fields remain available through explicit YAML
additions; nested `DLIGHTRAG_*` variables are an override mechanism, not a
second catalogue of normal application settings.

```text
constructor args > environment variables > .env > config.yaml > code defaults
```

Precedence determines the effective value; it does not assign ownership. Within
one deployment, configure a setting in one place rather than relying on a
higher-precedence source to mask a duplicate lower-precedence value. Nested
environment variables follow the settings path with `__` separators:

```bash
DLIGHTRAG_MODELS__EMBEDDING__API_KEY=...
DLIGHTRAG_STORAGE__POSTGRES__HOST=postgres
```

DlightRAG has nine top-level sections: `deployment`, `storage`, `models`,
`corpus`, `answer`, `runtime`, `access`, `interfaces`, and `observability`. Removed legacy
flat names are rejected rather than aliased. The ownership decision is recorded
in [ADR 0006](adr/0006-configuration-ownership-and-deployment-bindings.md).

## Configuration Ownership

Use the first matching rule:

| Concern | Canonical owner | Examples |
|---|---|---|
| Non-secret product behavior and integration choices | `config.yaml` | model selection, parser provider and endpoint, workspace identity, retrieval breadth, Answer policy, auth mode, access rules, observability behavior |
| Credentials | `.env` locally; an orchestrator Secret in production | provider API keys, PostgreSQL password, JWT verification key |
| Facts created by deployment topology | Compose or another deployment manifest | Service DNS, container listener/transport binding, volumes, probes, resource limits, PostgreSQL server tuning |
| Stable low-level mechanics | Code defaults until measurement requires an explicit override | retry/backoff, cache bounds, parser polling, vector thresholds |

Environment overrides are appropriate for Secrets, topology bindings, and
short-lived operational exceptions. Do not copy ordinary model, retrieval, or
Answer policy from `config.yaml` into Compose or Kubernetes manifests.

For filesystem settings, prefer mounting storage at the configured or default
application path. Override the application path only when the deployment cannot
align its mount. For infrastructure endpoints such as PostgreSQL Service DNS,
the deployment manifest may own the typed setting; do not also place that value
in its mounted `config.yaml`.

## What Belongs In `config.yaml`

Keep model/provider settings, parser sidecars, workspace identity,
high-level concurrency, retrieval breadth, Answer policy, auth mode, access
rules, and non-secret observability settings in YAML. Keep credentials out of
YAML, and keep container topology out of it.

Usually leave these at code defaults unless measurement proves otherwise:

- storage backend literals and raw LightRAG parser rules
- retry/backoff, queue, HNSW, image-compression, and parser polling internals
- per-stage ingestion workers that already match LightRAG defaults
- BM25 index signatures, RRF constants, and exact-vector thresholds
- thumbnail, highlight-cache, and URL-signing internals

## Container And Kubernetes Contract

The process reads `config.yaml` from its current working directory. The bundled
image uses `/app`, so Compose grants the canonical file through its top-level
`configs` resource at `/app/config.yaml`; this keeps configuration distinct from
data volumes. A Kubernetes ConfigMap should mount the same file at the same
path. Compose/Kubernetes mount the file but do not interpret its fields or
derive volume topology from them. Inject only credentials from Secrets and the
minimal topology bindings required by that workload. In particular:

- mount corpus storage at `/app/dlightrag_storage`, matching the checked-in
  `deployment.working_dir: ./dlightrag_storage`;
- mount the shared Agent Workspace, when enabled, at
  `/home/app/.dlightrag/agent_workspaces`, matching the application default;
- inject the PostgreSQL Service DNS through
  `DLIGHTRAG_STORAGE__POSTGRES__HOST` unless the mounted YAML owns it;
- bind container listeners and select the MCP network transport in the workload
  manifest because these choices vary by process role;
- keep a development-only insecure-listener waiver beside the manifest's
  loopback-only host publication; production deployments configure an auth mode
  in YAML instead of inheriting that waiver;
- keep ports, Services, volumes, probes, resource requests/limits, and database
  server tuning entirely outside DlightRAG application configuration.

This contract lets an operator understand product behavior from one typed YAML
file while reviewing credentials and infrastructure through their native
orchestrator surfaces.

## Parser Sidecars

Configure exactly one `mineru` or `docling` block. DlightRAG derives the
LightRAG wildcard parser rule. With neither block, the code default is local
MinerU at `http://127.0.0.1:8210`; if both exist, MinerU wins only for backward
compatibility. Parser changes affect new parses, not existing indexed data.

### MinerU (default)

The checked-in Docker configuration reaches a host service through
`host.docker.internal`:

```yaml
corpus:
  sidecars:
    mineru:
      api_mode: local
      local_endpoint: http://host.docker.internal:8210
      language: ch
      backend: hybrid-engine
```

| Field | Default | Notes |
|---|---|---|
| `api_mode` | `local` | `local` or `official` |
| `local_endpoint` | `http://127.0.0.1:8210` | Use the Docker host alias from containers |
| `official_endpoint` | `https://mineru.net` | Used only in official mode |
| `api_token` | unset | Prefer the nested environment variable |
| `language` | `ch` | OCR hint, separate from extraction output language |
| `backend` | `hybrid-engine` | `pipeline`, `vlm-engine`, or `hybrid-engine` |
| `poll_interval_seconds` | `5` | Parser polling interval |
| `max_polls` | `1440` | Two-hour default polling window |

`pipeline` avoids VLM transcription artifacts but handles complex layouts less
well. `MINERU_HYBRID_EFFORT=high` improves dense multi-panel figure detection at
roughly five times the parse time. DlightRAG leaves MinerU image analysis off;
its own VLM sidecar describes extracted figures instead.

The installer supports MinerU 3.4.5 through the reviewed 3.x API range. Manage
it with `make mineru-install` and `make mineru-service-*`; see
[Parser Services](operations.md#parser-services).

### Docling

```yaml
corpus:
  sidecars:
    docling:
      endpoint: http://host.docker.internal:5001
      do_formula_enrichment: true
      force_ocr: true
      code_formula_preset: granite_docling
```

| Field | Default | Notes |
|---|---|---|
| `endpoint` | `http://127.0.0.1:5001` | External docling-serve endpoint |
| `do_formula_enrichment` | `true` | Turning it off drops PDF formula text |
| `force_ocr` | `true` | Set `false` for reliable born-digital PDF text layers |
| `code_formula_preset` | `granite_docling` | MPS preset; use YAML `null` on CUDA/XPU/CPU |
| `poll_interval_seconds` | `5` | Parser polling interval |
| `max_polls` | `1440` | Two-hour default polling window |

DlightRAG always requests PDF heading hierarchy; this requires docling-serve
1.30.0+ and docling-jobkit 3.3.0+. The Docling service owns its OCR engine and
languages. The optional Compose CPU profile uses `http://docling:5001` with
`code_formula_preset: null`.

Both parser services need an HTTP keep-alive longer than the five-second poll
interval. DlightRAG's MinerU launcher and stock docling-serve use 60 seconds.

### Figure VLM

```yaml
corpus:
  sidecars:
    vlm:
      enabled: true
      max_image_bytes: 5242880
      min_image_pixel: 80
      surrounding_leading_max_tokens: 256
      surrounding_trailing_max_tokens: 256
```

| Field | Default | Meaning |
|---|---|---|
| `corpus.sidecars.vlm.enabled` | `true` | Analyze parser-extracted figures |
| `corpus.sidecars.vlm.max_image_bytes` | `5242880` | Maximum source image bytes |
| `corpus.sidecars.vlm.min_image_pixel` | `80` | Minimum image side accepted |
| `corpus.sidecars.vlm.surrounding_leading_max_tokens` | `256` | Leading text supplied to figure analysis; `null` disables the cap |
| `corpus.sidecars.vlm.surrounding_trailing_max_tokens` | `256` | Trailing text cap; `null` disables it |
| `corpus.parser.chunk_options` | `{}` | Advanced LightRAG parser/chunk keyword arguments |
| `corpus.extraction.use_json` | `true` | Request structured extraction |
| `corpus.extraction.language` | `English` | Generated entity/relation and keyword language |
| `corpus.extraction.entity_type_prompt_file` | unset | `.yml`/`.yaml` filename under `prompts/entity_type/` |

Extraction language does not configure OCR or translate existing graph data.

## Embeddings

One embedding space is shared by ingestion and every retrieval leg. Never mix
models, dimensions, or vector spaces in one workspace; use a new workspace or a
complete offline rebuild.

### Providers

| `provider` | Typical model | Visual support | Dimension wire field |
|---|---|---|---|
| `openai` | `text-embedding-3-large` | Text only | `dimensions` for supported models |
| `openai_compatible` | Deployment-defined | Text only | Never sent; response is validated |
| `voyage` | `voyage-multimodal-3.5` | Native text+image fusion | `output_dimension` |
| `gemini` | `gemini-embedding-2` | Native content aggregation | `outputDimensionality` |
| `jina` | `jina-embeddings-v4` | Native text+image fusion | `dimensions` |
| `cohere` | `embed-v4.0` | Native mixed-input fusion | `output_dimension` |
| `azure_cohere` | `Cohere-embed-v4` | Native mixed-input fusion | `output_dimension` |

Unknown model names resolve conservatively to text-only operation.
`openai_compatible` does not invent vendor-specific image, dimension, or task
fields. Azure OpenAI v1 roots and Azure Cohere deployment scoring roots are
supported by their corresponding adapters.

### Fields

| Field | Default | Meaning |
|---|---|---|
| `provider` | `voyage` | Protocol adapter |
| `model` | `voyage-multimodal-3.5` | Exact model or deployment identifier |
| `api_key` | unset | Put secrets in `.env` |
| `base_url` | provider default | Protocol root or accepted complete endpoint |
| `dim` | `1024` | Vector/schema dimension; every response is validated |
| `max_token_size` | `8192` | LightRAG embedding-content truncation budget |
| `input_modality` | `auto` | `auto`, `text`, or `multimodal` |
| `startup_probe` | `true` | Verify configured visual paths |
| `timeout` | `120` | Request timeout in seconds |
| `max_concurrency` | `16` | Process-wide scheduler admission |
| `batch_size` | `64` | LightRAG embedding batch size |

`auto` enables native fused document vectors and image-query retrieval for
known multimodal models. `text` disables both image paths. `multimodal` requires
them and makes probe failure fatal. Fused output replaces the canonical chunk
vector; it never creates a second visual document vector.

```yaml
models:
  embedding:
    provider: voyage
    model: voyage-multimodal-3.5
    base_url: https://api.voyageai.com/v1
    dim: 1024
    input_modality: auto
    startup_probe: true
```

For a host-side local endpoint used from Compose, replace `127.0.0.1` with
`host.docker.internal`. See [Offline Vector Storage Rebuild](operations.md#offline-vector-storage-rebuild)
before changing an existing workspace's vector space.

## Chat Models

`provider` identifies the SDK/protocol, not the vendor:

| `provider` | Transport | Typical endpoints |
|---|---|---|
| `openai` | Chat Completions | OpenAI, DeepSeek, OpenRouter, Azure OpenAI, vLLM, Ollama, other compatible APIs |
| `anthropic` | Anthropic native SDK | Claude |
| `gemini` | Google GenAI SDK | Gemini |

Select OpenAI-compatible vendors with `base_url`; unknown provider names are
rejected.

### Role Configuration

```yaml
models:
  chat:
    default:
      provider: openai
      model: openai/gpt-5.6
      base_url: https://openrouter.ai/api/v1
    roles:
      extract:
        provider: openai
        model: deepseek-v4-flash
        base_url: https://api.deepseek.com
```

A role override is complete, not a partial merge. Missing or incomplete roles
fall back to `models.chat.default` as a whole. The same fields apply to the
default and each `extract`, `keyword`, `query`, or `vlm` override:

| Field | Default | Meaning |
|---|---|---|
| `provider` | `openai` | `openai`, `anthropic`, or `gemini` protocol |
| `model` | required | Exact model/deployment ID |
| `api_key` | unset | Endpoint credential |
| `base_url` | provider default | Optional API root |
| `structured_output` | `auto` | `auto`, `json_schema`, or `json_object` |
| `temperature` | unset | Nonnegative provider temperature |
| `timeout` | `240` | Request timeout seconds |
| `max_retries` | `3` | Provider retry count |
| `reasoning` | unset | Typed reasoning level |
| `agentic_reasoning` | inherits `reasoning` | Research-specific level; explicit `null` disables |
| `model_kwargs` | `{}` | Provider-specific ordinary options |
| `agentic_model_kwargs` | `{}` | Shallow Research overlay |

`models.max_concurrency` defaults to `16` and limits process-wide AI-provider
requests.

### Model Catalogue And Reasoning

Model profiles are keyed by normalized `provider`, exact `model`, and normalized
`base_url`. Resolution order is:

```text
PostgreSQL runtime overlay > models.catalogue > built-in catalogue > fallback
```

A profile defines context/input/output limits, image support, and optionally a
reasoning format with all seven typed levels: `off`, `minimal`, `low`, `medium`,
`high`, `xhigh`, and `max`. Unsupported levels map to `null`; non-off requests
clamp to the nearest supported level. Uncatalogued endpoints use best-effort
protocol mapping and surface provider rejection.

```yaml
models:
  catalogue:
    - provider: openai
      model: vendor/new-model
      base_url: https://api.vendor.example/v1
      profile:
        context_window_tokens: 262144
        max_input_tokens: null
        max_output_tokens: 32768
        supports_images: true
        reasoning:
          format: openai
          levels:
            off: none
            minimal: null
            low: low
            medium: medium
            high: high
            xhigh: null
            max: null
  chat:
    default:
      reasoning: max
```

Startup catalogue changes require restart. Runtime overlay operations and
revision rules are in [Interfaces](interfaces.md#model-catalogue-and-profile-memory).

`agentic_reasoning` inherits `reasoning`. When typed reasoning is configured,
raw provider reasoning keys in `model_kwargs` are rejected to keep one owner for
translation. `agentic_model_kwargs` is a shallow overlay for Research calls.

### Structured Output

`structured_output` defaults to `auto`: strict schema output where supported,
with an OpenAI-compatible fallback to `json_object`. Set `json_object` only for
an endpoint known not to support strict schemas. Anthropic native supports
`auto`/`json_schema`, not the lower-confidence `json_object` mode.

```yaml
models:
  chat:
    roles:
      extract:
        provider: openai
        model: deepseek-v4-flash
        base_url: https://api.deepseek.com
        structured_output: json_object
```

## Reranking

Accepted strategy literals are:

| Strategy | Transport | Image policy |
|---|---|---|
| `chat_llm_reranker` | Configured chat endpoint | `auto` uses its vision probe |
| `jina_reranker` | Jina `/v1/rerank` | Explicit multimodal supported by suitable models |
| `aliyun_reranker` | Alibaba Model Studio | Explicit multimodal supported by Qwen VL rerank |
| `local_reranker` | Standard `{model,query,documents,top_n}` `/rerank` | Endpoint-defined |
| `voyage_reranker` | Voyage `/v1/rerank` | Text only |
| `cohere_reranker` | Cohere `/v2/rerank` | Text only |
| `azure_cohere` | Azure Cohere rerank | Text only |

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Enable final fused-candidate reranking |
| `strategy` | `chat_llm_reranker` | One literal above |
| `provider` | unset | Independent chat-rerank provider |
| `model`, `api_key`, `base_url` | unset | Rerank endpoint |
| `input_modality` | `auto` | `auto`, `text`, or `multimodal` |
| `score_threshold` | unset | Hard nonnegative post-rerank cutoff |
| `max_concurrency` | `8` | Rerank request admission |
| `batch_size` | `8` | Candidate batch/list size |
| `temperature` | unset | Chat reranker temperature |
| `model_kwargs` | `{}` | Provider-specific options |

Each HTTP strategy validates its required `api_key` and/or `base_url`; invalid
configuration fails startup rather than changing strategy. Text-only strategies
reject explicit multimodal mode.

## Remote Sources

`source_uri` is stable provenance. `download_uri` is the durable S3, Azure, or
queryless public HTTPS locator used when no local copy is retained.

```yaml
corpus:
  ingestion:
    retain_remote_source_files: false
    url_max_bytes: 104857600
    url_private_host_allowlist: []
  sources:
    blob_connection_string: null
    azure_sas_expiry: 3600
    s3_presign_expiry: 3600
    s3_region: null
```

Signed URLs with query or fragment tokens are not durable locators. Retain the
file or provide a separate queryless `download_uri`. Non-retained custom
`AsyncDataSource` connectors must provide the locator directly or through
`download_uri_for_key`; invalid contracts fail before parsing.

URL ingest rejects private hosts unless allowlisted, HTTPS-to-HTTP redirects,
and oversized downloads. `blob_connection_string` is the Azure credential;
`azure_sas_expiry` and `s3_presign_expiry` bound projected URLs; `s3_region`
overrides SDK discovery. Prefer
`DLIGHTRAG_CORPUS__SOURCES__BLOB_CONNECTION_STRING` for the secret. S3 uses the
standard AWS credential chain. Deleting DlightRAG data never deletes provider
objects. See [Sources](interfaces.md#sources).

## PostgreSQL And Process Role

DlightRAG requires PostgreSQL 18 for KV, graph, document status, BM25, metadata,
and Operational State. LightRAG's deployment-static storage names normally stay
at these exact defaults:

```yaml
storage:
  lightrag:
    vector_storage: PGVectorStorage
    graph_storage: PGTableGraphStorage
    kv_storage: PGKVStorage
    doc_status_storage: PGDocStatusStorage
  postgres:
    pool_min_size: 2
    pool_max_size: 16
    lightrag_pool_max_size: 16
```

The domain and LightRAG pools are separate; multiply their sum by process count
and stay below PostgreSQL `max_connections`. `kv_storage`, `graph_storage`, and
`doc_status_storage` are fixed to the values above. The only supported vector
alternative is LightRAG's `MilvusVectorDBStorage`; `PGGraphStorage`, AGE, Qdrant,
and other combinations are rejected rather than substituted.

For a writer deployment using Milvus or a Milvus-compatible Zilliz endpoint,
install the deterministic client extra (`dlightrag[milvus]`) and select:

```yaml
storage:
  lightrag:
    vector_storage: MilvusVectorDBStorage
    milvus_uri: https://example-compatible-endpoint
    milvus_db_name: default
```

Keep `milvus_token` in `DLIGHTRAG_STORAGE__LIGHTRAG__MILVUS_TOKEN`. Resolved
DlightRAG values overwrite inherited `MILVUS_URI`, `MILVUS_TOKEN`, and
`MILVUS_DB_NAME`; an unset optional binding leaves the corresponding upstream
environment behavior untouched. DlightRAG never connects to Milvus for health
checks and never creates, copies, or drops Milvus infrastructure itself. Zilliz
uses `MilvusVectorDBStorage`, not a separate storage class. Reader processes
must use `PGVectorStorage`: LightRAG 1.5.7 has no public nonmutating reader attach
for an external vector adapter.

| Field | Default | Meaning |
|---|---|---|
| `deployment.service_role` | `writer` | `writer` or `reader` |
| `deployment.workspace` | `default` | Default workspace |
| `deployment.working_dir` | `./dlightrag_storage` | Corpus/input/artifact root; resolved absolute |
| `storage.lightrag.vector_storage` | `PGVectorStorage` | `PGVectorStorage` or explicit `MilvusVectorDBStorage` |
| `storage.lightrag.milvus_uri` | unset | Optional `MILVUS_URI` bridge; may be a Milvus-compatible Zilliz URI |
| `storage.lightrag.milvus_token` | unset | Optional secret `MILVUS_TOKEN` bridge |
| `storage.lightrag.milvus_db_name` | unset | Optional `MILVUS_DB_NAME` bridge |
| `storage.postgres.host` | `localhost` | PostgreSQL host |
| `storage.postgres.port` | `5432` | PostgreSQL port |
| `storage.postgres.user` | `dlightrag` | Login role |
| `storage.postgres.password` | `dlightrag` | Password; override in `.env` |
| `storage.postgres.database` | `dlightrag` | Database |
| `storage.postgres.ssl_mode` | unset | `disable`, `allow`, `prefer`, `require`, `verify-ca`, or `verify-full` |
| `ssl_cert`, `ssl_key`, `ssl_root_cert`, `ssl_crl` | unset | TLS file paths |
| `pool_min_size`, `pool_max_size` | `2`, `16` | Domain pool bounds |
| `lightrag_pool_max_size` | `16` | Separate LightRAG pool maximum |
| `command_timeout`, `acquire_timeout` | `60`, `30` | SQL command/acquire seconds |
| `session_settings` | `{}` | asyncpg session parameters |
| `statement_cache_size` | driver default | Prepared-statement cache size |
| `connection_retries` | `10` | Startup connection attempts |
| `connection_retry_backoff` / `_max` | `3` / `30` | Retry delay/cap seconds |
| `pool_close_timeout` | `5` | Shutdown close wait seconds |

- `writer` provisions schema, ingests, and serves every API.
- `reader` serves retrieval and durable answers but rejects corpus mutations.
  It validates pre-migrated schema and performs no LightRAG DDL.

Both roles use the same primary endpoint. Run writer migrations before readers.
Multi-host deployments need one shared POSIX `deployment.working_dir` mounted at
the same absolute path. Production sizing, SSL, indexes, and role details are in
[PostgreSQL](postgresql.md).

## RunRuntime, Ingestion Concurrency, And Limits

| Field | Default | Scope |
|---|---|---|
| `models.max_concurrency` | `16` | All provider requests in one process |
| `runtime.query.worker_concurrency` | `16` | Query-lane runs per process |
| `runtime.query.max_nonterminal_runs` | `30000` | Query-lane deployment-wide nonterminal admission limit |
| `runtime.corpus_mutation.worker_concurrency` | `2` | Corpus Mutation Run workers per writer process |
| `runtime.corpus_mutation.max_nonterminal_runs` | `1000` | Corpus Mutation deployment-wide nonterminal admission limit |
| `corpus.ingestion.pipeline.max_concurrency` | `16` | One workspace's LightRAG pipeline |
| `models.embedding.max_concurrency` | `16` | Embedding calls |
| `models.embedding.batch_size` | `64` | LightRAG embedding batch size |
| `corpus.ingestion.chunk_token_size` | `2000` | LightRAG chunk size |
| `corpus.ingestion.replace_default` | `false` | Default replacement policy |
| `corpus.ingestion.retain_remote_source_files` | `false` | Default remote-byte retention |
| `corpus.ingestion.max_upload_bytes` | `104857600` | One ingest file |
| `corpus.ingestion.url_max_bytes` | `104857600` | One URL download |
| `corpus.ingestion.url_private_host_allowlist` | `[]` | Explicit private URL hosts |
| `interfaces.max_upload_size_mb` | `512` | General multipart receive cap |

Advanced stage defaults:

```yaml
corpus:
  ingestion:
    pipeline:
      max_parallel_insert: 3
      max_parallel_parse_native: 5
      max_parallel_parse_mineru: 2
      max_parallel_parse_docling: 2
      max_parallel_analyze: 5
      queue_size_parse: 20
      queue_size_analyze: 100
      queue_size_insert: 4
```

Embedding batches split automatically at provider input-count, token, and
inline-image limits while preserving order.

## Retrieval

```yaml
corpus:
  retrieval:
    top_k: 40
    chunk_top_k: 20
    direct_visual_top_k: 20
    timeout: 300
    bm25_enabled: true
```

`top_k` controls graph/entity breadth; `chunk_top_k` controls text/visual chunk
candidates. BM25 candidate breadth follows the chunk budget. `/answer` packs
evidence against the query model's remaining input capacity.

Advanced fields:

| Field | Default | Meaning |
|---|---|---|
| `timeout` | `300` | Claimed top-level Retrieval planning/search seconds; excludes queue residence and does not wrap Answer Runs |
| `bm25_k1` | `1.2` | BM25 term-frequency saturation |
| `bm25_b` | `0.75` | BM25 length normalization |
| `bm25_profiles` | built-in language set | pg_textsearch index signatures and language labels |
| `rrf_k` | `60` | Reciprocal-rank fusion constant |
| `metadata_filter_exact_vector_threshold` | `8192` | Exact vector scoring cutoff for a filtered candidate set |
| `max_entity_tokens` | `6000` | KG entity context ceiling |
| `max_relation_tokens` | `8000` | KG relation context ceiling |
| `max_total_tokens` | `40000` | Total LightRAG context ceiling |
| `kg_chunk_pick_method` | `VECTOR` | `VECTOR` or `WEIGHT` |
| `kg_entity_types` | `[]` | Empty uses LightRAG's general taxonomy |

Enabling BM25 for existing data or changing profiles requires
[Workspace BM25 Rebuild](operations.md#workspace-bm25-rebuild). Algorithm and budget semantics live in
[Retrieval and Answer](retrieval-answer.md).

Workspace partition promotion is disabled until a benchmark supplies either
`corpus.promotion.doc_threshold` or `chunk_threshold`. Advanced worker defaults
are `lease_seconds: 1800`, `retry_backoff_seconds: 600`, and
`claim_poll_seconds: 5.0`. Visual routes use
`corpus.visual_assets.thumb_max_px: 300` and `thumb_cache_size: 256`.

## RunRuntime Lanes And Retention

```yaml
runtime:
  query:
    worker_concurrency: 16
    max_nonterminal_runs: 30000
  corpus_mutation:
    worker_concurrency: 2
    max_nonterminal_runs: 1000
  run_retention_days: 365
```

One RunRuntime schedules top-level Retrieval and Answer Runs on the Query Lane
and all ingest, replace, exact delete, retry, and reset Runs on the Corpus
Mutation Lane. Each lane has a per-process worker bound and a deployment-wide
nonterminal admission limit. Deployment configuration owns process count and
therefore total active Run capacity. The [Slice 6 validation
report](validation/run-runtime-slice-6.md) records the 10k-client fake-executor
campaign, admission-limit rejection, single-process worker occupancy,
measurements, and limitations.

`run_retention_days` is the per-Run retention selection for terminal Answer
Runs, their event logs, linked Web turns, and unreferenced Run blobs; the default
is 365 days. Top-level Retrieval and Corpus Mutation Runs each use a fixed
seven-day selection. Nonterminal Runs are not retention-pruned, and Conversation
rows do not extend model history. Full lifecycle rules are in
[RunRuntime and durable query execution](durable-answer-runs.md).

## Answer Generation And Attachments

```yaml
answer:
  generation:
    max_images: 12
    max_attachments: 6
    max_attachment_bytes: 104857600
    max_total_attachment_bytes: 134217728
    image_max_bytes: 3000000
    image_max_total_bytes: 24000000
    image_max_px: 1536
    image_max_pixels: 40000000
    image_min_px: 1024
    image_quality: 89
    image_min_quality: 79
```

Attachments are run-scoped Resources. Full bytes do not enter model context;
text is decoded/converted and figures are inspected on demand. `query_images`
is a separate retrieve-only path limited to three current images. The final
answer image count is clamped to the query model's discovered capability.

## Research Agent

```yaml
answer:
  agent:
    execution_environment: trust   # disabled | trust | sandbox
    workspace_root: null
    skills_root: null              # absolute path; null → ~/.dlightrag/skills
    owner_skills_root: null        # absolute path; null → ~/.dlightrag/owner_skills
    outbound_mcp: []
    publication:
      max_artifacts: 20
      max_file_bytes: 31457280
      max_total_bytes: 104857600
      workspace_max_bytes: 1073741824
      preview_image_max_pixels: 16000000
      preview_image_max_edge: 4096
      original_image_max_pixels: 64000000
      original_image_max_edge: 8000
      active_html_max_bytes: 20971520
  conversations:
    active_html_preview_enabled: true
```

`trust` runs rooted tools as the service user; Bash still has process-level
filesystem and network access. `disabled` removes path, Bash, spill, and
`attach_artifact`. That publication-authority tool is available only to the
parent Research Session; Fast and Child Sessions cannot authorize publication.
This distribution has no sandbox backend, so selecting `sandbox` fails rather
than downgrading.

An explicit workspace root must be absolute, must not overlap
`deployment.working_dir`, and must be the same shared RWX path on every worker.
Published artifacts fail whole when over budget; they are not truncated.
Interactive HTML is separately opt-in and isolated by the Web artifact boundary.
See [Security](security.md#answer-artifact-browser-boundary).

Outbound MCP endpoints are deployment-allowlisted. Each entry has `name`,
`transport`, and a nonempty unique `tools` list. `stdio` requires `command` and
optional `args`; `streamable-http` requires `url`. The other endpoint kind is
forbidden in each mode.

```yaml
answer:
  agent:
    outbound_mcp:
      - name: analytics
        transport: streamable-http
        url: https://mcp.example.com/mcp
        tools: [lookup_metric]
```

There is no endpoint discovery, marketplace, OAuth service, or plugin runtime.
Research discovers Skill metadata from two tiers and loads content on demand.
The global root is `answer.agent.skills_root`, defaulting to
`~/.dlightrag/skills`; it is operator-provisioned and read-only for the answer
agent. The bundled Compose stack keeps `skills_root: null`, mounts the
operator's `${COMPOSE_GLOBAL_SKILLS_DIR:-$HOME/.dlightrag/skills}` at that default
container path, and refuses to create a missing host source as root. The setup
wizard prepares the directory; manual operators must create it before
`docker compose up`.

The per-owner root is `answer.agent.owner_skills_root`, defaulting to
`~/.dlightrag/owner_skills`; users write their own skills only through the
validated `publish_skill`/`delete_skill` tools, bounded by a 20-skill / 20MiB
owner quota. Owner names shadow global names for that owner. Each worker must
see the same shared paths. A reference creator workflow ships at
`examples/skills/skill-creator/SKILL.md`; copy it into the global skills root
to guide conversational skill creation for all users.

## Public Web Sources

```yaml
answer:
  web_sources:
    exa:
      api_key: null     # DLIGHTRAG_ANSWER__WEB_SOURCES__EXA__API_KEY
    tavily:
      api_key: null     # DLIGHTRAG_ANSWER__WEB_SOURCES__TAVILY__API_KEY
    search_providers: null
    extract_providers: null
```

Exa and Tavily are optional provider adapters. A `null` provider order derives
an Exa-then-Tavily chain from configured keys; `[]` disables that operation.
Set each list explicitly to give Search and Extract independent failover order.
Every provider named in a list must have a key. The setup wizard can configure
Exa, Tavily, both with one shared order, or independent Search/Extract orders.

Research exposes provider-neutral `search_web` controls for result count,
domains, date range, and `fast`/`balanced`/`deep` effort. Its `read` tool can
also admit one anonymous public HTTP(S) URL with only `User-Agent`, `Accept`, and
`Accept-Language` presentation options. DlightRAG validates every redirect and
resolved address against the shared SSRF policy, pins validated DNS targets,
fetches direct HTTP first, and uses the Extract chain only when direct retrieval
fails or produces no usable text. It supplies no cookies or browser automation.

## Citations And Highlights

Citation validation is always enabled. Semantic highlights default on for Web
Inspector Sources and off for other answer callers unless requested.

```yaml
answer:
  citations:
    highlights:
      enabled: true
      timeout: 10.0
      max_concurrency: 8
      batch_size: 8
      max_input_chars: 4096
      cache_size: 500
```

Set `enabled: false` to disable highlight extraction on every interface. Public
citation shapes are in [Interfaces](interfaces.md#citations).

## Access And Interfaces

Code defaults bind listeners to loopback with no auth; MCP defaults to `stdio`.
The checked-in Compose config explicitly selects `streamable-http` on port 8101.

| Field | Default | Meaning |
|---|---|---|
| `interfaces.api.host`, `.port` | `127.0.0.1`, `8100` | REST/Web bind |
| `interfaces.mcp.transport` | `stdio` | `stdio` or `streamable-http` |
| `interfaces.mcp.host`, `.port` | `127.0.0.1`, `8101` | HTTP MCP bind |
| `interfaces.mcp.allowed_hosts` | local hosts | Host-header allowlist |
| `interfaces.mcp.allowed_origins` | local origins | Origin allowlist |
| `interfaces.mcp.resource_server_url` | unset | Public RFC 9728 resource URL |
| `interfaces.max_upload_size_mb` | `512` | General multipart receive cap |
| `access.auth_mode` | `none` | `none`, `simple`, or `jwt` |
| `access.api_token` | unset | Shared token for `simple` |
| `access.allow_insecure_no_auth` | `false` | Permit non-loopback no-auth bind |
| `access.jwt_verification_key` | unset | Static HMAC/public key |
| `access.jwt_jwks_url` | unset | Rotating JWKS endpoint |
| `access.jwt_issuer`, `.jwt_audience` | unset | Expected claims |
| `access.jwt_algorithm` | `HS256` | Accepted signing algorithm |
| `access.cors_allow_origins` | `["*"]` | REST browser origins |
| `access.web_identity` | disabled | Edge, issuer, audience, optional JWKS |
| `access.control.mode` | `allow_all` | `allow_all` or `jwt_claims` |
| `access.control.rules` | `[]` | Claim/workspace/action mappings |

Do not expose listeners without auth and ingress protection. Security semantics
are in [Security](security.md); payload contracts are in
[Interfaces](interfaces.md).

CLI and evaluation clients use `DLIGHTRAG_API_URL`, optional
`DLIGHTRAG_API_TOKEN`, and `DLIGHTRAG_CLIENT_TIMEOUT` (default 120 seconds).

## Observability

Tracing activates only when both Langfuse keys are set in `.env`.

| Field | Default | Meaning |
|---|---|---|
| `log_level` | `info` | Application logging level |
| `langfuse_public_key`, `langfuse_secret_key` | unset | Both required; keep in `.env` |
| `langfuse_host` | `https://cloud.langfuse.com` | Trace destination |
| `langfuse_trace_sensitive_data` | `true` | Suppress raw content/IDs when false |
| `langfuse_export_external_spans` | `false` | Export third-party OTEL spans |
| `langfuse_environment` | unset | Environment label |
| `langfuse_release` | unset | Release label |
| `langfuse_sample_rate` | `1.0` | Export fraction |
| `langfuse_timeout` | SDK default | Export timeout |
| `langfuse_flush_at` | SDK default | Buffered event count |
| `langfuse_flush_interval` | SDK default | Flush cadence |

Memory traces include counts and character totals, never record bodies. Run the
bundled stack with the [Langfuse runbook](operations.md#local-langfuse-observability).

## Advanced LightRAG Fields

These fields are supported but normally left at code defaults:

```yaml
corpus:
  ingestion:
    chunk_token_size: 2000
  retrieval:
    kg_chunk_pick_method: VECTOR
    max_entity_tokens: 6000
    max_relation_tokens: 8000
    max_total_tokens: 40000
    kg_entity_types: []
storage:
  lightrag:
    vector_index_type: HNSW_HALFVEC
    hnsw_m: 32
    hnsw_ef_construction: 256
    hnsw_ef_search: 256
    # Bounded scalar adapter options. Milvus accepts only LightRAG's documented
    # index/metric/HNSW/SQ/IVF keys; credentials do not belong here.
    vector_db_kwargs: {}
```

`vector_db_kwargs` accepts at most 16 scalar entries and 4096 encoded bytes.
Secrets and endpoint URIs are rejected from this mapping. Milvus visual-fusion
vector overwrite and PostgreSQL hot-workspace promotion are unsupported, and
those combinations fail deterministically.

An empty `kg_entity_types` uses LightRAG's general taxonomy. For stronger domain
control, set `corpus.extraction.entity_type_prompt_file` to a file under
`prompts/entity_type/`.
