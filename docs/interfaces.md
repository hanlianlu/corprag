# Interfaces

This document owns public REST, MCP, Web, and in-process request/response
contracts. Configuration belongs in [Configuration](configuration.md), runtime
behavior in [Retrieval and Answer](retrieval-answer.md), durable lifecycle rules
in [RunRuntime and durable query execution](durable-answer-runs.md), and authorization in
[Security](security.md).

## Choosing An Interface

| Interface | Use when | Ingestion |
|---|---|---|
| REST | DlightRAG runs as a service | Durable Corpus Mutation Runs |
| MCP | An agent connects over stdio or streamable HTTP | Durable Corpus Mutation Runs |
| Web | A browser user uploads and chats | Durable Corpus Mutation Runs |
| In-process Application | Your process owns DlightRAG and its dependencies | Durable Corpus Mutation Runs |

Remote clients should not import `dlightrag`; use REST, MCP, or Web. Configure
models, PostgreSQL, credentials, and the parser once, then reuse the service or
Application instance.

```python
from dlightrag import DlightragConfig, create_application

application = await create_application(DlightragConfig())
try:
    ...
finally:
    await application.aclose()
```

## Common Answer Terms

| Term | Meaning |
|---|---|
| `contexts` | Retrieved evidence. `/answer` returns only what the answer model saw. |
| `sources` | Document-level groups containing chunks, pages, and media routes. `/answer` returns only cited sources. |
| `references` | Compact cited-document projection derived from validated inline citations. |
| `evidence_images` | Cited visual evidence available for rendering. |
| `parts` | Ordered Markdown, Artifact, and explicitly inline Evidence Image parts. |
| `usage` | Root, child, and inclusive provider usage when available. |
| `evidence` | Counts of admitted chunks, entities, relationships, and cited sources. |
| `parent_run_id`, `continuation_kind` | Follow-up/fork lineage. |

Answer **attachments** are files or HTTP(S) references used only by one answer.
They never become workspace documents or appear in `/retrieve`. The separate
`query_images` input belongs only to `/retrieve` and performs knowledge-base
visual search.

## Ingestion

### REST

`POST /runs/corpus/ingest` and `/runs/corpus/replace` accept JSON and
return `202 Accepted` with a durable `corpus_mutation` Run descriptor.
Single and batch multipart uploads use the corresponding `/upload` and
`/uploads` suffixes. A single upload may provide a 64-character hexadecimal
`content_sha256`; mismatch rejects acceptance and deletes the Run-exclusive
staging directory. Every REST mutation requires `Idempotency-Key`.

```bash
curl -X POST http://localhost:8100/runs/corpus/ingest \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: ingest-docs-1' \
  -d '{"source_type":"local","path":"docs"}'

curl -X POST http://localhost:8100/runs/corpus/ingest \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: ingest-s3-1' \
  -d '{"source_type":"s3","bucket":"my-bucket","prefix":"docs/"}'

curl -X POST http://localhost:8100/runs/corpus/ingest \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: ingest-url-1' \
  -d '{"source_type":"url","url":"https://cdn.example.com/report.pdf"}'
```

| Field | Required for | Description |
|---|---|---|
| `source_type` | all | `local`, `azure_blob`, `s3`, or `url` |
| `path` | local | File/directory relative to managed `input_dir/<workspace>` |
| `container_name` | Azure | Container name |
| `blob_path` | Azure single | Object path; exclusive with `prefix` |
| `bucket` | S3 | Bucket name |
| `s3_key` | S3 single | Object key; exclusive with `prefix` |
| `s3_region` | S3 optional | Per-request region override |
| `prefix` | Azure/S3 batch | Prefix; omit or use `""` for the whole container/bucket |
| `url` / `urls` | URL | One URL or a batch; mutually exclusive |
| `filename` | URL optional | Parser filename when the URL path lacks an extension |
| `source_uri` / `source_uris` | URL optional | Stable provenance, independent of the fetch URL |
| `download_uri` / `download_uris` | URL optional | Durable S3, Azure, or queryless public HTTPS locator |
| `documents` | optional | Explicit manifest with per-document metadata |
| `retain_source_file` | remote optional | Keep fetched bytes; otherwise a durable locator is required |
| `replace` | optional | Purge an existing document before enqueueing replacement |
| `workspace` | optional | Target, default `default` |
| `title`, `author` | optional | Built-in document metadata |
| `metadata` | optional | Custom metadata object |

`source_uri` is identity, never a download address. A signed URL containing a
query or fragment is not durable; use `retain_source_file: true` or supply a
separate queryless locator. S3 uses the standard AWS credential chain. Payloads
never carry access keys.

Per-document metadata uses a manifest:

```json
{
  "source_type": "s3",
  "bucket": "my-bucket",
  "metadata": {"source_system": "s3-prod"},
  "documents": [
    {"key": "docs/a.pdf", "metadata": {"department": "legal"}}
  ]
}
```

### In-process And MCP

```python
from dlightrag.application.corpus_admin import IngestSpec

creation = await application.corpus_mutations.create_ingest(
    workspace="default",
    spec=IngestSpec(source_type="s3", bucket="my-bucket", prefix="docs/"),
    submitted_by="operator-1",
    idempotency_key="ingest-s3-1",
)
run = await application.runs.get(owner_id="default", run_id=creation.run.run_id)
```

MCP `ingest` exposes the REST source arguments and returns the common Run
descriptor. MCP also exposes `retry_files`, `delete_files`, `reset_corpus`, and
`resume_corpus_run`; use `get_run` and `cancel_run` for their shared lifecycle.

### Runs And Results

Corpus Mutation actions are `ingest`, `replace`, `delete`, `retry`, and `reset`.
They use the common `queued`, `running`, `succeeded`, `failed`, and `cancelled`
states and the common `GET|DELETE /runs/{run_id}` plus
`GET /runs/{run_id}/events` observation routes. Cancellation closes at the
durable upstream handoff. A multi-document mutation with any failed document is
`failed`, not partially successful. Acceptance is atomically bounded by the
validated 1,000 nonterminal Corpus Mutation fuse; a full lane returns HTTP 503
before inserting a Run. Accepted work remains durable and independently claims
at no more than two mutations deployment-wide.

The terminal result carries the action, stable `track_id`, bounded per-document
outcomes, `document_count`, and `details_truncated`. An ambiguous destructive
outcome remains `running` with `phase=waiting_for_repair` and exposes bounded
`repair_reason` and `repair_remedy` guidance. An authorized operator repairs the
upstream state and explicitly uses `POST /runs/{run_id}/resume`; Corpus Reset is
the only destructive supersession path.

```json
{
  "action": "ingest",
  "track_id": "dlightrag-corpus-0199a0a0-0000-7000-8000-000000000001",
  "document_count": 1,
  "details_truncated": false,
  "documents": [{"document_id": "file-doc-abc123", "status": "ready", "phase": "finalized"}]
}
```

### Metadata

Custom keys need no declaration and are immediately filterable through
`filters.custom`. Matching is case-insensitive without rewriting stored values
or keys. Built-ins such as `filename`, `filename_stem`, `file_extension`,
`title`, and `author` are reserved. Set title/author through their fields.
`creation_date` is the one built-in accepted under `metadata`; it must be ISO
8601 and is filtered with `creation_date_from`/`creation_date_to`.
`_dlightrag_finalization_complete` is internal and is rejected from caller
metadata.

Metadata GET, update, search, planner schema, and search statistics expose only
Product Documents whose internal finalization marker is exactly true. Pending,
failed-finalization, legacy-unproven, and direct-LightRAG-bypass rows behave as
not found on these surfaces.

`POST /metadata/search` returns document IDs ordered by `doc_id`, with `limit`
(1–100, default 50) and a signed opaque `cursor`. The cursor is bound to the
workspace, request filters, and filename match mode. Invalid or cross-workspace
cursors return 422 before storage access.

## Retrieval And Answer

### Request Fields

| Field | Endpoint | Default | Description |
|---|---|---|---|
| `query` | both | required | Query text |
| `mode` | answer | `auto` | `auto`, `fast`, or `research` |
| `workspace` | both | configured default | One workspace |
| `workspaces` | both | unset | Explicit federated workspace list |
| `all_workspaces` | both | `false` | Every workspace visible to the caller; exclusive with explicit selection |
| `top_k` | both | config | KG entity/relationship breadth |
| `chunk_top_k` | both | config | Text/visual candidate breadth |
| `federated_rerank` | retrieve | `false` | Rerank the merged multi-workspace candidate pool when a reranker is configured |
| `bm25_query` | retrieve | query-derived | Optional lexical override; REST/MCP cap it at 1,024 characters |
| `query_images` | retrieve | none | Up to three current images for visual search |
| `attachments` | answer | none | Link descriptors or multipart files used only by this answer |
| `semantic_highlights` | answer | `false` | Add answer-aware source highlights when globally enabled |
| `history` | answer | none | Up to 100 caller-supplied user/assistant messages |
| `filters` | both | none | Built-in and custom metadata filters |

`all_workspaces` is authorization-relative. `None` and `[]` mean omission;
`"*"` and `"all"` are ordinary workspace names. Ingestion remains
single-workspace.

### REST

```bash
# Accept a durable Retrieval run.
RETRIEVAL_RUN=$(curl -sS -X POST http://localhost:8100/retrieve \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: retrieval-example-1' \
  -d '{"query":"key findings","all_workspaces":true}' | jq -r .run_id)
curl http://localhost:8100/runs/$RETRIEVAL_RUN

# Accept a durable Answer run.
ANSWER_RUN=$(curl -sS -X POST http://localhost:8100/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"key findings","semantic_highlights":true}' | jq -r .run_id)

# Read status/result and follow events.
curl http://localhost:8100/runs/$ANSWER_RUN
curl -N -H 'Last-Event-ID: 12' http://localhost:8100/runs/$ANSWER_RUN/events

# Attach an HTTPS resource.
curl -X POST http://localhost:8100/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"summarize this","attachments":[{"url":"https://cdn.example.com/report.pdf","filename":"report.pdf"}]}'

# Upload resources: one JSON request part plus repeated attachments.
curl -X POST http://localhost:8100/answer \
  -F 'request={"query":"summarize this"};type=application/json' \
  -F 'attachments=@report.pdf' \
  -F 'attachments=@figure.png'
```

There is no public `/query` route, inline remote Retrieval result, or ephemeral
Answer mode. `POST /retrieve` and `POST /answer` persist a Run and return HTTP
202. A Retrieval descriptor differs only in `run_kind: "retrieval"`:

```json
{
  "run_id": "019…",
  "run_kind": "answer",
  "lane": "query",
  "status": "queued",
  "status_url": "/runs/019…",
  "events_url": "/runs/019…/events",
  "cancel_url": "/runs/019…"
}
```

### Run Lifecycle and Answer Endpoints

| Operation | Contract |
|---|---|
| `POST /retrieve` | Accept a Retrieval Run and return its descriptor. Optional `Idempotency-Key` replays the same normalized request; conflicting reuse returns 409. |
| `POST /answer` | Accept an Answer Run. Optional `Idempotency-Key` replays the same normalized request; conflicting reuse returns 409. |
| `GET /runs` | List this owner's runs oldest-first; `after` + `limit` (1–100, default 50). |
| `GET /runs/{run_id}` | Return common kind, lane, status, cancellation, phase, progress, error, and terminal result fields. |
| `GET /runs/{run_id}/events` | Reconnectable SSE; resume with `Last-Event-ID` or integer `after`. |
| `DELETE /runs/{run_id}` | Idempotent cancellation; 200 if terminal, otherwise 202. |
| `POST /runs/{run_id}/resume` | Requeue an authorized Corpus Mutation Run from `phase=waiting_for_repair`; 409 for every other lifecycle state. |
| `GET /answer/{run_id}/artifacts` | List Answer Artifact descriptors/outcome; 409 before a result exists. |
| `GET /answer/{run_id}/artifacts/{resource_id}` | Stream Answer Artifact bytes with Range support; `download=true` forces attachment. |
| `GET /answer/{run_id}/artifacts/{resource_id}/presentation` | Project an available Markdown Artifact as typed `AnswerResponse`, including that Artifact's validated citation sources. |
| `POST /answer/{run_id}/steer` | Queue an instruction for live Research. |
| `POST /answer/{run_id}/follow-up` | Create a child run using the selected terminal answer as context. |
| `POST /answer/{run_id}/fork` | Create a sibling branch from accepted context. |
| `GET /answer/{run_id}/transcript` | Return bounded canonical ancestry. |
| `GET /answer/{run_id}/children` | Newest-first keyset page (`limit` 1–100, default 50). |

Run status is `queued`, `running`, `succeeded`, `failed`, or `cancelled`. Phase is
an executor-owned string. Retrieval uses `planning` and `searching`; Answer uses
`routing`, `planning`, `searching`, `researching`, and `generating`. Unknown,
pruned, and foreign-owner Runs all return 404. A retained Run whose event log was
trimmed returns 410 from the event endpoint; status/result remains readable.
Disconnecting a client never cancels a Run.

The common SSE terminal and progress events are:

| Event | Payload |
|---|---|
| `progress` | Current executor-owned `phase` |
| `done` | Terminal success with full kind-specific `result`, or cancellation without one |
| `error` | Terminal `{kind, message}` failure |

Answer may additionally emit `token`, `reset`, `tool_start`, `tool_progress`,
and `tool_end`. Retrieval emits only `progress` and one terminal event.

Each durable sequence is the SSE `id`. Supplying conflicting header/query
cursors returns 400. Without a cursor, replay starts at sequence 1. Ten-second
comment keepalives consume no sequence. Exactly one terminal event is committed.

Canonical successful Retrieval result after reader projection:

```json
{
  "contexts": {"chunks": [], "entities": [], "relationships": []},
  "sources": [],
  "trace": {},
  "image_descriptions": []
}
```

Runtime storage omits authorization-dependent download/image URLs and query-image
bytes. `GET /runs/{run_id}` and terminal SSE reads project source and visual URLs
against the caller's current permissions.

Canonical successful Answer result:

```json
{
  "answer": "The key findings are... [1-1] [2-3]",
  "contexts": {"chunks": [], "entities": [], "relationships": []},
  "references": [{"id": "1", "title": "report.pdf"}],
  "sources": [],
  "evidence_images": [],
  "parts": [{"type": "markdown", "text": "The key findings are... [1-1]"}],
  "artifacts": [],
  "artifact_outcome": {"status": "complete", "issues": []},
  "usage": {},
  "evidence": {},
  "trace": {},
  "image_descriptions": []
}
```

`trace.bm25_enabled` reports lexical-lane participation. If one retrieval lane
fails and the other succeeds, the result continues with `bm25_error_type` or
`lightrag_error_type`; `lightrag_mix_chunk_count` records the pre-fusion
LightRAG count.

### In-process Application

```python
from dlightrag.application.access import DEPLOYMENT_OWNER_ID
from dlightrag.application.answer_runs import AnswerRequest
from dlightrag.application.retrieval import RetrieveProjection, RetrieveRequest

retrieval_run = await application.retrieval.create(
    request=RetrieveRequest(query="What changed?", workspaces=("default",)),
    owner_id=DEPLOYMENT_OWNER_ID,
    idempotency_key="retrieval-example-1",
)
retrieved = await application.retrieval.wait(
    owner_id=DEPLOYMENT_OWNER_ID,
    run_id=retrieval_run.run.run_id,
    projection=RetrieveProjection(
        downloadable_workspaces=None,
        visual_workspaces=None,
    ),
)

answer = await application.answers.answer(
    AnswerRequest(
        query="What changed?",
        workspaces=("default",),
        semantic_highlights=True,
    ),
    owner_id=DEPLOYMENT_OWNER_ID,
)

async for event in application.answers.answer_stream(
    AnswerRequest(query="What changed?", workspaces=("default",)),
    owner_id=DEPLOYMENT_OWNER_ID,
):
    print(event.event_type, event.payload)
```

`RetrievalService.retrieve()` is the create-and-wait convenience; cancelling that
await only detaches and does not cancel its accepted Run. `RunService` exposes
common status, listing, subscription, and cancellation.

Files/URLs become `ResourceInput` values through
`AnswerAttachment.from_path/from_bytes/from_url` and
`resource_inputs_from_attachments`. `AnswerService` owns Answer-specific
steering, continuation, transcript, and roster methods. There is no separate
public Python SDK for remote callers.

### MCP Server

MCP `retrieve` and `answer` return only durable descriptors; poll `get_run` for
the canonical result. A tool result puts typed JSON in `structuredContent` and
formatted equivalent JSON in its first text block. Expected validation or
authorization failures set `isError: true`; protocol failures remain JSON-RPC
errors.

Registered public tool names are:

- query/run: `retrieve`, `answer`, `get_run`, `cancel_run`, `list_runs`,
  `steer_answer_run`, `follow_up_answer_run`, `fork_answer_run`,
  `get_answer_transcript`, `list_answer_children`, `list_answer_artifacts`,
  `read_answer_artifact`
- corpus: `list_workspaces`, `get_capabilities`,
  `get_workspace_storage_status`, `create_workspace`, `ingest`, `retry_files`,
  `list_files`, `delete_files`, `reset_corpus`, `resume_corpus_run`
- model catalogue: `get_model_catalogue`, `upsert_model_catalogue_entry`,
  `remove_model_catalogue_entry`
- memory: `list_memories`, `remember_memory`, `forget_memory`,
  `undo_memory_change`, `get_memory_settings`, `set_memory_enabled`,
  `clear_memory`

### Web

Web routes under `/web/api/*` are browser contracts, not compatibility aliases
for REST. `GET /web/api/bootstrap` returns authorized workspace state, Files
target, attachment limits, and image capability—never bearer or edge tokens.
Route families cover:

- `/conversations`, `/conversations/{id}/history`, and
  `/runs/{run_id}/attachments/{ordinal}` (plus `/thumbnail`);
- `/answer`, submission reconciliation, status/resume/steer/children,
  follow-up/fork/cancel, Artifacts/presentation, and events; and
- Files/upload and same-origin `/corpus-runs/{run_id}`
  status/events/cancel/resume, workspaces, images, Memory, and model catalogue.

`/web/` is unpersisted New Chat;
`/web/conversations/{conversation_id}` selects a durable owner-scoped
conversation. The URL is authoritative for reload and browser history.

`POST /web/api/answer` accepts an optional conversation ID, query, attachments,
and search workspaces. It returns HTTP 202 with canonical `{conversation, turn}`.
For a first submission, the server creates conversation, turn, blobs, and run in
one transaction from the owner-scoped `submission_id`. On an ambiguous result,
use `GET /web/api/answer-submissions/{submission_id}`; the browser must not
blindly repeat the POST.

The Web event stream follows the same durable sequence as REST but projects a
typed `AnswerPresentation` (`answer_text`, `parts`, `sources`,
`evidence_images`, `artifacts`, and `artifact_outcome`). Conversation history
uses the same shape. Pending, failed, and cancelled turns remain visible;
only succeeded turns become model history.

History defaults to the newest 40 turns and accepts a signed cursor plus a limit
up to 100. Attachments are owner-scoped, content-addressed run blobs and are
re-registered lazily for follow-ups. Count, per-file, and total-byte limits are
validated before acceptance; read failures after acceptance produce a terminal
error rather than silent omission. Lifecycle details are centralized in
[RunRuntime and durable query execution](durable-answer-runs.md).

## Contexts

`contexts` always contains `chunks`, `entities`, and `relationships`. Public
REST/Web responses use image routes rather than inline base64. In-process
internals may carry bounded `image_data` for model use.

### Chunk

```json
{
  "chunk_id": "abc123",
  "reference_id": "1",
  "file_path": "report.pdf",
  "content": "Page text...",
  "page_number": 2,
  "image_url": "/images/default/abc123?size=full",
  "thumbnail_url": "/images/default/abc123?size=thumb",
  "image_mime_type": "image/png",
  "relevance_score": 0.87
}
```

| Field | Meaning |
|---|---|
| `chunk_id` | Unique chunk ID |
| `reference_id` | Document-level citation ID |
| `file_path` | Display basename, not provenance or download authority |
| `content` | Chunk text |
| `page_number` | Optional 1-based display page |
| `image_url`, `thumbnail_url`, `image_mime_type` | Optional visual route metadata |
| `relevance_score` | Optional 0–1 rerank score |
| `metadata` | Extra metadata |
| `_workspace` | Source workspace for federated retrieval |

### Entity And Relationship

Entity rows contain `entity_name`, `entity_type`, `description`, `source_id`,
and optional `reference_id`. Relationship rows contain `src_id`, `tgt_id`,
`description`, `source_id`, and optional `reference_id`. `source_id` is a
comma-separated list of supporting chunk IDs.

## Sources

A source groups one document's chunks in citation-index order:

```json
{
  "id": "1",
  "title": "report.pdf",
  "type": "file",
  "source_uri": "local://default/docs/report.pdf",
  "download_url": "/files/raw/doc-a1b2c3?workspace=default",
  "cited_chunk_ids": ["abc123"],
  "chunks": [{
    "chunk_id": "abc123",
    "chunk_idx": 1,
    "page_number": 2,
    "content": "Page text...",
    "image_url": null,
    "thumbnail_url": null,
    "highlight_phrases": null
  }]
}
```

`source_uri` is stable provenance. `download_url` is an authorized projection
(`/files/raw/{document_id}` for REST,
`/web/api/files/raw/{document_id}` for Web); MCP transport-neutral payloads
leave it null. `retrieve` returns all retrieved sources. `answer` returns cited
sources only and sets `cited_chunk_ids`. Projecting a historical result does not
re-authorize its source: the live download and visual routes return 404 after a
document is hidden.

## Citations

`references` is the compact `{id, title}` projection of validated cited sources.
Inline citations accept:

| Format | Meaning |
|---|---|
| `[1-2]` | Document/reference 1, its second chunk (1-based) |
| `[3]` | Document/reference 3 |

Resolve `[1-2]` by finding source `id: "1"`, then chunk `chunk_idx: 2`.
`page_number` helps navigation but does not affect citation validity.

Visual bytes are read through authenticated routes. Both full and thumbnail
reads return 404 when the chunk has no attributable, currently finalized
Product Document; cached thumbnails do not bypass that check:

| Interface | Reference |
|---|---|
| REST | `/images/{workspace}/{chunk_id}?size=thumb|full` |
| Web | `/web/api/images/{workspace}/{chunk_id}?size=thumb|full` |
| MCP | REST-style URL when a reachable REST route exists; no MCP binary stream |
| Application | Render references; internals may also expose `image_data` |

## Multimodal Inputs

Use answer attachments for question-local documents/images:

```bash
curl -X POST http://localhost:8100/answer \
  -F 'request={"query":"What does this show?"};type=application/json' \
  -F 'attachments=@photo.png'
```

Use `query_images` only for visual retrieval against the knowledge base:

```json
{
  "query": "diagrams like this",
  "query_images": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64>"}}
  ]
}
```

Image support is a deployment capability. Discover it through REST
`GET /health`, MCP `get_capabilities`, or
`await application.answers.capabilities()`. Unsupported/unknown image input
fails closed with `CURRENT_IMAGES_UNSUPPORTED` or
`ANSWER_IMAGE_CAPABILITY_UNKNOWN`.

## Workspace And File Management

| Route | Contract |
|---|---|
| `GET /workspaces` | Page the authorized workspace catalogue. |
| `POST /workspaces` | Create an empty workspace (201; duplicate 409). |
| `GET /workspaces/{workspace}/storage` | Read operator storage/promotion state. |
| `POST /runs/corpus/reset` | Accept Corpus Reset while retaining Workspace identity and history. |
| `GET /files` | Page processed files for one workspace. |
| `POST /runs/corpus/delete` | Accept durable deletion by exact path, name, or document ID. |
| `GET /files/failed` | Page failed documents. |
| `POST /runs/corpus/retry` | Accept retry for exact document IDs or `all_retryable`. |
| `GET /files/raw/{document_id:path}` | Stream or redirect one authorized source. |
| `POST /metadata/search` | Page matching document IDs. |
| `GET /metadata/{doc_id}` | Read one document's metadata. |
| `POST /metadata/{doc_id}` | Merge nonempty custom metadata. |

REST uses resource-oriented operations:

```bash
curl -X POST http://localhost:8100/workspaces \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"Research Notes"}'

curl -X POST http://localhost:8100/runs/corpus/reset \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: reset-research-notes-1' \
  -d '{"workspace":"research_notes"}'
```

`GET /workspaces` orders by workspace ID and pages with `limit` (default 50,
maximum 100) plus a signed cursor. Access filtering happens after catalog
paging. The response contains `workspaces`, `records`, and `next_cursor`. MCP
`list_workspaces` returns only the first 50 plus `has_more`.

Corpus deletion and reset are durable Run actions rather than direct mutation
routes. Reset preserves the Workspace registry, access scope, and history. Web
Files uses a workspace-bound signed keyset cursor, defaults
to 50 files (maximum 100), and orders by `updated_at DESC, id ASC`; processed
rows appear only after Product Document finalization. The failed-files view
remains administrative and available for repair regardless of publication.

## Model Catalogue And Profile Memory

| REST route | Contract |
|---|---|
| `GET /models/catalogue` | Effective runtime overlay with revision. |
| `PUT /models/catalogue` | Upsert one complete endpoint profile under optimistic revision; requires `model_catalogue.write`. |
| `DELETE /models/catalogue` | Remove one overlay entry under optimistic revision; requires `model_catalogue.write`. |
| `GET /memory` | Newest-first active records with `limit` 1–100 and signed cursor. |
| `POST /memory` | Remember one owner-scoped Profile Memory record. |
| `DELETE /memory/{memory_id}` | Forget a record. |
| `POST /memory/changes/{change_id}/undo` | Apply the compensating undo. |
| `GET|PUT /memory/settings` | Read/change the owner capability switch. |
| `POST /memory/clear` | Clear owner records; returns 204. |

Memory cursors are signed and owner-independent as tokens; owner scope remains
an authenticated query predicate. Invalid cursors return 422 before storage.
When Memory is disabled, mutation/recall operations are unavailable except
reading/changing the setting.

## Health And Errors

`GET /health` is liveness: it returns in-process state without model, parser,
corpus, or database I/O. Its bounded `components` map distinguishes `process`,
`operational_state`, `run_coordinator`, `cancellation_listener`,
`corpus_storage`, `parser`, and `providers`; details and warnings use fixed
sanitized text. The four LightRAG storage class names and
`answer_image_capability` are also reported. Degraded state remains HTTP 200.

`GET /ready` checks only the authority required to durably admit and coordinate
Runs: Application Operational State plus the injected writable Operational
State database probe. It does not probe the reader corpus or external vector,
parser, or model providers. The endpoint returns fixed-detail HTTP 503 when
that authority is unavailable. Readiness checks are single-flighted and
memoized for two seconds.

General errors are `{detail, error_type, error_kind?}` where `error_type` is
`validation`, `auth`, `unavailable`, `configuration`, or `internal`. Stable
answer error kinds are:

- `CURRENT_IMAGES_UNSUPPORTED`, `CURRENT_IMAGE_LIMIT_EXCEEDED`,
  `CURRENT_DOCUMENT_PARSE_FAILED`, `ANSWER_IMAGE_CAPABILITY_UNKNOWN`,
  `ANSWER_INPUT_OVERFLOW`, `MODEL_CAPABILITY_UNAVAILABLE`,
  `unsupported_resource_capability`, and `ANSWER_RESOURCE_INVALID`;
- `invalid_tool_configuration`, `unsupported_answer_mode`, `routing_failed`,
  `tool_contract_changed`, `run_abandoned`, and `run_execution_failed`; and
- `ANSWER_STREAM_FAILED`.

Stable top-level Retrieval terminal error kinds are `retrieval_timeout`,
`retrieval_failed`, `retrieval_model_changed`,
`retrieval_context_policy_changed`, `retrieval_input_incompatible`, and
`retrieval_input_missing`. `run_abandoned` is common to both durable Query kinds.

Internal exception text and schema detail are not public.

Accepted Retrieval and Answer Runs queue under worker saturation while the Query
Lane has fewer than 30,000 nonterminal Runs; the deployment-wide fuse rejects
later admission with HTTP 503. Corpus Mutation admission uses its independent
validated 1,000-Run fuse and the same pre-insert 503 behavior. The controlled
failure and full-fuse evidence is linked from the
[Slice 6 validation report](validation/run-runtime-slice-6.md). Queue residence
has no application timeout. Top-level Retrieval applies
`corpus.retrieval.timeout` only during claimed
execution and reports `retrieval_timeout` terminally. Explicit transient corpus
or provider interruptions defer Retrieval and Answer with a durable bounded
retry checkpoint, release Query compute capacity, and later resume the same
Run. Authentication, unsupported configuration/schema, invalid input,
deterministic model rejection, context overflow, and unknown exceptions remain
terminal. Attachment total-byte
overflow returns HTTP 413 before buffering. Generic rate, connection, and
volumetric controls belong at ingress;
see [Security](security.md#ingress-responsibilities).
