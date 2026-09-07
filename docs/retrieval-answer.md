# Retrieval And Answer

This document owns how queries become contexts, answers, sources, and citations.
Payloads live in [Interfaces](interfaces.md), fields in
[Configuration](configuration.md), runtime ownership in
[Architecture](architecture.md), and recovery in
[RunRuntime and durable query execution](durable-answer-runs.md).

DlightRAG always uses LightRAG `mix` as its graph/vector base. It adds metadata
filtering, optional direct image retrieval, PostgreSQL BM25, RRF fusion,
provenance hydration, reranking, answer packing, and citation validation.

- Top-level `/retrieve` is a durable, owner-scoped Run. It is
  knowledge-base-only and may take `query_images`.
- `/answer` creates a durable Answer Run from a query plus optional attachments,
  then resolves `auto | fast | research`.
- Retrieval inside an Answer is an internal Retrieval Stage under the Answer
  Run; it never creates another Run.
- `auto` considers the valid mode set and conversation context. When both paths
  are legal, routing defaults to Research unless the turn is corpus-grounded.

Every accepted Retrieval pins normalized query/options, authorized search scope,
the required `extract` and optional `vlm` model profiles, capability facts, and
policy revisions. Every accepted Answer additionally pins bounded history and
Resources. The Web conversation
layer wraps the Answer pipeline; it does not define another path.

## Ingestion Shape

```text
source
  -> LightRAG parser/routing
  -> LightRAG chunks, entities, relationships, vectors, and document status
  -> optional fused visual-vector alignment
  -> DlightRAG metadata and BM25 maintenance
```

The configured MinerU or Docling block produces one internal wildcard parser
rule. Unsupported suffixes use LightRAG's native/legacy route. Parser outputs
remain aligned to the LightRAG document record.

A successful drawing has one canonical chunk. LightRAG supplies its VLM text;
when native multimodal embedding is active, DlightRAG replaces the same chunk's
vector with one fused text+image vector. It does not add a second visual-only
chunk. Text-only or safely downgraded `auto` mode retains LightRAG's text vector;
explicit multimodal probe failure aborts startup.

## Query Pipeline

For a top-level request, the Application resolves authorization and accepts a
`run_kind=retrieval` Query-lane Run before this pipeline executes. The same raw
pipeline begins directly inside an Answer without nested Run acceptance.

```text
Retrieval Stage
  -> use accepted authorized concrete workspaces and warm them
  -> plan lexical terms and optional metadata filters once
  -> per workspace:
       LightRAG mix
       optional direct image-vector retrieval
       PostgreSQL BM25
       RRF fusion + dedup + candidate limit
       provenance/image hydration
       final rerank
  -> federated round-robin merge
  -> reference canonicalization
  -> optional answer packing/generation
```

`RetrievalPlanner` sees the query, schema, bounded prior turns when appropriate,
and current-image hints. It never sees answer attachment bytes/text/manifests.
Explicit BM25 terms and metadata filters remain authoritative. A Research KB
tool's chosen semantic query is preserved while the planner derives only its
supporting lexical/filter/image context.

The `LightRAG mix` and BM25 lanes degrade independently. If one fails, the other
may return results and trace records `lightrag_error_type` or `bm25_error_type`.
If both fail, retrieval raises the LightRAG error with BM25 chained. Trace
`lightrag_mix_chunk_count` records the LightRAG count before fusion;
`contexts.chunks` is the final fused/reranked set.

Top-level Retrieval uses `planning` and `searching` durable phases. Corpus
unavailability returns a deferred Runtime outcome with bounded exponential
backoff, releasing Query execution capacity until `next_attempt_at`. The
configured retrieval timeout bounds claimed planning/search execution and is a
terminal `retrieval_timeout`; queue residence is not part of that timeout.
Unexpected execution failures settle as sanitized `retrieval_failed` errors.

### BM25

BM25 queries the same `LIGHTRAG_DOC_CHUNKS` rows. Ingestion labels each chunk's
language. Startup maintains partial pg_textsearch indexes for configured
languages plus a full-table `simple` fallback. Supported query languages select
their profile; unknown/ambiguous languages use `simple`.

Changing profile signatures, `k1`, or `b` for an existing workspace requires an
offline BM25 rebuild. Disabling BM25 removes only this PostgreSQL lane;
Resource lexical search remains run-scoped and in memory.

## Product Document Visibility And Metadata In-Filtering

Product Document visibility is always-on: directly attributable chunk evidence
is admitted only when its document metadata row exists and
`_dlightrag_finalization_complete` is exactly true. Missing, NULL, false, and
out-of-band LightRAG documents are unpublished. PostgreSQL applies `IS TRUE` in
metadata, graph-chunk, BM25, and vector queries. Unscoped ANN/BM25 first rank a
bounded over-fetch window and then correlate those candidates to metadata;
non-pushdown vector stores similarly post-filter only bounded hit IDs. A short
result is preferable to leakage.

Named filters map to typed columns: `filename`, `file_extension`, `title`,
`author`, and creation-date bounds. Arbitrary keys use `filters.custom` against
one JSONB column, with case-insensitive comparison.

Filename matching is corpus-aware: exact stored name/stem first, then literal
substring. The caller/planner supplies one `filename` value rather than guessing
which operator will match unseen data.

- Explicit caller filters are strict; zero candidates means zero results.
- Inferred filters carry confidence/evidence for observability. If they resolve
  to no candidates or retrieve no chunks, DlightRAG retries unfiltered.
- Non-empty inferred candidates constrain semantic and BM25 legs.

Every chunk-producing leg enforces visibility, with any user filter as an
additional scope:

- `FilteredVectorStorage` returns immediately for empty filtered candidates,
  uses exact scoring for small filtered sets, HNSW iterative scan for larger
  sets, and a bounded visibility-only path without a user filter.
- Graph entity/relation legs resolve source chunks by ID, so
  `FilteredChunkStore` visibility-checks that bounded lookup too.
- The wrappers use a context variable only for the optional user filter;
  visibility itself is not optional. Ingest and delete use their mutation
  collaborators rather than treating product reads as an internal identity API.

The filter controls quotable chunk evidence. It does not rewrite LightRAG's
corpus-level entity/relationship summaries, which may merge descriptions from
multiple documents and have no separable per-document share. Trace
`metadata_kg_chunks_dropped` counts graph-referenced chunks rejected by scope.
Bounded fallback paths may also report `visibility_strategy`,
`visibility_dropped`, and `visibility_shortfall`; these are not snapshot
isolation claims.

## Multimodal Retrieval

`query_images` adds two transient paths:

```text
query image
  |-- VLM description -> LightRAG text/KG + BM25
  `-- native image embedding -> fused visual chunks (when active)
```

Document visuals use fused VLM-description+image vectors; query images use
image-only query vectors. Provider adapters apply official query/document task
semantics and split batches at provider input/token/image limits without
reordering. RRF and dedup resolve overlap between semantic and visual hits.

With text modality, direct image embedding is skipped but VLM descriptions may
still drive text retrieval. `auto` can make that safe downgrade after probe
failure; explicit multimodal mode fails startup.

## Fusion And Reranking

DlightRAG disables LightRAG's query reranker. It reranks the fused set after
provenance hydration so LightRAG, BM25, and direct-image candidates compete in
one list with page/image data attached.

Ranker classes are:

- chat-model listwise reranking, optionally with images when the selected model
  passed its vision probe;
- multimodal or text HTTP `/rerank` adapters;
- Voyage, Cohere, and Azure Cohere text rerankers.

A configured score threshold is hard: candidates below it disappear, even if a
workspace then contributes none. Runtime reranker failure falls back to the
pre-rerank fused order. Configuration failures (for example, a selected provider
without credentials) fail startup instead of changing strategy.

Reranking has its own image budget. Oversized visual candidates fall back to
text where available; unbounded data URIs are never sent.

## Multi-Workspace Retrieval

The planner runs once, selected workspaces execute concurrently, and each runs
the complete filtering/fusion/rerank pipeline. The federation layer tags chunks
with `_workspace`, canonicalizes references, round-robin interleaves the
per-workspace lists, and applies the configured output budget plus its
per-workspace fairness floor.

By default, round-robin preserves representation without pretending scores from
different workspace/model calls are calibrated. If `federated_rerank` is true
and the configured reranker is available, DlightRAG reranks the merged candidate
pool once across Workspaces. An unavailable/build-failed reranker uses the capped
interleave; a runtime rerank failure does the same and records its error type in
trace rather than failing Retrieval.

## Answer Orchestration

`AnswerExecutor` owns the workflow; `AnswerOrchestrator` prepares typed Host
context, tools, and effects.

### Fast

Fast performs planning, KB retrieval, and one lightweight generation call. It
uses shared Context Contribution, Evidence, citation, model-call, usage, and
Agent Session infrastructure, but creates no Agent Operation, workspace, tools,
publication, or Profile Memory interaction.

### Research

Research drives `AgentSessionRuntime` over one selected Lane. Its closed
run-local registry may include:

- knowledge-base, resource, and optional provider-neutral public Web tools;
- rooted file/Bash tools when execution is enabled;
- Profile Memory tools for the parent (children recall only);
- progressive `load_skill`;
- allowlisted outbound MCP tools; and
- bounded foreground child Sessions with explicit snapshots and Evidence
  return.

Tool errors return to the model for correction; they do not terminate research.
A no-tool assistant turn ends the run, and that text is the answer. The parent
Research Session authorizes a root Workspace file for publication only through
`attach_artifact`; Fast and Child Sessions do not receive that product tool. A
successful attachment binds the root's normalized relative path, label, media
capability, byte size, and raw-content digest in the same settlement as the tool
result. Reattaching the same path replaces its intent and moves it to the latest
settlement position.

The Answer remains the default deliverable; Workspace and publication-tool
availability do not imply that Research should create an Artifact. The Agent uses
a separate Artifact when the user requests one, when a complete deliverable is
too long or structurally rich for one practical Answer, or when a visual,
interactive, or downloadable surface materially improves use. If the Artifact
contains the complete deliverable, the Answer is a concise orientation and
handoff rather than a substantial copy. Explicit requests for both inline and
file versions are the exception. Independent citation validation governs support
on each surface, not duplicated prose.

At the terminal boundary, the Host verifies every attachment against current
bytes and publishes each valid root plus the safe transitive dependency closure
reachable through Markdown/HTML `artifact:` links. Those links are placement
syntax, not publication authority: an unattached answer link fails validation,
and an attached root omitted by the answer receives a trailing link in attachment
settlement order. Dependencies are published but are not auto-placed. Failed or
stale attachments receive the single bounded correction pass. There is no
reserved filename, privileged Artifact role, or hidden finalizer call.

Native tool-turn text deltas are streamed optimistically when the provider
supports them. They are transient presentation: the Host resets them when the
same turn contains tool calls, a provider attempt fails or is cancelled after
emitting text, a follow-up/correction continues the Session, interrupted
generation is recovered, or citation/Artifact finalization changes the terminal
text. Persisted Request Snapshots, Assistant Turns, tool settlements, and the
canonical result remain the recovery authorities.

Both modes produce the same canonical result: ordered `parts`, cited `sources`,
`references`, `evidence_images`, Artifacts/outcome, usage, and Evidence counts.

## Context And Model Budgets

Each call uses an immutable model profile pinned by normalized provider, model,
and endpoint. It supplies context, input, output, image, and reasoning facts.
Uncatalogued endpoints first receive the best-effort fallback profile; acceptance
fails only if the resolved profile still cannot provide usable capacity.

The Context Policy independently reserves output, dynamic context, safety,
retained tail, episodic continuation, and minimum input. Evidence, history,
resource windows, tool schemas, and observations share the measured residual.
Parallel tools divide the exact next-control residual before execution. Provider
output is limited by both model output capacity and remaining physical context.

Full attachment bytes never enter model context. Only bounded text windows,
capped observations, and budgeted images do.

## Answer Input And Packing

The answer model receives structured messages, not raw `contexts` JSON:

```text
system policy
bounded prior text history (when supplied)
current user message:
  User-attached images
  User-attached document evidence, labeled [att-N-M]
  Knowledge-base evidence, labeled [N-M]
  Knowledge Graph Context
  Question (last)
```

Each citation marker is defined once on the evidence it labels. Attachment
`att-N` identities exist only for that answer; they are not durable chunks or
vectors. Retrieved document images are preceded by their text label and sent
only when they fit.

`top_k` controls KG breadth. `chunk_top_k` controls retrieved text/visual
candidates. Answer retrieval over-fetches candidates, then packs them to the
query model's remaining input capacity and image budget.

- Pure visual chunks whose image cannot fit are removed and later candidates
  backfill them.
- Mixed text+image chunks keep text when the image is skipped.
- Final exact serialization removes whole chunks from the reranked tail and
  rebuilds prompt/citation indexes until it fits; it never truncates a chunk,
  reorders survivors, or retrieves again.
- Returned contexts/sources use the final admitted chunks. Use `/retrieve` for
  the broader pre-answer set.

Images already within JPEG/PNG/WebP limits pass through unchanged. Recompression
honors configured quality and geometry floors; images that still do not fit are
skipped rather than degraded further. Focused VLM inspection is a separate call
with per-call limits and does not consume the final answer image budget.

DlightRAG uses LightRAG `aquery_data()` as the context/reference seed rather than
`aquery_llm()`, because final evidence may include BM25, direct visual, federated,
and reranked results.

## Citation And Presentation Finalization

After generation, DlightRAG validates inline markers against the final packed
context:

- unknown markers are removed;
- a chunk marker pointing only to Markdown headings degrades to its document
  marker because the chunk supports no factual claim;
- generated bibliography tails are discarded; and
- cited sources/references are derived from the surviving inline markers.

Finalization also derives `evidence_images` and ordered Markdown/Artifact/image
`parts`; transports never trust model-generated Markdown image URLs. Every
published Markdown Artifact is citation-finalized against the same admitted
context and stores its cited sources under that Artifact's resource identity.
Artifact presentations therefore resolve their own citation indexes without
borrowing sources from the chat Answer or another Artifact. Streaming may expose
tokens immediately, but the final `done` result contains normalized text and
authoritative metadata.

Semantic highlights run only after citation validation. They enrich cited source
chunks with phrases from the finalized answer. Web requests them by default;
REST, MCP, and Application callers opt in. `/retrieve` never emits them. Timeout
or failure leaves original sources unchanged.

## Answer Attachments And Resources

Attachments are run-scoped Resources, never workspace data. `ResourceRegistry`
owns inline bytes or lazy public HTTP(S) references for one answer. Research may
also admit a public URL through `read(url=...)`; only `User-Agent`, `Accept`, and
`Accept-Language` are configurable. The shared public-HTTP boundary validates
scheme, credentials, every redirect, resolved addresses, HTTPS downgrade, byte
limits, and pixel limits. Validated DNS targets are pinned for the connection.
Settled fetched bytes and their canonical locators replay from owner-scoped blob
storage after recovery rather than making a second network request.

`read` is deterministic:

- UTF-8 and CSV decode directly;
- HTML, PDF, DOCX, PPTX, and XLSX use selected offline MarkItDown converters;
- OOXML archives pass zip-bomb preflight;
- each acquired URL is one fixed snapshot for the run; and
- opaque signed cursors continue bounded text without exposing offsets or
  provider locators. A focused first window starts near the best match, then
  continuation rotates through the entire document without skipping content.

`inspect` sends a bounded image or PDF page to the VLM role (default model as
fallback) and marks output `derived_by_vlm` with its exact locator. It does not
accept arbitrary bounding boxes; page and embedded-visual handles provide
structural narrowing.

When Exa or Tavily is configured, Research can search Web passages as peer
evidence through one provider-neutral tool. Search and Extract use independently
ordered failover chains. Failover occurs only for provider failures, never to
seek a subjectively better result; malformed individual results are dropped and
reported, and an empty successful result stops the chain. Result URLs become
inert resource handles and are fetched only by explicit `read` under normal
guards. Direct anonymous HTTP is attempted first; the configured Extract chain
is a bounded internal fallback, not a model-visible tool. DlightRAG supplies no
cookies, authenticated browser session, or Playwright; callers must attach
protected bytes/screenshots.

Web uploads are content-addressed blobs owned by durable runs. Follow-up
re-registers historical attachments lazily, newest first within limits. There
is no parsed attachment table or vector cache. Deleting conversations/runs and
retention cleanup release only blobs with no surviving reference.
