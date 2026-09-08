# Run Runtime and Scaling Target

> Accepted and implemented. The bounded local control-plane evidence and its limitations are recorded in [RunRuntime Slice 6 validation](validation/run-runtime-slice-6.md).

## Goals and boundaries

- Mechanically exercise 10,000 local Query control-plane submissions without OOM, deadlock, restart loops, or uncontrolled fake-executor pressure. This is not evidence for 10,000 concurrent online users.
- Replace inline Retrieval, Answer-only durable execution, and Ingest Jobs with one durable Run lifecycle.
- Bound active expensive work and durable backlog without tenant quotas, billing, or per-user fairness policy.
- Keep LightRAG responsible for its own corpus storage composition and pipeline behavior.
- Keep provisioning, replica placement, autoscaling, database and provider capacity, networking, TLS, DDoS/WAF, backup, and disaster recovery outside this repository.

The target is a breaking upgrade. It does not retain parallel legacy lifecycle endpoints or compatibility persistence, and it requires no migration guide for those old product contracts.

## Workload and acceptance assumptions

- One trusted organization, 10–100 Corpus Workspaces, and normally 1–10 Workspaces per Query.
- Expected Query mix is 10% Retrieval, 40% Fast Answer, and 50% Research Answer.
- The deployment-wide Query nonterminal admission limit defaults to 30,000 Runs; the recorded campaign submitted 10,000 Query Runs and did not fill that limit.
- Retrieval `top_k` and `chunk_top_k` remain bounded; current target maxima are ten times configured values, presently 400 and 200.
- The Research working-set guard and Answer attachment maximum are 128 MiB. In trust-mode execution, only filesystem/container infrastructure can enforce a hard storage quota.
- Existing Child Session concurrency remains four per parent. Additional Bash, browser, Web-search, or tenant capacity classes require load-test evidence.

In the recorded one-process local campaign, Run submission while its lane's admission limit has room, status, cancellation, and event-cursor reconnect remained responsive. Accepted Runs are not lost or settled terminally twice, and workers resume draining after offered load falls. Exact latency, memory, event-loop, database, and provider thresholds are evidence-driven rather than invented here.

## Domain model

```text
RunRuntime
├── run_kind=retrieval, lane=query
├── run_kind=answer, lane=query
│   ├── Fast strategy
│   └── Research strategy
└── run_kind=corpus_mutation, lane=corpus_mutation
    ├── ingest
    ├── replace
    ├── delete
    ├── retry
    └── reset
```

A top-level Retrieval is durable. Retrieval inside an Answer is an internal Retrieval Stage under that Answer's capacity and recovery authority, never a nested Run.

**Run** is a durable user-submitted operation with common state, events, cancellation, and terminal result. Kind is data on this one record, not a separate lifecycle class or runtime.

**Query Lane** is the execution lane shared by owner-scoped Retrieval and Answer kinds. There is no `QueryRun` aggregate, class, module, or runtime.

**Corpus Mutation** is a Workspace-scoped run kind whose action is `ingest`, `replace`, `delete`, `retry`, or `reset`.

**RunRuntime** owns common lifecycle, dispatch, recovery, event ordering, and terminal settlement while operation executors own operation-specific behavior and results. It is not a generic workflow engine.

**Execution Lane** is a scheduling and capacity class. The only lanes are Query and Corpus Mutation.

**Operational State** is structured authority for Runs, events, leases, idempotency, Agent Sessions, Web Conversations, Workspace identity/access, and blob references.

**Run Blob Plane** contains only DlightRAG-owned immutable Answer attachments, fetched Web resources, spills, and Published Artifact bytes.

**Product Document** is the consistency unit spanning LightRAG state and DlightRAG metadata/readiness, source locators, BM25 labels, visual fusion when enabled, and retained source/file sidecars.

## Public Run contract

Creation remains use-case-shaped rather than accepting an untyped generic workflow request:

```text
POST /retrieve  -> 202 run handle with kind=retrieval
POST /answer    -> 202 run handle with kind=answer
POST /runs/corpus/ingest  -> 202 run handle with kind=corpus_mutation
POST /runs/corpus/replace|delete|retry|reset -> 202 corpus_mutation run handle
```

Observation is common:

```text
GET    /runs/{run_id}
DELETE /runs/{run_id}
GET    /runs/{run_id}/events
```

Answer-only steer, follow-up, fork, transcript, children, and Artifact interfaces remain associated with an `AnswerRun`. A client disconnect only detaches its observer; it never cancels the Run.

Application use cases own authentication, authorization, normalized input, capability and size validation, retention selection, and result projection. They submit a generic prepared envelope; `RunRuntime` does not import Answer or RAG transport models. Retrieval and Answer are visible to their submitting owner. Corpus Mutations are visible to callers authorized for their Workspace and retain `submitted_by` for audit.

Every submission has a stable client-generated submission key. Uniqueness is scoped by Run kind and submitter. The Application computes a normalized fingerprint; identical key and input returns the existing Run, while changed input conflicts atomically.

Blob ingest is streamed to a bounded temporary file while the server computes SHA-256, then atomically committed before Run acceptance. An optional client `content_sha256` permits an early replay/conflict check and is verified during streaming; it is not mandatory. `RequestBodyLimitMiddleware`, route-specific limits, bounded chunks, and atomic staging enforce body limits. DlightRAG adds no deployment-wide pre-body reservation or upload TTL ledger. Ingress infrastructure owns connection, rate, slow-client, and volumetric controls.

## Lifecycle, events, and retention

The only public lifecycle states are:

```text
queued | running | succeeded | failed | cancelled
```

A Corpus Mutation with one or more failed documents is `failed` and preserves every per-document and per-phase result. There is no partial lifecycle state and no successful partial outcome.

Infrastructure interruption reclaims the same Run through a fenced lease. Transient dependency unavailability records a durable checkpoint and `next_attempt_at`, releases execution capacity, and retries with bounded exponential backoff. Explicit invalid input, unsupported capability, or deterministic operation failure is terminal. The runtime executes declared deferred or terminal outcomes; it does not blindly reinterpret every exception as retryable.

An ambiguous destructive mutation remains `running` with `phase=waiting_for_repair` and an explicit repair reason/remedy. It counts toward the nonterminal admission limit and preserves the Workspace mutation barrier but releases its local execution slot. Repair completion lets the same Run reconcile and continue. An administrator-confirmed Corpus Reset is the only destructive supersession path; it records `superseded_by_run_id` on the waiting prior Run. There is no unsafe unlock that preserves unknown corpus state.

SSE is a lossy notification channel, not an unbounded per-subscriber buffer. The durable ordered Run event log is authoritative. Clients reconnect with a sequence cursor; the `RunStore` provides semantic wait-for-change notification, with bounded polling only as a safety fallback.

Answer terminal history remains 365 days; Retrieval and Corpus Mutation terminal history remain seven days through one per-Run `purge_after` policy. Nonterminal Runs are not retention-pruned.

## Execution lanes and admission

```text
Query Lane
  retrieval + answer run kinds

Corpus Mutation Lane
  ingest + replace + delete + retry + reset
```

Each lane has bounded per-process execution and an independent deployment-wide nonterminal admission limit. Claim ordering favors eligible older Runs, but only Corpus Mutation guarantees durable FIFO within each Workspace; temporarily ineligible work is skipped rather than causing lane-wide head-of-line blocking. Local worker limits bound each process while nonterminal limits prevent a dependency outage from growing a durable queue without bound. Deployment configuration owns process count and therefore total active capacity.

The defaults/targets are 16 Query workers per process and 30,000 Query nonterminal Runs, plus two Corpus Mutation workers per writer process and a 1,000-Run mutation limit. The deterministic one-process, one-database campaign submitted 10,000 Query Runs plus 1,000 recorded mutation Runs to fake executors, exercised occupancy of 16 and two, reached only the mutation admission limit, preserved lane independence and per-Workspace mutation FIFO, and drained. The [validation report](validation/run-runtime-slice-6.md) records exact measurements and limitations. A separate test with multiple coordinator objects sharing PostgreSQL in one test process shows additive object-local slots without double claims; it is not multi-process or multi-host evidence. `ModelScheduler` remains process-local Engine AI protection, LightRAG owns its stage queues and parser/embedding/LLM concurrency, and providers/operators own external quotas and service capacity.

Across the deployment, at most one `corpus_mutation` run owns a Workspace at a time. While calling LightRAG's in-process pipeline, that Run retains its fenced lease and a local Corpus Mutation execution slot. LightRAG owns concurrency within the pipeline. Different Workspaces can be processed by different writer replicas concurrently.

This durable per-Workspace Corpus Mutation FIFO/barrier is unrelated to Agent `AccessScheduler` mutual exclusion, which is process-local, conflict-based, and not FIFO. A deferred or `waiting_for_repair` Run releases compute capacity but keeps the logical Workspace mutation barrier. Later mutations for that Workspace remain queued. This deployment-wide ownership is required because LightRAG's process-shared pipeline coordination covers a pre-fork process group, not independent replicas.

The combined Application process topology remains. Writer-capable processes claim Corpus Mutations; reader-capable processes do not mutate corpus state. A service process that may write Operational State can durably accept an authorized mutation even when no writer or corpus dependency is currently healthy, provided the lane's nonterminal admission limit has room. No ingress-only, Query-worker, or mutation-worker process mode is added without evidence.

## Corpus Mutation contract

### Common rules

Every mutation uses a stable LightRAG `track_id`, durable phase checkpoints, idempotent DlightRAG finalizers, and inspected upstream outcomes. It commits `handoff_started_at` before non-idempotent upstream effects. Recovery first calls `aget_docs_by_track_id(track_id)`:

- existing upstream state is reconciled instead of duplicated;
- missing state with a complete staged source is re-enqueued with the same `track_id`;
- an incomplete or missing required source fails closed.

Cancellation is allowed only before the durable `handoff_started_at` compare-and-set. After that point the mutation is not cancellable because LightRAG may already have effects. Cancellation never promises rollback.

Source bytes from multipart upload, URL, S3, or other supported acquisition land in the Workspace `input_dir` layout. They remain there after processing. Only an explicit delete/reset removes originals and sidecars, and requested physical deletion is a completion condition rather than best effort.

### Ingest

Ingest stages source content to a Run-exclusive path, records `handoff_started`, enqueues with the stable `track_id`, executes the in-process LightRAG pipeline while holding capacity, then completes every required DlightRAG projection. A batch attempts its declared documents and records each result; any failed document makes the Run `failed`.

### Delete

Delete first sets document readiness false so DlightRAG direct document surfaces stop exposing it, then calls public `adelete_by_doc_id` and inspects the complete `DeletionResult`. Confirmed `success` or `not_found` completes the upstream contract leg. A contract-tested rejection that proves no upstream write occurred may restore visibility. A result indicating partial or uncertain mutation enters `waiting_for_repair`.

After upstream confirmation, deletion idempotently clears DlightRAG metadata/source locators, BM25 and visual relationships, and requested source/sidecar files. Success means every DlightRAG-owned and requested file effect has completed; internal LightRAG storage durability remains upstream-owned. LightRAG 1.5.7 can log a final `_insert_done()` failure after returning public success, so observable delete convergence under the supported default PostgreSQL composition is an integration gate rather than a reason to call private flush functions. Ownership uncertainty prevents physical deletion and cannot produce false success.

### Replace

Replace is deliberately non-atomic and resumable rather than pretending to snapshot LightRAG. It stages the replacement at a Run-exclusive path, hides the old document, confirms `adelete_by_doc_id` success or `not_found`, then enqueues the replacement with its stable `track_id` and completes all projections. It never deletes old files/sidecars before upstream deletion is confirmed and never restores LightRAG document status from a snapshot.

### Retry

Retry accepts explicit document IDs or a selector for all currently retryable documents. On its first successful execution, it snapshots the exact cohort into the durable checkpoint; later recovery never absorbs newly failed documents.

The cohort includes LightRAG `FAILED` documents and LightRAG `PROCESSED` documents whose DlightRAG finalization is incomplete. A processed-but-unfinalized document retries only its missing DlightRAG phases. A failed document is hidden, its complete owned retained source is verified, public `adelete_by_doc_id` must return `success` or `not_found`, and that source is re-enqueued without physical deletion using the Retry Run's stable `track_id`. This path inherits replace's destructive checkpoints and repair behavior. Already ready documents are idempotent successes; identity or source mismatch fails closed. DlightRAG never uses LightRAG's private process-local all-failed retry mailbox or directly rewrites document status.

### Corpus Reset

Corpus Reset is a Workspace-scoped FIFO mutation barrier. Earlier ordinary mutations finish before it; later submissions execute against the empty corpus. It removes the Workspace's LightRAG corpus state, DlightRAG corpus projections and maintenance counters, and source/sidecar files. It preserves durable Run/event history, Conversations, Agent Sessions, historical Artifacts, Workspace access control, and logical Workspace identity.

An administrator may explicitly use Reset to supersede a `waiting_for_repair` Run. The Reset then owns the existing barrier; if Reset itself becomes ambiguous it enters `waiting_for_repair` rather than reporting success.

## Product Document consistency and visibility

A Product Document is ready only after every enabled required phase completes:

- LightRAG processing reported complete through its public contract;
- DlightRAG metadata and source-locator finalization;
- BM25 language/label finalization;
- visual fusion when enabled;
- retained source and sidecar lifecycle effects requested by the action.

The `_dlightrag_finalization_complete=true` marker gates DlightRAG direct document visibility. If product finalization fails after LightRAG reaches `PROCESSED`, that native status remains unchanged and retry replays only incomplete idempotent finalizers. Semantic and lexical chunk retrieval, visual retrieval, metadata filtering/search/counts, file panels, and downloads must enforce it. Metadata field statistics count only finalized rows. A storage implementation without pushdown may over-fetch and post-filter; temporary recall loss is safer than leaking an unfinished document.

This marker does not turn LightRAG's shared graph into an MVCC snapshot. Query Runs may execute during mutation and follow LightRAG's eventual-consistency contract; shared entity/relationship data can change incrementally. Mutation success promises final convergence across all projections, not atomic query snapshots during processing. DlightRAG therefore adds no deployment-wide Query-versus-mutation read/write gate.

Direct upload or scan into the same LightRAG Workspace outside DlightRAG is unsupported. Such a write has no authoritative DlightRAG metadata, source intent, or completed consistency barrier and remains excluded from DlightRAG product surfaces.

## LightRAG storage and files

LightRAG owns the deployment-static composition named by:

```text
kv_storage
vector_storage
graph_storage
doc_status_storage
```

DlightRAG resolves these names through LightRAG registration and `verify_storage_implementation()`. Defaults remain exactly:

```text
PGKVStorage
PGVectorStorage
PGTableGraphStorage
PGDocStatusStorage
```

`PGTableGraphStorage` is mandatory as the default graph choice; it must not drift to AGE or `PGGraphStorage`. The only explicit vector alternative is `MilvusVectorDBStorage`; Zilliz uses that adapter's compatible URI/token rather than a new class. Writer support is deployed. Reader processes remain PostgreSQL-vector-only because LightRAG 1.5.7 exposes no public nonmutating external-vector attach lifecycle.

The current explicit database credential/environment bridge remains. Resolved DlightRAG deployment bindings have explicit override priority over inherited LightRAG environment values; unset bindings leave upstream defaults intact. Configuration that requests an unsupported combination fails startup rather than silently degrading a feature.

DlightRAG owns narrow auxiliary persistence and behavior only: metadata governance/filtering, BM25 labels, visual fusion, source locators, authorization projection, federation, and composite completion. PostgreSQL may remain the first implementation for these narrow ports without becoming a universal corpus abstraction.

Original sources, thumbnails, and parser sidecars use LightRAG/RAGAnything's `input_dir` and file lifecycle on a deployment-provided shared POSIX volume. DlightRAG adds locator/digest and authorization projections but no separate artifact-storage seam and no duplicate corpus copy in `RunBlobStore`.

Changing storage composition remains external operator/infrastructure responsibility. DlightRAG does not provision resources, dual-write, copy data, track transfer watermarks, or coordinate infrastructure cutover. Drain and cutover automation are not repository-owned scope.

## Persistence seams

Operational State remains distinct from LightRAG corpus storage and from Run bytes. Each owner keeps a narrow semantic port such as `RunStore`, `AgentSessionRepository`, `ConversationStore`, `MemoryStore`, or `WorkspaceRegistry`. The private composition root selects a coherent Adapter family, currently PostgreSQL. Callers do not receive a universal CRUD store, and cross-owner atomicity is exposed only through a purpose-built narrow transaction seam.

`RunStore` owns atomic acceptance and idempotency conflict detection, claims, fenced leases, events, cancellation settlement, deployment-wide nonterminal admission limits, Workspace mutation ownership, wakeup, and logical pruning. SQL transactions, `SKIP LOCKED`, advisory mechanisms, indexes, and notifications remain private to the PostgreSQL Adapter.

`RunBlobStore` persists and streams only DlightRAG-owned Answer attachments, fetched Web resources, spills, and Published Artifact bytes. Operational State remains authoritative for their owner scope, digest, references, visibility, and retention. The first implementation keeps PostgreSQL chunked `BYTEA`; a future object-store Adapter requires complete-before-reference staging and orphan cleanup but adds no authorization authority.

The mutable Agent Workspace remains Answer execution state rather than Operational State or the Run Blob Plane. Final Published Artifacts enter `RunBlobStore`; uncommitted workspace files do not.

## Health and responsibility split

Operational State is required for admission, lifecycle, leases, and event authority. Its unavailability makes the service not ready. Corpus storage, parser, or provider degradation does not take down the control plane: the bounded component view reports `degraded`, eligible submissions are accepted while their lane's nonterminal admission limit has room, and dependency interruptions durably defer when operation policy permits. Liveness reports process health independently and performs no dependency I/O.

The repository owns:

- Application authorization, request preparation, normalized fingerprints, bounded streaming, and product projections;
- common Run lifecycle, dispatch, recovery, lane admission, and Workspace mutation ownership;
- DlightRAG-specific metadata, BM25, visual, source-locator, federation, evidence, and consistency semantics;
- storage-neutral narrow Operational and blob ports with current PostgreSQL Adapters;
- health, readiness, degradation, and overload signals.

LightRAG/RAGAnything owns:

- KV/vector/graph/document-status interfaces and storage registration;
- parser routing, staged ingest, pipeline recovery, KG/vector construction, and core retrieval;
- source/input layout, parser sidecar identities, and rebuild meaning.

Deployment infrastructure and external operators own:

- database, vector service, parser, model provider, ingress, and filesystem provisioning and capacity;
- replica count, placement, autoscaling, networking, TLS, connection/rate/slow-client policy, and DDoS/WAF;
- credentials, backup, restore, disaster recovery, and blue-green storage transfer/cutover execution;
- consistently mounted configuration and shared POSIX volume guarantees.

The repository does not inspect an orchestrator, create service clusters, infer infrastructure capacity, or prescribe new worker deployments.

## Implementation and evidence boundary

The canonical domain-language, architecture, interface, operations, security,
configuration, PostgreSQL, and RunRuntime documentation describes the deployed
common Runtime, durable Retrieval/Answer/Corpus Mutation behavior, and Product
Document visibility barrier. The old Ingest Job lifecycle, unchecked deletion,
document-status snapshot restoration, and best-effort required projections are
not compatibility paths.

The evidence decisions are closed in the [Slice 6 validation report](validation/run-runtime-slice-6.md): Corpus Mutation retains two workers per writer process and a 1,000-Run deployment-wide nonterminal admission limit; broad survival tripwires and measured latency/memory/drain results are environment-described rather than product SLOs. The deterministic fake-executor campaign covers Retrieval across 1, 10, 50, and 100 Workspaces and the repository-owned failure/recovery matrix. The default PostgreSQL 18 integration gate verifies observable delete convergence across `PGKVStorage`, `PGVectorStorage`, `PGTableGraphStorage`, and `PGDocStatusStorage` without claiming a storage-neutral proof of LightRAG internals. Queue latency remains measured rather than a pass/fail criterion; service survival, bounded pressure, durable control-plane responsiveness, and eventual drain are hard gates.
