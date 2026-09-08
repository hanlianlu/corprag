# RunRuntime And Durable Query And Corpus Mutation Execution

This document owns the common Retrieval, Answer, and Corpus Mutation Run lifecycle, fencing,
recovery, event persistence, resource blobs, and the Web conversation adapter.
Public endpoint shapes live in
[Interfaces](interfaces.md); retrieval and generation behavior lives in
[Retrieval and Answer](retrieval-answer.md); PostgreSQL deployment details live
in [PostgreSQL](postgresql.md).

`dlightrag.engine.runtime` owns storage-neutral lifecycle records, store ports,
fenced sessions, subscriptions, cancellation listening, and `RunCoordinator`.
Composition registers operation-specific Retrieval, Answer, and Corpus Mutation
executors; Runtime does not import their request models. `PGRunStore` implements the operational
runtime port and `PGRunBlobStore` implements the immutable PostgreSQL `BYTEA`
blob seam without creating an Engine dependency on PostgreSQL.

## Guarantees And Limits

- One operation-neutral `RunRuntime` owns every durable run ID and lifecycle;
  top-level Retrieval and Answer executors share the Query Lane, while ingest,
  replace, exact delete, retry, and reset use the Corpus Mutation Lane.
- An Answer's internal Retrieval Stage executes directly under that Answer's
  capacity and recovery authority; it is never a nested Run.
- Disconnecting a client never cancels its Run.
- Retrieval recovery re-executes pinned normalized input; Answer recovery
  restores complete typed Agent Operation state or exact Fast stages.
- Committed Answer effects do not execute again; pending effects replay only
  under the pinned plan and contract.
- Every interface observes the same common state and terminal result. REST,
  same-origin browser, MCP, Python client, and Application observers use the
  same Run identity and durable sequence.
- Corpus mutations are FIFO within one Workspace and may proceed concurrently
  across Workspaces. Their per-process worker bound and deployment-wide
  nonterminal admission limit are independent from the Query Lane.

DlightRAG does **not** promise exactly-once execution for an interrupted
read-only tool batch or exactly-once token generation before final result
staging. It adds no external workflow framework, detached/background agents,
permission platform, sandbox backend, MCP registry/OAuth service, or object
storage shim.

## Lifecycle

`POST /retrieve`, `POST /answer`, and Corpus Mutation acceptance validate and
persist a Run before returning a descriptor (HTTP surfaces return 202). There
is no inline remote Retrieval result, temporary Answer mode, `stream` request
field, or separate ingest lifecycle.

```text
accept  -> Run + bounded immutable prepared input
claim   -> oldest lane-eligible row; fencing epoch++; lease
execute -> operation-owned phases/checkpoints and durable events
finish  -> canonical result + exactly one terminal event (one transaction)
recover -> reclaim an expired lease and execute from durable authority
```

Retrieval acceptance pins its normalized query/options, authorized Workspace
set, the required Extract and optional VLM profiles/fingerprints, capability
facts, and policy revisions. Recovery can therefore repeat planning and search
without trusting mutable request state. Answer acceptance additionally commits routing, Session
state, and attachment references through its purpose-built transaction seam.

An execution slot is one of `runtime.query.worker_concurrency` local runs. The
coordinator reserves a slot **before** claiming a row, so a worker never holds a
lease while waiting for local capacity. The Query default is 16 workers per
process; Corpus Mutation defaults to two workers per writer process. Multiple
processes contribute additive local slots, while PostgreSQL row locks prevent a
Run from being double-claimed. Deployment configuration owns process count and
total active Run capacity. Model-provider and LightRAG pipeline concurrency are
independent from both Run lanes. The [Slice 6 validation
report](validation/run-runtime-slice-6.md) records the failure matrix,
admission-limit rejection, single-process occupancy, 10k-client measurements,
broad survival thresholds, and limitations.

A free worker claims the oldest eligible queued or expired-running row with
`FOR UPDATE SKIP LOCKED`. It sweeps bounded batches at startup, after local
completion, and once per second so work from another host does not depend on a
process-local wakeup.

Accepted Retrieval and Answer Runs queue while Query slots are busy, up to the
deployment-wide `runtime.query.max_nonterminal_runs` nonterminal admission limit
(default 30,000). Corpus Mutation Runs queue independently up to
`runtime.corpus_mutation.max_nonterminal_runs` (default 1,000). Reaching a lane's
admission limit rejects new acceptance before storing a Run; already
accepted work remains durable. Answer has no wall-clock timeout. A top-level
Retrieval's `corpus.retrieval.timeout` begins only after it is claimed and bounds
its planning/search execution, not queue residence; expiry fails that Run with
`retrieval_timeout`. Individual LLM, embedding, rerank, URL, resource, and
parser calls retain their own timeouts.

## Leases, Fencing, And Recovery

Every claim increments a monotonic fencing epoch. A heartbeat renews only an
unexpired lease matching owner and epoch. If a guarded write updates no row, the
worker rereads it; expiration or owner/epoch divergence means lease loss. The
stale worker cancels and joins local work, performs no transition, and releases
its slot. It never revives a lease.

Recovery uses two distinct counters:

- `durable_progress_version` advances only on fenced model-turn, compaction,
  effect, or Fast-stage settlements.
- `reclaims_without_progress` counts consecutive expired-lease reclaims with no
  such progress.

Four consecutive no-progress reclaims fail the run as `run_abandoned`. A long
run that settles progress between crashes may survive more than four restarts.
The bound is not configurable.

The sweeper needs no execution slot to finalize a cancel-pending row without a
live lease or abandon a no-progress run. Cancellation takes precedence over
reclaim and abandonment.

### Corpus Mutation Handoff And Repair

A Corpus Mutation keeps one stable `track_id` for upstream reconciliation. It
commits `handoff_started_at` before the first destructive or otherwise
non-idempotent LightRAG effect. An expired lease after that point cannot merely
repeat the effect: the executor first reconciles authoritative public LightRAG
state. It proceeds only when reconciliation proves the effect is complete or
safe to continue.

If reconciliation cannot prove a safe outcome, the same nonterminal Run enters
`waiting_for_repair`, releases its lease and local execution slot, and exposes bounded
`repair_reason` and `repair_remedy` fields. It is not failed and no replacement
Run is created. An authorized operator repairs upstream state, then explicitly
resumes this same Run through REST, same-origin browser, MCP, Python, or the
Application Run service. Resume records repair confirmation and requeues only
that identity.

A full Corpus Reset may instead name one waiting Run to supersede. The old Run
becomes terminal `repair_superseded`, linked to the reset Run. Reset hides first,
preserves Corpus Workspace identity and mutation history, and then clears
corpus content. It is not Workspace Delete.

Ingest finalization has a separate Product Document publication barrier. If
LightRAG has committed `PROCESSED` but DlightRAG finalization is incomplete, the
upstream status remains `PROCESSED` while metadata keeps the document hidden.
Retry selectors include that document and execute only the missing idempotent
same-ID finalization path; product code never rewrites LightRAG status.

### Graceful Shutdown

The coordinator first stops claiming. Active workers may finish a settlement or
terminal transaction during shutdown grace. Then remaining work is cancelled
and joined:

- owned cancel-pending runs become terminal `cancelled` with `done`;
- other owned nonterminal runs return from `running` to `queued`, clearing the
  lease while preserving durable state and cancellation fields.

This handoff does not increment crash-reclaim counters. Generation interrupted
by shutdown or crash emits `reset` when reclaimed.

## Cancellation And Controls

Deleting a queued Run terminalizes it immediately. Deleting a running Run sets
`cancel_requested_at`; the coordinator signals the owning worker, and executors
also observe cancellation at their stable boundaries. A Corpus Mutation rejects
cancellation once its upstream handoff has started unless an operation-owned
safe settlement wins the row-lock race. If the lease expires
first, the sweeper terminalizes it.

Cancellation and successful finalization serialize on the run-row lock.
Success is allowed only while no cancellation is pending. Cancelling a terminal
run is an idempotent no-op. REST returns 200 when terminal and 202 while a live
worker still must observe cancellation.

Steer instructions enter an ordered inbox and are consumed at stable checkpoints
as durable `ControlMessage` entries. A terminal-race steer/follow-up creates a
fresh linked Operation. Fork creates a new Lane in the same Agent Session.
Endpoint details are in
[Interfaces](interfaces.md#run-lifecycle-and-answer-endpoints).

Closing an SSE subscriber or cancelling a caller-awaited Application
`RetrievalService.retrieve()` / `stream()` or `AnswerService.answer()` /
`answer_stream()` call only detaches the caller. Explicit Run cancellation is
the sole client action that sets `cancel_requested_at`.

## Idempotency And Pinned Input

Run IDs are UUIDv7. Every accepted envelope has a submission key unique within
`(run_kind, submitted_by)`; when a caller omits its optional key, the Application
uses the generated run ID:

- REST: `Idempotency-Key`
- MCP/Application: `idempotency_key`
- Web Answer submission: `submission_id`

A matching normalized replay returns the existing run with current status;
conflicting input returns 409. No key always creates a new run. The key expires
with the run row.

The fingerprint hashes the kind-specific normalized public input. Retrieval
includes query, authorized Workspace set, limits, lexical/filter options,
federated-rerank choice, and query images; Answer additionally includes bounded
history and ordered Resource descriptors. It excludes headers, temporary paths,
authorization-dependent projected URLs, secrets, and later execution output. A
replay returns before planning, URL fetches, image descriptions, or history
projection are repeated.

The immutable prepared input stores the execution facts each kind needs. Both
pin relevant endpoint fingerprints, effective model profiles, and
catalogue/context-policy revisions. Retrieval also retains current-image bytes
only while nonterminal; its terminal accepted envelope keeps their count and
SHA-256 identities, not their bytes. Answer retains its accepted
history/Resources/scope, image descriptions, and Agent Plan. Recovery uses these
pinned facts; provider credentials remain deployment state. An incompatible
model fingerprint, model-catalogue revision, or context-policy revision fails
closed at startup and again before execution rather than silently running with
changed semantics.

## Durable Events

Events have gap-free, monotonically increasing per-Run sequences. Retrieval
and Corpus Mutation emit `progress` plus one terminal `done` or `error`; Corpus
Mutation also persists handoff, deferral, and repair phases on the Run. Answer may additionally
emit `token`, `reset`, and `tool_start` / `tool_progress` / `tool_end`.

Appending locks the Run row, consumes its next sequence, and checks live lease
owner/epoch. Answer token text is coalesced into bounded chunks. Tool events
store only name, status, elapsed time, output byte count, spill state, call
identity, and attachment count—never stdout/stderr.

A terminal transaction stores status/error/result and appends exactly one
terminal event:

- success: `done` with complete canonical result;
- cancellation: `done` with `status="cancelled"`, no result;
- failure: `error` with public kind/message.

SSE closes after replaying the terminal event. Retrieval publishes no contexts
before its terminal result. Intermediate Answer contexts are not published
because Research may change them. `progress` is last-writer-wins and may move
backward after recovery. Answer `reset` invalidates all previously streamed
draft text before a tool-bearing turn, provider retry or failure, continuing
follow-up/correction, interrupted regeneration, or canonical citation/Artifact
rewrite. Only a successful `done.result` is terminal result authority.

Stored results/events contain transport-neutral source identities, never
projection URLs or inline image bytes. REST rechecks current permissions and
projects fresh download/visual URLs; MCP keeps download URLs null and projects
permitted visual routes. Neither modifies stored events. Trusted Application
callers provide an explicit `RetrieveProjection` when they want a projected
Retrieval result. Event retention may end before the Run row: then SSE returns
410 while status continues to expose the result.

## PostgreSQL State

### `dlightrag_runs`

One operation-neutral row owns:

- globally unique UUIDv7 run ID, `run_kind`, execution `lane`, submitter, access
  scope, mandatory submission key, and request fingerprint;
- bounded `prepared_input_json` while queued/running;
- status, phase, stop reason, cancellation time;
- lease owner/expiration and fencing epoch;
- durable progress/reclaim counters and next event sequence;
- retention policy, `purge_after`, retry eligibility, and opaque executor checkpoint;
- final result or terminal error; and
- created/updated/started/finished/event-trim timestamps.

Answer routing and continuation lineage remain in Answer-owned projection tables.

The row is the sole authority for lifecycle. Research state lives in the Agent
Session's immutable parent-linked entries and closed typed registers. Fast stage
state is stored under deterministic stage identities.

Every accepted Run stores its own retention selection and receives
`purge_after` at terminal settlement. Answer uses the configured
`runtime.run_retention_days` floor (default 365 days); top-level Retrieval and
Corpus Mutation use seven days. Nonterminal Runs are never retention-pruned. Sweeps use bounded
`SKIP LOCKED` batches. Conversation turns do not extend Answer retention.
Deleting the last routed Answer Run makes its Agent Session tree eligible for
cleanup; shared Sessions and child trees still referenced by Runs survive.

### `dlightrag_run_events`

Rows are keyed by run and sequence and cascade with the run. Worker writes
require the active lease/epoch. Queued cancellation and sweeper transitions use
the same lock/predicate, preventing duplicate terminal events.

### `dlightrag_blobs` And `dlightrag_blob_chunks`

Raw bytes are content-addressed inside one owner namespace. Every non-final
chunk is exactly 1,048,576 bytes. One transaction writes all chunks, then inserts
the blob metadata; metadata therefore means complete. There is no independent
blob-retention clock and no cross-owner deduplication.

Accepted uploads link atomically during run acceptance. A fetched Web Resource
links only after HTTP(S)/redirect/DNS/SSRF/byte validation settles its effect.
Its resource catalog row retains the canonical locator, provenance capabilities,
and replay ordinal. Recovery reads those stored bytes without live DNS validation
because it performs no network request; a settled Resource remains one fixed
snapshot, while an acquisition that admitted no Evidence may retry later.

### `dlightrag_answer_run_artifacts`

This join table records ordered request attachments and terminal Published
Artifacts: safe filename, MIME type, ordinal, resource kind, and deterministic
transform locator. Fetched Web snapshots use `dlightrag_answer_resources` plus
the shared Blob tables instead. The artifact table references `(owner, digest)` with
`ON DELETE RESTRICT`, so insertion takes the PostgreSQL key-share lock that
serializes against blob deletion.

Deleting a run removes only its references. Blob deletion occurs in one
transaction only when no references survive; the foreign key protects a
concurrent reuse. Deterministic conversion is recomputed from stored bytes.
VLM inspection prose remains run Evidence rather than a cross-run cache.

### `dlightrag_answer_artifact_attachments`

One row per `(owner_id, run_id, relative_path)` is the durable Root Artifact
Attachment authority. It records the display label, raw SHA-256 digest and byte
size, presentation capability, originating Session/Effect Intent, attachment
time, and monotonic settlement order. It is distinct from
`dlightrag_answer_run_artifacts`: attachment rows authorize workspace roots;
run-artifact rows reference owner-visible published bytes and other run
resources.

The row cascades with its run. Reattaching the same path replaces the authority
and assigns the latest settlement order. Workspace inventory refresh or deletion
does not erase the attachment: terminal publication must instead compare it
with the current raw bytes and fail closed when the file is missing or stale.

## Agent Session Recovery

Each routing row authorizes one Agent Session/Lane. Immutable entries preserve
parent-linked ancestry; Lane registers select branches without copying shared
history. Research restores complete `OperationState` and invokes the same pure
`NextAction` interpreter used live.

Before a provider call, Runtime commits the exact request snapshot and attempt.
Assistant settlement records the complete response and ordered Tool Batch Plan.
Tool clearance, effect settlement, ToolResult placement, Host deltas, and
progress then commit under the lease/epoch predicate.

Recovery treats effects by contract:

- `replayable`: reconcile or dispatch again under the unchanged contract;
- `never`: settle as `outcome_unknown`;
- changed contract: settle `tool_contract_changed` without dispatch.

`attach_artifact` is replayable because it only validates current workspace
bytes and produces authority through settlement. Its model-visible `ToolResult`
and `ArtifactAttachmentUpdate` commit atomically; a crash cannot commit one
without the other.

Image state stores resource/corpus identities, never data URIs. A missing corpus
visual drops only its image while preserving text/citation; a missing attachment
blob fails the run.

`spawn_agent` is replayable because child IDs derive from parent effect intent
and terminal roster rows persist parent-visible outcome/Evidence. Replay
re-merges that state without creating or driving another child. An interrupted
ordinary read-only batch may execute again.

### Fast Recovery

Fast shares the Session Entry Tree but never enters the Agent interpreter.
Acceptance atomically appends `UserMessage` plus `HostTurnReservation`. Before
assistant settlement, the Host stages the complete canonical result at a
deterministic final-generation stage. If the process crashes after the assistant
commit, recovery terminalizes from the staged result without retrieval or
generation.

Failure before staging clears the reservation and preserves unanswered input.
After staging, failure/cancellation preserves it so replay can commit the exact
assistant without Lane interleaving. Interrupted pre-staging generation emits
`reset`; token generation is not exactly-once.

## Web Conversation Adapter

A Web conversation owns principal-scoped navigation/history, not execution.
The run-creation transaction inserts/reuses the run, input blobs/artifact
references, and one conversation turn keyed by `submission_id`. Admission
failure leaves no empty conversation, and history exists before HTTP 202.

History returns chronological keyset pages (newest 40 by default, maximum 100
per request). Queued/running turns remain resubscribable pending entries; failed
and cancelled turns remain until run retention. Only succeeded turns become
model history, projected from the run rather than copied.

A follow-up adds a linked turn. Fork atomically opens a conversation branch with
parent lineage. Conversation deletion removes linked runs in one transaction;
workers cannot append after the fenced rows disappear. Cascades remove events
and references, then ownership-safe cleanup removes unreferenced blobs.

If retention removes the last turn/routed run before the empty conversation row
is swept, a later reuse starts a fresh `main` Lane and Session. Conversation
identity never preserves hidden model history.

Browser SSE uses the same event sequence, resume, 410, and detach semantics as
REST; only terminal projection differs (`AnswerPresentation` rather than stored
canonical JSON). The browser reconciles ambiguous acceptance through the
owner-scoped submission lookup and never blindly repeats POST.

## Reader Role And Artifact Topology

A `reader` is corpus-read-only, not operationally read-only. It can execute
Retrieval and Answer Runs and write operational state, events, Answer Artifacts,
and conversations, while CorpusAdmin and the LightRAG pool reject corpus
mutation/DDL. Both roles use the same writable primary; writers migrate before
readers validate schema and serve traffic.

LightRAG parser artifacts use `file://` paths under the configured working
directory. Multi-process/multi-host deployments therefore mount one shared
POSIX artifact tree at the same absolute path. Direct object-storage resolution
is outside this contract.

## Failure And Security Rules

- Owner scope applies to every run, event, and Artifact lookup.
- Workspace authorization is resolved before acceptance; pinned workspace scope
  is not retroactively revoked by later policy changes.
- Attachment count, item, total-byte, and pixel limits apply before acceptance.
- Deployments must monitor PostgreSQL/blob growth and enforce ingress limits;
  DlightRAG intentionally has no aggregate queue byte quota.
- Public URLs retain scheme, redirect, DNS, SSRF, and byte checks.
- Tool errors shown to models are sanitized; operator traces retain detail.
- Duplicate run-local tool names fail before a model call.
- A stale worker cannot append, commit, or delete after lease loss.

The contract is verified by unit state-transition tests; PostgreSQL integration
tests for claiming, fencing, per-Workspace mutation FIFO, cross-Workspace
concurrency, post-handoff recovery, repair resume, cancellation, pruning, and ownership;
transport/reconnect tests; process-restart tests; and the full `make ci` gate.
