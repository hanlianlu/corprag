# DlightRAG Domain Language

DlightRAG is a durable research-and-retrieval product that joins reusable AI, agent, and corpus capabilities without merging their ownership.

## Product Surfaces

**Application**:
The local in-process DlightRAG product surface that exposes Answer, retrieval, and corpus-administration capabilities and owns their shared lifetime.
_Avoid_: Manager, service locator

**Answer**:
The default owner-visible terminal prose of an Answer Run, shown in the conversation surface. It directly satisfies the request unless a separate deliverable is warranted; when a Published Artifact carries the complete deliverable, the Answer is a concise orientation and handoff rather than a second copy.
_Avoid_: Answer Run, Published Artifact, parallel report body

**Run**:
A durable submitted operation with one common identity, lifecycle, event log, cancellation path, and terminal result. `run_kind` and `lane` are data on this record rather than subclasses.
_Avoid_: job, task, `QueryRun`, per-operation lifecycle record

**RunRuntime**:
The single operation-neutral lifecycle coordinator and store contract. It dispatches Retrieval and Answer executors on the Query Lane and Corpus Mutation executors on the Corpus Mutation Lane by `run_kind`.
_Avoid_: Query runtime, Answer runtime, Ingest Job coordinator, generic workflow engine

**Query Lane**:
The execution and capacity lane for retrieval and answer kinds. It is not an aggregate, class, module, or separate runtime.
_Avoid_: QueryRun, query service

**Corpus Mutation Lane**:
The bounded execution and capacity lane for Workspace-scoped ingest, replace,
exact delete, retry, and reset Runs. Runs are FIFO within one Corpus Workspace
and may execute concurrently across Workspaces; the lane has a separate fuse
from the Query Lane.
_Avoid_: Ingest Job queue, per-action coordinator, Workspace Delete

**Corpus Mutation Run**:
A Workspace-scoped Run with `run_kind=corpus_mutation` and
`lane=corpus_mutation`. It keeps a stable `track_id`, records the upstream
handoff before non-idempotent corpus effects, and retains its terminal row for
at least seven days.
_Avoid_: Ingest Job, corpus task, per-action lifecycle

**Answer Run**:
An owner-scoped Run with `run_kind=answer` and `lane=query`.
_Avoid_: separate Answer lifecycle, request

**Answer Service**:
The product capability that validates and accepts Answer input and owns Answer-specific steer, follow-up, fork, transcript, child roster, and terminal projection. Common status, events, cancellation, and listing belong to the Run service.
_Avoid_: Answer manager, transport-specific runtime service

**Continuation**:
A new durable Answer Run with `parent_run_id` and kind `follow_up` or `fork`. Follow-up includes the selected answer as context; fork starts a sibling branch from its accepted context.
_Avoid_: in-place run mutation, session checkout, hidden conversation copy

**Retrieval**:
An owner-scoped Run with `run_kind=retrieval` and `lane=query` that returns corpus Evidence without generating an Answer. Top-level Retrieval is durable; retrieval inside an Answer is an internal Retrieval Stage, never a nested Run.
_Avoid_: Fast Answer, inline Retrieval, `QueryRun`

**Fast Answer**:
A durable Answer Run that plans, retrieves, and generates without an Agent Operation or research workspace. Its Host turn still commits User and Assistant Entries to the routed Agent Session.
_Avoid_: Retrieval, non-durable fast path

**Web Conversation**:
The browser conversation surface that creates Answer Runs; it is a transport, not a research capability. The thread shows the run's terminal answer and opens its Published Artifacts in Artifact Canvas.
_Avoid_: Web Search, Web, Web Channel when meaning public-Web retrieval; chat column as the report

**Web Search**:
The optional capability that discovers open-Web sources and admits their passages as Evidence inside an Answer Run. It is not the direct reading of a known Web Resource.
_Avoid_: Web UI, Web Conversation, Web Channel, URL read

**Answer Mode**:
The caller-facing selector `auto | fast | research`. Omitted means `auto` on REST, MCP, Web Conversation, and Python.
_Avoid_: research bool, inferred path, Web Search as a mode

**Valid Mode Set**:
The capability-derived subset of `{fast, research}` that one Prepared Input may legally resolve to.
_Avoid_: requested mode, router guess, heuristic from Web Search being configured

**Resolved Mode**:
The durable `fast` or `research` value written once after routing; crash recovery does not recompute it.
_Avoid_: Answer Mode when the request was `auto`

**Corpus Administration**:
The product capability for corpus workspace lifecycle, ingestion, files, metadata, visual assets, and reset. Product mutation acceptance belongs to the Corpus Mutation Service; common observation and cancellation belong to Run Service.
_Avoid_: Workspace manager, direct WorkspaceRag call from an interface

**Upstream Handoff**:
The durable Corpus Mutation boundary committed before a destructive or otherwise
non-idempotent LightRAG effect. Recovered work after handoff proceeds only after
authoritative public reconciliation or explicit repair confirmation.
_Avoid_: checkpoint-only hint, private status mailbox

**Waiting For Repair**:
The nonterminal `waiting_for_repair` phase of the same Corpus Mutation Run when
public reconciliation cannot prove a safe repeat. Public `repair_reason` and
`repair_remedy` guide an authorized operator, who repairs upstream state and
explicitly resumes that same Run.
_Avoid_: failed replacement Run, automatic destructive replay, new retry job

**Reset Supersession**:
An authorized full Corpus Reset may explicitly supersede one
`waiting_for_repair` mutation. The old Run becomes terminal with its superseding
Run identity; reset preserves Corpus Workspace identity and history rather than
recreating the Workspace.
_Avoid_: Workspace Delete, hidden repair abandonment

**Authorized Workspace Set**:
The concrete canonical Corpus Workspace ids a caller may use after authentication and access expansion are complete.
_Avoid_: Workspace selector, raw claims

**Access Rule**:
One deployment-configured allow mapping from a verified JWT claim value to
Workspace patterns and Action patterns. Rules are not durable user memberships.
_Avoid_: Database role, Workspace membership, deny policy

**Action Preset**:
A named expansion (`reader`, `editor`, or `admin`) into product Actions inside an
Access Rule. It is not a stored role assignment and never bypasses the rule's
Workspace match.
_Avoid_: Role record, user role, permission group

## Corpus And Research

**Corpus Workspace**:
A named, authorized corpus scope whose ingestion and retrieval state is owned by one corpus runtime.
_Avoid_: Workspace when it could mean an agent filesystem

**Product Document**:
One directly visible corpus consistency unit: a LightRAG document plus required
DlightRAG metadata/source locator, BM25 labels, retained source/sidecar effects,
and enabled visual fusion. It is published only while
`_dlightrag_finalization_complete` is exactly true. A LightRAG `PROCESSED` row
whose product finalizers are incomplete stays upstream `PROCESSED`, remains
hidden through this marker, and retries only the same idempotent finalization
path; DlightRAG does not rewrite upstream status. Shared graph summaries remain
eventually consistent rather than document-snapshot isolated.
_Avoid_: LightRAG processed row, metadata row, graph snapshot

**Agent Session**:
The owner-scoped durable parent-linked Entry Tree shared by routed Fast and Research Answer Runs. Stable Lane heads select branches, and the tree remains durable only while at least one Answer Run routing row names it; a Web Conversation identity alone is not a second history authority.
_Avoid_: Answer Run, linear journal, thread

**Agent Session Repository**:
The narrow Host-facing read/transaction seam for immutable Session snapshots. Runtime mutations go through exact-register CAS transactions; Hosts do not receive append/fork/archive storage shortcuts.
_Avoid_: generic store, lifecycle hook registry, mutable transcript

**Child Session**:
A first-class foreground Agent Session inside one parent Answer Run. It has explicit parent/call lineage and ContextSnapshot, bounded depth and concurrency, pinned plan/model/tool allowlist/budget/Host state, and an independently renewed lease and fencing epoch.
_Avoid_: child run, detached job, mission, background swarm

**Subagent Roster**:
The status/wait/cancel projection for Child Sessions created by `spawn_agent`. Child Evidence is admitted into the parent ledger before the parent effect settles.
_Avoid_: Delegate Research, workflow engine, child queue

**Prepared Input**:
The immutable, bounded execution description accepted for a new Answer Run after authorization, capability resolution, and profile pinning. Resolve never rewrites it.
_Avoid_: Request payload, checkpoint

**Routing Record**:
The Answer-owned one-to-one durable mapping from an Answer Run to requested/valid/resolved mode and its canonical Agent Session/Lane. It is the transcript authorization anchor and does not make Answer Run and Agent Operation identities equivalent.
_Avoid_: Prepared Input, second policy source, inferred research bool

**Public Request Fingerprint**:
The digest of the normalized caller request used for owner-scoped idempotency before Prepared Input is built. Canonical Answer Mode is part of that digest: omitted equals `auto`.
_Avoid_: Prepared-input hash

## Long-Term Memory

**Memory Record**:
An owner-scoped, non-citable remembered preference or fact that may be recalled across separate conversations. It is never Evidence and never a citation source.
_Avoid_: Compaction Summary, Journal Entry, PriorTurns, Evidence, Artifact, remembered citation

**Memory Subject**:
The owner identity a Memory store scopes every operation to; bound by the host (DlightRAG: JWT owner or stable local single-user owner; stdio MCP: `--subject`) and never accepted from a model tool argument. Shared simple-auth callers are not personal Memory subjects.
_Avoid_: caller-selected namespace, conversation id, Agent Session

**RecallResult**:
The structured outcome of one query-aware Memory recall: selected records (exact matches pinned first, then chronological), the raw leg candidates the fusion consumed, degradation flags, and the recalled body character cost (rendering overhead excluded). Never a prompt fragment.
_Avoid_: prompt text, standing block, packed string

**Memory Operation**:
One host-bound, owner-scoped remember, forget, or undo request carrying a stable idempotency key and trusted provenance. The storage seam validates, settles, and journals it atomically; an Answer Run may impose an atomic mutation limit without changing the package interface.
_Avoid_: Silent promotion, transcript scan, model aside, adapter-owned mutation

**Memory Operation Receipt**:
The replay-stable result of one Memory Operation: change identity, action, outcome, affected record identities, safe body, provenance, and supersede/undo links. Answer projects it into a durable product event; Agent Core never interprets it.
_Avoid_: parsed tool prose, telemetry payload, UI-only notification

**Memory Operation Journal**:
The package-owned owner-scoped ledger that makes operation replay, changed-input rejection, mutation caps, optimistic conflicts, and compensating Undo one transaction with record transitions.
_Avoid_: Answer Event Log, Agent Journal, browser cache, out-of-band undo snapshot

**Memory Record Lifecycle**:
`active → superseded` or `active → forgotten`. Forget is idempotent and leaves a non-recallable tombstone; Undo is a new compensating operation rather than a reverse state transition. Non-active history follows the shared retention floor. Clear Profile Memory is the explicit exception: it physically removes that owner's package records and operation journal without rewriting Answer or Conversation history.
_Avoid_: state rollback, silent transcript promotion, confidence score

**Retention Floor**:
The selected per-Run clock bounding terminal Run rows and event logs. Answer uses the configured selection (365 days by default), which also bounds routed Agent Session history and superseded Memory history; top-level Retrieval and Corpus Mutation each use seven days. Reclamation may happen later, never earlier.
_Avoid_: deadline, SLA, per-aggregate TTL, inactivity expiry

**Conversation History Page**:
One signed turn-number keyset page from Web history. It bounds one read, not retention or exact model recovery.
_Avoid_: snapshot, max_turns, trim window, retention window

**File Panel Page**:
One workspace-bound signed keyset page of finalized Product Documents whose
LightRAG status is processed, in durable server order. It bounds
presentation/read work, not inventory or retention; failed-file administration
is a separate repair view.
_Avoid_: full file snapshot, OFFSET page, workspace catalog scan, retention window

## Web Conversation Streaming

**Event Cursor**:
The client's consumed position in one Answer Run's event stream, carried across reconnects. It bounds replay, not retention, and never outlives its run.
_Avoid_: Conversation History Page, offset, bookmark, retention marker

**Attachment Lease**:
The exactly-once settlement of a locally attached file's ownership when a submission is handed to an Answer Run: accept keeps it with the stored turn, restore returns it to the composer, discard releases it. A lease is never shared between submissions.
_Avoid_: upload, blob registry, attachment copy

**Optimistic Turn**:
The local turn a Web Conversation shows while a submission handshake is in flight. It is swapped for the stored turn on acceptance and dropped on failure.
_Avoid_: placeholder message, fake reply, client-side Answer Run

**Turn Projection**:
The browser-side fold of one Answer Run's event stream into a presentable turn view. Replaying the same events yields the same view.
_Avoid_: answer parser, incremental rendering state, second transcript

**Wire Format**:
The snake_case field spelling of REST and SSE payloads at the transport boundary. It is translated exactly once at the consuming edge and never appears beyond it.
_Avoid_: DTO leak, dual naming convention, camelCase server contract

## Session Entries And Effects

**Session Entry**:
An immutable semantic fact with one physical `parent_entry_id` in an Agent Session Tree. The closed variants are UserMessage, AssistantMessage, ToolResult, ControlMessage, and Compaction; transaction sequence records commit order but never defines ancestry.
_Avoid_: generic journal row, custom entry, mutable message

**Agent Session Tree**:
The immutable Entry set plus stable Lane heads and Lane state. Branch ancestry follows parent links; Fork creates a new Lane at a stable checkpoint without copying or deleting shared Entries.
_Avoid_: linear log, DAG merge, navigation operation

**Context Projection**:
The bounded model-facing projection of one selected Lane ancestry. Exactly one active branch-local compaction summary precedes the retained suffix; historical summaries remain immutable audit facts and never Evidence.
_Avoid_: authority, checkpoint, transcript snapshot

**Context Contribution**:
A typed model-ready contribution with source, authority, citable and compressible facts. It unifies projection ordering without merging the contributor's storage ownership.
_Avoid_: global prompt registry, second state authority

**Compaction Summary**:
Typed continuation memory for one contiguous branch-ancestry prefix, validated and rendered by the framework but never treated as citable Evidence.
_Avoid_: Free-form checkpoint, evidence summary, Memory Record

**Effect Intent**:
A durable declaration of one validated tool operation and the replay contract under which it may execute.
_Avoid_: Tool call when referring to durable recovery identity

**Effect Settlement**:
The atomic durable outcome of an Effect Intent, including its ordered result and any host-owned updates.
_Avoid_: Tool result when referring to the full commit boundary

**Durable Progress**:
A monotonically increasing run fact advanced only by live fenced execution boundaries that recovery must not repeat.
_Avoid_: Heartbeat, phase, turn count, recovery interrupt, Workspace Epoch handoff

**Fast Stage**:
A deterministic Answer Run progress boundary inside Fast. Fast shares the Agent Session Entry Tree through a HostTurnReservation but creates no Agent Operation.
_Avoid_: Agent step, retrieval request, Session replacement

## Evidence And Resources

**Evidence**:
Citable, run-scoped source material with durable identity and content/locator integrity.
_Avoid_: Summary, agent prose

**Web Resource**:
A run-scoped public HTTP(S) source admitted from a caller, Web Search, or an Agent-selected URL and represented by a Resource Handle. Admission origin and acquisition method are independent provenance facts. Within one Answer Run, the same normalized URL resolves to its first successfully admitted durable snapshot rather than silently refetching mutable content. Raw Bash output is model context, not Web Evidence.
_Avoid_: Web Search result, URL attachment, raw URL, Bash output

**Resource Handle**:
An opaque owner/run-scoped identity through which prepared, fetched, evidence-backed, or spilled content remains addressable across recovery.
_Avoid_: File path, URL, blob id

**Blob**:
Owner-scoped immutable content addressed by digest and stored independently from the references that keep it reachable.
_Avoid_: Artifact when referring only to stored bytes

**Orphan Blob**:
A complete Blob that currently has no run or Resource Handle reference and is therefore eligible for grace-delayed cleanup.
_Avoid_: Failed artifact

**Spill**:
Private full tool output retained for the life of an active Answer Run and addressed only through a Resource Handle.
_Avoid_: Artifact, report, Journal Entry, Blob when referring to the handle

**Root Artifact Attachment**:
A run-scoped publication authorization settled by the parent Research Session for one normalized Agent Workspace path at an exact raw-content digest. It carries the root's display label and settlement order; answer links may place it but cannot create the authorization.
_Avoid_: Published Artifact, Workspace file, `artifact:` link, upload attachment

**Published Artifact**:
An optional owner-visible, run-scoped output descriptor created by fenced publication of an authorized Root Artifact Attachment or its validated dependency bytes. It is a separate reading, presentation, or download surface for a user-requested file, an impractically long or structurally rich deliverable, or content materially improved by that surface. It carries stable resource identity, validated media type, presentation capability, publication status, and an owner-scoped data plane without exposing a Workspace path.
_Avoid_: Answer, parallel Answer, routine Research by-product, Spill, Blob when referring to the reference rather than the bytes, unattached workspace file

**Artifact Canvas**:
The Web Feature that opens any presentable Published Artifact and owns loading, safe renderer selection, side/wide/fullscreen layout, focus restoration, and preview teardown.
_Avoid_: Report Pane, universal Panel abstraction, same-DOM active HTML

**Active HTML Preview**:
An explicit opt-in rendering mode for a self-contained HTML Artifact inside an opaque-origin, script-enabled sandboxed iframe. It is isolated from DlightRAG credentials and DOM; its CSP blocks normal external loads, but it is not a server execution sandbox or an absolute network-egress guarantee.
_Avoid_: trusted content, same-origin iframe, Sandbox service, zero-egress claim

**Publication**:
The fenced terminal transaction that makes staged Agent Workspace files owner-visible as Published Artifacts.
_Avoid_: Staging, Spill settlement, a second model call

**Agent Loop**:
The product-neutral event-driven turn cycle that stops when the model emits no tool call or cancellation is observed; provider errors produce an error stop. Research hosts it through durable boundaries; Fast does not enter it.
_Avoid_: workflow engine, max-agent-turn policy, READY protocol, Fast Answer

## Execution And Workspace

**Execution Environment**:
The adapter behind exactly three modes: `disabled`, host-trusted `trust`, and `sandbox`. Trust grants the Agent the host user's process and network authority, so evidence-tool guidance cannot enforce network egress. Sandbox is the only seam that may enforce egress policy and has an explicit unavailable failure unless trusted host code supplies a backend.
_Avoid_: implicit downgrade, permission catalog, approval prompt, shell-command filtering as a security boundary

**Agent Skill**:
A progressively disclosed `SKILL.md` package discovered from the operator-global root or one owner's published skills; owner names shadow global for that owner. Metadata is projected first; contained references are read only through `load_skill` and Skill code is never executed. Users write their own skills only through the validated `publish_skill` tool.
_Avoid_: owner Profile Memory, marketplace plugin, arbitrary extension

**Outbound MCP Tool**:
A deployment-declared and allowlisted remote tool invoked through a foreground stdio or streamable-HTTP MCP session.
_Avoid_: MCP registry, marketplace, OAuth platform

**Agent Workspace**:
The model-visible filesystem rooted at the active Workspace Epoch's workspace directory.
_Avoid_: Corpus Workspace, working_dir, workspace when it could mean a corpus scope

**Workspace Epoch**:
One filesystem generation of an Agent Workspace; recovery copies a stable observation of the recorded epoch and never executes in the old one.
_Avoid_: Durable Progress, Fencing Epoch, checkpoint

**Workspace Inventory**:
The current Workspace Epoch's path, type, size, and digest observation of an Agent Workspace.
_Avoid_: Journal Entry, checkpoint, historical epoch listing

## Configuration And Deployment

**Application Configuration**:
The non-secret operator choices that govern DlightRAG product behavior and integrations independently of how a process is packaged or scheduled.
_Avoid_: credentials, container topology, duplicated environment override

**Deployment Binding**:
A value created by the runtime topology or process role to connect DlightRAG to its environment, such as service discovery or a listener binding. It adapts Application Configuration to the deployment without becoming a second owner of product policy.
_Avoid_: application default, product setting, secret

## Operations

**Fencing Epoch**:
The monotonically increasing write-generation of one Run lease; every durable write is predicated on the current epoch and a live lease.
_Avoid_: Workspace Epoch, Durable Progress

**Full Development Reset**:
The explicit replacement of development database and local runtime/corpus data; see [Operations](operations.md#full-development-reset).
_Avoid_: Migration, Workspace reset, compatibility cutover
