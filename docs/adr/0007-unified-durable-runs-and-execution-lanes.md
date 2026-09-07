# Unified durable Runs and execution lanes

## Status

Accepted, implemented, and failure/load validated. Slice 1 established common
Runtime and Answer, Slice 2 added top-level Retrieval, Slice 5 moved Corpus
Mutation onto the same RunRuntime, and the [Slice 6 validation](../validation/run-runtime-slice-6.md)
confirmed the lane bounds and failure behavior.

## Context

At decision time, DlightRAG split caller-awaited Retrieval, durable Answer Runs,
and durable Ingest Jobs across different lifecycle and concurrency machinery.
That split could not give a deployment one coherent overload, recovery,
cancellation, or observation model for 10,000 online users.

## Decision

One storage-neutral `RunRuntime` owns the lifecycle of the closed `run_kind` values `retrieval`, `answer`, and `corpus_mutation`. These are values on one Run record, not separate runtime classes or modules. Corpus Mutation actions are `ingest`, `replace`, `delete`, `retry`, and `reset`. Creation remains use-case-shaped, returns HTTP 202, and shares `/runs/{run_id}` status, cancellation, and event observation. A stable submission key and normalized fingerprint make identical retries return the existing Run and changed input conflict.

The only public states are `queued`, `running`, `succeeded`, `failed`, and `cancelled`. A multi-document Corpus Mutation with any failed document is `failed` and retains every per-document result; there is no partial lifecycle state or successful partial outcome. An ambiguous destructive mutation stays `running` with `phase=waiting_for_repair` until it can safely resume or an administrator explicitly supersedes it with Corpus Reset.

`RunRuntime` has a Query Lane for Retrieval and Answer and a Corpus Mutation Lane for all five mutation actions. Query is only an execution lane; there is no `QueryRun` aggregate or runtime. An Answer's internal retrieval is a Retrieval Stage, not a nested Run. Slice 1 configures the Query lane with per-process concurrency 16, an atomic deployment-wide active ceiling of 16, and a deployment-wide nonterminal acceptance fuse of 30,000.

Across the deployment, at most one Corpus Mutation Run owns a given Workspace at a time. While it executes LightRAG's in-process pipeline it holds its Run lease and active permit; LightRAG owns parallelism within that pipeline. Different Workspaces may execute concurrently. A deferred or `waiting_for_repair` Run releases compute capacity but preserves the Workspace mutation barrier, so later mutations cannot compound uncertain state.

## Consequences

The migration was sliced. Slice 1 removed the Answer-only lifecycle and moved
Answer through the common Runtime. Slice 2 made top-level Retrieval durable on
the same Query Lane while keeping Answer's internal Retrieval Stage direct.
Slice 5 moved Corpus Mutation execution onto its dedicated lane in the same
Runtime; Slice 6 retained the Query `16 / 16 / 30,000` and Corpus Mutation
`2 / 2 / 1,000` bounds from controlled evidence. No parallel Answer, inline
top-level Retrieval, or Ingest Job lifecycle
is retained as a compatibility path. The implementation keeps the combined Application
process topology and existing writer/reader capabilities: only writer-capable
processes will claim Corpus Mutations, while operationally writable service
processes may accept them durably. No ingress-only or worker-only process modes
are introduced without evidence.

This preserves ADR 0001's Application → Engine dependency direction: Runtime accepts generic prepared envelopes and injected executors rather than importing Answer or RAG request types.
