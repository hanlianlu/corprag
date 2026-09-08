# Unified durable Runs and execution lanes

## Status

Accepted, implemented, and validated for the documented local control-plane evidence. Slice 1 established common
Runtime and Answer, Slice 2 added top-level Retrieval, Slice 5 moved Corpus
Mutation onto the same RunRuntime, and the [Slice 6 validation](../validation/run-runtime-slice-6.md)
confirms the local worker bounds, the Corpus Mutation admission limit, and
failure behavior. The 30,000 Query limit remains a configured default/target,
not a reached load-test boundary.

## Context

At decision time, DlightRAG split caller-awaited Retrieval, durable Answer Runs,
and durable Ingest Jobs across different lifecycle and concurrency machinery.
That split could not give a deployment one coherent overload, recovery,
cancellation, or observation model for the recorded 10,000 local Query submissions.

## Decision

One storage-neutral `RunRuntime` owns the lifecycle of the closed `run_kind` values `retrieval`, `answer`, and `corpus_mutation`. These are values on one Run record, not separate runtime classes or modules. Corpus Mutation actions are `ingest`, `replace`, `delete`, `retry`, and `reset`. Creation remains use-case-shaped, returns HTTP 202, and shares `/runs/{run_id}` status, cancellation, and event observation. A stable submission key and normalized fingerprint make identical retries return the existing Run and changed input conflict.

The only public states are `queued`, `running`, `succeeded`, `failed`, and `cancelled`. A multi-document Corpus Mutation with any failed document is `failed` and retains every per-document result; there is no partial lifecycle state or successful partial outcome. An ambiguous destructive mutation stays `running` with `phase=waiting_for_repair` until it can safely resume or an administrator explicitly supersedes it with Corpus Reset.

`RunRuntime` has a Query Lane for Retrieval and Answer and a Corpus Mutation Lane for all five mutation actions. Query is only an execution lane; there is no `QueryRun` aggregate or runtime. An Answer's internal retrieval is a Retrieval Stage, not a nested Run. The Query lane uses 16 workers per process and a deployment-wide nonterminal admission limit of 30,000. Corpus Mutation uses two workers per writer process and an independent 1,000-Run admission limit. Deployment configuration owns process count and total active capacity.

Across the deployment, at most one Corpus Mutation Run owns a given Workspace at a time. While it executes LightRAG's in-process pipeline it holds its fenced Run lease and one local execution slot; LightRAG owns parallelism within that pipeline. Different Workspaces may execute concurrently across writer processes. A deferred or `waiting_for_repair` Run releases its local slot but preserves the Workspace mutation barrier, so later mutations cannot compound uncertain state.

## Consequences

The migration was sliced. Slice 1 removed the Answer-only lifecycle and moved
Answer through the common Runtime. Slice 2 made top-level Retrieval durable on
the same Query Lane while keeping Answer's internal Retrieval Stage direct.
Slice 5 moved Corpus Mutation execution onto its dedicated lane in the same
Runtime; Slice 6 validated Query `16 / 30,000` and Corpus Mutation
`2 / 1,000` with controlled one-process fake-executor evidence: all local worker slots and the mutation limit were exercised, while the 30,000 Query limit and multi-host behavior were not. No parallel Answer, inline
top-level Retrieval, or Ingest Job lifecycle
is retained as a compatibility path. The implementation keeps the combined Application
process topology and existing writer/reader capabilities: only writer-capable
processes will claim Corpus Mutations, while operationally writable service
processes may accept them durably. No ingress-only or worker-only process modes
are introduced without evidence.

This preserves ADR 0001's Application → Engine dependency direction: Runtime accepts generic prepared envelopes and injected executors rather than importing Answer or RAG request types.
