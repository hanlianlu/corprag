# Treat a Product Document as one consistency unit

## Status

Accepted, implemented, and fault-validated. The Product Document visibility
barrier and durable Corpus Mutation Run lifecycle are implemented; the
[Slice 6 matrix](../validation/run-runtime-slice-6.md) records phase-boundary,
repair, and supported PostgreSQL 18 convergence evidence.

## Context

A Product Document is not only a LightRAG document. It also has DlightRAG-owned metadata and readiness, source locators, BM25 labels, and, when enabled, visual fusion. Reporting success after changing only LightRAG leaves a document partially visible or partially deleted.

## Decision

Corpus Mutation actions use stable `track_id` identities, durable checkpoints, and idempotent phases across every relevant projection. `ingest`, `replace`, `delete`, `retry`, and `reset` succeed only after the public LightRAG contract reports completion and required DlightRAG projections and requested physical file effects have completed. DlightRAG inspects and trusts upstream public outcomes; it never claims to prove arbitrary LightRAG storage internals, snapshots or restores them, or directly rewrites LightRAG document status.

Original source bytes, thumbnails, and parser sidecars remain in LightRAG/RAGAnything's `input_dir` file layout on a deployment-provided shared POSIX volume. DlightRAG does not introduce an artifact-store abstraction for these files. Source locators and digests are narrow product projections, not ownership of a second file lifecycle.

A document becomes visible on DlightRAG's direct document surfaces only when
`_dlightrag_finalization_complete` is exactly true. Missing, NULL, and false
markers fail closed. Semantic and graph-referenced chunks, BM25, direct visual
results, metadata search/schema/counts, processed-file listing, source download,
and full/thumbnail visual reads enforce that marker. PostgreSQL pushes the
predicate into bounded reads; non-pushdown storage over-fetches a bounded
candidate window and post-filters, preferring temporary recall loss to leaking
unfinished documents. Normal queries otherwise follow LightRAG's
eventual-consistency contract during mutation; DlightRAG does not add a
cross-Pod read/write snapshot gate around the upstream graph.

Delete and replace hide the old document before the first destructive LightRAG call. Visibility is restored only when upstream gives a clear, contract-tested rejection proving that no write occurred. A public `DeletionResult` of `success` or `not_found` completes the upstream contract leg; internal storage flush durability remains LightRAG-owned. An ambiguous partial destructive result keeps the Run `running` with `phase=waiting_for_repair`, releases its local execution slot, and blocks later mutations in that Workspace. The only destructive escape is an administrator-confirmed Corpus Reset that records which stuck Run it supersedes.

Replace is a non-atomic, resumable delete-then-ingest operation: stage the replacement at a Run-exclusive source path, require confirmed upstream delete success or `not_found`, enqueue with the stable `track_id`, and resume finalization from durable checkpoints. File and sidecar cleanup cannot precede confirmed deletion. Direct ingestion into the same LightRAG Workspace outside DlightRAG is unsupported because it bypasses the consistency barrier.

## Consequences

Projection failure is a failed Run with preserved per-document and per-phase results, never a successful partial result. Retry first reconciles upstream state by `track_id`. Already processed documents retry only missing DlightRAG projections. A LightRAG `FAILED` document is retried through inspected public per-document deletion followed by retained-source re-enqueue with the Retry Run's stable `track_id`; the private process-local all-failed retry mailbox is not a product contract. Requested source deletion is a completion condition and an uncertain ownership or I/O result cannot be reported as success.
