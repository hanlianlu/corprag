# Keep corpus storage composition owned by LightRAG

## Status

Accepted and implemented in Slice 4. The [Slice 6 validation](../validation/run-runtime-slice-6.md)
adds supported PostgreSQL 18 observable delete-convergence evidence without
claiming storage-neutral proof of LightRAG internals.

## Context

LightRAG already composes KV, vector, graph, and document-status storage. Wrapping that composition in a second DlightRAG-wide storage abstraction would duplicate upstream contracts, misrepresent Milvus/Zilliz as a complete corpus replacement, and make DlightRAG responsible for infrastructure migration.

## Decision

The corpus storage composition is deployment-static and remains LightRAG's responsibility. DlightRAG keeps the four configured storage class names—`kv_storage`, `vector_storage`, `graph_storage`, and `doc_status_storage`—and resolves them through LightRAG's registration and `verify_storage_implementation()` contract.

The defaults remain exactly `PGKVStorage`, `PGVectorStorage`, `PGTableGraphStorage`, and `PGDocStatusStorage`. `PGTableGraphStorage` is the default graph implementation and must never be silently replaced with AGE or `PGGraphStorage`. Milvus or Zilliz may replace only `vector_storage` when explicitly selected; they are not complete peer corpus implementations.

DlightRAG retains its explicit database-credential/environment bridge. Resolved DlightRAG deployment bindings take precedence and are deliberately applied over inherited LightRAG environment values; unset bindings continue to use upstream behavior. Unsupported configured combinations fail startup rather than silently dropping a requested capability.

DlightRAG owns only its narrow product behavior and auxiliary projections: metadata governance and filtering, BM25 labels, visual fusion, source locators, federation, authorization projection, and composite mutation completion. It does not add an aggregate corpus backend, per-Workspace storage binding or epoch, migration coordinator, or complete-backend conformance suite.

A storage change remains an operator/infrastructure blue-green procedure. DlightRAG does not provision databases, execute infrastructure migration, dual-write, or manage cutover state. The durable Corpus Mutation Run executor is implemented and validated; operators and infrastructure, not this repository, own drain and cutover automation.

## Consequences

PostgreSQL remains the default composition, including `PGTableGraphStorage`; selecting `MilvusVectorDBStorage` changes only the vector leg. Zilliz uses that same adapter with a compatible URI/token. Writer processes use LightRAG's public storage lifecycle. Reader processes remain PostgreSQL-vector-only because LightRAG 1.5.7 exposes no public nonmutating external-vector attach lifecycle. Infrastructure owns topology, capacity, credentials, backup, and recovery, while LightRAG owns the storage interfaces and DlightRAG remains limited to product semantics it actually adds.
