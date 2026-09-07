# Separate the Run Blob Plane from Operational State

## Status

Accepted and implemented for the current PostgreSQL `PGRunBlobStore` Adapter.
A future object-store Adapter remains target-only.

## Context

Answer attachments, fetched Web resources, spills, and Published Artifact content are DlightRAG-owned immutable bytes. The current PostgreSQL implementation stores them in `dlightrag_blob_chunks` and related `BYTEA` columns beside Run lifecycle, events, leases, and conversations. At scale, byte volume has different backup, streaming, and storage-pressure characteristics from transactional Operational State, even though Operational State must remain authoritative for owner/run scope, digest, references, retention, and visibility.

## Decision

A storage-neutral `RunBlobStore` persists and streams only the Run Blob Plane: Answer attachments, fetched Web resources, spills, and Published Artifact bytes. It never stores deployed corpus originals, thumbnails, or parser sidecars, which remain in LightRAG/RAGAnything's file layout. The Run store persists structured authority and blob references rather than byte content.

The first implementation keeps PostgreSQL chunked `BYTEA` as the only Adapter and adds no object-storage infrastructure. A future object-store Adapter must durably stage complete content before an Operational State transaction references it, verify digest and size, and clean unreferenced staging without treating the Blob Store as an authorization authority.

## Consequences

Operational State migration no longer inherently includes large Run bytes, but must preserve their identities and references. Infrastructure owns object-store provisioning, capacity, backup, and hard quotas; DlightRAG owns content identity, reference safety, access projection, and user-visible retention. Published Artifact settlement and Answer acceptance remain atomic at the authority layer even when byte persistence uses a separate physical system.
