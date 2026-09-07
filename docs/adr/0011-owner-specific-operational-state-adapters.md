# Keep Operational State behind owner-specific semantic Adapters

## Status

Accepted, implemented, and fault-validated for the current PostgreSQL Adapter
family and its purpose-built cross-owner transaction seams. The
[Slice 6 matrix](../validation/run-runtime-slice-6.md) includes controlled
accept, claim, heartbeat, authorization, and readiness failures.

## Context

Operational State contains distinct authorities: Runs, Agent Sessions, Web Conversations, Memory, Workspace identity and access, model catalogue state, and blob references. Making that plane one caller-facing `OperationalBackend` would create a broad shallow Interface and merge Application/Engine owners merely because PostgreSQL currently persists them together.

## Decision

Each owner retains a narrow semantic persistence port such as `RunStore`, `AgentSessionRepository`, `ConversationStore`, `MemoryStore`, or `WorkspaceRegistry`. The private composition root selects a coherent Operational Adapter Family, currently PostgreSQL. Cross-owner atomicity is represented only by purpose-built narrow transaction seams, such as accepting a Web turn and its Run together; callers never receive a universal CRUD or transaction Interface.

## Consequences

Infrastructure may replace the Operational database, but a new repository Adapter family must satisfy the owner contracts and required atomic compositions rather than one monolithic backend contract. This preserves the Application → Engine dependency DAG and keeps PostgreSQL transactions private to concrete Adapters.
