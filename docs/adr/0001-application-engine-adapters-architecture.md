# Application, Engine, and Adapters

DlightRAG exposes three visible code zones plus one private composition root: inbound Adapters call Application use cases and consume transport-neutral contracts from their owning Engine modules, Application owns product orchestration, Engine owns execution and Answer domain contracts, and outbound Adapters implement narrow ports owned by Application or Engine. This keeps the request path short without flattening distinct execution and persistence responsibilities.

## Status

Accepted and implemented for source ownership and dependency direction. Pending product interfaces and persistence changes are governed by later ADRs without changing this decision.

## Decision

The installable product uses these zones:

1. **Application** — product use cases, authorization, caller-facing service projections, configuration, lifecycle, and health.
2. **Engine** — AI, Agent, Runtime, RAG, and Answer execution/domain contracts as sibling owners with an explicit dependency DAG.
3. **Adapters** — HTTP and MCP inbound protocols plus concrete outbound mechanisms such as PostgreSQL and observability.

The ordinary request path is:

```text
HTTP / MCP -> Application -> Engine
```

A private root composition module wires concrete Adapters into Application and then leaves the request path. Application does not import concrete persistence or transport implementations. Inbound Adapters call Application facades and may import transport-neutral Answer contracts from their owning Engine leaf modules; they do not import Answer executors or orchestration internals. No module anywhere under `engine` may import Application, directly or indirectly, and Engine does not import Adapters.

Engine ownership remains a DAG rather than a directory hierarchy:

- AI depends on no other product module;
- Agent may depend on AI;
- Runtime may depend on Agent but not Answer or RAG;
- RAG may depend on AI;
- Answer may depend on AI, Agent, RAG, and Runtime.

A shared Runtime therefore accepts generic prepared envelopes and composition-injected operation executors; it does not import Answer or RAG request models. Concrete persistence is exposed through owner-specific semantic ports rather than through a universal database or corpus interface.

Offline index repair remains an installed PostgreSQL Adapter operator command rather than an Application use case. Its Adapter-side composition may call Engine behavior while writers are stopped; no composition entrypoint lives under Engine and there is no exception to the Engine-to-Application prohibition. Corpus Reset remains an authorized Application operation because it mutates product corpus state through the running service.

Application is the only public in-process Python facade. REST is the public remote interface; internal HTTP clients used by CLI or evaluation remain Adapter machinery rather than a second product SDK.

## Consequences

The source tree communicates ownership directly instead of introducing classification-only layers or feature-local copies of shared persistence. Fast and Research remain Answer strategies, the generic Agent kernel remains product-neutral, and RAG remains the Engine owner that composes upstream LightRAG behavior with DlightRAG-specific corpus semantics.

This ADR intentionally does not freeze REST paths, lifecycle states, configuration keys, storage classes, or persistence schemas. Those contracts are governed by their owning decisions and may change without weakening the Application → Engine dependency direction or the Engine → Application prohibition recorded here.
