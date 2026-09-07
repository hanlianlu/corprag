# Public Web Resource Acquisition and Provider-Neutral Web Sources

Research admits a known public HTTP(S) URL through `read`, not through a separate browser-opening tool. The Host turns a successful read into a durable Web Resource and Evidence snapshot. Web Search and hosted extraction use provider-neutral seams with separate Exa and Tavily adapters; arbitrary Bash network output remains non-citable model context.

## Status

Accepted and implemented. Current-state architecture, security, retrieval, and configuration documentation describe the landed vertical slice.

## Context

Research already has two partial paths to public Web content. Exa Web Search discovers sources and can provide hosted content fallback, while `ResourceRegistry` directly fetches caller URL attachments through HTTP machinery owned by RAG corpus ingestion. A trusted Agent can also run `curl` through Bash, but Bash records process output and workspace effects rather than a Host-attested binding between a public locator, admitted content, and Evidence identity.

Adding Agent-selected URL reads and a second provider exposes four mismatches:

1. caller URL attachments, search results, and Agent-selected URLs are acquisition origins of the same kind of resource, not three unrelated source kinds;
2. public-URL transport safety is shared mechanics, but it currently lives under Corpus ownership;
3. the Exa implementation and the model-facing Web Search tool share one provider-specific file;
4. an Exa-specific key, usage field, and fallback path cannot describe ordered Exa/Tavily capabilities.

Browser automation is a different capability. It owns rendered state and interaction, whereas this decision concerns read-only source acquisition and evidence admission.

## Decision

### Web Resource and `read(url)`

A **Web Resource** is a run-scoped public HTTP(S) source represented by a Resource Handle. Its resource kind is Web; its admission origin (`caller`, `search`, or `agent`) and acquisition method (`direct_http`, `exa_extract`, or `tavily_extract`) are independent provenance facts. Web Search discovers Web Resources but is not the direct reading of one.

Extend the existing `read` tool with a URL input. One call validates, acquires, admits, and returns the first bounded view plus its Resource Handle; no separate `open_url` or `register_url` step exists. Existing path and handle inputs remain mutually exclusive with URL input. Generic Agent Core owns only the optional multi-target `read` contract and delegates resource-id or URL targets through a Host callback; it never fetches a URL or imports Answer. A Host without URL-reading capability does not advertise that input branch. Answer owns URL admission rather than registering a second tool named `read`.

Keep the model contract terse and layered. The tool description says only that it reads one workspace path, Resource Handle, or public HTTP(S) URL and that URL reads create or reuse a citable Run snapshot. Field descriptions explain individual inputs; validation enforces target-specific combinations; results print the exact target and cursor required to continue. `path`, `resource_id`, and `url` are mutually exclusive. URL input accepts focus and optional bounded HTTP presentation fields but not a cursor; continuation uses the returned Resource Handle. HTTP overrides are invalid with a cursor, local resource, spill, or existing Web snapshot and are never silently ignored.

The Host, not the model, owns the admission transaction:

```text
normalize URL
  -> validate public HTTP(S) target
  -> acquire a bounded representation
  -> persist the admitted bytes or extracted text
  -> bind locator, digest, origin, and acquisition metadata
  -> issue/reuse a Resource Handle
  -> admit Evidence
  -> settle the tool effect
```

The same normalized URL reuses the first successfully admitted snapshot within one Answer Run, including its Child Sessions. A failed attempt creates no Evidence and does not pin a failure; a later call may retry. Agent-supplied URLs, caller link handles, and Search-discovered handles converge on one Resource Registry acquisition path; the model tool does not own a second fetch implementation. The model-facing reader must not pre-materialize every link through visual inspection before reaching the fallback-aware acquisition path. Direct HTTP uses the existing Resource Registry format dispatch for text, HTML, PDF, Office, and visual resources. Hosted extraction is a fallback for suitable textual Web content and never substitutes extracted text for original binary bytes.

ResourceRegistry-backed handles receive a durable catalog and admitted-representation record rather than process-local identity. Direct bytes and the complete exact hosted-extraction text are content-addressed before the owning Tool Effect settles; recovery hydrates the same Resource Handle and representation without another network call. Continuations are durable or self-contained and remain valid after process recovery. This closes the existing gap between persisted fetched blobs and the documented Resource Handle recovery contract; Workspace spills keep their separate Answer-owned Agent Workspace Epoch recovery path.

URL-shaped reads are a Research capability independent of Agent Execution Environment mode. They remain available when local execution is `disabled`, beside unrestricted Bash when it is `trust`, and outside any future sandbox's process-egress policy. Fast has no tools and does not gain hidden Web acquisition: a caller-provided document link continues to make Fast invalid, while a caller-provided current-image link may be pinned before execution through its existing input-normalization path. That image path and Corpus URL ingestion reuse shared public-HTTP mechanics but do not acquire Research Evidence or invoke hosted extraction.

Public HTTP and HTTPS are accepted. Direct acquisition is an anonymous GET and sends no model-supplied body, authorization, cookies, or arbitrary header map. For a Web Resource that has not yet acquired a successful snapshot, the Agent may override only `User-Agent`, `Accept`, and `Accept-Language` through explicit bounded fields; values reject control characters, affect Direct HTTP only, and may not replace an already admitted snapshot. This lets the Agent retry a rejected representation without turning `read` into a general HTTP client. The reader does not fetch, interpret, or enforce `robots.txt`, add domain policy, or sleep on `Retry-After`; ordinary HTTP failure feeds the declared Extract fallback or returns an error. URLs with embedded credentials and non-public targets are rejected, every redirect is revalidated, and HTTPS cannot downgrade to HTTP. The final normalized URL is the citable source identity; the requested URL and redirect chain remain acquisition trace, and requested/final identities alias the same Resource Handle. Validation must bind to the actual connection destination or verify its peer address; pre-resolving a hostname and then allowing the HTTP client to resolve it again is not a sufficient DNS-rebinding boundary.

Ordinary query parameters remain part of URL identity because they may select the content. Agent-selected URLs reject known credential/signature parameters. Caller-provided signed URL attachments may transport private bytes, but their complete acquisition URL remains private and their citation locator is the Resource Handle rather than the signed URL. Fragments do not participate in HTTP identity.

### Bash and execution trust

Raw Bash output is model context, not Web Evidence. It receives neither a Web Resource identity nor a Host-attested content-to-URL binding. `curl` therefore remains useful for trusted exploratory work but cannot establish a citable public source by itself.

Trust mode intentionally grants the Agent the host user's process and network authority, so tool guidance cannot enforce egress. Bash and its existing ability to invoke `curl`, Python, or other network clients remain unchanged. Shell-command string filtering is rejected as a security boundary because the same network access is available through other executables and language runtimes. A future sandbox or outbound proxy may enforce egress; this decision does not pretend that trust mode does. This vertical slice adds no general `http_request` tool; a future sandbox may inject one separately without widening `read`.

### Shared public-HTTP mechanics

Move public-URL validation and bounded byte acquisition from RAG Corpus to the private shared module `dlightrag.engine.public_http`. It owns URL normalization, DNS and peer policy, redirect handling, timeouts, and response-size enforcement. Answer and RAG may depend on it; it must not import either owner.

The module does not own document conversion, Resource Handles, Evidence, persistence, Web Search, hosted extraction, or provider fallback. It is also not a generic authenticated API client: Exa and Tavily own their fixed provider endpoints, credentials, request schemas, and response parsing.

A single Engine module is preferred over a new top-level `engine.web` or foundation package. Public HTTP acquisition is shared mechanics, not another peer product owner.

### Provider-neutral Web sources

Replace the provider-specific `answer/tools/web.py` ownership with:

```text
dlightrag/engine/answer/web_sources/
  __init__.py
  contracts.py
  service.py
  exa.py
  tavily.py

dlightrag/engine/answer/tools/web_search.py
```

`web_sources` is named for both discovery and hosted extraction; extraction is not search. Its package facade is narrow. Provider-neutral contracts describe search requests and hits, extraction requests and results, usage, and a closed error taxonomy. `service.py` hides provider selection and fallback; adapters normalize provider responses, while Answer records available cost diagnostics. `exa.py` and `tavily.py` separately own their provider APIs. The model-facing tool is a thin adapter over the service and never selects providers.

Tavily is integrated through its REST API as a first-class provider, not as an outbound MCP tool. It implements separate Search and Extract contracts. Search snippets are admitted immediately as provider-extracted Evidence, with acquisition metadata that does not claim they are full-page snapshots; the Agent reads a returned Resource Handle only when it needs more content. Provider-generated answer prose is not authoritative Evidence; DlightRAG admits normalized source hits or exact extracted representations and retains the actual source URL as the citation locator. Provider identity remains acquisition metadata and trace data.

Search and Extract have independent ordered provider chains. When order is not explicitly configured, it is derived from configured credentials in the canonical `exa`, then `tavily` order. An explicitly named provider without required credentials is a configuration error. Direct HTTP always precedes the hosted Extract chain and is not itself a provider entry. Search stops at the first successful provider response, including a valid empty result; Extract requires a non-empty usable representation. The first implementation does not fan out, merge, or calibrate rankings across providers, and the model does not select a provider.

The model-facing Search contract exposes only provider-neutral research intent: one query, `max_results` (default 10, range 1–20), included/excluded domains, optional start/end publication dates, and relative `effort` (`fast`, `balanced`, or `deep`; default `balanced`). Adapters map effort to the nearest provider-native latency/relevance tier without enabling provider-generated answers. Provider-specific type, depth, topic, and provider choice remain behind the seam. Repeating an identical Search request performs a new provider call; discovery results are not Run snapshots, while Evidence still deduplicates identical passages and URLs. A partially malformed response admits its complete valid hits and reports the dropped count; a structurally valid response with zero valid hits is a successful empty result and does not trigger quality-based fallback.

Fallback proceeds on timeout, network unavailability, rate limiting, exhausted quota, authentication failure, provider server failure, malformed provider response, or empty/failed extraction. It does not trigger from a provider-specific guess that otherwise valid Search results are semantically weak. Provider adapters neither retry, back off, park, nor circuit-break across tool calls; each call tries the configured chain anew. Exhaustion returns one bounded attempt summary without exposing secrets or raw provider payloads. A successful fallback tells the Agent which provider succeeded and gives a short normalized reason for each preceding degradation; available cost and degradation diagnostics remain observable without exposing raw payloads.

Provider configuration moves from the Exa-implied `answer.web_search.api_key` to a provider-neutral `answer.web_sources` namespace containing Search and Extract order plus separate provider credentials. This is a forward-only clean break rather than a permanent runtime compatibility alias.

### Setup journey

`prerequisite_setup.py` gives optional Web providers their own step after required models and document parsing, plus a separate `Web research` entry under Change settings. Skipping is the default and never makes setup incomplete: public `read(url)` remains available, while Search and hosted extraction are reported as unconfigured. The secret-free summary shows Direct URL read availability, effective Search and Extract order, and whether each provider key is set.

The novice path chooses none, Exa, Tavily, or both. A single provider becomes both chains; when both are selected, the user can choose one common order or independent Search/Extract priority. Provider keys are written only to `.env`; behavior and order remain in `config.yaml`. Applying Web settings follows the existing confirm, backup, canonical-load validation, and rollback journey and never resets Corpus data.

The wizard may explicitly migrate a detected legacy `DLIGHTRAG_ANSWER__WEB_SEARCH__API_KEY` to the Exa provider key after confirmation and then remove the legacy entry. This one-time setup migration does not make the old key a runtime alias. Setup validates configuration and reports credentials as `set`, not `verified`; it does not make live provider calls whose transient network or quota failures would block onboarding.

### Bounds and usage

Do not add a cumulative Web-only research quota such as maximum URLs, total Run bytes, or provider calls. Parent and Child Sessions share Resource identity, snapshot deduplication, Evidence, and aggregated usage—not an exhaustible Web allowance.

Mandatory local safety bounds remain: per-response bytes after decoding, redirects, timeouts, model-visible windows, and process-wide concurrency admission. Keep available provider cost and fallback diagnostics provider-neutral; do not invent incomplete cross-provider accounting. If deployment cost governance later requires hard Run limits, design one cross-capability Run Resource Policy covering models, Child Sessions, Web providers, media, and storage rather than a Web-only counter.

## Why this shape

`read(url)` is not valuable because its HTTP client is more capable than `curl`; it is valuable because the Host can attest and durably settle the relationship between locator, representation, Resource Handle, and Evidence. Keeping direct transport, provider APIs, resource admission, and the model tool as separate owners preserves that guarantee without making one Web utility a god module.

Provider-specific files contain unstable external schemas. The narrow service facade prevents Exa or Tavily fields, costs, and failure conventions from leaking into tools, resources, citations, or configuration consumers. Independent Search and Extract chains preserve capability semantics while allowing either provider to be primary or fallback.

Avoiding an arbitrary URL-count quota preserves open-ended research. Retaining per-operation bounds and shared concurrency protects worker availability without confusing system safety with research quality.

## Rejected alternatives

- **Use Bash `curl` as the Web reader.** It provides bytes but no Host-attested source identity, Evidence admission, deterministic snapshot, or enforceable request policy.
- **Add `open_url`.** It suggests navigation or browser state and duplicates the existing Resource reader.
- **Restrict Agent reads to caller or Search URLs.** Prevents following citations and reading known official sources without improving the public-target security boundary.
- **HTTPS only.** Excludes legitimate legacy public sources; HTTP remains explicit provenance with no HTTPS downgrade.
- **Keep shared HTTP under Corpus.** Makes Answer depend on another capability owner's ingestion implementation.
- **Create an `engine.web` owner.** Adds a peer architecture category for shared mechanics and overloads Web with browser, search, and transport meanings.
- **Put Exa and Tavily in `tools/web.py`.** Couples model-tool presentation to two external APIs and widens every future provider change.
- **Expose Tavily only through outbound MCP.** Generic MCP results do not carry DlightRAG's Resource, Evidence, settlement, and provider-usage semantics.
- **Use one provider order for Search and Extract.** Treats distinct capabilities as interchangeable.
- **Fallback on subjective result quality.** Produces unstable cost and provider-specific routing behavior.
- **Add a Web-only cumulative Run budget.** Arbitrarily truncates research and leaves more expensive model work governed differently.
- **Turn `read` into a general HTTP client.** Custom methods, request bodies, credentials, and sessions have different side-effect, replay, and secrecy semantics from citable read-only acquisition. Trust-mode Bash already supplies unrestricted network freedom; a future sandbox may inject a separate non-citable `http_request` capability.
- **Block network commands by shell-string inspection.** Easily bypassed and creates a false security claim.

## Consequences

The configuration key for existing Exa deployments changes and must be documented as a clean migration. Web Search remains unavailable when no Search provider is configured, while direct public URL reads still work and merely lack hosted extraction fallback. Health and traces must use provider-neutral capability and usage terms.

This decision lands before product operation, so changed `read` and `search_web` schemas replace the current contracts directly. The implementation does not add legacy tool variants, active-Run deployment gates, or pinned-plan migration machinery for development data; developers reset or cancel incompatible local Runs when necessary.

Public HTTP extraction must close the current DNS validation/use gap before Agent-selected URLs are enabled. Tests must cover actual-target policy, redirects, decompression/byte limits, same-Run snapshot reuse, failed-attempt retry, full process recovery and continuation, Child evidence admission, provider ordering, effort mapping, partial malformed results, fallback classification, and the absence of Exa/Tavily fields outside their adapters.

Browser automation, rendered-page interaction, authenticated browsing, and general crawling remain outside this decision.
