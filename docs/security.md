# Security

This document owns authentication, identity-provider boundaries, authorization,
ingress responsibility, resource handling, and browser Artifact isolation. Field
defaults live in [Configuration](configuration.md); public payloads in
[Interfaces](interfaces.md).

DlightRAG verifies credentials and maps claims to workspace/actions. It does
not issue OAuth tokens, manage users/passwords, or provide an identity-provider
login system.

## Authentication Modes

| Mode | Intended use |
|---|---|
| `none` | Loopback development only |
| `simple` | One shared bearer behind a trusted internal boundary |
| `jwt` | Externally issued, user-scoped signed tokens |

A non-loopback REST/MCP listener with `none` is refused unless
`access.allow_insecure_no_auth: true`. With browser credentials, replace wildcard
CORS with explicit origins.

### Simple Bearer

```yaml
# config.yaml
access:
  auth_mode: simple
```

```bash
# .env or an orchestrator Secret
DLIGHTRAG_ACCESS__API_TOKEN=<generated-by-openssl-rand-base64-32>
```

Clients send `Authorization: Bearer <generated>`. REST may accept `X-User-Id`
for request scope; MCP remains one anonymous principal. `simple` is admission
control, not multi-user authorization.

### Static JWT

JWT mode requires `sub`, which becomes `user_id`.

```yaml
# config.yaml
access:
  auth_mode: jwt
  jwt_algorithm: HS256
```

```bash
# .env or an orchestrator Secret
DLIGHTRAG_ACCESS__JWT_VERIFICATION_KEY=<key-or-public-pem>
```

Use the shared secret for `HS*`; use issuer public-key PEM for `RS*`/`ES*`.
Issuer/audience claims are validated when configured. DlightRAG never signs,
renews, or mints tokens.

### JWKS / OIDC

Prefer JWKS for rotating keys from Entra, Okta, Auth0, Keycloak, and similar
issuers:

```yaml
access:
  auth_mode: jwt
  jwt_algorithm: RS256
  jwt_jwks_url: https://login.example.com/.well-known/jwks.json
  jwt_issuer: https://login.example.com/tenant/v2.0
  jwt_audience: api://dlightrag
```

`issuer` and `audience` are required with `jwt_jwks_url`. Audience may be one
value or a list; any match passes.

### MCP OAuth Discovery

DlightRAG can advertise its MCP HTTP listener as an OAuth 2.1 resource server;
the external issuer still authenticates users and issues tokens.

```yaml
access:
  auth_mode: jwt
  jwt_algorithm: RS256
  jwt_jwks_url: https://auth.example.com/.well-known/jwks.json
  jwt_issuer: https://auth.example.com
  jwt_audience: api://dlightrag-rest
interfaces:
  mcp:
    transport: streamable-http
    resource_server_url: https://rag.example.com/mcp
```

This publishes RFC 9728 metadata and a `WWW-Authenticate` discovery challenge.
The externally reachable URL cannot be inferred from a bind address. For native
MCP OAuth it is also the exact expected token audience, independent of broader
REST/Web audience settings. Omit it when MCP clients already hold directly
supplied JWTs.

## Edge-Asserted Web Identity

Web can verify a credential forwarded by an authenticating edge. REST and MCP
continue to verify their own bearer JWTs and never accept edge assertions.

```yaml
access:
  auth_mode: jwt
  web_identity:
    edge: cloudflare        # cloudflare | azure | aws
    issuer: https://<team>.cloudflareaccess.com
    audience: <application-aud-tag>
```

| Edge | Verified credential | Required configuration |
|---|---|---|
| Cloudflare Access | `Cf-Access-Jwt-Assertion`, fallback `CF_Authorization` JWT cookie | Team issuer + application AUD |
| Azure Easy Auth | `X-MS-TOKEN-AAD-ID-TOKEN` | Entra issuer + App Registration client ID |
| AWS Amplify/CloudFront | Forwarded `Authorization` bearer | IdP issuer + JWKS URL + app client ID |

Owner identity is `(iss, sub)`, so changing issuer creates a different owner even
for the same human. Azure's unsigned `X-MS-CLIENT-PRINCIPAL` is display-only.
Missing/invalid edge credentials return 401; DlightRAG renders no login page.

The origin must accept traffic only from the configured edge. Cryptographic
verification does not make arbitrary header injection safe. State-changing Web
routes also require exact same-origin `Origin` and a double-submit
`dlightrag_web_csrf` cookie echoed as `X-CSRF-Token`.

### Entra Example

Use one API App Registration with access-token version 2, an exposed delegated
scope, and optionally App Roles assigned to groups.

```yaml
access:
  auth_mode: jwt
  jwt_algorithm: RS256
  jwt_jwks_url: https://login.microsoftonline.com/<TENANT_ID>/discovery/v2.0/keys
  jwt_issuer: https://login.microsoftonline.com/<TENANT_ID>/v2.0
  jwt_audience: <API_CLIENT_ID>
```

Common mistakes:

- v2 tokens use the `/v2.0` issuer and client-ID GUID audience; v1 differs.
- Entra signs with RS256, not the HS256 default.
- App Roles provide stable strings in `roles`; raw `groups` contains object IDs
  and can overage around 200 groups.
- A client must request the exposed API scope to receive the API audience.

Map App Roles through `access.control.rules`, described below.

## Ingress Responsibilities

The application enforces semantic invariants:

- authentication, authorization, and owner scope;
- idempotency and changed-input conflict;
- URL redirect/DNS/SSRF checks and upload/fetch byte, archive, part, and pixel
  bounds;
- a streaming receive cap before parsers;
- durable lease fencing;
- client/model output sanitization; and
- provider concurrency and per-call timeouts.

Ingress owns TLS/certificates, DDoS and volumetric protection, WAF signatures,
IP/geo/bot policy, request quotas/rates, and connection caps. DlightRAG ships no
in-process WAF or rate limiter. SIEM systems such as Sentinel observe/correlate;
they are not an inline blocker.

Accepted Retrieval and Answer Runs queue rather than fail under local worker
saturation, up to the deployment-wide nonterminal Query-lane fuse (30,000 by
default). Corpus Mutation Runs use a separate validated 1,000-Run admission
fuse and deployment-wide active bound of two. The controlled full-fuse,
authorization, sanitation, and 10k-client evidence is recorded in the
[Slice 6 validation report](validation/run-runtime-slice-6.md). Monitor
PostgreSQL/blob growth and rate-limit acceptance before either safety bound.
`none` and `simple` collapse callers into one deployment owner and require an
already restricted network boundary.

`GET /health` and `GET /ready` are unauthenticated by design. Health performs
no database, corpus, parser, or model I/O. Readiness short-caches only the
writable Operational State verdict; it does not probe corpus or provider state.
Both surfaces expose only fixed component details and storage class names, never
credentials, endpoint URIs, or inherited environment values.

### Per-Surface Front Doors

REST and Web share the API process (default port 8100); MCP uses a separate
listener (8101). A browser redirect proxy may front only `/web`, while direct
REST/MCP clients supply their bearer tokens. All tokens must remain verifiable
under the one configured JWT policy. If browser and REST audiences differ, list
both under `jwt_audience`; native MCP OAuth still requires the exact public MCP
resource URL as audience.

## Authorization Model

Authentication asks who; authorization asks which product action on which
workspace.

```text
verified JWT claims
  -> deployment Access Rules
  -> canonical Workspace + Action
  -> allow or deny
```

A rule matches claim name/value, workspace pattern, and action pattern. Rules
combine with OR semantics; there are no deny rules, and no allow match means
deny.

```yaml
access:
  auth_mode: jwt
  control:
    mode: jwt_claims
    rules:
      - claim: roles
        value: finance.editors
        workspaces: [finance]
        actions: [editor]
      - claim: roles
        value: legal.readers
        workspaces: [legal]
        actions: [reader]
```

`jwt_claims` requires JWT auth and at least one rule. Local development defaults
to `allow_all`. Claim values may be strings or list members. Workspace patterns
are a canonical ID or `*`. Action patterns may be exact, `*`, a prefix such as
`workspace.*`, or a preset.

### Actions And Presets

| Action | Meaning |
|---|---|
| `workspace.query` | Retrieve/answer |
| `workspace.ingest` | Start ingestion/replacement/retry, including repair resume for those actions |
| `workspace.list_files` | List files |
| `workspace.delete_files` | Delete files, including repair resume for deletion |
| `workspace.download_source` | Download retained source |
| `workspace.read_metadata` | Read metadata |
| `workspace.update_metadata` | Update metadata |
| `workspace.read_visual_asset` | Read rendered visuals |
| `workspace.create`, `.reset` | Workspace creation and identity-preserving Corpus Reset, including repair resume/supersession for reset |
| `workspace.storage_status` | Read storage/promotion state |
| `model_catalogue.write` | Change deployment-wide model catalogue |

| Preset | Expansion |
|---|---|
| `reader` | query, list/download files, read metadata/visual assets |
| `editor` | reader + ingest and metadata/file mutation |
| `admin` | every action |

Presets affect only Actions, never Workspace matching. Deployment-wide actions
require `workspaces: ["*"]`.

### Source Of Truth And Revocation

The IdP owns users/claims; deployment configuration owns claim-to-workspace
rules. DlightRAG stores no users, custom roles, invitations, or membership ACLs.
PostgreSQL is trusted application storage without row-level security; this is
not database-enforced tenant isolation.

Explicit workspace requests are checked before acceptance. `all_workspaces`
expands only to currently authorized query workspaces. Source/visual routes
recheck permission against the actual workspace.

An accepted Retrieval or Answer Run pins its resolved Workspace set for
execution, not mutable claims. Later rule or IdP changes do not revoke execution
of that Run; follow-up/fork recheck current access. Corpus Mutation status,
events, cancellation, and repair resumption recheck access to the Run's declared
Workspace and fail closed. REST status/event projection and MCP status
projection recheck current source-download and visual-asset actions; canonical
results never persist authorization-dependent URLs. Trusted Application callers
supply their own Retrieval projection. JWT changes become visible when a new
token arrives, so use short lifetimes where revocation latency matters.

Use a policy/membership store for user-managed, deny, hierarchy, or resource-level
policy. Use separate deployments/databases or PostgreSQL RLS where regulation
requires database-enforced tenant separation.

## Source Download Boundary

Public source payloads expose stable `source_uri` and, on HTTP surfaces, a
projected `download_url` containing only document ID and workspace. They never
expose local paths or stored locators. REST/Web download routes recheck
`workspace.download_source` before streaming retained bytes or redirecting to
Azure, S3, or queryless HTTPS. Authorization is necessary but not sufficient:
the metadata row must also carry `_dlightrag_finalization_complete=true`.
Guessed IDs for pending, failed-finalization, legacy-unproven, or direct
LightRAG-bypass documents return 404. Full and thumbnail image routes enforce
the same rule, including on cache hits.

Signed/query-bearing URLs are fetch credentials, not durable locators. Retain
the bytes or provide a separate queryless `download_uri`; signed queries never
become public provenance or source-contract logs.

Durable Corpus Mutation Runs retain the bounded accepted fetch input, upstream
handoff checkpoint, and operator repair confirmation needed for recovery until
the fixed seven-day terminal retention floor permits pruning. Treat Operational State as secret
storage and restrict its access.

## Answer Resources And Execution

Attachments are bounded by count, per-item bytes, total bytes, archive expansion,
and decoded pixels before orchestration. Link resources and agent-supplied
`read(url=...)` targets admit only anonymous HTTP(S) without embedded credentials,
signed query credentials, fragments, localhost/private destinations, or unsafe
redirects. The shared egress boundary repeats scheme/host/DNS/SSRF checks at
every redirect, pins the validated address for each connection, and never permits
HTTPS to downgrade to HTTP. Agent reads can vary only `User-Agent`, `Accept`, and
`Accept-Language`; cookies, authorization, arbitrary headers, and browser sessions
are unavailable. A successful acquisition becomes one immutable run snapshot.

MarkItDown runs without plugins/network. OOXML files pass central-directory
zip-bomb checks before conversion. Full bytes never enter model context—only
bounded text windows, safe observations, and budgeted images. Only Evidence
ledger entries become citable; Profile Memory, Skills metadata, and incidental
child summaries do not.

Admitted bytes are content-addressed within one owner. A fetched resource is
stored only after validation and is linked atomically before its effect settles,
so recovery reads the same bytes. Deduplication never crosses owners.

Only a settled parent-Research `attach_artifact` call authorizes a workspace
root for publication. Fast and Child Sessions do not receive that capability,
and answer or Markdown `artifact:` links grant no authority. At the terminal
boundary, the Host rechecks the attachment's raw digest and size, validates its
safe dependency closure, and fails closed for missing, stale, or unattached
roots. Published descriptors and bytes remain owner/run scoped. See
[ADR 0004](adr/0004-structured-artifact-attachment-authority.md).

Execution modes:

- `disabled`: no local execution tools;
- `trust`: rooted file tools, but Bash retains all host/container filesystem and
  network authority of the service user; and
- `sandbox`: fails because this distribution ships no backend; it never
  downgrades to trust.

Root checks are not a shell sandbox. Outbound MCP endpoints/tools are deployment
allowlists; there is no discovery, OAuth brokerage, or management plane. Protect
tool credentials and egress at deployment level. Public Web Search and Extract
exist only for explicitly configured Exa/Tavily provider chains; provider failures
may fail over, while successful empty results do not.

All Agent/child/Fast mutations are fenced by owner, run lease, epoch, and
register sequence. A completed child outcome is persisted so replay cannot
re-enter it. A staged Fast result replays without another model call.

## Answer Artifact Browser Boundary

Artifact descriptors expose validated media and owner-scoped URLs, never raw
blob bytes or Agent Workspace paths. Authenticated data/Markdown uses
`Cache-Control: private, no-store`; active/unknown formats download with
`nosniff`.

Published SVG is sanitized of scripts, handlers, external loads, and nested SVG
data URLs, then served under CSP sandbox. PDF preview is sandboxed without
same-origin capability.

HTML never executes as a same-origin document. After explicit consent, the
browser inserts authenticated inert bytes into one `srcdoc` iframe with
`sandbox="allow-scripts"` but without same-origin, forms, popups, downloads,
frames, workers, storage, device permissions, or application bridge. A prepended
CSP blocks normal fetch/subresource paths. The wrapper's only parent signal is a
private one-way Escape-close token removed before Artifact code executes. Close
or switch destroys the iframe.

This isolates DlightRAG cookies, storage, and authenticated DOM. It provides no
CPU/memory quota, executes no server code, and is not an absolute browser-egress
guarantee. Chromium is the active-preview security regression baseline.

## Deployment Posture

| Deployment | Recommended posture |
|---|---|
| Local | Loopback REST/MCP + `none` |
| Trusted internal | `simple` behind network restriction |
| Enterprise multi-user | `jwt` + JWKS; `jwt_claims` for workspace policy |

Public MCP requires non-loopback bind, authentication, and explicit
`interfaces.mcp.allowed_hosts`/`allowed_origins`; browser clients also need
`access.cors_allow_origins`. Host/Origin DNS-rebinding protection remains active
even with bearer auth.
