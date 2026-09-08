# Operations

This document owns maintenance and recovery runbooks. PostgreSQL tuning lives in
[PostgreSQL](postgresql.md); fields/defaults live in
[Configuration](configuration.md). These commands are not normal query traffic.

Deployments follow the
[configuration ownership and container contract](configuration.md#container-and-kubernetes-contract):
mount one non-secret `config.yaml`, inject credentials from a Secret, and keep
only topology bindings in Compose or Kubernetes manifests. The rationale is
recorded in [ADR 0006](adr/0006-configuration-ownership-and-deployment-bindings.md).

## pg_textsearch 1.4 Upgrade

The PostgreSQL image pins the extension binary, while PostgreSQL records the SQL
extension version inside each database. Rebuilding the image does not update an
existing data volume, and `postgres/init.sql` runs only for a new volume. Upgrade
both sides in this order:

```bash
# Build first to minimize downtime.
docker compose build --pull postgres

# Quiesce application traffic and take a logical backup.
docker compose stop dlightrag-api dlightrag-mcp
mkdir -p ~/.dlightrag/backups
backup="$HOME/.dlightrag/backups/dlightrag-before-pg_textsearch-1.4-$(date +%Y%m%d-%H%M%S).dump"
docker compose exec -T postgres sh -ec \
  'pg_dump --format=custom --username="$POSTGRES_USER" --dbname="$POSTGRES_DB"' \
  > "$backup"

# Load the new preloaded library, then update the extension catalog.
docker compose up -d --wait --force-recreate postgres
docker compose exec -T postgres sh -ec \
  'psql --set=ON_ERROR_STOP=1 --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" \
   --command="ALTER EXTENSION pg_textsearch UPDATE TO '\''1.4.0'\''"'

# Both values must report 1.4.0; filtered top-K seeding should be on / 3.
docker compose exec -T postgres sh -ec \
  'psql --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" \
   --command="SELECT extversion FROM pg_extension WHERE extname = '\''pg_textsearch'\''" \
   --command="SHOW pg_textsearch.library_version" \
   --command="SHOW pg_textsearch.filtered_seed" \
   --command="SHOW pg_textsearch.filtered_seed_margin"'

docker compose up -d dlightrag-api dlightrag-mcp
```

The upstream 1.3.1→1.4.0 path preserves existing BM25 indexes and does not
require `REINDEX`. A later maintenance-window `REINDEX` is optional to reclaim
pages freed by the old binary. Use
[Workspace BM25 Rebuild](#workspace-bm25-rebuild) instead when changing
DlightRAG language profiles, `bm25_k1`, or `bm25_b`.

## Full Development Reset

`scripts/reset_development.py` erases the complete development environment:
PostgreSQL data/migration ledger, LightRAG corpus/KG/vector/status data,
DlightRAG Runs/answers/Web state, and local runtime/corpus files. It is never
exposed through REST, Web, or MCP.

```bash
# Read-only previews; safe while services run.
uv run scripts/reset_development.py --mode docker --dry-run
uv run scripts/reset_development.py --mode native --dry-run

# Docker: remove app volumes, start only empty PostgreSQL, verify extensions.
uv run scripts/reset_development.py --mode docker
# or
make dev-reset

# Native: replace the dedicated database public schema and empty working files.
uv run scripts/reset_development.py --mode native
```

Interactive runs require the exact database name. `--yes` skips only that
prompt, never target validation. Native mode refuses non-loopback hosts unless
`--allow-remote-reset` and other sessions unless `--force-disconnect`. Docker
reset leaves API/MCP/readers/writers stopped; start one writer separately so it
creates baseline schema.

This differs from `scripts/reset_workspace.py`, which resets authorized Corpus
Workspaces in a running deployment. Neither delegates to the other.

## RunRuntime And Durable Query And Corpus Mutation Runs

- The current RunRuntime has no legacy Ingest Job or Answer-only compatibility reader.
  External drain/cutover execution remains operator/infrastructure responsibility.
- Roll one compatible writer first so it migrates, then readers. Workers sharing
  a database must use compatible model roles, execution mode, MCP allowlists,
  and Answer policy.
- Mount one shared POSIX `deployment.working_dir` for corpus artifacts. With
  trusted/sandboxed Research, mount one shared RWX
  `answer.agent.workspace_root` on every worker (Compose:
  `/home/app/.dlightrag/agent_workspaces`).
- Graceful shutdown fenced-requeues unfinished work; crash recovery waits for
  lease expiry. Four no-progress reclaims fail as `run_abandoned`.
- Monitor `dlightrag_runs`, `dlightrag_run_events`, `dlightrag_blobs`, and
  `dlightrag_blob_chunks`. Each process defaults to 16 Query workers, and each
  writer process defaults to two Corpus Mutation workers. Deployment
  configuration owns process count and total active capacity. New acceptance is
  rejected when the lane's deployment-wide nonterminal admission limit is
  reached: 30,000 Query Runs or 1,000 Corpus Mutation Runs by default. These
  values and their limitations are recorded in the [Slice 6 validation
  report](validation/run-runtime-slice-6.md).
- Route traffic with `GET /ready`; it probes only writable Operational State.
  Use `GET /health` for I/O-free liveness and the bounded corpus/parser/provider
  degradation view. A corpus or provider outage does not remove readiness:
  accepted eligible Runs defer durably while their lane's admission limit has room.

Run the repository-owned failure matrix, fake-model PG18 convergence gate, and
opt-in fake-only load campaign with `make validate-runtime`. `runtime-faults`
fails on any P0/P1/P2 Python or frontend recovery regression; `runtime-pg18`
requires the supported PostgreSQL 18 image/extensions; `load-runtime` prints
`RUN_RUNTIME_LOAD PASS` only when every correctness/survival gate passes and
writes bounded local evidence to `.test-results/load-runtime/`. A reported
latency percentile is measurement, not a production SLO. The command uses local
PostgreSQL and LightRAG with fakes, but no paid provider or external parser.

Retention sweeps run hourly in bounded batches without cron. Each terminal Run
receives `purge_after` from its accepted retention selection: Answer uses
`runtime.run_retention_days` (default 365 days), while top-level Retrieval and
Corpus Mutation use seven days. Nonterminal Runs are never retention-pruned. Event logs may expire
before a retained Run, after which SSE returns 410 and status still serves the
result. Exact lifecycle rules are in
[RunRuntime and durable query execution](durable-answer-runs.md).

## Trusted Publisher Prerequisite

One tag publishes lockstep `dlightrag` and `dlightrag-memory` projects. Configure
this GitHub trusted publisher on both PyPI projects before tagging:

- repository `hanlianlu/DlightRAG`
- workflow `.github/workflows/publish.yml`
- environment `pypi`

The workflow uses GitHub OIDC (`id-token: write`) and has no token fallback.
Publishing is not transactional; rerun after fixing a missing project/binding.

## Parser Services

Keep exactly one `corpus.sidecars.mineru` or `.docling` block. The checked-in
Docker configuration reaches host MinerU through
`http://host.docker.internal:8210`.

```bash
make mineru-install
make mineru-service-install
make mineru-service-status
make mineru-service-logs
make mineru-service-stop
```

Use `make mineru-api` for foreground operation where a background user service
is unavailable.

Optional Compose Docling CPU:

```bash
docker compose --profile docling up -d
```

Point its block at `http://docling:5001` with `code_formula_preset: null`. It
publishes only `127.0.0.1:5001`; do not run it beside a host Docling service on
the same port. Independently managed Docling endpoints are also supported.

## Product Document Finalization And Failed Ingestion Cleanup

A LightRAG `processed` status alone does not publish a Product Document.
DlightRAG's processed file panel, retrieval evidence, metadata surfaces,
downloads, and image routes require the finalization marker to be exactly true.
A failure in metadata/source finalization, BM25 labeling, required retained
source/sidecar work, or enabled visual fusion leaves the marker false while the
native LightRAG status remains `processed`. DlightRAG never rewrites that status
to represent product-finalization failure. Re-ingest the same retained source
to replay only the idempotent same-ID finalizers; do not manually set the marker. Documents written directly through LightRAG have no completion proof
and remain excluded.

Failed documents are terminal and are not automatically retried. First inspect
the workspace:

```bash
curl 'http://127.0.0.1:8100/files/failed?workspace=personel'
```

If the stored source/download locator is still available, retry every failed
document with the currently configured parser:

```bash
curl -X POST http://127.0.0.1:8100/runs/corpus/retry \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: retry-personel-failed-1' \
  -d '{"workspace":"personel","selector":"all_retryable"}'
```

If retry is unwanted or the source is unavailable, accept exact durable
deletion by filename:

```bash
curl -X POST http://127.0.0.1:8100/runs/corpus/delete \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: delete-personel-failed-1' \
  -d '{"workspace":"personel","filenames":["failed.pdf"]}'
```

Deletion writes the false visibility marker before invoking LightRAG, then
cascades the failed status, full document, metadata, chunks/vectors/KG when
present, source file, and `.parsed`/`.mineru_raw`/`.docling_raw` directories.
An ambiguous deletion can therefore reduce recall but cannot leave the document
directly visible.
The terminal Corpus Mutation Run remains as operational history for at least
seven days.

### Repairing An Ambiguous Mutation

If Run status reports `phase=waiting_for_repair`, do not submit a replacement
mutation and do not manually rewrite the Run row. Inspect `repair_reason` and
`repair_remedy`, repair or verify authoritative LightRAG public state, then
resume the same Run:

```bash
curl -X POST http://127.0.0.1:8100/runs/$RUN_ID/resume
```

The caller must still hold the action permission implied by that Run. Resume
keeps the same Run ID and `track_id` and returns the Run to its Workspace FIFO.
If repair is inappropriate and a full corpus reset is required, accept one reset
naming `supersedes_run_id`; only reset may terminally supersede a waiting
mutation while preserving Workspace identity and history.

## Workspace BM25 Rebuild

`dlightrag-rebuild-bm25` creates configured pg_textsearch indexes and refreshes
`dlightrag_bm25_language` for existing chunks. It neither parses, calls models,
rebuilds vectors, nor changes sources.

Run it after enabling BM25 on an existing corpus or changing BM25 profiles,
`k1`, or `b`:

```bash
# Stop every API, MCP, ingest, and reader process using the workspace.
uv run dlightrag-rebuild-bm25 --yes

# Installed package:
dlightrag-rebuild-bm25 --env-file /absolute/path/to/.env --yes
```

The configured role must be `writer` and BM25 must be enabled. Restart services
only after completion. `--batch-size N` bounds language-label transactions.
Vector rebuild targets `chunks` and `all` already run this maintenance.

## Offline Vector Storage Rebuild

`dlightrag-rebuild-vdb` rebuilds LightRAG vectors from existing graph/chunk rows
using the configured workspace, embedding model, BM25 labels, and visual
alignment. It does not ingest/parse files or create document status.

Use it for LightRAG upgrade guidance, missing/stale rows, failed `check`, or an
intentional embedding change whose vector schema already supports the dimension.
Use failed-file retry—not this command—for ingestion failures.

| Target | Writes | Behavior |
|---|---:|---|
| `check` | no | Compare graph records with entity/relation vectors |
| `graph` | yes | Rebuild entity and relationship vectors |
| `chunks` | yes | Rebuild chunk vectors, labels, and fused visual alignment |
| `all` | yes | Run graph + chunks maintenance |

`graph`, `chunks`, and `all` require `--yes`. Stop every writer to the same
LightRAG storage before them.

### Native Or Installed

```bash
uv run dlightrag-rebuild-vdb --target check
uv run dlightrag-rebuild-vdb --target all --yes

# Installed package:
dlightrag-rebuild-vdb --env-file /absolute/path/to/.env --target check
dlightrag-rebuild-vdb --env-file /absolute/path/to/.env --target all --yes
```

| Flag | Meaning |
|---|---|
| `--env-file PATH` | Load explicit environment file |
| `--batch-size N` | Source rows per rebuild batch |
| `--no-restore-sidecar-alignment` | Skip fused visual-vector restoration |

### Docker Compose

```bash
# Check is read-only.
docker compose run --rm dlightrag-api dlightrag-rebuild-vdb --target check

# Stop writers, rebuild in the app image, restart.
docker compose stop dlightrag-api dlightrag-mcp
docker compose run --rm dlightrag-api dlightrag-rebuild-vdb --target all --yes
docker compose up -d dlightrag-api dlightrag-mcp
```

After `chunks`/`all`, DlightRAG refreshes BM25 language labels and replaces
canonical drawing vectors with fused VLM-description+image vectors when direct
multimodal embedding is active. Skip alignment only for diagnosis or intentional
text-only deployments.

Before destructive production rebuilds: back up PostgreSQL, use the service's
same `.env`/`config.yaml`/workspace/model, and do not change dimensions without
a migrated/recreated vector schema. Inspect any nonzero exit before restart.

## Local Langfuse Observability

The repository can run an isolated local Langfuse Compose project in
`../langfuse-local` (override with `LANGFUSE_LOCAL_DIR`).

```bash
make langfuse-up
```

Open <http://localhost:3300> with user `admin@localhost.local`; read the generated
password from `../langfuse-local/.env`. Traces appear after a model call.

| Target | Behavior |
|---|---|
| `make langfuse-stack` | Download/patch the official Compose file |
| `make langfuse-bootstrap` | Sync project credentials to both env files |
| `make langfuse-up` | Bootstrap and start |
| `make langfuse-down` | Stop |
| `make langfuse-restart` | Re-sync and recreate Web/worker |
| `make langfuse-status` | Show containers |
| `make langfuse-logs` | Follow Web/worker logs |
| `make langfuse-health` | Check host endpoint |
| `make langfuse-reset CONFIRM=1` | Delete all local traces |

### Connection And Keys

`scripts/langfuse/headless.py` writes one key pair to the Langfuse
`LANGFUSE_INIT_PROJECT_*` variables and DlightRAG's
`DLIGHTRAG_OBSERVABILITY__LANGFUSE_{PUBLIC,SECRET}_KEY`. Both are required;
without either, tracing is disabled. Initialization variables are read only when
Langfuse first creates its database; rotate established credentials in the UI or
reset the local stack.

Set `observability.langfuse_host` according to where DlightRAG runs:

| DlightRAG | Trace host |
|---|---|
| Docker Compose | `http://host.docker.internal:3300` |
| Native | `http://localhost:3300` |

Browsers and `make langfuse-health` always use `http://localhost:3300`.
`config.yaml` owns the nonsecret host; bootstrap writes only secrets.

Compose reads `.env` only when creating a container. After keys/host change:

```bash
docker compose up -d --force-recreate dlightrag-api dlightrag-mcp
docker compose logs dlightrag-api | grep -i 'langfuse tracing'
```

A log target of `https://cloud.langfuse.com` means the local host setting was
not loaded.

### Cost And Recovery

DlightRAG always reports tokens. Configure matching model prices in Langfuse to
compute cost. OpenRouter can instead return charged cost when enabled only on
its model block:

```yaml
models:
  chat:
    default:
      provider: openai
      base_url: https://openrouter.ai/api/v1
      model_kwargs:
        usage:
          include: true
```

Do not send that vendor-specific option to providers that reject it.

To disable tracing, clear both Langfuse keys and recreate app containers. If the
UI password is lost, read `LANGFUSE_INIT_USER_PASSWORD` from the local env. If
the env file was deleted, `make langfuse-bootstrap` recovers the API key pair
from DlightRAG's `.env`, but cannot change the established UI password.

Last-resort reset (deletes only Langfuse traces):

```bash
make langfuse-bootstrap
make langfuse-reset CONFIRM=1
make langfuse-up
grep LANGFUSE_INIT_USER_PASSWORD ../langfuse-local/.env
```
