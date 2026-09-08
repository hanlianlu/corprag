# RunRuntime Slice 6 validation

**Status:** complete for the repository-owned control plane

**Load evidence capture:** 2026-09-08T02:54:08Z

**Decision:** retain Query `16 workers/process + 30,000 nonterminal` and Corpus Mutation `2 workers/writer process + 1,000 nonterminal`

This report closes the failure/load evidence boundary in
[the RunRuntime target](../run-runtime-and-scaling-target.md). It does not claim
that 10,000 expensive operations execute at once: the load model is 10,000
addressable client submissions in one trusted organization, while one process
bounds expensive work to 16 Query and two Corpus Mutation executions through
coordinator worker concurrency. Deployment configuration owns process count and
total active capacity. This report also does not transfer
provider, parser, LightRAG storage, PostgreSQL provisioning, networking, ingress,
autoscaling, backup, or disaster-recovery ownership into this repository.

## Reproducible commands

```bash
# Repository-owned P0/P1/P2 fault injection (fresh PG fixtures, fake dependencies,
# frontend reconnect/repair tests)
make runtime-faults

# Fake models against the supported real PG18 LightRAG composition, including delete
make runtime-pg18

# Opt-in deterministic 10k campaign; never calls an external parser or provider
make load-runtime

# Run all three gates
make validate-runtime
```

`runtime-faults` fails closed when PostgreSQL is unreachable and passes only
when the Python fault suites, frontend unit suite, and frontend browser suite
all pass. `load-runtime` prints
`RUN_RUNTIME_LOAD PASS|FAIL`, the accepted row count, duration, throughput, and
drain time. `FAIL` means at least one correctness/survival gate failed; latency
percentiles and queue residence remain evidence rather than product SLOs. The
bounded local details are regenerated under `.test-results/load-runtime/` and
are intentionally gitignored.

## Failure matrix

Every `PASS` below names an executable test or command. “Residual owner” names
what remains after the repository-owned invariant; external provisioning is
never presented as DlightRAG implementation work.

### P0 — data, authorization, duplicate-effect, and lifecycle safety

| ID | Owned fault and affected kind/lane | Injected mechanism | Expected public/durable state | Safety invariant | Recovery/remedy | Automated evidence | Result | Residual owner |
|---|---|---|---|---|---|---|---|---|
| P0-01 | Concurrent acceptance, replay, changed-input conflict; Answer/Retrieval Query and Corpus Mutation | Same submission key raced, then fingerprint changed | One queued row; replay returns it; changed input conflicts; reached admission limit inserts no row | Atomic idempotency and no duplicate effect | Retry the same key/input or choose a new key for new input | `test_answer_runs_pg.py::TestCreation`, `test_retrieval_runs_pg.py::test_retrieval_acceptance_claim_and_terminal_result_use_the_generic_store`, `test_run_runtime_lane_independence_pg.py::test_atomic_replay_conflict_authorization_and_lane_local_admission_limits`, `make load-runtime` | PASS | Caller owns stable keys; Operational State availability is infrastructure-owned |
| P0-02 | Operational State accept/claim/heartbeat/write interruption; all kinds/lanes | Store exceptions are injected while PostgreSQL remains the authority; startup schema faults are controlled | Acceptance does not claim success without a durable row; claim is retried; heartbeat exception is not authoritative lease loss; `/ready` is not ready when Operational State cannot serve | No invented acceptance, no unfenced continuation | Restore Operational State; same queued/leased Run is retried or reclaimed | `test_run_runtime_lane_independence_pg.py::test_operational_state_outages_do_not_invent_acceptance_or_drop_queued_runs`, `test_answer_run_coordinator.py::TestHeartbeatResilience`, `test_application.py::test_a_startup_schema_failure_closes_the_application`, `test_api_server.py::TestReadinessEndpoint` | PASS | Database provisioning/failover remains infrastructure-owned |
| P0-03 | Lease expiry, crash reclaim, fencing, duplicate workers; all kinds | Lease is backdated; multiple workers race claim; stale epoch writes after reclaim | One new owner/epoch; stale heartbeat/event/checkpoint/terminal writes return no commit | At most one authoritative writer and one terminal settlement | Reclaim the same Run; abandon only after bounded no-progress reclaims | `test_answer_runs_pg.py::TestLeaseFencing`, `test_retrieval_runs_pg.py::test_retrieval_reclaim_and_both_cancellation_states_use_the_common_lifecycle`, `test_run_runtime_lane_independence_pg.py::test_graceful_shutdown_requeues_and_crash_reclaim_fences_post_handoff_worker` | PASS | Host/process restart is operator-owned |
| P0-04 | Multiple coordinators/claimers and Workspace ownership; both lanes | Six mutation claimers race three Workspace heads plus one same-Workspace successor; two Query coordinators each reserve two local slots | Three unique mutation heads claim concurrently; the successor remains queued; Query occupancy reaches four additive slots with no duplicate ID | No Run is double-claimed; one owning mutation per Workspace; process-local slots add across coordinators | Finish/release/defer the Workspace head; then its FIFO successor becomes eligible | `test_run_runtime_lane_independence_pg.py::test_concurrent_claimers_add_slots_without_double_claiming_or_breaking_fifo`, `test_multiple_coordinators_contribute_additive_local_slots` | PASS | Process count and placement are deployment-owned |
| P0-05 | Event race, cursor sequencing, terminal duplication, trim | Concurrent fenced appends, reconnect cursor, terminal race, expired retention | Gap-free ordered events; exactly one `done|error`; trimmed terminal row remains authoritative | SSE can be lossy but durable event authority is bounded and ordered | Reconnect with `after_sequence`; terminal status/result remains after trim | `test_answer_runs_pg.py::TestEvents`, `test_answer_runs_pg.py::TestRetention`, `test_corpus_mutation_phase_faults_pg.py::test_transient_ingest_recovers_the_same_track_with_exponential_checkpoint`, `make load-runtime` | PASS | Slow-client connection policy is ingress-owned |
| P0-06 | Cancellation before versus after external handoff; all kinds | Cancel queued/running work and race `start_handoff` CAS | Before handoff: cancelled/pending and one terminal event. After handoff: explicit `rejected` and mutation continues/repairs | Cancellation never promises rollback and cannot label an ambiguous effect cancelled | Observe terminal state; after handoff use repair/resume or authorized Reset | `test_corpus_mutation_runs_pg.py::test_pre_handoff_cancel_is_terminal_once_and_workspace_authorization_is_closed`, `test_corpus_mutation_runs_pg.py::test_corpus_repair_resume_is_explicit_and_keeps_the_workspace_barrier`, `test_retrieval_runs_pg.py::test_retrieval_reclaim_and_both_cancellation_states_use_the_common_lifecycle`, `test_corpus_mutations.py::test_public_operation_finishes_after_outer_task_cancellation` | PASS | Upstream effect meaning remains LightRAG-owned |
| P0-07 | Graceful shutdown versus hard crash; all kinds | Coordinator closes a blocked fake executor; separate lease is expired after handoff | Graceful work is requeued and its local slot released; crash keeps handoff/checkpoint and reclaims with a new fence | No shutdown write is torn and no crash creates a second Run | Start another writer; reclaim same Run and reconcile public upstream state | `test_run_runtime_lane_independence_pg.py::test_graceful_shutdown_requeues_and_crash_reclaim_fences_post_handoff_worker`, `test_answer_run_coordinator.py::test_graceful_shutdown_requeues_without_crash_recovery` | PASS | Process supervisor remains operator-owned |
| P0-08 | Transient dependency versus terminal taxonomy; all kinds | Typed parser/provider/corpus transient failures, then recovery; auth/4xx/schema/context/unknown errors | Transient: queued `phase=deferred`, bounded 2/4/…/60 second delay with no lease or local slot. Non-retryable: failed with bounded public error | The local slot is released without dropping the Run or Workspace barrier; unknown errors are never blindly retried | Dependency recovers and same Run is claimed; terminal input/config must be corrected in a new submission | `test_dependency_classification.py`, `test_retrieval_runs.py`, `test_corpus_mutations.py::test_transient_dependency_deferral_uses_bounded_exponential_backoff`, `test_corpus_mutation_phase_faults_pg.py::test_transient_ingest_recovers_the_same_track_with_exponential_checkpoint` | PASS | Dependency capacity/credentials remain provider/operator-owned |
| P0-09 | Workspace/owner authorization and result/error leakage; all kinds | Wrong owner/Workspace reads, cancel, event cursor; private paths/tokens in injected failures; 101 result rows | Unauthorized lookup is unknown/empty; public result caps at 100; repair/error text is bounded and secret-free | Operational rows and diagnostics never cross authorization scopes | Authenticate for the owning scope; inspect private logs only through operator controls | `test_run_runtime_lane_independence_pg.py::test_atomic_replay_conflict_authorization_and_lane_local_admission_limits`, `test_corpus_mutation_runs_pg.py::test_pre_handoff_cancel_is_terminal_once_and_workspace_authorization_is_closed`, `test_corpus_mutations.py::test_public_result_is_bounded_and_drops_paths_and_diagnostics`, `test_corpus_admin.py::test_public_failure_diagnostic_redacts_common_private_values` | PASS | Identity provider and ingress authentication remain external |
| P0-10 | Staging digest mismatch, incomplete source, and staging orphan; Corpus Mutation ingest/replace | Wrong SHA-256, missing accepted source, acceptance failure/replay | Stage is atomically removed or terminal `corpus_source_unavailable`; no private path in result | No partial source admission and no second corpus copy in RunBlobStore | Restage complete bytes and submit a new Run; exact replay removes the unaccepted stage | `test_corpus_mutations.py::test_stage_upload_digest_mismatch_removes_the_run_exclusive_stage`, `test_corpus_mutations.py::test_discard_staged_run_owns_the_private_stage_layout`, `test_corpus_mutation_phase_faults_pg.py::test_missing_staged_source_fails_terminally_without_public_path` | PASS | Shared POSIX volume provisioning remains infrastructure-owned |
| P0-11 | Public LightRAG `track_id` reconciliation after handoff; ingest/replace | Crash after `handoff_started`; public track lookup returns processed state | Same Run succeeds via finalization; source admission is not called again; stable track is retained | No duplicate source enqueue/effect after an ambiguous worker exit | Reconcile `aget_docs_by_track_id`; re-enqueue only when tracked state is absent and staged source is complete | `test_corpus_mutation_phase_faults_pg.py::test_public_track_reconciliation_does_not_repeat_source_admission` | PASS | Public LightRAG contract/storage durability remains upstream-owned |
| P0-12 | Product Document ingest/replace phase faults: source staging, enqueue/parser, LightRAG processing, visual fusion, BM25 labels, metadata/source locator, readiness promotion | The PG Runtime test injects typed failures at the executor's public `aingest` aggregate seam for six dependency labels; engine unit tests separately fault exact visual, BM25, and combined metadata/source/readiness finalizers before completion | Run defers/fails according to taxonomy; `_dlightrag_finalization_complete` is never committed early | Unfinished documents remain hidden and required finalizers are idempotent | Same Run reclaims/reconciles and replays only incomplete finalizers | `test_corpus_mutation_phase_faults_pg.py::test_ingest_owned_phase_faults_defer_without_leaking_capacity_or_diagnostics`, `test_unified_ingestion_engine.py::test_required_product_document_finalizer_fault_replays_before_readiness`, `test_document_visibility.py` | PASS | Parser/model/storage service capacity remains external |
| P0-13 | Delete/replace/retry destructive phases: hide, public delete, source/sidecar removal, projection cleanup | Ambiguous exception after handoff and explicit `waiting_for_repair` outcome | Run remains running at `waiting_for_repair` without a lease or local slot and is not cancellable; later same-Workspace mutation is blocked | No false success, visibility leak, or unsafe unlock; uncertain physical state stays explicit | Inspect public state, repair, then explicitly resume same Run | `test_corpus_mutation_phase_faults_pg.py::test_ambiguous_destructive_phase_waits_for_repair_and_keeps_fifo`, `test_unified_ingestion_engine.py` replacement/retry cancellation tests, `test_cleanup.py`, `make runtime-faults` | PASS | LightRAG internal flush/status implementation is upstream-owned |
| P0-14 | Corpus Reset drop failure and explicit supersession | Reset fake returns/raises ambiguous outcome; wrong-Workspace and authorized supersession attempted | Ambiguous Reset waits for repair. Authorized same-Workspace Reset terminally supersedes prior waiting Run with audit link/event | Reset is the only destructive supersession; no generic unsafe unlock | Correct scope and submit confirmed Reset naming the waiting Run | `test_corpus_mutation_phase_faults_pg.py::test_ambiguous_destructive_phase_waits_for_repair_and_keeps_fifo`, `test_corpus_mutation_runs_pg.py::test_reset_explicitly_supersedes_only_the_same_workspace_waiting_run`, `test_workspace_rag_reset.py` | PASS | Storage drop implementation remains LightRAG-owned |
| P0-15 | Supported default PostgreSQL delete convergence | Fake LLM/embed ingest+replace creates exact graph node/edge identities, visibility crash marker is injected, then public delete runs against PG18 | Document disappears from `PGKVStorage`, `PGVectorStorage`, the proven non-empty `PGTableGraphStorage` rows, `PGDocStatusStorage`, DlightRAG metadata and direct retrieval | Public success requires observable convergence; no private LightRAG flush/status calls | Same public delete/repair contract; test cleans its Workspace | `test_pg18_lightrag_smoke.py::test_unified_text_ingest_replace_and_filtered_retrieval` under the PG18 command above | PASS | PostgreSQL/extension provisioning remains infrastructure-owned |

### P1 — bounded availability, recovery, degradation, and capacity isolation

| ID | Owned fault | Injection and expected state | Safety/recovery invariant | Automated evidence | Result | Residual owner |
|---|---|---|---|---|---|---|
| P1-01 | Query pressure must not starve mutation; mutation pressure must not starve Query | Reach one lane's reduced nonterminal admission limit while the other lane still accepts/claims/progresses; fill local workers independently | Independent admission limits and local semaphores; no global Query-vs-mutation lock | `test_run_runtime_lane_independence_pg.py`, `make load-runtime` | PASS | Provider/LightRAG internal concurrency remains external/upstream |
| P1-02 | Deferred/repair work must release local slots without breaking barriers | One-slot coordinator yields alpha to deferred/repair state and starts beta; waiting alpha blocks alpha successor; next attempt recovers same Run | Local slot release, Workspace barrier retention, skip-ineligible head | `test_run_runtime_lane_independence_pg.py::test_defer_and_repair_release_the_local_slot`, `test_defer_releases_lease_preserves_barrier_then_recovers_same_run`, `test_corpus_mutation_runs_pg.py` | PASS | Dependency restoration is operator-owned |
| P1-03 | Reader role can accept Operational State but cannot claim/mutate corpus | Default PG18 writer provisions; reader attaches and attempts writes | Reader serves retrieval and durable Query acceptance; corpus writes fail; no mutation executor is registered | `test_pg18_lightrag_smoke.py::test_reader_role_attaches_read_only_and_rejects_writes`, `test_reader_role.py`, `test_application.py::test_a_reader_validates_the_durable_schema_it_does_not_own` | PASS | Replica topology is infrastructure-owned |
| P1-04 | Health boundary under corpus/provider/parser versus Operational State faults | Component degradation and readiness probe failures are controlled | `/health` remains I/O-free 200; only Operational State affects `/ready`; other dependencies remain ready+degraded | `test_api_server.py::TestHealthEndpoint`, `test_api_server.py::TestReadinessEndpoint`, `test_application_health.py`, `make runtime-faults` | PASS | ASGI process health and ingress routing remain server/infra-owned |
| P1-05 | 10k addressable client control-plane pressure | 10,000 Query submissions plus 1,000 mutation submissions covering all five production actions and one admission-limit rejection; fake executors hold 16+2 local slots | No lost Run, duplicate terminal, unbounded executor, deadlock, or starvation; eventual drain | `make load-runtime` and `tests/unit/test_run_runtime_load_contract.py` | PASS | This is not provider, parser, network, or fleet capacity evidence |
| P1-06 | Retrieval fan-out shape | Deterministic top-level Retrieval requests cover 1/10/50/100 Workspaces | One local Query slot owns the request; no 100-document paid ingest is fabricated | `make load-runtime` (`retrieval_workspace_shapes`) | PASS | Corpus content/quality is release-operator evidence |
| P1-07 | Operational scans at backlog | `EXPLAIN (ANALYZE, BUFFERS)` at 200 Query and 1,000 mutation nonterminal rows | Claim/event cursor indexes are available; planner may choose a cheaper compact seqscan for count/FIFO | `test_run_runtime_query_plans_pg.py::test_runtime_backlog_queries_use_bounded_indexes`, `.test-results/load-runtime/explain-*.txt` from `make load-runtime` | PASS | PostgreSQL tuning/provisioning remains infrastructure-owned |
| P1-08 | Queue cleanup and retention pressure | Terminal drain, event trim/prune, fresh database forced drop | Nonterminal work is never retention-pruned; accepted eligible work drains; temporary DB is removed | `test_answer_runs_pg.py::TestRetention`, isolated fixtures, `make load-runtime` | PASS | Backup/DR remains infrastructure-owned |
| P1-09 | Automatic PostgreSQL regression coverage omission | CI suite contract inspects workflow | Retrieval, Corpus Mutation, phase-fault, lane-independence, and query-plan PG suites run in automatic fresh-container CI and skips fail the job | `test_ci_workflow_contract.py::test_pg_integration_job_rejects_skips_without_external_evaluation` | PASS | CI runner/container availability is platform-owned |

### P2 — operator, UI, and observability quality

| ID | Owned fault | Expected behavior | Automated evidence | Result | Residual owner |
|---|---|---|---|---|---|
| P2-01 | Observer disconnect/reconnect and cursor reset | Disconnect never cancels; reconnect resumes from durable cursor and handles trim/reset | `frontend/lib/run-controller.test.ts`, `frontend/stores/answer-event-cursor-store.test.ts`, `frontend-test` in `make runtime-faults` | PASS | Ingress slow-client policy remains external |
| P2-02 | Failed-file repair UI loses session/run context | UI binds Workspace+Run, exposes retry/repair resume, and does not guess terminal success | `frontend/ui/failed-file-recovery.browser.test.ts`, browser step in `make runtime-faults` | PASS | Human repair decision remains operator-owned |
| P2-03 | Health/diagnostic payload grows or leaks details | Bounded component vocabulary and sanitized public diagnostics | `test_application_health.py::test_component_view_is_bounded_and_recovery_clears_stale_warning`, `test_corpus_admin.py` diagnostic tests | PASS | Private logging backend/retention remains operator-owned |
| P2-04 | Operator misreads load output or reruns paid dependencies | One documented command prints explicit PASS/FAIL and writes bounded sanitized evidence; fake-only contract is asserted | `tests/unit/test_run_runtime_load_contract.py`, `make load-runtime` | PASS | Hardware-to-production extrapolation remains an operator decision |

## Captured 10k result

Environment: Apple arm64, 16 logical CPUs, Darwin; Python 3.14.7 and PostgreSQL
18.6 (Debian). The workload used one fresh isolated database. The frozen capture
completed at `2026-09-08T02:54:08.991094+00:00`; a rerun regenerates the
gitignored evidence and must not be represented by these numbers without
updating this section. No database URL, credential, absolute local path, raw
log, or external provider response is retained here.

Traffic was exact and deterministic:

- 10,000 Query submissions: 1,000 Retrieval, 4,000 Fast Answer, 5,000 Research Answer;
- Retrieval covered 1, 10, 50, and 100 Workspaces; Answer used 1 or 10;
- 1,000 Corpus Mutation submissions spread FIFO pressure over 100 Workspaces, with exactly 200 each of ingest, replace, delete, retry, and reset;
- one additional mutation at the 1,000-Run nonterminal admission limit was rejected before insert;
- fake Query/Corpus executors emitted durable phase/events and held 16/two local coordinator slots;
- 29 queued cancellation requests, sampled owner/status reads, unauthorized reads, and 400 event-cursor operations;
- fresh isolated PostgreSQL was force-dropped after the run.

Captured result: **PASS**, 11,000 accepted and 11,000 terminal rows, zero lost
Runs, zero duplicate terminal Runs, total duration 70.654 s, accepted throughput
155.689 Runs/s, and post-offer drain 39.924 s.

| Operation | Samples | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| accept | 11,000 | 2.642 ms | 3.432 ms | 4.479 ms | 27.584 ms |
| status/owner lookup | 592 | 0.557 ms | 0.811 ms | 1.468 ms | 2.727 ms |
| cancel | 29 | 1.755 ms | 7.174 ms | 10.347 ms | 10.347 ms |
| event-cursor page/reconnect | 400 | 0.619 ms | 0.832 ms | 0.914 ms | 1.449 ms |

Queue residence was measured, not gated: p50 18.609003 s, p95 37.793724 s,
p99 39.375605 s. Maximum event-loop lag was 17.876 ms. Process high-water RSS
grew 3,571,712 bytes from a 109,707,264-byte idle high-water mark. The largest
observed database connection count was 12. Fake executor in-flight maxima were
exactly 16 Query and two Mutation, bounded solely by the one coordinator's
configured worker concurrency.

Broad survival tripwires were 900 s total, 300 s drain, 30 s control-operation
p99, 5 s event-loop lag, and 2 GiB RSS growth. These deliberately loose values
reject collapse and are not latency SLOs.

At the seeded backlog, normal PostgreSQL planning used
`idx_dlightrag_runs_claim` for Query claim and the event primary key for cursor
reads. The planner deliberately chose a compact sequential scan for the
1,000-row mutation admission count (0.104 ms captured execution) and a hash
anti-join for Workspace FIFO (0.528 ms), while the forced-index structural test
proves `idx_dlightrag_runs_claim` and `idx_dlightrag_runs_mutation_fifo` remain
usable. No additional index is justified at this bounded scale.

## Runtime-limit decision and limitations

Corpus Mutation retains two workers per writer process and a 1,000-Run
deployment-wide nonterminal admission limit. The campaign reached both local
worker slots, rejected before insert at the admission limit, kept Query
responsive, preserved per-Workspace FIFO, and drained. Query retains 16 workers
per process and a 30,000-Run deployment-wide nonterminal admission limit.
Process count and total active capacity remain deployment-owned; this campaign
does not justify either as provider throughput or fleet capacity.

This single-process fake-executor campaign proves the repository-owned durable
control plane only. The PG18 E2E gate separately proves observable default
storage delete convergence with fake models. Neither is a claim about paid
providers, real parser throughput, LightRAG private internals, shared-volume or
database provisioning, network/ingress behavior, autoscaling, backup/DR, or a
10,000-operation simultaneous execution topology.
