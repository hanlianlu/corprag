# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL adapter family for the operation-neutral RunRuntime.

The generic run row, claim/lease/event lifecycle, and retention logic live here.
Answer-owned Session, evidence, and resource tables remain narrow projections
linked to the generic row; callers consume owner-specific Protocols rather than
one universal operational-storage interface.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, cast

import asyncpg

from dlightrag.adapters.postgres.answer.memory_settings import (
    MEMORY_SETTINGS_DDL,
    MEMORY_SETTINGS_SCHEMA_TABLE,
)
from dlightrag.adapters.postgres.answer.session_repository import (
    PGAgentSessionRepository,
    PGProgressStore,
)
from dlightrag.adapters.postgres.answer.workspace import PGWorkspaceStore
from dlightrag.adapters.postgres.core._migrations import (
    ForeignKeyRequirement,
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)
from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.adapters.postgres.core._pool import pg_pool
from dlightrag.adapters.postgres.runtime._terminal import TerminalStatus, finish_fenced_run
from dlightrag.adapters.postgres.runtime.run_blob_store import BlobSizeConflict, write_blob_content
from dlightrag.application.answer_runs import (
    ChildRosterPageRequest,
    ChildRosterRowPage,
)
from dlightrag.engine.agent.session.ids import SessionId
from dlightrag.engine.agent.tool_content import decode_tool_content, tool_content_message_fields
from dlightrag.engine.answer.runs.routing import RoutingAcceptance, RoutingRecord
from dlightrag.engine.runtime.cancellation import (
    RunCancellationListener,
    cancellation_notify_key,
)
from dlightrag.engine.runtime.contracts import RunKind, RunLane, RunPhase
from dlightrag.engine.runtime.errors import RunSchemaError
from dlightrag.engine.runtime.policy import (
    DEFAULT_RUN_RETENTION_SECONDS,
    MAX_RECLAIMS_WITHOUT_PROGRESS,
    RUN_ABANDONED_ERROR_KIND,
    RUN_LEASE_SECONDS,
)
from dlightrag.engine.runtime.records import (
    CancellationOutcome,
    ClaimedRun,
    IdempotencyKeyConflict,
    LeaseRenewal,
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
    ReclaimDecision,
    ReclaimState,
    RunAccessScope,
    RunAdmissionLimitExceededError,
    RunArtifactReference,
    RunCreation,
    RunDeletion,
    RunEvent,
    RunExecutionContext,
    RunFetchedResource,
    RunRecord,
    ShutdownOutcome,
    SweepOutcome,
    TerminalOutcome,
    advance_reclaim,
    parse_run_id,
    require_prepared_input_bounds,
)
from dlightrag.engine.runtime.settlements import ArtifactAttachmentUpdate

RUN_MIGRATION_SCOPE = "runs"

_ABANDONED_ERROR_MESSAGE = "Run exceeded its reclaim-without-progress bound."
_BATCH_LIMIT = 200
_EVENT_PAGE_LIMIT = 500
DEFAULT_QUERY_MAX_NONTERMINAL_RUNS = 30_000
# Validated by the deterministic bounded-control-plane campaign documented in
# docs/validation/run-runtime-slice-6.md.
DEFAULT_CORPUS_MUTATION_MAX_NONTERMINAL_RUNS = 1_000

_MIGRATE_ANSWER_RUNTIME = """
DO $$
BEGIN
    IF to_regclass('dlightrag_answer_runs') IS NOT NULL
       AND to_regclass('dlightrag_runs') IS NULL THEN
        IF EXISTS (
            SELECT 1 FROM dlightrag_answer_runs
            WHERE status IN ('queued', 'running')
        ) THEN
            RAISE EXCEPTION
                'cannot rename Answer runtime schema while nonterminal runs exist';
        END IF;
        ALTER TABLE dlightrag_answer_runs RENAME TO dlightrag_runs;
    END IF;
    IF to_regclass('dlightrag_answer_run_events') IS NOT NULL
       AND to_regclass('dlightrag_run_events') IS NULL THEN
        ALTER TABLE dlightrag_answer_run_events RENAME TO dlightrag_run_events;
    END IF;
END $$
"""

_MIGRATE_RUN_COLUMNS = """
DO $$
BEGIN
    IF to_regclass('dlightrag_runs') IS NULL THEN
        RETURN;
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'dlightrag_runs' AND column_name = 'idempotency_key'
    ) THEN
        ALTER TABLE dlightrag_runs RENAME COLUMN idempotency_key TO submission_key;
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'dlightrag_runs' AND column_name = 'workspace_epoch'
    ) THEN
        ALTER TABLE dlightrag_runs RENAME COLUMN workspace_epoch TO agent_workspace_epoch;
    END IF;
END $$
"""

_ALTER_RUN_RUNTIME = """
ALTER TABLE dlightrag_runs
    ADD COLUMN IF NOT EXISTS run_kind TEXT NOT NULL DEFAULT 'answer',
    ADD COLUMN IF NOT EXISTS lane TEXT NOT NULL DEFAULT 'query',
    ADD COLUMN IF NOT EXISTS submitted_by TEXT,
    ADD COLUMN IF NOT EXISTS access_scope_kind TEXT NOT NULL DEFAULT 'owner',
    ADD COLUMN IF NOT EXISTS retention_seconds BIGINT NOT NULL DEFAULT 31536000,
    ADD COLUMN IF NOT EXISTS purge_after TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS checkpoint_json JSONB,
    ADD COLUMN IF NOT EXISTS handoff_started_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS superseded_by_run_id UUID;
UPDATE dlightrag_runs
SET submitted_by = owner_id,
    submission_key = COALESCE(submission_key, 'legacy:' || run_id::text),
    purge_after = CASE
        WHEN finished_at IS NOT NULL AND purge_after IS NULL
        THEN finished_at + make_interval(secs => retention_seconds::double precision)
        ELSE purge_after
    END;
ALTER TABLE dlightrag_runs
    ALTER COLUMN submitted_by SET NOT NULL,
    ALTER COLUMN submission_key SET NOT NULL;
"""

_NORMALIZE_RUN_CONSTRAINTS = """
ALTER TABLE dlightrag_runs
    DROP CONSTRAINT IF EXISTS dlightrag_answer_runs_status_check,
    DROP CONSTRAINT IF EXISTS dlightrag_answer_runs_phase_check,
    DROP CONSTRAINT IF EXISTS dlightrag_answer_runs_counter_check,
    DROP CONSTRAINT IF EXISTS dlightrag_answer_runs_lease_check,
    DROP CONSTRAINT IF EXISTS dlightrag_answer_runs_terminal_check,
    DROP CONSTRAINT IF EXISTS dlightrag_answer_runs_result_check,
    DROP CONSTRAINT IF EXISTS dlightrag_answer_runs_error_check,
    DROP CONSTRAINT IF EXISTS dlightrag_answer_runs_prepared_input_check,
    DROP CONSTRAINT IF EXISTS dlightrag_answer_runs_workspace_epoch_check,
    DROP CONSTRAINT IF EXISTS dlightrag_runs_kind_check,
    DROP CONSTRAINT IF EXISTS dlightrag_runs_lane_check,
    DROP CONSTRAINT IF EXISTS dlightrag_runs_scope_check,
    DROP CONSTRAINT IF EXISTS dlightrag_runs_status_check,
    DROP CONSTRAINT IF EXISTS dlightrag_runs_phase_check,
    DROP CONSTRAINT IF EXISTS dlightrag_runs_counter_check,
    DROP CONSTRAINT IF EXISTS dlightrag_runs_lease_check,
    DROP CONSTRAINT IF EXISTS dlightrag_runs_permit_check,
    DROP CONSTRAINT IF EXISTS dlightrag_runs_terminal_check,
    DROP CONSTRAINT IF EXISTS dlightrag_runs_result_check,
    DROP CONSTRAINT IF EXISTS dlightrag_runs_error_check,
    DROP CONSTRAINT IF EXISTS dlightrag_runs_prepared_input_check,
    DROP CONSTRAINT IF EXISTS dlightrag_runs_workspace_epoch_check;
ALTER TABLE dlightrag_runs
    ADD CONSTRAINT dlightrag_runs_kind_check
        CHECK (run_kind IN ('retrieval', 'answer', 'corpus_mutation')),
    ADD CONSTRAINT dlightrag_runs_lane_check
        CHECK (lane IN ('query', 'corpus_mutation')),
    ADD CONSTRAINT dlightrag_runs_scope_check
        CHECK (access_scope_kind IN ('owner', 'workspace')),
    ADD CONSTRAINT dlightrag_runs_status_check
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    ADD CONSTRAINT dlightrag_runs_counter_check
        CHECK (fencing_epoch >= 0 AND next_event_sequence >= 1
               AND durable_progress_version >= 0
               AND last_reclaim_progress_version >= 0
               AND reclaims_without_progress >= 0 AND retention_seconds >= 1),
    ADD CONSTRAINT dlightrag_runs_lease_check
        CHECK ((lease_owner IS NULL) = (lease_expires_at IS NULL)),
    ADD CONSTRAINT dlightrag_runs_terminal_check
        CHECK ((status IN ('succeeded', 'failed', 'cancelled')) = (finished_at IS NOT NULL)),
    ADD CONSTRAINT dlightrag_runs_result_check
        CHECK (status <> 'succeeded' OR result_json IS NOT NULL),
    ADD CONSTRAINT dlightrag_runs_error_check
        CHECK ((status = 'failed') = (error_kind IS NOT NULL)),
    ADD CONSTRAINT dlightrag_runs_prepared_input_check
        CHECK ((status IN ('queued', 'running')) = (prepared_input_json IS NOT NULL)),
    ADD CONSTRAINT dlightrag_runs_workspace_epoch_check
        CHECK (agent_workspace_epoch IS NULL OR agent_workspace_epoch >= 1);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dlightrag_runs_global_id ON dlightrag_runs (run_id);
DROP INDEX IF EXISTS idx_dlightrag_answer_runs_idempotency;
DROP INDEX IF EXISTS idx_dlightrag_runs_idempotency;
CREATE UNIQUE INDEX idx_dlightrag_runs_submission
    ON dlightrag_runs (run_kind, submitted_by, submission_key);
"""

# ─────────────────────────────────────────────────────────────────
# Final clean-break baseline schema
# ─────────────────────────────────────────────────────────────────

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS dlightrag_runs (
    owner_id            TEXT        NOT NULL,
    run_id              UUID        NOT NULL,
    run_kind            TEXT        NOT NULL,
    lane                TEXT        NOT NULL,
    submitted_by        TEXT        NOT NULL,
    access_scope_kind   TEXT        NOT NULL,
    submission_key      TEXT        NOT NULL,
    prepared_input_json JSONB,
    accepted_input_json JSONB       NOT NULL DEFAULT '{}'::jsonb,
    request_fingerprint TEXT        NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'queued',
    phase               TEXT,
    stop_reason         TEXT,
    cancel_requested_at TIMESTAMPTZ,
    lease_owner         TEXT,
    lease_expires_at    TIMESTAMPTZ,
    fencing_epoch       BIGINT      NOT NULL DEFAULT 0,
    durable_progress_version       BIGINT  NOT NULL DEFAULT 0,
    last_reclaim_progress_version  BIGINT  NOT NULL DEFAULT 0,
    reclaims_without_progress      INTEGER NOT NULL DEFAULT 0,
    next_event_sequence BIGINT      NOT NULL DEFAULT 1,
    events_trimmed_at   TIMESTAMPTZ,
    result_json         JSONB,
    error_kind          TEXT,
    error_message       TEXT,
    retention_seconds   BIGINT      NOT NULL,
    purge_after         TIMESTAMPTZ,
    next_attempt_at     TIMESTAMPTZ,
    checkpoint_json     JSONB,
    handoff_started_at  TIMESTAMPTZ,
    superseded_by_run_id UUID,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    agent_workspace_epoch BIGINT,
    PRIMARY KEY (owner_id, run_id),
    UNIQUE (run_id),
    CONSTRAINT dlightrag_runs_kind_check
        CHECK (run_kind IN ('retrieval', 'answer', 'corpus_mutation')),
    CONSTRAINT dlightrag_runs_lane_check
        CHECK (lane IN ('query', 'corpus_mutation')),
    CONSTRAINT dlightrag_runs_scope_check
        CHECK (access_scope_kind IN ('owner', 'workspace')),
    CONSTRAINT dlightrag_runs_status_check
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    CONSTRAINT dlightrag_runs_counter_check
        CHECK (fencing_epoch >= 0 AND next_event_sequence >= 1
               AND durable_progress_version >= 0
               AND last_reclaim_progress_version >= 0
               AND reclaims_without_progress >= 0 AND retention_seconds >= 1),
    CONSTRAINT dlightrag_runs_lease_check
        CHECK ((lease_owner IS NULL) = (lease_expires_at IS NULL)),
    CONSTRAINT dlightrag_runs_terminal_check
        CHECK ((status IN ('succeeded', 'failed', 'cancelled')) = (finished_at IS NOT NULL)),
    CONSTRAINT dlightrag_runs_result_check
        CHECK (status <> 'succeeded' OR result_json IS NOT NULL),
    CONSTRAINT dlightrag_runs_error_check
        CHECK ((status = 'failed') = (error_kind IS NOT NULL)),
    CONSTRAINT dlightrag_runs_prepared_input_check
        CHECK ((status IN ('queued', 'running')) = (prepared_input_json IS NOT NULL)),
    CONSTRAINT dlightrag_runs_workspace_epoch_check
        CHECK (agent_workspace_epoch IS NULL OR agent_workspace_epoch >= 1)
)
"""

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS dlightrag_run_events (
    owner_id       TEXT        NOT NULL,
    run_id         UUID        NOT NULL,
    event_sequence BIGINT      NOT NULL,
    event_type     TEXT        NOT NULL,
    payload        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, event_sequence),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_run_events_sequence_check
        CHECK (event_sequence >= 1)
)
"""

_NORMALIZE_RUN_EVENTS = """
ALTER TABLE dlightrag_run_events
    DROP CONSTRAINT IF EXISTS dlightrag_answer_run_events_type_check,
    DROP CONSTRAINT IF EXISTS dlightrag_run_events_type_check,
    DROP CONSTRAINT IF EXISTS dlightrag_answer_run_events_sequence_check,
    DROP CONSTRAINT IF EXISTS dlightrag_run_events_sequence_check;
ALTER TABLE dlightrag_run_events
    ADD CONSTRAINT dlightrag_run_events_sequence_check
        CHECK (event_sequence >= 1);
"""

_ENFORCE_RUN_EVENT_CONSTRAINTS = """
CREATE OR REPLACE FUNCTION public.dlightrag_enforce_run_event_constraints()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $$
DECLARE
    parent_status TEXT;
    parent_lease_owner TEXT;
    parent_lease_expires_at TIMESTAMPTZ;
    parent_next_event_sequence BIGINT;
    parent_result JSONB;
    parent_error_kind TEXT;
    parent_error_message TEXT;
BEGIN
    SELECT status, lease_owner, lease_expires_at, next_event_sequence,
           result_json, error_kind, error_message
    INTO parent_status, parent_lease_owner, parent_lease_expires_at,
         parent_next_event_sequence, parent_result, parent_error_kind,
         parent_error_message
    FROM public.dlightrag_runs
    WHERE owner_id = NEW.owner_id AND run_id = NEW.run_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'run event parent does not exist'
            USING ERRCODE = '23503';
    END IF;
    IF NEW.event_sequence >= parent_next_event_sequence THEN
        RAISE EXCEPTION 'run event sequence must precede the parent next sequence'
            USING ERRCODE = '23514';
    END IF;
    IF jsonb_typeof(NEW.payload) IS DISTINCT FROM 'object' THEN
        RAISE EXCEPTION 'run event payload must be a JSON object'
            USING ERRCODE = '23514';
    END IF;

    IF NEW.event_type = 'done' THEN
        IF jsonb_typeof(NEW.payload->'status') IS DISTINCT FROM 'string'
           OR NEW.payload->>'status' IS DISTINCT FROM parent_status THEN
            RAISE EXCEPTION 'done event status does not match its parent'
                USING ERRCODE = '23514';
        END IF;
        IF parent_status = 'succeeded' THEN
            IF NOT NEW.payload ? 'result'
               OR NEW.payload->'result' IS DISTINCT FROM parent_result THEN
                RAISE EXCEPTION 'succeeded run event payload does not match its parent'
                    USING ERRCODE = '23514';
            END IF;
        ELSIF parent_status = 'cancelled' THEN
            IF NEW.payload ? 'result' THEN
                RAISE EXCEPTION 'cancelled run event payload cannot contain a result'
                    USING ERRCODE = '23514';
            END IF;
        ELSE
            RAISE EXCEPTION 'done event requires a succeeded or cancelled parent'
                USING ERRCODE = '23514';
        END IF;
    ELSIF NEW.event_type = 'error' THEN
        IF parent_status IS DISTINCT FROM 'failed' THEN
            RAISE EXCEPTION 'error event requires a failed parent'
                USING ERRCODE = '23514';
        END IF;
        IF jsonb_typeof(NEW.payload->'kind') IS DISTINCT FROM 'string'
           OR jsonb_typeof(NEW.payload->'message') IS DISTINCT FROM 'string'
           OR NEW.payload->>'kind' IS DISTINCT FROM parent_error_kind
           OR NEW.payload->>'message' IS DISTINCT FROM parent_error_message
           OR (NEW.payload ? 'result'
               AND NEW.payload->'result' IS DISTINCT FROM parent_result) THEN
            RAISE EXCEPTION 'failed run event payload does not match its parent'
                USING ERRCODE = '23514';
        END IF;
    ELSIF parent_status IS DISTINCT FROM 'running'
          OR parent_lease_owner IS NULL
          OR parent_lease_expires_at IS NULL
          OR parent_lease_expires_at < NOW() THEN
        RAISE EXCEPTION 'nonterminal event requires a live leased parent'
            USING ERRCODE = '23514';
    END IF;

    RETURN NEW;
END
$$
"""

_CREATE_RUN_EVENT_CONSTRAINT_TRIGGER = """
DROP TRIGGER IF EXISTS trg_dlightrag_run_events_enforce
    ON public.dlightrag_run_events;
CREATE TRIGGER trg_dlightrag_run_events_enforce
    BEFORE INSERT OR UPDATE ON public.dlightrag_run_events
    FOR EACH ROW
    EXECUTE FUNCTION public.dlightrag_enforce_run_event_constraints();
"""

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_sessions (
    owner_id             TEXT        NOT NULL,
    session_id           UUID        NOT NULL,
    lease_run_id         UUID        NOT NULL,
    commit_sequence      BIGINT      NOT NULL DEFAULT 0,
    fencing_epoch        BIGINT      NOT NULL,
    last_sequence        BIGINT      NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, session_id),
    CONSTRAINT dlightrag_agent_sessions_commit_sequence_check CHECK (commit_sequence >= 0),
    CONSTRAINT dlightrag_agent_sessions_fencing_check CHECK (fencing_epoch >= 1),
    CONSTRAINT dlightrag_agent_sessions_sequence_check CHECK (last_sequence >= 0)
)
"""

_CREATE_ENTRIES = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_session_entries (
    owner_id       TEXT        NOT NULL,
    session_id     UUID        NOT NULL,
    sequence       BIGINT      NOT NULL,
    entry_id       UUID        NOT NULL,
    parent_entry_id UUID,
    entry_type     TEXT        NOT NULL,
    schema_version INTEGER     NOT NULL,
    timestamp      TIMESTAMPTZ NOT NULL,
    payload_json   JSONB       NOT NULL,
    PRIMARY KEY (owner_id, session_id, sequence),
    UNIQUE (entry_id),
    UNIQUE (owner_id, session_id, entry_id),
    FOREIGN KEY (owner_id, session_id)
        REFERENCES dlightrag_agent_sessions (owner_id, session_id) ON DELETE CASCADE,
    FOREIGN KEY (owner_id, session_id, parent_entry_id)
        REFERENCES dlightrag_agent_session_entries
            (owner_id, session_id, entry_id)
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT dlightrag_agent_session_entries_sequence_check CHECK (sequence >= 1),
    CONSTRAINT dlightrag_agent_session_entries_type_check CHECK (entry_type IN (
        'user_message', 'assistant_message', 'tool_result',
        'control_message', 'compaction'
    )),
    CONSTRAINT dlightrag_agent_session_entries_version_check CHECK (schema_version >= 1)
)
"""

_CREATE_SESSION_REGISTERS = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_session_registers (
    owner_id       TEXT        NOT NULL,
    session_id     UUID        NOT NULL,
    register_kind  TEXT        NOT NULL,
    register_key   TEXT        NOT NULL,
    sequence       BIGINT      NOT NULL,
    payload_json   JSONB       NOT NULL,
    PRIMARY KEY (owner_id, session_id, register_kind, register_key),
    FOREIGN KEY (owner_id, session_id)
        REFERENCES dlightrag_agent_sessions (owner_id, session_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_agent_session_registers_kind_check
        CHECK (register_kind IN (
            'lane_head', 'lane_state', 'operation_meta', 'operation_state',
            'request_snapshot', 'tool_arguments', 'pending_input', 'host_turn_reservation',
            'context_projection', 'session_fault'
        )),
    CONSTRAINT dlightrag_agent_session_registers_sequence_check CHECK (sequence >= 1)
)
"""

_CREATE_STAGES = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_run_stages (
    owner_id          TEXT        NOT NULL,
    run_id            UUID        NOT NULL,
    stage_intent_id   UUID        NOT NULL,
    stage_name        TEXT        NOT NULL,
    progress_version  BIGINT      NOT NULL,
    state             JSONB       NOT NULL,
    state_digest      TEXT        NOT NULL,
    settled_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, stage_intent_id),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_run_stages_name_check
        CHECK (stage_name IN ('planner', 'retrieval', 'final_generation')),
    CONSTRAINT dlightrag_answer_run_stages_digest_check
        CHECK (state_digest ~ '^[0-9a-f]{64}$')
)
"""

_CREATE_EVIDENCE = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_evidence (
    owner_id        TEXT        NOT NULL,
    run_id          UUID        NOT NULL,
    session_id      UUID        NOT NULL,
    intent_id       UUID        NOT NULL,
    result_ordinal  INTEGER     NOT NULL,
    content_digest  TEXT        NOT NULL,
    locator_digest  TEXT        NOT NULL,
    content         BYTEA       NOT NULL,
    locator         BYTEA       NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, session_id, intent_id, result_ordinal),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_evidence_ordinal_check CHECK (result_ordinal >= 0),
    CONSTRAINT dlightrag_answer_evidence_digest_check
        CHECK (content_digest ~ '^[0-9a-f]{64}$' AND locator_digest ~ '^[0-9a-f]{64}$')
)
"""

_CREATE_RESOURCES = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_resources (
    owner_id       TEXT        NOT NULL,
    run_id         UUID        NOT NULL,
    resource_id    TEXT        NOT NULL,
    kind           TEXT        NOT NULL,
    safe_name      TEXT        NOT NULL,
    media_type     TEXT        NOT NULL,
    capabilities   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    ordinal        INTEGER,
    blob_digest    TEXT,
    locator_digest TEXT,
    source_locator BYTEA,
    session_id     UUID,
    intent_id      UUID,
    result_ordinal INTEGER,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, resource_id),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_resources_kind_check
        CHECK (kind IN ('accepted_blob', 'evidence', 'fetched_blob', 'committed_spill')),
    CONSTRAINT dlightrag_answer_resources_blob_link_check
        CHECK ((kind = 'accepted_blob' AND blob_digest IS NOT NULL)
               OR (kind = 'fetched_blob'
                   AND blob_digest IS NOT NULL AND locator_digest IS NOT NULL
                   AND (capabilities->>'resource_kind' IS DISTINCT FROM 'web'
                        OR source_locator IS NOT NULL))
               OR (kind = 'evidence' AND locator_digest IS NOT NULL)
               OR (kind = 'committed_spill'
                   AND blob_digest IS NULL AND locator_digest IS NULL))
)
"""

_CREATE_BLOBS = """
CREATE TABLE IF NOT EXISTS dlightrag_blobs (
    owner_id   TEXT        NOT NULL,
    digest     TEXT        NOT NULL,
    byte_size  BIGINT      NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, digest),
    CONSTRAINT dlightrag_blobs_digest_check CHECK (digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT dlightrag_blobs_size_check CHECK (byte_size >= 0)
)
"""

_CREATE_BLOB_CHUNKS = """
CREATE TABLE IF NOT EXISTS dlightrag_blob_chunks (
    owner_id    TEXT    NOT NULL,
    digest      TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     BYTEA   NOT NULL,
    PRIMARY KEY (owner_id, digest, chunk_index),
    FOREIGN KEY (owner_id, digest)
        REFERENCES dlightrag_blobs (owner_id, digest) ON DELETE CASCADE,
    CONSTRAINT dlightrag_blob_chunks_index_check CHECK (chunk_index >= 0)
)
"""

_CREATE_RUN_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_run_artifacts (
    owner_id          TEXT        NOT NULL,
    run_id            UUID        NOT NULL,
    resource_id       TEXT        NOT NULL,
    reference_kind    TEXT        NOT NULL,
    ordinal           INTEGER     NOT NULL,
    digest            TEXT        NOT NULL,
    filename          TEXT        NOT NULL,
    mime_type         TEXT        NOT NULL,
    transform_locator JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, resource_id),
    UNIQUE (owner_id, run_id, reference_kind, ordinal),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    FOREIGN KEY (owner_id, digest)
        REFERENCES dlightrag_blobs (owner_id, digest) ON DELETE RESTRICT,
    CONSTRAINT dlightrag_answer_run_artifacts_kind_check
        CHECK (reference_kind IN
               ('current_attachment', 'history_attachment', 'fetched_resource',
                'published_artifact')),
    CONSTRAINT dlightrag_answer_run_artifacts_ordinal_check
        CHECK (ordinal >= 0)
)
"""

_CREATE_ROUTING = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_run_routing (
    owner_id                 TEXT        NOT NULL,
    run_id                   UUID        NOT NULL,
    requested_mode           TEXT        NOT NULL,
    valid_modes              TEXT[]      NOT NULL,
    resolved_mode            TEXT,
    model_fingerprints       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    context_policy_revision  TEXT        NOT NULL,
    agent_session_id         UUID        NOT NULL,
    agent_lane_id            TEXT        NOT NULL,
    source_lane_id           TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_run_routing_requested_check
        CHECK (requested_mode IN ('auto', 'fast', 'research')),
    CONSTRAINT dlightrag_answer_run_routing_valid_check
        CHECK (COALESCE(array_length(valid_modes, 1), 0) >= 1
               AND valid_modes <@ ARRAY['fast', 'research']::text[]),
    CONSTRAINT dlightrag_answer_run_routing_resolved_check
        CHECK (resolved_mode IS NULL
               OR (resolved_mode IN ('fast', 'research')
                   AND resolved_mode = ANY (valid_modes)))
)
"""

_CREATE_CHILD_SESSIONS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_child_sessions (
    owner_id           TEXT        NOT NULL,
    run_id             UUID        NOT NULL,
    child_session_id   UUID        NOT NULL,
    parent_session_id  UUID        NOT NULL,
    parent_call_id     TEXT        NOT NULL,
    parent_intent_id   UUID,
    status             TEXT        NOT NULL,
    summary            TEXT,
    objective          TEXT,
    context_mode       TEXT,
    model_role         TEXT,
    tools_json         JSONB,
    usage_json         JSONB,
    depth              INTEGER     NOT NULL,
    context_snapshot_json JSONB    NOT NULL,
    plan_json          JSONB,
    budget_json        JSONB,
    host_state_json    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    lease_owner        TEXT,
    lease_expires_at   TIMESTAMPTZ,
    fencing_epoch      BIGINT      NOT NULL DEFAULT 0,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, child_session_id),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_child_sessions_status_check
        CHECK (status IN ('running', 'succeeded', 'failed', 'cancelled')),
    CONSTRAINT dlightrag_answer_child_sessions_depth_check CHECK (depth >= 1),
    CONSTRAINT dlightrag_answer_child_sessions_fencing_check CHECK (fencing_epoch >= 0)
)
"""

_CREATE_AGENT_CONTROLS = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_controls (
    owner_id         TEXT        NOT NULL,
    run_id           UUID        NOT NULL,
    control_sequence BIGINT      NOT NULL,
    kind             TEXT        NOT NULL,
    content          TEXT        NOT NULL,
    consumed_at      TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, control_sequence),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_agent_controls_sequence_check CHECK (control_sequence >= 1),
    CONSTRAINT dlightrag_agent_controls_kind_check CHECK (kind IN ('steer', 'follow_up')),
    CONSTRAINT dlightrag_agent_controls_content_check CHECK (char_length(content) BETWEEN 1 AND 20000)
)
"""


_CREATE_INDEXES = (
    # Claim and sweep scan nonterminal rows oldest-first across every owner.
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_runs_claim "
    "ON dlightrag_runs (lane, next_attempt_at, created_at, run_id) "
    "WHERE status IN ('queued', 'running')",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_dlightrag_runs_submission "
    "ON dlightrag_runs (run_kind, submitted_by, submission_key)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_dlightrag_runs_global_id ON dlightrag_runs (run_id)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_runs_retention "
    "ON dlightrag_runs (purge_after) WHERE purge_after IS NOT NULL",
    # Reconnect/notification rescans page only this worker's live cancellations.
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_runs_cancel_pending "
    "ON dlightrag_runs (lease_owner, created_at, run_id) "
    "WHERE cancel_requested_at IS NOT NULL AND status = 'running'",
    # Exactly one terminal event per run, enforced durably rather than by convention.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_dlightrag_run_events_terminal "
    "ON dlightrag_run_events (owner_id, run_id) "
    "WHERE event_type IN ('done', 'error')",
    # Reverse lookup for ownership-safe blob cleanup and the RESTRICT foreign key.
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_run_artifacts_digest "
    "ON dlightrag_answer_run_artifacts (owner_id, digest)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_evidence_run "
    "ON dlightrag_answer_evidence (owner_id, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_resources_run "
    "ON dlightrag_answer_resources (owner_id, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_resources_blob "
    "ON dlightrag_answer_resources (owner_id, blob_digest) "
    "WHERE blob_digest IS NOT NULL",
    # Newest-first bounded child-roster keyset pages ride one exact order.
    "CREATE INDEX IF NOT EXISTS idx_answer_child_sessions_roster "
    "ON dlightrag_answer_child_sessions "
    "(owner_id, run_id, created_at DESC, child_session_id DESC)",
)

_CREATE_WORKSPACE_INVENTORY = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_workspace_inventory (
    owner_id        TEXT        NOT NULL,
    run_id          UUID        NOT NULL,
    relative_path   TEXT        NOT NULL,
    entry_type      TEXT        NOT NULL,
    mode            INTEGER,
    size_bytes      BIGINT      NOT NULL,
    content_digest  TEXT,
    PRIMARY KEY (owner_id, run_id, relative_path),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE
)
"""

_CREATE_ARTIFACT_ATTACHMENT_ORDER = """
CREATE SEQUENCE IF NOT EXISTS dlightrag_answer_artifact_attachment_order_seq
"""

_CREATE_ARTIFACT_ATTACHMENTS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_artifact_attachments (
    owner_id        TEXT        NOT NULL,
    run_id          UUID        NOT NULL,
    relative_path   TEXT        NOT NULL,
    label           TEXT        NOT NULL,
    content_digest  TEXT        NOT NULL,
    size_bytes      BIGINT      NOT NULL,
    presentation    TEXT        NOT NULL,
    session_id      UUID        NOT NULL,
    intent_id       UUID        NOT NULL,
    attachment_order BIGINT     NOT NULL DEFAULT nextval(
        'dlightrag_answer_artifact_attachment_order_seq'
    ),
    attached_at     TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (owner_id, run_id, relative_path),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_artifact_attachments_digest_check
        CHECK (content_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT dlightrag_answer_artifact_attachments_size_check
        CHECK (size_bytes >= 0),
    CONSTRAINT dlightrag_answer_artifact_attachments_presentation_check
        CHECK (presentation IN ('image', 'markdown', 'html', 'pdf', 'text', 'download'))
)
"""

_CREATE_CORPUS_MUTATION_WINDOWS = """
CREATE TABLE IF NOT EXISTS dlightrag_corpus_mutation_windows (
    run_id          UUID        NOT NULL,
    window_number   INTEGER     NOT NULL,
    workspace       TEXT        NOT NULL,
    docs             BIGINT      NOT NULL,
    chunks           BIGINT      NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, window_number),
    FOREIGN KEY (run_id) REFERENCES dlightrag_runs (run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_corpus_mutation_windows_nonnegative
        CHECK (window_number > 0 AND docs >= 0 AND chunks >= 0)
)
"""

_CLEAN_BREAK_CORPUS_MUTATIONS = """
DROP TABLE IF EXISTS dlightrag_failed_retry_items CASCADE;
DROP TABLE IF EXISTS dlightrag_ingest_counters CASCADE;
DROP TABLE IF EXISTS dlightrag_ingest_jobs CASCADE;
"""

_CREATE_COMMITTED_SPILLS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_committed_spills (
    owner_id        TEXT        NOT NULL,
    run_id          UUID        NOT NULL,
    resource_id     TEXT        NOT NULL,
    content_digest  TEXT        NOT NULL,
    size_bytes      BIGINT      NOT NULL,
    session_id      UUID        NOT NULL,
    intent_id       UUID        NOT NULL,
    PRIMARY KEY (owner_id, run_id, resource_id),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE
)
"""

# The baseline bakes the current schema directly into CREATE statements. Later
# migrations advance initialized databases without adding runtime compatibility paths.

RUN_MIGRATIONS = (
    Migration(
        "run_runtime_v1",
        "Rename terminal Answer rows and create the operation-neutral RunRuntime schema",
        (
            _MIGRATE_ANSWER_RUNTIME,
            _MIGRATE_RUN_COLUMNS,
            _CREATE_RUNS,
            _ALTER_RUN_RUNTIME,
            _NORMALIZE_RUN_CONSTRAINTS,
            _CREATE_EVENTS,
            _NORMALIZE_RUN_EVENTS,
            _CREATE_SESSIONS,
            _CREATE_ENTRIES,
            _CREATE_SESSION_REGISTERS,
            _CREATE_STAGES,
            _CREATE_EVIDENCE,
            _CREATE_RESOURCES,
            _CREATE_BLOBS,
            _CREATE_BLOB_CHUNKS,
            _CREATE_RUN_ARTIFACTS,
            _CREATE_ROUTING,
            _CREATE_CHILD_SESSIONS,
            _CREATE_AGENT_CONTROLS,
            *_CREATE_INDEXES,
            _CREATE_WORKSPACE_INVENTORY,
            _CREATE_ARTIFACT_ATTACHMENT_ORDER,
            _CREATE_ARTIFACT_ATTACHMENTS,
            _CREATE_COMMITTED_SPILLS,
            *MEMORY_SETTINGS_DDL,
        ),
    ),
    Migration(
        "child_roster_index",
        "Index child sessions for bounded newest-first roster pages",
        (
            "CREATE INDEX IF NOT EXISTS idx_answer_child_sessions_roster "
            "ON dlightrag_answer_child_sessions "
            "(owner_id, run_id, created_at DESC, child_session_id DESC)",
        ),
    ),
    Migration(
        "worker_cancel_pending_index",
        "Index bounded worker-local cancellation rescans",
        (
            "CREATE INDEX IF NOT EXISTS idx_dlightrag_runs_cancel_pending "
            "ON dlightrag_runs (lease_owner, created_at, run_id) "
            "WHERE cancel_requested_at IS NOT NULL AND status = 'running'",
        ),
    ),
    Migration(
        "write_model_published_artifact_kind",
        "Restrict durable output references to the unified publication kind",
        (
            "ALTER TABLE dlightrag_answer_run_artifacts "
            "DROP CONSTRAINT dlightrag_answer_run_artifacts_kind_check",
            "ALTER TABLE dlightrag_answer_run_artifacts "
            "ADD CONSTRAINT dlightrag_answer_run_artifacts_kind_check "
            "CHECK (reference_kind IN "
            "('current_attachment', 'history_attachment', 'fetched_resource', "
            "'published_artifact'))",
        ),
    ),
    Migration(
        "write_model_root_artifact_attachments",
        "Stage root Artifact attachments through settled Agent tool effects",
        (_CREATE_ARTIFACT_ATTACHMENT_ORDER, _CREATE_ARTIFACT_ATTACHMENTS),
    ),
    Migration(
        "write_model_web_resource_catalog",
        "Persist fetched Web locators for process-independent resource recovery",
        (
            "ALTER TABLE dlightrag_answer_resources ADD COLUMN IF NOT EXISTS source_locator BYTEA",
            "ALTER TABLE dlightrag_answer_resources "
            "DROP CONSTRAINT dlightrag_answer_resources_blob_link_check",
            "ALTER TABLE dlightrag_answer_resources "
            "ADD CONSTRAINT dlightrag_answer_resources_blob_link_check "
            "CHECK ((kind = 'accepted_blob' AND blob_digest IS NOT NULL) "
            "OR (kind = 'fetched_blob' AND blob_digest IS NOT NULL "
            "AND locator_digest IS NOT NULL "
            "AND (capabilities->>'resource_kind' IS DISTINCT FROM 'web' "
            "OR source_locator IS NOT NULL)) "
            "OR (kind = 'evidence' AND locator_digest IS NOT NULL) "
            "OR (kind = 'committed_spill' AND blob_digest IS NULL "
            "AND locator_digest IS NULL))",
        ),
    ),
    Migration(
        "corpus_mutation_runtime",
        "Add fenced Corpus Mutation checkpoints and remove the legacy ingest lifecycle",
        (
            "ALTER TABLE dlightrag_runs ADD COLUMN IF NOT EXISTS handoff_started_at TIMESTAMPTZ",
            "ALTER TABLE dlightrag_runs ADD COLUMN IF NOT EXISTS superseded_by_run_id UUID",
            _CREATE_CORPUS_MUTATION_WINDOWS,
            _CLEAN_BREAK_CORPUS_MUTATIONS,
            "CREATE INDEX IF NOT EXISTS idx_dlightrag_runs_mutation_fifo "
            "ON dlightrag_runs (owner_id, created_at, run_id) "
            "WHERE lane = 'corpus_mutation' AND status IN ('queued', 'running')",
        ),
    ),
    Migration(
        "normalize_run_event_constraints",
        "Enforce event sequence, parent lifecycle, and terminal payload integrity",
        (_ENFORCE_RUN_EVENT_CONSTRAINTS, _CREATE_RUN_EVENT_CONSTRAINT_TRIGGER),
    ),
    Migration(
        "remove_run_active_permit",
        "Remove the obsolete Run occupancy column",
        (
            "ALTER TABLE dlightrag_runs "
            "DROP CONSTRAINT IF EXISTS dlightrag_runs_permit_check, "
            "DROP COLUMN IF EXISTS active_permit",
        ),
    ),
)

RUN_SCHEMA_TABLES = (
    TableRequirement(
        name="dlightrag_runs",
        columns=(
            "owner_id",
            "run_id",
            "run_kind",
            "lane",
            "submitted_by",
            "access_scope_kind",
            "submission_key",
            "prepared_input_json",
            "accepted_input_json",
            "request_fingerprint",
            "status",
            "phase",
            "stop_reason",
            "cancel_requested_at",
            "lease_owner",
            "lease_expires_at",
            "fencing_epoch",
            "durable_progress_version",
            "last_reclaim_progress_version",
            "reclaims_without_progress",
            "next_event_sequence",
            "events_trimmed_at",
            "result_json",
            "error_kind",
            "error_message",
            "retention_seconds",
            "purge_after",
            "next_attempt_at",
            "checkpoint_json",
            "handoff_started_at",
            "superseded_by_run_id",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
            "agent_workspace_epoch",
        ),
        primary_key=("owner_id", "run_id"),
        checks=(
            "dlightrag_runs_kind_check",
            "dlightrag_runs_lane_check",
            "dlightrag_runs_scope_check",
            "dlightrag_runs_status_check",
            "dlightrag_runs_counter_check",
            "dlightrag_runs_lease_check",
            "dlightrag_runs_terminal_check",
            "dlightrag_runs_result_check",
            "dlightrag_runs_error_check",
            "dlightrag_runs_prepared_input_check",
            "dlightrag_runs_workspace_epoch_check",
        ),
        indexes=(
            "idx_dlightrag_runs_claim",
            "idx_dlightrag_runs_retention",
            "idx_dlightrag_runs_cancel_pending",
        ),
        unique_indexes=(
            "idx_dlightrag_runs_submission",
            "idx_dlightrag_runs_global_id",
        ),
    ),
    TableRequirement(
        name="dlightrag_corpus_mutation_windows",
        columns=("run_id", "window_number", "workspace", "docs", "chunks", "created_at"),
        primary_key=("run_id", "window_number"),
        foreign_keys=(ForeignKeyRequirement(columns=("run_id",), references="dlightrag_runs"),),
        checks=("dlightrag_corpus_mutation_windows_nonnegative",),
    ),
    TableRequirement(
        name="dlightrag_run_events",
        columns=(
            "owner_id",
            "run_id",
            "event_sequence",
            "event_type",
            "payload",
            "created_at",
        ),
        primary_key=("owner_id", "run_id", "event_sequence"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=("dlightrag_run_events_sequence_check",),
        unique_indexes=("idx_dlightrag_run_events_terminal",),
    ),
    TableRequirement(
        name="dlightrag_agent_sessions",
        columns=(
            "owner_id",
            "session_id",
            "lease_run_id",
            "commit_sequence",
            "fencing_epoch",
            "last_sequence",
            "created_at",
            "updated_at",
        ),
        primary_key=("owner_id", "session_id"),
        checks=(
            "dlightrag_agent_sessions_commit_sequence_check",
            "dlightrag_agent_sessions_fencing_check",
            "dlightrag_agent_sessions_sequence_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_agent_session_entries",
        columns=(
            "owner_id",
            "session_id",
            "sequence",
            "entry_id",
            "parent_entry_id",
            "entry_type",
            "schema_version",
            "timestamp",
            "payload_json",
        ),
        primary_key=("owner_id", "session_id", "sequence"),
        unique=(
            ("entry_id",),
            ("owner_id", "session_id", "entry_id"),
        ),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "session_id"),
                references="dlightrag_agent_sessions",
            ),
            ForeignKeyRequirement(
                columns=("owner_id", "session_id", "parent_entry_id"),
                references="dlightrag_agent_session_entries",
            ),
        ),
        checks=(
            "dlightrag_agent_session_entries_sequence_check",
            "dlightrag_agent_session_entries_type_check",
            "dlightrag_agent_session_entries_version_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_agent_session_registers",
        columns=(
            "owner_id",
            "session_id",
            "register_kind",
            "register_key",
            "sequence",
            "payload_json",
        ),
        primary_key=(
            "owner_id",
            "session_id",
            "register_kind",
            "register_key",
        ),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "session_id"),
                references="dlightrag_agent_sessions",
            ),
        ),
        checks=(
            "dlightrag_agent_session_registers_kind_check",
            "dlightrag_agent_session_registers_sequence_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_run_stages",
        columns=(
            "owner_id",
            "run_id",
            "stage_intent_id",
            "stage_name",
            "progress_version",
            "state",
            "state_digest",
            "settled_at",
        ),
        primary_key=("owner_id", "run_id", "stage_intent_id"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=(
            "dlightrag_answer_run_stages_name_check",
            "dlightrag_answer_run_stages_digest_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_evidence",
        columns=(
            "owner_id",
            "run_id",
            "session_id",
            "intent_id",
            "result_ordinal",
            "content_digest",
            "locator_digest",
            "content",
            "locator",
            "created_at",
        ),
        primary_key=("owner_id", "run_id", "session_id", "intent_id", "result_ordinal"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=(
            "dlightrag_answer_evidence_ordinal_check",
            "dlightrag_answer_evidence_digest_check",
        ),
        indexes=("idx_dlightrag_answer_evidence_run",),
    ),
    TableRequirement(
        name="dlightrag_answer_resources",
        columns=(
            "owner_id",
            "run_id",
            "resource_id",
            "kind",
            "safe_name",
            "media_type",
            "capabilities",
            "ordinal",
            "blob_digest",
            "locator_digest",
            "source_locator",
            "session_id",
            "intent_id",
            "result_ordinal",
            "created_at",
        ),
        primary_key=("owner_id", "run_id", "resource_id"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=(
            "dlightrag_answer_resources_kind_check",
            "dlightrag_answer_resources_blob_link_check",
        ),
        indexes=(
            "idx_dlightrag_answer_resources_run",
            "idx_dlightrag_answer_resources_blob",
        ),
    ),
    TableRequirement(
        name="dlightrag_blobs",
        columns=("owner_id", "digest", "byte_size", "created_at"),
        primary_key=("owner_id", "digest"),
        checks=(
            "dlightrag_blobs_digest_check",
            "dlightrag_blobs_size_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_blob_chunks",
        columns=("owner_id", "digest", "chunk_index", "content"),
        primary_key=("owner_id", "digest", "chunk_index"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "digest"), references="dlightrag_blobs"),
        ),
        checks=("dlightrag_blob_chunks_index_check",),
    ),
    TableRequirement(
        name="dlightrag_answer_run_artifacts",
        columns=(
            "owner_id",
            "run_id",
            "resource_id",
            "reference_kind",
            "ordinal",
            "digest",
            "filename",
            "mime_type",
            "transform_locator",
            "created_at",
        ),
        primary_key=("owner_id", "run_id", "resource_id"),
        unique=(("owner_id", "run_id", "reference_kind", "ordinal"),),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
            ForeignKeyRequirement(columns=("owner_id", "digest"), references="dlightrag_blobs"),
        ),
        checks=(
            "dlightrag_answer_run_artifacts_kind_check",
            "dlightrag_answer_run_artifacts_ordinal_check",
        ),
        indexes=("idx_dlightrag_answer_run_artifacts_digest",),
    ),
    TableRequirement(
        name="dlightrag_answer_workspace_inventory",
        columns=(
            "owner_id",
            "run_id",
            "relative_path",
            "entry_type",
            "mode",
            "size_bytes",
            "content_digest",
        ),
        primary_key=("owner_id", "run_id", "relative_path"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_artifact_attachments",
        columns=(
            "owner_id",
            "run_id",
            "relative_path",
            "label",
            "content_digest",
            "size_bytes",
            "presentation",
            "session_id",
            "intent_id",
            "attachment_order",
            "attached_at",
        ),
        primary_key=("owner_id", "run_id", "relative_path"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=(
            "dlightrag_answer_artifact_attachments_digest_check",
            "dlightrag_answer_artifact_attachments_size_check",
            "dlightrag_answer_artifact_attachments_presentation_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_run_routing",
        columns=(
            "owner_id",
            "run_id",
            "requested_mode",
            "valid_modes",
            "resolved_mode",
            "model_fingerprints",
            "context_policy_revision",
            "agent_session_id",
            "agent_lane_id",
            "source_lane_id",
            "created_at",
            "updated_at",
        ),
        primary_key=("owner_id", "run_id"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=(
            "dlightrag_answer_run_routing_requested_check",
            "dlightrag_answer_run_routing_valid_check",
            "dlightrag_answer_run_routing_resolved_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_child_sessions",
        columns=(
            "owner_id",
            "run_id",
            "child_session_id",
            "parent_session_id",
            "parent_call_id",
            "parent_intent_id",
            "status",
            "summary",
            "objective",
            "context_mode",
            "model_role",
            "tools_json",
            "usage_json",
            "depth",
            "context_snapshot_json",
            "plan_json",
            "budget_json",
            "host_state_json",
            "lease_owner",
            "lease_expires_at",
            "fencing_epoch",
            "created_at",
            "updated_at",
        ),
        primary_key=("owner_id", "run_id", "child_session_id"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        indexes=("idx_answer_child_sessions_roster",),
        checks=(
            "dlightrag_answer_child_sessions_status_check",
            "dlightrag_answer_child_sessions_depth_check",
            "dlightrag_answer_child_sessions_fencing_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_agent_controls",
        columns=(
            "owner_id",
            "run_id",
            "control_sequence",
            "kind",
            "content",
            "consumed_at",
            "created_at",
        ),
        primary_key=("owner_id", "run_id", "control_sequence"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=(
            "dlightrag_agent_controls_sequence_check",
            "dlightrag_agent_controls_kind_check",
            "dlightrag_agent_controls_content_check",
        ),
    ),
    MEMORY_SETTINGS_SCHEMA_TABLE,
    TableRequirement(
        name="dlightrag_answer_committed_spills",
        columns=(
            "owner_id",
            "run_id",
            "resource_id",
            "content_digest",
            "size_bytes",
            "session_id",
            "intent_id",
        ),
        primary_key=("owner_id", "run_id", "resource_id"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
    ),
)

#: ``(expression, output name)`` for every column :func:`run_record` reads.
_RUN_COLUMN_SPECS: tuple[tuple[str, str], ...] = (
    ("owner_id", "owner_id"),
    ("run_id::text", "run_id"),
    ("run_kind", "run_kind"),
    ("lane", "lane"),
    ("submitted_by", "submitted_by"),
    ("access_scope_kind", "access_scope_kind"),
    ("submission_key", "submission_key"),
    ("request_fingerprint", "request_fingerprint"),
    ("prepared_input_json", "prepared_input"),
    ("accepted_input_json", "accepted_input"),
    ("status", "status"),
    ("phase", "phase"),
    ("stop_reason", "stop_reason"),
    ("cancel_requested_at", "cancel_requested_at"),
    ("lease_owner", "lease_owner"),
    ("lease_expires_at", "lease_expires_at"),
    ("fencing_epoch", "fencing_epoch"),
    ("durable_progress_version", "durable_progress_version"),
    ("last_reclaim_progress_version", "last_reclaim_progress_version"),
    ("reclaims_without_progress", "reclaims_without_progress"),
    ("next_event_sequence", "next_event_sequence"),
    ("events_trimmed_at", "events_trimmed_at"),
    ("result_json", "result_json"),
    ("error_kind", "error_kind"),
    ("error_message", "error_message"),
    ("created_at", "created_at"),
    ("updated_at", "updated_at"),
    ("started_at", "started_at"),
    ("finished_at", "finished_at"),
    ("purge_after", "purge_after"),
    ("next_attempt_at", "next_attempt_at"),
    ("checkpoint_json", "checkpoint_json"),
    ("handoff_started_at", "handoff_started_at"),
    ("superseded_by_run_id::text", "superseded_by_run_id"),
    ("agent_workspace_epoch", "agent_workspace_epoch"),
)


def run_columns(alias: str = "") -> str:
    """Project one run row's columns, optionally through a join alias."""
    prefix = f"{alias}." if alias else ""
    return ",\n".join(f"{prefix}{expression} AS {name}" for expression, name in _RUN_COLUMN_SPECS)


_RUN_COLUMNS = run_columns()
_LIST_RUNS = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_runs
WHERE owner_id = $1 ORDER BY created_at, run_id LIMIT $2
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant
_LIST_RUNS_AFTER = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_runs
WHERE owner_id = $1 AND (created_at, run_id) > (
 SELECT created_at, run_id FROM dlightrag_runs
 WHERE owner_id = $1 AND run_id = $2)
ORDER BY created_at, run_id LIMIT $3
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_ACTIVE_REQUIREMENTS_FRONTIER = """
SELECT created_at, run_id
FROM dlightrag_runs
WHERE run_kind IN ('answer', 'retrieval')
  AND status IN ('queued', 'running')
  AND cancel_requested_at IS NULL
  AND NOT (status = 'running' AND lease_expires_at < NOW()
           AND reclaims_without_progress >= $1)
ORDER BY created_at DESC, run_id DESC
LIMIT 1
"""
_ACTIVE_REQUIREMENTS_FIRST_PAGE = """
SELECT created_at, run_id, run_kind, prepared_input_json
FROM dlightrag_runs
WHERE run_kind IN ('answer', 'retrieval')
  AND status IN ('queued', 'running')
  AND cancel_requested_at IS NULL
  AND NOT (status = 'running' AND lease_expires_at < NOW()
           AND reclaims_without_progress >= $1)
  AND (created_at, run_id) <= ($2::timestamptz, $3::uuid)
ORDER BY created_at, run_id
LIMIT $4
"""
_ACTIVE_REQUIREMENTS_AFTER = """
SELECT created_at, run_id, run_kind, prepared_input_json
FROM dlightrag_runs
WHERE run_kind IN ('answer', 'retrieval')
  AND status IN ('queued', 'running')
  AND cancel_requested_at IS NULL
  AND NOT (status = 'running' AND lease_expires_at < NOW()
           AND reclaims_without_progress >= $1)
  AND (created_at, run_id) <= ($2::timestamptz, $3::uuid)
  AND (created_at, run_id) > ($4::timestamptz, $5::uuid)
ORDER BY created_at, run_id
LIMIT $6
"""

_CANCEL_PENDING_FRONTIER = """
SELECT created_at, run_id
FROM dlightrag_runs
WHERE cancel_requested_at IS NOT NULL
  AND status = 'running'
  AND lease_owner = $1
  AND lease_expires_at > NOW()
ORDER BY created_at DESC, run_id DESC
LIMIT 1
"""
_CANCEL_PENDING_FIRST_PAGE = """
SELECT owner_id, run_id, created_at
FROM dlightrag_runs
WHERE cancel_requested_at IS NOT NULL
  AND status = 'running'
  AND lease_owner = $1
  AND lease_expires_at > NOW()
  AND (created_at, run_id) <= ($2::timestamptz, $3::uuid)
ORDER BY created_at, run_id
LIMIT $4
"""
_CANCEL_PENDING_AFTER = """
SELECT owner_id, run_id, created_at
FROM dlightrag_runs
WHERE cancel_requested_at IS NOT NULL
  AND status = 'running'
  AND lease_owner = $1
  AND lease_expires_at > NOW()
  AND (created_at, run_id) <= ($2::timestamptz, $3::uuid)
  AND (created_at, run_id) > ($4::timestamptz, $5::uuid)
ORDER BY created_at, run_id
LIMIT $6
"""

_INSERT_RUN = f"""
INSERT INTO dlightrag_runs (
    owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind,
    submission_key, prepared_input_json, accepted_input_json,
    request_fingerprint, retention_seconds
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10, $11)
ON CONFLICT (run_kind, submitted_by, submission_key) DO NOTHING
RETURNING {_RUN_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_SELECT_RUN_BY_KEY = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_runs
WHERE run_kind = $1 AND submitted_by = $2 AND submission_key = $3
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_COUNT_NONTERMINAL_LANE = """
SELECT COUNT(*) FROM dlightrag_runs
WHERE lane = $1 AND status IN ('queued', 'running')
"""

_SELECT_RUN = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_runs
WHERE owner_id = $1 AND run_id = $2
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_SELECT_RUN_FOR_UPDATE = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_runs
WHERE owner_id = $1 AND run_id = $2
FOR UPDATE
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_SELECT_RUN_GLOBAL = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_runs
WHERE run_id = $1
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_RECORD_CORPUS_WINDOW = """
WITH inserted AS (
    INSERT INTO dlightrag_corpus_mutation_windows (
        run_id, window_number, workspace, docs, chunks
    ) VALUES ($1, $2, $3, $4, $5)
    ON CONFLICT (run_id, window_number) DO NOTHING
    RETURNING 1
), updated AS (
    UPDATE dlightrag_workspace_meta
    SET ingested_docs_total = ingested_docs_total + $4,
        ingested_chunks_total = ingested_chunks_total + $5,
        promotion_state = CASE
            WHEN promotion_state = 'none' AND storage_tier = 'shared'
             AND (($6::bigint IS NOT NULL AND ingested_docs_total + $4 >= $6)
               OR ($7::bigint IS NOT NULL AND ingested_chunks_total + $5 >= $7))
            THEN 'pending' ELSE promotion_state END,
        updated_at = NOW()
    WHERE workspace = $3 AND EXISTS (SELECT 1 FROM inserted)
    RETURNING promotion_state
), promoted AS (
    INSERT INTO dlightrag_promotion_jobs (workspace, state)
    SELECT $3, 'pending' FROM updated WHERE promotion_state = 'pending'
    ON CONFLICT DO NOTHING
)
SELECT EXISTS (SELECT 1 FROM inserted)
"""

_SELECT_EVENTS = """
SELECT event_sequence, event_type, payload, created_at
FROM dlightrag_run_events
WHERE owner_id = $1 AND run_id = $2 AND event_sequence > $3
ORDER BY event_sequence
LIMIT $4
"""

_SELECT_CLAIM_CANDIDATE = """
SELECT r.owner_id, r.run_id
FROM dlightrag_runs r
WHERE r.run_kind = ANY($2::text[])
  AND r.lane = ANY($3::text[])
  AND r.cancel_requested_at IS NULL
  AND (r.next_attempt_at IS NULL OR r.next_attempt_at <= NOW())
  AND (
      r.status = 'queued'
      OR (r.status = 'running' AND r.lease_expires_at < NOW()
          AND r.reclaims_without_progress < $1)
  )
  AND (
      r.lane <> 'corpus_mutation'
      OR NOT EXISTS (
          SELECT 1 FROM dlightrag_runs earlier
          WHERE earlier.lane = 'corpus_mutation'
            AND earlier.owner_id = r.owner_id
            AND earlier.status IN ('queued', 'running')
            AND (earlier.created_at, earlier.run_id) < (r.created_at, r.run_id)
      )
  )
ORDER BY r.created_at, r.run_id
LIMIT 1
FOR UPDATE OF r SKIP LOCKED
"""

_LOCK_AGENT_SESSION_IF_PRESENT = """
SELECT 1
FROM dlightrag_agent_sessions
WHERE owner_id = $1 AND session_id = $2
FOR UPDATE
"""

_INSERT_ROUTING = """
INSERT INTO dlightrag_answer_run_routing (
    owner_id, run_id, requested_mode, valid_modes, resolved_mode,
    model_fingerprints, context_policy_revision,
    agent_session_id, agent_lane_id, source_lane_id
)
VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10)
"""

_SELECT_ROUTING = """
SELECT requested_mode, valid_modes, resolved_mode,
       agent_session_id::text, agent_lane_id, source_lane_id
FROM dlightrag_answer_run_routing
WHERE owner_id = $1 AND run_id = $2
"""

_RESOLVE_ROUTING = """
UPDATE dlightrag_answer_run_routing AS rt
SET resolved_mode = $5,
    updated_at = NOW()
FROM dlightrag_runs AS r
WHERE rt.owner_id = r.owner_id AND rt.run_id = r.run_id
  AND rt.owner_id = $1 AND rt.run_id = $2
  AND r.lease_owner = $3 AND r.fencing_epoch = $4
  AND r.status = 'running' AND r.lease_expires_at > NOW()
  AND (rt.resolved_mode IS NULL OR rt.resolved_mode = $5)
RETURNING rt.resolved_mode
"""

_HOLD_RUN_LEASE = """
SELECT 1
FROM dlightrag_runs
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
FOR UPDATE
"""

_UPSERT_CHILD_SESSION = """
INSERT INTO dlightrag_answer_child_sessions (
    owner_id, run_id, child_session_id, parent_session_id, parent_call_id,
    parent_intent_id, status, objective, context_mode, model_role, tools_json,
    depth, context_snapshot_json, plan_json, budget_json, host_state_json
)
VALUES (
    $1, $2, $3, $4, $5, $6, 'running', $7, $8, $9, $10,
    $11, $12::jsonb, $13::jsonb, $14::jsonb, $15::jsonb
)
ON CONFLICT (owner_id, run_id, child_session_id) DO UPDATE
SET parent_intent_id = COALESCE(
        dlightrag_answer_child_sessions.parent_intent_id,
        EXCLUDED.parent_intent_id
    ),
    plan_json = COALESCE(EXCLUDED.plan_json, dlightrag_answer_child_sessions.plan_json),
    budget_json = COALESCE(EXCLUDED.budget_json, dlightrag_answer_child_sessions.budget_json),
    host_state_json = COALESCE(
        EXCLUDED.host_state_json, dlightrag_answer_child_sessions.host_state_json
    ),
    updated_at = NOW()
"""

_CLAIM_CHILD_SESSION = """
UPDATE dlightrag_answer_child_sessions
SET lease_owner = $4,
    lease_expires_at = NOW() + ($5 * INTERVAL '1 second'),
    fencing_epoch = fencing_epoch + 1,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND status = 'running'
  AND (lease_expires_at IS NULL OR lease_expires_at < NOW() OR lease_owner = $4)
RETURNING fencing_epoch
"""

_RENEW_CHILD_SESSION_LEASE = """
UPDATE dlightrag_answer_child_sessions
SET lease_expires_at = NOW() + ($6 * INTERVAL '1 second'),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND lease_owner = $4 AND fencing_epoch = $5
  AND status = 'running' AND lease_expires_at > NOW()
RETURNING 1
"""

_SELECT_CHILD_SESSION = """
SELECT child_session_id, status, summary, parent_intent_id,
       objective, context_mode, model_role, tools_json, usage_json,
       depth, context_snapshot_json, plan_json, budget_json, host_state_json,
       lease_owner, lease_expires_at, fencing_epoch
FROM dlightrag_answer_child_sessions
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
"""

_SELECT_CHILD_SESSIONS = """
SELECT child_session_id, parent_session_id, parent_call_id, parent_intent_id,
       status, summary, objective, context_mode, model_role, tools_json, usage_json,
       depth, context_snapshot_json, plan_json, budget_json, host_state_json,
       lease_owner, lease_expires_at, fencing_epoch, created_at, updated_at
FROM dlightrag_answer_child_sessions
WHERE owner_id = $1 AND run_id = $2
ORDER BY created_at, child_session_id
"""

_CHILD_ROSTER_COLUMNS = """
child_session_id, parent_session_id, parent_call_id, parent_intent_id,
status, summary, objective, context_mode, model_role, tools_json, usage_json,
depth, context_snapshot_json, plan_json, budget_json, host_state_json,
lease_owner, lease_expires_at, fencing_epoch, created_at, updated_at
"""

_SELECT_CHILD_SESSIONS_FIRST_PAGE = f"""
SELECT {_CHILD_ROSTER_COLUMNS}
FROM dlightrag_answer_child_sessions
WHERE owner_id = $1 AND run_id = $2
ORDER BY created_at DESC, child_session_id DESC
LIMIT $3
"""  # noqa: S608 - interpolates only the trusted column constant

_SELECT_CHILD_SESSIONS_AFTER = f"""
SELECT {_CHILD_ROSTER_COLUMNS}
FROM dlightrag_answer_child_sessions
WHERE owner_id = $1 AND run_id = $2
  AND (created_at < $3::timestamptz
       OR (created_at = $3::timestamptz AND child_session_id < $4::uuid))
ORDER BY created_at DESC, child_session_id DESC
LIMIT $5
"""  # noqa: S608 - interpolates only the trusted column constant

_SELECT_AGENT_TRANSCRIPT = """
WITH RECURSIVE authorized AS (
    SELECT agent_session_id, agent_lane_id
    FROM dlightrag_answer_run_routing
    WHERE owner_id = $1 AND run_id = $2 AND agent_session_id = $3
), ancestry AS (
    SELECT e.entry_id, e.parent_entry_id, e.sequence, e.entry_type, e.payload_json
    FROM authorized AS a
    JOIN dlightrag_agent_session_entries AS e
      ON e.owner_id = $1 AND e.session_id = a.agent_session_id
    JOIN dlightrag_agent_session_registers AS r
      ON r.owner_id = e.owner_id AND r.session_id = e.session_id
     AND r.register_kind = 'lane_head' AND r.register_key = a.agent_lane_id
     AND e.entry_id = NULLIF(r.payload_json->>'entry_id', '')::uuid
    UNION ALL
    SELECT parent.entry_id, parent.parent_entry_id, parent.sequence,
           parent.entry_type, parent.payload_json
    FROM dlightrag_agent_session_entries AS parent
    JOIN ancestry AS child ON child.parent_entry_id = parent.entry_id
    WHERE parent.owner_id = $1 AND parent.session_id = $3
)
SELECT entry_type, payload_json
FROM ancestry
WHERE entry_type IN ('user_message', 'assistant_message', 'tool_result', 'control_message')
ORDER BY sequence DESC
LIMIT $4
"""

_LOCK_CONTROL_RUN = """
SELECT r.status, rt.requested_mode, rt.resolved_mode
FROM dlightrag_runs AS r
JOIN dlightrag_answer_run_routing AS rt
  ON rt.owner_id = r.owner_id AND rt.run_id = r.run_id
WHERE r.owner_id = $1 AND r.run_id = $2
FOR UPDATE OF r
"""

_NEXT_CONTROL_SEQUENCE = """
SELECT COALESCE(MAX(control_sequence), 0) + 1
FROM dlightrag_agent_controls
WHERE owner_id = $1 AND run_id = $2
"""

_INSERT_CONTROL = """
INSERT INTO dlightrag_agent_controls (
    owner_id, run_id, control_sequence, kind, content
)
VALUES ($1, $2, $3, $4, $5)
"""

_SELECT_PENDING_CONTROLS = """
SELECT control_sequence, kind, content, created_at
FROM dlightrag_agent_controls
WHERE owner_id = $1 AND run_id = $2 AND consumed_at IS NULL
ORDER BY control_sequence
FOR UPDATE
"""

_CONSUME_CONTROLS = """
UPDATE dlightrag_agent_controls
SET consumed_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND control_sequence = ANY($3::bigint[]) AND consumed_at IS NULL
"""

_BIND_CHILD_PARENT_INTENT = """
UPDATE dlightrag_answer_child_sessions
SET parent_intent_id = $4, updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND parent_intent_id IS NULL
"""

_FINISH_CHILD_SESSION = """
UPDATE dlightrag_answer_child_sessions
SET status = $4,
    summary = $5,
    usage_json = $6,
    host_state_json = jsonb_set(
        host_state_json,
        '{terminal_outcome}',
        $7::jsonb,
        true
    ),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
"""

_CLAIM_RUN = f"""
UPDATE dlightrag_runs
SET status = 'running',
    lease_owner = $3,
    lease_expires_at = NOW() + ($4 * INTERVAL '1 second'),
    fencing_epoch = fencing_epoch + 1,
    reclaims_without_progress = $5,
    last_reclaim_progress_version = $6,
    started_at = COALESCE(started_at, NOW()),
    next_attempt_at = NULL,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND status IN ('queued', 'running')
RETURNING {_RUN_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_HEARTBEAT = """
UPDATE dlightrag_runs
SET lease_expires_at = NOW() + ($5 * INTERVAL '1 second'),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
RETURNING (cancel_requested_at IS NOT NULL) AS cancel_requested
"""

_WRITE_CHECKPOINT = """
UPDATE dlightrag_runs
SET checkpoint_json = $5::jsonb,
    phase = COALESCE($6::text, phase),
    durable_progress_version = durable_progress_version + 1,
    lease_expires_at = NOW() + ($7 * INTERVAL '1 second'),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
RETURNING 1
"""

_START_HANDOFF = """
UPDATE dlightrag_runs
SET handoff_started_at = COALESCE(handoff_started_at, NOW()),
    checkpoint_json = $5::jsonb,
    durable_progress_version = durable_progress_version + 1,
    lease_expires_at = NOW() + ($6 * INTERVAL '1 second'),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
  AND (handoff_started_at IS NOT NULL OR cancel_requested_at IS NULL)
RETURNING 1
"""

_APPEND_EVENT = """
WITH bumped AS (
    UPDATE dlightrag_runs
    SET next_event_sequence = next_event_sequence + 1,
        phase = COALESCE($5::text, phase),
        lease_expires_at = NOW() + ($8 * INTERVAL '1 second'),
        updated_at = NOW()
    WHERE owner_id = $1 AND run_id = $2
      AND lease_owner = $3 AND fencing_epoch = $4
      AND status = 'running' AND lease_expires_at > NOW()
    RETURNING next_event_sequence - 1 AS event_sequence
), inserted AS (
    INSERT INTO dlightrag_run_events (
        owner_id, run_id, event_sequence, event_type, payload
    )
    SELECT $1, $2, event_sequence, $6::text, $7::jsonb FROM bumped
    RETURNING event_sequence
)
SELECT event_sequence FROM inserted
"""

_FINALIZE_UNLEASED = """
WITH bumped AS (
    UPDATE dlightrag_runs AS r
    SET status = $3::text,
        cancel_requested_at = CASE
            WHEN $3::text = 'cancelled' THEN COALESCE(r.cancel_requested_at, NOW())
            ELSE r.cancel_requested_at
        END,
        stop_reason = NULL,
        error_kind = $4::text,
        error_message = $5::text,
        phase = NULL,
        prepared_input_json = NULL,
        lease_owner = NULL,
        lease_expires_at = NULL,
        finished_at = NOW(),
        purge_after = NOW() + make_interval(secs => retention_seconds::double precision),
        updated_at = NOW(),
        next_event_sequence = r.next_event_sequence + 1
    WHERE (r.owner_id, r.run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
      AND r.status IN ('queued', 'running')
      AND (r.lease_expires_at IS NULL OR r.lease_expires_at < NOW())
    RETURNING r.owner_id, r.run_id, r.next_event_sequence - 1 AS event_sequence
), inserted AS (
    INSERT INTO dlightrag_run_events (
        owner_id, run_id, event_sequence, event_type, payload
    )
    SELECT owner_id, run_id, event_sequence, $6::text, $7::jsonb FROM bumped
    RETURNING event_sequence
)
SELECT count(*)::int FROM inserted
"""

_SUPERSEDE_WAITING_MUTATION = """
WITH updated AS (
    UPDATE dlightrag_runs
    SET status = 'failed', phase = NULL,
        error_kind = 'repair_superseded',
        error_message = 'Waiting mutation was superseded by an authorized Corpus Reset.',
        result_json = jsonb_build_object(
            'action', COALESCE(accepted_input_json->>'action', 'unknown'),
            'superseded_by_run_id', $3::uuid,
            'repair_reason', 'Superseded by an authorized full Corpus Reset.'
        ),
        prepared_input_json = NULL,
        superseded_by_run_id = $3::uuid,
        lease_owner = NULL, lease_expires_at = NULL,
        finished_at = NOW(),
        purge_after = NOW() + make_interval(secs => retention_seconds::double precision),
        updated_at = NOW(), next_event_sequence = next_event_sequence + 1
    WHERE owner_id = $1 AND run_id = $2
      AND run_kind = 'corpus_mutation' AND status = 'running'
      AND phase = 'waiting_for_repair'
      AND lease_owner IS NULL AND lease_expires_at IS NULL
    RETURNING owner_id, run_id, next_event_sequence - 1 AS event_sequence,
              result_json
), inserted AS (
    INSERT INTO dlightrag_run_events (
        owner_id, run_id, event_sequence, event_type, payload
    )
    SELECT owner_id, run_id, event_sequence, 'error',
           jsonb_build_object(
               'kind', 'repair_superseded',
               'message', 'Waiting mutation was superseded by an authorized Corpus Reset.',
               'result', result_json
           )
    FROM updated
    RETURNING 1
)
SELECT count(*)::int FROM inserted
"""

_REQUEST_CANCELLATION = """
WITH updated AS (
    UPDATE dlightrag_runs
    SET cancel_requested_at = COALESCE(cancel_requested_at, NOW()),
        updated_at = NOW()
    WHERE owner_id = $1 AND run_id = $2
      AND status = 'running'
      AND handoff_started_at IS NULL
    RETURNING 1
), notified AS (
    SELECT pg_notify('dlightrag_run_cancel', $3) FROM updated
)
SELECT count(*)::int FROM notified
"""

_REQUEUE_RUN = """
UPDATE dlightrag_runs
SET status = 'queued',
    lease_owner = NULL,
    lease_expires_at = NULL,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
  AND cancel_requested_at IS NULL
RETURNING 1
"""

_DEFER_RUN = """
UPDATE dlightrag_runs
SET status = 'queued', phase = 'deferred', checkpoint_json = $5::jsonb,
    next_attempt_at = $6, lease_owner = NULL, lease_expires_at = NULL,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND lease_owner = $3
  AND fencing_epoch = $4 AND status = 'running' AND lease_expires_at > NOW()
RETURNING 1
"""

_RESUME_REPAIR = """
UPDATE dlightrag_runs
SET status = 'queued', phase = 'repair_resumed', next_attempt_at = NOW(),
    checkpoint_json = jsonb_set(
        COALESCE(checkpoint_json, '{}'::jsonb),
        '{repair_resume_confirmed}',
        'true'::jsonb,
        TRUE
    ),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND status = 'running' AND phase = 'waiting_for_repair'
  AND lease_owner IS NULL AND lease_expires_at IS NULL
RETURNING 1
"""

_WAIT_FOR_REPAIR = """
UPDATE dlightrag_runs
SET phase = 'waiting_for_repair', checkpoint_json = $5::jsonb,
    next_attempt_at = NULL, lease_owner = NULL, lease_expires_at = NULL,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND lease_owner = $3
  AND fencing_epoch = $4 AND status = 'running' AND lease_expires_at > NOW()
RETURNING 1
"""

_SELECT_CANCEL_PENDING = """
SELECT owner_id, run_id
FROM dlightrag_runs
WHERE cancel_requested_at IS NOT NULL
  AND status IN ('queued', 'running')
  AND (lease_expires_at IS NULL OR lease_expires_at < NOW())
ORDER BY updated_at
LIMIT $1
FOR UPDATE SKIP LOCKED
"""

_INSERT_RESOURCE = """
INSERT INTO dlightrag_answer_resources (
    owner_id, run_id, resource_id, kind, safe_name, media_type, capabilities,
    ordinal, blob_digest, locator_digest, source_locator,
    session_id, intent_id, result_ordinal
)
VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12, $13, $14)
ON CONFLICT (owner_id, run_id, resource_id) DO NOTHING
"""

_INSERT_RUN_ARTIFACT = """
INSERT INTO dlightrag_answer_run_artifacts (
    owner_id, run_id, resource_id, reference_kind, ordinal, digest,
    filename, mime_type, transform_locator
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
ON CONFLICT (owner_id, run_id, resource_id) DO NOTHING
"""

_SELECT_RUN_ARTIFACTS = """
SELECT resource_id, reference_kind, ordinal, digest, filename, mime_type,
       transform_locator, created_at
FROM dlightrag_answer_run_artifacts
WHERE owner_id = $1 AND run_id = $2
ORDER BY reference_kind, ordinal
"""

_SELECT_RUN_FETCHED_RESOURCES = """
SELECT resource_id, ordinal, blob_digest, safe_name, media_type, source_locator, capabilities
FROM dlightrag_answer_resources
WHERE owner_id = $1 AND run_id = $2 AND kind = 'fetched_blob'
  AND capabilities->>'resource_kind' IN ('web', 'tool_attachment')
  AND ordinal IS NOT NULL AND source_locator IS NOT NULL
ORDER BY ordinal, resource_id
"""

_SELECT_ARTIFACT_ATTACHMENTS = """
SELECT relative_path, label, content_digest, size_bytes, presentation,
       session_id::text, intent_id::text
FROM dlightrag_answer_artifact_attachments
WHERE owner_id = $1 AND run_id = $2
ORDER BY attachment_order
"""

_SELECT_RUN_DIGESTS = """
SELECT owner_id, digest
FROM dlightrag_answer_run_artifacts
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
UNION
SELECT owner_id, blob_digest AS digest
FROM dlightrag_answer_resources
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
  AND blob_digest IS NOT NULL
"""

_DELETE_RUNS = """
WITH deleted AS (
    DELETE FROM dlightrag_runs
    WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
    RETURNING 1
)
SELECT count(*)::int FROM deleted
"""

_SELECT_RUN_AGENT_SESSIONS = """
SELECT owner_id, agent_session_id AS session_id
FROM dlightrag_answer_run_routing
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
UNION
SELECT owner_id, child_session_id AS session_id
FROM dlightrag_answer_child_sessions
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
"""

_LOCK_AGENT_SESSION_CANDIDATES = """
SELECT owner_id, session_id
FROM dlightrag_agent_sessions
WHERE (owner_id, session_id) IN (
    SELECT * FROM unnest($1::text[], $2::uuid[])
)
ORDER BY owner_id, session_id
FOR UPDATE
"""

_DELETE_UNREFERENCED_AGENT_SESSIONS = """
DELETE FROM dlightrag_agent_sessions AS sessions
WHERE (sessions.owner_id, sessions.session_id) IN (
    SELECT * FROM unnest($1::text[], $2::uuid[])
)
AND NOT EXISTS (
    SELECT 1 FROM dlightrag_answer_run_routing AS routing
    WHERE routing.owner_id = sessions.owner_id
      AND routing.agent_session_id = sessions.session_id
)
RETURNING 1
"""

# Retention order: references first, then blob chunks/metadata only when no
# run/resource reference remains. Resources cascade with their run; orphan
# reference rows are gone with the run row too.
_DELETE_UNREFERENCED_BLOBS = """
WITH referenced AS (
    SELECT DISTINCT owner_id, digest FROM dlightrag_answer_run_artifacts
    UNION
    SELECT DISTINCT owner_id, blob_digest AS digest FROM dlightrag_answer_resources
        WHERE blob_digest IS NOT NULL
), candidates AS (
    SELECT b.owner_id, b.digest
    FROM dlightrag_blobs AS b
    WHERE (b.owner_id, b.digest) IN (SELECT * FROM unnest($1::text[], $2::text[]))
      AND NOT EXISTS (
          SELECT 1 FROM referenced AS r
          WHERE r.owner_id = b.owner_id AND r.digest = b.digest
      )
    FOR UPDATE SKIP LOCKED
), deleted AS (
    DELETE FROM dlightrag_blobs AS b
    USING candidates AS c
    WHERE b.owner_id = c.owner_id AND b.digest = c.digest
    RETURNING 1
)
SELECT count(*)::int FROM deleted
"""

_SELECT_EXPIRED_RUNS = """
SELECT runs.owner_id, runs.run_id
FROM dlightrag_runs AS runs
WHERE runs.status IN ('succeeded', 'failed', 'cancelled')
  AND runs.purge_after <= NOW()
ORDER BY runs.purge_after
LIMIT $1
FOR UPDATE OF runs SKIP LOCKED
"""

_SELECT_TRIMMABLE_RUNS = """
SELECT owner_id, run_id
FROM dlightrag_runs
WHERE status IN ('succeeded', 'failed', 'cancelled')
  AND purge_after <= NOW()
  AND events_trimmed_at IS NULL
ORDER BY purge_after
LIMIT $1
FOR UPDATE SKIP LOCKED
"""

_DELETE_EVENTS_FOR_RUNS = """
DELETE FROM dlightrag_run_events
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
"""

_MARK_EVENTS_TRIMMED = """
UPDATE dlightrag_runs
SET events_trimmed_at = NOW(),
    updated_at = NOW()
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
"""


def _digest_pairs(rows: Sequence[Any]) -> tuple[list[str], list[str]]:
    """Split reference rows into parallel owner and digest lists."""
    owners: list[str] = []
    digests: list[str] = []
    for row in rows:
        owners.append(str(row["owner_id"]))
        digests.append(str(row["digest"]))
    return owners, digests


async def _delete_unreferenced_agent_sessions(conn: Any, rows: Sequence[Any]) -> int:
    candidates = {(str(row["owner_id"]), uuid.UUID(str(row["session_id"]))) for row in rows}
    if not candidates:
        return 0
    ordered = sorted(candidates, key=lambda item: (item[0], str(item[1])))
    owners = [owner for owner, _session_id in ordered]
    session_ids = [session_id for _owner, session_id in ordered]
    await conn.fetch(_LOCK_AGENT_SESSION_CANDIDATES, owners, session_ids)
    deleted = await conn.fetch(_DELETE_UNREFERENCED_AGENT_SESSIONS, owners, session_ids)
    return len(deleted)


async def _try_delete_unreferenced(
    conn: Any, owners: Sequence[str], digests: Sequence[str]
) -> int | None:
    """Delete one savepointed blob batch, or None when a concurrent reference wins."""
    try:
        async with conn.transaction():
            deleted = await conn.fetchval(_DELETE_UNREFERENCED_BLOBS, owners, digests)
    except asyncpg.RestrictViolationError:
        return None
    except asyncpg.PostgresError:
        raise
    return int(deleted or 0)


async def _delete_unreferenced(conn: Any, owners: Sequence[str], digests: Sequence[str]) -> int:
    """Delete blobs no run or resource still references, yielding to concurrent links.

    Each delete runs inside its own savepoint. A RESTRICT raised by a reference
    that beat the reference check must not abort the caller's transaction, or the
    run deletion it already performed would silently roll back and retention would
    never advance past a contended batch. One contended blob must not shield the
    rest either, so a failed batch is retried digest by digest.
    """
    if not owners:
        return 0
    deleted = await _try_delete_unreferenced(conn, owners, digests)
    if deleted is not None:
        return deleted
    if len(owners) == 1:
        return 0
    survivors = 0
    for owner, digest in zip(owners, digests, strict=True):
        survivors += await _try_delete_unreferenced(conn, [owner], [digest]) or 0
    return survivors


def _require_owner(owner_id: str) -> str:
    owner = str(owner_id).strip()
    if not owner:
        raise ValueError("owner_id cannot be empty")
    return owner


def _validate_envelope(
    envelope: PreparedRunEnvelope, run_id: str
) -> tuple[str, uuid.UUID, str, str]:
    submitter = _require_owner(envelope.submitted_by)
    owner = _require_owner(envelope.access_scope.scope_id)
    if envelope.access_scope.kind == "owner" and owner != submitter:
        raise ValueError("owner-scoped runs must be submitted by their owner")
    if not envelope.submission_key:
        raise ValueError("submission_key must be non-empty")
    if not envelope.request_fingerprint:
        raise ValueError("request_fingerprint must be non-empty")
    if envelope.retention_seconds < 1:
        raise ValueError("retention_seconds must be positive")
    run_uuid = parse_run_id(run_id)
    if run_uuid is None:
        raise ValueError("run_id must be a canonical UUID")
    require_prepared_input_bounds(envelope.payload)
    payload = json.dumps(dict(envelope.payload), ensure_ascii=False, sort_keys=True)
    accepted_input = json.dumps(dict(envelope.accepted_input), ensure_ascii=False, sort_keys=True)
    return owner, run_uuid, payload, accepted_input


def _json_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        loaded = json.loads(value)
        return dict(loaded) if isinstance(loaded, dict) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _child_roster_row(row: Any) -> dict[str, Any]:
    return {
        "child_session_id": str(row["child_session_id"]),
        "parent_session_id": str(row["parent_session_id"]),
        "parent_call_id": str(row["parent_call_id"]),
        "parent_intent_id": (
            str(row["parent_intent_id"]) if row["parent_intent_id"] is not None else None
        ),
        "status": str(row["status"]),
        "summary": row["summary"],
        "objective": row["objective"],
        "context": row["context_mode"],
        "model_role": row["model_role"],
        "tools": _json_value(row["tools_json"]),
        "usage": _json_value(row["usage_json"]),
        "depth": int(row["depth"]),
        "context_snapshot": _json_value(row["context_snapshot_json"]),
        "plan": _json_value(row["plan_json"]),
        "budget": _json_value(row["budget_json"]),
        "host_state": _json_value(row["host_state_json"]),
        "fencing_epoch": int(row["fencing_epoch"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _optional_int(row: Any, name: str) -> int | None:
    try:
        value = row[name]
    except KeyError, TypeError:
        return None
    return int(value) if value is not None else None


def run_record(row: Any) -> RunRecord:
    """Project one stored run row into the storage-neutral Runtime record."""
    prepared = row["prepared_input"]
    return RunRecord(
        run_id=str(row["run_id"]),
        run_kind=cast(RunKind, str(row["run_kind"])),
        lane=cast(RunLane, str(row["lane"])),
        submitted_by=str(row["submitted_by"]),
        access_scope=RunAccessScope(
            kind=cast(Literal["owner", "workspace"], str(row["access_scope_kind"])),
            scope_id=str(row["owner_id"]),
        ),
        submission_key=str(row["submission_key"]),
        request_fingerprint=str(row["request_fingerprint"]),
        prepared_input=_json_object(prepared) if prepared is not None else None,
        accepted_input=_json_object(row["accepted_input"]),
        status=row["status"],
        phase=row["phase"],
        stop_reason=row["stop_reason"],
        cancel_requested_at=row["cancel_requested_at"],
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
        fencing_epoch=int(row["fencing_epoch"]),
        durable_progress_version=int(row["durable_progress_version"]),
        last_reclaim_progress_version=int(row["last_reclaim_progress_version"]),
        reclaims_without_progress=int(row["reclaims_without_progress"]),
        next_event_sequence=int(row["next_event_sequence"]),
        events_trimmed_at=row["events_trimmed_at"],
        result=_json_object(row["result_json"]) if row["result_json"] is not None else None,
        error_kind=row["error_kind"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        purge_after=row["purge_after"],
        next_attempt_at=row["next_attempt_at"],
        checkpoint=(
            _json_object(row["checkpoint_json"]) if row["checkpoint_json"] is not None else None
        ),
        handoff_started_at=row["handoff_started_at"],
        superseded_by_run_id=(
            str(row["superseded_by_run_id"]) if row["superseded_by_run_id"] is not None else None
        ),
        agent_workspace_epoch=_optional_int(row, "agent_workspace_epoch"),
    )


def _event_record(row: Any) -> RunEvent:
    return RunEvent(
        sequence=int(row["event_sequence"]),
        event_type=row["event_type"],
        payload=_json_object(row["payload"]),
        created_at=row["created_at"],
    )


def _reference_record(row: Any) -> RunArtifactReference:
    return RunArtifactReference(
        resource_id=str(row["resource_id"]),
        reference_kind=row["reference_kind"],
        ordinal=int(row["ordinal"]),
        digest=str(row["digest"]),
        filename=str(row["filename"]),
        mime_type=str(row["mime_type"]),
        transform_locator=_json_object(row["transform_locator"]),
        created_at=row["created_at"],
    )


class PGRunStore(PostgresOperationRunner):
    """Generic durable lifecycle plus Answer-owned PostgreSQL projections."""

    def __init__(
        self,
        *,
        pool: ConnectionPool | None = None,
        retention_seconds: int = DEFAULT_RUN_RETENTION_SECONDS,
        query_max_nonterminal_runs: int = DEFAULT_QUERY_MAX_NONTERMINAL_RUNS,
        corpus_mutation_max_nonterminal_runs: int = DEFAULT_CORPUS_MUTATION_MAX_NONTERMINAL_RUNS,
        promotion_doc_threshold: int | None = None,
        promotion_chunk_threshold: int | None = None,
    ) -> None:
        super().__init__(pool=pool)
        self._retention_seconds = retention_seconds
        self._query_max_nonterminal_runs = max(1, int(query_max_nonterminal_runs))
        self._corpus_mutation_max_nonterminal_runs = max(
            1, int(corpus_mutation_max_nonterminal_runs)
        )
        self._promotion_doc_threshold = promotion_doc_threshold
        self._promotion_chunk_threshold = promotion_chunk_threshold
        self._initialized = False

    async def _run_read[T](self, operation: Callable[[Any], Awaitable[T]]) -> T:
        return await self._run(operation)

    async def _run_write[T](self, operation: Callable[[Any], Awaitable[T]]) -> T:
        return await self._run_once(operation)

    async def initialize(self, *, validate_only: bool = False) -> None:
        """Create the RunRuntime and Answer projection schema, or validate a reader."""
        if self._initialized:
            return

        async def _operation(conn: Any) -> None:
            if validate_only:
                await verify_migrations(
                    conn,
                    scope=RUN_MIGRATION_SCOPE,
                    migrations=RUN_MIGRATIONS,
                    tables=RUN_SCHEMA_TABLES,
                    schema_error=RunSchemaError,
                )
                return
            await apply_migrations(
                conn,
                scope=RUN_MIGRATION_SCOPE,
                migrations=RUN_MIGRATIONS,
                schema_error=RunSchemaError,
            )

        await self._run(_operation)
        self._initialized = True

    # -- acceptance ---------------------------------------------------
    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        run_kind: RunKind,
    ) -> RunCreation | None:
        """Replay a matching operation-scoped key before preparation happens."""
        owner = _require_owner(owner_id)

        async def _operation(conn: Any) -> RunCreation | None:
            row = await conn.fetchrow(_SELECT_RUN_BY_KEY, run_kind, owner, idempotency_key)
            if row is None:
                return None
            return _require_replay_match(
                owner=owner,
                key=idempotency_key,
                stored_fingerprint=str(row["request_fingerprint"]),
                fingerprint=idempotency_fingerprint,
                row=row,
            )

        return await self._run_read(_operation)

    async def create_run(
        self,
        *,
        envelope: PreparedRunEnvelope,
        run_id: str,
        resources: Sequence[Mapping[str, object]] = (),
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
    ) -> RunCreation:
        """Accept one purpose-built Answer envelope on the generic runtime."""
        return await self.accept_run(
            envelope=envelope,
            run_id=run_id,
            resources=resources,
            blobs=artifacts,
            references=references,
            routing=routing,
        )

    async def accept_run(
        self,
        *,
        envelope: PreparedRunEnvelope,
        run_id: str,
        resources: Sequence[Mapping[str, object]] = (),
        blobs: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
    ) -> RunCreation:
        """Atomically accept one operation input and its generic run row."""
        if envelope.run_kind in {"answer", "retrieval"} and envelope.lane != "query":
            raise ValueError("Answer and Retrieval runs execute on the query lane")
        if envelope.run_kind == "corpus_mutation" and envelope.lane != "corpus_mutation":
            raise ValueError("Corpus Mutation runs execute on the corpus_mutation lane")
        if envelope.run_kind == "corpus_mutation" and envelope.access_scope.kind != "workspace":
            raise ValueError("Corpus Mutation runs require workspace access scope")
        if envelope.supersedes_run_id is not None and envelope.run_kind != "corpus_mutation":
            raise ValueError("only Corpus Mutation runs may supersede a waiting mutation")
        superseded_uuid = (
            parse_run_id(envelope.supersedes_run_id)
            if envelope.supersedes_run_id is not None
            else None
        )
        if envelope.supersedes_run_id is not None and superseded_uuid is None:
            raise ValueError("supersedes_run_id is invalid")
        if envelope.run_kind != "answer" and (resources or blobs or references or routing):
            raise ValueError("non-Answer runs cannot carry Answer-owned projections")
        if any(reference.reference_kind == "fetched_resource" for reference in references):
            raise ValueError("fetched_resource references cannot be run creation inputs")
        owner, run_uuid, prepared_json, accepted_json = _validate_envelope(envelope, run_id)

        async def _operation(conn: Any) -> RunCreation:
            async with conn.transaction():
                replayed = await conn.fetchrow(
                    _SELECT_RUN_BY_KEY,
                    envelope.run_kind,
                    envelope.submitted_by,
                    envelope.submission_key,
                )
                if replayed is not None:
                    return _require_replay_match(
                        owner=owner,
                        key=envelope.submission_key,
                        stored_fingerprint=str(replayed["request_fingerprint"]),
                        fingerprint=envelope.request_fingerprint,
                        row=replayed,
                    )
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    f"dlightrag:run-accept:{envelope.lane}",
                )
                replayed = await conn.fetchrow(
                    _SELECT_RUN_BY_KEY,
                    envelope.run_kind,
                    envelope.submitted_by,
                    envelope.submission_key,
                )
                if replayed is not None:
                    return _require_replay_match(
                        owner=owner,
                        key=envelope.submission_key,
                        stored_fingerprint=str(replayed["request_fingerprint"]),
                        fingerprint=envelope.request_fingerprint,
                        row=replayed,
                    )
                if superseded_uuid is not None:
                    superseded = await conn.fetchval(
                        _SUPERSEDE_WAITING_MUTATION,
                        owner,
                        superseded_uuid,
                        run_uuid,
                    )
                    if int(superseded or 0) != 1:
                        raise ValueError(
                            "supersedes_run_id is not this Workspace's waiting mutation"
                        )
                nonterminal = int(await conn.fetchval(_COUNT_NONTERMINAL_LANE, envelope.lane) or 0)
                max_nonterminal = (
                    self._query_max_nonterminal_runs
                    if envelope.lane == "query"
                    else self._corpus_mutation_max_nonterminal_runs
                )
                if nonterminal >= max_nonterminal:
                    raise RunAdmissionLimitExceededError(
                        "Deployment-wide nonterminal admission limit reached"
                    )
                await self._write_blobs(conn, owner, blobs)
                row = await conn.fetchrow(
                    _INSERT_RUN,
                    owner,
                    run_uuid,
                    envelope.run_kind,
                    envelope.lane,
                    envelope.submitted_by,
                    envelope.access_scope.kind,
                    envelope.submission_key,
                    prepared_json,
                    accepted_json,
                    envelope.request_fingerprint,
                    envelope.retention_seconds,
                )
                if row is None:
                    replayed = await conn.fetchrow(
                        _SELECT_RUN_BY_KEY,
                        envelope.run_kind,
                        envelope.submitted_by,
                        envelope.submission_key,
                    )
                    if replayed is None:
                        raise RuntimeError("run insert reported a vanished conflict")
                    return _require_replay_match(
                        owner=owner,
                        key=envelope.submission_key,
                        stored_fingerprint=str(replayed["request_fingerprint"]),
                        fingerprint=envelope.request_fingerprint,
                        row=replayed,
                    )
                for resource in resources:
                    await conn.execute(
                        _INSERT_RESOURCE,
                        owner,
                        run_uuid,
                        str(resource["resource_id"]),
                        "accepted_blob",
                        str(resource["safe_name"]),
                        str(resource["media_type"]),
                        json.dumps(resource.get("capabilities") or {}, ensure_ascii=False),
                        int(str(resource["ordinal"])),
                        str(resource["blob_digest"]),
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                for reference in references:
                    await conn.execute(
                        _INSERT_RUN_ARTIFACT,
                        owner,
                        run_uuid,
                        reference.resource_id,
                        reference.reference_kind,
                        reference.ordinal,
                        reference.digest,
                        reference.filename,
                        reference.mime_type,
                        json.dumps(dict(reference.transform_locator), ensure_ascii=False),
                    )
                if envelope.run_kind == "answer":
                    await self._insert_routing(
                        conn, owner, run_uuid, routing, prepared_input=envelope.payload
                    )
                return RunCreation(run=run_record(row), replayed=False)

        return await self._run_write(_operation)

    async def record_corpus_window(
        self,
        *,
        run_id: str,
        workspace: str,
        window_number: int,
        docs: int,
        chunks: int,
    ) -> bool:
        """Idempotently account one successful mutation window for promotion."""
        run_uuid = parse_run_id(run_id)
        if run_uuid is None or window_number < 1 or docs < 0 or chunks < 0:
            raise ValueError("invalid corpus mutation window")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                return bool(
                    await conn.fetchval(
                        _RECORD_CORPUS_WINDOW,
                        run_uuid,
                        window_number,
                        _require_owner(workspace),
                        docs,
                        chunks,
                        self._promotion_doc_threshold,
                        self._promotion_chunk_threshold,
                    )
                )

        return await self._run_write(_operation)

    async def _write_publications(
        self, conn: Any, owner: str, run_uuid: uuid.UUID, publications: Sequence[Any]
    ) -> None:
        planned = tuple((item, PendingArtifact(content=item.content)) for item in publications)
        await self._write_blobs(conn, owner, tuple(blob for _item, blob in planned))
        for index, (item, blob) in enumerate(planned):
            await conn.execute(
                _INSERT_RUN_ARTIFACT,
                owner,
                run_uuid,
                item.resource_id,
                item.reference_kind,
                index,
                blob.digest,
                item.filename,
                item.mime_type,
                "{}",
            )

    async def _write_blobs(self, conn: Any, owner: str, blobs: Sequence[PendingArtifact]) -> None:
        """Acquire new blob identities in one canonical order per transaction."""
        for blob in sorted(blobs, key=lambda item: item.digest):
            await self._write_blob(conn, owner, blob)

    async def _write_blob(self, conn: Any, owner: str, blob: PendingArtifact) -> None:
        try:
            await write_blob_content(
                conn,
                owner_id=owner,
                digest=blob.digest,
                content=blob.content,
            )
        except BlobSizeConflict as exc:
            raise ValueError("blob digest collision with a different byte size") from exc

    async def create_run_in(
        self,
        conn: Any,
        *,
        envelope: PreparedRunEnvelope,
        run_id: str,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
    ) -> RunCreation:
        """Create or replay one run inside a transaction the caller already owns.

        This is the composition seam another durable table uses to link its own
        row to the accepted run atomically. It performs no transaction control
        of its own, so the caller's commit is what makes the run and its link
        durable together. ``request`` is the bounded accepted execution input.
        """
        if envelope.run_kind != "answer" or envelope.lane != "query":
            raise ValueError("Web Answer acceptance requires answer kind on the query lane")
        if envelope.access_scope.kind != "owner":
            raise ValueError("Web Answer runs require owner access scope")
        if envelope.supersedes_run_id is not None:
            raise ValueError("Web Answer runs cannot supersede another run")
        owner, run_uuid, payload, envelope_json = _validate_envelope(envelope, run_id)
        if any(reference.reference_kind == "fetched_resource" for reference in references):
            # A fetched resource is worker-fenced run state, never accepted input.
            raise ValueError("fetched_resource references cannot be run creation inputs")
        existing = await conn.fetchrow(
            _SELECT_RUN_BY_KEY,
            envelope.run_kind,
            envelope.submitted_by,
            envelope.submission_key,
        )
        if existing is not None:
            return _require_replay_match(
                owner=owner,
                key=envelope.submission_key,
                stored_fingerprint=str(existing["request_fingerprint"]),
                fingerprint=envelope.request_fingerprint,
                row=existing,
            )
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtext($1))",
            f"dlightrag:run-accept:{envelope.lane}",
        )
        existing = await conn.fetchrow(
            _SELECT_RUN_BY_KEY,
            envelope.run_kind,
            envelope.submitted_by,
            envelope.submission_key,
        )
        if existing is not None:
            return _require_replay_match(
                owner=owner,
                key=envelope.submission_key,
                stored_fingerprint=str(existing["request_fingerprint"]),
                fingerprint=envelope.request_fingerprint,
                row=existing,
            )
        nonterminal = int(await conn.fetchval(_COUNT_NONTERMINAL_LANE, envelope.lane) or 0)
        if nonterminal >= self._query_max_nonterminal_runs:
            raise RunAdmissionLimitExceededError(
                "Deployment-wide nonterminal admission limit reached"
            )
        await self._write_blobs(conn, owner, artifacts)
        row = await conn.fetchrow(
            _INSERT_RUN,
            owner,
            run_uuid,
            envelope.run_kind,
            envelope.lane,
            envelope.submitted_by,
            envelope.access_scope.kind,
            envelope.submission_key,
            payload,
            envelope_json,
            envelope.request_fingerprint,
            envelope.retention_seconds,
        )
        if row is None:
            existing = await conn.fetchrow(
                _SELECT_RUN_BY_KEY,
                envelope.run_kind,
                envelope.submitted_by,
                envelope.submission_key,
            )
            if existing is None:
                raise RuntimeError("answer run insert reported a vanished conflict")
            if str(existing["request_fingerprint"]) != envelope.request_fingerprint:
                raise IdempotencyKeyConflict(
                    "idempotency key was reused with different request input"
                )
            return RunCreation(run=run_record(existing), replayed=True)
        for reference in references:
            await conn.execute(
                _INSERT_RUN_ARTIFACT,
                owner,
                run_uuid,
                reference.resource_id,
                reference.reference_kind,
                reference.ordinal,
                reference.digest,
                reference.filename,
                reference.mime_type,
                json.dumps(dict(reference.transform_locator), ensure_ascii=False),
            )
        await self._insert_routing(conn, owner, run_uuid, routing, prepared_input=envelope.payload)
        return RunCreation(run=run_record(row), replayed=False)

    async def _insert_routing(
        self,
        conn: Any,
        owner: str,
        run_uuid: uuid.UUID,
        routing: RoutingAcceptance | None,
        *,
        prepared_input: Mapping[str, Any],
    ) -> None:
        record = routing or RoutingAcceptance.fallback(prepared_input)
        session_uuid = uuid.UUID(record.agent_session_id)
        await conn.fetchval(_LOCK_AGENT_SESSION_IF_PRESENT, owner, session_uuid)
        await conn.execute(
            _INSERT_ROUTING,
            owner,
            run_uuid,
            record.requested_mode,
            list(record.valid_modes),
            record.resolved_mode,
            json.dumps(dict(record.model_fingerprints), ensure_ascii=False),
            record.context_policy_revision,
            uuid.UUID(record.agent_session_id),
            record.agent_lane_id,
            record.source_lane_id,
        )

    async def load_routing(self, *, owner_id: str, run_id: str) -> RoutingRecord | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None

        async def _operation(conn: Any) -> RoutingRecord | None:
            row = await conn.fetchrow(_SELECT_ROUTING, owner, run_uuid)
            if row is None:
                return None
            valid = tuple(str(item) for item in (row["valid_modes"] or ()))
            return RoutingRecord(
                requested_mode=str(row["requested_mode"]),
                valid_modes=valid,
                resolved_mode=row["resolved_mode"],
                agent_session_id=str(row["agent_session_id"]),
                agent_lane_id=str(row["agent_lane_id"]),
                source_lane_id=(str(row["source_lane_id"]) if row["source_lane_id"] else None),
            )

        return await self._run_read(_operation)

    async def resolve(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        resolved_mode: str,
    ) -> str | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            raise ValueError("run_id must be a canonical UUID")

        async def _operation(conn: Any) -> str | None:
            value = await conn.fetchval(
                _RESOLVE_ROUTING,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                resolved_mode,
            )
            return str(value) if value is not None else None

        return await self._run_write(_operation)

    async def upsert_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        parent_session_id: str,
        parent_call_id: str,
        worker_id: str,
        fencing_epoch: int,
        parent_intent_id: str | None = None,
        objective: str | None = None,
        context_mode: str | None = None,
        model_role: str | None = None,
        tools: Sequence[str] | None = None,
        depth: int = 1,
        context_snapshot: Mapping[str, Any] | None = None,
        plan: Mapping[str, Any] | None = None,
        budget: Mapping[str, Any] | None = None,
        host_state: Mapping[str, Any] | None = None,
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        parent_uuid = parse_run_id(parent_session_id)
        intent_uuid = parse_run_id(parent_intent_id) if parent_intent_id is not None else None
        if run_uuid is None or child_uuid is None or parent_uuid is None:
            raise ValueError("child session ids must be canonical UUIDs")
        if parent_intent_id is not None and intent_uuid is None:
            raise ValueError("parent intent id must be a canonical UUID")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return False
                await conn.execute(
                    _UPSERT_CHILD_SESSION,
                    owner,
                    run_uuid,
                    child_uuid,
                    parent_uuid,
                    parent_call_id,
                    intent_uuid,
                    objective,
                    context_mode,
                    model_role,
                    json.dumps(list(tools)) if tools is not None else None,
                    depth,
                    json.dumps(dict(context_snapshot or {}), ensure_ascii=False),
                    json.dumps(dict(plan), ensure_ascii=False) if plan is not None else None,
                    json.dumps(dict(budget), ensure_ascii=False) if budget is not None else None,
                    json.dumps(dict(host_state or {}), ensure_ascii=False),
                )
                return True

        return await self._run_write(_operation)

    async def claim_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        worker_id: str,
        fencing_epoch: int,
    ) -> int | None:
        """Acquire the Child's independent lease under the live parent run claim."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        if run_uuid is None or child_uuid is None:
            raise ValueError("child session ids must be canonical UUIDs")

        async def _operation(conn: Any) -> int | None:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return None
                value = await conn.fetchval(
                    _CLAIM_CHILD_SESSION,
                    owner,
                    run_uuid,
                    child_uuid,
                    worker_id,
                    RUN_LEASE_SECONDS,
                )
                return int(value) if value is not None else None

        return await self._run_write(_operation)

    async def heartbeat_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        worker_id: str,
        fencing_epoch: int,
        child_fencing_epoch: int,
    ) -> bool:
        """Renew one unexpired Child lease under its live parent run claim."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        if run_uuid is None or child_uuid is None:
            raise ValueError("child session ids must be canonical UUIDs")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return False
                renewed = await conn.fetchval(
                    _RENEW_CHILD_SESSION_LEASE,
                    owner,
                    run_uuid,
                    child_uuid,
                    worker_id,
                    child_fencing_epoch,
                    RUN_LEASE_SECONDS,
                )
                return renewed is not None

        return await self._run_write(_operation)

    async def load_child_session(
        self, *, owner_id: str, run_id: str, child_session_id: str
    ) -> dict[str, Any] | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        if run_uuid is None or child_uuid is None:
            return None

        async def _operation(conn: Any) -> dict[str, Any] | None:
            row = await conn.fetchrow(_SELECT_CHILD_SESSION, owner, run_uuid, child_uuid)
            if row is None:
                return None
            return {
                "child_session_id": str(row["child_session_id"]),
                "status": str(row["status"]),
                "summary": row["summary"],
                "parent_intent_id": (
                    str(row["parent_intent_id"]) if row["parent_intent_id"] is not None else None
                ),
                "objective": row["objective"],
                "context": row["context_mode"],
                "model_role": row["model_role"],
                "tools": _json_value(row["tools_json"]),
                "usage": _json_value(row["usage_json"]),
                "depth": int(row["depth"]),
                "context_snapshot": _json_value(row["context_snapshot_json"]),
                "plan": _json_value(row["plan_json"]),
                "budget": _json_value(row["budget_json"]),
                "host_state": _json_value(row["host_state_json"]),
                "fencing_epoch": int(row["fencing_epoch"]),
            }

        return await self._run_read(_operation)

    async def list_child_sessions(
        self, *, owner_id: str, run_id: str
    ) -> tuple[dict[str, Any], ...]:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[dict[str, Any], ...]:
            rows = await conn.fetch(_SELECT_CHILD_SESSIONS, owner, run_uuid)
            return tuple(_child_roster_row(row) for row in rows)

        return await self._run_read(_operation)

    async def list_child_sessions_page(
        self,
        *,
        owner_id: str,
        run_id: str,
        page: ChildRosterPageRequest,
    ) -> ChildRosterRowPage:
        """Return one physical limit+1 newest-first keyset roster page."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        validated = ChildRosterPageRequest(limit=page.limit, cursor=page.cursor)
        cursor = validated.cursor
        if run_uuid is None:
            return ChildRosterRowPage(children=(), has_more=False, fetched_rows=0)
        if cursor is not None and cursor.run_id != run_uuid:
            raise ValueError("child-roster cursor belongs to another run")
        fetch_limit = validated.limit + 1

        async def _operation(conn: Any) -> ChildRosterRowPage:
            if cursor is None:
                rows = await conn.fetch(
                    _SELECT_CHILD_SESSIONS_FIRST_PAGE,
                    owner,
                    run_uuid,
                    fetch_limit,
                )
            else:
                rows = await conn.fetch(
                    _SELECT_CHILD_SESSIONS_AFTER,
                    owner,
                    run_uuid,
                    cursor.created_at,
                    cursor.child_session_id,
                    fetch_limit,
                )
            fetched_rows = len(rows)
            return ChildRosterRowPage(
                children=tuple(_child_roster_row(row) for row in rows[: validated.limit]),
                has_more=fetched_rows > validated.limit,
                fetched_rows=fetched_rows,
            )

        return await self._run_read(_operation)

    async def load_agent_transcript(
        self,
        *,
        owner_id: str,
        run_id: str,
        session_id: str,
        limit: int,
    ) -> tuple[dict[str, Any], ...]:
        """Project one canonical parent Session ancestry without exposing storage rows."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        session_uuid = parse_run_id(session_id)
        if run_uuid is None or session_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[dict[str, Any], ...]:
            rows = await conn.fetch(
                _SELECT_AGENT_TRANSCRIPT,
                owner,
                run_uuid,
                session_uuid,
                max(1, min(int(limit), 100)),
            )
            messages: list[dict[str, Any]] = []
            for row in reversed(rows):
                payload = _json_object(row["payload_json"])
                entry_type = str(row["entry_type"])
                if entry_type in {"user_message", "control_message"}:
                    messages.append({"role": "user", "content": payload.get("content")})
                elif entry_type == "assistant_message":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": payload.get("content") or "",
                            "tool_calls": list(payload.get("tool_calls") or ()),
                        }
                    )
                elif entry_type == "tool_result":
                    outcome = str(payload.get("outcome") or "failed")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(payload.get("call_id") or ""),
                            "name": str(payload.get("tool_name") or ""),
                            **tool_content_message_fields(
                                decode_tool_content(payload.get("content"))
                            ),
                            "is_error": outcome != "succeeded",
                        }
                    )
            return tuple(messages)

        return await self._run_read(_operation)

    async def enqueue_agent_control(
        self,
        *,
        owner_id: str,
        run_id: str,
        kind: str,
        content: str,
    ) -> dict[str, Any] | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        text = content.strip()
        if run_uuid is None or kind not in {"steer", "follow_up"} or not text:
            return None

        async def _operation(conn: Any) -> dict[str, Any] | None:
            async with conn.transaction():
                run = await conn.fetchrow(_LOCK_CONTROL_RUN, owner, run_uuid)
                if run is None or str(run["status"]) not in {"queued", "running"}:
                    return None
                resolved = str(run["resolved_mode"] or "")
                requested = str(run["requested_mode"] or "")
                if resolved != "research" and not (not resolved and requested == "research"):
                    return None
                sequence = int(await conn.fetchval(_NEXT_CONTROL_SEQUENCE, owner, run_uuid) or 1)
                await conn.execute(_INSERT_CONTROL, owner, run_uuid, sequence, kind, text)
                return {
                    "run_id": run_id,
                    "control_sequence": sequence,
                    "kind": kind,
                    "content": text,
                }

        return await self._run_write(_operation)

    async def load_pending_agent_controls(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
    ) -> tuple[dict[str, Any], ...] | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None

        async def _operation(conn: Any) -> tuple[dict[str, Any], ...] | None:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return None
                rows = await conn.fetch(_SELECT_PENDING_CONTROLS, owner, run_uuid)
                return tuple(
                    {
                        "control_sequence": int(row["control_sequence"]),
                        "kind": str(row["kind"]),
                        "content": str(row["content"]),
                        "created_at": row["created_at"],
                    }
                    for row in rows
                )

        return await self._run_write(_operation)

    async def acknowledge_agent_controls(
        self,
        *,
        owner_id: str,
        run_id: str,
        control_sequences: Sequence[int],
        worker_id: str,
        fencing_epoch: int,
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return False

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return False
                if control_sequences:
                    await conn.execute(
                        _CONSUME_CONTROLS,
                        owner,
                        run_uuid,
                        [int(value) for value in control_sequences],
                    )
                return True

        return await self._run_write(_operation)

    async def finish_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        status: str,
        summary: str,
        outcome: Mapping[str, Any],
        worker_id: str,
        fencing_epoch: int,
        usage: Mapping[str, int] | None = None,
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        if run_uuid is None or child_uuid is None:
            raise ValueError("child session ids must be canonical UUIDs")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return False
                tag = await conn.execute(
                    _FINISH_CHILD_SESSION,
                    owner,
                    run_uuid,
                    child_uuid,
                    status,
                    summary,
                    json.dumps(dict(usage)) if usage is not None else None,
                    json.dumps(dict(outcome), ensure_ascii=False, sort_keys=True),
                )
                return not str(tag).endswith(" 0")

        return await self._run_write(_operation)

    async def bind_child_parent_intent(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        parent_intent_id: str,
        worker_id: str,
        fencing_epoch: int,
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        intent_uuid = parse_run_id(parent_intent_id)
        if run_uuid is None or child_uuid is None or intent_uuid is None:
            raise ValueError("child session ids must be canonical UUIDs")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return False
                tag = await conn.execute(
                    _BIND_CHILD_PARENT_INTENT, owner, run_uuid, child_uuid, intent_uuid
                )
                return not str(tag).endswith(" 0")

        return await self._run_write(_operation)

    async def delete_runs_in(
        self, conn: Any, *, owner_id: str, run_ids: Sequence[str]
    ) -> RunDeletion:
        """Delete owned runs and orphaned blobs inside a caller-owned transaction.

        The composition seam a linked table uses so deleting its own rows and the
        runs they referenced is one atomic act; a lease-fenced worker can no
        longer append to a run whose row disappeared.
        """
        owner = _require_owner(owner_id)
        run_uuids = [parsed for parsed in (parse_run_id(value) for value in run_ids) if parsed]
        if not run_uuids:
            return RunDeletion(runs=0, artifacts=0)
        owners = [owner] * len(run_uuids)
        pairs = await conn.fetch(_SELECT_RUN_DIGESTS, owners, run_uuids)
        session_rows = await conn.fetch(_SELECT_RUN_AGENT_SESSIONS, owners, run_uuids)
        deleted = await conn.fetchval(_DELETE_RUNS, owners, run_uuids)
        await _delete_unreferenced_agent_sessions(conn, session_rows)
        artifacts = await _delete_unreferenced(conn, *_digest_pairs(pairs))
        return RunDeletion(runs=int(deleted or 0), artifacts=artifacts)

    async def iter_active_run_requirements(
        self,
        *,
        page_size: int = _BATCH_LIMIT,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Stream active-run compatibility facts in bounded keyset pages."""
        cap = max(1, min(int(page_size), _BATCH_LIMIT))

        async def _frontier(conn: Any) -> Any:
            return await conn.fetchrow(
                _ACTIVE_REQUIREMENTS_FRONTIER,
                MAX_RECLAIMS_WITHOUT_PROGRESS,
            )

        upper = await self._run_read(_frontier)
        if upper is None:
            return
        upper_position = (upper["created_at"], upper["run_id"])
        position: tuple[Any, Any] | None = None
        while True:

            async def _page(
                conn: Any,
                after: tuple[Any, Any] | None = position,
            ) -> list[Any]:
                if after is None:
                    return await conn.fetch(
                        _ACTIVE_REQUIREMENTS_FIRST_PAGE,
                        MAX_RECLAIMS_WITHOUT_PROGRESS,
                        *upper_position,
                        cap,
                    )
                return await conn.fetch(
                    _ACTIVE_REQUIREMENTS_AFTER,
                    MAX_RECLAIMS_WITHOUT_PROGRESS,
                    *upper_position,
                    *after,
                    cap,
                )

            rows = await self._run_read(_page)
            if not rows:
                return
            for row in rows:
                yield {
                    "run_kind": str(row["run_kind"]),
                    "prepared_input": _json_value(row["prepared_input_json"]),
                }
            if len(rows) < cap:
                return
            next_position = (rows[-1]["created_at"], rows[-1]["run_id"])
            if position is not None and next_position <= position:
                raise RuntimeError("active-run requirement cursor did not advance")
            position = next_position

    async def delete_runs(self, *, owner_id: str, run_ids: Sequence[str]) -> RunDeletion:
        """Delete owned runs and orphaned blobs in one dedicated transaction."""

        async def _operation(conn: Any) -> RunDeletion:
            async with conn.transaction():
                return await self.delete_runs_in(conn, owner_id=owner_id, run_ids=run_ids)

        return await self._run_write(_operation)

    # -- reads --------------------------------------------------------
    async def get_run(self, *, owner_id: str, run_id: str) -> RunRecord | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None

        async def _operation(conn: Any) -> RunRecord | None:
            row = await conn.fetchrow(_SELECT_RUN, owner, run_uuid)
            return run_record(row) if row is not None else None

        return await self._run_read(_operation)

    async def get_run_global(self, *, run_id: str) -> RunRecord | None:
        """Read by globally unique id for a later fail-closed authorization check."""
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None

        async def _operation(conn: Any) -> RunRecord | None:
            row = await conn.fetchrow(_SELECT_RUN_GLOBAL, run_uuid)
            return run_record(row) if row is not None else None

        return await self._run_read(_operation)

    async def list_runs(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> tuple[RunRecord, ...]:
        owner = _require_owner(owner_id)
        cap = max(1, min(int(limit), 100))
        after = parse_run_id(after_run_id) if after_run_id else None

        async def _operation(conn: Any) -> tuple[RunRecord, ...]:
            if after is None:
                rows = await conn.fetch(_LIST_RUNS, owner, cap)
            else:
                rows = await conn.fetch(_LIST_RUNS_AFTER, owner, after, cap)
            return tuple(run_record(row) for row in rows)

        return await self._run_read(_operation)

    async def list_run_artifacts(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunArtifactReference, ...]:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[RunArtifactReference, ...]:
            rows = await conn.fetch(_SELECT_RUN_ARTIFACTS, owner, run_uuid)
            return tuple(_reference_record(row) for row in rows)

        return await self._run_read(_operation)

    async def list_fetched_resources(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunFetchedResource, ...]:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[RunFetchedResource, ...]:
            rows = await conn.fetch(_SELECT_RUN_FETCHED_RESOURCES, owner, run_uuid)
            return tuple(
                RunFetchedResource(
                    resource_id=str(row["resource_id"]),
                    ordinal=int(row["ordinal"]),
                    digest=str(row["blob_digest"]),
                    filename=str(row["safe_name"]),
                    mime_type=str(row["media_type"]),
                    source_locator=bytes(row["source_locator"]),
                    capabilities=_json_object(row["capabilities"]),
                )
                for row in rows
            )

        return await self._run_read(_operation)

    async def list_artifact_attachments(
        self, *, owner_id: str, run_id: str
    ) -> tuple[ArtifactAttachmentUpdate, ...]:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[ArtifactAttachmentUpdate, ...]:
            rows = await conn.fetch(_SELECT_ARTIFACT_ATTACHMENTS, owner, run_uuid)
            return tuple(
                ArtifactAttachmentUpdate(
                    relative_path=str(row["relative_path"]),
                    label=str(row["label"]),
                    content_digest=str(row["content_digest"]),
                    size_bytes=int(row["size_bytes"]),
                    presentation=str(row["presentation"]),
                    session_id=str(row["session_id"]),
                    intent_id=str(row["intent_id"]),
                )
                for row in rows
            )

        return await self._run_read(_operation)

    async def read_event_page(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> tuple[RunEvent, ...]:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[RunEvent, ...]:
            rows = await conn.fetch(
                _SELECT_EVENTS,
                owner,
                run_uuid,
                max(0, int(after_sequence)),
                _EVENT_PAGE_LIMIT,
            )
            return tuple(_event_record(row) for row in rows)

        return await self._run_read(_operation)

    # -- cancellation -------------------------------------------------
    async def iter_cancel_pending(
        self,
        *,
        worker_id: str,
        page_size: int = _BATCH_LIMIT,
    ) -> AsyncIterator[tuple[str, str]]:
        """Stream this worker's live cancel-pending leases in bounded pages."""
        cap = max(1, min(int(page_size), _BATCH_LIMIT))

        async def _frontier(conn: Any) -> Any:
            return await conn.fetchrow(_CANCEL_PENDING_FRONTIER, worker_id)

        upper = await self._run_read(_frontier)
        if upper is None:
            return
        upper_position = (upper["created_at"], upper["run_id"])
        position: tuple[Any, Any] | None = None
        while True:

            async def _page(
                conn: Any,
                after: tuple[Any, Any] | None = position,
            ) -> list[Any]:
                if after is None:
                    return await conn.fetch(
                        _CANCEL_PENDING_FIRST_PAGE,
                        worker_id,
                        *upper_position,
                        cap,
                    )
                return await conn.fetch(
                    _CANCEL_PENDING_AFTER,
                    worker_id,
                    *upper_position,
                    *after,
                    cap,
                )

            rows = await self._run_read(_page)
            if not rows:
                return
            for row in rows:
                yield str(row["owner_id"]), str(row["run_id"])
            if len(rows) < cap:
                return
            next_position = (rows[-1]["created_at"], rows[-1]["run_id"])
            if position is not None and next_position <= position:
                raise RuntimeError("cancel-pending cursor did not advance")
            position = next_position

    def build_cancellation_listener(
        self,
        *,
        worker_id: str,
        on_cancel: Callable[[str, str], Awaitable[None]],
    ) -> RunCancellationListener:
        """Build the dedicated reconnecting LISTEN listener for this store."""

        async def _open_connection() -> Any:
            if self._operation_pool is not None:
                return await self._operation_pool.acquire().__aenter__()
            pool = await pg_pool.get()
            return await pool.acquire().__aenter__()

        def _rescan() -> AsyncIterator[tuple[str, str]]:
            return self.iter_cancel_pending(worker_id=worker_id)

        return RunCancellationListener(
            open_connection=_open_connection,
            rescan=_rescan,
            on_cancel=on_cancel,
        )

    async def request_cancellation(self, *, owner_id: str, run_id: str) -> CancellationOutcome:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return CancellationOutcome(outcome="unknown", run=None)

        async def _operation(conn: Any) -> CancellationOutcome:
            async with conn.transaction():
                row = await conn.fetchrow(_SELECT_RUN_FOR_UPDATE, owner, run_uuid)
                if row is None:
                    return CancellationOutcome(outcome="unknown", run=None)
                run = run_record(row)
                if run.terminal:
                    return CancellationOutcome(outcome="already_terminal", run=run)
                if run.handoff_started_at is not None:
                    return CancellationOutcome(outcome="rejected", run=run)
                if run.status == "queued":
                    finalized = await self._finalize_cancelled_queued(conn, owner, run_uuid)
                    if not finalized:
                        raise RuntimeError("locked queued run could not be cancelled")
                    return CancellationOutcome(
                        outcome="cancelled",
                        run=run_record(await conn.fetchrow(_SELECT_RUN, owner, run_uuid)),
                    )
                updated = await conn.fetchval(
                    _REQUEST_CANCELLATION,
                    owner,
                    run_uuid,
                    cancellation_notify_key(owner_id=owner, run_id=str(run_uuid)),
                )
                if int(updated or 0) != 1:
                    current = run_record(await conn.fetchrow(_SELECT_RUN, owner, run_uuid))
                    return CancellationOutcome(outcome="rejected", run=current)
                return CancellationOutcome(
                    outcome="pending",
                    run=run_record(await conn.fetchrow(_SELECT_RUN, owner, run_uuid)),
                )

        return await self._run_write(_operation)

    async def _finalize_cancelled_queued(self, conn: Any, owner: str, run_uuid: uuid.UUID) -> bool:
        finalized = await conn.fetchval(
            _FINALIZE_UNLEASED,
            [owner],
            [run_uuid],
            "cancelled",
            None,
            None,
            "done",
            json.dumps({"status": "cancelled"}),
        )
        return int(finalized or 0) == 1

    # -- claim and writes ---------------------------------------------
    async def claim_next(
        self,
        *,
        worker_id: str,
        run_kinds: Sequence[RunKind] = ("answer",),
        lanes: Sequence[RunLane] = ("query",),
    ) -> ClaimedRun | None:
        """Claim the oldest eligible registered kind without double-claiming."""
        worker = str(worker_id).strip()
        if not worker:
            raise ValueError("worker_id cannot be empty")
        requested_lanes = tuple(dict.fromkeys(lanes))
        if len(requested_lanes) != 1:
            raise ValueError("claim_next requires exactly one execution lane")
        lane = requested_lanes[0]

        async def _operation(conn: Any) -> ClaimedRun | None:
            while True:
                candidate = await conn.fetchrow(
                    _SELECT_CLAIM_CANDIDATE,
                    MAX_RECLAIMS_WITHOUT_PROGRESS,
                    list(run_kinds),
                    [lane],
                )
                if candidate is None:
                    return None
                locked = await conn.fetchrow(
                    _SELECT_RUN, candidate["owner_id"], candidate["run_id"]
                )
                if locked is None:
                    continue
                if locked["status"] == "queued":
                    decision = ReclaimDecision(
                        abandoned=False,
                        reclaims_without_progress=int(locked["reclaims_without_progress"]),
                        last_reclaim_progress_version=int(locked["durable_progress_version"]),
                    )
                else:
                    decision = advance_reclaim(
                        ReclaimState(
                            durable_progress_version=int(locked["durable_progress_version"]),
                            last_reclaim_progress_version=int(
                                locked["last_reclaim_progress_version"]
                            ),
                            reclaims_without_progress=int(locked["reclaims_without_progress"]),
                        )
                    )
                if decision.abandoned:
                    await conn.execute(
                        _FINALIZE_UNLEASED,
                        [candidate["owner_id"]],
                        [candidate["run_id"]],
                        "failed",
                        RUN_ABANDONED_ERROR_KIND,
                        _ABANDONED_ERROR_MESSAGE,
                        "error",
                        json.dumps(
                            {
                                "kind": RUN_ABANDONED_ERROR_KIND,
                                "message": _ABANDONED_ERROR_MESSAGE,
                            }
                        ),
                    )
                    continue
                row = await conn.fetchrow(
                    _CLAIM_RUN,
                    candidate["owner_id"],
                    candidate["run_id"],
                    worker,
                    RUN_LEASE_SECONDS,
                    decision.reclaims_without_progress,
                    decision.last_reclaim_progress_version,
                )
                if row is None:
                    continue
                return self._claim_from_row(row, worker)

        async def _wrapped(conn: Any) -> ClaimedRun | None:
            async with conn.transaction():
                return await _operation(conn)

        return await self._run_write(_wrapped)

    def _claim_from_row(self, row: Any, worker: str) -> ClaimedRun:
        run = run_record(row)
        owner = run.owner_id
        run_uuid = parse_run_id(run.run_id)
        if run_uuid is None:
            raise RuntimeError("claimed run id is not a canonical UUID")
        if run.run_kind != "answer":
            return ClaimedRun(
                run=run,
                execution=RunExecutionContext(
                    owner_id=owner,
                    run_id=run.run_id,
                    worker_id=worker,
                    lease_owner=worker,
                    fencing_epoch=run.fencing_epoch,
                ),
            )
        prepared = run.prepared_input or run.accepted_input or {}
        raw_session_id = prepared.get("agent_session_id")
        if not raw_session_id:
            raise RuntimeError("claimed Answer run has no canonical Agent Session mapping")
        primary_session_id = SessionId(str(raw_session_id))
        execution = RunExecutionContext(
            owner_id=owner,
            run_id=run.run_id,
            worker_id=worker,
            lease_owner=worker,
            fencing_epoch=run.fencing_epoch,
            session_repository=PGAgentSessionRepository(
                pool=self._operation_pool,
                owner_id=owner,
                run_id=run_uuid,
                worker_id=worker,
                lease_owner=worker,
                fencing_epoch=run.fencing_epoch,
                primary_session_id=primary_session_id,
            ),
            progress_store=PGProgressStore(
                pool=self._operation_pool,
                owner_id=owner,
                run_id=run_uuid,
                worker_id=worker,
                lease_owner=worker,
                fencing_epoch=run.fencing_epoch,
            ),
            workspace_store=PGWorkspaceStore(
                pool=self._operation_pool,
                owner_id=owner,
                run_id=run_uuid,
                worker_id=worker,
                lease_owner=worker,
                fencing_epoch=run.fencing_epoch,
            ),
        )
        return ClaimedRun(run=run, execution=execution)

    async def heartbeat(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> LeaseRenewal:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return LeaseRenewal(renewed=False, cancel_requested=False)

        async def _operation(conn: Any) -> LeaseRenewal:
            row = await conn.fetchrow(
                _HEARTBEAT,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                RUN_LEASE_SECONDS,
            )
            if row is None:
                return LeaseRenewal(renewed=False, cancel_requested=False)
            return LeaseRenewal(renewed=True, cancel_requested=bool(row["cancel_requested"]))

        return await self._run_write(_operation)

    async def write_checkpoint(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
        phase: RunPhase | None = None,
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return False

        async def _operation(conn: Any) -> bool:
            value = await conn.fetchval(
                _WRITE_CHECKPOINT,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                json.dumps(dict(checkpoint), ensure_ascii=False),
                phase,
                RUN_LEASE_SECONDS,
            )
            return value is not None

        return await self._run_write(_operation)

    async def start_handoff(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return False

        async def _operation(conn: Any) -> bool:
            value = await conn.fetchval(
                _START_HANDOFF,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                json.dumps(dict(checkpoint), ensure_ascii=False),
                RUN_LEASE_SECONDS,
            )
            return value is not None

        return await self._run_write(_operation)

    async def record_phase(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        phase: RunPhase,
    ) -> int | None:
        return await self.append_event(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            phase=phase,
            event_type="progress",
            payload={"phase": phase},
        )

    async def append_event(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        phase: RunPhase | None,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> int | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None

        async def _operation(conn: Any) -> int | None:
            sequence = await conn.fetchval(
                _APPEND_EVENT,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                phase,
                event_type,
                json.dumps(dict(payload), ensure_ascii=False),
                RUN_LEASE_SECONDS,
            )
            return int(sequence) if sequence is not None else None

        return await self._run_write(_operation)

    # -- terminal transitions -----------------------------------------
    async def finish_success(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        result: Mapping[str, object],
        stop_reason: str | None = None,
        publications: Sequence[Any] = (),
    ) -> TerminalOutcome:
        return await self._finish_run(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            status="succeeded",
            stop_reason=stop_reason,
            result=result,
            error_kind=None,
            error_message=None,
            event_type="done",
            payload={"status": "succeeded", "result": result},
            withhold_on_cancel=True,
            publications=publications,
        )

    async def finish_failure(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        error_kind: str,
        error_message: str,
        result: Mapping[str, object] | None = None,
    ) -> TerminalOutcome:
        return await self._finish_run(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            status="failed",
            stop_reason=None,
            result=result,
            error_kind=error_kind,
            error_message=error_message,
            event_type="error",
            payload={"kind": error_kind, "message": error_message},
            withhold_on_cancel=True,
        )

    async def finish_cancelled(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> TerminalOutcome:
        return await self._finish_run(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            status="cancelled",
            stop_reason=None,
            result=None,
            error_kind=None,
            error_message=None,
            event_type="done",
            payload={"status": "cancelled"},
            withhold_on_cancel=False,
        )

    async def _finish_run(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        status: TerminalStatus,
        stop_reason: str | None,
        result: Mapping[str, object] | None,
        error_kind: str | None,
        error_message: str | None,
        event_type: str,
        payload: Mapping[str, Any],
        withhold_on_cancel: bool,
        publications: Sequence[Any] = (),
    ) -> TerminalOutcome:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return TerminalOutcome(committed=False, status=None, event_sequence=None)

        async def _operation(conn: Any) -> TerminalOutcome:
            async with conn.transaction():
                outcome = await finish_fenced_run(
                    conn,
                    owner_id=owner,
                    run_id=run_uuid,
                    lease_owner=worker_id,
                    fencing_epoch=fencing_epoch,
                    status=status,
                    stop_reason=stop_reason,
                    result=result,
                    error_kind=error_kind,
                    error_message=error_message,
                    event_type=event_type,
                    payload=payload,
                    withhold_on_cancel=withhold_on_cancel,
                )
                # Preserve publication and spill cleanup ownership for the
                # requested transition; a cancellation that beat it owns only
                # its terminal row and event.
                if outcome.committed and outcome.status == status:
                    if status == "succeeded" and publications:
                        await self._write_publications(conn, owner, run_uuid, publications)
                    await conn.execute(
                        "DELETE FROM dlightrag_answer_committed_spills"
                        " WHERE owner_id = $1 AND run_id = $2",
                        owner,
                        run_uuid,
                    )
                    await conn.execute(
                        "DELETE FROM dlightrag_answer_resources"
                        " WHERE owner_id = $1 AND run_id = $2 AND kind = 'committed_spill'",
                        owner,
                        run_uuid,
                    )
                return outcome

        return await self._run_write(_operation)

    async def defer(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
        next_attempt_at: Any,
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return False

        async def _operation(conn: Any) -> bool:
            value = await conn.fetchval(
                _DEFER_RUN,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                json.dumps(dict(checkpoint), ensure_ascii=False),
                next_attempt_at,
            )
            return value is not None

        return await self._run_write(_operation)

    async def resume_repair(self, *, owner_id: str, run_id: str) -> bool:
        """Explicitly make one waiting mutation claimable again without a new Run."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return False

        async def _operation(conn: Any) -> bool:
            value = await conn.fetchval(_RESUME_REPAIR, owner, run_uuid)
            return value is not None

        return await self._run_write(_operation)

    async def wait_for_repair(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return False

        async def _operation(conn: Any) -> bool:
            value = await conn.fetchval(
                _WAIT_FOR_REPAIR,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                json.dumps(dict(checkpoint), ensure_ascii=False),
            )
            return value is not None

        return await self._run_write(_operation)

    async def release_for_shutdown(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> ShutdownOutcome:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return "lease_lost"

        async def _operation(conn: Any) -> ShutdownOutcome:
            async with conn.transaction():
                row = await conn.fetchrow(_REQUEUE_RUN, owner, run_uuid, worker_id, fencing_epoch)
                if row is not None:
                    return "requeued"
                current = await conn.fetchrow(_SELECT_RUN, owner, run_uuid)
                if current is not None and current["cancel_requested_at"] is not None:
                    outcome = await finish_fenced_run(
                        conn,
                        owner_id=owner,
                        run_id=run_uuid,
                        lease_owner=worker_id,
                        fencing_epoch=fencing_epoch,
                        status="cancelled",
                        stop_reason=None,
                        result=None,
                        error_kind=None,
                        error_message=None,
                        event_type="done",
                        payload={"status": "cancelled"},
                        withhold_on_cancel=False,
                        cancel_requested=True,
                    )
                    return "cancelled" if outcome.committed else "lease_lost"
                return "lease_lost"

        return await self._run_write(_operation)

    # -- sweep and retention ------------------------------------------
    async def sweep_once(self) -> SweepOutcome:
        """Finalize cancel-pending and reclaim-poisoned rows without a slot."""

        async def _operation(conn: Any) -> SweepOutcome:
            async with conn.transaction():
                pending = await conn.fetch(_SELECT_CANCEL_PENDING, _BATCH_LIMIT)
                cancelled = 0
                if pending:
                    owners = [row["owner_id"] for row in pending]
                    run_ids = [row["run_id"] for row in pending]
                    cancelled = await conn.fetchval(
                        _FINALIZE_UNLEASED,
                        owners,
                        run_ids,
                        "cancelled",
                        None,
                        None,
                        "done",
                        json.dumps({"status": "cancelled"}),
                    )
                poisoned = await conn.fetch(
                    "SELECT owner_id, run_id FROM dlightrag_runs"
                    " WHERE status = 'running' AND lease_expires_at < NOW()"
                    " AND reclaims_without_progress >= $1"
                    " AND cancel_requested_at IS NULL"
                    " ORDER BY updated_at LIMIT $2"
                    " FOR UPDATE SKIP LOCKED",
                    MAX_RECLAIMS_WITHOUT_PROGRESS,
                    _BATCH_LIMIT,
                )
                abandoned = 0
                if poisoned:
                    owners = [row["owner_id"] for row in poisoned]
                    run_ids = [row["run_id"] for row in poisoned]
                    abandoned = await conn.fetchval(
                        _FINALIZE_UNLEASED,
                        owners,
                        run_ids,
                        "failed",
                        RUN_ABANDONED_ERROR_KIND,
                        _ABANDONED_ERROR_MESSAGE,
                        "error",
                        json.dumps(
                            {
                                "kind": RUN_ABANDONED_ERROR_KIND,
                                "message": _ABANDONED_ERROR_MESSAGE,
                            }
                        ),
                    )
                return SweepOutcome(cancelled=int(cancelled), abandoned=int(abandoned))

        return await self._run_write(_operation)

    async def trim_expired_event_logs(self) -> int:
        async def _operation(conn: Any) -> int:
            async with conn.transaction():
                rows = await conn.fetch(_SELECT_TRIMMABLE_RUNS, _BATCH_LIMIT)
                if not rows:
                    return 0
                owners = [row["owner_id"] for row in rows]
                run_ids = [row["run_id"] for row in rows]
                await conn.execute(_DELETE_EVENTS_FOR_RUNS, owners, run_ids)
                await conn.execute(_MARK_EVENTS_TRIMMED, owners, run_ids)
                return len(rows)

        return await self._run_write(_operation)

    async def prune_expired_runs(self) -> RunDeletion:
        """Retention order: delete runs (references cascade), then unreferenced blobs."""

        async def _operation(conn: Any) -> RunDeletion:
            async with conn.transaction():
                rows = await conn.fetch(_SELECT_EXPIRED_RUNS, _BATCH_LIMIT)
                if not rows:
                    return RunDeletion(runs=0, artifacts=0)
                owners = [row["owner_id"] for row in rows]
                run_ids = [row["run_id"] for row in rows]
                digest_rows = await conn.fetch(_SELECT_RUN_DIGESTS, owners, run_ids)
                session_rows = await conn.fetch(_SELECT_RUN_AGENT_SESSIONS, owners, run_ids)
                deleted = await conn.fetchval(_DELETE_RUNS, owners, run_ids)
                await _delete_unreferenced_agent_sessions(conn, session_rows)
                artifacts = await _delete_unreferenced(conn, *_digest_pairs(digest_rows))
                return RunDeletion(runs=int(deleted), artifacts=artifacts)

        return await self._run_write(_operation)


def _require_replay_match(
    *,
    owner: str,
    key: str | None,
    stored_fingerprint: str,
    fingerprint: str,
    row: Any,
) -> RunCreation:
    if stored_fingerprint != fingerprint:
        raise IdempotencyKeyConflict(
            f"owner {owner} reused idempotency key {key} with different normalized input"
        )
    return RunCreation(run=run_record(row), replayed=True)


__all__ = [
    "RUN_MIGRATIONS",
    "RUN_MIGRATION_SCOPE",
    "RUN_SCHEMA_TABLES",
    "PGRunStore",
    "run_columns",
    "run_record",
]
