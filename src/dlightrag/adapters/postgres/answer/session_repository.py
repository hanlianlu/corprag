# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Claim-bound PostgreSQL Session, progress, evidence, resource, and blob repositories.

Every bound store embeds owner, run, worker, lease owner, and fencing epoch at
claim time; its public methods carry no fencing parameters. Each mutating
transaction starts by locking the run row under the live lease/epoch predicate,
so a stale or lost lease changes zero rows.

Runtime transitions atomically commit HostDelta, ordered Entries, exact typed
registers, Session commit sequence, and durable run progress.

Session mutation is repository-exclusive and validated inductively: every
committed state was produced by this adapter while holding the Session row lock,
so a new transaction validates its delta plus the affected Lane pairs rather
than rescanning immutable history or unrelated register payloads. Out-of-band
row edits and pre-existing unrelated corruption are outside the transaction
interface; full snapshot loading still decodes every persisted Entry/register
and fails when a caller explicitly reads corrupt state.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from dlightrag.adapters.postgres.core._operations import ConnectionPool
from dlightrag.adapters.postgres.core._pool import pg_pool
from dlightrag.adapters.postgres.runtime._terminal import finish_fenced_run
from dlightrag.adapters.postgres.runtime.run_blob_store import BlobSizeConflict, write_complete_blob
from dlightrag.engine.agent.session.effects import JsonValue
from dlightrag.engine.agent.session.entries import (
    SessionEntry,
)
from dlightrag.engine.agent.session.ids import (
    EntryId,
    IntentId,
    LaneId,
    SessionId,
    StageIntentId,
)
from dlightrag.engine.agent.session.registers import (
    DeleteRegister,
    LaneHead,
    LaneState,
    RegisterRecord,
    RegisterRef,
    RegisterWrite,
    SetRegister,
    decode_register,
)
from dlightrag.engine.agent.session.repository import (
    AgentSessionSnapshot,
)
from dlightrag.engine.agent.session.transactions import (
    RegisterConflict,
    SessionTransaction,
    TransactionCommit,
    TransactionLeaseLost,
    TransactionOutcome,
)
from dlightrag.engine.runtime.progress import (
    StageCommit,
    StageCommitResult,
    StageConflict,
    StageEvidenceConflict,
    StageLeaseLost,
    StageProgressConflict,
    StageRecord,
    StageTerminalCommit,
    StageTerminalCommitResult,
)
from dlightrag.engine.runtime.settlements import (
    CompleteBlobDescriptor,
    EffectHostUpdate,
    MemoryOperationSettlement,
    OpaqueEvidenceResourceWrite,
    OpaqueEvidenceWrite,
    OpaqueFetchedResourceWrite,
)

_LEASE_PREDICATE = """
SELECT 1
FROM dlightrag_runs
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
FOR UPDATE
"""

_LOCK_PROGRESS_RUN = """
SELECT durable_progress_version,
       cancel_requested_at IS NOT NULL AS cancel_requested
FROM dlightrag_runs
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
FOR UPDATE
"""

_CHILD_LEASE_PREDICATE = """
SELECT 1
FROM dlightrag_answer_child_sessions
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND lease_owner = $4 AND fencing_epoch = $5
  AND status = 'running' AND lease_expires_at > NOW()
FOR UPDATE
"""

_LOCK_SESSION = """
SELECT lease_run_id::text, commit_sequence, fencing_epoch, last_sequence
FROM dlightrag_agent_sessions
WHERE owner_id = $1 AND session_id = $2
FOR UPDATE
"""

_CREATE_SESSION = """
INSERT INTO dlightrag_agent_sessions (
    owner_id, session_id, lease_run_id, commit_sequence, fencing_epoch
)
VALUES ($1, $2, $3, 0, $4)
ON CONFLICT (owner_id, session_id) DO NOTHING
"""

_ACTIVE_SESSION_RUN = """
SELECT 1
FROM dlightrag_runs
WHERE owner_id = $1 AND run_id = $2
  AND status = 'running' AND lease_expires_at > NOW()
"""

_INSERT_ENTRIES = """
INSERT INTO dlightrag_agent_session_entries (
    owner_id, session_id, sequence, entry_id, parent_entry_id,
    entry_type, schema_version, timestamp, payload_json
)
SELECT $1, $2, input.sequence, input.entry_id, input.parent_entry_id,
       input.entry_type, input.schema_version, input.entry_timestamp, input.payload_json
FROM jsonb_to_recordset($3::jsonb) AS input(
    sequence BIGINT,
    entry_id UUID,
    parent_entry_id UUID,
    entry_type TEXT,
    schema_version INTEGER,
    entry_timestamp TIMESTAMPTZ,
    payload_json JSONB
)
"""

_ADVANCE_SESSION = """
UPDATE dlightrag_agent_sessions
SET commit_sequence = commit_sequence + 1,
    last_sequence = last_sequence + $3,
    updated_at = NOW()
WHERE owner_id = $1 AND session_id = $2
RETURNING commit_sequence
"""

_ADVANCE_PROGRESS = """
UPDATE dlightrag_runs
SET durable_progress_version = durable_progress_version + 1,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
"""

_INSERT_MEMORY_OPERATION_EVENT = """
WITH bumped AS (
    UPDATE dlightrag_runs
    SET next_event_sequence = next_event_sequence + 1,
        updated_at = NOW()
    WHERE owner_id = $1 AND run_id = $2
      AND lease_owner = $3 AND fencing_epoch = $4
      AND status = 'running' AND lease_expires_at > NOW()
    RETURNING next_event_sequence - 1 AS event_sequence
)
INSERT INTO dlightrag_run_events (
    owner_id, run_id, event_sequence, event_type, payload
)
SELECT $1, $2, event_sequence, 'memory_operation_settled', $5::jsonb
FROM bumped
"""

_SELECT_SESSION_SNAPSHOT = """
SELECT commit_sequence, last_sequence
FROM dlightrag_agent_sessions
WHERE owner_id = $1 AND session_id = $2
"""

_SELECT_ENTRIES = """
SELECT sequence, entry_id::text, parent_entry_id::text, entry_type,
       schema_version, timestamp, payload_json
FROM dlightrag_agent_session_entries
WHERE owner_id = $1 AND session_id = $2
ORDER BY sequence
"""

_SELECT_ENTRY_DELTA = """
SELECT sequence, entry_id::text, parent_entry_id::text, entry_type,
       schema_version, timestamp, payload_json
FROM dlightrag_agent_session_entries
WHERE owner_id = $1 AND session_id = $2
  AND sequence > $3 AND sequence <= $4
ORDER BY sequence
"""

_SELECT_REGISTERS = """
SELECT register_kind, register_key, sequence, payload_json
FROM dlightrag_agent_session_registers
WHERE owner_id = $1 AND session_id = $2
ORDER BY register_kind, register_key
"""

_SELECT_EXPECTED_REGISTERS = """
WITH expected(register_kind, register_key, ordinal) AS (
    SELECT *
    FROM unnest($3::text[], $4::text[]) WITH ORDINALITY
)
SELECT expected.ordinal, expected.register_kind, expected.register_key, stored.sequence
FROM expected
LEFT JOIN dlightrag_agent_session_registers AS stored
  ON stored.owner_id = $1 AND stored.session_id = $2
 AND stored.register_kind = expected.register_kind
 AND stored.register_key = expected.register_key
ORDER BY expected.ordinal
"""

_SELECT_ENTRY_IDENTITIES = """
SELECT entry_id::text
FROM dlightrag_agent_session_entries
WHERE owner_id = $1 AND session_id = $2
  AND entry_id = ANY($3::uuid[])
"""

_SELECT_AFFECTED_LANES = """
SELECT register_kind, register_key,
       CASE
           WHEN register_kind = 'lane_state' AND register_key = ANY($4::text[])
           THEN payload_json
           ELSE NULL
       END AS payload_json
FROM dlightrag_agent_session_registers
WHERE owner_id = $1 AND session_id = $2
  AND register_kind IN ('lane_head', 'lane_state')
  AND register_key = ANY($3::text[])
"""

_SET_REGISTERS = """
INSERT INTO dlightrag_agent_session_registers (
    owner_id, session_id, register_kind, register_key, sequence, payload_json
)
SELECT $1, $2, input.register_kind, input.register_key, $3, input.payload_json
FROM jsonb_to_recordset($4::jsonb) AS input(
    register_kind TEXT,
    register_key TEXT,
    payload_json JSONB
)
ON CONFLICT (owner_id, session_id, register_kind, register_key)
DO UPDATE SET sequence = EXCLUDED.sequence, payload_json = EXCLUDED.payload_json
"""

_DELETE_REGISTERS = """
DELETE FROM dlightrag_agent_session_registers AS stored
USING unnest($3::text[], $4::text[]) AS input(register_kind, register_key)
WHERE stored.owner_id = $1 AND stored.session_id = $2
  AND stored.register_kind = input.register_kind
  AND stored.register_key = input.register_key
"""

_INSERT_EVIDENCE = """
INSERT INTO dlightrag_answer_evidence (
    owner_id, run_id, session_id, intent_id, result_ordinal,
    content_digest, locator_digest, content, locator
)
SELECT $1, $2, input.session_id, input.intent_id, input.result_ordinal,
       input.content_digest, input.locator_digest, input.content, input.locator
FROM unnest(
    $3::uuid[], $4::uuid[], $5::integer[],
    $6::text[], $7::text[], $8::bytea[], $9::bytea[]
) WITH ORDINALITY AS input(
    session_id, intent_id, result_ordinal,
    content_digest, locator_digest, content, locator, ordinality
)
ORDER BY input.ordinality
ON CONFLICT (owner_id, run_id, session_id, intent_id, result_ordinal) DO NOTHING
"""

_SELECT_EVIDENCE_CONFLICT = """
SELECT 1
FROM unnest(
    $3::uuid[], $4::uuid[], $5::integer[], $6::text[], $7::text[]
) AS input(session_id, intent_id, result_ordinal, content_digest, locator_digest)
LEFT JOIN dlightrag_answer_evidence AS stored
  ON stored.owner_id = $1 AND stored.run_id = $2
 AND stored.session_id = input.session_id
 AND stored.intent_id = input.intent_id
 AND stored.result_ordinal = input.result_ordinal
WHERE stored.content_digest IS NULL
   OR stored.content_digest IS DISTINCT FROM input.content_digest
   OR stored.locator_digest IS DISTINCT FROM input.locator_digest
LIMIT 1
"""

_INSERT_RESOURCE = """
INSERT INTO dlightrag_answer_resources (
    owner_id, run_id, resource_id, kind, safe_name, media_type, capabilities,
    ordinal, blob_digest, locator_digest, source_locator,
    session_id, intent_id, result_ordinal
)
VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12, $13, $14)
ON CONFLICT (owner_id, run_id, resource_id) DO UPDATE
SET capabilities = EXCLUDED.capabilities || jsonb_build_object(
    'resource_aliases',
    to_jsonb(ARRAY(
        SELECT DISTINCT alias
        FROM jsonb_array_elements_text(
            COALESCE(dlightrag_answer_resources.capabilities->'resource_aliases', '[]'::jsonb)
            || COALESCE(EXCLUDED.capabilities->'resource_aliases', '[]'::jsonb)
        ) AS merged(alias)
        ORDER BY alias
    ))
)
WHERE dlightrag_answer_resources.kind = 'fetched_blob'
  AND EXCLUDED.kind = 'fetched_blob'
  AND dlightrag_answer_resources.blob_digest = EXCLUDED.blob_digest
  AND dlightrag_answer_resources.locator_digest = EXCLUDED.locator_digest
"""

_SELECT_RESOURCE_DIGESTS = """
SELECT kind, blob_digest, locator_digest
FROM dlightrag_answer_resources
WHERE owner_id = $1 AND run_id = $2 AND resource_id = $3
"""


class _EvidenceIdentityConflict(Exception):
    """Rolls back the current settlement transaction as an EvidenceConflict."""


def _uuid(value: Any) -> Any:
    """Coerce a canonical id value to a PostgreSQL UUID parameter."""
    return uuid.UUID(str(value))


async def _write_evidence_identities(
    conn: Any,
    *,
    owner_id: str,
    run_id: Any,
    writes: Sequence[OpaqueEvidenceWrite],
) -> None:
    """Insert and verify one ordered evidence identity batch in two statements."""
    if not writes:
        return
    session_ids = [_uuid(write.session_id) for write in writes]
    intent_ids = [_uuid(write.intent_id) for write in writes]
    ordinals = [write.result_ordinal for write in writes]
    content_digests = [write.content_digest for write in writes]
    locator_digests = [write.locator_digest for write in writes]
    await conn.execute(
        _INSERT_EVIDENCE,
        owner_id,
        run_id,
        session_ids,
        intent_ids,
        ordinals,
        content_digests,
        locator_digests,
        [write.content for write in writes],
        [write.locator for write in writes],
    )
    if (
        await conn.fetchval(
            _SELECT_EVIDENCE_CONFLICT,
            owner_id,
            run_id,
            session_ids,
            intent_ids,
            ordinals,
            content_digests,
            locator_digests,
        )
        is not None
    ):
        raise _EvidenceIdentityConflict()


class PGAgentSessionRepository:
    """One claim-bound Agent Session Repository over PostgreSQL.

    All Session writes must cross :meth:`transact`. The locked Session row is
    the serialization point, and each successful transaction preserves the
    Entry-tree and Lane-pair invariants. Validation therefore probes only
    incoming Entry identities/external parents and main/touched Lane pairs;
    this hot path is deliberately not a scrubber for unrelated out-of-band
    database corruption.
    """

    def __init__(
        self,
        *,
        pool: ConnectionPool | None,
        owner_id: str,
        run_id: Any,
        worker_id: str,
        lease_owner: str,
        fencing_epoch: int,
        primary_session_id: SessionId | None = None,
        child_session_id: SessionId | None = None,
    ) -> None:
        self._pool = pool
        self._owner_id = owner_id
        self._run_id = run_id
        self._worker_id = worker_id
        self._lease_owner = lease_owner
        self._fencing_epoch = fencing_epoch
        self._primary_session_id = primary_session_id
        self._child_session_id = child_session_id

    def for_child(
        self, child_session_id: SessionId, *, fencing_epoch: int
    ) -> PGAgentSessionRepository:
        """Bind the same HostDelta owner to one independently leased Child Session."""
        return PGAgentSessionRepository(
            pool=self._pool,
            owner_id=self._owner_id,
            run_id=self._run_id,
            worker_id=self._worker_id,
            lease_owner=self._lease_owner,
            fencing_epoch=fencing_epoch,
            primary_session_id=self._primary_session_id,
            child_session_id=child_session_id,
        )

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[Any]:
        pool = self._pool if self._pool is not None else await pg_pool.get()
        async with pool.acquire() as conn:
            yield conn

    async def load(self, session_id: SessionId) -> AgentSessionSnapshot:
        """Fully decode and validate one coherent Session corruption-scrub snapshot."""
        async with self._connection() as conn:
            async with conn.transaction(isolation="repeatable_read", readonly=True):
                session_row = await conn.fetchrow(
                    _SELECT_SESSION_SNAPSHOT, self._owner_id, _uuid(session_id.value)
                )
                if session_row is None:
                    return AgentSessionSnapshot(
                        session_id=session_id,
                        commit_sequence=0,
                        last_entry_sequence=0,
                        entries=(),
                        registers=(),
                    )
                commit_sequence = int(session_row["commit_sequence"])
                last_entry_sequence = int(session_row["last_sequence"])
                entries = await self._load_entries(
                    conn,
                    session_id,
                    expected_last_sequence=last_entry_sequence,
                )
                registers = await self._load_registers(conn, session_id)
                return AgentSessionSnapshot(
                    session_id=session_id,
                    commit_sequence=commit_sequence,
                    last_entry_sequence=last_entry_sequence,
                    entries=tuple(entries),
                    registers=tuple(registers),
                )

    async def refresh(
        self,
        session_id: SessionId,
        *,
        previous: AgentSessionSnapshot,
    ) -> AgentSessionSnapshot:
        """Read only the immutable Entry suffix and exact registers after advancement."""
        if previous.session_id != session_id:
            raise ValueError("Agent Session refresh snapshot belongs to another Session")
        async with self._connection() as conn:
            # Metadata, Entry suffix, and current registers must describe one commit.
            async with conn.transaction(isolation="repeatable_read", readonly=True):
                session_row = await conn.fetchrow(
                    _SELECT_SESSION_SNAPSHOT, self._owner_id, _uuid(session_id.value)
                )
                commit_sequence = 0 if session_row is None else int(session_row["commit_sequence"])
                last_entry_sequence = (
                    0 if session_row is None else int(session_row["last_sequence"])
                )
                if (
                    commit_sequence < previous.commit_sequence
                    or last_entry_sequence < previous.last_entry_sequence
                ):
                    raise ValueError("Agent Session refresh cursor regressed")
                if commit_sequence == previous.commit_sequence:
                    if last_entry_sequence != previous.last_entry_sequence:
                        raise ValueError("Agent Session refresh metadata is inconsistent")
                    return previous
                if session_row is None:
                    raise ValueError("Agent Session refresh metadata is inconsistent")
                entries = await self._load_entry_delta(
                    conn,
                    session_id,
                    after_sequence=previous.last_entry_sequence,
                    through_sequence=last_entry_sequence,
                )
                registers = await self._load_registers(conn, session_id)
                # Tuple concatenation reuses every decoded immutable old Entry reference.
                return AgentSessionSnapshot(
                    session_id=session_id,
                    commit_sequence=commit_sequence,
                    last_entry_sequence=last_entry_sequence,
                    entries=previous.entries + tuple(entries),
                    registers=tuple(registers),
                    selected_lane_id=previous.selected_lane_id,
                )

    async def _load_entries(
        self,
        conn: Any,
        session_id: SessionId,
        *,
        expected_last_sequence: int,
    ) -> list[SessionEntry]:
        rows = await conn.fetch(_SELECT_ENTRIES, self._owner_id, _uuid(session_id.value))
        if len(rows) != expected_last_sequence or any(
            int(row["sequence"]) != expected for expected, row in enumerate(rows, start=1)
        ):
            raise ValueError("Agent Session full Entry rows are not gap-free")
        entries = [_decode_entry(row, session_id=session_id) for row in rows]
        if len({entry.entry_id for entry in entries}) != len(entries) or any(
            entry.session_id != session_id for entry in entries
        ):
            raise ValueError("Agent Session full Entry identities are corrupt")
        return entries

    async def _load_entry_delta(
        self,
        conn: Any,
        session_id: SessionId,
        *,
        after_sequence: int,
        through_sequence: int,
    ) -> list[SessionEntry]:
        rows = await conn.fetch(
            _SELECT_ENTRY_DELTA,
            self._owner_id,
            _uuid(session_id.value),
            after_sequence,
            through_sequence,
        )
        expected_count = through_sequence - after_sequence
        if len(rows) != expected_count or any(
            int(row["sequence"]) != after_sequence + offset
            for offset, row in enumerate(rows, start=1)
        ):
            raise ValueError("Agent Session refresh Entry delta is not gap-free")
        return [_decode_entry(row, session_id=session_id) for row in rows]

    async def _load_registers(self, conn: Any, session_id: SessionId) -> list[RegisterRecord]:
        rows = await conn.fetch(
            _SELECT_REGISTERS,
            self._owner_id,
            _uuid(session_id.value),
        )
        records: list[RegisterRecord] = []
        for row in rows:
            ref = RegisterRef(
                kind=str(row["register_kind"]),  # type: ignore[arg-type]
                key=str(row["register_key"]),
            )
            value = decode_register(
                kind=ref.kind,
                payload=_json_payload(row["payload_json"]),
            )
            if value.ref != ref:
                raise ValueError("Agent Session register payload identity is corrupt")
            records.append(RegisterRecord(value=value, sequence=int(row["sequence"])))
        return records

    async def transact(
        self,
        *,
        session_id: SessionId,
        fencing_epoch: int,
        transaction: SessionTransaction[EffectHostUpdate],
    ) -> TransactionOutcome:
        """Apply one exact-register-CAS Session transaction."""
        if fencing_epoch != self._fencing_epoch:
            return TransactionLeaseLost()
        if self._child_session_id is not None and session_id != self._child_session_id:
            return TransactionLeaseLost()
        if (
            self._child_session_id is None
            and self._primary_session_id is not None
            and session_id != self._primary_session_id
        ):
            return TransactionLeaseLost()
        async with self._connection() as conn:
            async with conn.transaction():
                if await self._hold_lease(conn) is None:
                    return TransactionLeaseLost()
                await conn.execute(
                    _CREATE_SESSION,
                    self._owner_id,
                    _uuid(session_id.value),
                    self._run_id,
                    fencing_epoch,
                )
                session_row = await conn.fetchrow(
                    _LOCK_SESSION, self._owner_id, _uuid(session_id.value)
                )
                stored_epoch = int(session_row["fencing_epoch"])
                stored_run_id = str(session_row["lease_run_id"])
                current_run_id = str(self._run_id)
                if stored_run_id != current_run_id:
                    if (
                        await conn.fetchval(
                            _ACTIVE_SESSION_RUN,
                            self._owner_id,
                            _uuid(stored_run_id),
                        )
                        is not None
                    ):
                        return TransactionLeaseLost()
                    await conn.execute(
                        "UPDATE dlightrag_agent_sessions"
                        " SET lease_run_id = $3, fencing_epoch = $4"
                        " WHERE owner_id = $1 AND session_id = $2",
                        self._owner_id,
                        _uuid(session_id.value),
                        self._run_id,
                        fencing_epoch,
                    )
                elif stored_epoch > fencing_epoch:
                    return TransactionLeaseLost()
                elif stored_epoch < fencing_epoch:
                    await conn.execute(
                        "UPDATE dlightrag_agent_sessions SET fencing_epoch = $3"
                        " WHERE owner_id = $1 AND session_id = $2",
                        self._owner_id,
                        _uuid(session_id.value),
                        fencing_epoch,
                    )
                current_sequence = int(session_row["commit_sequence"])
                conflict = await self._check_register_expectations(conn, session_id, transaction)
                if conflict is not None:
                    return conflict
                await self._validate_transaction_entries(
                    conn,
                    session_id,
                    transaction.entries,
                    last_sequence=int(session_row["last_sequence"]),
                )
                await self._validate_transaction_registers(conn, session_id, transaction)
                next_sequence = current_sequence + 1
                last_entry_sequence = int(session_row["last_sequence"])
                entry_sequences = tuple(
                    range(
                        last_entry_sequence + 1,
                        last_entry_sequence + 1 + len(transaction.entries),
                    )
                )
                if transaction.host_delta is not None:
                    settlement = transaction.host_delta
                    await self._write_host_update(
                        conn,
                        session_id,
                        settlement.intent_id,
                        settlement.value,
                    )
                    if settlement.value.memory_operation is not None:
                        await conn.execute(
                            _INSERT_MEMORY_OPERATION_EVENT,
                            self._owner_id,
                            self._run_id,
                            self._lease_owner,
                            self._fencing_epoch,
                            json.dumps(
                                _memory_event_payload(
                                    session_id,
                                    settlement.intent_id,
                                    settlement.value.memory_operation,
                                ),
                                ensure_ascii=False,
                            ),
                        )
                await self._insert_entries(conn, session_id, transaction.entries, entry_sequences)
                await self._write_registers(
                    conn,
                    session_id,
                    transaction.register_writes,
                    sequence=next_sequence,
                )
                register_sequences = tuple(
                    (write.ref, next_sequence) for write in transaction.register_writes
                )
                await conn.execute(
                    _ADVANCE_SESSION,
                    self._owner_id,
                    _uuid(session_id.value),
                    len(transaction.entries),
                )
                if transaction.advances_durable_progress:
                    await conn.execute(_ADVANCE_PROGRESS, self._owner_id, self._run_id)
                return TransactionCommit(
                    commit_sequence=next_sequence,
                    appended_sequences=entry_sequences,
                    register_sequences=register_sequences,
                )

    async def _check_register_expectations(
        self,
        conn: Any,
        session_id: SessionId,
        transaction: SessionTransaction[EffectHostUpdate],
    ) -> RegisterConflict | None:
        if not transaction.expectations:
            return None
        rows = await conn.fetch(
            _SELECT_EXPECTED_REGISTERS,
            self._owner_id,
            _uuid(session_id.value),
            [expectation.ref.kind for expectation in transaction.expectations],
            [expectation.ref.key for expectation in transaction.expectations],
        )
        if len(rows) != len(transaction.expectations):
            raise ValueError("Agent Session register expectation query returned incomplete results")
        for expectation, row in zip(transaction.expectations, rows, strict=True):
            actual = int(row["sequence"]) if row["sequence"] is not None else None
            if actual != expectation.sequence:
                return RegisterConflict(
                    ref=expectation.ref,
                    expected_sequence=expectation.sequence,
                    current_sequence=actual,
                )
        return None

    async def _validate_transaction_registers(
        self,
        conn: Any,
        session_id: SessionId,
        transaction: SessionTransaction[EffectHostUpdate],
    ) -> None:
        main_lane = LaneId.main().value
        touched_lanes = {
            write.ref.key
            for write in transaction.register_writes
            if write.ref.kind in {"lane_head", "lane_state"}
        }
        affected_lanes = [main_lane, *sorted(touched_lanes - {main_lane})]
        advanced_lanes = {
            write.value.lane_id.value
            for write in transaction.register_writes
            if transaction.entries
            and isinstance(write, SetRegister)
            and isinstance(write.value, LaneHead)
            and write.value.entry_id == transaction.entries[-1].entry_id
        }
        rows = await conn.fetch(
            _SELECT_AFFECTED_LANES,
            self._owner_id,
            _uuid(session_id.value),
            affected_lanes,
            sorted(advanced_lanes),
        )
        refs: set[RegisterRef] = set()
        advanced_states: dict[str, LaneState] = {}
        for row in rows:
            ref = RegisterRef(
                kind=str(row["register_kind"]),  # type: ignore[arg-type]
                key=str(row["register_key"]),
            )
            refs.add(ref)
            if ref.kind == "lane_state" and ref.key in advanced_lanes:
                state = decode_register(
                    kind=ref.kind,
                    payload=_json_payload(row["payload_json"]),
                )
                if not isinstance(state, LaneState) or state.ref != ref:
                    raise ValueError("Agent Session Lane State payload identity is corrupt")
                advanced_states[ref.key] = state
        for write in transaction.register_writes:
            if write.ref.kind not in {"lane_head", "lane_state"}:
                continue
            if isinstance(write, SetRegister):
                refs.add(write.ref)
                if isinstance(write.value, LaneState) and write.ref.key in advanced_lanes:
                    advanced_states[write.ref.key] = write.value
            elif isinstance(write, DeleteRegister):
                refs.discard(write.ref)
                if write.ref.kind == "lane_state":
                    advanced_states.pop(write.ref.key, None)
        for lane_id in affected_lanes:
            has_head = RegisterRef("lane_head", lane_id) in refs
            has_state = RegisterRef("lane_state", lane_id) in refs
            if has_head != has_state or (lane_id == main_lane and not has_head):
                raise ValueError("Session registers require complete main and Lane pairs")
        for lane_id in advanced_lanes:
            state = advanced_states.get(lane_id)
            if state is not None and state.archived:
                raise ValueError("an archived Lane is not writable")

    async def _validate_transaction_entries(
        self,
        conn: Any,
        session_id: SessionId,
        entries: Sequence[SessionEntry],
        *,
        last_sequence: int,
    ) -> None:
        if not entries:
            return
        incoming: list[EntryId] = []
        incoming_set: set[EntryId] = set()
        roots = 1 if last_sequence > 0 else 0
        for entry in entries:
            if entry.session_id != session_id:
                raise ValueError("transaction Entry belongs to another Session")
            if entry.entry_id in incoming_set:
                raise ValueError("transaction Entry identity already exists")
            if entry.parent_entry_id is None:
                if incoming or roots:
                    raise ValueError("only the first Session Entry can be a root")
                roots += 1
            incoming.append(entry.entry_id)
            incoming_set.add(entry.entry_id)
        external_parents: list[EntryId] = []
        probed: set[EntryId] = set()
        known_local: set[EntryId] = set()
        for entry in entries:
            parent_id = entry.parent_entry_id
            if parent_id is not None:
                if parent_id in incoming_set and parent_id not in known_local:
                    raise ValueError("transaction Entry parent is missing")
                if parent_id not in incoming_set and parent_id not in probed:
                    external_parents.append(parent_id)
                    probed.add(parent_id)
            known_local.add(entry.entry_id)
        probe_ids = [*incoming, *external_parents]
        rows = await conn.fetch(
            _SELECT_ENTRY_IDENTITIES,
            self._owner_id,
            _uuid(session_id.value),
            [_uuid(entry_id.value) for entry_id in probe_ids],
        )
        stored = {EntryId(str(row["entry_id"])) for row in rows}
        known_local = set()
        for entry in entries:
            if entry.entry_id in stored:
                raise ValueError("transaction Entry identity already exists")
            if (
                entry.parent_entry_id is not None
                and entry.parent_entry_id not in known_local
                and entry.parent_entry_id not in stored
            ):
                raise ValueError("transaction Entry parent is missing")
            known_local.add(entry.entry_id)

    async def _insert_entries(
        self,
        conn: Any,
        session_id: SessionId,
        entries: Sequence[SessionEntry],
        sequences: Sequence[int],
    ) -> None:
        if not entries:
            return
        payload = [
            {
                "sequence": sequence,
                "entry_id": entry.entry_id.value,
                "parent_entry_id": (
                    entry.parent_entry_id.value if entry.parent_entry_id is not None else None
                ),
                "entry_type": entry.entry_type,
                "schema_version": entry.schema_version,
                "entry_timestamp": entry.timestamp.isoformat(),
                "payload_json": entry.canonical_payload(),
            }
            for entry, sequence in zip(entries, sequences, strict=True)
        ]
        await conn.execute(
            _INSERT_ENTRIES,
            self._owner_id,
            _uuid(session_id.value),
            json.dumps(payload, ensure_ascii=False),
        )

    async def _write_registers(
        self,
        conn: Any,
        session_id: SessionId,
        writes: Sequence[RegisterWrite],
        *,
        sequence: int,
    ) -> None:
        sets = [write for write in writes if isinstance(write, SetRegister)]
        deletes = [write for write in writes if isinstance(write, DeleteRegister)]
        if sets:
            payload = [
                {
                    "register_kind": write.ref.kind,
                    "register_key": write.ref.key,
                    "payload_json": write.value.canonical_payload(),
                }
                for write in sets
            ]
            await conn.execute(
                _SET_REGISTERS,
                self._owner_id,
                _uuid(session_id.value),
                sequence,
                json.dumps(payload, ensure_ascii=False),
            )
        if deletes:
            await conn.execute(
                _DELETE_REGISTERS,
                self._owner_id,
                _uuid(session_id.value),
                [write.ref.kind for write in deletes],
                [write.ref.key for write in deletes],
            )

    async def _hold_lease(self, conn: Any) -> Any:
        if self._child_session_id is not None:
            return await conn.fetchval(
                _CHILD_LEASE_PREDICATE,
                self._owner_id,
                self._run_id,
                _uuid(self._child_session_id.value),
                self._lease_owner,
                self._fencing_epoch,
            )
        return await conn.fetchval(
            _LEASE_PREDICATE,
            self._owner_id,
            self._run_id,
            self._lease_owner,
            self._fencing_epoch,
        )

    async def load_evidence(self, session_id: SessionId) -> list[OpaqueEvidenceWrite]:
        """Return this run's durable evidence writes, oldest first (adapter read)."""
        async with self._connection() as conn:
            rows = await conn.fetch(
                "SELECT session_id::text, intent_id::text, result_ordinal,"
                " content_digest, locator_digest, content, locator"
                " FROM dlightrag_answer_evidence"
                " WHERE owner_id = $1 AND run_id = $2 AND session_id = $3"
                " ORDER BY created_at, result_ordinal",
                self._owner_id,
                self._run_id,
                _uuid(session_id.value),
            )
        return [
            OpaqueEvidenceWrite(
                session_id=str(row["session_id"]),
                intent_id=str(row["intent_id"]),
                result_ordinal=int(row["result_ordinal"]),
                content_digest=str(row["content_digest"]),
                locator_digest=str(row["locator_digest"]),
                content=bytes(row["content"]),
                locator=bytes(row["locator"]),
            )
            for row in rows
        ]

    async def _write_host_update(
        self,
        conn: Any,
        session_id: SessionId,
        intent_id: IntentId,
        update: EffectHostUpdate,
    ) -> str:
        await _write_evidence_identities(
            conn,
            owner_id=self._owner_id,
            run_id=self._run_id,
            writes=update.evidence,
        )
        for resource in update.resources:
            await self._write_evidence_resource(conn, resource)
        # New blob identities use one canonical lock order across transactions;
        # resource/evidence projection retains the Host update's original order.
        for fetched in sorted(update.fetched, key=lambda item: item.complete_blob.digest):
            await self._write_complete_blob(conn, fetched.complete_blob)
        for fetched in update.fetched:
            await self._write_fetched_resource(conn, fetched.resource)
        await _write_evidence_identities(
            conn,
            owner_id=self._owner_id,
            run_id=self._run_id,
            writes=tuple(write for fetched in update.fetched for write in fetched.evidence),
        )

        if update.committed_outputs:
            from dlightrag.adapters.postgres.answer.workspace import _upsert_spill
            from dlightrag.engine.runtime.workspace import CommittedSpillRecord

            for output in update.committed_outputs:
                await _upsert_spill(
                    conn,
                    self._owner_id,
                    self._run_id,
                    CommittedSpillRecord(
                        resource_id=output.resource_id,
                        content_digest=output.content_digest,
                        size_bytes=output.size_bytes,
                        session_id=output.session_id,
                        intent_id=output.intent_id,
                    ),
                )

        inventory = update.workspace_inventory
        if inventory is not None:
            if inventory.replace_all:
                await conn.execute(
                    "DELETE FROM dlightrag_answer_workspace_inventory"
                    " WHERE owner_id = $1 AND run_id = $2",
                    self._owner_id,
                    self._run_id,
                )
            else:
                for path in inventory.deletes:
                    await conn.execute(
                        "DELETE FROM dlightrag_answer_workspace_inventory"
                        " WHERE owner_id = $1 AND run_id = $2 AND relative_path = $3",
                        self._owner_id,
                        self._run_id,
                        path,
                    )
            for record in inventory.upserts:
                await conn.execute(
                    "INSERT INTO dlightrag_answer_workspace_inventory ("
                    " owner_id, run_id, relative_path, entry_type, mode, size_bytes, content_digest)"
                    " VALUES ($1, $2, $3, $4, $5, $6, $7)"
                    " ON CONFLICT (owner_id, run_id, relative_path) DO UPDATE SET"
                    " entry_type = EXCLUDED.entry_type, mode = EXCLUDED.mode,"
                    " size_bytes = EXCLUDED.size_bytes, content_digest = EXCLUDED.content_digest",
                    self._owner_id,
                    self._run_id,
                    record.relative_path,
                    record.entry_type,
                    record.mode,
                    record.size_bytes,
                    record.content_digest,
                )

        attachment = update.artifact_attachment
        if attachment is not None:
            if attachment.session_id != session_id.value or attachment.intent_id != intent_id.value:
                raise ValueError("Artifact attachment provenance mismatches its settlement")
            await conn.execute(
                "INSERT INTO dlightrag_answer_artifact_attachments ("
                " owner_id, run_id, relative_path, label, content_digest, size_bytes,"
                " presentation, session_id, intent_id)"
                " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)"
                " ON CONFLICT (owner_id, run_id, relative_path) DO UPDATE SET"
                " label = EXCLUDED.label, content_digest = EXCLUDED.content_digest,"
                " size_bytes = EXCLUDED.size_bytes, presentation = EXCLUDED.presentation,"
                " session_id = EXCLUDED.session_id, intent_id = EXCLUDED.intent_id,"
                " attachment_order = EXCLUDED.attachment_order,"
                " attached_at = EXCLUDED.attached_at",
                self._owner_id,
                self._run_id,
                attachment.relative_path,
                attachment.label,
                attachment.content_digest,
                attachment.size_bytes,
                attachment.presentation,
                _uuid(attachment.session_id),
                _uuid(attachment.intent_id),
            )
        return _host_update_digest(update)

    async def _write_evidence_resource(self, conn: Any, write: OpaqueEvidenceResourceWrite) -> None:
        await conn.execute(
            _INSERT_RESOURCE,
            self._owner_id,
            self._run_id,
            write.resource_id,
            "evidence",
            write.safe_name,
            write.media_type,
            json.dumps(write.capabilities, ensure_ascii=False),
            None,
            None,
            write.locator_digest,
            None,
            write.session_id,
            write.intent_id,
            write.result_ordinal,
        )
        row = await conn.fetchrow(
            _SELECT_RESOURCE_DIGESTS, self._owner_id, self._run_id, write.resource_id
        )
        if (
            row is None
            or row["kind"] != "evidence"
            or row["locator_digest"] != write.locator_digest
        ):
            raise _EvidenceIdentityConflict()

    async def _write_complete_blob(self, conn: Any, blob: CompleteBlobDescriptor) -> None:
        try:
            await write_complete_blob(
                conn,
                owner_id=self._owner_id,
                digest=blob.digest,
                total_bytes=blob.total_bytes,
                chunks=blob.chunks,
            )
        except BlobSizeConflict as exc:
            raise _EvidenceIdentityConflict() from exc

    async def _write_fetched_resource(self, conn: Any, write: OpaqueFetchedResourceWrite) -> None:
        await conn.execute(
            _INSERT_RESOURCE,
            self._owner_id,
            self._run_id,
            write.resource_id,
            "fetched_blob",
            write.safe_name,
            write.media_type,
            json.dumps(write.capabilities, ensure_ascii=False),
            write.ordinal,
            write.blob_digest,
            write.source_locator_digest,
            write.source_locator,
            write.session_id,
            write.intent_id,
            None,
        )
        row = await conn.fetchrow(
            _SELECT_RESOURCE_DIGESTS, self._owner_id, self._run_id, write.resource_id
        )
        if (
            row is None
            or row["kind"] != "fetched_blob"
            or row["blob_digest"] != write.blob_digest
            or row["locator_digest"] != write.source_locator_digest
        ):
            raise _EvidenceIdentityConflict()


class PGProgressStore:
    """One claim-bound RunProgressStore over PostgreSQL."""

    def __init__(
        self,
        *,
        pool: ConnectionPool | None,
        owner_id: str,
        run_id: Any,
        worker_id: str,
        lease_owner: str,
        fencing_epoch: int,
    ) -> None:
        self._pool = pool
        self._owner_id = owner_id
        self._run_id = run_id
        self._worker_id = worker_id
        self._lease_owner = lease_owner
        self._fencing_epoch = fencing_epoch

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[Any]:
        pool = self._pool if self._pool is not None else await pg_pool.get()
        async with pool.acquire() as conn:
            yield conn

    async def load_stage(self, stage_intent_id: StageIntentId) -> StageRecord | None:
        async with self._connection() as conn:
            row = await conn.fetchrow(
                "SELECT stage_intent_id::text, stage_name, progress_version, state,"
                " state_digest, settled_at::text"
                " FROM dlightrag_answer_run_stages"
                " WHERE owner_id = $1 AND run_id = $2 AND stage_intent_id = $3",
                self._owner_id,
                self._run_id,
                _uuid(stage_intent_id.value),
            )
            if row is None:
                return None
            return StageRecord(
                stage_intent_id=stage_intent_id,
                stage_name=str(row["stage_name"]),
                progress_version=int(row["progress_version"]),
                state=_json_payload(row["state"]),
                state_digest=str(row["state_digest"]),
                evidence_count=0,
                settled_at=row["settled_at"],
            )

    async def settle_terminal(
        self,
        *,
        expected_progress_version: int,
        stage_intent_id: StageIntentId,
        state: JsonValue,
        result: Mapping[str, Any],
    ) -> StageTerminalCommitResult:
        """Atomically settle the final Fast stage or an earlier cancellation."""
        async with self._connection() as conn:
            async with conn.transaction():
                locked = await conn.fetchrow(
                    _LOCK_PROGRESS_RUN,
                    self._owner_id,
                    self._run_id,
                    self._lease_owner,
                    self._fencing_epoch,
                )
                if locked is None:
                    return StageLeaseLost()
                progress = int(locked["durable_progress_version"])
                if bool(locked["cancel_requested"]):
                    terminal = await finish_fenced_run(
                        conn,
                        owner_id=self._owner_id,
                        run_id=self._run_id,
                        lease_owner=self._lease_owner,
                        fencing_epoch=self._fencing_epoch,
                        status="succeeded",
                        stop_reason=None,
                        result=result,
                        error_kind=None,
                        error_message=None,
                        event_type="done",
                        payload={"status": "succeeded", "result": result},
                        withhold_on_cancel=True,
                        cancel_requested=True,
                    )
                    if (
                        not terminal.committed
                        or terminal.status != "cancelled"
                        or terminal.event_sequence is None
                    ):
                        return StageLeaseLost()
                    return StageTerminalCommit(
                        progress_version=progress,
                        stage_intent_id=stage_intent_id,
                        status="cancelled",
                        terminal_event_sequence=terminal.event_sequence,
                    )
                if progress != expected_progress_version:
                    return StageProgressConflict(
                        expected_progress_version=expected_progress_version,
                        current_progress_version=progress,
                    )
                existing = await conn.fetchrow(
                    "SELECT state_digest FROM dlightrag_answer_run_stages"
                    " WHERE owner_id = $1 AND run_id = $2 AND stage_intent_id = $3"
                    " FOR UPDATE",
                    self._owner_id,
                    self._run_id,
                    _uuid(stage_intent_id.value),
                )
                state_json = json.dumps(state, ensure_ascii=False, sort_keys=True)
                state_digest = _sha256(state_json)
                if existing is not None and existing["state_digest"] != state_digest:
                    return StageConflict(stage_intent_id=stage_intent_id)
                if existing is None:
                    await conn.execute(
                        "INSERT INTO dlightrag_answer_run_stages ("
                        " owner_id, run_id, stage_intent_id, stage_name,"
                        " progress_version, state, state_digest)"
                        " VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)",
                        self._owner_id,
                        self._run_id,
                        _uuid(stage_intent_id.value),
                        "final_generation",
                        expected_progress_version,
                        state_json,
                        state_digest,
                    )
                terminal = await finish_fenced_run(
                    conn,
                    owner_id=self._owner_id,
                    run_id=self._run_id,
                    lease_owner=self._lease_owner,
                    fencing_epoch=self._fencing_epoch,
                    status="succeeded",
                    stop_reason=None,
                    result=result,
                    error_kind=None,
                    error_message=None,
                    event_type="done",
                    payload={"status": "succeeded", "result": result},
                    withhold_on_cancel=True,
                    cancel_requested=False,
                )
                if (
                    not terminal.committed
                    or terminal.status != "succeeded"
                    or terminal.event_sequence is None
                ):
                    return StageLeaseLost()
                await conn.execute(
                    "UPDATE dlightrag_runs SET"
                    " durable_progress_version = durable_progress_version + 1"
                    " WHERE owner_id = $1 AND run_id = $2",
                    self._owner_id,
                    self._run_id,
                )
                return StageTerminalCommit(
                    progress_version=expected_progress_version + 1,
                    stage_intent_id=stage_intent_id,
                    status="succeeded",
                    terminal_event_sequence=terminal.event_sequence,
                )

    async def settle_stage(
        self,
        *,
        expected_progress_version: int,
        stage_intent_id: StageIntentId,
        stage_name: str,
        state: JsonValue,
        evidence: Sequence[Any],
    ) -> StageCommitResult:
        async with self._connection() as conn:
            try:
                async with conn.transaction():
                    locked = await conn.fetchrow(
                        _LOCK_PROGRESS_RUN,
                        self._owner_id,
                        self._run_id,
                        self._lease_owner,
                        self._fencing_epoch,
                    )
                    if locked is None:
                        return StageLeaseLost()
                    progress = int(locked["durable_progress_version"])
                    if progress != expected_progress_version:
                        return StageProgressConflict(
                            expected_progress_version=expected_progress_version,
                            current_progress_version=progress,
                        )
                    existing = await conn.fetchrow(
                        "SELECT state_digest FROM dlightrag_answer_run_stages"
                        " WHERE owner_id = $1 AND run_id = $2 AND stage_intent_id = $3"
                        " FOR UPDATE",
                        self._owner_id,
                        self._run_id,
                        _uuid(stage_intent_id.value),
                    )
                    state_json = json.dumps(state, ensure_ascii=False, sort_keys=True)
                    state_digest = _sha256(state_json)
                    if existing is not None:
                        if existing["state_digest"] != state_digest:
                            return StageConflict(stage_intent_id=stage_intent_id)
                        return StageCommit(
                            progress_version=expected_progress_version,
                            stage_intent_id=stage_intent_id,
                            evidence_count=0,
                        )
                    await _write_evidence_identities(
                        conn,
                        owner_id=self._owner_id,
                        run_id=self._run_id,
                        writes=evidence,
                    )
                    await conn.execute(
                        "INSERT INTO dlightrag_answer_run_stages ("
                        " owner_id, run_id, stage_intent_id, stage_name,"
                        " progress_version, state, state_digest)"
                        " VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)",
                        self._owner_id,
                        self._run_id,
                        _uuid(stage_intent_id.value),
                        stage_name,
                        expected_progress_version,
                        state_json,
                        state_digest,
                    )
                    await conn.execute(_ADVANCE_PROGRESS, self._owner_id, self._run_id)
                    return StageCommit(
                        progress_version=expected_progress_version + 1,
                        stage_intent_id=stage_intent_id,
                        evidence_count=len(evidence),
                    )
            except _EvidenceIdentityConflict:
                return StageEvidenceConflict()


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _memory_event_payload(
    session_id: SessionId,
    intent_id: IntentId,
    operation: MemoryOperationSettlement,
) -> dict[str, Any]:
    return {
        "body": operation.body,
        "change_id": operation.change_id,
        "intent_id": intent_id.value,
        "kind": operation.kind,
        "memory_ids": list(operation.memory_ids),
        "operation": operation.operation,
        "outcome": operation.outcome,
        "session_id": session_id.value,
        "supersedes_id": operation.supersedes_id,
        "target_change_id": operation.target_change_id,
    }


def _host_update_digest(update: EffectHostUpdate) -> str:
    payload = {
        "evidence": [
            {
                "session_id": write.session_id,
                "intent_id": write.intent_id,
                "result_ordinal": write.result_ordinal,
                "content_digest": write.content_digest,
                "locator_digest": write.locator_digest,
            }
            for write in update.evidence
        ],
        "resources": [
            {
                "resource_id": resource.resource_id,
                "locator_digest": resource.locator_digest,
            }
            for resource in update.resources
        ],
        "fetched": [
            {
                "resource_id": item.resource.resource_id,
                "blob_digest": item.resource.blob_digest,
                "source_locator_digest": item.resource.source_locator_digest,
            }
            for item in update.fetched
        ],
        "committed_outputs": [
            {
                "resource_id": output.resource_id,
                "content_digest": output.content_digest,
                "size_bytes": output.size_bytes,
            }
            for output in update.committed_outputs
        ],
        "workspace_inventory": (
            None
            if update.workspace_inventory is None
            else {
                "replace_all": update.workspace_inventory.replace_all,
                "upserts": [record.relative_path for record in update.workspace_inventory.upserts],
                "deletes": list(update.workspace_inventory.deletes),
            }
        ),
        "artifact_attachment": (
            None
            if update.artifact_attachment is None
            else {
                "relative_path": update.artifact_attachment.relative_path,
                "label": update.artifact_attachment.label,
                "content_digest": update.artifact_attachment.content_digest,
                "size_bytes": update.artifact_attachment.size_bytes,
                "presentation": update.artifact_attachment.presentation,
                "session_id": update.artifact_attachment.session_id,
                "intent_id": update.artifact_attachment.intent_id,
            }
        ),
        "memory_operation": (
            None
            if update.memory_operation is None
            else {
                "operation": update.memory_operation.operation,
                "outcome": update.memory_operation.outcome,
                "change_id": update.memory_operation.change_id,
                "memory_ids": list(update.memory_operation.memory_ids),
                "kind": update.memory_operation.kind,
                "body": update.memory_operation.body,
                "supersedes_id": update.memory_operation.supersedes_id,
                "target_change_id": update.memory_operation.target_change_id,
            }
        ),
    }
    return _sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _json_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        loaded = json.loads(value)
        return dict(loaded) if isinstance(loaded, dict) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _decode_entry(row: Any, *, session_id: SessionId) -> SessionEntry:
    from dlightrag.engine.agent.session.entries import ENTRY_TYPE_TO_CLASS, decode_entry_payload
    from dlightrag.engine.agent.session.ids import EntryId

    entry_class = ENTRY_TYPE_TO_CLASS.get(str(row["entry_type"]))
    if entry_class is None:
        raise ValueError(f"unknown Session Entry type in storage: {row['entry_type']}")
    payload = _json_payload(row["payload_json"])
    return decode_entry_payload(
        entry_type=str(row["entry_type"]),
        entry_id=EntryId(str(row["entry_id"])),
        session_id=session_id,
        sequence=int(row["sequence"]),
        timestamp=row["timestamp"],
        payload=payload,
        parent_entry_id=(
            EntryId(str(row["parent_entry_id"])) if row["parent_entry_id"] is not None else None
        ),
    )


__all__ = [
    "PGAgentSessionRepository",
    "PGProgressStore",
]
