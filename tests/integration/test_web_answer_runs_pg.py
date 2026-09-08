# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for run-linked Web conversations on PostgreSQL 18.

Exercises the real contract against a live database: the baseline schema and its
foreign keys, the single transaction that creates a run, its uploaded bytes, and
its conversation turn, owner-wide submission idempotency and its conflicts,
concurrent replay, conversation deletion with ownership-safe artifact cleanup,
the unified run/Session retention floor, and the cascade that removes a pruned
run's visible turn while leaving only empty conversation navigation state.

Every test runs inside a throwaway database created and dropped per test, so the
configured administrative database is never mutated. Connection settings come
from the shared integration-test PostgreSQL environment; skipped if unavailable.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.runtime.run_store import (
    PGRunStore,
)
from dlightrag.adapters.postgres.web.web_conversations import (
    PGWebConversationStore,
)
from dlightrag.application.web_conversations import (
    ConversationCursor,
    ConversationHistoryPageRequest,
    ConversationPageRequest,
    ConversationSubmissionConflict,
)
from dlightrag.engine.answer.runs.envelope import accepted_input_envelope
from dlightrag.engine.runtime.records import (
    IdempotencyKeyConflict,
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
    RunAccessScope,
    RunDeletion,
    artifact_digest,
    run_request_fingerprint,
)
from tests.conftest import FingerprintingRunStore
from tests.integration.pg_conn import PG_CONN_KWARGS

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_PG_CONN_KWARGS: dict[str, Any] = PG_CONN_KWARGS

_OWNER = "principal-alpha"
_OTHER_OWNER = "principal-beta"
_MAX_TURNS = 100


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def db_name() -> AsyncIterator[str]:
    """Provision an isolated throwaway database and yield its name."""
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")

    name = f"dlightrag_weblink_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    try:
        yield name
    finally:
        admin = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        finally:
            await admin.close()


@pytest.fixture
async def pool(db_name: str) -> AsyncIterator[Any]:
    """Yield a pool bound to this test's throwaway database."""
    created = await asyncpg.create_pool(
        **{**_PG_CONN_KWARGS, "database": db_name}, min_size=1, max_size=8
    )
    try:
        yield created
    finally:
        await created.close()


@pytest.fixture
async def runs(pool: Any) -> FingerprintingRunStore:
    store = FingerprintingRunStore(pool=pool)
    await store.initialize()
    return store


@pytest.fixture
async def store(pool: Any, runs: FingerprintingRunStore) -> PGWebConversationStore:
    created = PGWebConversationStore(pool=pool, run_store=runs)
    await created.initialize()
    return created


@dataclass
class _DeletionMetrics:
    queries: int = 0
    outer_transactions: int = 0
    savepoints: int = 0
    max_fetched_rows: int = 0


class _MeasuredTransaction:
    def __init__(self, conn: _MeasuredConnection, transaction: Any) -> None:
        self._conn = conn
        self._transaction = transaction

    async def __aenter__(self) -> Any:
        outer = self._conn.transaction_depth == 0
        entered = await self._transaction.__aenter__()
        self._conn.transaction_depth += 1
        if outer:
            self._conn.metrics.outer_transactions += 1
        else:
            self._conn.metrics.savepoints += 1
        return entered

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        try:
            return await self._transaction.__aexit__(exc_type, exc, traceback)
        finally:
            self._conn.transaction_depth -= 1


class _MeasuredConnection:
    def __init__(self, conn: Any, metrics: _DeletionMetrics) -> None:
        self._conn = conn
        self.metrics = metrics
        self.transaction_depth = 0

    def transaction(self, *args: Any, **kwargs: Any) -> _MeasuredTransaction:
        return _MeasuredTransaction(self, self._conn.transaction(*args, **kwargs))

    def _query(self, rows: int = 0) -> None:
        self.metrics.queries += 1
        self.metrics.max_fetched_rows = max(self.metrics.max_fetched_rows, rows)

    async def fetch(self, query: str, *args: Any) -> Any:
        rows = await self._conn.fetch(query, *args)
        self._query(len(rows))
        return rows

    async def fetchrow(self, query: str, *args: Any) -> Any:
        row = await self._conn.fetchrow(query, *args)
        self._query(1 if row is not None else 0)
        return row

    async def fetchval(self, query: str, *args: Any) -> Any:
        value = await self._conn.fetchval(query, *args)
        self._query(1 if value is not None else 0)
        return value

    async def execute(self, query: str, *args: Any) -> Any:
        value = await self._conn.execute(query, *args)
        self._query()
        return value


class _MeasuredAcquire:
    def __init__(self, acquire: Any, metrics: _DeletionMetrics) -> None:
        self._acquire = acquire
        self._metrics = metrics

    async def __aenter__(self) -> _MeasuredConnection:
        conn = await self._acquire.__aenter__()
        return _MeasuredConnection(conn, self._metrics)

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        return await self._acquire.__aexit__(exc_type, exc, traceback)


class _MeasuredPool:
    def __init__(self, pool: Any, metrics: _DeletionMetrics) -> None:
        self._pool = pool
        self._metrics = metrics

    def acquire(self) -> _MeasuredAcquire:
        return _MeasuredAcquire(self._pool.acquire(), self._metrics)


class _MeasuredRunStore(PGRunStore):
    def __init__(self, *, pool: Any, fail_on_call: int | None = None) -> None:
        super().__init__(pool=pool)
        self.fail_on_call = fail_on_call
        self.delete_calls = 0
        self.max_run_ids = 0

    async def delete_runs_in(
        self,
        conn: Any,
        *,
        owner_id: str,
        run_ids: Sequence[str],
    ) -> RunDeletion:
        self.delete_calls += 1
        self.max_run_ids = max(self.max_run_ids, len(run_ids))
        if self.delete_calls == self.fail_on_call:
            raise RuntimeError("injected deletion batch failure")
        return await super().delete_runs_in(conn, owner_id=owner_id, run_ids=run_ids)


def _measured_store(
    pool: Any,
    *,
    fail_on_call: int | None = None,
) -> tuple[PGWebConversationStore, _MeasuredRunStore, _DeletionMetrics]:
    metrics = _DeletionMetrics()
    measured_runs = _MeasuredRunStore(pool=pool, fail_on_call=fail_on_call)
    measured = PGWebConversationStore(
        pool=_MeasuredPool(pool, metrics),
        run_store=measured_runs,
    )
    measured._initialized = True  # pyright: ignore[reportPrivateUsage]
    return measured, measured_runs, metrics


def _request(query: str = "why", **extra: Any) -> dict[str, Any]:
    return {
        "query": query,
        "workspaces": ["alpha"],
        "history": [],
        "agent_session_id": "00000000-0000-7000-8000-000000000001",
        "agent_lane_id": "main",
        **extra,
    }


def _turn_envelope(
    *, owner: str, request: dict[str, Any], submission_id: str, fingerprint: str
) -> PreparedRunEnvelope:
    return PreparedRunEnvelope(
        run_kind="answer",
        lane="query",
        submitted_by=owner,
        access_scope=RunAccessScope(kind="owner", scope_id=owner),
        submission_key=submission_id,
        request_fingerprint=fingerprint,
        payload=request,
        accepted_input=accepted_input_envelope(request),
        retention_seconds=365 * 24 * 60 * 60,
    )


async def _conversation(store: PGWebConversationStore, owner: str = _OWNER) -> str:
    row = await store.create_conversation(owner)
    return str(row["conversation_id"])


async def _submit(
    store: PGWebConversationStore,
    conversation_id: str,
    *,
    owner: str = _OWNER,
    submission_id: str | None = None,
    request: dict[str, Any] | None = None,
    artifacts: list[PendingArtifact] | None = None,
    references: list[PendingArtifactReference] | None = None,
    idempotency_fingerprint: str | None = None,
    create_conversation: bool = False,
):
    effective_request = {
        **(request if request is not None else _request()),
        "agent_session_id": conversation_id,
    }
    effective_submission_id = submission_id or str(uuid.uuid4())
    fingerprint = (
        idempotency_fingerprint
        if idempotency_fingerprint is not None
        else run_request_fingerprint(effective_request)
    )
    return await store.create_answer_turn(
        principal_id=owner,
        conversation_id=conversation_id,
        submission_id=effective_submission_id,
        envelope=_turn_envelope(
            owner=owner,
            request=effective_request,
            submission_id=effective_submission_id,
            fingerprint=fingerprint,
        ),
        run_id=str(uuid.uuid7()),
        artifacts=artifacts or [],
        references=references or [],
        title_hint="why",
        create_conversation=create_conversation,
    )


async def _finish(pool: Any, run_id: str, *, status: str, error: str | None = None) -> None:
    """Drive one run to a terminal state the way its worker eventually would."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs "
            "SET status = $2, finished_at = NOW(), prepared_input_json = NULL, "
            "    result_json = CASE WHEN $2 = 'succeeded' "
            '        THEN \'{"answer": "done"}\'::jsonb ELSE NULL END, '
            "    error_kind = $3, error_message = $3 "
            "WHERE run_id = $1",
            uuid.UUID(run_id),
            status,
            error,
        )


async def _backdate_finish(pool: Any, run_id: str, *, days: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs "
            "SET finished_at = NOW() - ($2 * INTERVAL '1 day'), "
            "purge_after = NOW() - ($2 * INTERVAL '1 day') "
            "+ make_interval(secs => retention_seconds::double precision) WHERE run_id = $1",
            uuid.UUID(run_id),
            days,
        )


async def _count(pool: Any, table: str, **where: Any) -> int:
    clause = " AND ".join(f"{column} = ${index}" for index, column in enumerate(where, start=1))
    async with pool.acquire() as conn:
        return int(
            await conn.fetchval(
                f"SELECT count(*)::int FROM {table}"  # noqa: S608 - fixed table/column names
                + (f" WHERE {clause}" if clause else ""),
                *where.values(),
            )
        )


async def _bulk_linked_runs(
    conn: Any,
    conversation_id: str,
    total: int,
) -> None:
    await conn.execute(
        """
        INSERT INTO dlightrag_runs (
            owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind,
            submission_key, prepared_input_json, accepted_input_json,
            request_fingerprint, retention_seconds
        )
        SELECT $1, md5('run:' || series::text)::uuid, 'answer', 'query', $1, 'owner',
               md5('submission:' || series::text), '{}'::jsonb, '{}'::jsonb,
               md5('fingerprint:' || series::text), 31536000
        FROM generate_series(1, $2) AS series
        """,
        _OWNER,
        total,
    )
    await conn.execute(
        """
        INSERT INTO web_conversation_turns (
            turn_id, principal_id, conversation_id, turn_number,
            submission_id, answer_run_id
        )
        SELECT md5('turn:' || series::text)::uuid, $1, $2::uuid, series,
               md5('submission:' || series::text)::uuid,
               md5('run:' || series::text)::uuid
        FROM generate_series(1, $3) AS series
        """,
        _OWNER,
        conversation_id,
        total,
    )


# ---------------------------------------------------------------------------
# One transaction
# ---------------------------------------------------------------------------


async def test_a_submission_commits_the_run_bytes_and_turn_together(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    content = b"chart-bytes"
    digest = artifact_digest(content)

    creation = await _submit(
        store,
        conversation_id,
        artifacts=[PendingArtifact(content=content)],
        references=[
            PendingArtifactReference(
                resource_id="attachment-1",
                reference_kind="current_attachment",
                ordinal=1,
                digest=digest,
                filename="chart.png",
                mime_type="image/png",
            )
        ],
    )

    assert creation is not None
    assert creation.replayed is False
    assert creation.turn.turn_number == 1
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT t.answer_run_id::text AS run_id, r.status "
            "FROM web_conversation_turns AS t "
            "JOIN dlightrag_runs AS r "
            "  ON r.owner_id = t.principal_id AND r.run_id = t.answer_run_id "
            "WHERE t.turn_id = $1::text::uuid",
            creation.turn.turn_id,
        )
    assert row["run_id"] == creation.turn.answer_run_id
    assert row["status"] == "queued"
    assert await _count(pool, "dlightrag_blobs", owner_id=_OWNER, digest=digest) == 1
    assert await _count(pool, "dlightrag_answer_run_artifacts", owner_id=_OWNER) == 1


async def test_first_submission_creates_conversation_run_and_turn_together(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = str(uuid.uuid4())

    creation = await _submit(
        store,
        conversation_id,
        create_conversation=True,
    )

    assert creation is not None
    assert creation.summary["conversation_id"] == conversation_id
    assert creation.turn.turn_number == 1
    assert await _count(pool, "web_conversations") == 1
    assert await _count(pool, "dlightrag_runs") == 1
    assert await _count(pool, "web_conversation_turns") == 1


async def test_concurrent_first_submission_replays_one_atomic_conversation(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = str(uuid.uuid4())
    submission_id = str(uuid.uuid4())

    results = await asyncio.gather(
        *(
            _submit(
                store,
                conversation_id,
                submission_id=submission_id,
                create_conversation=True,
            )
            for _ in range(5)
        )
    )

    accepted = [result for result in results if result is not None]
    assert len(accepted) == 5
    assert {result.summary["conversation_id"] for result in accepted} == {conversation_id}
    assert len({result.turn.answer_run_id for result in accepted}) == 1
    assert await _count(pool, "web_conversations") == 1
    assert await _count(pool, "dlightrag_runs") == 1
    assert await _count(pool, "web_conversation_turns") == 1


async def test_a_submission_to_an_unknown_conversation_writes_nothing(
    store: PGWebConversationStore, pool: Any
) -> None:
    creation = await _submit(store, str(uuid.uuid4()))

    assert creation is None
    assert await _count(pool, "dlightrag_runs") == 0
    assert await _count(pool, "web_conversation_turns") == 0


async def test_a_foreign_conversation_is_never_written_to(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store, _OTHER_OWNER)

    creation = await _submit(store, conversation_id, owner=_OWNER)

    assert creation is None
    assert await _count(pool, "dlightrag_runs") == 0


# ---------------------------------------------------------------------------
# Owner-wide submission idempotency
# ---------------------------------------------------------------------------


async def test_forked_conversation_maps_to_a_new_lane_in_the_parent_session(
    store: PGWebConversationStore,
    pool: Any,
) -> None:
    parent_conversation = await _conversation(store)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions"
            " (owner_id, session_id, lease_run_id, commit_sequence, fencing_epoch)"
            " VALUES ($1, $2, $2, 0, 1)",
            _OWNER,
            uuid.UUID(parent_conversation),
        )
    parent_seed = await store.submission_seed(_OWNER, parent_conversation, attachment_limit=0)
    assert parent_seed is not None
    parent = parent_seed.head
    branch_conversation = str(uuid.uuid4())
    branch_lane = str(uuid.uuid4())
    request = {
        **_request("branch question"),
        "agent_session_id": parent.agent_session_id,
        "agent_lane_id": branch_lane,
        "source_lane_id": parent.agent_lane_id,
        "parent_run_id": str(uuid.uuid4()),
        "continuation_kind": "fork",
    }
    submission_id = str(uuid.uuid4())
    created = await store.create_answer_turn(
        principal_id=_OWNER,
        conversation_id=branch_conversation,
        submission_id=submission_id,
        envelope=_turn_envelope(
            owner=_OWNER,
            request=request,
            submission_id=submission_id,
            fingerprint=run_request_fingerprint(request),
        ),
        run_id=str(uuid.uuid7()),
        artifacts=(),
        references=(),
        title_hint="branch",
        create_conversation=True,
        forked_from_conversation_id=parent_conversation,
    )
    assert created is not None
    branch_seed = await store.submission_seed(_OWNER, branch_conversation, attachment_limit=0)
    assert branch_seed is not None
    branch = branch_seed.head
    assert branch.agent_session_id == parent.agent_session_id
    assert branch.agent_lane_id == branch_lane
    assert branch.agent_lane_id != parent.agent_lane_id


async def test_replaying_a_submission_returns_the_same_run_and_turn(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    submission_id = str(uuid.uuid4())

    first = await _submit(store, conversation_id, submission_id=submission_id)
    second = await _submit(store, conversation_id, submission_id=submission_id)

    assert first is not None and second is not None
    assert second.replayed is True
    assert second.turn.turn_id == first.turn.turn_id
    assert second.turn.answer_run_id == first.turn.answer_run_id
    assert await _count(pool, "dlightrag_runs") == 1
    assert await _count(pool, "web_conversation_turns") == 1


async def test_public_fingerprint_controls_atomic_web_replay(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    submission_id = str(uuid.uuid4())
    public_fingerprint = run_request_fingerprint(_request())

    first = await _submit(
        store,
        conversation_id,
        submission_id=submission_id,
        request=_request(pinned_models=[{"revision": "old"}]),
        idempotency_fingerprint=public_fingerprint,
    )
    replay = await _submit(
        store,
        conversation_id,
        submission_id=submission_id,
        request=_request(pinned_models=[{"revision": "new"}]),
        idempotency_fingerprint=public_fingerprint,
    )

    assert first is not None and replay is not None
    assert replay.replayed is True
    assert replay.turn.answer_run_id == first.turn.answer_run_id
    with pytest.raises(ConversationSubmissionConflict):
        await _submit(
            store,
            conversation_id,
            submission_id=submission_id,
            request=_request(pinned_models=[{"revision": "old"}]),
            idempotency_fingerprint=run_request_fingerprint(_request("changed")),
        )
    assert await _count(pool, "dlightrag_runs") == 1


async def test_concurrent_identical_submissions_create_exactly_one_turn(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    submission_id = str(uuid.uuid4())

    results = await asyncio.gather(
        *(_submit(store, conversation_id, submission_id=submission_id) for _ in range(5))
    )

    run_ids = {result.turn.answer_run_id for result in results if result is not None}
    assert len(run_ids) == 1
    assert await _count(pool, "dlightrag_runs") == 1
    assert await _count(pool, "web_conversation_turns") == 1


async def test_reusing_a_submission_with_different_input_is_a_conflict(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    submission_id = str(uuid.uuid4())
    await _submit(store, conversation_id, submission_id=submission_id)

    with pytest.raises(ConversationSubmissionConflict):
        await _submit(
            store,
            conversation_id,
            submission_id=submission_id,
            request=_request("a different question"),
        )

    assert await _count(pool, "dlightrag_runs") == 1


async def test_reusing_a_submission_in_another_conversation_is_a_conflict(
    store: PGWebConversationStore, pool: Any
) -> None:
    first_conversation = await _conversation(store)
    second_conversation = await _conversation(store)
    submission_id = str(uuid.uuid4())
    await _submit(store, first_conversation, submission_id=submission_id)

    with pytest.raises(ConversationSubmissionConflict):
        await _submit(store, second_conversation, submission_id=submission_id)

    assert await _count(pool, "web_conversation_turns") == 1


async def test_the_same_submission_in_two_conversations_at_once_is_a_conflict(
    store: PGWebConversationStore, pool: Any
) -> None:
    """A race loses to the unique key, and losing is a conflict, not a server fault.

    Both callers see no existing turn and both replay the one run the submission
    id owns, so the second turn insert is what discovers the reuse.
    """
    first_conversation = await _conversation(store)
    second_conversation = await _conversation(store)
    submission_id = str(uuid.uuid4())
    content = b"chart-bytes"
    references = [
        PendingArtifactReference(
            resource_id="attachment-1",
            reference_kind="current_attachment",
            ordinal=1,
            digest=artifact_digest(content),
            filename="chart.png",
            mime_type="image/png",
        )
    ]

    outcomes = await asyncio.gather(
        *(
            _submit(
                store,
                conversation_id,
                submission_id=submission_id,
                artifacts=[PendingArtifact(content=content)],
                references=list(references),
            )
            for conversation_id in (first_conversation, second_conversation)
        ),
        return_exceptions=True,
    )

    accepted = [item for item in outcomes if not isinstance(item, BaseException)]
    rejected = [item for item in outcomes if isinstance(item, BaseException)]
    assert len(accepted) == 1
    assert [type(error) for error in rejected] == [ConversationSubmissionConflict]
    assert await _count(pool, "dlightrag_runs") == 1
    assert await _count(pool, "web_conversation_turns") == 1
    assert await _count(pool, "dlightrag_blobs") == 1
    assert await _count(pool, "dlightrag_answer_run_artifacts") == 1


async def test_the_submission_key_is_owner_wide_not_conversation_scoped(
    store: PGWebConversationStore, runs: FingerprintingRunStore
) -> None:
    """The run's idempotency key is the submission id in the owner's namespace."""
    conversation_id = await _conversation(store)
    submission_id = str(uuid.uuid4())

    creation = await _submit(store, conversation_id, submission_id=submission_id)

    assert creation is not None
    record = await runs.get_run(owner_id=_OWNER, run_id=creation.turn.answer_run_id)
    assert record is not None
    assert record.submission_key == submission_id


async def test_two_principals_may_use_the_same_submission_id(
    store: PGWebConversationStore, pool: Any
) -> None:
    submission_id = str(uuid.uuid4())
    mine = await _conversation(store, _OWNER)
    theirs = await _conversation(store, _OTHER_OWNER)

    await _submit(store, mine, owner=_OWNER, submission_id=submission_id)
    await _submit(store, theirs, owner=_OTHER_OWNER, submission_id=submission_id)

    assert await _count(pool, "web_conversation_turns") == 2
    assert await _count(pool, "dlightrag_runs") == 2


async def test_a_run_created_outside_a_conversation_keeps_the_same_key_namespace(
    store: PGWebConversationStore, runs: FingerprintingRunStore
) -> None:
    conversation_id = await _conversation(store)
    submission_id = str(uuid.uuid4())
    await _submit(store, conversation_id, submission_id=submission_id)

    other_request = _request("something else")
    with pytest.raises(IdempotencyKeyConflict):
        await runs.create_run(
            owner_id=_OWNER,
            prepared_input=other_request,
            idempotency_fingerprint=run_request_fingerprint(other_request),
            idempotency_key=submission_id,
        )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


async def test_keyset_pages_traverse_ten_thousand_rows_once_with_bounded_fetches(
    store: PGWebConversationStore,
    pool: Any,
) -> None:
    total = 10_037
    foreign_total = 137
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO web_conversations (
                principal_id, conversation_id, agent_session_id, agent_lane_id,
                created_at, updated_at
            )
            SELECT
                $1,
                md5($1 || ':' || series::text)::uuid,
                md5($1 || ':' || series::text)::uuid,
                'main',
                TIMESTAMPTZ '2025-01-01 00:00:00+00',
                TIMESTAMPTZ '2026-01-01 00:00:00+00'
                    - ((series - 1) * INTERVAL '1 microsecond')
            FROM generate_series(1, $2) AS series
            """,
            _OWNER,
            total,
        )
        await conn.execute(
            """
            INSERT INTO web_conversations (
                principal_id, conversation_id, agent_session_id, agent_lane_id,
                created_at, updated_at
            )
            SELECT
                $1,
                md5($1 || ':' || series::text)::uuid,
                md5($1 || ':' || series::text)::uuid,
                'main',
                TIMESTAMPTZ '2025-01-01 00:00:00+00',
                TIMESTAMPTZ '2027-01-01 00:00:00+00'
                    - (series * INTERVAL '1 microsecond')
            FROM generate_series(1, $2) AS series
            """,
            _OTHER_OWNER,
            foreign_total,
        )
        expected_owner = [
            str(row["conversation_id"])
            for row in await conn.fetch(
                "SELECT conversation_id FROM web_conversations "
                "WHERE principal_id = $1 "
                "ORDER BY updated_at DESC, conversation_id DESC",
                _OWNER,
            )
        ]

    limit = 73
    request = ConversationPageRequest(limit=limit)
    seen: list[str] = []
    fetched_counts: list[int] = []
    page_sizes: list[int] = []
    while True:
        result = await store.list_conversations(_OWNER, page=request)
        fetched_counts.append(result.fetched_rows)
        page_sizes.append(len(result.items))
        seen.extend(str(row["conversation_id"]) for row in result.items)
        assert len(result.items) <= limit
        assert result.fetched_rows <= limit + 1
        if not result.has_more:
            break
        last = result.items[-1]
        request = ConversationPageRequest(
            limit=limit,
            cursor=ConversationCursor(
                updated_at=last["updated_at"],
                conversation_id=uuid.UUID(str(last["conversation_id"])),
            ),
        )

    assert seen == expected_owner
    assert len(seen) == total
    assert len(set(seen)) == total
    assert page_sizes[0] == limit
    assert page_sizes[len(page_sizes) // 2] == limit
    assert 0 < page_sizes[-1] <= limit
    assert fetched_counts[0] == limit + 1
    assert fetched_counts[len(fetched_counts) // 2] == limit + 1
    assert fetched_counts[-1] == page_sizes[-1]
    foreign = await store.list_conversations(
        _OTHER_OWNER,
        page=ConversationPageRequest(limit=100),
    )
    assert len(foreign.items) == 100
    assert not set(seen).intersection(str(row["conversation_id"]) for row in foreign.items)


async def test_identical_timestamp_ties_paginate_by_uuid_desc_exactly_once(
    store: PGWebConversationStore,
    pool: Any,
) -> None:
    total = 1_001
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO web_conversations (
                principal_id, conversation_id, agent_session_id, agent_lane_id,
                created_at, updated_at
            )
            SELECT
                $1,
                md5('tie:' || series::text)::uuid,
                md5('tie:' || series::text)::uuid,
                'main',
                TIMESTAMPTZ '2026-01-01 00:00:00+00',
                TIMESTAMPTZ '2026-01-02 00:00:00+00'
            FROM generate_series(1, $2) AS series
            """,
            _OWNER,
            total,
        )
        expected = [
            str(row["conversation_id"])
            for row in await conn.fetch(
                "SELECT conversation_id FROM web_conversations "
                "WHERE principal_id = $1 ORDER BY conversation_id DESC",
                _OWNER,
            )
        ]

    seen: list[str] = []
    request = ConversationPageRequest(limit=37)
    while True:
        result = await store.list_conversations(_OWNER, page=request)
        seen.extend(str(row["conversation_id"]) for row in result.items)
        assert result.fetched_rows <= 38
        if not result.has_more:
            break
        last = result.items[-1]
        request = ConversationPageRequest(
            limit=37,
            cursor=ConversationCursor(
                updated_at=last["updated_at"],
                conversation_id=uuid.UUID(str(last["conversation_id"])),
            ),
        )

    assert seen == expected
    assert len(set(seen)) == total


async def test_keyset_traversal_has_standard_concurrent_touch_and_create_semantics(
    store: PGWebConversationStore,
    pool: Any,
) -> None:
    """Rows moved above an established cursor are skipped, never duplicated."""
    ids = [str(uuid.uuid4()) for _ in range(3)]
    async with pool.acquire() as conn:
        for position, conversation_id in enumerate(ids):
            await conn.execute(
                "INSERT INTO web_conversations ("
                "principal_id, conversation_id, agent_session_id, agent_lane_id, updated_at"
                ") VALUES ($1, $2::uuid, $2::uuid, 'main', "
                "TIMESTAMPTZ '2026-01-03 00:00:00+00' - ($3 * INTERVAL '1 day'))",
                _OWNER,
                conversation_id,
                position,
            )

    first = await store.list_conversations(
        _OWNER,
        page=ConversationPageRequest(limit=2),
    )
    assert first.has_more
    first_ids = [str(row["conversation_id"]) for row in first.items]
    last = first.items[-1]
    cursor = ConversationCursor(
        updated_at=last["updated_at"],
        conversation_id=uuid.UUID(str(last["conversation_id"])),
    )
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE web_conversations SET updated_at = "
            "TIMESTAMPTZ '2027-01-01 00:00:00+00' "
            "WHERE principal_id = $1 AND conversation_id = $2::uuid",
            _OWNER,
            ids[-1],
        )
        created = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO web_conversations ("
            "principal_id, conversation_id, agent_session_id, agent_lane_id, updated_at"
            ") VALUES ($1, $2::uuid, $2::uuid, 'main', "
            "TIMESTAMPTZ '2027-01-02 00:00:00+00')",
            _OWNER,
            created,
        )

    second = await store.list_conversations(
        _OWNER,
        page=ConversationPageRequest(limit=2, cursor=cursor),
    )
    second_ids = [str(row["conversation_id"]) for row in second.items]

    assert not set(first_ids).intersection(second_ids)
    assert ids[-1] not in second_ids
    assert created not in second_ids


async def test_a_snapshot_projects_each_turn_from_its_run(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    pending = await _submit(store, conversation_id, request=_request("first"))
    done = await _submit(store, conversation_id, request=_request("second"))
    assert pending is not None and done is not None
    await _finish(pool, done.turn.answer_run_id, status="succeeded")

    page = await store.history_page(
        _OWNER,
        conversation_id,
        page=ConversationHistoryPageRequest(limit=_MAX_TURNS),
    )

    assert page is not None
    assert [turn.turn_number for turn in page.turns] == [1, 2]
    assert [turn.run.status for turn in page.turns] == ["queued", "succeeded"]
    assert page.turns[0].run.request_input()["query"] == "first"
    assert page.turns[1].run.result == {"answer": "done"}


async def test_a_run_is_only_findable_by_its_owner(store: PGWebConversationStore) -> None:
    conversation_id = await _conversation(store)
    creation = await _submit(store, conversation_id)
    assert creation is not None

    assert await store.find_turn_by_run(_OWNER, creation.turn.answer_run_id) is not None
    assert await store.find_turn_by_run(_OTHER_OWNER, creation.turn.answer_run_id) is None


# ---------------------------------------------------------------------------
# Deletion and retention
# ---------------------------------------------------------------------------


async def _hold_conversation_lock(conn: Any, conversation_id: str) -> Any:
    """Take the lock a submission holds, so a deleter has to wait behind it."""
    transaction = conn.transaction()
    await transaction.start()
    await conn.fetchrow(
        "SELECT 1 FROM web_conversations "
        "WHERE principal_id = $1 AND conversation_id = $2::text::uuid FOR UPDATE",
        _OWNER,
        conversation_id,
    )
    return transaction


async def _link_turn(conn: Any, runs: FingerprintingRunStore, conversation_id: str) -> str:
    """Create the run and its turn the way an accepted submission would."""
    request = _request("late")
    creation = await runs.create_run_in(
        conn,
        owner_id=_OWNER,
        request=request,
        idempotency_fingerprint=run_request_fingerprint(request),
    )
    await conn.execute(
        "INSERT INTO web_conversation_turns "
        "(turn_id, principal_id, conversation_id, turn_number, submission_id, answer_run_id) "
        "VALUES ($1, $2, $3::text::uuid, 1, $4, $5::text::uuid)",
        uuid.uuid4(),
        _OWNER,
        conversation_id,
        uuid.uuid4(),
        creation.run.run_id,
    )
    return creation.run.run_id


async def test_deleting_a_conversation_never_orphans_a_turn_committed_behind_it(
    store: PGWebConversationStore, runs: FingerprintingRunStore, pool: Any
) -> None:
    """The deleter must snapshot run ids under the conversation lock, not before it."""
    conversation_id = await _conversation(store)

    async with pool.acquire() as holder:
        transaction = await _hold_conversation_lock(holder, conversation_id)
        deletion = asyncio.create_task(store.delete_conversation(_OWNER, conversation_id))
        await asyncio.sleep(0.2)
        await _link_turn(holder, runs, conversation_id)
        await transaction.commit()

    assert await deletion is True
    assert await _count(pool, "web_conversation_turns") == 0
    assert await _count(pool, "dlightrag_runs") == 0


async def test_deleting_every_conversation_never_orphans_a_turn_committed_behind_it(
    store: PGWebConversationStore, runs: FingerprintingRunStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)

    async with pool.acquire() as holder:
        transaction = await _hold_conversation_lock(holder, conversation_id)
        deletion = asyncio.create_task(store.delete_all_conversations(_OWNER))
        await asyncio.sleep(0.2)
        await _link_turn(holder, runs, conversation_id)
        await transaction.commit()

    assert await deletion == 1
    assert await _count(pool, "web_conversation_turns") == 0
    assert await _count(pool, "dlightrag_runs") == 0


async def test_two_delete_all_callers_wait_for_a_concurrent_submission_without_deadlock(
    store: PGWebConversationStore, runs: FingerprintingRunStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)

    async with pool.acquire() as holder:
        transaction = await _hold_conversation_lock(holder, conversation_id)
        first = asyncio.create_task(store.delete_all_conversations(_OWNER))
        second = asyncio.create_task(store.delete_all_conversations(_OWNER))
        await asyncio.sleep(0.2)
        late_run_id = await _link_turn(holder, runs, conversation_id)
        await transaction.commit()

    results = await asyncio.wait_for(asyncio.gather(first, second), timeout=10)
    assert sorted(results) == [0, 1]
    assert await runs.get_run(owner_id=_OWNER, run_id=late_run_id) is None
    assert await _count(pool, "web_conversations") == 0
    assert await _count(pool, "web_conversation_turns") == 0
    assert await _count(pool, "dlightrag_runs") == 0


async def test_deleting_a_conversation_deletes_its_runs_and_frees_its_bytes(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    content = b"only-reference"
    digest = artifact_digest(content)
    creation = await _submit(
        store,
        conversation_id,
        artifacts=[PendingArtifact(content=content)],
        references=[
            PendingArtifactReference(
                resource_id="attachment-1",
                reference_kind="current_attachment",
                ordinal=1,
                digest=digest,
                filename="chart.png",
                mime_type="image/png",
            )
        ],
    )
    assert creation is not None
    await _finish(pool, creation.turn.answer_run_id, status="succeeded")

    assert await store.delete_conversation(_OWNER, conversation_id) is True

    assert await _count(pool, "dlightrag_runs") == 0
    assert await _count(pool, "dlightrag_answer_run_artifacts") == 0
    assert await _count(pool, "dlightrag_blobs") == 0
    assert await _count(pool, "web_conversation_turns") == 0


async def test_deletion_keeps_bytes_another_run_still_references(
    store: PGWebConversationStore, pool: Any
) -> None:
    content = b"shared-bytes"
    digest = artifact_digest(content)
    reference = PendingArtifactReference(
        resource_id="attachment-1",
        reference_kind="current_attachment",
        ordinal=1,
        digest=digest,
        filename="chart.png",
        mime_type="image/png",
    )
    doomed = await _conversation(store)
    kept = await _conversation(store)
    await _submit(
        store, doomed, artifacts=[PendingArtifact(content=content)], references=[reference]
    )
    await _submit(store, kept, artifacts=[PendingArtifact(content=content)], references=[reference])

    await store.delete_conversation(_OWNER, doomed)

    assert await _count(pool, "dlightrag_blobs", owner_id=_OWNER, digest=digest) == 1
    assert await _count(pool, "dlightrag_runs") == 1


async def test_deleting_every_conversation_deletes_every_linked_run(
    store: PGWebConversationStore, pool: Any
) -> None:
    mine = await _conversation(store, _OWNER)
    theirs = await _conversation(store, _OTHER_OWNER)
    await _submit(store, mine, owner=_OWNER)
    await _submit(store, theirs, owner=_OTHER_OWNER)

    assert await store.delete_all_conversations(_OWNER) == 1

    assert await _count(pool, "dlightrag_runs", owner_id=_OWNER) == 0
    assert await _count(pool, "dlightrag_runs", owner_id=_OTHER_OWNER) == 1


async def test_ten_thousand_linked_runs_delete_with_a_bounded_client_working_set(
    store: PGWebConversationStore,
    pool: Any,
) -> None:
    total = 10_000
    batch_size = 128
    batch_count = 79
    digest = "a" * 64
    conversation_id = await _conversation(store)
    async with pool.acquire() as conn:
        await _bulk_linked_runs(conn, conversation_id, total)
        await conn.execute(
            "INSERT INTO dlightrag_blobs (owner_id, digest, byte_size) VALUES ($1, $2, 0)",
            _OWNER,
            digest,
        )
        await conn.execute(
            """
            INSERT INTO dlightrag_answer_run_artifacts (
                owner_id, run_id, resource_id, reference_kind, ordinal,
                digest, filename, mime_type
            )
            SELECT $1, md5('run:' || series::text)::uuid, 'attachment',
                   'current_attachment', 0, $2, 'attachment.bin',
                   'application/octet-stream'
            FROM generate_series(1, $3) AS series
            """,
            _OWNER,
            digest,
            total,
        )
    measured, measured_runs, metrics = _measured_store(pool)

    assert await measured.delete_conversation(_OWNER, conversation_id) is True

    assert measured_runs.delete_calls == batch_count
    assert measured_runs.max_run_ids == batch_size
    assert metrics.max_fetched_rows == batch_size
    assert metrics.outer_transactions == 1
    assert metrics.savepoints == batch_count
    assert metrics.queries == (5 * batch_count) + 4
    assert await _count(pool, "web_conversations") == 0
    assert await _count(pool, "web_conversation_turns") == 0
    assert await _count(pool, "dlightrag_runs") == 0
    assert await _count(pool, "dlightrag_answer_run_artifacts") == 0
    assert await _count(pool, "dlightrag_blobs") == 0


async def test_ten_thousand_conversations_delete_in_bounded_locked_batches(
    store: PGWebConversationStore,
    pool: Any,
) -> None:
    total = 10_003
    foreign_total = 17
    batch_count = 79
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO web_conversations (
                principal_id, conversation_id, agent_session_id, agent_lane_id
            )
            SELECT $1, md5('delete-all:' || series::text)::uuid,
                   md5('delete-all:' || series::text)::uuid, 'main'
            FROM generate_series(1, $2) AS series
            """,
            _OWNER,
            total,
        )
        await conn.execute(
            """
            INSERT INTO web_conversations (
                principal_id, conversation_id, agent_session_id, agent_lane_id
            )
            SELECT $1, md5('foreign-delete-all:' || series::text)::uuid,
                   md5('foreign-delete-all:' || series::text)::uuid, 'main'
            FROM generate_series(1, $2) AS series
            """,
            _OTHER_OWNER,
            foreign_total,
        )
    measured, measured_runs, metrics = _measured_store(pool)

    assert await measured.delete_all_conversations(_OWNER) == total

    assert measured_runs.delete_calls == 0
    assert metrics.max_fetched_rows == 128
    assert metrics.outer_transactions == 1
    assert metrics.savepoints == 0
    assert metrics.queries == (4 * batch_count) + 1
    assert await _count(pool, "web_conversations", principal_id=_OWNER) == 0
    assert await _count(pool, "web_conversations", principal_id=_OTHER_OWNER) == foreign_total


async def test_failure_after_multiple_batches_rolls_back_runs_blobs_sessions_and_conversation(
    store: PGWebConversationStore,
    pool: Any,
) -> None:
    total = 300
    conversation_id = await _conversation(store)
    async with pool.acquire() as conn:
        await _bulk_linked_runs(conn, conversation_id, total)
        await conn.execute(
            """
            INSERT INTO dlightrag_blobs (owner_id, digest, byte_size)
            SELECT $1,
                   md5('blob-a:' || series::text) || md5('blob-b:' || series::text),
                   0
            FROM generate_series(1, $2) AS series
            """,
            _OWNER,
            total,
        )
        await conn.execute(
            """
            INSERT INTO dlightrag_answer_run_artifacts (
                owner_id, run_id, resource_id, reference_kind, ordinal,
                digest, filename, mime_type
            )
            SELECT $1, md5('run:' || series::text)::uuid, 'attachment',
                   'current_attachment', 0,
                   md5('blob-a:' || series::text) || md5('blob-b:' || series::text),
                   'attachment.bin', 'application/octet-stream'
            FROM generate_series(1, $2) AS series
            """,
            _OWNER,
            total,
        )
        await conn.execute(
            """
            INSERT INTO dlightrag_agent_sessions (
                owner_id, session_id, lease_run_id, fencing_epoch
            )
            SELECT $1, md5('session:' || series::text)::uuid,
                   md5('run:' || series::text)::uuid, 1
            FROM generate_series(1, $2) AS series
            """,
            _OWNER,
            total,
        )
        await conn.execute(
            """
            INSERT INTO dlightrag_answer_run_routing (
                owner_id, run_id, requested_mode, valid_modes, resolved_mode,
                context_policy_revision, agent_session_id, agent_lane_id
            )
            SELECT $1, md5('run:' || series::text)::uuid, 'fast', ARRAY['fast'],
                   'fast', 'test', md5('session:' || series::text)::uuid, 'main'
            FROM generate_series(1, $2) AS series
            """,
            _OWNER,
            total,
        )
    measured, measured_runs, metrics = _measured_store(pool, fail_on_call=3)

    with pytest.raises(RuntimeError, match="injected deletion batch failure"):
        await measured.delete_conversation(_OWNER, conversation_id)

    assert measured_runs.delete_calls == 3
    assert measured_runs.max_run_ids == 128
    assert metrics.max_fetched_rows == 128
    assert metrics.outer_transactions == 1
    assert await _count(pool, "web_conversations") == 1
    assert await _count(pool, "web_conversation_turns") == total
    assert await _count(pool, "dlightrag_runs") == total
    assert await _count(pool, "dlightrag_answer_run_artifacts") == total
    assert await _count(pool, "dlightrag_blobs") == total
    assert await _count(pool, "dlightrag_answer_run_routing") == total
    assert await _count(pool, "dlightrag_agent_sessions") == total


async def test_a_shared_fork_session_is_cleaned_only_after_its_final_routing_reference(
    store: PGWebConversationStore,
    pool: Any,
) -> None:
    parent_conversation = await _conversation(store)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions"
            " (owner_id, session_id, lease_run_id, commit_sequence, fencing_epoch)"
            " VALUES ($1, $2, $2, 0, 1)",
            _OWNER,
            uuid.UUID(parent_conversation),
        )
    parent_creation = await _submit(store, parent_conversation)
    assert parent_creation is not None
    fork_conversation = str(uuid.uuid4())
    fork_lane = str(uuid.uuid4())
    request = {
        **_request("fork"),
        "agent_session_id": parent_conversation,
        "agent_lane_id": fork_lane,
        "source_lane_id": "main",
        "parent_run_id": parent_creation.turn.answer_run_id,
        "continuation_kind": "fork",
    }
    submission_id = str(uuid.uuid4())
    fork_creation = await store.create_answer_turn(
        principal_id=_OWNER,
        conversation_id=fork_conversation,
        submission_id=submission_id,
        envelope=_turn_envelope(
            owner=_OWNER,
            request=request,
            submission_id=submission_id,
            fingerprint=run_request_fingerprint(request),
        ),
        run_id=str(uuid.uuid7()),
        title_hint="fork",
        create_conversation=True,
        forked_from_conversation_id=parent_conversation,
    )
    assert fork_creation is not None

    assert await store.delete_conversation(_OWNER, parent_conversation) is True
    assert await _count(pool, "dlightrag_agent_sessions") == 1
    assert await _count(pool, "dlightrag_answer_run_routing") == 1

    assert await store.delete_conversation(_OWNER, fork_conversation) is True
    assert await _count(pool, "dlightrag_agent_sessions") == 0
    assert await _count(pool, "dlightrag_answer_run_routing") == 0


async def test_a_successful_linked_run_prunes_after_the_retention_floor(
    store: PGWebConversationStore, runs: FingerprintingRunStore, pool: Any
) -> None:
    """The unified retention clock applies to conversation-linked runs too."""
    conversation_id = await _conversation(store)
    creation = await _submit(store, conversation_id)
    assert creation is not None
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions"
            " (owner_id, session_id, lease_run_id, commit_sequence, fencing_epoch)"
            " VALUES ($1, $2, $3, 0, 1)",
            _OWNER,
            uuid.UUID(conversation_id),
            uuid.UUID(creation.turn.answer_run_id),
        )
    await _finish(pool, creation.turn.answer_run_id, status="succeeded")
    await _backdate_finish(pool, creation.turn.answer_run_id, days=370)

    deletion = await runs.prune_expired_runs()

    assert deletion.runs == 1
    assert await _count(pool, "web_conversation_turns") == 0
    assert await _count(pool, "web_conversations") == 1
    assert await _count(pool, "dlightrag_agent_sessions") == 0

    replacement = await _submit(
        store,
        conversation_id,
        request=_request("after retention"),
    )
    assert replacement is not None
    assert replacement.turn.turn_number == 2


async def test_empty_conversation_rebases_to_a_fresh_main_lane_after_session_retention(
    store: PGWebConversationStore,
    runs: FingerprintingRunStore,
    pool: Any,
) -> None:
    conversation_id = await _conversation(store)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE web_conversations SET agent_lane_id = 'expired-branch'"
            " WHERE principal_id = $1 AND conversation_id = $2",
            _OWNER,
            uuid.UUID(conversation_id),
        )

    creation = await _submit(
        store,
        conversation_id,
        request=_request(agent_lane_id="expired-branch", source_lane_id="main"),
    )

    assert creation is not None
    assert creation.summary["agent_lane_id"] == "main"
    run = await runs.get_run(owner_id=_OWNER, run_id=creation.turn.answer_run_id)
    assert run is not None
    assert run.prepared_input is not None
    assert run.prepared_input["agent_lane_id"] == "main"
    assert run.prepared_input["source_lane_id"] is None
    routing = await runs.load_routing(owner_id=_OWNER, run_id=creation.turn.answer_run_id)
    assert routing is not None
    assert routing.agent_lane_id == "main"
    assert routing.source_lane_id is None


async def test_session_delete_wins_race_before_empty_conversation_acceptance(
    store: PGWebConversationStore,
    runs: FingerprintingRunStore,
    pool: Any,
) -> None:
    conversation_id = await _conversation(store)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE web_conversations SET agent_lane_id = 'expired-branch'"
            " WHERE principal_id = $1 AND conversation_id = $2",
            _OWNER,
            uuid.UUID(conversation_id),
        )
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions"
            " (owner_id, session_id, lease_run_id, commit_sequence, fencing_epoch)"
            " VALUES ($1, $2, $2, 0, 1)",
            _OWNER,
            uuid.UUID(conversation_id),
        )
        transaction = conn.transaction()
        await transaction.start()
        await conn.execute(
            "DELETE FROM dlightrag_agent_sessions WHERE owner_id = $1 AND session_id = $2",
            _OWNER,
            uuid.UUID(conversation_id),
        )
        submission = asyncio.create_task(
            _submit(
                store,
                conversation_id,
                request=_request(agent_lane_id="expired-branch", source_lane_id="main"),
            )
        )
        await asyncio.sleep(0.05)
        assert not submission.done()
        await transaction.commit()

    creation = await asyncio.wait_for(submission, timeout=5)
    assert creation is not None
    routing = await runs.load_routing(owner_id=_OWNER, run_id=creation.turn.answer_run_id)
    assert routing is not None
    assert routing.agent_lane_id == "main"
    assert routing.source_lane_id is None


async def test_an_expired_event_log_is_still_trimmed_for_a_linked_run(
    store: PGWebConversationStore, runs: FingerprintingRunStore, pool: Any
) -> None:
    """Event trim uses the same retention clock, not a separate exemption."""
    conversation_id = await _conversation(store)
    creation = await _submit(store, conversation_id)
    assert creation is not None
    await _finish(pool, creation.turn.answer_run_id, status="succeeded")
    await _backdate_finish(pool, creation.turn.answer_run_id, days=370)

    assert await runs.trim_expired_event_logs() == 1

    record = await runs.get_run(owner_id=_OWNER, run_id=creation.turn.answer_run_id)
    assert record is not None
    assert record.events_trimmed_at is not None
    assert record.result == {"answer": "done"}


@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_a_failed_or_cancelled_linked_run_prunes_and_cascades_its_turn(
    store: PGWebConversationStore, runs: FingerprintingRunStore, pool: Any, status: str
) -> None:
    conversation_id = await _conversation(store)
    creation = await _submit(store, conversation_id)
    assert creation is not None
    await _finish(
        pool,
        creation.turn.answer_run_id,
        status=status,
        error="answer_stream_failed" if status == "failed" else None,
    )
    await _backdate_finish(pool, creation.turn.answer_run_id, days=370)

    deletion = await runs.prune_expired_runs()

    assert deletion.runs == 1
    assert await _count(pool, "web_conversation_turns") == 0
    assert await _count(pool, "web_conversations") == 1


async def test_an_unlinked_successful_run_still_prunes(
    store: PGWebConversationStore, runs: FingerprintingRunStore, pool: Any
) -> None:
    request = _request()
    creation = await runs.create_run(
        owner_id=_OWNER,
        prepared_input=request,
        idempotency_fingerprint=run_request_fingerprint(request),
    )
    await _finish(pool, creation.run.run_id, status="succeeded")
    await _backdate_finish(pool, creation.run.run_id, days=370)

    assert (await runs.prune_expired_runs()).runs == 1


async def test_deleting_a_run_row_cascades_its_conversation_turn(
    store: PGWebConversationStore, runs: FingerprintingRunStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    creation = await _submit(store, conversation_id)
    assert creation is not None

    await runs.delete_runs(owner_id=_OWNER, run_ids=[creation.turn.answer_run_id])

    assert await _count(pool, "web_conversation_turns") == 0


async def test_a_turn_cannot_reference_a_run_another_principal_owns(
    store: PGWebConversationStore, runs: FingerprintingRunStore, pool: Any
) -> None:
    """The foreign key carries the principal, so the link is owner-scoped."""
    conversation_id = await _conversation(store)
    request = _request()
    foreign = await runs.create_run(
        owner_id=_OTHER_OWNER,
        prepared_input=request,
        idempotency_fingerprint=run_request_fingerprint(request),
    )

    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
            await conn.execute(
                "INSERT INTO web_conversation_turns "
                "(turn_id, principal_id, conversation_id, turn_number, submission_id, "
                " answer_run_id) "
                "VALUES ($1::text::uuid, $2, $3::text::uuid, 1, $4::text::uuid, $5::text::uuid)",
                str(uuid.uuid4()),
                _OWNER,
                conversation_id,
                str(uuid.uuid4()),
                foreign.run.run_id,
            )


async def test_the_accepted_envelope_survives_the_terminal_transition(
    store: PGWebConversationStore, runs: FingerprintingRunStore, pool: Any
) -> None:
    """Blocker 1 regression: finish clears prepared input, not the envelope."""
    conversation_id = await _conversation(store)
    content = b"keep-me"
    digest = artifact_digest(content)
    reference = PendingArtifactReference(
        resource_id="attachment-1",
        reference_kind="current_attachment",
        ordinal=1,
        digest=digest,
        filename="chart.png",
        mime_type="image/png",
    )
    creation = await _submit(
        store,
        conversation_id,
        request=_request(
            "remember me",
            attachments=[
                {
                    "ordinal": 1,
                    "digest": digest,
                    "filename": "chart.png",
                    "mime_type": "image/png",
                }
            ],
        ),
        artifacts=[PendingArtifact(content=content)],
        references=[reference],
    )
    assert creation is not None
    await _finish(pool, creation.turn.answer_run_id, status="succeeded")

    record = await runs.get_run(owner_id=_OWNER, run_id=creation.turn.answer_run_id)
    assert record is not None
    assert record.prepared_input is None
    envelope = record.request_input()
    assert envelope["query"] == "remember me"
    assert envelope["workspaces"] == ["alpha"]
    assert envelope["attachments"][0]["filename"] == "chart.png"


async def test_submission_seed_keeps_first_attachment_ordinals_and_skips_incomplete_legacy_rows(
    store: PGWebConversationStore,
    pool: Any,
) -> None:
    conversation_id = await _conversation(store)
    contents = [f"attachment-{ordinal}".encode() for ordinal in range(3)]
    digests = [artifact_digest(content) for content in contents]
    creation = await _submit(
        store,
        conversation_id,
        request=_request(
            "attachments",
            attachments=[
                {
                    "ordinal": ordinal,
                    "digest": digests[ordinal],
                    "filename": f"attachment-{ordinal}.txt",
                    "mime_type": "text/plain",
                    "byte_size": len(contents[ordinal]),
                }
                for ordinal in range(3)
            ],
        ),
        artifacts=[PendingArtifact(content=content) for content in contents],
        references=[
            PendingArtifactReference(
                resource_id=f"attachment-{ordinal}",
                reference_kind="current_attachment",
                ordinal=ordinal,
                digest=digests[ordinal],
                filename=f"attachment-{ordinal}.txt",
                mime_type="text/plain",
            )
            for ordinal in range(3)
        ],
    )
    assert creation is not None
    await _finish(pool, creation.turn.answer_run_id, status="succeeded")

    seed = await store.submission_seed(_OWNER, conversation_id, attachment_limit=2)
    assert seed is not None
    assert [item.source_ordinal for item in seed.attachments] == [0, 1]

    legacy = await _submit(store, conversation_id, request=_request("legacy"))
    assert legacy is not None
    await _finish(pool, legacy.turn.answer_run_id, status="succeeded")
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs "
            "SET accepted_input_json = jsonb_set(accepted_input_json, '{attachments}', $2::jsonb) "
            "WHERE owner_id = $1 AND run_id = $3::uuid",
            _OWNER,
            '[{"ordinal": 0, "filename": "legacy.txt", "mime_type": "text/plain"}]',
            uuid.UUID(legacy.turn.answer_run_id),
        )

    complete_seed = await store.submission_seed(_OWNER, conversation_id, attachment_limit=6)
    assert complete_seed is not None
    assert [item.source_ordinal for item in complete_seed.attachments] == [0, 1, 2]
    assert all(item.digest != "None" for item in complete_seed.attachments)


async def test_an_empty_conversation_is_reclaimed_after_its_turns_age_out(
    store: PGWebConversationStore, runs: FingerprintingRunStore, pool: Any
) -> None:
    """Turns live and die with their runs; the empty row is then reclaimed."""
    conversation_id = await _conversation(store)
    creation = await _submit(store, conversation_id)
    assert creation is not None
    await _finish(pool, creation.turn.answer_run_id, status="succeeded")
    await _backdate_finish(pool, creation.turn.answer_run_id, days=370)

    assert (await runs.prune_expired_runs()).runs == 1
    assert await store.prune_empty_conversations() == 1

    assert await _count(pool, "web_conversations") == 0
    assert await _count(pool, "web_conversation_turns") == 0


async def test_a_conversation_with_live_turns_is_never_reclaimed(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    await _submit(store, conversation_id)

    assert await store.prune_empty_conversations() == 0
    assert await _count(pool, "web_conversations") == 1


async def test_205_turn_keyset_traversal_is_stable_across_a_concurrent_append(
    store: PGWebConversationStore,
) -> None:
    conversation_id = await _conversation(store)
    for number in range(1, 206):
        created = await _submit(store, conversation_id, request=_request(f"turn {number}"))
        assert created is not None

    page = await store.history_page(
        _OWNER,
        conversation_id,
        page=ConversationHistoryPageRequest(limit=40),
    )
    assert page is not None
    assert page.fetched_rows == 41
    assert [turn.turn_number for turn in page.turns] == list(range(166, 206))
    seen = [turn.turn_number for turn in page.turns]
    cursor = page.next_cursor

    appended = await _submit(store, conversation_id, request=_request("concurrent append"))
    assert appended is not None and appended.turn.turn_number == 206
    while cursor is not None:
        page = await store.history_page(
            _OWNER,
            conversation_id,
            page=ConversationHistoryPageRequest(limit=40, cursor=cursor),
        )
        assert page is not None
        assert page.fetched_rows <= 41
        seen.extend(turn.turn_number for turn in page.turns)
        cursor = page.next_cursor

    assert sorted(seen) == list(range(1, 206))
    assert len(seen) == len(set(seen))
    assert 206 not in seen


async def test_older_turns_remain_durable_and_keyset_reachable(
    store: PGWebConversationStore, pool: Any
) -> None:
    """A presentation page is a read bound; its cursor keeps older turns reachable."""
    conversation_id = await _conversation(store)
    for index in range(3):
        request = {
            **_request(f"question {index}"),
            "agent_session_id": conversation_id,
        }
        submission_id = str(uuid.uuid4())
        await store.create_answer_turn(
            principal_id=_OWNER,
            conversation_id=conversation_id,
            submission_id=submission_id,
            envelope=_turn_envelope(
                owner=_OWNER,
                request=request,
                submission_id=submission_id,
                fingerprint=run_request_fingerprint(request),
            ),
            run_id=str(uuid.uuid7()),
            title_hint="why",
        )

    assert await _count(pool, "web_conversation_turns") == 3
    assert await _count(pool, "dlightrag_runs") == 3

    page = await store.history_page(
        _OWNER,
        conversation_id,
        page=ConversationHistoryPageRequest(limit=2),
    )
    assert page is not None
    assert [turn.turn_number for turn in page.turns] == [2, 3]
    assert page.next_cursor is not None
    older = await store.history_page(
        _OWNER,
        conversation_id,
        page=ConversationHistoryPageRequest(limit=2, cursor=page.next_cursor),
    )
    assert older is not None
    assert [turn.turn_number for turn in older.turns] == [1]


# ---------------------------------------------------------------------------
# Retention/deletion lock order
# ---------------------------------------------------------------------------


async def test_conversation_deletion_takes_the_same_lock_order_as_run_retention(
    store: PGWebConversationStore, db_name: str, pool: Any
) -> None:
    """Both deletion paths must lock the run row before its conversation turn.

    Retention now runs on every process, so a conversation delete racing a prune
    is ordinary traffic. The blocking connection reproduces exactly what
    ``prune_expired_runs`` does: lock the expired run, then delete it and let the
    cascade take its turn. If conversation deletion removed the turn first, the
    two transactions would hold each other's next lock and PostgreSQL would abort
    one of them.
    """
    conversation_id = await _conversation(store)
    creation = await _submit(store, conversation_id)
    assert creation is not None
    run_uuid = uuid.UUID(creation.turn.answer_run_id)

    blocker = await asyncpg.connect(**{**_PG_CONN_KWARGS, "database": db_name})
    try:
        transaction = blocker.transaction()
        await transaction.start()
        await blocker.fetchrow(
            "SELECT run_id FROM dlightrag_runs WHERE owner_id = $1 AND run_id = $2 FOR UPDATE",
            _OWNER,
            run_uuid,
        )

        deleting = asyncio.create_task(store.delete_conversation(_OWNER, conversation_id))
        await asyncio.sleep(0.3)
        assert not deleting.done()

        await blocker.execute(
            "DELETE FROM dlightrag_runs WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            run_uuid,
        )
        await transaction.commit()

        assert await asyncio.wait_for(deleting, timeout=10) is True
    finally:
        await blocker.close()

    assert await _count(pool, "web_conversations") == 0
    assert await _count(pool, "web_conversation_turns") == 0


# ---------------------------------------------------------------------------
# Reader validation
# ---------------------------------------------------------------------------


async def test_a_reader_validates_the_migrated_schema_without_ddl(pool: Any) -> None:
    writer = PGWebConversationStore(pool=pool)
    await writer.initialize()

    reader = PGWebConversationStore(pool=pool)
    await reader.initialize(validate_only=True)

    conversation_id = await _conversation(reader)
    creation = await _submit(reader, conversation_id)
    assert creation is not None
