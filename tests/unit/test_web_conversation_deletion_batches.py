# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded-working-set contracts for PostgreSQL Web conversation deletion."""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dlightrag.adapters.postgres.web import web_conversations as pg_web
from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore
from dlightrag.engine.runtime.records import RunDeletion


@dataclass
class _DeletionState:
    conversations: dict[str, str]
    runs: dict[str, list[str]]
    blobs: set[str]
    sessions: set[str]
    routing: dict[str, str]


class _Transaction:
    def __init__(self, conn: _RecordingConnection) -> None:
        self._conn = conn
        self._snapshot: _DeletionState | None = None

    async def __aenter__(self) -> None:
        self._conn.transactions += 1
        self._snapshot = copy.deepcopy(self._conn.state)

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is not None:
            assert self._snapshot is not None
            self._conn.state = self._snapshot


class _RecordingConnection:
    def __init__(self, state: _DeletionState, *, malformed_run_row: bool = False) -> None:
        self.state = state
        self.malformed_run_row = malformed_run_row
        self.transactions = 0
        self.fetch_calls = 0
        self.fetchrow_calls = 0
        self.fetchval_calls = 0
        self.execute_calls = 0
        self.max_fetched_rows = 0
        self.max_pending_session_cleanups = 0
        self._pending_session_cleanups: list[str] = []

    def transaction(self) -> _Transaction:
        return _Transaction(self)

    def _record_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.max_fetched_rows = max(self.max_fetched_rows, len(rows))
        return rows

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls += 1
        assert not self._pending_session_cleanups
        if query == pg_web._LOCK_PRINCIPAL_CONVERSATION_BATCH:  # pyright: ignore[reportPrivateUsage]
            principal_id, limit = args
            assert principal_id == "owner"
            rows = [
                {"conversation_id": conversation_id, "agent_session_id": session_id}
                for conversation_id, session_id in sorted(self.state.conversations.items())[
                    : int(limit)
                ]
            ]
            return self._record_rows(rows)
        if query == pg_web._SELECT_LINKED_CONVERSATION_BATCH:  # pyright: ignore[reportPrivateUsage]
            principal_id, conversation_ids, limit = args
            assert principal_id == "owner"
            rows = [
                {"conversation_id": conversation_id}
                for conversation_id in sorted(conversation_ids)
                if self.state.runs.get(conversation_id)
            ][: int(limit)]
            return self._record_rows(rows)
        if query == pg_web._DELETE_CONVERSATION_BATCH:  # pyright: ignore[reportPrivateUsage]
            principal_id, conversation_ids, limit = args
            assert principal_id == "owner"
            assert len(conversation_ids) <= int(limit)
            rows: list[dict[str, Any]] = []
            for conversation_id in sorted(conversation_ids):
                assert not self.state.runs.get(conversation_id)
                session_id = self.state.conversations.pop(conversation_id, None)
                if session_id is not None:
                    self.state.runs.pop(conversation_id, None)
                    rows.append(
                        {"conversation_id": conversation_id, "agent_session_id": session_id}
                    )
            self._pending_session_cleanups = [row["agent_session_id"] for row in rows]
            self.max_pending_session_cleanups = max(
                self.max_pending_session_cleanups,
                len(self._pending_session_cleanups),
            )
            return self._record_rows(rows)
        if query == pg_web._SELECT_CONVERSATION_RUN_BATCH:  # pyright: ignore[reportPrivateUsage]
            principal_id, conversation_id, limit = args
            assert principal_id == "owner"
            run_ids = self.state.runs.get(str(conversation_id), [])[: int(limit)]
            if self.malformed_run_row and run_ids:
                self.malformed_run_row = False
                return self._record_rows([{"answer_run_id": "not-a-uuid"}])
            return self._record_rows([{"answer_run_id": run_id} for run_id in run_ids])
        raise AssertionError(f"unexpected fetch query: {query}")

    async def fetchrow(self, query: str, *args: Any) -> dict[str, str] | None:
        self.fetchrow_calls += 1
        assert not self._pending_session_cleanups
        if query != pg_web._LOCK_CONVERSATION:  # pyright: ignore[reportPrivateUsage]
            raise AssertionError(f"unexpected fetchrow query: {query}")
        principal_id, conversation_id = args
        assert principal_id == "owner"
        session_id = self.state.conversations.get(str(conversation_id))
        return {"agent_session_id": session_id} if session_id is not None else None

    async def fetchval(self, query: str, *args: Any) -> str | None:
        self.fetchval_calls += 1
        assert not self._pending_session_cleanups
        if query != pg_web._DELETE_CONVERSATION:  # pyright: ignore[reportPrivateUsage]
            raise AssertionError(f"unexpected fetchval query: {query}")
        principal_id, conversation_id = args
        assert principal_id == "owner"
        conversation_id = str(conversation_id)
        assert not self.state.runs.get(conversation_id)
        session_id = self.state.conversations.pop(conversation_id, None)
        if session_id is not None:
            self.state.runs.pop(conversation_id, None)
            self._pending_session_cleanups = [session_id]
            self.max_pending_session_cleanups = max(self.max_pending_session_cleanups, 1)
        return session_id

    async def execute(self, query: str, *args: Any) -> str:
        self.execute_calls += 1
        principal_id, session_values = args
        assert principal_id == "owner"
        if query == pg_web._DELETE_AGENT_SESSION_IF_UNREFERENCED:  # pyright: ignore[reportPrivateUsage]
            session_ids = [str(session_values)]
        elif query == pg_web._DELETE_AGENT_SESSION_BATCH_IF_UNREFERENCED:  # pyright: ignore[reportPrivateUsage]
            session_ids = [str(value) for value in session_values]
        else:
            raise AssertionError(f"unexpected execute query: {query}")
        assert session_ids == self._pending_session_cleanups
        self._pending_session_cleanups = []
        for session_id in session_ids:
            if session_id not in self.state.routing.values():
                self.state.sessions.discard(session_id)
        return f"DELETE {len(session_ids)}"


class _Acquire:
    def __init__(self, conn: _RecordingConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> _RecordingConnection:
        return self._conn

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None


class _Pool:
    def __init__(self, conn: _RecordingConnection) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


class _DeletingRunStore:
    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        malformed_result: bool = False,
        make_no_progress: bool = False,
    ) -> None:
        self.fail_on_call = fail_on_call
        self.malformed_result = malformed_result
        self.make_no_progress = make_no_progress
        self.calls = 0
        self.max_run_ids = 0

    async def delete_runs_in(
        self,
        conn: _RecordingConnection,
        *,
        owner_id: str,
        run_ids: list[str],
    ) -> RunDeletion:
        assert owner_id == "owner"
        self.calls += 1
        self.max_run_ids = max(self.max_run_ids, len(run_ids))
        if self.fail_on_call == self.calls:
            raise RuntimeError("injected batch failure")
        if self.make_no_progress:
            return RunDeletion(runs=0, artifacts=0)

        selected = set(run_ids)
        deleted = 0
        for conversation_id, linked_ids in conn.state.runs.items():
            survivors = [run_id for run_id in linked_ids if run_id not in selected]
            deleted += len(linked_ids) - len(survivors)
            conn.state.runs[conversation_id] = survivors
        for run_id in selected:
            conn.state.blobs.discard(run_id)
            session_id = conn.state.routing.pop(run_id, None)
            if session_id is not None and session_id not in conn.state.routing.values():
                conn.state.sessions.discard(session_id)
        if self.malformed_result:
            return cast(Any, SimpleNamespace(runs="many", artifacts=0))
        return RunDeletion(runs=deleted, artifacts=deleted)


def _id(value: int) -> str:
    return str(uuid.UUID(int=value))


def _state(*, conversations: int, runs_per_first: int = 0) -> _DeletionState:
    conversation_rows = {_id(index + 1): _id(1_000_000 + index) for index in range(conversations)}
    linked_runs = {conversation_id: [] for conversation_id in conversation_rows}
    if conversation_rows and runs_per_first:
        first = min(conversation_rows)
        linked_runs[first] = [_id(2_000_000 + index) for index in range(runs_per_first)]
    all_runs = {run_id for run_ids in linked_runs.values() for run_id in run_ids}
    return _DeletionState(
        conversations=conversation_rows,
        runs=linked_runs,
        blobs=set(all_runs),
        sessions=set(conversation_rows.values()),
        routing={},
    )


def _store(conn: _RecordingConnection, run_store: _DeletingRunStore) -> PGWebConversationStore:
    store = PGWebConversationStore(
        pool=cast(Any, _Pool(conn)),
        run_store=cast(Any, run_store),
    )
    store._initialized = True  # pyright: ignore[reportPrivateUsage]
    return store


@pytest.mark.asyncio
async def test_one_ten_thousand_run_history_uses_bounded_sequences_in_one_transaction() -> None:
    total = 10_037
    state = _state(conversations=1, runs_per_first=total)
    conversation_id = next(iter(state.conversations))
    conn = _RecordingConnection(state)
    runs = _DeletingRunStore()

    assert await _store(conn, runs).delete_conversation("owner", conversation_id) is True

    expected_batches = math.ceil(total / pg_web._DELETE_BATCH_SIZE)  # pyright: ignore[reportPrivateUsage]
    assert runs.calls == expected_batches
    assert runs.max_run_ids == pg_web._DELETE_BATCH_SIZE  # pyright: ignore[reportPrivateUsage]
    assert conn.max_fetched_rows == pg_web._DELETE_BATCH_SIZE  # pyright: ignore[reportPrivateUsage]
    assert conn.transactions == 1
    assert conn.fetch_calls == expected_batches + 1
    assert not conn.state.conversations
    assert not conn.state.runs
    assert not conn.state.blobs


@pytest.mark.asyncio
async def test_delete_all_counts_ten_thousand_rows_without_accumulating_sessions() -> None:
    total = 10_003
    conn = _RecordingConnection(_state(conversations=total))
    runs = _DeletingRunStore()

    assert await _store(conn, runs).delete_all_conversations("owner") == total

    conversation_batches = math.ceil(total / pg_web._DELETE_BATCH_SIZE)  # pyright: ignore[reportPrivateUsage]
    assert conn.fetch_calls == (3 * conversation_batches) + 1
    assert conn.max_fetched_rows == pg_web._DELETE_BATCH_SIZE  # pyright: ignore[reportPrivateUsage]
    assert conn.max_pending_session_cleanups == pg_web._DELETE_BATCH_SIZE  # pyright: ignore[reportPrivateUsage]
    assert conn.execute_calls == conversation_batches
    assert conn.fetchval_calls == 0
    assert conn.transactions == 1
    assert runs.calls == 0
    assert not conn.state.conversations
    assert not conn.state.sessions


@pytest.mark.asyncio
async def test_failure_after_multiple_run_batches_rolls_back_every_deletion() -> None:
    state = _state(conversations=1, runs_per_first=400)
    conversation_id = next(iter(state.conversations))
    state.routing = {run_id: _id(3_000_000 + index) for index, run_id in enumerate(state.blobs)}
    state.sessions.update(state.routing.values())
    before = copy.deepcopy(state)
    conn = _RecordingConnection(state)
    runs = _DeletingRunStore(fail_on_call=3)

    with pytest.raises(RuntimeError, match="injected batch failure"):
        await _store(conn, runs).delete_conversation("owner", conversation_id)

    assert runs.calls == 3
    assert conn.transactions == 1
    assert conn.state == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connection_options", "run_store_options", "message"),
    (
        pytest.param(
            {"malformed_run_row": True},
            {},
            "non-canonical UUID",
            id="malformed-id",
        ),
        pytest.param(
            {},
            {"malformed_result": True},
            "invalid deletion result",
            id="malformed-store-result",
        ),
        pytest.param(
            {},
            {"make_no_progress": True},
            "no forward progress",
            id="run-id-collision",
        ),
    ),
)
async def test_malformed_deletion_inputs_fail_closed(
    connection_options: dict[str, Any],
    run_store_options: dict[str, Any],
    message: str,
) -> None:
    state = _state(conversations=1, runs_per_first=3)
    conversation_id = next(iter(state.conversations))
    before = copy.deepcopy(state)
    conn = _RecordingConnection(state, **connection_options)

    with pytest.raises(RuntimeError, match=message):
        await _store(conn, _DeletingRunStore(**run_store_options)).delete_conversation(
            "owner", conversation_id
        )

    assert conn.state == before


@pytest.mark.asyncio
async def test_missing_and_empty_single_conversation_results_are_unchanged() -> None:
    state = _state(conversations=1)
    conversation_id = next(iter(state.conversations))
    conn = _RecordingConnection(state)
    store = _store(conn, _DeletingRunStore())

    assert await store.delete_conversation("owner", _id(999_999_999)) is False
    assert conversation_id in conn.state.conversations
    assert await store.delete_conversation("owner", conversation_id) is True
    assert conn.transactions == 2


def test_deletion_selectors_are_stably_ordered_limited_keysets() -> None:
    queries = (
        pg_web._SELECT_CONVERSATION_RUN_BATCH,  # pyright: ignore[reportPrivateUsage]
        pg_web._LOCK_PRINCIPAL_CONVERSATION_BATCH,  # pyright: ignore[reportPrivateUsage]
        pg_web._SELECT_LINKED_CONVERSATION_BATCH,  # pyright: ignore[reportPrivateUsage]
        pg_web._DELETE_CONVERSATION_BATCH,  # pyright: ignore[reportPrivateUsage]
    )
    for query in queries:
        normalized = " ".join(query.upper().split())
        assert " ORDER BY " in normalized
        assert " LIMIT " in normalized
        assert " OFFSET " not in normalized
    principal_lock = " ".join(
        pg_web._LOCK_PRINCIPAL_CONVERSATION_BATCH.upper().split()  # pyright: ignore[reportPrivateUsage]
    )
    assert "SKIP LOCKED" not in principal_lock
    assert not hasattr(pg_web, "_SELECT_PRINCIPAL_RUNS")
