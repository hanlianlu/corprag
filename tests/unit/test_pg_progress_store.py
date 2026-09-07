# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Recording-connection tests for PostgreSQL Fast progress settlement."""

import hashlib
import uuid
from typing import Any, cast

from dlightrag.adapters.postgres.answer.session_repository import PGProgressStore
from dlightrag.engine.agent.session.ids import StageIntentId
from dlightrag.engine.runtime.progress import StageCommit, StageTerminalCommit
from dlightrag.engine.runtime.settlements import OpaqueEvidenceWrite


class _Context:
    def __init__(self, value: Any) -> None:
        self._value = value

    async def __aenter__(self) -> Any:
        return self._value

    async def __aexit__(self, *args: Any) -> None:
        return None


class _RecordingConnection:
    def __init__(self) -> None:
        self.queries: list[tuple[str, str]] = []
        self.query_args: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self) -> _Context:
        return _Context(None)

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.queries.append(("fetchrow", query))
        self.query_args.append((query, args))
        if "FROM dlightrag_runs" in query:
            return {"durable_progress_version": 0, "cancel_requested": False}
        if "FROM dlightrag_answer_run_stages" in query:
            return None
        raise AssertionError(f"unexpected fetchrow: {query}")

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.queries.append(("fetchval", query))
        self.query_args.append((query, args))
        if "UPDATE dlightrag_runs" in query and "event_sequence" in query:
            return 7
        if "LEFT JOIN dlightrag_answer_evidence" in query:
            return None
        raise AssertionError(f"unexpected fetchval: {query}")

    async def execute(self, query: str, *args: Any) -> str:
        self.queries.append(("execute", query))
        self.query_args.append((query, args))
        return "OK"


class _Pool:
    def __init__(self, connection: _RecordingConnection) -> None:
        self._connection = connection

    def acquire(self) -> _Context:
        return _Context(self._connection)


def _store(connection: _RecordingConnection) -> PGProgressStore:
    run_id = uuid.uuid4()
    return PGProgressStore(
        pool=cast(Any, _Pool(connection)),
        owner_id="owner",
        run_id=run_id,
        worker_id="worker",
        lease_owner="worker",
        fencing_epoch=1,
    )


def _stage_id() -> StageIntentId:
    return StageIntentId.deterministic(run_id=str(uuid.uuid4()), name="fast:stage")


def _evidence(count: int) -> tuple[OpaqueEvidenceWrite, ...]:
    session_id = str(uuid.uuid4())
    intent_id = str(uuid.uuid4())
    content = b"evidence"
    locator = b"locator"
    return tuple(
        OpaqueEvidenceWrite(
            session_id=session_id,
            intent_id=intent_id,
            result_ordinal=ordinal,
            content_digest=hashlib.sha256(content).hexdigest(),
            locator_digest=hashlib.sha256(locator).hexdigest(),
            content=content,
            locator=locator,
        )
        for ordinal in range(count)
    )


def _assert_single_run_lock_without_progress_reselect(
    connection: _RecordingConnection,
) -> None:
    run_reads = [
        query
        for kind, query in connection.queries
        if kind in {"fetchrow", "fetchval"} and "FROM dlightrag_runs" in query
    ]
    assert len(run_reads) == 1
    assert "durable_progress_version" in run_reads[0]
    assert "cancel_requested_at" in run_reads[0]
    assert "FOR UPDATE" in run_reads[0]


async def test_settle_stage_uses_the_initial_fenced_row_for_progress() -> None:
    connection = _RecordingConnection()

    outcome = await _store(connection).settle_stage(
        expected_progress_version=0,
        stage_intent_id=_stage_id(),
        stage_name="planner",
        state={"plan": "fixed"},
        evidence=(),
    )

    assert isinstance(outcome, StageCommit)
    _assert_single_run_lock_without_progress_reselect(connection)


async def test_settle_stage_batches_evidence_in_two_ordered_array_statements() -> None:
    connection = _RecordingConnection()
    evidence = _evidence(3)

    outcome = await _store(connection).settle_stage(
        expected_progress_version=0,
        stage_intent_id=_stage_id(),
        stage_name="retrieval",
        state={"results": []},
        evidence=evidence,
    )

    assert isinstance(outcome, StageCommit)
    assert outcome.evidence_count == 3
    evidence_queries = [
        (kind, query) for kind, query in connection.queries if "dlightrag_answer_evidence" in query
    ]
    assert [kind for kind, _query in evidence_queries] == ["execute", "fetchval"]
    insert = evidence_queries[0][1]
    assert "unnest(" in insert
    assert "$3::uuid[]" in insert
    assert "$8::bytea[]" in insert
    assert "WITH ORDINALITY" in insert
    assert "ORDER BY input.ordinality" in insert
    assert (
        "ON CONFLICT (owner_id, run_id, session_id, intent_id, result_ordinal) DO NOTHING" in insert
    )
    assert "jsonb_to_recordset" not in insert
    assert "DO UPDATE" not in insert
    insert_args = next(args for query, args in connection.query_args if query == insert)
    probe = evidence_queries[1][1]
    probe_args = next(args for query, args in connection.query_args if query == probe)
    assert all(len(values) == 3 for values in insert_args[2:])
    assert all(len(values) == 3 for values in probe_args[2:])
    assert len(insert_args) == 9
    assert len(probe_args) == 7
    _assert_single_run_lock_without_progress_reselect(connection)


async def test_settle_terminal_uses_the_initial_fenced_row_for_progress() -> None:
    connection = _RecordingConnection()

    outcome = await _store(connection).settle_terminal(
        expected_progress_version=0,
        stage_intent_id=_stage_id(),
        state={"result": {"answer": "fixed"}},
        result={"answer": "fixed"},
    )

    assert isinstance(outcome, StageTerminalCommit)
    assert outcome.status == "succeeded"
    assert outcome.terminal_event_sequence == 7
    _assert_single_run_lock_without_progress_reselect(connection)
