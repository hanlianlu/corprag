# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Fast Host turns share the canonical Agent Session tree without an Operation."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from dlightrag.engine.agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    UserMessageEntry,
)
from dlightrag.engine.agent.session.fold import project_session_messages
from dlightrag.engine.agent.session.ids import EntryId, LaneId, ProjectionId, SessionId
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.projection import ContextProjection, projection_source_digest
from dlightrag.engine.agent.session.registers import (
    DeleteRegister,
    HostTurnReservation,
    LaneHead,
    LaneState,
    RegisterRecord,
    RegisterRef,
    SetRegister,
    decode_register,
)
from dlightrag.engine.agent.session.repository import (
    AgentSessionSnapshot,
    project_transaction_commit,
)
from dlightrag.engine.agent.session.runtime import (
    AgentSessionRuntime,
    OperationConflictError,
    SessionLeaseLostError,
)
from dlightrag.engine.agent.session.transactions import (
    RegisterConflict,
    RegisterExpectation,
    SessionTransaction,
    TransactionCommit,
    TransactionLeaseLost,
)
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.answer.execution.executor import (
    AnswerExecutor,
    _project_fast_history_before_current_user,
    _reserve_agent_session_boundary,
)
from dlightrag.engine.answer.fast import FastSessionHost, ensure_session_lane
from dlightrag.engine.answer.history import HistoryProjectionTarget
from dlightrag.engine.runtime.errors import RunExecutionError


async def _no_settled_result() -> None:
    return None


class _CompactionFaultTransactions:
    """Inject one deterministic ambiguous apply or genuine CAS conflict."""

    def __init__(self, store: MemoryAgentSessionRepository[None], *, applied: bool) -> None:
        self._store = store
        self._applied = applied
        self.injected = False
        self.load_calls = 0
        self.refresh_calls = 0

    async def load(self, session_id: SessionId):
        self.load_calls += 1
        return await self._store.load(session_id)

    async def refresh(self, session_id: SessionId, *, previous):
        self.refresh_calls += 1
        return await self._store.refresh(session_id, previous=previous)

    async def transact(self, *, session_id, fencing_epoch, transaction):
        if not self.injected and any(
            isinstance(entry, CompactionEntry) for entry in transaction.entries
        ):
            self.injected = True
            if self._applied:
                await self._store.transact(
                    session_id=session_id,
                    fencing_epoch=fencing_epoch,
                    transaction=transaction,
                )
                raise RuntimeError("commit applied but acknowledgement was lost")
            snapshot = await self._store.load(session_id)
            reservation = next(
                record
                for record in snapshot.registers
                if isinstance(record.value, HostTurnReservation)
            )
            await self._store.transact(
                session_id=session_id,
                fencing_epoch=fencing_epoch,
                transaction=SessionTransaction.from_parts(
                    register_writes=[SetRegister(reservation.value)],
                    expectations=[RegisterExpectation(reservation.ref, reservation.sequence)],
                ),
            )
            return await self._store.transact(
                session_id=session_id,
                fencing_epoch=fencing_epoch,
                transaction=transaction,
            )
        return await self._store.transact(
            session_id=session_id,
            fencing_epoch=fencing_epoch,
            transaction=transaction,
        )


async def _fast_host(
    repository: Any,
    session_id: SessionId,
    *,
    load_settled_result: Any = _no_settled_result,
) -> FastSessionHost:
    return FastSessionHost(
        repository=repository,
        initial_snapshot=await repository.load(session_id),
        load_settled_result=load_settled_result,
        fencing_epoch=1,
    )


async def _seed_history(
    repository: MemoryAgentSessionRepository[None],
    session_id: SessionId,
    *,
    count: int,
) -> tuple[UserMessageEntry, ...]:
    entries: list[UserMessageEntry] = []
    parent: EntryId | None = None
    for index in range(count):
        entry = UserMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=datetime.now(UTC),
            parent_entry_id=parent,
            content=f"history {index}",
        )
        entries.append(entry)
        parent = entry.entry_id
    head = LaneHead(LaneId.main(), parent)
    state = LaneState(LaneId.main())
    await repository.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            entries=entries,
            register_writes=[SetRegister(head), SetRegister(state)],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(state.ref, None),
            ],
        ),
    )
    return tuple(entries)


class _CountingFastRepository(MemoryAgentSessionRepository[None]):
    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0
        self.refresh_calls = 0
        self.decoded_rows = 0
        self.loaded_snapshots: list[AgentSessionSnapshot] = []

    async def load(self, session_id: SessionId) -> AgentSessionSnapshot:
        self.load_calls += 1
        snapshot = await super().load(session_id)
        self.decoded_rows += len(snapshot.entries)
        self.loaded_snapshots.append(snapshot)
        return snapshot

    async def refresh(
        self,
        session_id: SessionId,
        *,
        previous: AgentSessionSnapshot,
    ) -> AgentSessionSnapshot:
        self.refresh_calls += 1
        snapshot = await super().refresh(session_id, previous=previous)
        self.decoded_rows += snapshot.last_entry_sequence - previous.last_entry_sequence
        return snapshot

    async def authoritative(self, session_id: SessionId) -> AgentSessionSnapshot:
        return await super().load(session_id)


class _UncommittedMutationRepository(_CountingFastRepository):
    async def refresh(
        self,
        session_id: SessionId,
        *,
        previous: AgentSessionSnapshot,
    ) -> AgentSessionSnapshot:
        snapshot = await super().refresh(session_id, previous=previous)
        return replace(
            snapshot,
            registers=(RegisterRecord(LaneState(LaneId.main()), 1),),
        )


class _ForcedFastRepository(_CountingFastRepository):
    forced: str | None = None

    async def transact(self, **kwargs: Any):
        if self.forced == "conflict":
            return RegisterConflict(RegisterRef("lane_head", LaneId.main().value), None, 1)
        if self.forced == "lease":
            return TransactionLeaseLost()
        if self.forced == "exception":
            raise RuntimeError("transaction acknowledgement lost")
        outcome = await super().transact(**kwargs)
        if self.forced == "noncontiguous":
            assert isinstance(outcome, TransactionCommit)
            return TransactionCommit(
                commit_sequence=outcome.commit_sequence + 1,
                appended_sequences=outcome.appended_sequences,
                register_sequences=tuple(
                    (ref, sequence + 1) for ref, sequence in outcome.register_sequences
                ),
            )
        return outcome


class _SlowFastRepository(_CountingFastRepository):
    def __init__(self) -> None:
        super().__init__()
        self.block_transaction = False
        self.transaction_started = asyncio.Event()
        self.release_transaction = asyncio.Event()

    async def transact(self, **kwargs: Any):
        if self.block_transaction:
            self.transaction_started.set()
            await self.release_transaction.wait()
        return await super().transact(**kwargs)


class _CountingCompactionModel:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_text(self, **_kwargs):
        self.calls += 1
        yield "## Goal\nPreserve the prior turn."


async def _run_fault_injected_compaction(*, applied: bool):
    store = MemoryAgentSessionRepository[None]()
    transactions = _CompactionFaultTransactions(store, applied=applied)
    session_id = SessionId.new()
    host = await _fast_host(transactions, session_id)
    old_question = "question " * 4_000
    old_answer = "answer " * 4_000
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="old",
        idempotency_key="old-key",
        content=old_question,
    )
    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="old",
        content=old_answer,
    )
    current = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="current",
        idempotency_key="current-key",
        content="current question",
    )

    def measure(messages, projected_summary=""):
        return len(projected_summary) + sum(
            len(str(item.get("content") or "")) for item in messages
        )

    model = _CountingCompactionModel()

    class _Models:
        @staticmethod
        def query_tool_model():
            return model

    executor = object.__new__(AnswerExecutor)
    executor._models = cast(Any, _Models())
    profile = ModelProfile(context_window_tokens=100_000)
    compacted, trace, committed = await executor._compact_fast_history_if_needed(
        host=host,
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="current",
        accepted_user_entry_id=current.user_entry_id,
        targets=(
            HistoryProjectionTarget(
                "fast_generation",
                profile,
                measure,
                proactive_compaction=True,
                require_full_dynamic_reserve=True,
            ),
        ),
        compaction_model_profile=profile,
    )
    return store, session_id, transactions, model, compacted, trace, committed


def test_fast_host_turn_reservation_round_trips_closed_register_codec() -> None:
    value = HostTurnReservation(
        lane_id=LaneId.main(),
        reservation_id="run-1",
        idempotency_key="submission-1",
        user_entry_id=EntryId.new(),
    )
    assert (
        decode_register(
            kind="host_turn_reservation",
            payload=value.canonical_payload(),
        )
        == value
    )


def test_transaction_commit_projection_stamps_entries_and_applies_register_writes() -> None:
    session_id = SessionId.new()
    lane_id = LaneId.new()
    user = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        sequence=1,
        content="question",
        acceptance_id="run",
    )
    reservation = HostTurnReservation(lane_id, "run", "request", user.entry_id)
    snapshot = AgentSessionSnapshot(
        session_id=session_id,
        commit_sequence=1,
        last_entry_sequence=1,
        entries=(user,),
        registers=(
            RegisterRecord(LaneHead(lane_id, user.entry_id), 1),
            RegisterRecord(LaneState(lane_id), 1),
            RegisterRecord(reservation, 1),
        ),
        selected_lane_id=lane_id,
    )
    assistant = AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        parent_entry_id=user.entry_id,
        content="answer",
        stop_reason="stop",
        acceptance_id="run",
    )
    transaction = SessionTransaction.from_parts(
        entries=[assistant],
        register_writes=[
            SetRegister(LaneHead(lane_id, assistant.entry_id)),
            DeleteRegister(reservation.ref),
        ],
        expectations=[
            RegisterExpectation(LaneHead(lane_id, None).ref, 1),
            RegisterExpectation(reservation.ref, 1),
        ],
    )
    commit = TransactionCommit(
        commit_sequence=2,
        appended_sequences=(2,),
        register_sequences=(
            (LaneHead(lane_id, None).ref, 2),
            (reservation.ref, 2),
        ),
    )

    projected = project_transaction_commit(snapshot, transaction, commit)

    assert projected is not None
    assert projected.selected_lane_id == lane_id
    assert projected.entries[0] is user
    assert projected.entries[1].sequence == 2
    assert projected.entries[1].entry_id == assistant.entry_id
    assert next(
        record for record in projected.registers if record.ref == LaneHead(lane_id, None).ref
    ) == RegisterRecord(LaneHead(lane_id, assistant.entry_id), 2)
    assert all(record.ref != reservation.ref for record in projected.registers)
    assert (
        project_transaction_commit(
            snapshot,
            transaction,
            TransactionCommit(
                commit_sequence=3,
                appended_sequences=(2,),
                register_sequences=commit.register_sequences,
            ),
        )
        is None
    )


@pytest.mark.asyncio
async def test_routing_boundary_refresh_observes_a_turn_settled_after_the_initial_load() -> None:
    repository = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    initial = await repository.load(session_id)
    earlier = await _fast_host(repository, session_id)
    await earlier.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="earlier",
        idempotency_key="earlier-key",
        content="earlier question",
    )
    await earlier.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="earlier",
        content="earlier answer",
    )

    boundary = await _reserve_agent_session_boundary(
        repository,
        session_id=session_id,
        fencing_epoch=1,
        previous=initial,
    )

    assert [
        message["content"] for message in project_session_messages(boundary.tree.ancestry(), None)
    ] == ["earlier question", "earlier answer"]


@pytest.mark.asyncio
async def test_fast_host_rejects_refresh_mutation_without_commit_advancement() -> None:
    repository = _UncommittedMutationRepository()
    session_id = SessionId.new()
    host = await _fast_host(repository, session_id)

    with pytest.raises(ValueError, match="changed without commit advancement"):
        await host.snapshot(session_id)


@pytest.mark.asyncio
async def test_fast_host_decodes_long_history_once_across_accept_complete_fail_and_replay() -> None:
    repository = _CountingFastRepository()
    session_id = SessionId.new()
    await _seed_history(repository, session_id, count=1000)
    canonical = await repository.load(session_id)
    host = FastSessionHost(
        repository=repository,
        initial_snapshot=canonical,
        load_settled_result=_no_settled_result,
        fencing_epoch=1,
    )

    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="completed",
        idempotency_key="completed-key",
        content="new question",
    )
    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="completed",
        content="new answer",
    )
    failed = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="failed",
        idempotency_key="failed-key",
        content="failed question",
    )
    await host.fail(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="failed",
    )
    replay = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="failed",
        idempotency_key="failed-key",
        content="failed question",
    )
    boundary = await host.snapshot(session_id)

    assert failed.created is True
    assert replay.created is False
    assert replay.user_entry_id == failed.user_entry_id
    assert repository.load_calls == 1
    assert repository.refresh_calls == 6
    assert repository.decoded_rows == 1000
    assert all(boundary.entries[index] is canonical.entries[index] for index in range(1000))

    # The replaced raw-load seam decoded the whole growing snapshot at each boundary.
    legacy_full_loads = 1 + repository.refresh_calls
    legacy_decoded_rows = 1000 + sum((1000, 1001, 1002, 1003, 1003, 1003))
    assert legacy_full_loads == 7
    assert legacy_decoded_rows == 7012


@pytest.mark.asyncio
async def test_first_host_refresh_observes_lane_fork_committed_after_initial_snapshot() -> None:
    repository = _CountingFastRepository()
    session_id = SessionId.new()
    await _seed_history(repository, session_id, count=2)
    canonical = await repository.load(session_id)
    lane_id = LaneId.new()
    await ensure_session_lane(
        repository=repository,
        snapshot=canonical,
        fencing_epoch=1,
        session_id=session_id,
        lane_id=lane_id,
        source_lane_id=LaneId.main(),
    )
    host = FastSessionHost(
        repository=repository,
        initial_snapshot=canonical,
        load_settled_result=_no_settled_result,
        fencing_epoch=1,
    )

    accepted = await host.accept(
        session_id=session_id,
        lane_id=lane_id,
        reservation_id="branch-run",
        idempotency_key="branch-key",
        content="branch question",
    )
    boundary = await host.snapshot(session_id, selected_lane_id=lane_id)

    assert accepted.created is True
    assert boundary.tree.ancestry(lane_id)[-1].entry_id == accepted.user_entry_id
    assert repository.load_calls == 1
    assert repository.decoded_rows == 2
    assert boundary.selected_lane_id == lane_id
    assert canonical.selected_lane_id == LaneId.main()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("forced", "error"),
    [
        ("conflict", OperationConflictError),
        ("lease", SessionLeaseLostError),
        ("exception", RuntimeError),
    ],
)
async def test_fast_host_invalidates_cache_on_non_authoritative_transaction_outcome(
    forced: str,
    error: type[BaseException],
) -> None:
    repository = _ForcedFastRepository()
    session_id = SessionId.new()
    host = await _fast_host(repository, session_id)
    repository.forced = forced

    with pytest.raises(error):
        await host.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id="forced",
            idempotency_key="forced-key",
            content="question",
        )

    assert session_id not in host._snapshots  # pyright: ignore[reportPrivateUsage]
    repository.forced = None
    await host.snapshot(session_id)
    assert repository.load_calls == 2


@pytest.mark.asyncio
async def test_noncontiguous_commit_metadata_invalidates_instead_of_caching_speculation() -> None:
    repository = _ForcedFastRepository()
    session_id = SessionId.new()
    host = await _fast_host(repository, session_id)
    repository.forced = "noncontiguous"

    accepted = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="malformed",
        idempotency_key="malformed-key",
        content="question",
    )

    assert session_id not in host._snapshots  # pyright: ignore[reportPrivateUsage]
    repository.forced = None
    authoritative = await host.snapshot(session_id)
    assert authoritative.commit_sequence == 1
    assert authoritative.entries[-1].entry_id == accepted.user_entry_id
    assert repository.load_calls == 2


@pytest.mark.asyncio
async def test_concurrent_host_snapshot_waits_for_commit_projection_without_regression() -> None:
    repository = _SlowFastRepository()
    session_id = SessionId.new()
    host = await _fast_host(repository, session_id)
    repository.block_transaction = True
    accepted_task = asyncio.create_task(
        host.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id="concurrent",
            idempotency_key="concurrent-key",
            content="question",
        )
    )
    await repository.transaction_started.wait()
    snapshot_task = asyncio.create_task(host.snapshot(session_id))
    await asyncio.sleep(0)
    assert snapshot_task.done() is False

    repository.release_transaction.set()
    accepted, boundary = await asyncio.gather(accepted_task, snapshot_task)

    assert boundary.commit_sequence == 1
    assert boundary.entries[-1].entry_id == accepted.user_entry_id
    assert host._snapshots[session_id] is boundary  # pyright: ignore[reportPrivateUsage]
    assert repository.load_calls == 1


@pytest.mark.asyncio
async def test_fast_turn_accepts_user_and_reservation_then_settles_assistant() -> None:
    store = MemoryAgentSessionRepository[None]()
    settled_result: dict[str, Any] | None = None

    async def load_settled_result() -> dict[str, Any] | None:
        return settled_result

    session_id = SessionId.new()
    host = await _fast_host(store, session_id, load_settled_result=load_settled_result)

    accepted = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-1",
        idempotency_key="submission-1",
        content="question",
    )
    replay = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-1",
        idempotency_key="submission-1",
        content="question",
    )
    assert accepted.created is True
    assert replay.created is False
    reserved = await store.load(session_id)
    assert [type(entry) for entry in reserved.tree.ancestry()] == [UserMessageEntry]
    assert any(isinstance(record.value, HostTurnReservation) for record in reserved.registers)

    settled_result = {
        "answer": "answer",
        "contexts": {"chunks": []},
        "sources": [],
        "artifacts": [],
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }
    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-1",
        content="answer",
        usage={"input_tokens": 2, "output_tokens": 1},
    )
    settled = await store.load(session_id)
    assert [type(entry) for entry in settled.tree.ancestry()] == [
        UserMessageEntry,
        AssistantMessageEntry,
    ]
    assert not any(isinstance(record.value, HostTurnReservation) for record in settled.registers)
    assistant = settled.tree.ancestry()[-1]
    assert isinstance(assistant, AssistantMessageEntry)
    assert assistant.acceptance_id == "run-1"
    assert assistant.provider_state is None
    settled_replay = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-1",
        idempotency_key="submission-1",
        content="question",
    )
    assert settled_replay.settled is True
    assert settled_replay.settled_payload == settled_result
    assert len((await store.load(session_id)).entries) == 2
    assert (
        await host.fail(
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id="run-1",
        )
        is None
    )


@pytest.mark.asyncio
async def test_staged_fast_result_settles_active_reservation_without_regeneration() -> None:
    store = MemoryAgentSessionRepository[None]()
    settled_result: dict[str, Any] | None = None

    async def load_settled_result() -> dict[str, Any] | None:
        return settled_result

    session_id = SessionId.new()
    host = await _fast_host(store, session_id, load_settled_result=load_settled_result)
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-crashed",
        idempotency_key="submission-crashed",
        content="question",
    )
    settled_result = {
        "answer": "durable answer",
        "contexts": {"chunks": [{"chunk_id": "c1"}]},
        "sources": [{"id": "1"}],
        "artifacts": [{"resource_id": "report"}],
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }

    assert (
        await host.fail(
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id="run-crashed",
        )
        is None
    )
    staged = await store.load(session_id)
    assert any(isinstance(record.value, HostTurnReservation) for record in staged.registers)

    replay = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-crashed",
        idempotency_key="submission-crashed",
        content="question",
    )

    assert replay.settled_payload == settled_result
    assert replay.progress_advanced is True
    snapshot = await store.load(session_id)
    assert [entry.entry_type for entry in snapshot.tree.ancestry()] == [
        "user_message",
        "assistant_message",
    ]
    assistant = snapshot.tree.ancestry()[-1]
    assert isinstance(assistant, AssistantMessageEntry)
    assert assistant.content == "durable answer"
    assert not any(isinstance(record.value, HostTurnReservation) for record in snapshot.registers)


@pytest.mark.asyncio
async def test_fast_compaction_omits_a_failed_user_between_succeeded_and_current_turns() -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    host = await _fast_host(store, session_id)
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-succeeded",
        idempotency_key="submission-succeeded",
        content="successful question",
    )
    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-succeeded",
        content="successful answer",
    )
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-failed",
        idempotency_key="submission-failed",
        content="failed question",
    )
    await host.fail(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-failed",
    )
    current = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-current",
        idempotency_key="submission-current",
        content="current question",
    )
    snapshot = await store.load(session_id)
    ancestry = snapshot.tree.ancestry()
    projection = ContextProjection(
        projection_id=ProjectionId.new(),
        covered_through_sequence=ancestry[1].sequence,
        first_retained_sequence=ancestry[2].sequence,
        summary='{"goal":"successful turn"}',
        covered_through_entry_id=ancestry[1].entry_id,
        first_retained_entry_id=ancestry[2].entry_id,
        source_digest=projection_source_digest([entry.entry_id for entry in ancestry[:2]]),
    )

    history = _project_fast_history_before_current_user(
        snapshot,
        lane_id=LaneId.main(),
        projection=projection,
        accepted_user_entry_id=current.user_entry_id,
    )

    assert len(history.messages) == 1
    assert "successful turn" in str(history.messages[0]["content"])
    assert all(message.get("content") != "failed question" for message in history.messages)
    assert all(message.get("content") != "current question" for message in history.messages)


@pytest.mark.asyncio
async def test_fast_replay_reinstalls_reservation_on_durable_compaction_checkpoint() -> None:
    store = MemoryAgentSessionRepository[None]()
    settled_result: dict[str, Any] | None = None

    async def load_settled_result() -> dict[str, Any] | None:
        return settled_result

    session_id = SessionId.new()
    host = await _fast_host(store, session_id, load_settled_result=load_settled_result)
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-old",
        idempotency_key="submission-old",
        content="old question",
    )
    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-old",
        content="old answer",
    )
    accepted = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-current",
        idempotency_key="submission-current",
        content="current question",
    )
    before = await store.load(session_id)
    replay_history = _project_fast_history_before_current_user(
        before,
        lane_id=LaneId.main(),
        projection=None,
        accepted_user_entry_id=accepted.user_entry_id,
    )
    assert [message["content"] for message in replay_history.messages] == [
        "old question",
        "old answer",
    ]
    ancestry = before.tree.ancestry()
    projection = ContextProjection(
        projection_id=ProjectionId.new(),
        covered_through_sequence=ancestry[1].sequence,
        first_retained_sequence=ancestry[2].sequence,
        summary='{"goal":"old turn"}',
        covered_through_entry_id=ancestry[1].entry_id,
        first_retained_entry_id=ancestry[2].entry_id,
        source_digest=projection_source_digest([entry.entry_id for entry in ancestry[:2]]),
    )

    projected_history = _project_fast_history_before_current_user(
        before,
        lane_id=LaneId.main(),
        projection=projection,
        accepted_user_entry_id=accepted.user_entry_id,
    )
    assert len(projected_history.messages) == 1
    assert "old turn" in str(projected_history.messages[0]["content"])
    assert all(
        message.get("content") != "current question" for message in projected_history.messages
    )

    await host.commit_compaction(
        snapshot=before,
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-current",
        projection=projection,
    )
    compacted = await store.load(session_id)
    assert [entry.entry_type for entry in compacted.tree.ancestry()] == [
        "user_message",
        "assistant_message",
        "user_message",
        "compaction",
    ]
    assert compacted.active_projection == projection

    await host.fail(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-current",
    )
    failed = await store.load(session_id)
    assert not any(isinstance(record.value, HostTurnReservation) for record in failed.registers)
    with pytest.raises(OperationConflictError, match="changed"):
        await host.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id="run-current",
            idempotency_key="submission-current",
            content="changed question",
        )
    assert len((await store.load(session_id)).entries) == 4
    reinstalled = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-current",
        idempotency_key="submission-current",
        content="current question",
    )
    assert reinstalled.created is False
    assert reinstalled.user_entry_id == accepted.user_entry_id
    assert len((await store.load(session_id)).entries) == 4

    class StaleRepository:
        async def load(self, _session_id: SessionId):
            return failed

        async def refresh(self, target: SessionId, *, previous):
            return await store.refresh(target, previous=previous)

        async def transact(self, **kwargs):
            return await store.transact(**kwargs)

    stale_runtime = AgentSessionRuntime[None](
        repository=StaleRepository(),  # type: ignore[arg-type]
        effects=cast(Any, object()),
        tools=(),
        fencing_epoch=1,
    )
    with pytest.raises(OperationConflictError, match="lane_state"):
        await stale_runtime.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            idempotency_key="research-raced-with-fast-reinstall",
            content="research question",
            plan=AgentRunPlan(
                model_role="query",
                context_policy_revision="test",
                tools=(),
            ),
        )
    after_conflict = await store.load(session_id)
    lane_state = after_conflict.tree.lane(LaneId.main()).state.value
    assert isinstance(lane_state, LaneState)
    assert lane_state.active_operation_id is None
    assert any(isinstance(record.value, HostTurnReservation) for record in after_conflict.registers)
    assert len(after_conflict.entries) == 4

    settled_result = {
        "answer": "current answer",
        "contexts": {"chunks": []},
        "sources": [],
        "artifacts": [],
        "usage": {"input_tokens": 4, "output_tokens": 2},
    }
    replay = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-current",
        idempotency_key="submission-current",
        content="current question",
    )

    assert replay.user_entry_id == accepted.user_entry_id
    assert replay.settled_payload == settled_result
    recovered = await store.load(session_id)
    latest = recovered.tree.ancestry()[-1]
    checkpoint = recovered.tree.ancestry()[-2]
    assert isinstance(checkpoint, CompactionEntry)
    assert isinstance(latest, AssistantMessageEntry)
    assert latest.parent_entry_id == checkpoint.entry_id
    assert latest.acceptance_id == "run-current"
    assert not any(isinstance(record.value, HostTurnReservation) for record in recovered.registers)


@pytest.mark.asyncio
async def test_fast_compaction_satisfies_smaller_extract_and_larger_query_profiles() -> None:
    store = _CountingFastRepository()
    session_id = SessionId.new()
    host = await _fast_host(store, session_id)
    old_question = "question " * 4_000
    old_answer = "answer " * 4_000
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="old",
        idempotency_key="old-key",
        content=old_question,
    )
    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="old",
        content=old_answer,
    )
    current = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="current",
        idempotency_key="current-key",
        content="current question",
    )

    def planner_measure(messages, projected_summary=""):
        return len(projected_summary) + sum(
            len(str(item.get("content") or "")) for item in messages
        )

    def generation_measure(messages, projected_summary=""):
        del messages, projected_summary
        return 100

    class _ToolModel:
        @staticmethod
        async def stream_text(**_kwargs):
            yield "## Goal\nPreserve the prior turn."

    class _Models:
        @staticmethod
        def query_tool_model():
            return _ToolModel()

    executor = object.__new__(AnswerExecutor)
    executor._models = cast(Any, _Models())
    query_profile = ModelProfile(context_window_tokens=1_000_000)
    extract_profile = ModelProfile(context_window_tokens=100_000)
    compacted, trace, committed = await executor._compact_fast_history_if_needed(
        host=host,
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="current",
        accepted_user_entry_id=current.user_entry_id,
        targets=(
            HistoryProjectionTarget(
                "fast_planner",
                extract_profile,
                planner_measure,
                proactive_compaction=True,
                require_full_dynamic_reserve=True,
            ),
            HistoryProjectionTarget(
                "fast_generation",
                query_profile,
                generation_measure,
                proactive_compaction=True,
                require_full_dynamic_reserve=True,
            ),
        ),
        compaction_model_profile=query_profile,
    )

    assert committed is True
    planner_trace = trace["fast_compaction_targets"]["fast_planner"]
    generation_trace = trace["fast_compaction_targets"]["fast_generation"]
    assert planner_trace["input_tokens_before"] > planner_trace["input_limit_tokens"]
    assert planner_trace["input_tokens_after"] <= planner_trace["input_limit_tokens"]
    assert generation_trace["input_tokens_before"] <= generation_trace["input_limit_tokens"]
    assert all(message.get("content") != "current question" for message in compacted.messages)
    snapshot = await host.snapshot(session_id)
    assert isinstance(snapshot.tree.ancestry()[-1], CompactionEntry)
    assert snapshot.active_projection is not None
    assert store.load_calls == 1
    assert store.decoded_rows == 0


@pytest.mark.asyncio
async def test_ambiguous_fast_compaction_commit_recovers_applied_projection() -> None:
    (
        store,
        session_id,
        transactions,
        model,
        compacted,
        trace,
        committed,
    ) = await _run_fault_injected_compaction(applied=True)

    assert transactions.injected is True
    assert transactions.load_calls == 2
    assert model.calls == 1
    assert committed is True
    assert trace["fast_compaction_recovered"] is True
    assert "input_tokens_before" not in trace
    assert all(message.get("content") != "current question" for message in compacted.messages)
    snapshot = await store.load(session_id)
    assert sum(isinstance(entry, CompactionEntry) for entry in snapshot.entries) == 1


@pytest.mark.asyncio
async def test_genuine_fast_compaction_cas_conflict_reprepares_from_reload() -> None:
    (
        store,
        session_id,
        transactions,
        model,
        _compacted,
        trace,
        committed,
    ) = await _run_fault_injected_compaction(applied=False)

    assert transactions.injected is True
    assert transactions.load_calls == 2
    assert model.calls == 2
    assert committed is True
    assert trace["fast_compaction_attempt"] > 1
    assert trace["fast_compaction_retries"][-1]["stage"] == "commit"
    snapshot = await store.load(session_id)
    assert sum(isinstance(entry, CompactionEntry) for entry in snapshot.entries) == 1


@pytest.mark.asyncio
async def test_failed_fast_compaction_commits_nothing_and_traces_failure(caplog) -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    host = await _fast_host(store, session_id)
    old_question = "question " * 4_000
    old_answer = "answer " * 4_000
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="old",
        idempotency_key="old-key",
        content=old_question,
    )
    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="old",
        content=old_answer,
    )
    current = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="current",
        idempotency_key="current-key",
        content="current question",
    )

    class _Synthesizer:
        @staticmethod
        def history_input_measure(_query: str):
            def measure(messages, projected_summary=""):
                return len(projected_summary) + sum(
                    len(str(item.get("content") or "")) for item in messages
                )

            return measure

    class _ToolModel:
        @staticmethod
        async def stream_text(**_kwargs):
            yield "summary without a required goal heading"

    class _Models:
        @staticmethod
        def answer_synthesizer(_profile):
            return _Synthesizer()

        @staticmethod
        def query_tool_model():
            return _ToolModel()

    executor = object.__new__(AnswerExecutor)
    executor._models = cast(Any, _Models())
    profile = ModelProfile(context_window_tokens=100_000)
    with pytest.raises(RunExecutionError) as caught:
        await executor._compact_fast_history_if_needed(
            host=host,
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id="current",
            accepted_user_entry_id=current.user_entry_id,
            targets=(
                HistoryProjectionTarget(
                    "fast_generation",
                    profile,
                    _Synthesizer.history_input_measure("current question"),
                    proactive_compaction=True,
                    require_full_dynamic_reserve=True,
                ),
            ),
            compaction_model_profile=profile,
        )

    assert caught.value.kind == "compaction_failed"
    snapshot = await store.load(session_id)
    assert not any(isinstance(entry, CompactionEntry) for entry in snapshot.entries)
    assert snapshot.active_projection is None
    failure_records = [record for record in caplog.records if hasattr(record, "trace")]
    assert failure_records[-1].trace["compaction_failed"]["attempts"]


@pytest.mark.asyncio
async def test_staged_fast_result_rejects_an_interleaved_lane_head() -> None:
    store = MemoryAgentSessionRepository[None]()
    settled_result: dict[str, Any] | None = None

    async def load_settled_result() -> dict[str, Any] | None:
        return settled_result

    session_id = SessionId.new()
    host = await _fast_host(store, session_id, load_settled_result=load_settled_result)
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-crashed",
        idempotency_key="submission-crashed",
        content="original question",
    )
    await host.fail(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-crashed",
    )
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-other",
        idempotency_key="submission-other",
        content="interleaved question",
    )
    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-other",
        content="interleaved answer",
    )
    settled_result = {
        "answer": "durable answer",
        "contexts": {"chunks": []},
        "sources": [],
        "artifacts": [],
        "usage": {},
    }

    with pytest.raises(OperationConflictError, match="accepted lane head"):
        await host.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id="run-crashed",
            idempotency_key="submission-crashed",
            content="original question",
        )

    snapshot = await store.load(session_id)
    assert [getattr(entry, "acceptance_id", None) for entry in snapshot.tree.ancestry()] == [
        "run-crashed",
        "run-other",
        "run-other",
    ]


@pytest.mark.asyncio
async def test_fast_failure_keeps_replay_user_but_omits_it_from_model_history() -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    host = await _fast_host(store, session_id)
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-failed",
        idempotency_key="submission-failed",
        content="unanswered",
    )

    await host.fail(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-failed",
    )
    snapshot = await store.load(session_id)
    [entry] = snapshot.tree.ancestry()
    assert isinstance(entry, UserMessageEntry)
    assert entry.content == "unanswered"
    assert project_session_messages(snapshot.tree.ancestry(), None) == []
    assert not any(isinstance(record.value, HostTurnReservation) for record in snapshot.registers)

    recovered = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-failed",
        idempotency_key="submission-failed",
        content="unanswered",
    )
    assert recovered.created is False
    assert len((await store.load(session_id)).entries) == 1
    await host.fail(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-failed",
    )
    with pytest.raises(OperationConflictError, match="changed"):
        await host.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id="run-failed",
            idempotency_key="submission-failed",
            content="different",
        )


@pytest.mark.asyncio
async def test_runtime_accept_rejects_a_fast_reservation_on_the_same_lane() -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    host = await _fast_host(store, session_id)
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-fast",
        idempotency_key="submission-fast",
        content="fast question",
    )
    runtime = AgentSessionRuntime[None](
        repository=store,
        effects=cast(Any, object()),
        tools=(),
        fencing_epoch=1,
    )

    with pytest.raises(OperationConflictError, match="active Host turn"):
        await runtime.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            idempotency_key="research-run",
            content="research question",
            plan=AgentRunPlan(
                model_role="query",
                context_policy_revision="test",
                tools=(),
            ),
        )
