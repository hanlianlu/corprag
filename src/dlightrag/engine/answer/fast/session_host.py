# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-owned Host operations over the canonical Agent Session Tree."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from dlightrag.engine.agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    UserMessageEntry,
)
from dlightrag.engine.agent.session.ids import EntryId, LaneId, SessionId
from dlightrag.engine.agent.session.projection import ContextProjection, projection_source_digest
from dlightrag.engine.agent.session.registers import (
    ContextProjectionRegister,
    DeleteRegister,
    HostTurnReservation,
    LaneHead,
    LaneState,
    RegisterRecord,
    RegisterRef,
    SessionFault,
    SetRegister,
)
from dlightrag.engine.agent.session.repository import (
    AgentSessionRepository,
    AgentSessionSnapshot,
    project_transaction_commit,
    validate_snapshot_refresh,
)
from dlightrag.engine.agent.session.runtime import OperationConflictError, SessionLeaseLostError
from dlightrag.engine.agent.session.transactions import (
    RegisterConflict,
    RegisterExpectation,
    SessionTransaction,
    TransactionCommit,
    TransactionLeaseLost,
)
from dlightrag.engine.runtime.coordinator import LeaseLostError
from dlightrag.engine.runtime.errors import RunExecutionError


@dataclass(frozen=True, slots=True)
class AcceptedFastTurn:
    session_id: SessionId
    lane_id: LaneId
    reservation_id: str
    user_entry_id: EntryId
    created: bool
    settled_payload: Mapping[str, Any] | None = None
    progress_advanced: bool = False

    @property
    def settled(self) -> bool:
        return self.settled_payload is not None


class FastSessionHost:
    """Atomically reserve and settle Fast Host turns on one Session Lane."""

    def __init__(
        self,
        *,
        repository: AgentSessionRepository[Any],
        initial_snapshot: AgentSessionSnapshot,
        load_settled_result: Callable[[], Awaitable[Mapping[str, Any] | None]],
        fencing_epoch: int,
    ) -> None:
        self._repository = repository
        self._snapshots = {initial_snapshot.session_id: initial_snapshot}
        self._snapshot_locks = {initial_snapshot.session_id: asyncio.Lock()}
        self._load_settled_result = load_settled_result
        self._fencing_epoch = fencing_epoch

    async def snapshot(
        self,
        session_id: SessionId,
        *,
        selected_lane_id: LaneId | None = None,
        force_reload: bool = False,
    ) -> AgentSessionSnapshot:
        """Return one refreshed boundary without re-decoding its historical prefix."""
        async with self._snapshot_lock(session_id):
            if force_reload:
                self._snapshots.pop(session_id, None)
            previous = self._snapshots.get(session_id)
            try:
                snapshot = (
                    await self._repository.load(session_id)
                    if previous is None
                    else await self._repository.refresh(session_id, previous=previous)
                )
            except BaseException:
                self._snapshots.pop(session_id, None)
                raise
            if previous is not None:
                try:
                    validate_snapshot_refresh(
                        session_id,
                        previous=previous,
                        snapshot=snapshot,
                    )
                except ValueError:
                    self._snapshots.pop(session_id, None)
                    raise
            elif snapshot.session_id != session_id:
                self._snapshots.pop(session_id, None)
                raise ValueError("Agent Session repository returned another Session")
            self._snapshots[session_id] = snapshot
            if selected_lane_id is not None and snapshot.selected_lane_id != selected_lane_id:
                return replace(snapshot, selected_lane_id=selected_lane_id)
            return snapshot

    def _snapshot_lock(self, session_id: SessionId) -> asyncio.Lock:
        lock = self._snapshot_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._snapshot_locks[session_id] = lock
        return lock

    async def accept(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
        reservation_id: str,
        idempotency_key: str,
        content: Any,
    ) -> AcceptedFastTurn:
        snapshot = await self.snapshot(session_id)
        _reject_fault(snapshot)
        settled_payload = await self._load_settled_result()
        reservation_ref = RegisterRef("host_turn_reservation", lane_id.value)
        existing = _register(snapshot, reservation_ref)
        if existing is not None:
            value = existing.value
            if not isinstance(value, HostTurnReservation):
                raise TypeError("Host turn reservation register has the wrong value type")
            if value.reservation_id != reservation_id or value.idempotency_key != idempotency_key:
                raise OperationConflictError("Lane already owns another Fast Host turn")
            user = _accepted_user(snapshot, lane_id, reservation_id)
            if user.entry_id != value.user_entry_id or user.content != content:
                raise OperationConflictError("Fast turn replay changed its accepted content")
            if settled_payload is None:
                return AcceptedFastTurn(
                    session_id,
                    lane_id,
                    reservation_id,
                    value.user_entry_id,
                    created=False,
                )
            ancestry = snapshot.tree.ancestry(lane_id)
            latest = ancestry[-1] if ancestry else None
            if isinstance(latest, AssistantMessageEntry):
                if latest.acceptance_id != reservation_id or not _is_reserved_turn_head(
                    snapshot,
                    user_entry_id=user.entry_id,
                    head_entry_id=latest.parent_entry_id,
                ):
                    raise OperationConflictError("settled Fast result lost its accepted lane head")
                if latest.content != _settled_answer(settled_payload):
                    raise OperationConflictError(
                        "settled Fast Assistant disagrees with its durable Host result"
                    )
            elif latest is None or not _is_reserved_turn_head(
                snapshot,
                user_entry_id=user.entry_id,
                head_entry_id=latest.entry_id,
            ):
                raise OperationConflictError("settled Fast result lost its reserved User lane head")
            await self.complete(
                session_id=session_id,
                lane_id=lane_id,
                reservation_id=reservation_id,
                content=_settled_answer(settled_payload),
                usage=_settled_usage(settled_payload),
            )
            return AcceptedFastTurn(
                session_id,
                lane_id,
                reservation_id,
                value.user_entry_id,
                created=False,
                settled_payload=settled_payload,
                progress_advanced=True,
            )

        try:
            lane = snapshot.tree.lane(lane_id)
        except KeyError:
            if snapshot.entries or lane_id != LaneId.main():
                raise OperationConflictError(f"unknown Lane: {lane_id.value}") from None
            if settled_payload is not None:
                raise OperationConflictError(
                    "settled Fast result lost its accepted User Entry"
                ) from None
            parent = None
            head_sequence = None
            state_sequence = None
            lane_state = LaneState(lane_id)
            state_write = True
        else:
            lane_state = lane.state.value
            if not isinstance(lane_state, LaneState):
                raise TypeError("Lane State register has the wrong value type")
            if lane.archived:
                raise OperationConflictError("an archived Lane cannot accept a Fast turn")
            if lane_state.active_operation_id is not None:
                raise OperationConflictError("Lane already owns an Agent operation")
            parent = lane.head_entry_id
            head_sequence = lane.head.sequence
            state_sequence = lane.state.sequence
            state_write = False
            ancestry = snapshot.tree.ancestry(lane_id)
            if ancestry:
                latest = ancestry[-1]
                if (
                    isinstance(latest, AssistantMessageEntry)
                    and latest.acceptance_id == reservation_id
                ):
                    user = _accepted_user(snapshot, lane_id, reservation_id)
                    if user.content != content:
                        raise OperationConflictError(
                            "Fast turn replay changed its accepted content"
                        )
                    if settled_payload is None:
                        raise OperationConflictError(
                            "settled Fast turn lost its durable Host result"
                        )
                    if not _is_reserved_turn_head(
                        snapshot,
                        user_entry_id=user.entry_id,
                        head_entry_id=latest.parent_entry_id,
                    ) or latest.content != _settled_answer(settled_payload):
                        raise OperationConflictError(
                            "settled Fast Assistant disagrees with its durable Host result"
                        )
                    return AcceptedFastTurn(
                        session_id,
                        lane_id,
                        reservation_id,
                        user.entry_id,
                        created=False,
                        settled_payload=settled_payload,
                    )
                replay_user = _unanswered_user_at_head(snapshot, lane_id)
                if replay_user is not None and replay_user.acceptance_id == reservation_id:
                    if replay_user.content != content:
                        raise OperationConflictError(
                            "Fast turn replay changed its accepted content"
                        )
                    reservation = HostTurnReservation(
                        lane_id=lane_id,
                        reservation_id=reservation_id,
                        idempotency_key=idempotency_key,
                        user_entry_id=replay_user.entry_id,
                    )
                    await self._transact(
                        session_id,
                        SessionTransaction.from_parts(
                            register_writes=[
                                SetRegister(lane_state),
                                SetRegister(reservation),
                            ],
                            expectations=[
                                RegisterExpectation(lane.head.ref, lane.head.sequence),
                                RegisterExpectation(lane_state.ref, state_sequence),
                                RegisterExpectation(reservation.ref, None),
                            ],
                        ),
                    )
                    if settled_payload is None:
                        return AcceptedFastTurn(
                            session_id,
                            lane_id,
                            reservation_id,
                            replay_user.entry_id,
                            created=False,
                        )
                    await self.complete(
                        session_id=session_id,
                        lane_id=lane_id,
                        reservation_id=reservation_id,
                        content=_settled_answer(settled_payload),
                        usage=_settled_usage(settled_payload),
                    )
                    return AcceptedFastTurn(
                        session_id,
                        lane_id,
                        reservation_id,
                        replay_user.entry_id,
                        created=False,
                        settled_payload=settled_payload,
                        progress_advanced=True,
                    )

        if settled_payload is not None:
            raise OperationConflictError("settled Fast result lost its accepted lane head")
        message = UserMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=datetime.now(UTC),
            parent_entry_id=parent,
            content=content,
            acceptance_id=reservation_id,
        )
        reservation = HostTurnReservation(
            lane_id=lane_id,
            reservation_id=reservation_id,
            idempotency_key=idempotency_key,
            user_entry_id=message.entry_id,
        )
        writes = [
            SetRegister(LaneHead(lane_id, message.entry_id)),
            *((SetRegister(lane_state),) if state_write else ()),
            SetRegister(reservation),
        ]
        expectations = [
            RegisterExpectation(writes[0].ref, head_sequence),
            *((RegisterExpectation(lane_state.ref, state_sequence),) if state_write else ()),
            RegisterExpectation(reservation.ref, None),
        ]
        await self._transact(
            session_id,
            SessionTransaction.from_parts(
                entries=[message],
                register_writes=writes,
                expectations=expectations,
            ),
        )
        return AcceptedFastTurn(
            session_id,
            lane_id,
            reservation_id,
            message.entry_id,
            created=True,
            progress_advanced=True,
        )

    async def commit_compaction(
        self,
        *,
        snapshot: AgentSessionSnapshot,
        session_id: SessionId,
        lane_id: LaneId,
        reservation_id: str,
        projection: ContextProjection,
    ) -> TransactionCommit:
        """CAS one projection against the exact snapshot it was prepared from."""
        if snapshot.session_id != session_id:
            raise ValueError("Fast compaction snapshot belongs to another Session")
        _reject_fault(snapshot)
        reservation = _reservation(snapshot, lane_id)
        if reservation is None:
            raise OperationConflictError("Fast Host turn reservation identity changed")
        reservation_value = reservation.value
        if not isinstance(reservation_value, HostTurnReservation):
            raise TypeError("Host turn reservation register has the wrong value type")
        if reservation_value.reservation_id != reservation_id:
            raise OperationConflictError("Fast Host turn reservation identity changed")
        user = _accepted_user(snapshot, lane_id, reservation_id)
        lane = snapshot.tree.lane(lane_id)
        head = lane.head
        if not isinstance(head.value, LaneHead):
            raise TypeError("Lane Head register has the wrong value type")
        ancestry = snapshot.tree.ancestry(lane_id)
        latest = ancestry[-1] if ancestry else None
        if latest is None or not _is_reserved_turn_head(
            snapshot,
            user_entry_id=user.entry_id,
            head_entry_id=latest.entry_id,
        ):
            raise OperationConflictError("Fast Host turn lane head changed before compaction")
        previous_projection = _projection(snapshot, lane_id)
        if previous_projection is not None:
            previous_value = previous_projection.value
            if not isinstance(previous_value, ContextProjectionRegister):
                raise TypeError("Context projection register has the wrong value type")
            if (
                projection.covered_through_sequence
                <= previous_value.projection.covered_through_sequence
            ):
                raise OperationConflictError("Fast compaction projection does not advance")
        if not _projection_belongs_to_lane(snapshot, lane_id, projection):
            raise OperationConflictError("Fast compaction projection source changed")

        entry = CompactionEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=datetime.now(UTC),
            parent_entry_id=head.value.entry_id,
            projection_id=projection.projection_id,
            summary=projection.summary,
            covered_through_sequence=projection.covered_through_sequence,
            first_retained_sequence=projection.first_retained_sequence,
            covered_through_entry_id=projection.covered_through_entry_id,
            first_retained_entry_id=projection.first_retained_entry_id,
            source_digest=projection.source_digest,
        )
        projection_value = ContextProjectionRegister(lane_id, projection)
        return await self._transact(
            session_id,
            SessionTransaction.from_parts(
                entries=[entry],
                register_writes=[
                    SetRegister(LaneHead(lane_id, entry.entry_id)),
                    SetRegister(projection_value),
                ],
                expectations=[
                    RegisterExpectation(head.ref, head.sequence),
                    RegisterExpectation(
                        projection_value.ref,
                        previous_projection.sequence if previous_projection is not None else None,
                    ),
                    RegisterExpectation(reservation.ref, reservation.sequence),
                ],
            ),
        )

    async def complete(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
        reservation_id: str,
        content: str,
        usage: Any | None = None,
        cost: Any | None = None,
    ) -> TransactionCommit | None:
        snapshot = await self.snapshot(session_id)
        _reject_fault(snapshot)
        reservation_record = _reservation(snapshot, lane_id)
        if reservation_record is None:
            ancestry = snapshot.tree.ancestry(lane_id)
            if (
                ancestry
                and isinstance(ancestry[-1], AssistantMessageEntry)
                and ancestry[-1].acceptance_id == reservation_id
            ):
                return None
            raise OperationConflictError("Fast Host turn reservation is not active")
        if not isinstance(reservation_record.value, HostTurnReservation):
            raise TypeError("Host turn reservation register has the wrong value type")
        if reservation_record.value.reservation_id != reservation_id:
            raise OperationConflictError("Fast Host turn reservation identity changed")
        head = snapshot.tree.lane(lane_id).head
        if not isinstance(head.value, LaneHead):
            raise TypeError("Lane Head register has the wrong value type")
        ancestry = snapshot.tree.ancestry(lane_id)
        latest = ancestry[-1] if ancestry else None
        if isinstance(latest, AssistantMessageEntry):
            if (
                latest.acceptance_id != reservation_id
                or not _is_reserved_turn_head(
                    snapshot,
                    user_entry_id=reservation_record.value.user_entry_id,
                    head_entry_id=latest.parent_entry_id,
                )
                or latest.content != content
            ):
                raise OperationConflictError("Fast Host turn lane head changed")
            return await self._transact(
                session_id,
                SessionTransaction.from_parts(
                    register_writes=[DeleteRegister(reservation_record.ref)],
                    expectations=[
                        RegisterExpectation(
                            reservation_record.ref,
                            reservation_record.sequence,
                        )
                    ],
                ),
            )
        if latest is None or not _is_reserved_turn_head(
            snapshot,
            user_entry_id=reservation_record.value.user_entry_id,
            head_entry_id=latest.entry_id,
        ):
            raise OperationConflictError("Fast Host turn lane head changed")
        entry = AssistantMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=datetime.now(UTC),
            parent_entry_id=head.value.entry_id,
            content=content,
            stop_reason="stop",
            usage=usage,
            cost=cost,
            acceptance_id=reservation_id,
        )
        return await self._transact(
            session_id,
            SessionTransaction.from_parts(
                entries=[entry],
                register_writes=[
                    SetRegister(LaneHead(lane_id, entry.entry_id)),
                    DeleteRegister(reservation_record.ref),
                ],
                expectations=[
                    RegisterExpectation(head.ref, head.sequence),
                    RegisterExpectation(reservation_record.ref, reservation_record.sequence),
                ],
            ),
        )

    async def fail(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
        reservation_id: str,
    ) -> TransactionCommit | None:
        if await self._load_settled_result() is not None:
            return None
        snapshot = await self.snapshot(session_id)
        reservation = _reservation(snapshot, lane_id)
        if reservation is None:
            ancestry = snapshot.tree.ancestry(lane_id)
            if ancestry and isinstance(ancestry[-1], AssistantMessageEntry):
                if ancestry[-1].acceptance_id == reservation_id:
                    return None
            unanswered = _unanswered_user_at_head(snapshot, lane_id)
            if unanswered is not None and unanswered.acceptance_id == reservation_id:
                return None
            raise OperationConflictError("Fast Host turn reservation is not active")
        if not isinstance(reservation.value, HostTurnReservation):
            raise TypeError("Host turn reservation register has the wrong value type")
        if reservation.value.reservation_id != reservation_id:
            raise OperationConflictError("Fast Host turn reservation identity changed")
        return await self._transact(
            session_id,
            SessionTransaction.from_parts(
                register_writes=[DeleteRegister(reservation.ref)],
                expectations=[RegisterExpectation(reservation.ref, reservation.sequence)],
            ),
        )

    async def _transact(
        self,
        session_id: SessionId,
        transaction: SessionTransaction[Any],
    ) -> TransactionCommit:
        # Repository outcomes and cache publication are ordered per Session.
        async with self._snapshot_lock(session_id):
            try:
                outcome = await self._repository.transact(
                    session_id=session_id,
                    fencing_epoch=self._fencing_epoch,
                    transaction=transaction,
                )
            except BaseException:
                # The write may have committed before its acknowledgement was lost.
                self._snapshots.pop(session_id, None)
                raise
            if isinstance(outcome, RegisterConflict):
                self._snapshots.pop(session_id, None)
                raise OperationConflictError(
                    f"register {outcome.ref.kind}:{outcome.ref.key} changed"
                )
            if isinstance(outcome, TransactionLeaseLost):
                self._snapshots.pop(session_id, None)
                raise SessionLeaseLostError(session_id.value)
            previous = self._snapshots.get(session_id)
            if previous is not None:
                projected = project_transaction_commit(previous, transaction, outcome)
                if projected is None:
                    self._snapshots.pop(session_id, None)
                else:
                    self._snapshots[session_id] = projected
            return outcome


async def ensure_session_lane(
    *,
    repository: AgentSessionRepository[Any],
    snapshot: AgentSessionSnapshot,
    fencing_epoch: int,
    session_id: SessionId,
    lane_id: LaneId,
    source_lane_id: LaneId | None,
) -> None:
    """Open or fork a Lane from the executor's authoritative cold-load boundary."""
    if snapshot.session_id != session_id:
        raise ValueError("Agent Session Lane snapshot belongs to another Session")
    try:
        snapshot.tree.lane(lane_id)
        return
    except KeyError:
        pass
    if source_lane_id is None:
        if snapshot.commit_sequence == 0 and lane_id == LaneId.main():
            return
        raise RunExecutionError("agent_session_conflict", "Agent Lane mapping is missing.")
    try:
        source = snapshot.tree.lane(source_lane_id)
        target = source.head_entry_id
        if target is None or not snapshot.tree.is_stable_checkpoint(target):
            raise ValueError("source Lane is not a stable fork checkpoint")
        head = LaneHead(lane_id, target)
        state = LaneState(lane_id)
        outcome = await repository.transact(
            session_id=session_id,
            fencing_epoch=fencing_epoch,
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(head), SetRegister(state)],
                expectations=[
                    RegisterExpectation(head.ref, None),
                    RegisterExpectation(state.ref, None),
                ],
            ),
        )
    except (KeyError, ValueError) as exc:
        raise RunExecutionError(
            "agent_session_conflict",
            "The Agent Session changed before its branch could be created.",
        ) from exc
    if isinstance(outcome, TransactionLeaseLost):
        raise LeaseLostError
    if isinstance(outcome, RegisterConflict):
        raise RunExecutionError(
            "agent_session_conflict",
            "The Agent Lane changed before its branch could be created.",
        )


def _register(snapshot: AgentSessionSnapshot, ref: RegisterRef) -> RegisterRecord | None:
    return next((record for record in snapshot.registers if record.ref == ref), None)


def _reject_fault(snapshot: AgentSessionSnapshot) -> None:
    if any(isinstance(record.value, SessionFault) for record in snapshot.registers):
        raise OperationConflictError("a faulted Session cannot accept a Fast turn")


def _accepted_user(
    snapshot: AgentSessionSnapshot,
    lane_id: LaneId,
    reservation_id: str,
) -> UserMessageEntry:
    user = next(
        (
            entry
            for entry in reversed(snapshot.tree.ancestry(lane_id))
            if isinstance(entry, UserMessageEntry) and entry.acceptance_id == reservation_id
        ),
        None,
    )
    if user is None:
        raise OperationConflictError("Fast turn lost its accepted User Entry")
    return user


def _settled_answer(payload: Mapping[str, Any]) -> str:
    answer = payload.get("answer")
    if not isinstance(answer, str):
        raise OperationConflictError("settled Fast Host result has no canonical answer")
    return answer


def _settled_usage(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    usage = payload.get("usage")
    return usage if isinstance(usage, Mapping) else None


def _unanswered_user_at_head(
    snapshot: AgentSessionSnapshot,
    lane_id: LaneId,
) -> UserMessageEntry | None:
    """Return the User at a Head made only of its following checkpoints."""
    ancestry = snapshot.tree.ancestry(lane_id)
    if not ancestry:
        return None
    by_id = {entry.entry_id: entry for entry in ancestry}
    current = ancestry[-1]
    seen: set[EntryId] = set()
    while isinstance(current, CompactionEntry) and current.entry_id not in seen:
        seen.add(current.entry_id)
        if current.parent_entry_id is None:
            return None
        parent = by_id.get(current.parent_entry_id)
        if parent is None:
            return None
        current = parent
    return current if isinstance(current, UserMessageEntry) else None


def _projection_belongs_to_lane(
    snapshot: AgentSessionSnapshot,
    lane_id: LaneId,
    projection: ContextProjection,
) -> bool:
    """Verify a candidate's branch identities against its preparation snapshot."""
    branch = [
        entry for entry in snapshot.tree.ancestry(lane_id) if not isinstance(entry, CompactionEntry)
    ]
    covered_index = next(
        (
            index
            for index, entry in enumerate(branch)
            if entry.entry_id == projection.covered_through_entry_id
        ),
        None,
    )
    if covered_index is None:
        return False
    if (
        projection_source_digest([entry.entry_id for entry in branch[: covered_index + 1]])
        != projection.source_digest
    ):
        return False
    if projection.first_retained_entry_id is None:
        return covered_index == len(branch) - 1
    retained_index = next(
        (
            index
            for index, entry in enumerate(branch)
            if entry.entry_id == projection.first_retained_entry_id
        ),
        None,
    )
    return retained_index is not None and retained_index > covered_index


def _is_reserved_turn_head(
    snapshot: AgentSessionSnapshot,
    *,
    user_entry_id: EntryId,
    head_entry_id: EntryId | None,
) -> bool:
    """Whether a Head is the accepted User or its compaction-checkpoint chain."""
    by_id = {entry.entry_id: entry for entry in snapshot.entries}
    current = head_entry_id
    seen: set[EntryId] = set()
    while current is not None and current not in seen:
        if current == user_entry_id:
            return True
        seen.add(current)
        entry = by_id.get(current)
        if not isinstance(entry, CompactionEntry):
            return False
        current = entry.parent_entry_id
    return False


def _projection(
    snapshot: AgentSessionSnapshot,
    lane_id: LaneId,
) -> RegisterRecord | None:
    record = _register(snapshot, RegisterRef("context_projection", lane_id.value))
    if record is None:
        return None
    if not isinstance(record.value, ContextProjectionRegister):
        raise TypeError("Context projection register has the wrong value type")
    return record


def _reservation(
    snapshot: AgentSessionSnapshot,
    lane_id: LaneId,
) -> RegisterRecord | None:
    record = _register(snapshot, RegisterRef("host_turn_reservation", lane_id.value))
    if record is None:
        return None
    if not isinstance(record.value, HostTurnReservation):
        raise TypeError("Host turn reservation register has the wrong value type")
    return record


__all__ = ["AcceptedFastTurn", "FastSessionHost", "ensure_session_lane"]
