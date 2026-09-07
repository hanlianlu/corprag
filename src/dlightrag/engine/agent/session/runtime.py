# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Deep durable Agent Session Runtime.

The Runtime is the sole interpreter of total OperationState. It plans one
closed action, persists every external-effect intent, executes through typed
ports, and atomically settles Runtime state, Entry placement, and HostDelta.
Recovery drives the same interpreter over the current registers.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from dlightrag.engine.agent.session.effects import (
    ReplayPolicy,
    ToolResultEntry,
    ToolResultOutcome,
    canonical_json,
)
from dlightrag.engine.agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    ControlMessageEntry,
    SessionEntry,
    ToolResultMessageEntry,
    UserMessageEntry,
)
from dlightrag.engine.agent.session.ids import (
    AttemptId,
    EntryId,
    IntentId,
    LaneId,
    OperationId,
    SessionId,
)
from dlightrag.engine.agent.session.interpreter import (
    TOOL_DISPOSITION_OUTCOME,
    AssembleProviderRequest,
    BeginToolEffect,
    CallProvider,
    CloseCancellationPosition,
    CommitSyntheticToolResult,
    CompleteOperation,
    ConsumeSteer,
    ContinueAfterToolBatch,
    FinishCancellation,
    NextAction,
    NoAction,
    RecoverToolEffect,
    RunCompaction,
    next_action,
)
from dlightrag.engine.agent.session.operation import (
    AcceptedSteer,
    Cancelling,
    CompactionPending,
    CompletionReady,
    OperationCancelled,
    OperationCompleted,
    OperationFailed,
    OperationMeta,
    ProviderRequestPending,
    ReadyForProvider,
    RunOperationState,
    TerminalFailureKind,
    ToolBatchItem,
    ToolBatchPlan,
    ToolBatchReady,
    ToolCallDisposition,
    ToolEffectPending,
)
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.projection import ContextProjection
from dlightrag.engine.agent.session.registers import (
    ContextProjectionRegister,
    DeleteRegister,
    FollowUpInput,
    HostTurnReservation,
    LaneHead,
    LaneState,
    OperationMetaRegister,
    OperationStateRegister,
    PendingInput,
    RegisterRecord,
    RegisterRef,
    RegisterWrite,
    RequestSnapshot,
    SessionFault,
    SetRegister,
    ToolArguments,
)
from dlightrag.engine.agent.session.repository import (
    AgentSessionCursor,
    AgentSessionRepository,
    AgentSessionSnapshot,
    project_transaction_commit,
    validate_snapshot_refresh,
)
from dlightrag.engine.agent.session.transactions import (
    HostDeltaSettlement,
    RegisterConflict,
    RegisterExpectation,
    SessionTransaction,
    TransactionCommit,
    TransactionLeaseLost,
)
from dlightrag.engine.agent.tools.contracts import AgentTool
from dlightrag.engine.ai.messages import AssistantTurn
from dlightrag.engine.dependencies import ProviderUnavailableError

logger = logging.getLogger(__name__)


class AgentSessionRuntimeError(RuntimeError):
    """Base typed failure at the approved Runtime seam."""


class OperationConflictError(AgentSessionRuntimeError):
    """An exact operation/Lane register changed under this owner."""


class OperationNotFoundError(AgentSessionRuntimeError):
    """The requested operation is not present in this Session."""


class OperationIdempotencyConflict(AgentSessionRuntimeError):
    """An idempotency key was reused with another acceptance payload."""


class SessionLeaseLostError(AgentSessionRuntimeError):
    """The Runtime's fencing epoch no longer owns this Session."""


class AgentOperationCancelled(Exception):
    """A typed Host cancellation crossing an effect port without reclassification."""

    def __init__(self, reason: BaseException) -> None:
        super().__init__(str(reason))
        self.reason = reason


class OperationEffectFailed(Exception):
    """Typed ordinary effect/Plan/context outcome that leaves Session usable."""

    kind: TerminalFailureKind
    detail: str

    def __init__(self, kind: TerminalFailureKind, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


class ProviderContextOverflow(Exception):
    """The exact provider request exceeded context and requires compaction."""


class ProviderAttemptFailed(Exception):
    """Typed provider failure used for bounded durable recovery."""

    def __init__(self, detail: str, *, retryable: bool) -> None:
        super().__init__(detail)
        self.detail = detail
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class CompactionRequired:
    """Marker that the exact Plan-bound context requires compaction."""


@dataclass(frozen=True, slots=True)
class CompactionResult:
    entry: CompactionEntry
    projection: ContextProjection


@dataclass(frozen=True, slots=True)
class ToolEffectResult[HostDeltaT]:
    result: ToolResultEntry
    host_delta: HostDeltaT | None


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    session_id: SessionId
    lane_id: LaneId
    operation_id: OperationId
    meta: OperationMeta
    state: RunOperationState
    snapshot: AgentSessionSnapshot


type AgentSessionEventKind = Literal[
    "operation_accepted",
    "provider_request_committed",
    "provider_attempt_started",
    "provider_attempt_unknown",
    "provider_context_overflow",
    "assistant_committed",
    "tool_intent_committed",
    "tool_replay_started",
    "tool_result_committed",
    "tool_batch_completed",
    "steer_accepted",
    "steer_consumed",
    "follow_up_queued",
    "cancel_requested",
    "operation_completed",
    "operation_cancelled",
    "operation_failed",
    "compaction_required",
    "compaction_retry",
    "compaction_committed",
    "model_start",
    "model_end",
    "provider_delta",
    "tool_start",
    "tool_update",
]


@dataclass(frozen=True, slots=True)
class AgentSessionEvent:
    """Observe-only event; durable events carry their commit sequence."""

    kind: AgentSessionEventKind
    session_id: SessionId
    lane_id: LaneId
    operation_id: OperationId
    commit_sequence: int | None
    data: Mapping[str, Any]
    ephemeral: bool = False


EventSink = Callable[[AgentSessionEvent], Awaitable[None]]


class AgentRuntimeEffects[HostDeltaT](Protocol):
    """Closed effect adapters. Context assembly must be side-effect free."""

    async def assemble_request(
        self, context: RuntimeContext
    ) -> RequestSnapshot | CompactionRequired: ...

    async def call_provider(
        self,
        context: RuntimeContext,
        request: RequestSnapshot,
        attempt_id: AttemptId,
        emit_ephemeral: EventSink,
    ) -> AssistantTurn: ...

    async def execute_tool(
        self,
        context: RuntimeContext,
        item: ToolBatchItem,
        arguments: Mapping[str, Any],
        attempt_id: AttemptId,
        emit_ephemeral: EventSink,
    ) -> ToolEffectResult[HostDeltaT]: ...

    async def compact(
        self,
        context: RuntimeContext,
        attempt: int,
    ) -> CompactionResult: ...


@dataclass(frozen=True, slots=True)
class SteerCommand:
    command_id: str
    content: Any


@dataclass(frozen=True, slots=True)
class FollowUpCommand:
    command_id: str
    input_id: str
    idempotency_key: str
    content: Any


type RuntimeControl = SteerCommand | FollowUpCommand


class RuntimeControlPort(Protocol):
    async def poll(self, context: RuntimeContext) -> tuple[RuntimeControl, ...]: ...

    async def acknowledge(self, command_ids: tuple[str, ...]) -> bool: ...


@dataclass(frozen=True, slots=True)
class AcceptedOperation:
    """Accepted identity plus the authoritative Session cursor at its boundary."""

    session_id: SessionId
    lane_id: LaneId
    operation_id: OperationId
    created: bool
    cursor: AgentSessionCursor


@dataclass(frozen=True, slots=True)
class OperationView:
    context: RuntimeContext

    @property
    def state(self) -> RunOperationState:
        return self.context.state

    @property
    def terminal(self) -> bool:
        return isinstance(
            self.state,
            (OperationCompleted, OperationCancelled, OperationFailed),
        )


@dataclass(frozen=True, slots=True)
class AgentSessionSnapshotSeed[HostDeltaT]:
    """One repository-bound startup snapshot for exactly one Session Runtime."""

    repository: AgentSessionRepository[HostDeltaT]
    session_id: SessionId
    snapshot: AgentSessionSnapshot

    def __post_init__(self) -> None:
        if self.snapshot.session_id != self.session_id:
            raise ValueError("Agent Session seed snapshot belongs to another Session")


class AgentSessionRuntime[HostDeltaT]:
    """Accept, restore, drive, control, and close durable Agent work."""

    def __init__(
        self,
        *,
        repository: AgentSessionRepository[HostDeltaT],
        effects: AgentRuntimeEffects[HostDeltaT],
        tools: Sequence[AgentTool],
        fencing_epoch: int,
        provider_attempt_limit: int = 2,
        event_sink: EventSink | None = None,
        controls: RuntimeControlPort | None = None,
        initial_snapshot: AgentSessionSnapshotSeed[HostDeltaT] | None = None,
    ) -> None:
        if fencing_epoch < 1:
            raise ValueError("Agent Session Runtime fencing epoch must be positive")
        if provider_attempt_limit < 1:
            raise ValueError("provider attempt limit must be positive")
        self._repository = repository
        self._effects = effects
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("Agent Session Runtime Tool names must be unique")
        self._fencing_epoch = fencing_epoch
        self._provider_attempt_limit = provider_attempt_limit
        self._event_sink = event_sink
        self._controls = controls
        if initial_snapshot is not None and initial_snapshot.repository is not repository:
            raise ValueError("Agent Session seed belongs to another repository")
        self._snapshots: dict[SessionId, AgentSessionSnapshot] = (
            {initial_snapshot.session_id: initial_snapshot.snapshot}
            if initial_snapshot is not None
            else {}
        )
        self._snapshot_locks: dict[SessionId, asyncio.Lock] = (
            {initial_snapshot.session_id: asyncio.Lock()} if initial_snapshot is not None else {}
        )

    async def accept(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
        idempotency_key: str,
        content: Any,
        plan: AgentRunPlan,
    ) -> AcceptedOperation:
        """Atomically append UserMessage plus initial total Operation state."""
        if not idempotency_key:
            raise ValueError("Agent operation idempotency key cannot be empty")
        operation_id = OperationId.deterministic(idempotency_key=idempotency_key)
        acceptance_digest = _digest(
            {
                "session_id": session_id.value,
                "lane_id": lane_id.value,
                "idempotency_key": idempotency_key,
                "content": content,
                "plan_digest": plan.digest,
            }
        )
        snapshot = await self._refresh_snapshot(session_id)
        if _register(snapshot, RegisterRef("session_fault", "session")) is not None:
            raise AgentSessionRuntimeError("a faulted Session cannot accept new work")
        reservation = _register(
            snapshot,
            RegisterRef("host_turn_reservation", lane_id.value),
        )
        if reservation is not None:
            if not isinstance(reservation.value, HostTurnReservation):
                raise TypeError("Host turn reservation register has the wrong value type")
            raise OperationConflictError("Lane already owns an active Host turn")
        existing = _register(snapshot, RegisterRef("operation_meta", operation_id.value))
        if existing is not None:
            value = existing.value
            if not isinstance(value, OperationMetaRegister):
                raise TypeError("Operation Meta register has the wrong value type")
            if value.meta.acceptance_digest != acceptance_digest:
                raise OperationIdempotencyConflict(idempotency_key)
            return AcceptedOperation(
                session_id,
                lane_id,
                operation_id,
                created=False,
                cursor=snapshot.cursor,
            )

        meta = OperationMeta(
            operation_id=operation_id,
            lane_id=lane_id,
            idempotency_key=idempotency_key,
            acceptance_digest=acceptance_digest,
            plan_json=plan.canonical_json(),
            plan_digest=plan.digest,
        )
        state = ReadyForProvider(operation_id)
        message = UserMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            content=content,
        )
        try:
            lane = snapshot.tree.lane(lane_id)
        except KeyError:
            if snapshot.entries or lane_id != LaneId.main():
                raise OperationNotFoundError(f"unknown Lane: {lane_id.value}") from None
            parent = None
            lane_head_sequence = None
            lane_state_sequence = None
            lane_state = LaneState(lane_id, active_operation_id=operation_id.value)
        else:
            lane_state_value = lane.state.value
            if not isinstance(lane_state_value, LaneState):
                raise TypeError("Lane State register has the wrong value type")
            if lane.archived:
                raise ValueError("an archived Lane cannot accept an operation")
            if lane_state_value.active_operation_id is not None:
                raise OperationConflictError("Lane already owns an active operation")
            parent = lane.head_entry_id
            lane_head_sequence = lane.head.sequence
            lane_state_sequence = lane.state.sequence
            lane_state = replace(lane_state_value, active_operation_id=operation_id.value)
        message = replace(message, parent_entry_id=parent)
        pending_record = _register(snapshot, RegisterRef("pending_input", lane_id.value))
        pending_write: SetRegister | DeleteRegister | None = None
        if pending_record is not None:
            if not isinstance(pending_record.value, PendingInput):
                raise TypeError("Pending Input register has the wrong value type")
            queued = pending_record.value.items
            if queued:
                first = queued[0]
                if first.idempotency_key != idempotency_key or first.content_json != canonical_json(
                    content
                ):
                    raise OperationConflictError(
                        "Lane must accept its oldest pending follow-up first"
                    )
                pending_write = (
                    SetRegister(PendingInput(lane_id, queued[1:]))
                    if len(queued) > 1
                    else DeleteRegister(pending_record.ref)
                )
        writes = [
            SetRegister(LaneHead(lane_id, message.entry_id)),
            SetRegister(lane_state),
            SetRegister(OperationMetaRegister(meta)),
            SetRegister(OperationStateRegister(state)),
            *((pending_write,) if pending_write is not None else ()),
        ]
        expectations = [
            RegisterExpectation(writes[0].ref, lane_head_sequence),
            RegisterExpectation(writes[1].ref, lane_state_sequence),
            RegisterExpectation(writes[2].ref, None),
            RegisterExpectation(writes[3].ref, None),
            *(
                (RegisterExpectation(pending_record.ref, pending_record.sequence),)
                if pending_write is not None and pending_record is not None
                else ()
            ),
        ]
        commit = await self._transact(
            session_id,
            SessionTransaction.from_parts(
                entries=[message],
                register_writes=writes,
                expectations=expectations,
            ),
        )
        await self._emit_commit(
            "operation_accepted",
            session_id=session_id,
            lane_id=lane_id,
            operation_id=operation_id,
            commit=commit,
            data={"entry_id": message.entry_id.value},
        )
        return AcceptedOperation(
            session_id,
            lane_id,
            operation_id,
            created=True,
            cursor=AgentSessionCursor(
                commit_sequence=commit.commit_sequence,
                last_entry_sequence=commit.appended_sequences[-1],
            ),
        )

    async def restore(
        self,
        *,
        session_id: SessionId,
        operation_id: OperationId,
    ) -> OperationView:
        snapshot = await self._refresh_snapshot(session_id)
        meta_record = _require_register(snapshot, RegisterRef("operation_meta", operation_id.value))
        state_record = _require_register(
            snapshot, RegisterRef("operation_state", operation_id.value)
        )
        if not isinstance(meta_record.value, OperationMetaRegister):
            raise TypeError("Operation Meta register has the wrong value type")
        if not isinstance(state_record.value, OperationStateRegister):
            raise TypeError("Operation State register has the wrong value type")
        meta = meta_record.value.meta
        state = state_record.value.state
        snapshot = replace(snapshot, selected_lane_id=meta.lane_id)
        if state.operation_id != operation_id or meta.operation_id != operation_id:
            raise ValueError("Operation register identity is corrupt")
        return OperationView(
            RuntimeContext(
                session_id=session_id,
                lane_id=meta.lane_id,
                operation_id=operation_id,
                meta=meta,
                state=state,
                snapshot=snapshot,
            )
        )

    async def drive(
        self,
        *,
        session_id: SessionId,
        operation_id: OperationId,
    ) -> OperationView:
        """Drive the pure interpreter until terminal or external cancellation."""
        while True:
            view = await self.restore(session_id=session_id, operation_id=operation_id)
            if (
                _register(view.context.snapshot, RegisterRef("session_fault", "session"))
                is not None
                and not view.terminal
                and not isinstance(view.state, Cancelling)
            ):
                raise AgentSessionRuntimeError("Session is faulted")
            try:
                if await self._apply_controls(view):
                    continue
                action = next_action(view.state)
                if isinstance(action, NoAction):
                    return view
                await self._execute_action(view, action)
            except (
                asyncio.CancelledError,
                AgentOperationCancelled,
                OperationConflictError,
                ProviderUnavailableError,
                SessionLeaseLostError,
            ):
                raise
            except OperationEffectFailed as exc:
                refreshed = await self.restore(
                    session_id=session_id,
                    operation_id=operation_id,
                )
                await self._fail(refreshed, kind=exc.kind, detail=exc.detail)
            except Exception as exc:
                refreshed = await self.restore(
                    session_id=session_id,
                    operation_id=operation_id,
                )
                await self._fail(
                    refreshed,
                    kind="runtime_fault",
                    detail=f"Runtime invariant failed: {exc}",
                )

    async def _execute_action(self, view: OperationView, action: NextAction) -> None:
        """Execute one already-planned closed action and its atomic transition."""
        if isinstance(action, AssembleProviderRequest):
            await self._assemble_request(view, action.turn_number)
        elif isinstance(action, CallProvider):
            await self._call_provider(view)
        elif isinstance(action, CommitSyntheticToolResult):
            await self._settle_synthetic(view, action.item, action.outcome)
        elif isinstance(action, BeginToolEffect):
            await self._begin_tool(view, action.item)
        elif isinstance(action, RecoverToolEffect):
            if action.replay:
                await self._replay_tool(view, action.item)
            else:
                await self._settle_pending_unknown(view, action.item)
        elif isinstance(action, ContinueAfterToolBatch):
            await self._continue_after_tools(view)
        elif isinstance(action, ConsumeSteer):
            await self._consume_steer(view, action.control_id)
        elif isinstance(action, CompleteOperation):
            await self._complete(view)
        elif isinstance(action, RunCompaction):
            await self._compact(view, action.attempt)
        elif isinstance(action, CloseCancellationPosition):
            await self._close_cancel_position(
                view,
                action.item,
                outcome_unknown=action.outcome_unknown,
            )
        elif isinstance(action, FinishCancellation):
            await self._finish_cancel(view)
        else:
            raise AssertionError(f"unhandled NextAction: {type(action).__name__}")

    async def _apply_controls(self, view: OperationView) -> bool:
        if self._controls is None or not isinstance(
            view.state, (ReadyForProvider, CompletionReady)
        ):
            return False
        commands = await self._controls.poll(view.context)
        if not commands:
            return False
        acknowledged: list[str] = []
        for command in commands:
            if isinstance(command, SteerCommand):
                await self.steer(
                    session_id=view.context.session_id,
                    operation_id=view.context.operation_id,
                    control_id=command.command_id,
                    content=command.content,
                )
            else:
                await self.follow_up(
                    session_id=view.context.session_id,
                    lane_id=view.context.lane_id,
                    input_id=command.input_id,
                    idempotency_key=command.idempotency_key,
                    content=command.content,
                )
            acknowledged.append(command.command_id)
            view = await self.restore(
                session_id=view.context.session_id,
                operation_id=view.context.operation_id,
            )
        if not await self._controls.acknowledge(tuple(acknowledged)):
            raise SessionLeaseLostError(view.context.session_id.value)
        return True

    async def steer(
        self,
        *,
        session_id: SessionId,
        operation_id: OperationId,
        control_id: str,
        content: Any,
    ) -> None:
        """Durably enqueue one Steer without changing a persisted request."""
        view = await self.restore(session_id=session_id, operation_id=operation_id)
        state = view.state
        if _register(view.context.snapshot, RegisterRef("session_fault", "session")) is not None:
            raise AgentSessionRuntimeError("a faulted Session cannot be steered")
        if isinstance(state, (OperationCompleted, OperationCancelled, OperationFailed)):
            raise OperationConflictError("a terminal operation cannot be steered")
        if isinstance(state, Cancelling):
            raise OperationConflictError("a cancelling operation cannot be steered")
        steers = getattr(state, "steers", ())
        if any(item.control_id == control_id for item in steers):
            return
        plan = AgentRunPlan.from_payload(_json_object(view.context.meta.plan_json))
        if len(steers) >= plan.max_pending_steers:
            raise OperationConflictError("operation Steer inbox is full")
        updated = replace(state, steers=(*steers, AcceptedSteer.from_content(control_id, content)))
        await self._replace_state(view, updated, event="steer_accepted")

    async def follow_up(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
        input_id: str,
        idempotency_key: str,
        content: Any,
    ) -> None:
        """Queue bounded unaccepted input; it receives a fresh Plan when dequeued."""
        snapshot = await self._refresh_snapshot(session_id)
        if _register(snapshot, RegisterRef("session_fault", "session")) is not None:
            raise AgentSessionRuntimeError("a faulted Session cannot accept follow-up input")
        lane = snapshot.tree.lane(lane_id)
        state = lane.state.value
        if not isinstance(state, LaneState):
            raise TypeError("Lane State register has the wrong value type")
        if state.active_operation_id is None:
            raise OperationConflictError("idle Lane input must be accepted as a new operation")
        ref = RegisterRef("pending_input", lane_id.value)
        record = _register(snapshot, ref)
        current = ()
        if record is not None:
            if not isinstance(record.value, PendingInput):
                raise TypeError("Pending Input register has the wrong value type")
            current = record.value.items
        if any(item.input_id == input_id for item in current):
            return
        meta_record = _require_register(
            snapshot,
            RegisterRef("operation_meta", state.active_operation_id),
        )
        if not isinstance(meta_record.value, OperationMetaRegister):
            raise TypeError("Operation Meta register has the wrong value type")
        plan = AgentRunPlan.from_payload(_json_object(meta_record.value.meta.plan_json))
        if len(current) >= plan.max_pending_follow_ups:
            raise OperationConflictError("Lane follow-up FIFO is full")
        value = PendingInput(
            lane_id,
            (
                *current,
                FollowUpInput.from_content(
                    input_id=input_id,
                    idempotency_key=idempotency_key,
                    content=content,
                ),
            ),
        )
        commit = await self._transact(
            session_id,
            SessionTransaction.from_parts(
                register_writes=[SetRegister(value)],
                expectations=[RegisterExpectation(ref, record.sequence if record else None)],
            ),
        )
        await self._emit_commit(
            "follow_up_queued",
            session_id=session_id,
            lane_id=lane_id,
            operation_id=OperationId(state.active_operation_id),
            commit=commit,
            data={"input_id": input_id},
        )

    async def cancel(
        self,
        *,
        session_id: SessionId,
        operation_id: OperationId,
    ) -> None:
        """Durably request cancellation and stop all new effect dispatch."""
        view = await self.restore(session_id=session_id, operation_id=operation_id)
        state = view.state
        if isinstance(state, (OperationCompleted, OperationCancelled, OperationFailed, Cancelling)):
            return
        batch: ToolBatchPlan | None = None
        next_index = 0
        uncertain: int | None = None
        uncertain_attempt: AttemptId | None = None
        if isinstance(state, ToolBatchReady):
            batch = state.batch
            next_index = state.next_source_index
        elif isinstance(state, ToolEffectPending):
            batch = state.batch
            next_index = state.source_index
            uncertain = state.source_index
            uncertain_attempt = state.attempt_id
        cancelling = Cancelling(
            operation_id,
            turn_count=_turn_count(state),
            batch=batch,
            next_source_index=next_index,
            uncertain_source_index=uncertain,
            uncertain_attempt_id=uncertain_attempt,
        )
        await self._replace_state(view, cancelling, event="cancel_requested")

    async def close(
        self,
        *,
        session_id: SessionId,
        operation_id: OperationId,
    ) -> OperationView:
        """Finish a requested cancellation, or return an existing terminal state."""
        view = await self.restore(session_id=session_id, operation_id=operation_id)
        if not view.terminal and not isinstance(view.state, Cancelling):
            await self.cancel(session_id=session_id, operation_id=operation_id)
        return await self.drive(session_id=session_id, operation_id=operation_id)

    async def _assemble_request(self, view: OperationView, turn_number: int) -> None:
        assembled = await self._effects.assemble_request(view.context)
        if isinstance(assembled, CompactionRequired):
            plan = AgentRunPlan.from_payload(_json_object(view.context.meta.plan_json))
            state = CompactionPending(
                view.context.operation_id,
                turn_count=_turn_count(view.state),
                attempt=1,
                max_attempts=plan.compaction_attempt_limit,
                steers=getattr(view.state, "steers", ()),
            )
            await self._replace_state(view, state, event="compaction_required")
            return
        request = assembled
        if (
            request.operation_id != view.context.operation_id
            or request.turn_number != turn_number
            or request.plan_digest != view.context.meta.plan_digest
        ):
            raise ValueError("Context Assembler returned a mismatched Request Snapshot")
        state = ProviderRequestPending(
            view.context.operation_id,
            turn_number=turn_number,
            provider_attempts=0,
            steers=getattr(view.state, "steers", ()),
        )
        await self._write_state_and_request(view, state, request)

    async def _call_provider(self, view: OperationView) -> None:
        state = view.state
        if not isinstance(state, ProviderRequestPending):
            raise TypeError("provider action requires ProviderRequestPending")
        request_record = _require_register(
            view.context.snapshot,
            RegisterRef("request_snapshot", view.context.operation_id.value),
        )
        if not isinstance(request_record.value, RequestSnapshot):
            raise TypeError("Request Snapshot register has the wrong value type")
        attempt_id = AttemptId.new()
        started = replace(
            state,
            provider_attempts=state.provider_attempts + 1,
            attempt_ids=(*state.attempt_ids, attempt_id),
        )
        await self._replace_state(view, started, event="provider_attempt_started")
        refreshed = await self.restore(
            session_id=view.context.session_id,
            operation_id=view.context.operation_id,
        )
        try:
            assistant = await self._effects.call_provider(
                refreshed.context,
                request_record.value,
                attempt_id,
                self._ephemeral_sink(refreshed),
            )
        except asyncio.CancelledError, AgentOperationCancelled:
            raise
        except ProviderContextOverflow:
            plan = AgentRunPlan.from_payload(_json_object(view.context.meta.plan_json))
            compacting = CompactionPending(
                view.context.operation_id,
                turn_count=state.turn_number - 1,
                attempt=1,
                max_attempts=plan.compaction_attempt_limit,
                steers=started.steers,
            )
            await self._replace_state(
                refreshed,
                compacting,
                event="provider_context_overflow",
            )
            return
        except ProviderAttemptFailed as exc:
            if exc.retryable and started.provider_attempts < self._provider_attempt_limit:
                await self._emit_observe_only(
                    AgentSessionEvent(
                        kind="provider_attempt_unknown",
                        session_id=view.context.session_id,
                        lane_id=view.context.lane_id,
                        operation_id=view.context.operation_id,
                        commit_sequence=None,
                        data={"attempt_id": attempt_id.value},
                        ephemeral=True,
                    )
                )
                return
            if exc.retryable:
                # Keep ProviderAttemptStarted durable and let the owning Run
                # release compute capacity. Reclaim recovery resolves the
                # interrupted attempt before resuming this same operation.
                raise ProviderUnavailableError from exc
            await self._fail(
                refreshed,
                kind="provider_unavailable",
                detail=exc.detail,
            )
            return
        except Exception:
            await self._fail(
                refreshed,
                kind="provider_unavailable",
                detail="Model provider request failed",
            )
            return
        await self._commit_assistant(refreshed, assistant)

    async def _commit_assistant(self, view: OperationView, assistant: AssistantTurn) -> None:
        state = view.state
        if not isinstance(state, ProviderRequestPending):
            raise TypeError("Assistant settlement requires ProviderRequestPending")
        duplicate_ids = _duplicates(call.id for call in assistant.tool_calls)
        if duplicate_ids:
            await self._fail(
                view,
                kind="provider_invalid",
                detail=f"provider returned duplicate Tool call ids: {duplicate_ids}",
            )
            return
        entry = AssistantMessageEntry(
            entry_id=EntryId.new(),
            session_id=view.context.session_id,
            timestamp=_now(),
            content=assistant.text,
            reasoning=assistant.reasoning,
            stop_reason=assistant.stop_reason,
            tool_calls=assistant.tool_calls,
            usage=assistant.usage_details,
            cost=assistant.cost_details,
            provider_state=assistant.provider_state,
            parent_entry_id=_lane_head(view.context.snapshot, view.context.lane_id).entry_id,
        )
        batch, arguments = self._tool_batch_for_meta(view.context.meta, entry, assistant)
        if batch.items:
            next_state: RunOperationState = ToolBatchReady(
                view.context.operation_id,
                turn_number=state.turn_number,
                batch=batch,
                steers=state.steers,
            )
        else:
            next_state = CompletionReady(
                view.context.operation_id,
                turn_count=state.turn_number,
                assistant_entry_id=entry.entry_id,
                steers=state.steers,
            )
        snapshot = view.context.snapshot
        head_record = _require_register(snapshot, LaneHead(view.context.lane_id, None).ref)
        state_record = _state_record(snapshot, view.context.operation_id)
        request_record = _require_register(
            snapshot, RegisterRef("request_snapshot", view.context.operation_id.value)
        )
        writes = [
            SetRegister(LaneHead(view.context.lane_id, entry.entry_id)),
            SetRegister(OperationStateRegister(next_state)),
            DeleteRegister(request_record.ref),
            *(SetRegister(value) for value in arguments),
        ]
        expectations = [
            RegisterExpectation(head_record.ref, head_record.sequence),
            RegisterExpectation(state_record.ref, state_record.sequence),
            RegisterExpectation(request_record.ref, request_record.sequence),
            *(RegisterExpectation(value.ref, None) for value in arguments),
        ]
        commit = await self._transact(
            view.context.session_id,
            SessionTransaction.from_parts(
                entries=[entry],
                register_writes=writes,
                expectations=expectations,
            ),
        )
        await self._emit_commit(
            "assistant_committed",
            session_id=view.context.session_id,
            lane_id=view.context.lane_id,
            operation_id=view.context.operation_id,
            commit=commit,
            data={
                "entry_id": entry.entry_id.value,
                "turn_number": state.turn_number,
                "tool_positions": len(batch.items),
            },
        )

    def _tool_batch_for_meta(
        self,
        meta: OperationMeta,
        assistant_entry: AssistantMessageEntry,
        assistant: AssistantTurn,
    ) -> tuple[ToolBatchPlan, tuple[ToolArguments, ...]]:
        import json

        plan = AgentRunPlan.from_payload(json.loads(meta.plan_json))
        pinned = {tool.name: tool for tool in plan.tools}
        items: list[ToolBatchItem] = []
        arguments: list[ToolArguments] = []
        for index, call in enumerate(assistant.tool_calls):
            disposition: ToolCallDisposition = "executable"
            synthetic = ""
            intent_id: IntentId | None = None
            replay_policy: ReplayPolicy = "never"
            contract_version = 0
            schema = ""
            effective = ""
            canonical_input = ""
            pinned_tool = pinned.get(call.name)
            resolved = self._tools.get(call.name)
            if assistant.stop_reason == "length":
                disposition = "truncated_call"
                synthetic = (
                    f'Tool "{call.name}" was not executed because the provider response '
                    "was truncated."
                )
            elif pinned_tool is None:
                disposition = "unknown_tool" if resolved is None else "plan_denied"
                synthetic = f'Tool "{call.name}" is not allowed by the accepted Plan.'
            elif resolved is None:
                disposition = "contract_changed"
                synthetic = f'Tool "{call.name}" contract is unavailable.'
            elif (
                resolved.replay_policy != pinned_tool.replay_policy
                or resolved.contract_version != pinned_tool.contract_version
                or resolved.input_schema_digest != pinned_tool.input_schema_digest
            ):
                disposition = "contract_changed"
                synthetic = f'Tool "{call.name}" contract changed; call was not executed.'
            elif call.argument_error:
                disposition = "invalid_arguments"
                synthetic = f'Arguments for Tool "{call.name}" are invalid: {call.argument_error}'
            else:
                try:
                    validated = resolved.input_model.model_validate(call.arguments)
                except ValidationError as exc:
                    disposition = "invalid_arguments"
                    synthetic = f'Arguments for Tool "{call.name}" are invalid: {exc}'
                else:
                    canonical_input = canonical_json(validated.model_dump(mode="json"))
                    intent_id = IntentId.new()
                    replay_policy = pinned_tool.replay_policy
                    contract_version = pinned_tool.contract_version
                    schema = pinned_tool.input_schema_digest
                    effective = sha256(canonical_input.encode("utf-8")).hexdigest()
                    arguments.append(ToolArguments(intent_id, canonical_input))
            items.append(
                ToolBatchItem(
                    source_index=index,
                    call_id=call.id,
                    tool_name=call.name,
                    disposition=disposition,
                    result_entry_id=EntryId.new(),
                    intent_id=intent_id,
                    replay_policy=replay_policy,
                    contract_version=contract_version,
                    input_schema_digest=schema,
                    effective_input_digest=effective,
                    synthetic_message=synthetic,
                )
            )
        return ToolBatchPlan(assistant_entry.entry_id, tuple(items)), tuple(arguments)

    async def _settle_synthetic(
        self,
        view: OperationView,
        item: ToolBatchItem,
        outcome: ToolResultOutcome,
    ) -> None:
        result = ToolResultEntry.text(
            tool_name=item.tool_name,
            call_id=item.call_id,
            outcome=outcome,
            text=item.synthetic_message,
        )
        await self._append_tool_result(view, item, result, attempt_id=None, host_delta=None)

    async def _begin_tool(self, view: OperationView, item: ToolBatchItem) -> None:
        state = view.state
        if not isinstance(state, ToolBatchReady) or item.intent_id is None:
            raise TypeError("Tool start requires an executable ToolBatchReady item")
        pending = ToolEffectPending(
            view.context.operation_id,
            turn_number=state.turn_number,
            batch=state.batch,
            source_index=item.source_index,
            attempt_id=AttemptId.new(),
            steers=state.steers,
        )
        await self._replace_state(view, pending, event="tool_intent_committed")
        refreshed = await self.restore(
            session_id=view.context.session_id,
            operation_id=view.context.operation_id,
        )
        await self._execute_pending_tool(refreshed, item, recovery=False)

    async def _replay_tool(self, view: OperationView, item: ToolBatchItem) -> None:
        state = view.state
        if not isinstance(state, ToolEffectPending):
            raise TypeError("Tool recovery requires ToolEffectPending")
        rotated = replace(state, attempt_id=AttemptId.new())
        await self._replace_state(view, rotated, event="tool_replay_started")
        refreshed = await self.restore(
            session_id=view.context.session_id,
            operation_id=view.context.operation_id,
        )
        await self._execute_pending_tool(refreshed, item, recovery=True)

    async def _execute_pending_tool(
        self,
        view: OperationView,
        item: ToolBatchItem,
        *,
        recovery: bool,
    ) -> None:
        state = view.state
        if not isinstance(state, ToolEffectPending) or item.intent_id is None:
            raise TypeError("Tool execution requires ToolEffectPending")
        resolved = self._tools.get(item.tool_name)
        if (
            resolved is None
            or resolved.replay_policy != item.replay_policy
            or resolved.contract_version != item.contract_version
            or resolved.input_schema_digest != item.input_schema_digest
        ):
            await self._append_tool_result(
                view,
                item,
                ToolResultEntry.text(
                    tool_name=item.tool_name,
                    call_id=item.call_id,
                    outcome="tool_contract_changed",
                    text=f'Tool "{item.tool_name}" contract changed; call was not executed.',
                ),
                attempt_id=state.attempt_id,
                host_delta=None,
                advances_durable_progress=not recovery,
            )
            return
        args_record = _require_register(
            view.context.snapshot,
            RegisterRef("tool_arguments", item.intent_id.value),
        )
        if not isinstance(args_record.value, ToolArguments):
            raise TypeError("Tool Arguments register has the wrong value type")
        try:
            effect = await self._effects.execute_tool(
                view.context,
                item,
                args_record.value.arguments,
                state.attempt_id,
                self._ephemeral_sink(view),
            )
        except asyncio.CancelledError, AgentOperationCancelled:
            raise
        except Exception as exc:
            logger.warning("Agent Runtime Tool effect failed", exc_info=True)
            effect = ToolEffectResult(
                result=ToolResultEntry.text(
                    tool_name=item.tool_name,
                    call_id=item.call_id,
                    outcome="failed",
                    text=f'Tool "{item.tool_name}" failed: {exc}',
                ),
                host_delta=None,
            )
        await self._append_tool_result(
            view,
            item,
            effect.result,
            attempt_id=state.attempt_id,
            host_delta=effect.host_delta,
        )

    async def _settle_pending_unknown(self, view: OperationView, item: ToolBatchItem) -> None:
        result = ToolResultEntry.text(
            tool_name=item.tool_name,
            call_id=item.call_id,
            outcome="outcome_unknown",
            text=(
                f'Tool "{item.tool_name}" was not replayed because its prior effect may '
                "have happened; its outcome is unknown."
            ),
        )
        state = view.state
        attempt = state.attempt_id if isinstance(state, ToolEffectPending) else None
        await self._append_tool_result(
            view,
            item,
            result,
            attempt_id=attempt,
            host_delta=None,
            advances_durable_progress=False,
        )

    async def _append_tool_result(
        self,
        view: OperationView,
        item: ToolBatchItem,
        result: ToolResultEntry,
        *,
        attempt_id: AttemptId | None,
        host_delta: HostDeltaT | None,
        advances_durable_progress: bool = True,
    ) -> None:
        state = view.state
        if isinstance(state, ToolBatchReady):
            batch = state.batch
            turn_number = state.turn_number
            steers = state.steers
        elif isinstance(state, ToolEffectPending):
            batch = state.batch
            turn_number = state.turn_number
            steers = state.steers
        elif isinstance(state, Cancelling) and state.batch is not None:
            batch = state.batch
            turn_number = state.turn_count
            steers = ()
        else:
            raise TypeError("ToolResult settlement requires a Tool Batch state")
        durable_result = replace(result, details=None)
        entry = ToolResultMessageEntry(
            entry_id=item.result_entry_id,
            session_id=view.context.session_id,
            timestamp=_now(),
            parent_entry_id=_lane_head(view.context.snapshot, view.context.lane_id).entry_id,
            result=durable_result,
            intent_id=item.intent_id,
            source_index=item.source_index,
            contract_version=item.contract_version,
            input_schema_digest=item.input_schema_digest,
            replay_policy=item.replay_policy,
            attempt_id=attempt_id,
            effective_input_digest=item.effective_input_digest,
        )
        if isinstance(state, Cancelling):
            next_state: RunOperationState = replace(
                state,
                next_source_index=item.source_index + 1,
                uncertain_source_index=None,
                uncertain_attempt_id=None,
            )
        else:
            next_state = ToolBatchReady(
                view.context.operation_id,
                turn_number=turn_number,
                batch=batch,
                next_source_index=item.source_index + 1,
                steers=steers,
            )
        snapshot = view.context.snapshot
        head_record = _require_register(snapshot, LaneHead(view.context.lane_id, None).ref)
        state_record = _state_record(snapshot, view.context.operation_id)
        writes: list[RegisterWrite] = [
            SetRegister(LaneHead(view.context.lane_id, entry.entry_id)),
            SetRegister(OperationStateRegister(next_state)),
        ]
        expectations = [
            RegisterExpectation(head_record.ref, head_record.sequence),
            RegisterExpectation(state_record.ref, state_record.sequence),
        ]
        if item.intent_id is not None:
            args_record = _require_register(
                snapshot, RegisterRef("tool_arguments", item.intent_id.value)
            )
            writes.append(DeleteRegister(args_record.ref))
            expectations.append(RegisterExpectation(args_record.ref, args_record.sequence))
        delta = (
            HostDeltaSettlement(item.intent_id, host_delta)
            if item.intent_id is not None and host_delta is not None
            else None
        )
        commit = await self._transact(
            view.context.session_id,
            SessionTransaction.from_parts(
                entries=[entry],
                register_writes=writes,
                expectations=expectations,
                host_delta=delta,
                advances_durable_progress=advances_durable_progress,
            ),
        )
        await self._emit_commit(
            "tool_result_committed",
            session_id=view.context.session_id,
            lane_id=view.context.lane_id,
            operation_id=view.context.operation_id,
            commit=commit,
            data={
                "entry_id": entry.entry_id.value,
                "source_index": item.source_index,
                "outcome": durable_result.outcome,
            },
        )

    async def _continue_after_tools(self, view: OperationView) -> None:
        state = view.state
        if not isinstance(state, ToolBatchReady):
            raise TypeError("Tool Batch completion requires ToolBatchReady")
        ready = ReadyForProvider(
            view.context.operation_id,
            turn_count=state.turn_number,
            steers=state.steers,
        )
        await self._replace_state(view, ready, event="tool_batch_completed")

    async def _consume_steer(self, view: OperationView, control_id: str) -> None:
        state = view.state
        steers = getattr(state, "steers", ())
        if not steers or steers[0].control_id != control_id:
            raise OperationConflictError("Steer inbox changed before consumption")
        control = steers[0]
        entry = ControlMessageEntry(
            entry_id=EntryId.new(),
            session_id=view.context.session_id,
            timestamp=_now(),
            parent_entry_id=_lane_head(view.context.snapshot, view.context.lane_id).entry_id,
            control_id=control.control_id,
            content=control.content,
        )
        ready = ReadyForProvider(
            view.context.operation_id,
            turn_count=_turn_count(state),
            steers=steers[1:],
        )
        await self._append_entry_and_state(view, entry, ready, "steer_consumed")

    async def _complete(self, view: OperationView) -> None:
        state = view.state
        if not isinstance(state, CompletionReady):
            raise TypeError("completion requires CompletionReady")
        terminal = OperationCompleted(
            view.context.operation_id,
            turn_count=state.turn_count,
            assistant_entry_id=state.assistant_entry_id,
        )
        await self._terminalize(view, terminal, event="operation_completed")

    async def _compact(self, view: OperationView, attempt: int) -> None:
        state = view.state
        if not isinstance(state, CompactionPending):
            raise TypeError("Compaction action requires CompactionPending")
        try:
            result = await self._effects.compact(view.context, attempt)
        except Exception as exc:
            if attempt < state.max_attempts:
                await self._replace_state(
                    view,
                    replace(state, attempt=attempt + 1),
                    event="compaction_retry",
                )
            else:
                await self._fail(
                    view,
                    kind="compaction_failed",
                    detail=str(exc),
                )
            return
        if result.entry.session_id != view.context.session_id:
            raise ValueError("Compaction result belongs to another Session")
        entry = replace(
            result.entry,
            parent_entry_id=_lane_head(view.context.snapshot, view.context.lane_id).entry_id,
        )
        ready = ReadyForProvider(
            view.context.operation_id,
            turn_count=state.turn_count,
            steers=state.steers,
        )
        await self._append_entry_and_state(
            view,
            entry,
            ready,
            "compaction_committed",
            projection=result.projection,
        )

    async def _close_cancel_position(
        self,
        view: OperationView,
        item: ToolBatchItem,
        *,
        outcome_unknown: bool,
    ) -> None:
        if item.disposition != "executable":
            result = ToolResultEntry.text(
                tool_name=item.tool_name,
                call_id=item.call_id,
                outcome=TOOL_DISPOSITION_OUTCOME[item.disposition],
                text=item.synthetic_message,
            )
        else:
            outcome = "outcome_unknown" if outcome_unknown else "interrupted"
            detail = (
                "its prior effect may have happened and its outcome is unknown"
                if outcome_unknown
                else "it was cancelled before dispatch"
            )
            result = ToolResultEntry.text(
                tool_name=item.tool_name,
                call_id=item.call_id,
                outcome=outcome,
                text=f'Tool "{item.tool_name}" did not settle: {detail}.',
            )
        state = view.state
        attempt = (
            state.uncertain_attempt_id
            if isinstance(state, Cancelling) and outcome_unknown
            else None
        )
        await self._append_tool_result(
            view,
            item,
            result,
            attempt_id=attempt,
            host_delta=None,
        )

    async def _finish_cancel(self, view: OperationView) -> None:
        terminal = OperationCancelled(
            view.context.operation_id,
            turn_count=_turn_count(view.state),
        )
        await self._terminalize(view, terminal, event="operation_cancelled")

    async def _fail(
        self,
        view: OperationView,
        *,
        kind: TerminalFailureKind,
        detail: str,
    ) -> None:
        terminal = OperationFailed(
            view.context.operation_id,
            turn_count=_turn_count(view.state),
            kind=kind,
            detail=detail,
            provider_attempt_ids=(
                view.state.attempt_ids if isinstance(view.state, ProviderRequestPending) else ()
            ),
        )
        await self._terminalize(
            view,
            terminal,
            event="operation_failed",
            session_fault=(detail if kind == "runtime_fault" else None),
        )

    async def _terminalize(
        self,
        view: OperationView,
        terminal: RunOperationState,
        *,
        event: AgentSessionEventKind,
        session_fault: str | None = None,
    ) -> None:
        snapshot = view.context.snapshot
        lane_record = _require_register(snapshot, LaneState(view.context.lane_id).ref)
        lane = lane_record.value
        if not isinstance(lane, LaneState):
            raise TypeError("Lane State register has the wrong value type")
        if lane.active_operation_id != view.context.operation_id.value:
            raise OperationConflictError("Operation no longer owns its Lane")
        state_record = _state_record(snapshot, view.context.operation_id)
        writes: list[RegisterWrite] = [
            SetRegister(OperationStateRegister(terminal)),
            SetRegister(
                replace(
                    lane,
                    active_operation_id=None,
                    last_operation_id=view.context.operation_id.value,
                )
            ),
        ]
        expectations = [
            RegisterExpectation(state_record.ref, state_record.sequence),
            RegisterExpectation(lane_record.ref, lane_record.sequence),
        ]
        if session_fault is not None:
            fault = SessionFault(session_fault)
            writes.append(SetRegister(fault))
            expectations.append(RegisterExpectation(fault.ref, None))
        request = _register(
            snapshot, RegisterRef("request_snapshot", view.context.operation_id.value)
        )
        if request is not None:
            writes.append(DeleteRegister(request.ref))
            expectations.append(RegisterExpectation(request.ref, request.sequence))
        commit = await self._transact(
            view.context.session_id,
            SessionTransaction.from_parts(
                register_writes=writes,
                expectations=expectations,
            ),
        )
        await self._emit_commit(
            event,
            session_id=view.context.session_id,
            lane_id=view.context.lane_id,
            operation_id=view.context.operation_id,
            commit=commit,
            data={"state_type": terminal.state_type},
        )

    async def _replace_state(
        self,
        view: OperationView,
        state: RunOperationState,
        *,
        event: AgentSessionEventKind,
    ) -> TransactionCommit:
        record = _state_record(view.context.snapshot, view.context.operation_id)
        commit = await self._transact(
            view.context.session_id,
            SessionTransaction.from_parts(
                register_writes=[SetRegister(OperationStateRegister(state))],
                expectations=[RegisterExpectation(record.ref, record.sequence)],
            ),
        )
        await self._emit_commit(
            event,
            session_id=view.context.session_id,
            lane_id=view.context.lane_id,
            operation_id=view.context.operation_id,
            commit=commit,
            data={"state_type": state.state_type},
        )
        return commit

    async def _write_state_and_request(
        self,
        view: OperationView,
        state: RunOperationState,
        request: RequestSnapshot,
    ) -> None:
        state_record = _state_record(view.context.snapshot, view.context.operation_id)
        existing_request = _register(view.context.snapshot, request.ref)
        commit = await self._transact(
            view.context.session_id,
            SessionTransaction.from_parts(
                register_writes=[
                    SetRegister(OperationStateRegister(state)),
                    SetRegister(request),
                ],
                expectations=[
                    RegisterExpectation(state_record.ref, state_record.sequence),
                    RegisterExpectation(
                        request.ref,
                        existing_request.sequence if existing_request else None,
                    ),
                ],
            ),
        )
        await self._emit_commit(
            "provider_request_committed",
            session_id=view.context.session_id,
            lane_id=view.context.lane_id,
            operation_id=view.context.operation_id,
            commit=commit,
            data={"turn_number": request.turn_number},
        )

    async def _append_entry_and_state(
        self,
        view: OperationView,
        entry: SessionEntry,
        state: RunOperationState,
        event: AgentSessionEventKind,
        *,
        projection: ContextProjection | None = None,
    ) -> None:
        snapshot = view.context.snapshot
        head = _require_register(snapshot, LaneHead(view.context.lane_id, None).ref)
        state_record = _state_record(snapshot, view.context.operation_id)
        writes: list[RegisterWrite] = [
            SetRegister(LaneHead(view.context.lane_id, entry.entry_id)),
            SetRegister(OperationStateRegister(state)),
        ]
        expectations = [
            RegisterExpectation(head.ref, head.sequence),
            RegisterExpectation(state_record.ref, state_record.sequence),
        ]
        if projection is not None:
            value = ContextProjectionRegister(view.context.lane_id, projection)
            current = _register(snapshot, value.ref)
            writes.append(SetRegister(value))
            expectations.append(
                RegisterExpectation(value.ref, current.sequence if current else None)
            )
        commit = await self._transact(
            view.context.session_id,
            SessionTransaction.from_parts(
                entries=[entry],
                register_writes=writes,
                expectations=expectations,
            ),
        )
        await self._emit_commit(
            event,
            session_id=view.context.session_id,
            lane_id=view.context.lane_id,
            operation_id=view.context.operation_id,
            commit=commit,
            data={"entry_id": entry.entry_id.value},
        )

    def _snapshot_lock(self, session_id: SessionId) -> asyncio.Lock:
        lock = self._snapshot_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._snapshot_locks[session_id] = lock
        return lock

    async def _refresh_snapshot(self, session_id: SessionId) -> AgentSessionSnapshot:
        """Refresh and publish one base snapshot without allowing cache regression."""
        async with self._snapshot_lock(session_id):
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
            return snapshot

    async def _transact(
        self,
        session_id: SessionId,
        transaction: SessionTransaction[HostDeltaT],
    ) -> TransactionCommit:
        # The same per-Session lock orders repository outcomes and cache publication.
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

    async def _emit_commit(
        self,
        kind: AgentSessionEventKind,
        *,
        session_id: SessionId,
        lane_id: LaneId,
        operation_id: OperationId,
        commit: TransactionCommit,
        data: Mapping[str, Any],
    ) -> None:
        await self._emit_observe_only(
            AgentSessionEvent(
                kind=kind,
                session_id=session_id,
                lane_id=lane_id,
                operation_id=operation_id,
                commit_sequence=commit.commit_sequence,
                data=data,
            )
        )

    def _ephemeral_sink(self, view: OperationView) -> EventSink:
        async def emit(event: AgentSessionEvent) -> None:
            if not event.ephemeral or event.commit_sequence is not None:
                raise ValueError("effect streams may publish only ephemeral events")
            await self._emit_observe_only(event)

        return emit

    async def _emit_observe_only(self, event: AgentSessionEvent) -> None:
        if self._event_sink is None:
            return
        try:
            await self._event_sink(event)
        except Exception:
            logger.warning("Agent Session event sink failed", exc_info=True)


def _now() -> datetime:
    return datetime.now(UTC)


def _digest(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return sorted(repeated)


def _register(snapshot: AgentSessionSnapshot, ref: RegisterRef) -> RegisterRecord | None:
    return next((record for record in snapshot.registers if record.ref == ref), None)


def _require_register(snapshot: AgentSessionSnapshot, ref: RegisterRef) -> RegisterRecord:
    record = _register(snapshot, ref)
    if record is None:
        raise OperationNotFoundError(f"missing register {ref.kind}:{ref.key}")
    return record


def _state_record(snapshot: AgentSessionSnapshot, operation_id: OperationId) -> RegisterRecord:
    return _require_register(snapshot, RegisterRef("operation_state", operation_id.value))


def _lane_head(snapshot: AgentSessionSnapshot, lane_id: LaneId) -> LaneHead:
    value = _require_register(snapshot, LaneHead(lane_id, None).ref).value
    if not isinstance(value, LaneHead):
        raise TypeError("Lane Head register has the wrong value type")
    return value


def _turn_count(state: RunOperationState) -> int:
    if isinstance(state, ProviderRequestPending):
        return state.turn_number - 1
    if isinstance(state, (ToolBatchReady, ToolEffectPending)):
        return state.turn_number
    return int(getattr(state, "turn_count", 0))


def _json_object(value: str) -> dict[str, Any]:
    import json

    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("canonical Plan is not an object")
    return payload


__all__ = [
    "AcceptedOperation",
    "AgentOperationCancelled",
    "AgentRuntimeEffects",
    "AgentSessionEvent",
    "AgentSessionEventKind",
    "AgentSessionRuntime",
    "AgentSessionRuntimeError",
    "AgentSessionSnapshotSeed",
    "CompactionRequired",
    "CompactionResult",
    "EventSink",
    "FollowUpCommand",
    "OperationConflictError",
    "OperationEffectFailed",
    "OperationIdempotencyConflict",
    "OperationNotFoundError",
    "OperationView",
    "ProviderAttemptFailed",
    "ProviderContextOverflow",
    "RuntimeContext",
    "RuntimeControl",
    "RuntimeControlPort",
    "SessionLeaseLostError",
    "SteerCommand",
    "ToolEffectResult",
]
