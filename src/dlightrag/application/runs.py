# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner-scoped Application service and views for the common durable Run lifecycle."""

import datetime
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

from dlightrag.engine.runtime import (
    CancellationOutcome as RuntimeCancellationOutcome,
)
from dlightrag.engine.runtime import (
    RunCreation as RuntimeRunCreation,
)
from dlightrag.engine.runtime import (
    RunEvent as RuntimeRunEvent,
)
from dlightrag.engine.runtime import (
    RunRecord as RuntimeRunRecord,
)

RunStatus: TypeAlias = Literal[  # noqa: UP040 - preserve the inline OpenAPI enum
    "queued", "running", "succeeded", "failed", "cancelled"
]
RunKind: TypeAlias = Literal[  # noqa: UP040 - preserve caller contract introspection
    "retrieval", "answer", "corpus_mutation"
]
RunLane: TypeAlias = Literal["query", "corpus_mutation"]  # noqa: UP040
RunPhase: TypeAlias = str  # noqa: UP040 - executor-owned labels remain open
RunCancellationResult: TypeAlias = Literal[  # noqa: UP040
    "unknown", "cancelled", "pending", "already_terminal", "rejected"
]

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
_MAX_REPAIR_TEXT_CHARS = 512


def _repair_text(record: RuntimeRunRecord, key: str) -> str | None:
    if record.phase != "waiting_for_repair" or not isinstance(record.checkpoint, Mapping):
        return None
    value = record.checkpoint.get(key)
    return str(value)[:_MAX_REPAIR_TEXT_CHARS] if isinstance(value, str) and value else None


class IdempotencyKeyConflict(RuntimeError):
    """A caller reused a submission key with different normalized input."""


class RunCapacityExceededError(RuntimeError):
    """The deployment-wide nonterminal admission fuse is full."""


class RunRuntimeUnavailableError(RuntimeError):
    """No local common Run scheduler can safely accept new work."""


class RunCancelledError(RuntimeError):
    """The run this caller waited on was cancelled by its owner."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run {run_id} was cancelled")
        self.run_id = run_id


class RunFailedError(RuntimeError):
    """The run this caller waited on failed with one public error."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.error_kind = kind
        self.public_message = message


@dataclass(frozen=True, slots=True)
class RunView:
    """Caller-facing common lifecycle state without lease or persistence internals."""

    run_id: str
    run_kind: RunKind
    lane: RunLane
    submitted_by: str
    access_scope_kind: Literal["owner", "workspace"]
    access_scope_id: str
    status: RunStatus
    phase: RunPhase | None
    durable_progress_version: int
    next_event_sequence: int
    events_trimmed_at: datetime.datetime | None
    cancel_requested: bool
    result: Mapping[str, Any] | None
    error_kind: str | None
    error_message: str | None
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None
    request: Mapping[str, Any]
    repair_reason: str | None = None
    repair_remedy: str | None = None

    @classmethod
    def from_runtime(cls, record: RuntimeRunRecord) -> RunView:
        """Drop worker-only state at the Engine-to-Application boundary."""
        return cls(
            run_id=record.run_id,
            run_kind=record.run_kind,
            lane=record.lane,
            submitted_by=record.submitted_by,
            access_scope_kind=record.access_scope.kind,
            access_scope_id=record.access_scope.scope_id,
            status=record.status,
            phase=record.phase,
            durable_progress_version=record.durable_progress_version,
            next_event_sequence=record.next_event_sequence,
            events_trimmed_at=record.events_trimmed_at,
            cancel_requested=record.cancel_requested,
            result=dict(record.result) if record.result is not None else None,
            error_kind=record.error_kind,
            error_message=record.error_message,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            request=dict(record.request_input()),
            repair_reason=_repair_text(record, "repair_reason"),
            repair_remedy=_repair_text(record, "repair_remedy"),
        )

    def request_input(self) -> Mapping[str, Any]:
        """Return the bounded accepted request used by caller projections."""
        return self.request

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One caller-facing event in a Run's gap-free durable sequence."""

    sequence: int
    event_type: str
    payload: Mapping[str, Any]
    created_at: datetime.datetime

    @classmethod
    def from_runtime(cls, event: RuntimeRunEvent) -> RunEvent:
        return cls(
            sequence=event.sequence,
            event_type=event.event_type,
            payload=dict(event.payload),
            created_at=event.created_at,
        )


@dataclass(frozen=True, slots=True)
class RunCancellation:
    """Caller-facing result of an owner-scoped cancellation request."""

    outcome: RunCancellationResult
    run: RunView | None

    @classmethod
    def from_runtime(cls, outcome: RuntimeCancellationOutcome) -> RunCancellation:
        return cls(
            outcome=outcome.outcome,
            run=RunView.from_runtime(outcome.run) if outcome.run is not None else None,
        )


@dataclass(frozen=True, slots=True)
class RunCreation:
    """Caller-facing accepted Run, including idempotent replay state."""

    run: RunView
    replayed: bool

    @classmethod
    def from_runtime(cls, creation: RuntimeRunCreation) -> RunCreation:
        return cls(run=RunView.from_runtime(creation.run), replayed=bool(creation.replayed))


class RunRepository(Protocol):
    async def get_run(self, *, owner_id: str, run_id: str) -> RuntimeRunRecord | None: ...
    async def get_run_global(self, *, run_id: str) -> RuntimeRunRecord | None: ...
    async def list_runs(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> tuple[RuntimeRunRecord, ...]: ...
    async def request_cancellation(
        self, *, owner_id: str, run_id: str
    ) -> RuntimeCancellationOutcome: ...
    async def resume_repair(self, *, owner_id: str, run_id: str) -> bool: ...


class RunScheduler(Protocol):
    def cancel_local(self, owner_id: str, run_id: str) -> None: ...
    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncIterator[RuntimeRunEvent]: ...


class RunService:
    """The sole generic lifecycle authority exposed by Application."""

    def __init__(self, *, store: RunRepository, scheduler: RunScheduler) -> None:
        self._store = store
        self._scheduler = scheduler

    async def get(self, *, owner_id: str, run_id: str) -> RunView | None:
        record = await self._store.get_run(owner_id=owner_id, run_id=run_id)
        return RunView.from_runtime(record) if record is not None else None

    async def get_global(self, *, run_id: str) -> RunView | None:
        """Return a Run for a transport to authorize by its declared scope."""
        record = await self._store.get_run_global(run_id=run_id)
        return RunView.from_runtime(record) if record is not None else None

    async def list(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> tuple[RunView, ...]:
        records = await self._store.list_runs(
            owner_id=owner_id, after_run_id=after_run_id, limit=limit
        )
        return tuple(RunView.from_runtime(record) for record in records)

    async def cancel(self, *, owner_id: str, run_id: str) -> RunCancellation:
        outcome = await self._store.request_cancellation(owner_id=owner_id, run_id=run_id)
        if outcome.outcome == "pending":
            self._scheduler.cancel_local(owner_id, run_id)
        return RunCancellation.from_runtime(outcome)

    async def resume_repair(self, *, owner_id: str, run_id: str) -> bool:
        resumed = await self._store.resume_repair(owner_id=owner_id, run_id=run_id)
        if resumed:
            wake = getattr(self._scheduler, "wake", None)
            if callable(wake):
                wake()
        return resumed

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncIterator[RunEvent]:
        events = self._scheduler.subscribe(
            owner_id=owner_id, run_id=run_id, after_sequence=after_sequence
        )

        async def _views() -> AsyncIterator[RunEvent]:
            async for event in events:
                yield RunEvent.from_runtime(event)

        return _views()


__all__ = [
    "IdempotencyKeyConflict",
    "RunCancellation",
    "RunCancelledError",
    "RunCapacityExceededError",
    "RunCreation",
    "RunEvent",
    "RunFailedError",
    "RunKind",
    "RunLane",
    "RunRuntimeUnavailableError",
    "RunPhase",
    "RunRepository",
    "RunScheduler",
    "RunService",
    "RunStatus",
    "RunView",
]
