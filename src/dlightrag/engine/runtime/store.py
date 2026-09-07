# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Narrow operation-neutral ports owned by RunRuntime."""

import datetime
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Protocol

from dlightrag.engine.runtime.contracts import RunKind, RunLane, RunPhase
from dlightrag.engine.runtime.records import (
    ClaimedRun,
    LeaseRenewal,
    PreparedRunEnvelope,
    RunCreation,
    RunDeletion,
    RunEvent,
    RunRecord,
    ShutdownOutcome,
    SweepOutcome,
    TerminalOutcome,
)


class RunStore(Protocol):
    """Lifecycle operations used by the coordinator and Run application service."""

    async def accept_run(self, *, envelope: PreparedRunEnvelope, run_id: str) -> RunCreation: ...

    async def claim_next(
        self,
        *,
        worker_id: str,
        run_kinds: Sequence[RunKind],
        lanes: Sequence[RunLane],
    ) -> ClaimedRun | None: ...

    async def heartbeat(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> LeaseRenewal: ...

    async def record_phase(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        phase: RunPhase,
    ) -> int | None: ...

    async def write_checkpoint(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
        phase: RunPhase | None = None,
    ) -> bool: ...

    async def start_handoff(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
    ) -> bool: ...

    async def append_event(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        phase: RunPhase | None,
        event_type: str,
        payload: Mapping[str, object],
    ) -> int | None: ...

    async def finish_success(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        result: Mapping[str, object],
        stop_reason: str | None = None,
        publications: Sequence[object] = (),
    ) -> TerminalOutcome: ...

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
    ) -> TerminalOutcome: ...

    async def finish_cancelled(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> TerminalOutcome: ...

    async def defer(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
        next_attempt_at: datetime.datetime,
    ) -> bool: ...

    async def resume_repair(self, *, owner_id: str, run_id: str) -> bool: ...

    async def wait_for_repair(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
    ) -> bool: ...

    async def release_for_shutdown(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> ShutdownOutcome: ...

    async def sweep_once(self) -> SweepOutcome: ...
    async def trim_expired_event_logs(self) -> int: ...
    async def prune_expired_runs(self) -> RunDeletion: ...
    async def get_run(self, *, owner_id: str, run_id: str) -> RunRecord | None: ...
    async def get_run_global(self, *, run_id: str) -> RunRecord | None: ...
    async def list_runs(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> tuple[RunRecord, ...]: ...
    async def read_event_page(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> tuple[RunEvent, ...]: ...


class RunBlobStore(Protocol):
    """Opaque immutable bytes referenced by run-owned metadata."""

    def stream(
        self,
        *,
        owner_id: str,
        digest: str,
        offset: int = 0,
        length: int | None = None,
    ) -> AsyncIterator[bytes]: ...

    async def read(self, *, owner_id: str, digest: str) -> bytes | None: ...
    async def size(self, *, owner_id: str, digest: str) -> int | None: ...


__all__ = ["RunBlobStore", "RunStore"]
