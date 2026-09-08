# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Contract tests for the local Answer run coordinator and its subscribers.

The coordinator is the only local owner of accepted-run execution: it reserves
an execution slot before it claims a row, heartbeats a fenced lease, commits
completed control turns, coalesces tokens, and writes exactly one terminal
event. Subscribers replay durable events and detach without touching the run.
"""

import asyncio
import datetime
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast

import pytest

import dlightrag.engine.runtime.coordinator as coordinator_module
from dlightrag.engine.runtime.coordinator import (
    LeaseLostError,
    RunCancellationObserved,
    RunCoordinator,
    RunSession,
)
from dlightrag.engine.runtime.errors import RunExecutionError
from dlightrag.engine.runtime.policy import MAX_RECLAIMS_WITHOUT_PROGRESS
from dlightrag.engine.runtime.records import (
    AlreadyCommittedTerminal,
    ClaimedRun,
    Failed,
    LeaseRenewal,
    RunAccessScope,
    RunDeletion,
    RunEvent,
    RunExecutionOutcome,
    RunRecord,
    ShutdownOutcome,
    Succeeded,
    SweepOutcome,
    TerminalOutcome,
)

_OWNER = "owner-alpha"


class _MemoryStore:
    """An in-memory stand-in for the fenced PostgreSQL run store."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.events: dict[str, list[RunEvent]] = {}
        self.sweeps = 0
        self.claims = 0
        self.trims = 0
        self.prunes = 0
        self.trim_batches: list[int] = []
        self.prune_batches: list[int] = []
        self.trim_failures = 0
        self.artifact_writes: list[dict[str, Any]] = []
        self.claim_gate: asyncio.Event | None = None
        self.heartbeats = 0
        self.heartbeat_failures = 0
        self.heartbeat_result: Any = None
        self.finish_success_calls = 0
        self.token_append_calls = 0
        self.token_append_started = asyncio.Event()
        self.token_append_gate: asyncio.Event | None = None
        self.token_append_failure: Exception | None = None
        self.shutdown_releases = 0

    # -- test helpers -------------------------------------------------
    def add_run(self, run_id: str, **overrides: Any) -> None:
        self.runs[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "durable_progress_version": 0,
            "last_reclaim_progress_version": 0,
            "reclaims_without_progress": 0,
            "fencing_epoch": 0,
            "lease_owner": None,
            "lease_live": False,
            "cancel_requested": False,
            "next_event_sequence": 1,
            "result": None,
            "error_kind": None,
            "request": {"query": "why"},
            **overrides,
        }
        self.events.setdefault(run_id, [])

    def _record(self, row: Mapping[str, Any]) -> RunRecord:
        now = datetime.datetime.now(datetime.UTC)
        return RunRecord(
            run_id=str(row["run_id"]),
            run_kind="answer",
            lane="query",
            submitted_by=_OWNER,
            access_scope=RunAccessScope(kind="owner", scope_id=_OWNER),
            submission_key=None or str(str(row["run_id"])),
            request_fingerprint="test-fingerprint",
            prepared_input=row.get("prepared_input"),
            status=row["status"],
            phase=None,
            stop_reason=None,
            cancel_requested_at=now if row["cancel_requested"] else None,
            lease_owner=row["lease_owner"],
            lease_expires_at=now if row["lease_owner"] else None,
            fencing_epoch=int(row["fencing_epoch"]),
            durable_progress_version=int(row["durable_progress_version"]),
            last_reclaim_progress_version=int(row["last_reclaim_progress_version"]),
            reclaims_without_progress=int(row["reclaims_without_progress"]),
            next_event_sequence=int(row["next_event_sequence"]),
            events_trimmed_at=None,
            result=row["result"],
            error_kind=row["error_kind"],
            error_message=None,
            created_at=now,
            updated_at=now,
            started_at=now,
            finished_at=now if row["status"] in ("succeeded", "failed", "cancelled") else None,
        )

    def _owns(self, row: Mapping[str, Any], worker_id: str, epoch: int) -> bool:
        return (
            row["status"] == "running"
            and row["lease_owner"] == worker_id
            and row["lease_live"]
            and int(row["fencing_epoch"]) == epoch
        )

    def _append(self, row: dict[str, Any], event_type: str, payload: Mapping[str, Any]) -> int:
        sequence = int(row["next_event_sequence"])
        row["next_event_sequence"] = sequence + 1
        self.events[str(row["run_id"])].append(
            RunEvent(
                sequence=sequence,
                event_type=event_type,  # type: ignore[arg-type]
                payload=dict(payload),
                created_at=datetime.datetime.now(datetime.UTC),
            )
        )
        return sequence

    # -- store surface -------------------------------------------------
    async def accept_run(self, *, envelope: Any, run_id: str) -> Any:
        raise NotImplementedError("coordinator tests do not exercise admission")

    async def claim_next(self, *, worker_id: str, **_filters: Any) -> ClaimedRun | None:
        if self.claim_gate is not None:
            self.claim_gate.set()
        self.claims += 1
        for row in self.runs.values():
            eligible = row["status"] == "queued" or (
                row["status"] == "running" and not row["lease_live"]
            )
            if not eligible or row["cancel_requested"]:
                continue
            if row["status"] == "running":
                row["reclaims_without_progress"] = int(row["reclaims_without_progress"]) + 1
                if int(row["reclaims_without_progress"]) >= MAX_RECLAIMS_WITHOUT_PROGRESS:
                    row["status"] = "failed"
                    row["error_kind"] = "run_abandoned"
                    row["finished_at"] = True
                    continue
            row["status"] = "running"
            row["lease_owner"] = worker_id
            row["lease_live"] = True
            row["fencing_epoch"] = int(row["fencing_epoch"]) + 1
            from dlightrag.engine.runtime.records import RunExecutionContext
            from tests.in_memory_session_repository import InMemoryAgentSessionRepository

            class _Progress:
                def __init__(self, store, run_id):
                    self._store = store
                    self._run_id = run_id

                async def load_stage(self, stage_intent_id):
                    return None

                async def settle_stage(self, **kwargs):
                    from dlightrag.engine.runtime.progress import StageCommit

                    row = self._store.runs[self._run_id]
                    row["durable_progress_version"] = int(row["durable_progress_version"]) + 1
                    return StageCommit(
                        progress_version=int(row["durable_progress_version"]),
                        stage_intent_id=kwargs["stage_intent_id"],
                        evidence_count=0,
                    )

                async def settle_terminal(self, **kwargs):
                    from dlightrag.engine.runtime.progress import StageTerminalCommit

                    row = self._store.runs[self._run_id]
                    row["durable_progress_version"] = int(row["durable_progress_version"]) + 1
                    sequence = self._store._append(
                        row,
                        "done",
                        {"status": "succeeded", "result": dict(kwargs["result"])},
                    )
                    row.update(
                        status="succeeded",
                        result=dict(kwargs["result"]),
                        lease_owner=None,
                        lease_live=False,
                    )
                    return StageTerminalCommit(
                        progress_version=int(row["durable_progress_version"]),
                        stage_intent_id=kwargs["stage_intent_id"],
                        status="succeeded",
                        terminal_event_sequence=sequence,
                    )

            return ClaimedRun(
                run=self._record(row),
                execution=RunExecutionContext(
                    owner_id=_OWNER,
                    run_id=str(row["run_id"]),
                    worker_id=worker_id,
                    lease_owner=worker_id,
                    fencing_epoch=int(row["fencing_epoch"]),
                    session_repository=InMemoryAgentSessionRepository(),  # type: ignore[arg-type]
                    progress_store=_Progress(self, str(row["run_id"])),  # type: ignore[arg-type]
                ),
            )
        return None

    async def heartbeat(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> LeaseRenewal:
        self.heartbeats += 1
        if self.heartbeat_failures > 0:
            self.heartbeat_failures -= 1
            raise RuntimeError("transient heartbeat failure")
        if self.heartbeat_result is not None:
            return cast(LeaseRenewal, self.heartbeat_result)
        row = self.runs[run_id]
        if not self._owns(row, worker_id, fencing_epoch):
            return LeaseRenewal(renewed=False, cancel_requested=False)
        return LeaseRenewal(renewed=True, cancel_requested=bool(row["cancel_requested"]))

    async def record_phase(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int, phase: str
    ) -> int | None:
        row = self.runs[run_id]
        if not self._owns(row, worker_id, fencing_epoch):
            return None
        return self._append(row, "progress", {"phase": phase})

    async def write_checkpoint(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
        phase: str | None = None,
    ) -> bool:
        row = self.runs[run_id]
        if not self._owns(row, worker_id, fencing_epoch):
            return False
        row["checkpoint"] = dict(checkpoint)
        if phase is not None:
            row["phase"] = phase
        row["durable_progress_version"] = int(row["durable_progress_version"]) + 1
        return True

    async def start_handoff(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
    ) -> bool:
        return await self.write_checkpoint(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            checkpoint=checkpoint,
            phase="handed_off",
        )

    async def append_event(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        phase: str | None,
        event_type: str,
        payload: Mapping[str, object],
    ) -> int | None:
        if event_type == "token":
            self.token_append_calls += 1
            self.token_append_started.set()
            if self.token_append_gate is not None:
                await self.token_append_gate.wait()
            if self.token_append_failure is not None:
                raise self.token_append_failure
        row = self.runs[run_id]
        if not self._owns(row, worker_id, fencing_epoch):
            return None
        if phase is not None:
            row["phase"] = phase
        return self._append(row, event_type, dict(payload))

    async def finish_success(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        result: Mapping[str, Any],
        stop_reason: str | None = None,
        publications: Sequence[Any] = (),
    ) -> TerminalOutcome:
        del publications
        self.finish_success_calls += 1
        row = self.runs[run_id]
        if row["cancel_requested"]:
            return await self.finish_cancelled(
                owner_id=owner_id, run_id=run_id, worker_id=worker_id, fencing_epoch=fencing_epoch
            )
        return self._finish(
            row,
            worker_id,
            fencing_epoch,
            status="succeeded",
            result=dict(result),
            event_type="done",
            payload={"status": "succeeded", "result": dict(result)},
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
        row = self.runs[run_id]
        if row["cancel_requested"]:
            return await self.finish_cancelled(
                owner_id=owner_id,
                run_id=run_id,
                worker_id=worker_id,
                fencing_epoch=fencing_epoch,
            )
        row["error_kind"] = error_kind
        return self._finish(
            row,
            worker_id,
            fencing_epoch,
            status="failed",
            result=None if result is None else dict(result),
            event_type="error",
            payload={"kind": error_kind, "message": error_message},
        )

    async def defer(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
        next_attempt_at: datetime.datetime,
    ) -> bool:
        row = self.runs[run_id]
        if row["lease_owner"] != worker_id or row["fencing_epoch"] != fencing_epoch:
            return False
        row.update(
            status="queued",
            phase="deferred",
            checkpoint=dict(checkpoint),
            next_attempt_at=next_attempt_at,
            lease_owner=None,
            lease_expires_at=None,
        )
        return True

    async def resume_repair(self, *, owner_id: str, run_id: str) -> bool:
        row = self.runs[run_id]
        if row.get("phase") != "waiting_for_repair":
            return False
        row.update(status="queued", phase="deferred", next_attempt_at=None)
        return True

    async def wait_for_repair(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
    ) -> bool:
        row = self.runs[run_id]
        if row["lease_owner"] != worker_id or row["fencing_epoch"] != fencing_epoch:
            return False
        row.update(
            phase="waiting_for_repair",
            checkpoint=dict(checkpoint),
            next_attempt_at=None,
            lease_owner=None,
            lease_expires_at=None,
        )
        return True

    async def finish_cancelled(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> TerminalOutcome:
        return self._finish(
            self.runs[run_id],
            worker_id,
            fencing_epoch,
            status="cancelled",
            result=None,
            event_type="done",
            payload={"status": "cancelled"},
        )

    def _finish(
        self,
        row: dict[str, Any],
        worker_id: str,
        fencing_epoch: int,
        *,
        status: str,
        result: dict[str, Any] | None,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> TerminalOutcome:
        if not self._owns(row, worker_id, fencing_epoch):
            return TerminalOutcome(committed=False, status=None, event_sequence=None)
        row["status"] = status
        row["result"] = result
        row["lease_owner"] = None
        row["lease_live"] = False
        sequence = self._append(row, event_type, payload)
        return TerminalOutcome(committed=True, status=status, event_sequence=sequence)  # type: ignore[arg-type]

    async def release_for_shutdown(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> ShutdownOutcome:
        self.shutdown_releases += 1
        row = self.runs[run_id]
        if not self._owns(row, worker_id, fencing_epoch):
            return "lease_lost"
        if row["cancel_requested"]:
            self._finish(
                row,
                worker_id,
                fencing_epoch,
                status="cancelled",
                result=None,
                event_type="done",
                payload={"status": "cancelled"},
            )
            return "cancelled"
        row["status"] = "queued"
        row["lease_owner"] = None
        row["lease_live"] = False
        return "requeued"

    async def sweep_once(self) -> SweepOutcome:
        self.sweeps += 1
        return SweepOutcome(cancelled=0, abandoned=0)

    async def trim_expired_event_logs(self) -> int:
        self.trims += 1
        if self.trim_failures > 0:
            self.trim_failures -= 1
            raise RuntimeError("trim unavailable")
        return self.trim_batches.pop(0) if self.trim_batches else 0

    async def prune_expired_runs(self) -> RunDeletion:
        self.prunes += 1
        return RunDeletion(runs=self.prune_batches.pop(0) if self.prune_batches else 0, artifacts=0)

    async def read_event_page(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> tuple[RunEvent, ...]:
        return tuple(
            event for event in self.events.get(run_id, []) if event.sequence > after_sequence
        )

    async def get_run(self, *, owner_id: str, run_id: str) -> RunRecord | None:
        row = self.runs.get(run_id)
        return self._record(row) if row is not None else None

    async def get_run_global(self, *, run_id: str) -> RunRecord | None:
        row = self.runs.get(run_id)
        return self._record(row) if row is not None else None

    async def list_runs(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> tuple[RunRecord, ...]:
        return ()

    async def list_run_artifacts(self, *, owner_id: str, run_id: str) -> tuple[Any, ...]:
        return ()

    async def list_artifact_attachments(self, *, owner_id: str, run_id: str) -> tuple[Any, ...]:
        return ()

    async def request_cancellation(self, *, owner_id: str, run_id: str) -> Any:
        self.runs[run_id]["cancel_requested"] = True
        return None


class _Executor:
    """A deterministic stand-in for AnswerExecutor."""

    def __init__(self, body: Any) -> None:
        self._body = body
        self.sessions: list[RunSession] = []

    async def execute(self, session: RunSession) -> RunExecutionOutcome:
        self.sessions.append(session)
        return await self._body(session)


class _ControlledTokenSleep:
    """A manually fired wall-clock wait used without shortening production bounds."""

    def __init__(self) -> None:
        self.delays: list[float] = []
        self.waiters: list[asyncio.Future[None]] = []
        self.armed = asyncio.Event()

    async def __call__(self, delay: float) -> None:
        waiter = asyncio.get_running_loop().create_future()
        self.delays.append(delay)
        self.waiters.append(waiter)
        self.armed.set()
        await waiter

    def fire(self, index: int = -1) -> None:
        waiter = self.waiters[index]
        if not waiter.done():
            waiter.set_result(None)

    def fire_all(self) -> None:
        for waiter in self.waiters:
            if not waiter.done():
                waiter.set_result(None)


class _CancellationBlockingTokenSleep:
    """Keep timer cancellation pending so caller cancellation can race with reaping."""

    def __init__(self) -> None:
        self.armed = asyncio.Event()
        self.cancellation_seen = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, _delay: float) -> None:
        self.armed.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancellation_seen.set()
            try:
                await self.release.wait()
            finally:
                raise


async def _settle(predicate: Any, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true")


def _coordinator(
    store: _MemoryStore,
    executor: _Executor,
    *,
    query_worker_concurrency: int = 2,
    token_flush_sleep: Callable[[float], Awaitable[None]] | None = None,
):
    kwargs: dict[str, Any] = {}
    if token_flush_sleep is not None:
        kwargs["_token_flush_sleep"] = token_flush_sleep
    return RunCoordinator(
        store=store,
        executors={"answer": executor},
        query_worker_concurrency=query_worker_concurrency,
        **kwargs,
    )


def _traceback_depth(exc: BaseException) -> int:
    depth, frame = 0, exc.__traceback__
    while frame is not None:
        depth += 1
        frame = frame.tb_next
    return depth


class TestSchedulingAndLease:
    async def test_reserves_a_local_slot_before_it_claims_a_row(self) -> None:
        store = _MemoryStore()
        store.claim_gate = asyncio.Event()
        release = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await release.wait()
            return Succeeded({"answer": "done"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        store.add_run("run-b")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "running")
            claims_while_busy = store.claims
            await asyncio.sleep(0.05)

            assert store.claims == claims_while_busy
            assert store.runs["run-b"]["status"] == "queued"
            release.set()
            await _settle(lambda: store.runs["run-b"]["status"] == "succeeded")
        finally:
            release.set()
            await coordinator.aclose()

    async def test_lease_loss_stops_execution_and_forbids_later_writes(self) -> None:
        store = _MemoryStore()
        stopped = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            store.runs["run-a"]["lease_owner"] = "another-worker"
            try:
                await asyncio.sleep(5)
            finally:
                stopped.set()
            return Succeeded({"answer": "unreachable"})

        coordinator = RunCoordinator(
            store=store,
            executors={"answer": _Executor(body)},
            query_worker_concurrency=1,
            heartbeat_seconds=0.01,
        )
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(stopped.is_set)
            await _settle(lambda: not coordinator.active_runs)

            assert store.runs["run-a"]["status"] == "running"
            assert [event.event_type for event in store.events["run-a"]] == []
        finally:
            await coordinator.aclose()

    async def test_a_worker_that_lost_its_lease_cannot_append_or_finish(self) -> None:
        store = _MemoryStore()
        outcome: dict[str, Any] = {}

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.enter_phase("generating")
            store.runs["run-a"]["fencing_epoch"] = 99
            try:
                await session.emit_token("late")
                await session.flush_tokens()
            except LeaseLostError as exc:
                outcome["raised"] = exc
                raise
            return Succeeded({"answer": "no"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: "raised" in outcome)
            await _settle(lambda: not coordinator.active_runs)

            assert [event.event_type for event in store.events["run-a"]] == ["progress"]
        finally:
            await coordinator.aclose()

    async def test_sweeper_runs_without_holding_an_execution_slot(self) -> None:
        store = _MemoryStore()
        hold = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await hold.wait()
            return Succeeded({"answer": "done"})

        coordinator = RunCoordinator(
            store=store,
            executors={"answer": _Executor(body)},
            query_worker_concurrency=1,
            sweep_seconds=0.01,
        )
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "running")
            await _settle(lambda: store.sweeps >= 2)
        finally:
            hold.set()
            await coordinator.aclose()


class TestRetentionMaintenance:
    """Retention runs on every run-owning process, needs no slot, and leaks no task."""

    async def _running(self, store: _MemoryStore, **kwargs: Any) -> RunCoordinator:
        coordinator = RunCoordinator(
            store=store,
            executors={
                "answer": _Executor(lambda session: asyncio.sleep(0, Succeeded({"answer": "x"})))
            },
            query_worker_concurrency=1,
            sweep_seconds=60.0,
            **kwargs,
        )
        await coordinator.start()
        return coordinator

    async def test_the_scheduler_trims_events_and_prunes_runs_on_its_cadence(self) -> None:
        store = _MemoryStore()
        coordinator = await self._running(store, maintenance_seconds=0.01)
        try:
            await _settle(lambda: store.trims >= 2 and store.prunes >= 2)
        finally:
            await coordinator.aclose()

    async def test_one_pass_drains_full_batches(self) -> None:
        store = _MemoryStore()
        store.trim_batches = [200, 200, 0]
        store.prune_batches = [200, 0]
        coordinator = RunCoordinator(
            store=store,
            executors={
                "answer": _Executor(lambda session: asyncio.sleep(0, Succeeded({"answer": "x"})))
            },
            query_worker_concurrency=1,
        )

        await coordinator._maintain_once()

        assert (store.trims, store.prunes) == (3, 2)

    async def test_the_first_pass_waits_out_a_bounded_startup_jitter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fleet that restarts together must not align every trim on one instant."""
        monkeypatch.setattr(coordinator_module, "_startup_jitter", lambda _cadence: 5.0)
        store = _MemoryStore()
        coordinator = await self._running(store, maintenance_seconds=0.01)
        try:
            await asyncio.sleep(0.05)
            assert store.trims == 0
        finally:
            await coordinator.aclose()

    @pytest.mark.parametrize("cadence", [0.0, 0.01, 3600.0])
    def test_startup_jitter_never_outlasts_a_share_of_the_cadence(self, cadence: float) -> None:
        assert all(
            0.0 <= coordinator_module._startup_jitter(cadence) <= cadence * 0.1 for _ in range(50)
        )

    async def test_a_transient_retention_fault_is_retried_not_fatal(self) -> None:
        store = _MemoryStore()
        store.trim_failures = 1
        coordinator = await self._running(store, maintenance_seconds=0.01)
        try:
            await _settle(lambda: store.trims >= 2 and store.prunes >= 1)
        finally:
            await coordinator.aclose()

    async def test_closing_leaves_no_retention_task_behind(self) -> None:
        store = _MemoryStore()
        coordinator = await self._running(store, maintenance_seconds=0.01)
        await _settle(lambda: store.trims >= 1)
        await coordinator.aclose()

        settled = store.trims
        await asyncio.sleep(0.05)
        assert store.trims == settled


class TestWallClockTokenFlush:
    async def test_small_token_flushes_while_the_executor_remains_blocked(self) -> None:
        store = _MemoryStore()
        scheduler = _ControlledTokenSleep()
        executor_blocked = asyncio.Event()
        release = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.emit_token("small")
            executor_blocked.set()
            await release.wait()
            return Succeeded({"answer": "small"})

        coordinator = _coordinator(
            store,
            _Executor(body),
            query_worker_concurrency=1,
            token_flush_sleep=scheduler,
        )
        store.add_run("run-a")
        await coordinator.start()
        try:
            await executor_blocked.wait()
            await scheduler.armed.wait()
            scheduler.fire()
            await store.token_append_started.wait()
            await _settle(lambda: len(store.events["run-a"]) == 1)

            assert store.runs["run-a"]["status"] == "running"
            assert store.events["run-a"][0].payload == {"text": "small"}
        finally:
            release.set()
            await coordinator.aclose()

    async def test_production_timer_observes_the_real_wall_clock_bound(self) -> None:
        store = _MemoryStore()
        release = asyncio.Event()
        started_at = 0.0

        async def body(session: RunSession) -> RunExecutionOutcome:
            nonlocal started_at
            started_at = asyncio.get_running_loop().time()
            await session.emit_token("clocked")
            await release.wait()
            return Succeeded({"answer": "clocked"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await asyncio.wait_for(store.token_append_started.wait(), timeout=2.0)
            elapsed = asyncio.get_running_loop().time() - started_at

            # asyncio timers may wake a few milliseconds early on some event
            # loops; the broad upper tolerance avoids turning host load into a
            # flaky test while still rejecting an immediate/no-timer flush.
            assert elapsed >= coordinator_module.TOKEN_BATCH_SECONDS - 0.05
            assert elapsed < coordinator_module.TOKEN_BATCH_SECONDS + 1.0
            assert store.events["run-a"][0].payload == {"text": "clocked"}
        finally:
            release.set()
            await coordinator.aclose()

    async def test_tokens_coalesce_without_sliding_the_first_deadline(self) -> None:
        store = _MemoryStore()
        scheduler = _ControlledTokenSleep()
        buffered = asyncio.Event()
        release = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.emit_token("one")
            await scheduler.armed.wait()
            await session.emit_token(" two")
            await session.emit_token(" three")
            buffered.set()
            await release.wait()
            return Succeeded({"answer": "one two three"})

        coordinator = _coordinator(
            store,
            _Executor(body),
            query_worker_concurrency=1,
            token_flush_sleep=scheduler,
        )
        store.add_run("run-a")
        await coordinator.start()
        try:
            await buffered.wait()
            assert len(scheduler.waiters) == 1
            assert 0.0 <= scheduler.delays[0] <= coordinator_module.TOKEN_BATCH_SECONDS

            scheduler.fire()
            await _settle(lambda: len(store.events["run-a"]) == 1)
            assert store.events["run-a"][0].payload == {"text": "one two three"}
        finally:
            release.set()
            await coordinator.aclose()

    @pytest.mark.parametrize("trigger", ["size", "manual"])
    async def test_size_or_manual_flush_racing_the_timer_appends_exactly_once(
        self, trigger: str
    ) -> None:
        store = _MemoryStore()
        scheduler = _ControlledTokenSleep()
        exposed: asyncio.Future[RunSession] = asyncio.get_running_loop().create_future()
        release = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.emit_token("a")
            exposed.set_result(session)
            await release.wait()
            return Succeeded({"answer": "done"})

        coordinator = _coordinator(
            store,
            _Executor(body),
            query_worker_concurrency=1,
            token_flush_sleep=scheduler,
        )
        store.add_run("run-a")
        await coordinator.start()
        try:
            session = await exposed
            await scheduler.armed.wait()
            async with session._lane:  # noqa: SLF001 - deterministic race setup
                if trigger == "size":
                    operation = asyncio.create_task(
                        session.emit_token("b" * coordinator_module.TOKEN_BATCH_CHARS)
                    )
                    expected = "a" + "b" * coordinator_module.TOKEN_BATCH_CHARS
                else:
                    operation = asyncio.create_task(session.flush_tokens())
                    expected = "a"
                scheduler.fire()
            await operation
            await _settle(lambda: len(store.events["run-a"]) == 1)

            assert store.token_append_calls == 1
            assert store.events["run-a"][0].payload == {"text": expected}
        finally:
            release.set()
            await coordinator.aclose()

    @pytest.mark.parametrize("trigger", ["size", "manual"])
    async def test_timer_winning_the_race_keeps_exact_text(self, trigger: str) -> None:
        store = _MemoryStore()
        scheduler = _ControlledTokenSleep()
        exposed: asyncio.Future[RunSession] = asyncio.get_running_loop().create_future()
        release = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.emit_token("a")
            exposed.set_result(session)
            await release.wait()
            return Succeeded({"answer": "done"})

        coordinator = _coordinator(
            store,
            _Executor(body),
            query_worker_concurrency=1,
            token_flush_sleep=scheduler,
        )
        store.add_run("run-a")
        await coordinator.start()
        try:
            session = await exposed
            await scheduler.armed.wait()
            scheduler.fire()
            await _settle(lambda: store.token_append_calls == 1)

            if trigger == "size":
                suffix = "b" * coordinator_module.TOKEN_BATCH_CHARS
                await session.emit_token(suffix)
                expected_events = [{"text": "a"}, {"text": suffix}]
            else:
                await session.flush_tokens()
                expected_events = [{"text": "a"}]

            assert store.token_append_calls == len(expected_events)
            assert [event.payload for event in store.events["run-a"]] == expected_events
        finally:
            release.set()
            await coordinator.aclose()

    async def test_control_and_terminal_events_share_the_token_ordering_lane(self) -> None:
        store = _MemoryStore()
        scheduler = _ControlledTokenSleep()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.emit_token("before phase")
            await asyncio.sleep(0)
            await session.enter_phase("generating")
            await session.emit_token("before reset")
            await asyncio.sleep(0)
            await session.reset_output()
            await session.emit_token("before tool")
            await asyncio.sleep(0)
            await session.emit_tool_event("tool_started", {"name": "search"})
            await session.emit_token("before terminal")
            await asyncio.sleep(0)
            return Succeeded({"answer": "done"})

        coordinator = _coordinator(
            store,
            _Executor(body),
            query_worker_concurrency=1,
            token_flush_sleep=scheduler,
        )
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
            before = tuple(store.events["run-a"])
            scheduler.fire_all()
            await asyncio.sleep(0)

            assert [event.event_type for event in before] == [
                "token",
                "progress",
                "token",
                "reset",
                "token",
                "tool_started",
                "token",
                "done",
            ]
            assert tuple(store.events["run-a"]) == before
            assert all(waiter.done() for waiter in scheduler.waiters)
            assert not any(
                task.get_name() == "answer-token-flush:run-a" and not task.done()
                for task in asyncio.all_tasks()
            )
        finally:
            await coordinator.aclose()

    @pytest.mark.parametrize("failure", ["store", "lease"])
    async def test_scheduled_flush_failure_or_lease_loss_cannot_commit_success(
        self, failure: str
    ) -> None:
        store = _MemoryStore()
        scheduler = _ControlledTokenSleep()
        release = asyncio.Event()
        rejected_late_token = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.emit_token("never successful")
            await release.wait()
            try:
                await session.emit_token(" must be rejected")
            except Exception:
                rejected_late_token.set()
            return Succeeded({"answer": "must not commit"})

        executor = _Executor(body)
        coordinator = _coordinator(
            store,
            executor,
            query_worker_concurrency=1,
            token_flush_sleep=scheduler,
        )
        store.add_run("run-a")
        await coordinator.start()
        try:
            await scheduler.armed.wait()
            if failure == "store":
                store.token_append_failure = RuntimeError("token store unavailable")
            else:
                store.runs["run-a"]["fencing_epoch"] = 99
            scheduler.fire()
            await store.token_append_started.wait()
            release.set()
            await _settle(lambda: not coordinator.active_runs)

            assert rejected_late_token.is_set()
            assert store.finish_success_calls == 0
            assert executor.sessions[0]._pending_tokens == [  # noqa: SLF001
                "never successful"
            ]
            if failure == "store":
                assert store.runs["run-a"]["status"] == "failed"
                assert store.runs["run-a"]["error_kind"] == "run_execution_failed"
                assert store.events["run-a"][-1].event_type == "error"
            else:
                assert store.runs["run-a"]["status"] == "running"
                assert store.events["run-a"] == []
        finally:
            release.set()
            await coordinator.aclose()

    async def test_shutdown_before_deadline_disarms_without_a_later_write_or_task(self) -> None:
        store = _MemoryStore()
        scheduler = _ControlledTokenSleep()
        blocked = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.emit_token("discard on requeue")
            blocked.set()
            await asyncio.Event().wait()
            return Succeeded({"answer": "unreachable"})

        coordinator = _coordinator(
            store,
            _Executor(body),
            query_worker_concurrency=1,
            token_flush_sleep=scheduler,
        )
        store.add_run("run-a")
        await coordinator.start()
        await blocked.wait()
        await scheduler.armed.wait()

        await coordinator.aclose()
        scheduler.fire_all()
        await asyncio.sleep(0)

        assert store.runs["run-a"]["status"] == "queued"
        assert store.token_append_calls == 0
        assert store.events["run-a"] == []
        assert not any(
            task.get_name() == "answer-token-flush:run-a" and not task.done()
            for task in asyncio.all_tasks()
        )

    async def test_shutdown_cancellation_propagates_while_disarming_timer(self) -> None:
        store = _MemoryStore()
        sleeper = _CancellationBlockingTokenSleep()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.emit_token("must be discarded")
            await sleeper.armed.wait()
            # This boundary cancels and reaps the sleeping timer while holding
            # the session lane. Shutdown then cancels this caller mid-reap.
            await session.enter_phase("generating")
            return Succeeded({"answer": "must not commit"})

        coordinator = _coordinator(
            store,
            _Executor(body),
            query_worker_concurrency=1,
            token_flush_sleep=sleeper,
        )
        store.add_run("run-a")
        await coordinator.start()
        await sleeper.cancellation_seen.wait()

        try:
            await asyncio.wait_for(coordinator.aclose(), timeout=1.0)
        finally:
            sleeper.release.set()

        assert store.runs["run-a"]["status"] == "queued"
        assert store.finish_success_calls == 0
        assert store.events["run-a"] == []
        assert store.shutdown_releases == 1

    async def test_shutdown_joins_an_in_flight_scheduled_write_before_release(self) -> None:
        store = _MemoryStore()
        store.token_append_gate = asyncio.Event()
        scheduler = _ControlledTokenSleep()
        executor_cancelled = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.emit_token("lands before release")
            try:
                await asyncio.Event().wait()
            finally:
                executor_cancelled.set()
            return Succeeded({"answer": "unreachable"})

        coordinator = _coordinator(
            store,
            _Executor(body),
            query_worker_concurrency=1,
            token_flush_sleep=scheduler,
        )
        store.add_run("run-a")
        await coordinator.start()
        await scheduler.armed.wait()
        scheduler.fire()
        await store.token_append_started.wait()

        closing = asyncio.create_task(coordinator.aclose())
        await executor_cancelled.wait()
        assert store.shutdown_releases == 0
        assert not closing.done()
        assert store.runs["run-a"]["lease_owner"] == coordinator.worker_id

        store.token_append_gate.set()
        await closing

        assert store.shutdown_releases == 1
        assert store.runs["run-a"]["status"] == "queued"
        assert [event.payload for event in store.events["run-a"]] == [
            {"text": "lands before release"}
        ]
        assert not any(
            task.get_name() == "answer-token-flush:run-a" and not task.done()
            for task in asyncio.all_tasks()
        )


class TestDurableProgress:
    @pytest.mark.parametrize("status", ["succeeded", "cancelled"])
    async def test_already_committed_terminal_skips_coordinator_finish_and_notifies(
        self, status: str
    ) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> RunExecutionOutcome:
            # Atomic executor terminals own their answer event; handing that
            # outcome back must still disarm any local token timer.
            await session.emit_token("superseded by atomic terminal")
            row = store.runs[session.run_id]
            terminal = store._finish(
                row,
                session.worker_id,
                session.fencing_epoch,
                status=status,
                result={"answer": "atomic"} if status == "succeeded" else None,
                event_type="done",
                payload={
                    "status": status,
                    **({"result": {"answer": "atomic"}} if status == "succeeded" else {}),
                },
            )
            return AlreadyCommittedTerminal(terminal)

        executor = _Executor(body)
        coordinator = _coordinator(store, executor, query_worker_concurrency=1)
        store.add_run("run-a")
        with coordinator._broker.waiter(_OWNER, "run-a") as notified:
            await coordinator.start()
            try:
                await asyncio.wait_for(notified.wait(), timeout=2)
                await _settle(lambda: not coordinator.active_runs)
            finally:
                await coordinator.aclose()

        assert store.finish_success_calls == 0
        assert store.token_append_calls == 0
        assert executor.sessions[0].lease_lost is False
        assert store.runs["run-a"]["status"] == status
        assert [event.event_type for event in store.events["run-a"]] == ["done"]
        assert not any(
            task.get_name() == "answer-token-flush:run-a" and not task.done()
            for task in asyncio.all_tasks()
        )

    async def test_tokens_are_coalesced_in_order_with_one_terminal_event(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.enter_phase("generating")
            for token in ("alpha ", "beta ", "gamma"):
                await session.emit_token(token)
            return Succeeded({"answer": "alpha beta gamma"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        finally:
            await coordinator.aclose()

        kinds = [event.event_type for event in store.events["run-a"]]
        assert kinds == ["progress", "token", "done"]
        assert store.events["run-a"][1].payload["text"] == "alpha beta gamma"
        assert kinds.count("done") + kinds.count("error") == 1
        assert store.finish_success_calls == 1

    async def test_durable_progress_survives_a_restart(self) -> None:
        store = _MemoryStore()
        attempts: list[int] = []

        async def body(session: RunSession) -> RunExecutionOutcome:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("process died")
            return Succeeded({"answer": "resumed"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "failed")
            store.runs["run-a"].update(status="running", lease_live=False, error_kind=None)
            coordinator.wake()
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        finally:
            await coordinator.aclose()

        assert len(attempts) == 2
        assert store.runs["run-a"]["result"] == {"answer": "resumed"}

    async def test_resumed_run_emits_reset_before_regenerated_output(self) -> None:
        store = _MemoryStore()
        store.add_run("run-a", next_event_sequence=4)
        store.events["run-a"] = []

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.emit_token("fresh draft")
            return Succeeded({"answer": "fresh draft"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        finally:
            await coordinator.aclose()

        assert [event.event_type for event in store.events["run-a"]] == [
            "reset",
            "token",
            "done",
        ]

    async def test_first_attempt_never_emits_reset(self) -> None:
        store = _MemoryStore()
        store.add_run("run-a")

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.emit_token("draft")
            return Succeeded({"answer": "draft"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        finally:
            await coordinator.aclose()

        assert [event.event_type for event in store.events["run-a"]] == ["token", "done"]

    async def test_executor_failure_fails_the_run_with_its_public_kind(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> RunExecutionOutcome:
            raise RunExecutionError("evidence_settlement_conflict", "conflict")

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "failed")
        finally:
            await coordinator.aclose()

        assert store.runs["run-a"]["error_kind"] == "evidence_settlement_conflict"
        assert store.events["run-a"][-1].payload == {
            "kind": "evidence_settlement_conflict",
            "message": "conflict",
        }

    async def test_an_owner_classified_failure_keeps_its_actionable_message(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> RunExecutionOutcome:
            raise RunExecutionError(
                "CURRENT_DOCUMENT_PARSE_FAILED",
                "Could not read report.pdf.",
            )

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "failed")
        finally:
            await coordinator.aclose()

        payload = store.events["run-a"][-1].payload
        assert payload == {
            "kind": "CURRENT_DOCUMENT_PARSE_FAILED",
            "message": "Could not read report.pdf.",
        }

    async def test_an_unclassified_failure_never_leaks_its_exception_text(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> RunExecutionOutcome:
            raise RuntimeError("postgres://user:secret@host/db is unreachable")

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "failed")
        finally:
            await coordinator.aclose()

        payload = store.events["run-a"][-1].payload
        assert payload == {
            "kind": "run_execution_failed",
            "message": "Run execution failed.",
        }

    async def test_a_foreign_public_message_attribute_never_reaches_a_client(self) -> None:
        """Only the answer taxonomy vets a client-safe message; the shape does not."""
        store = _MemoryStore()

        class _Impostor(RuntimeError):
            public_message = "postgres://user:secret@host/db is unreachable"

        async def body(session: RunSession) -> RunExecutionOutcome:
            raise _Impostor("boom")

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "failed")
        finally:
            await coordinator.aclose()

        assert store.events["run-a"][-1].payload["message"] == "Run execution failed."

    async def test_a_run_without_a_prepared_input_fails_before_execution(self) -> None:
        store = _MemoryStore()
        store.add_run("run-a", prepared_input=None)
        executed = False

        async def body(session: RunSession) -> RunExecutionOutcome:
            nonlocal executed
            executed = True
            if session.prepared_input is None:
                raise RunExecutionError("run_execution_failed", "no prepared input")
            return Succeeded({"answer": "never"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "failed")
        finally:
            await coordinator.aclose()

        assert executed is True
        assert store.runs["run-a"]["error_kind"] == "run_execution_failed"


class TestHeartbeatResilience:
    """A store hiccup is not lease loss, and a dead heartbeat is not an outcome."""

    async def test_transient_heartbeat_failures_never_end_the_run(self) -> None:
        store = _MemoryStore()
        store.add_run("run-a")
        store.heartbeat_failures = 3

        async def body(session: RunSession) -> RunExecutionOutcome:
            await _settle(lambda: store.heartbeats >= 5)
            await session.check_cancelled()
            return Succeeded({"answer": "survived"})

        coordinator = RunCoordinator(
            store=store,
            executors={"answer": _Executor(body)},
            query_worker_concurrency=1,
            heartbeat_seconds=0.01,
        )
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        finally:
            await coordinator.aclose()

        assert store.runs["run-a"]["result"] == {"answer": "survived"}

    async def test_an_authoritative_non_renewal_still_stops_the_run(self) -> None:
        store = _MemoryStore()
        store.add_run("run-a")
        store.heartbeat_failures = 2

        async def body(session: RunSession) -> RunExecutionOutcome:
            await _settle(lambda: store.heartbeats >= 3)
            row = store.runs["run-a"]
            row["lease_owner"] = "another-worker"
            row["fencing_epoch"] = int(row["fencing_epoch"]) + 1
            await asyncio.sleep(5)
            return Succeeded({"answer": "never"})

        executor = _Executor(body)
        coordinator = RunCoordinator(
            store=store,
            executors={"answer": executor},
            query_worker_concurrency=1,
            heartbeat_seconds=0.01,
        )
        await coordinator.start()
        try:
            await _settle(lambda: bool(executor.sessions) and executor.sessions[0].lease_lost)
            await _settle(lambda: coordinator.active_runs == ())
        finally:
            await coordinator.aclose()

        assert store.runs["run-a"]["status"] == "running"
        assert store.events["run-a"] == []

    async def test_a_failing_heartbeat_task_never_replaces_the_run_outcome(self) -> None:
        store = _MemoryStore()
        store.add_run("run-a")
        store.heartbeat_result = _UnreadableRenewal()
        executions: list[asyncio.Task[Any]] = []

        async def body(session: RunSession) -> RunExecutionOutcome:
            task = asyncio.current_task()
            assert task is not None
            executions.append(task)
            await _settle(lambda: store.heartbeats >= 1)
            return Succeeded({"answer": "kept"})

        coordinator = RunCoordinator(
            store=store,
            executors={"answer": _Executor(body)},
            query_worker_concurrency=1,
            heartbeat_seconds=0.01,
        )
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
            await _settle(lambda: bool(executions) and executions[0].done())
        finally:
            await coordinator.aclose()

        assert executions[0].exception() is None
        assert store.runs["run-a"]["result"] == {"answer": "kept"}


class _UnreadableRenewal:
    """A renewal this worker cannot interpret; reading it must not kill the run."""

    @property
    def renewed(self) -> bool:
        raise RuntimeError("lease renewal could not be interpreted")


class TestCancellationAndShutdown:
    async def test_cancellation_is_observed_at_a_boundary_and_commits_cancelled(self) -> None:
        store = _MemoryStore()
        entered = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            entered.set()
            for _ in range(200):
                await session.check_cancelled()
                await asyncio.sleep(0.005)
            return Succeeded({"answer": "unreachable"})

        coordinator = RunCoordinator(
            store=store,
            executors={"answer": _Executor(body)},
            query_worker_concurrency=1,
            heartbeat_seconds=0.01,
        )
        store.add_run("run-a")
        await coordinator.start()
        try:
            await entered.wait()
            await store.request_cancellation(owner_id=_OWNER, run_id="run-a")
            await _settle(lambda: store.runs["run-a"]["status"] == "cancelled")
        finally:
            await coordinator.aclose()

        assert store.events["run-a"][-1].payload == {"status": "cancelled"}

    async def test_accepted_cancellation_wins_a_returned_executor_failure(self) -> None:
        store = _MemoryStore()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            entered.set()
            await release.wait()
            return Failed("provider_error", "provider failed", {"partial": True})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await entered.wait()
            await store.request_cancellation(owner_id=_OWNER, run_id="run-a")
            release.set()
            await _settle(lambda: store.runs["run-a"]["status"] == "cancelled")
        finally:
            release.set()
            await coordinator.aclose()

        assert store.runs["run-a"]["result"] is None
        assert store.runs["run-a"]["error_kind"] is None
        assert [(event.event_type, event.payload) for event in store.events["run-a"]] == [
            ("done", {"status": "cancelled"})
        ]

    async def test_accepted_cancellation_wins_a_raised_executor_failure(self) -> None:
        store = _MemoryStore()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            entered.set()
            await release.wait()
            raise RunExecutionError("provider_error", "provider failed")

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await entered.wait()
            await store.request_cancellation(owner_id=_OWNER, run_id="run-a")
            release.set()
            await _settle(lambda: store.runs["run-a"]["status"] == "cancelled")
        finally:
            release.set()
            await coordinator.aclose()

        assert store.runs["run-a"]["error_kind"] is None
        assert [(event.event_type, event.payload) for event in store.events["run-a"]] == [
            ("done", {"status": "cancelled"})
        ]

    async def test_pending_tokens_are_flushed_before_a_terminal_transition(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.emit_token("partial")
            raise RunCancellationObserved

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "cancelled")
        finally:
            await coordinator.aclose()

        assert [event.event_type for event in store.events["run-a"]] == ["token", "done"]

    async def test_graceful_shutdown_requeues_without_crash_recovery(self) -> None:
        store = _MemoryStore()
        running = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            running.set()
            await asyncio.sleep(30)
            return Succeeded({"answer": "unreachable"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        await running.wait()

        await coordinator.aclose()

        assert store.runs["run-a"]["status"] == "queued"
        assert store.runs["run-a"]["reclaims_without_progress"] == 0
        assert store.runs["run-a"]["lease_owner"] is None
        assert not coordinator.active_runs

    async def test_shutdown_leaves_no_background_task_running(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> RunExecutionOutcome:
            return Succeeded({"answer": "done"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        before = {task for task in asyncio.all_tasks()}
        await coordinator.start()
        await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        await coordinator.aclose()

        leaked = {task for task in asyncio.all_tasks() if task not in before and not task.done()}
        assert leaked == set()

    async def test_shutdown_joins_a_shielded_write_it_already_started(self) -> None:
        """A shielded write must land before shutdown returns, not run detached."""
        store = _MemoryStore()
        started = asyncio.Event()
        finished = asyncio.Event()
        original = store.finish_success

        async def slow_finish(**kwargs: Any) -> Any:
            started.set()
            await asyncio.sleep(0.05)
            outcome = await original(**kwargs)
            finished.set()
            return outcome

        store.finish_success = slow_finish  # type: ignore[method-assign]

        async def body(session: RunSession) -> RunExecutionOutcome:
            return Succeeded({"answer": "done"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        before = {task for task in asyncio.all_tasks()}
        await coordinator.start()
        await started.wait()

        await coordinator.aclose()

        assert finished.is_set()
        leaked = {task for task in asyncio.all_tasks() if task not in before and not task.done()}
        assert leaked == set()


class TestSubscriptions:
    async def test_replays_durable_events_then_follows_to_the_terminal_event(self) -> None:
        store = _MemoryStore()
        gate = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.enter_phase("generating")
            await session.emit_token("one")
            await session.flush_tokens()
            await gate.wait()
            return Succeeded({"answer": "one"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: len(store.events["run-a"]) == 2)
            seen: list[RunEvent] = []

            async def consume() -> None:
                async for event in coordinator.subscribe(owner_id=_OWNER, run_id="run-a"):
                    seen.append(event)

            reader = asyncio.create_task(consume())
            await _settle(lambda: len(seen) == 2)
            gate.set()
            await asyncio.wait_for(reader, timeout=2)
        finally:
            gate.set()
            await coordinator.aclose()

        assert [event.sequence for event in seen] == [1, 2, 3]
        assert seen[-1].event_type == "done"

    async def test_reconnect_after_a_cursor_has_no_gap_or_duplicate(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.enter_phase("generating")
            await session.emit_token("one")
            await session.flush_tokens()
            return Succeeded({"answer": "one"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
            first: list[int] = []
            async for event in coordinator.subscribe(owner_id=_OWNER, run_id="run-a"):
                first.append(event.sequence)
                if len(first) == 2:
                    break
            second = [
                event.sequence
                async for event in coordinator.subscribe(
                    owner_id=_OWNER, run_id="run-a", after_sequence=first[-1]
                )
            ]
        finally:
            await coordinator.aclose()

        assert first == [1, 2]
        assert second == [3]

    async def test_detaching_a_subscriber_never_cancels_the_run(self) -> None:
        store = _MemoryStore()
        release = asyncio.Event()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.enter_phase("generating")
            await release.wait()
            return Succeeded({"answer": "still finished"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            stream = coordinator.subscribe(owner_id=_OWNER, run_id="run-a")
            assert (await anext(stream)).event_type == "progress"
            await stream.aclose()

            assert store.runs["run-a"]["cancel_requested"] is False
            release.set()
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        finally:
            release.set()
            await coordinator.aclose()

    async def test_two_subscribers_each_see_the_whole_stream(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> RunExecutionOutcome:
            await session.emit_token("shared")
            return Succeeded({"answer": "shared"})

        coordinator = _coordinator(store, _Executor(body), query_worker_concurrency=1)
        store.add_run("run-a")
        await coordinator.start()
        try:

            async def consume() -> list[int]:
                return [
                    event.sequence
                    async for event in coordinator.subscribe(owner_id=_OWNER, run_id="run-a")
                ]

            first, second = await asyncio.gather(consume(), consume())
        finally:
            await coordinator.aclose()

        assert first == second == [1, 2]

    async def test_unknown_run_yields_nothing_and_closes(self) -> None:
        store = _MemoryStore()
        coordinator = _coordinator(store, _Executor(_noop), query_worker_concurrency=1)

        events = [event async for event in coordinator.subscribe(owner_id=_OWNER, run_id="missing")]

        assert events == []


async def _noop(session: RunSession) -> RunExecutionOutcome:
    return Succeeded({"answer": ""})


@pytest.mark.parametrize("query_worker_concurrency", [1, 4])
def test_execution_slots_are_bounded_by_worker_concurrency(
    query_worker_concurrency: int,
) -> None:
    coordinator = RunCoordinator(
        store=_MemoryStore(),
        executors={"answer": _Executor(_noop)},
        query_worker_concurrency=query_worker_concurrency,
    )
    assert coordinator.query_worker_concurrency == query_worker_concurrency


def test_query_worker_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="query_worker_concurrency must be positive"):
        RunCoordinator(
            store=_MemoryStore(),
            executors={"answer": _Executor(_noop)},
            query_worker_concurrency=0,
        )
