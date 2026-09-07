# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The local owner of accepted durable run execution.

One coordinator per process schedules, executes, and finalizes durable runs. It
reserves a local execution slot *before* it claims a row, so a worker never
holds a lease while waiting for capacity, and every durable write it makes is
predicated on its own lease owner and fencing epoch. Lease duration, heartbeat
cadence, sweep cadence, and token coalescing are fixed internal constants; the
public worker bound is ``runtime.query.worker_concurrency``. AI provider calls
and RAG pipeline work have independent admission owners.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Protocol, assert_never

from dlightrag.engine.runtime.contracts import RunKind, RunLane, RunPhase
from dlightrag.engine.runtime.errors import RunExecutionError
from dlightrag.engine.runtime.records import (
    AlreadyCommittedTerminal,
    ClaimedRun,
    Deferred,
    Failed,
    PendingPublication,
    RunEvent,
    RunExecutionOutcome,
    Succeeded,
    WaitingForRepair,
)
from dlightrag.engine.runtime.store import RunStore
from dlightrag.engine.runtime.subscription import RunEventBroker, follow_run_events

logger = logging.getLogger(__name__)

#: How often an idle worker rechecks the queue for another host's work.
SWEEP_SECONDS = 1.0
#: Cadence for renewing an owned run's storage lease.
RUN_HEARTBEAT_SECONDS = 20.0
#: How often each process runs the retention-floor event trim and run/artifact prune.
#: Every pass is bounded, ``SKIP LOCKED``, and idempotent, so running it on every
#: run-owning process needs no leader election and exposes no operator knob.
MAINTENANCE_SECONDS = 3600.0
#: Share of one cadence a process may defer its first retention pass by, so a
#: fleet that restarts together does not trim on the same instant.
_MAINTENANCE_JITTER_FRACTION = 0.1
#: Pause between drained retention batches, so a large backlog stays background
#: work rather than a burst against the pool.
_MAINTENANCE_BATCH_PAUSE_SECONDS = 0.05
#: How long a graceful shutdown waits for writes that were already in flight.
SHUTDOWN_WRITE_GRACE_SECONDS = 5.0
#: Coalescing bounds for durable token batches: a batch is committed as soon
#: as it reaches the character bound or, via the run's scheduled flush, no
#: later than the wall-clock bound after its first token even if the provider
#: stalls.
TOKEN_BATCH_CHARS = 512
TOKEN_BATCH_SECONDS = 0.25

_RUN_EXECUTION_FAILED = "run_execution_failed"
_RUN_EXECUTION_FAILED_MESSAGE = "Run execution failed."


def _startup_jitter(cadence: float) -> float:
    return random.uniform(0.0, max(0.0, cadence) * _MAINTENANCE_JITTER_FRACTION)  # noqa: S311


class RunCancellationObserved(Exception):
    """The run's owner requested cancellation and the worker observed it."""


class LeaseLostError(Exception):
    """This worker no longer owns the run and must persist nothing further."""


class DurableWrites:
    """Keeps every shielded durable write joinable across a shutdown.

    ``asyncio.shield`` stops a cancelled worker from tearing a small fenced write
    in half, but on its own it leaves that write running as an unreferenced task:
    a graceful shutdown could return before the requeue or terminal transition it
    already started ever reached PostgreSQL. Registering each one lets ``aclose``
    join them within the shutdown grace.
    """

    def __init__(self) -> None:
        self._writes: set[asyncio.Task[Any]] = set()

    def shield[T](self, operation: Awaitable[T]) -> Awaitable[T]:
        task = asyncio.ensure_future(operation)
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)
        return asyncio.shield(task)

    async def drain(self, timeout: float) -> None:
        """Wait out the writes already in flight; never start new ones."""
        deadline = asyncio.get_running_loop().time() + timeout
        while self._writes:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.warning("Shutdown left %d durable Run writes in flight", len(self._writes))
                return
            await asyncio.wait(tuple(self._writes), timeout=remaining)


class RunExecutor(Protocol):
    """Executes one claimed run or raises an owner-classified failure."""

    async def execute(self, session: RunSession) -> RunExecutionOutcome: ...


class RunSession:
    """One claimed run's fenced view of its durable state.

    Every write is predicated on this worker's lease owner and fencing epoch and
    is shielded, so a shutdown cancellation never tears a small fenced write in
    half. The first zero-row write means the lease is gone: the session latches
    closed so no later event, settlement, or terminal transition can be written
    by a worker the run no longer belongs to. Checkpoint and artifact methods
    are gone: Session settlements and acceptance carry those facts.

    Token durability: the first buffered token arms one wall-clock flush timer,
    so a stalled provider's text still lands within ``TOKEN_BATCH_SECONDS``.
    ``self._lane`` is the one serialization lane for token-buffer mutation,
    token writes, and control writes; the timer and executor race only through
    it, which keeps every batch whole and ordered before any control or terminal
    event. A timer failure is latched and re-raised at the
    next session boundary; ``aclose`` joins or cancels the timer and is called
    on every coordinator exit path.
    """

    def __init__(
        self,
        store: RunStore,
        claimed: ClaimedRun,
        *,
        broker: RunEventBroker,
        writes: DurableWrites,
        notify: Callable[[], None] | None = None,
        _token_flush_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        execution = claimed.execution
        run = claimed.run
        self.owner_id = execution.owner_id
        self.run_id = execution.run_id
        self.worker_id = execution.worker_id
        self.fencing_epoch = execution.fencing_epoch
        self.durable_progress_version = run.durable_progress_version
        self.execution = execution
        self.prepared_input: Mapping[str, Any] | None = run.prepared_input
        self.checkpoint: Mapping[str, Any] | None = run.checkpoint
        self.workspace_epoch: int | None = run.agent_workspace_epoch
        self._store = store
        self._broker = broker
        self._writes = writes
        self._notify = notify
        self._token_flush_sleep = _token_flush_sleep
        self._cancel_requested = run.cancel_requested
        self._handoff_started = run.handoff_started_at is not None
        self._lease_lost = False
        self._sealed = False
        self._lane = asyncio.Lock()
        self._pending_tokens: list[str] = []
        self._pending_chars = 0
        self._flush_deadline: float | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._background_failure: Exception | None = None
        # A run that already committed an event has a partial draft somewhere;
        # regenerated output must clear it before the first new token.
        self._reset_pending = run.next_event_sequence > 1
        self.pending_publications: list[PendingPublication] = []

    # -- state ---------------------------------------------------------
    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    @property
    def lease_lost(self) -> bool:
        return self._lease_lost

    @property
    def handoff_started(self) -> bool:
        return self._handoff_started

    def observe_cancellation(self) -> None:
        self._cancel_requested = True

    def observe_lease_loss(self) -> None:
        self._lease_lost = True

    async def check_cancelled(self) -> None:
        """Raise at a control boundary once the owner asked to cancel."""
        self._guard()
        if self._cancel_requested:
            raise RunCancellationObserved

    def _guard(self) -> None:
        """Refuse every further durable write once this session lost coherence."""
        if self._background_failure is not None:
            raise self._background_failure
        if self._lease_lost:
            raise LeaseLostError
        if self._sealed:
            raise RuntimeError("run session is closed")

    # -- durable writes -------------------------------------------------
    async def checkpoint_state(
        self, checkpoint: Mapping[str, object], *, phase: RunPhase | None = None
    ) -> None:
        """Replace the executor-owned bounded checkpoint under this run's fence."""
        async with self._lane:
            self._guard()
            await self._flush_locked()
            committed = await self._writes.shield(
                self._store.write_checkpoint(
                    owner_id=self.owner_id,
                    run_id=self.run_id,
                    worker_id=self.worker_id,
                    fencing_epoch=self.fencing_epoch,
                    checkpoint=checkpoint,
                    phase=phase,
                )
            )
            if not committed:
                self._lease_lost = True
                raise LeaseLostError

    async def begin_handoff(self, checkpoint: Mapping[str, object]) -> None:
        """Atomically cross the point after which cancellation is rejected."""
        async with self._lane:
            self._guard()
            await self._flush_locked()
            committed = await self._writes.shield(
                self._store.start_handoff(
                    owner_id=self.owner_id,
                    run_id=self.run_id,
                    worker_id=self.worker_id,
                    fencing_epoch=self.fencing_epoch,
                    checkpoint=checkpoint,
                )
            )
            if not committed:
                # Cancellation and handoff are a single-row race. A cancellation
                # that won is observed as cancellation, not as an ambiguous lease loss.
                if self._cancel_requested:
                    raise RunCancellationObserved
                self._lease_lost = True
                raise LeaseLostError
            self._handoff_started = True

    async def enter_phase(self, phase: RunPhase) -> None:
        async with self._lane:
            self._guard()
            await self._flush_locked()
            await self._fenced(
                self._store.record_phase(
                    owner_id=self.owner_id,
                    run_id=self.run_id,
                    worker_id=self.worker_id,
                    fencing_epoch=self.fencing_epoch,
                    phase=phase,
                )
            )

    async def emit_token(self, text: str) -> None:
        """Buffer text until its fixed age or size bound is reached."""
        if not text:
            return
        async with self._lane:
            # The timer can latch a failure while the executor is idle.  Check
            # before mutating the buffer so text is never accepted after that.
            self._guard()
            self._pending_tokens.append(text)
            self._pending_chars += len(text)
            now = asyncio.get_running_loop().time()
            if self._flush_deadline is None:
                self._flush_deadline = now + TOKEN_BATCH_SECONDS
                self._flush_task = asyncio.create_task(
                    self._flush_timer(self._flush_deadline),
                    name=f"run-token-flush:{self.run_id}",
                )
            if self._pending_chars >= TOKEN_BATCH_CHARS or now >= self._flush_deadline:
                await self._flush_locked()
        await self.check_cancelled()

    async def reset_output(self) -> None:
        """Clear the current streamed draft before a linked corrective operation."""
        async with self._lane:
            self._guard()
            await self._flush_locked()
            await self._fenced(
                self._store.append_event(
                    owner_id=self.owner_id,
                    run_id=self.run_id,
                    worker_id=self.worker_id,
                    fencing_epoch=self.fencing_epoch,
                    phase=None,
                    event_type="reset",
                    payload={},
                )
            )
            self._reset_pending = False

    async def emit_tool_event(
        self,
        event_type: str,
        payload: Mapping[str, object],
    ) -> None:
        """Commit one metadata-only tool lifecycle event for SSE subscribers."""
        async with self._lane:
            self._guard()
            await self._flush_locked()
            await self._fenced(
                self._store.append_event(
                    owner_id=self.owner_id,
                    run_id=self.run_id,
                    worker_id=self.worker_id,
                    fencing_epoch=self.fencing_epoch,
                    phase=None,
                    event_type=event_type,
                    payload=payload,
                )
            )

    async def flush_tokens(self) -> None:
        """Commit any buffered text; a reset clears the previous draft first."""
        async with self._lane:
            # Re-check after joining an in-flight scheduled flush: its failure
            # is latched while this coroutine waits on the lane.
            self._guard()
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        """Commit the buffered batch as one reset+token pair; caller holds ``_lane``."""
        if not self._pending_tokens:
            return
        text = "".join(self._pending_tokens)
        await self._disarm_timer_locked()
        try:
            if self._reset_pending:
                await self._fenced(
                    self._store.append_event(
                        owner_id=self.owner_id,
                        run_id=self.run_id,
                        worker_id=self.worker_id,
                        fencing_epoch=self.fencing_epoch,
                        phase=None,
                        event_type="reset",
                        payload={},
                    )
                )
                self._reset_pending = False
            await self._fenced(
                self._store.append_event(
                    owner_id=self.owner_id,
                    run_id=self.run_id,
                    worker_id=self.worker_id,
                    fencing_epoch=self.fencing_epoch,
                    phase=None,
                    event_type="token",
                    payload={"text": text},
                )
            )
        except Exception as exc:
            # The write's commit state is unknown after a storage failure.  Keep
            # the text for diagnosis and latch the failure rather than retrying
            # it (which could duplicate an append) or silently dropping it.
            self._background_failure = exc
            raise
        self._pending_tokens.clear()
        self._pending_chars = 0
        self._flush_deadline = None

    async def _flush_timer(self, deadline: float) -> None:
        """Flush one batch at its first-token deadline without a later API call."""
        try:
            delay = max(0.0, deadline - asyncio.get_running_loop().time())
            await self._token_flush_sleep(delay)
            async with self._lane:
                self._guard()
                if self._flush_deadline == deadline and self._pending_tokens:
                    await self._flush_locked()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Retrieve failures in this owned task and surface them at the next
            # executor/coordinator boundary; success is then impossible.
            self._background_failure = exc
        finally:
            if self._flush_task is asyncio.current_task():
                self._flush_task = None

    async def _disarm_timer_locked(self) -> None:
        """Cancel and reap the sleeping/lock-waiting timer while holding the lane."""
        task = self._flush_task
        if task is None or task is asyncio.current_task():
            return
        self._flush_task = None
        task.cancel()
        # ``return_exceptions`` converts only the child timer's cancellation
        # into a result. Cancellation of this caller still cancels ``gather``
        # and propagates, so shutdown cannot be mistaken for timer cleanup.
        result = (await asyncio.gather(task, return_exceptions=True))[0]
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            raise result

    async def _seal(self, *, flush_tokens: bool) -> None:
        """Disarm the timer and prevent future writes before terminal/release."""
        failure: Exception | None = None
        async with self._lane:
            if self._sealed:
                return
            await self._disarm_timer_locked()
            if flush_tokens:
                try:
                    self._guard()
                    await self._flush_locked()
                except Exception as exc:
                    failure = exc
            else:
                self._pending_tokens.clear()
                self._pending_chars = 0
                self._flush_deadline = None
            self._sealed = True
        if failure is not None:
            raise failure

    async def aclose(self) -> None:
        """Disarm this session without flushing; idempotent and join-safe."""
        await self._seal(flush_tokens=False)

    async def _fenced(self, operation: Awaitable[int | None]) -> None:
        self._guard()
        sequence = await self._writes.shield(operation)
        if sequence is None:
            self._lease_lost = True
            raise LeaseLostError
        self._broker.notify(self.owner_id, self.run_id)
        if self._notify is not None:
            self._notify()


class RunCoordinator:
    """Schedule, execute, and finalize this process's durable runs."""

    def __init__(
        self,
        *,
        store: RunStore,
        executors: Mapping[RunKind, RunExecutor],
        query_worker_concurrency: int,
        corpus_mutation_worker_concurrency: int = 1,
        worker_id: str | None = None,
        heartbeat_seconds: float = RUN_HEARTBEAT_SECONDS,
        sweep_seconds: float = SWEEP_SECONDS,
        maintenance_seconds: float = MAINTENANCE_SECONDS,
        _token_flush_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if query_worker_concurrency < 1:
            raise ValueError("query_worker_concurrency must be positive")
        if corpus_mutation_worker_concurrency < 1:
            raise ValueError("corpus_mutation_worker_concurrency must be positive")
        if not executors:
            raise ValueError("at least one run executor must be registered")
        self._store = store
        self._executors = dict(executors)
        self._run_kinds = tuple(self._executors)
        self._lane_kinds: dict[RunLane, tuple[RunKind, ...]] = {
            "query": tuple(kind for kind in self._run_kinds if kind != "corpus_mutation"),
            "corpus_mutation": tuple(kind for kind in self._run_kinds if kind == "corpus_mutation"),
        }
        self._lanes: tuple[RunLane, ...] = tuple(
            lane for lane, kinds in self._lane_kinds.items() if kinds
        )
        self._query_worker_concurrency = int(query_worker_concurrency)
        self._corpus_mutation_worker_concurrency = int(corpus_mutation_worker_concurrency)
        self._worker_id = worker_id or f"run-worker-{uuid.uuid4().hex}"
        self._heartbeat_seconds = heartbeat_seconds
        self._sweep_seconds = sweep_seconds
        self._maintenance_seconds = maintenance_seconds
        # Private deterministic test seam; production always uses asyncio.sleep
        # with the fixed TOKEN_BATCH_SECONDS bound.
        self._token_flush_sleep = _token_flush_sleep
        self._slots_by_lane: dict[RunLane, asyncio.Semaphore] = {
            "query": asyncio.Semaphore(self._query_worker_concurrency),
            "corpus_mutation": asyncio.Semaphore(self._corpus_mutation_worker_concurrency),
        }
        # Kept as the Query-lane test/introspection seam; scheduling uses the
        # lane-indexed pools above.
        self._slots = self._slots_by_lane["query"]
        self._broker = RunEventBroker()
        self._writes = DurableWrites()
        self._wake = asyncio.Event()
        self._acceptance_lock = asyncio.Lock()
        self._closing = False
        self._scheduler: asyncio.Task[None] | None = None
        self._mutation_scheduler: asyncio.Task[None] | None = None
        self._sweeper: asyncio.Task[None] | None = None
        self._maintainer: asyncio.Task[None] | None = None
        self._runs: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, RunSession] = {}

    @property
    def query_worker_concurrency(self) -> int:
        return self._query_worker_concurrency

    @property
    def corpus_mutation_worker_concurrency(self) -> int:
        return self._corpus_mutation_worker_concurrency

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def active_runs(self) -> tuple[str, ...]:
        return tuple(self._runs)

    @property
    def is_started(self) -> bool:
        """Whether this process can currently execute newly accepted runs."""
        tasks = [self._sweeper, self._maintainer]
        if self._lane_kinds["query"]:
            tasks.append(self._scheduler)
        if self._lane_kinds["corpus_mutation"]:
            tasks.append(self._mutation_scheduler)
        return not self._closing and all(task is not None and not task.done() for task in tasks)

    @contextlib.asynccontextmanager
    async def admission(self) -> AsyncIterator[bool]:
        """Keep shutdown from crossing one short durable acceptance write."""
        async with self._acceptance_lock:
            yield self.is_started

    async def start(self) -> None:
        """Begin claiming accepted runs and sweeping abandoned ones."""
        async with self._acceptance_lock:
            if self._scheduler is not None:
                return
            self._closing = False
            if self._lane_kinds["query"]:
                self._scheduler = asyncio.create_task(self._schedule_forever("query"))
            if self._lane_kinds["corpus_mutation"]:
                self._mutation_scheduler = asyncio.create_task(
                    self._schedule_forever("corpus_mutation")
                )
            self._sweeper = asyncio.create_task(self._sweep_forever())
            self._maintainer = asyncio.create_task(self._maintain_forever())

    def cancel_local(self, owner_id: str, run_id: str) -> None:
        """Signal a locally leased run's task; the listener re-read authority first.

        Cancelling the task interrupts the executor at its next control
        boundary; its shielded writes settle and the coordinator commits the
        single cancelled terminal transition.
        """
        task = self._runs.get(run_id)
        session = self._sessions.get(run_id)
        if (
            task is not None
            and not task.done()
            and not (session is not None and session.handoff_started)
        ):
            task.cancel()

    def wake(self) -> None:
        """Nudge this process after it accepted a run; polling remains the truth."""
        self._wake.set()

    async def aclose(self) -> None:
        """Stop claiming, let fenced writes settle, then requeue owned work."""
        async with self._acceptance_lock:
            self._closing = True
            self._wake.set()
            for task in (
                self._scheduler,
                self._mutation_scheduler,
                self._sweeper,
                self._maintainer,
            ):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            self._scheduler = None
            self._mutation_scheduler = None
            self._sweeper = None
            self._maintainer = None
            running = list(self._runs.values())
            for task in running:
                task.cancel()
            for task in running:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await self._writes.drain(SHUTDOWN_WRITE_GRACE_SECONDS)
            self._runs.clear()
            self._sessions.clear()

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncGenerator[RunEvent]:
        """Follow one run's durable events; detaching never mutates the run."""

        async def _is_finished() -> bool:
            run = await self._store.get_run(owner_id=owner_id, run_id=run_id)
            return run is None or run.terminal

        return follow_run_events(
            self._store,
            self._broker,
            owner_id=owner_id,
            run_id=run_id,
            after_sequence=after_sequence,
            is_finished=_is_finished,
        )

    # -- scheduling -----------------------------------------------------
    async def _schedule_forever(self, lane: RunLane = "query") -> None:
        slots = self._slots_by_lane[lane]
        run_kinds = self._lane_kinds[lane]
        while not self._closing:
            await slots.acquire()
            if self._closing:
                slots.release()
                return
            claimed: ClaimedRun | None = None
            try:
                claimed = await self._store.claim_next(
                    worker_id=self._worker_id,
                    run_kinds=run_kinds,
                    lanes=(lane,),
                )
            except Exception:
                logger.warning("%s Run claim failed", lane, exc_info=True)
            if claimed is None:
                slots.release()
                await self._idle()
                continue
            run_id = claimed.run.run_id
            task = asyncio.create_task(self._execute(claimed))
            self._runs[run_id] = task
            task.add_done_callback(
                lambda _task, key=run_id, owned_slots=slots: self._forget(key, owned_slots)
            )

    async def _idle(self) -> None:
        self._wake.clear()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._wake.wait(), timeout=self._sweep_seconds)

    def _forget(self, run_id: str, slots: asyncio.Semaphore | None = None) -> None:
        self._runs.pop(run_id, None)
        self._sessions.pop(run_id, None)
        (slots or self._slots).release()
        self._wake.set()

    async def _sweep_forever(self) -> None:
        """Finalize abandoned and cancel-pending rows without holding a slot."""
        while True:
            try:
                await self._store.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Run sweep failed", exc_info=True)
            await asyncio.sleep(self._sweep_seconds)

    async def _maintain_forever(self) -> None:
        """Apply the configured retention floor without reserving an execution slot.

        The first pass is deferred by a bounded share of the cadence so a fleet
        that restarts together spreads its trims out instead of aligning them.
        """
        await asyncio.sleep(_startup_jitter(self._maintenance_seconds))
        while True:
            try:
                await self._maintain_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Run retention failed", exc_info=True)
            await asyncio.sleep(self._maintenance_seconds)

    async def _maintain_once(self) -> None:
        """Drain both retention passes.

        Each pass is a bounded batch that strictly shrinks its own candidate set,
        so draining is finite; pausing between batches keeps a large backlog from
        monopolizing the pool, and a transient fault defers only the remainder.
        """
        while await self._store.trim_expired_event_logs() > 0:
            await asyncio.sleep(_MAINTENANCE_BATCH_PAUSE_SECONDS)
        while (await self._store.prune_expired_runs()).runs > 0:
            await asyncio.sleep(_MAINTENANCE_BATCH_PAUSE_SECONDS)

    # -- execution ------------------------------------------------------
    async def _execute(self, claimed: ClaimedRun) -> None:
        session = RunSession(
            self._store,
            claimed,
            broker=self._broker,
            writes=self._writes,
            notify=self._wake.set,
            _token_flush_sleep=self._token_flush_sleep,
        )
        self._sessions[session.run_id] = session
        heartbeat = asyncio.create_task(self._heartbeat_forever(session))
        try:
            executor = self._executors.get(claimed.run.run_kind)
            if executor is None:
                raise RunExecutionError(
                    "executor_unavailable",
                    "No executor is registered for this run kind.",
                )
            outcome = await executor.execute(session)
            if isinstance(outcome, Succeeded):
                # Sealing flushes pending text and joins/disarms its timer before
                # the terminal write, leaving one strict event order.
                await session._seal(flush_tokens=True)
                await self._finish_success(session, outcome.result)
            elif isinstance(outcome, Failed):
                await session._seal(flush_tokens=True)
                await self._finish_failure(
                    session,
                    outcome.error_kind,
                    outcome.error_message,
                    result=outcome.result,
                )
            elif isinstance(outcome, Deferred):
                await session.aclose()
                await self._store.defer(
                    owner_id=session.owner_id,
                    run_id=session.run_id,
                    worker_id=session.worker_id,
                    fencing_epoch=session.fencing_epoch,
                    checkpoint=outcome.checkpoint,
                    next_attempt_at=outcome.next_attempt_at,
                )
                self._wake.set()
            elif isinstance(outcome, WaitingForRepair):
                await session.aclose()
                await self._store.wait_for_repair(
                    owner_id=session.owner_id,
                    run_id=session.run_id,
                    worker_id=session.worker_id,
                    fencing_epoch=session.fencing_epoch,
                    checkpoint=outcome.checkpoint,
                )
                self._wake.set()
            elif isinstance(outcome, AlreadyCommittedTerminal):
                # The executor owns that atomic terminal; it must not leave a
                # local timer behind after handing control back.
                await session.aclose()
                self._broker.notify(session.owner_id, session.run_id)
                self._wake.set()
            else:
                assert_never(outcome)
        except asyncio.CancelledError:
            # Join or cancel the flush timer before the requeue so no token
            # write can land after this worker released the lease.
            await session.aclose()
            await self._release(session)
            raise
        except RunCancellationObserved:
            await self._finish_cancelled(session)
        except LeaseLostError:
            logger.info("Run %s lost its lease; leaving recovery to the next owner", session.run_id)
        except RunExecutionError as exc:
            await self._finish_failure(session, exc.kind, exc.public_message)
        except Exception:
            logger.warning(
                "Run %s failed with an unclassified executor error", session.run_id, exc_info=True
            )
            await self._finish_failure(
                session,
                _RUN_EXECUTION_FAILED,
                _RUN_EXECUTION_FAILED_MESSAGE,
            )
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            except Exception:
                # A dead heartbeat is never this run's outcome.
                logger.warning("Run %s heartbeat ended in failure", session.run_id, exc_info=True)
            await session.aclose()

    async def _heartbeat_forever(self, session: RunSession) -> None:
        """Renew an unexpired fenced lease and surface pending cancellation.

        A store that fails to answer is a transient fault, not lease loss: the
        renewal is retried on the next cadence and the run stays owned until the
        store authoritatively refuses to renew.
        """
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            try:
                renewal = await self._store.heartbeat(
                    owner_id=session.owner_id,
                    run_id=session.run_id,
                    worker_id=session.worker_id,
                    fencing_epoch=session.fencing_epoch,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Run %s heartbeat failed; retrying next cadence",
                    session.run_id,
                    exc_info=True,
                )
                continue
            if not renewal.renewed:
                session.observe_lease_loss()
                task = self._runs.get(session.run_id)
                if task is not None:
                    task.cancel()
                return
            if renewal.cancel_requested:
                session.observe_cancellation()

    async def _finish_success(self, session: RunSession, result: Mapping[str, Any]) -> None:
        if session.lease_lost:
            return
        outcome = await self._writes.shield(
            self._store.finish_success(
                owner_id=session.owner_id,
                run_id=session.run_id,
                worker_id=session.worker_id,
                fencing_epoch=session.fencing_epoch,
                result=result,
                publications=tuple(session.pending_publications),
            )
        )
        self._broker.notify(session.owner_id, session.run_id)
        if not outcome.committed:
            session.observe_lease_loss()

    async def _finish_failure(
        self,
        session: RunSession,
        kind: str,
        message: str,
        *,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        # Failure still preserves text when possible, but a failed/ambiguous
        # token append is never retried.  Either way the timer is gone before
        # the terminal transition.
        with contextlib.suppress(Exception):
            await session._seal(flush_tokens=True)
        if session.lease_lost:
            return
        with contextlib.suppress(Exception):
            await self._writes.shield(
                self._store.finish_failure(
                    owner_id=session.owner_id,
                    run_id=session.run_id,
                    worker_id=session.worker_id,
                    fencing_epoch=session.fencing_epoch,
                    error_kind=kind,
                    error_message=message,
                    **({"result": result} if result is not None else {}),
                )
            )
        self._broker.notify(session.owner_id, session.run_id)

    async def _finish_cancelled(self, session: RunSession) -> None:
        with contextlib.suppress(Exception):
            await session._seal(flush_tokens=True)
        if session.lease_lost:
            return
        await self._writes.shield(
            self._store.finish_cancelled(
                owner_id=session.owner_id,
                run_id=session.run_id,
                worker_id=session.worker_id,
                fencing_epoch=session.fencing_epoch,
            )
        )
        self._broker.notify(session.owner_id, session.run_id)

    async def _release(self, session: RunSession) -> None:
        """Requeue owned work on shutdown; this is not crash recovery."""
        if session.lease_lost:
            return
        with contextlib.suppress(Exception):
            await self._writes.shield(
                self._store.release_for_shutdown(
                    owner_id=session.owner_id,
                    run_id=session.run_id,
                    worker_id=session.worker_id,
                    fencing_epoch=session.fencing_epoch,
                )
            )
        self._broker.notify(session.owner_id, session.run_id)


__all__ = [
    "MAINTENANCE_SECONDS",
    "RUN_HEARTBEAT_SECONDS",
    "SHUTDOWN_WRITE_GRACE_SECONDS",
    "SWEEP_SECONDS",
    "TOKEN_BATCH_CHARS",
    "TOKEN_BATCH_SECONDS",
    "DurableWrites",
    "RunCoordinator",
    "RunExecutor",
    "LeaseLostError",
    "RunCancellationObserved",
    "RunSession",
]
