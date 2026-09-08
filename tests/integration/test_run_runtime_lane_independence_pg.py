# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""P0/P1 lane isolation, fencing, ordering, and recovery on one RunRuntime."""

from __future__ import annotations

import asyncio
import datetime
import uuid
from dataclasses import replace
from typing import Any, cast

import asyncpg
import pytest

from dlightrag.engine.runtime import (
    Deferred,
    IdempotencyKeyConflict,
    RunAdmissionLimitExceededError,
    RunCoordinator,
    RunSession,
    RunStore,
    Succeeded,
    WaitingForRepair,
)
from tests.integration.pg_conn import PG_CONN_KWARGS
from tests.integration.run_runtime_pg_harness import (
    TrackingExecutor,
    isolated_run_runtime,
    run_envelope,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def _pg_available() -> bool:
    try:
        connection = await asyncpg.connect(**PG_CONN_KWARGS)
        await connection.fetchval("SELECT 1")
        await connection.close()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
async def _require_postgres() -> None:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")


class _OneShotOperationalStateOutage:
    """Inject bounded store faults without replacing PostgreSQL authority."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self.accept_failures = {
            "answer": 1,
            "retrieval": 1,
            "corpus_mutation": 1,
        }
        self.claim_failures = {"query": 1, "corpus_mutation": 1}
        self.heartbeat_failures = 3

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    async def accept_run(self, *, envelope: Any, run_id: str) -> Any:
        remaining = self.accept_failures[envelope.run_kind]
        if remaining:
            self.accept_failures[envelope.run_kind] = remaining - 1
            raise ConnectionError("controlled Operational State acceptance outage")
        return await self._store.accept_run(envelope=envelope, run_id=run_id)

    async def claim_next(self, *, worker_id: str, run_kinds: Any, lanes: Any) -> Any:
        lane = str(lanes[0])
        remaining = self.claim_failures[lane]
        if remaining:
            self.claim_failures[lane] = remaining - 1
            raise ConnectionError("controlled Operational State claim outage")
        return await self._store.claim_next(
            worker_id=worker_id,
            run_kinds=run_kinds,
            lanes=lanes,
        )

    async def heartbeat(self, **ownership: Any) -> Any:
        if self.heartbeat_failures:
            self.heartbeat_failures -= 1
            raise ConnectionError("controlled Operational State heartbeat outage")
        return await self._store.heartbeat(**ownership)


async def test_operational_state_outages_do_not_invent_acceptance_or_drop_queued_runs() -> None:
    async with isolated_run_runtime("store_outage") as (store, pool):
        faulting_store = _OneShotOperationalStateOutage(store)
        envelopes = [
            run_envelope("retrieval", key="retrieval-outage"),
            run_envelope("answer", key="answer-outage"),
            run_envelope(
                "corpus_mutation",
                key="mutation-outage",
                workspace="outage-workspace",
            ),
        ]

        for envelope in envelopes:
            with pytest.raises(ConnectionError, match="acceptance outage"):
                await faulting_store.accept_run(
                    envelope=envelope,
                    run_id=str(uuid.uuid7()),
                )
        async with pool.acquire() as connection:
            assert await connection.fetchval("SELECT COUNT(*) FROM dlightrag_runs") == 0

        accepted = [
            await faulting_store.accept_run(
                envelope=envelope,
                run_id=str(uuid.uuid7()),
            )
            for envelope in envelopes
        ]
        query_executor = TrackingExecutor(delay_seconds=0.05)
        mutation_executor = TrackingExecutor(delay_seconds=0.05)
        coordinator = RunCoordinator(
            store=cast(RunStore, faulting_store),
            executors={
                "retrieval": query_executor,
                "answer": query_executor,
                "corpus_mutation": mutation_executor,
            },
            query_worker_concurrency=2,
            corpus_mutation_worker_concurrency=1,
            heartbeat_seconds=0.005,
            sweep_seconds=0.005,
        )
        await coordinator.start()
        try:
            async with asyncio.timeout(5):
                while True:
                    rows = [
                        await store.get_run(
                            owner_id=creation.run.access_scope.scope_id,
                            run_id=creation.run.run_id,
                        )
                        for creation in accepted
                    ]
                    if all(row is not None and row.terminal for row in rows):
                        break
                    await asyncio.sleep(0.01)
        finally:
            await coordinator.aclose()

        assert faulting_store.accept_failures == {
            "answer": 0,
            "retrieval": 0,
            "corpus_mutation": 0,
        }
        assert faulting_store.claim_failures == {"query": 0, "corpus_mutation": 0}
        assert faulting_store.heartbeat_failures == 0
        assert query_executor.calls == 2
        assert mutation_executor.calls == 1


async def test_atomic_replay_conflict_authorization_and_lane_local_admission_limits() -> None:
    async with isolated_run_runtime(
        "lane_accept",
        query_max_nonterminal_runs=2,
        corpus_mutation_max_nonterminal_runs=1,
    ) as (store, pool):
        mutation = run_envelope("corpus_mutation", key="mutation-replay", workspace="alpha")
        first, replay = await asyncio.gather(
            store.accept_run(envelope=mutation, run_id=str(uuid.uuid7())),
            store.accept_run(envelope=mutation, run_id=str(uuid.uuid7())),
        )
        assert {first.replayed, replay.replayed} == {False, True}
        assert first.run.run_id == replay.run.run_id

        with pytest.raises(IdempotencyKeyConflict):
            await store.accept_run(
                envelope=replace(mutation, request_fingerprint="changed-input"),
                run_id=str(uuid.uuid7()),
            )

        # A reached mutation admission limit neither blocks Query admission nor leaks state.
        query = await store.accept_run(
            envelope=run_envelope("retrieval", key="query-accepted"),
            run_id=str(uuid.uuid7()),
        )
        assert query.run.lane == "query"
        with pytest.raises(RunAdmissionLimitExceededError):
            await store.accept_run(
                envelope=run_envelope("corpus_mutation", key="mutation-rejected", workspace="beta"),
                run_id=str(uuid.uuid7()),
            )
        assert await store.get_run(owner_id="not-alpha", run_id=first.run.run_id) is None
        assert await store.read_event_page(owner_id="not-alpha", run_id=first.run.run_id) == ()
        assert (
            await store.request_cancellation(owner_id="not-alpha", run_id=first.run.run_id)
        ).outcome == "unknown"
        async with pool.acquire() as connection:
            assert (
                await connection.fetchval(
                    "SELECT COUNT(*) FROM dlightrag_runs WHERE lane = 'corpus_mutation'"
                )
                == 1
            )


async def test_concurrent_claimers_add_slots_without_double_claiming_or_breaking_fifo() -> None:
    async with isolated_run_runtime("lane_claim") as (store, _pool):
        first_alpha = await store.accept_run(
            envelope=run_envelope("corpus_mutation", key="alpha-1", workspace="alpha"),
            run_id=str(uuid.uuid7()),
        )
        second_alpha = await store.accept_run(
            envelope=run_envelope("corpus_mutation", key="alpha-2", workspace="alpha"),
            run_id=str(uuid.uuid7()),
        )
        beta = await store.accept_run(
            envelope=run_envelope("corpus_mutation", key="beta-1", workspace="beta"),
            run_id=str(uuid.uuid7()),
        )
        gamma = await store.accept_run(
            envelope=run_envelope("corpus_mutation", key="gamma-1", workspace="gamma"),
            run_id=str(uuid.uuid7()),
        )
        query = await store.accept_run(
            envelope=run_envelope("retrieval", key="query-1"), run_id=str(uuid.uuid7())
        )

        claims = await asyncio.gather(
            *(
                store.claim_next(
                    worker_id=f"writer-{index}",
                    run_kinds=("corpus_mutation",),
                    lanes=("corpus_mutation",),
                )
                for index in range(6)
            )
        )
        winners = [claim for claim in claims if claim is not None]
        winner_ids = [claim.run.run_id for claim in winners]
        assert len(winners) == 3
        assert len(set(winner_ids)) == len(winner_ids)
        assert set(winner_ids) == {
            first_alpha.run.run_id,
            beta.run.run_id,
            gamma.run.run_id,
        }
        assert second_alpha.run.run_id not in winner_ids
        assert (
            await store.claim_next(
                worker_id="blocked-successor",
                run_kinds=("corpus_mutation",),
                lanes=("corpus_mutation",),
            )
            is None
        )

        # Query claims independently while three Mutation writers own distinct Workspaces.
        query_claim = await store.claim_next(
            worker_id="query-worker", run_kinds=("retrieval",), lanes=("query",)
        )
        assert query_claim is not None and query_claim.run.run_id == query.run.run_id

        alpha_claim = next(claim for claim in winners if claim.run.run_id == first_alpha.run.run_id)
        settled = await store.finish_success(
            owner_id="alpha",
            run_id=first_alpha.run.run_id,
            worker_id=str(alpha_claim.run.lease_owner),
            fencing_epoch=alpha_claim.run.fencing_epoch,
            result={"ok": True},
        )
        assert settled.committed
        next_alpha = await store.claim_next(
            worker_id="writer-next",
            run_kinds=("corpus_mutation",),
            lanes=("corpus_mutation",),
        )
        assert next_alpha is not None and next_alpha.run.run_id == second_alpha.run.run_id


class _AdditiveSlotExecutor:
    def __init__(self, target: int) -> None:
        self._target = target
        self.started_ids: list[str] = []
        self.active = 0
        self.max_active = 0
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, session: RunSession) -> Succeeded:
        self.started_ids.append(session.run_id)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active == self._target:
            self.all_started.set()
        try:
            await self.release.wait()
            return Succeeded({"ok": True})
        finally:
            self.active -= 1


async def test_multiple_coordinators_contribute_additive_local_slots() -> None:
    async with isolated_run_runtime("lane_coordinators") as (store, _pool):
        accepted = [
            await store.accept_run(
                envelope=run_envelope("retrieval", key=f"query-{index}"),
                run_id=str(uuid.uuid7()),
            )
            for index in range(4)
        ]
        executor = _AdditiveSlotExecutor(target=4)
        coordinators = [
            RunCoordinator(
                store=store,
                executors={"retrieval": executor},
                query_worker_concurrency=2,
                corpus_mutation_worker_concurrency=1,
                sweep_seconds=0.02,
            )
            for _ in range(2)
        ]
        try:
            await asyncio.gather(*(coordinator.start() for coordinator in coordinators))
            for coordinator in coordinators:
                coordinator.wake()
            await asyncio.wait_for(executor.all_started.wait(), timeout=5)
            assert executor.max_active == 4
            assert len(executor.started_ids) == len(set(executor.started_ids)) == 4
            assert set(executor.started_ids) == {creation.run.run_id for creation in accepted}
            executor.release.set()
            async with asyncio.timeout(5):
                while True:
                    rows = [
                        await store.get_run(owner_id="load-owner", run_id=creation.run.run_id)
                        for creation in accepted
                    ]
                    if all(row is not None and row.terminal for row in rows):
                        break
                    await asyncio.sleep(0.01)
        finally:
            executor.release.set()
            await asyncio.gather(*(coordinator.aclose() for coordinator in coordinators))


class _YieldingMutationExecutor:
    def __init__(self, outcome: Deferred | WaitingForRepair) -> None:
        self._outcome = outcome
        self.starts: list[str] = []
        self.active = 0
        self.max_active = 0
        self.beta_started = asyncio.Event()

    async def execute(self, session: RunSession) -> Deferred | WaitingForRepair | Succeeded:
        workspace = str((session.prepared_input or {}).get("workspace"))
        self.starts.append(workspace)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if workspace == "alpha":
                return self._outcome
            self.beta_started.set()
            return Succeeded({"ok": True})
        finally:
            self.active -= 1


@pytest.mark.parametrize("outcome_kind", ["deferred", "waiting_for_repair"])
async def test_defer_and_repair_release_the_local_slot(outcome_kind: str) -> None:
    async with isolated_run_runtime(f"lane_yield_{outcome_kind}") as (store, _pool):
        if outcome_kind == "deferred":
            outcome: Deferred | WaitingForRepair = Deferred(
                checkpoint={"phase": "deferred_dependency"},
                next_attempt_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
            )
        else:
            outcome = WaitingForRepair(
                checkpoint={
                    "repair_reason": "Controlled uncertain outcome.",
                    "repair_remedy": "Inspect and resume.",
                }
            )
        executor = _YieldingMutationExecutor(outcome)
        first = await store.accept_run(
            envelope=run_envelope("corpus_mutation", key="alpha-yield", workspace="alpha"),
            run_id=str(uuid.uuid7()),
        )
        second = await store.accept_run(
            envelope=run_envelope("corpus_mutation", key="beta-after-yield", workspace="beta"),
            run_id=str(uuid.uuid7()),
        )
        coordinator = RunCoordinator(
            store=store,
            executors={"corpus_mutation": executor},
            query_worker_concurrency=1,
            corpus_mutation_worker_concurrency=1,
            sweep_seconds=0.02,
        )
        await coordinator.start()
        try:
            await asyncio.wait_for(executor.beta_started.wait(), timeout=5)
            async with asyncio.timeout(5):
                while True:
                    second_row = await store.get_run(owner_id="beta", run_id=second.run.run_id)
                    if second_row is not None and second_row.terminal:
                        break
                    await asyncio.sleep(0.01)
            first_row = await store.get_run(owner_id="alpha", run_id=first.run.run_id)
            assert first_row is not None and first_row.lease_owner is None
            assert first_row.phase == outcome_kind
            assert executor.starts == ["alpha", "beta"]
            assert executor.max_active == 1
        finally:
            await coordinator.aclose()


async def test_defer_releases_lease_preserves_barrier_then_recovers_same_run() -> None:
    async with isolated_run_runtime("lane_defer") as (store, pool):
        first = await store.accept_run(
            envelope=run_envelope("corpus_mutation", key="alpha-defer", workspace="alpha"),
            run_id=str(uuid.uuid7()),
        )
        second = await store.accept_run(
            envelope=run_envelope("corpus_mutation", key="alpha-behind", workspace="alpha"),
            run_id=str(uuid.uuid7()),
        )
        other = await store.accept_run(
            envelope=run_envelope("corpus_mutation", key="beta-free", workspace="beta"),
            run_id=str(uuid.uuid7()),
        )
        claim = await store.claim_next(
            worker_id="writer-1",
            run_kinds=("corpus_mutation",),
            lanes=("corpus_mutation",),
        )
        assert claim is not None and claim.run.run_id == first.run.run_id
        assert await store.defer(
            owner_id="alpha",
            run_id=first.run.run_id,
            worker_id="writer-1",
            fencing_epoch=claim.run.fencing_epoch,
            checkpoint={"phase": "deferred_dependency", "corpus_unavailable_attempt": 1},
            next_attempt_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        )
        deferred = await store.get_run(owner_id="alpha", run_id=first.run.run_id)
        assert deferred is not None
        assert deferred.status == "queued" and deferred.lease_owner is None
        other_claim = await store.claim_next(
            worker_id="writer-2",
            run_kinds=("corpus_mutation",),
            lanes=("corpus_mutation",),
        )
        assert other_claim is not None and other_claim.run.run_id == other.run.run_id
        assert other_claim.run.run_id != second.run.run_id
        await store.finish_success(
            owner_id="beta",
            run_id=other.run.run_id,
            worker_id="writer-2",
            fencing_epoch=other_claim.run.fencing_epoch,
            result={"ok": True},
        )
        async with pool.acquire() as connection:
            await connection.execute(
                "UPDATE dlightrag_runs SET next_attempt_at = NOW() - INTERVAL '1 second' "
                "WHERE run_id = $1",
                uuid.UUID(first.run.run_id),
            )
        recovered = await store.claim_next(
            worker_id="writer-3",
            run_kinds=("corpus_mutation",),
            lanes=("corpus_mutation",),
        )
        assert recovered is not None and recovered.run.run_id == first.run.run_id
        assert recovered.run.fencing_epoch == claim.run.fencing_epoch + 1


class _BlockingExecutor:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def execute(self, session: RunSession) -> Succeeded:
        del session
        self.started.set()
        await asyncio.Event().wait()
        return Succeeded({"unreachable": True})


async def test_graceful_shutdown_requeues_and_crash_reclaim_fences_post_handoff_worker() -> None:
    async with isolated_run_runtime("lane_shutdown") as (store, pool):
        blocking = _BlockingExecutor()
        coordinator = RunCoordinator(
            store=store,
            executors={"corpus_mutation": blocking},
            query_worker_concurrency=1,
            corpus_mutation_worker_concurrency=1,
            sweep_seconds=0.05,
        )
        graceful = await store.accept_run(
            envelope=run_envelope("corpus_mutation", key="graceful", workspace="alpha"),
            run_id=str(uuid.uuid7()),
        )
        await coordinator.start()
        coordinator.wake()
        await asyncio.wait_for(blocking.started.wait(), timeout=5)
        await coordinator.aclose()
        released = await store.get_run(owner_id="alpha", run_id=graceful.run.run_id)
        assert released is not None
        assert released.status == "queued" and released.lease_owner is None

        crashed = await store.claim_next(
            worker_id="crashed-writer",
            run_kinds=("corpus_mutation",),
            lanes=("corpus_mutation",),
        )
        assert crashed is not None and crashed.run.run_id == graceful.run.run_id
        assert await store.start_handoff(
            owner_id="alpha",
            run_id=graceful.run.run_id,
            worker_id="crashed-writer",
            fencing_epoch=crashed.run.fencing_epoch,
            checkpoint={"phase": "handoff_started", "track_id": "stable-track"},
        )
        assert (
            await store.request_cancellation(owner_id="alpha", run_id=graceful.run.run_id)
        ).outcome == "rejected"
        async with pool.acquire() as connection:
            await connection.execute(
                "UPDATE dlightrag_runs SET lease_expires_at = NOW() - INTERVAL '1 second' "
                "WHERE run_id = $1",
                uuid.UUID(graceful.run.run_id),
            )
        fresh = await store.claim_next(
            worker_id="recovery-writer",
            run_kinds=("corpus_mutation",),
            lanes=("corpus_mutation",),
        )
        assert fresh is not None and fresh.run.run_id == graceful.run.run_id
        assert fresh.run.handoff_started_at is not None
        assert fresh.run.checkpoint == {"phase": "handoff_started", "track_id": "stable-track"}
        assert not await store.write_checkpoint(
            owner_id="alpha",
            run_id=graceful.run.run_id,
            worker_id="crashed-writer",
            fencing_epoch=crashed.run.fencing_epoch,
            checkpoint={"unsafe": True},
        )
        stale_terminal = await store.finish_success(
            owner_id="alpha",
            run_id=graceful.run.run_id,
            worker_id="crashed-writer",
            fencing_epoch=crashed.run.fencing_epoch,
            result={"unsafe": True},
        )
        assert not stale_terminal.committed
