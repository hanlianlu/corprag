# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Opt-in deterministic 10k-client RunRuntime control-plane evidence."""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import platform
import resource
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from dlightrag.engine.runtime import (
    MAX_RECLAIMS_WITHOUT_PROGRESS,
    RunCapacityExceededError,
    RunCoordinator,
)
from tests.integration.run_runtime_pg_harness import (
    TrackingExecutor,
    isolated_run_runtime,
    run_envelope,
)
from tests.load.runtime_workload import latency_summary, mutation_action, query_submission

pytestmark = [pytest.mark.load_runtime, pytest.mark.asyncio]

_QUERY_SUBMISSIONS = 10_000
_MUTATION_SUBMISSIONS = 1_000
_SEED_QUERY = 200
_SEED_MUTATION = _MUTATION_SUBMISSIONS
_QUERY_ACTIVE = 16
_MUTATION_ACTIVE = 2
_QUERY_FUSE = 30_000
_MUTATION_FUSE = 1_000
_RESULTS = Path(".test-results/load-runtime")


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


async def _explain(connection: Any, sql: str, *args: Any) -> str:
    rows = await connection.fetch(f"EXPLAIN (ANALYZE, BUFFERS, COSTS OFF, TIMING OFF) {sql}", *args)
    return "\n".join(str(row["QUERY PLAN"]) for row in rows) + "\n"


async def _capture_backlog_plans(pool: Any) -> dict[str, str]:
    async with pool.acquire() as connection:
        await connection.execute("ANALYZE dlightrag_runs")
        claim = await _explain(
            connection,
            "SELECT r.owner_id, r.run_id FROM dlightrag_runs r "
            "WHERE r.run_kind = ANY($2::text[]) AND r.lane = ANY($3::text[]) "
            "AND r.cancel_requested_at IS NULL "
            "AND (r.next_attempt_at IS NULL OR r.next_attempt_at <= NOW()) "
            "AND (r.status = 'queued' OR "
            "(r.status = 'running' AND r.lease_expires_at < NOW() "
            "AND r.reclaims_without_progress < $1)) "
            "ORDER BY r.created_at, r.run_id LIMIT 1",
            MAX_RECLAIMS_WITHOUT_PROGRESS,
            ["answer", "retrieval"],
            ["query"],
        )
        fuse = await _explain(
            connection,
            "SELECT COUNT(*) FROM dlightrag_runs "
            "WHERE lane = $1 AND status IN ('queued', 'running')",
            "corpus_mutation",
        )
        mutation_fifo = await _explain(
            connection,
            "SELECT r.owner_id, r.run_id FROM dlightrag_runs r "
            "WHERE r.run_kind = 'corpus_mutation' AND r.lane = 'corpus_mutation' "
            "AND r.cancel_requested_at IS NULL "
            "AND (r.next_attempt_at IS NULL OR r.next_attempt_at <= NOW()) "
            "AND r.status = 'queued' AND NOT EXISTS ("
            "SELECT 1 FROM dlightrag_runs earlier "
            "WHERE earlier.lane = 'corpus_mutation' "
            "AND earlier.owner_id = r.owner_id "
            "AND earlier.status IN ('queued', 'running') "
            "AND (earlier.created_at, earlier.run_id) < (r.created_at, r.run_id)) "
            "ORDER BY r.created_at, r.run_id LIMIT 1",
        )
    return {"claim": claim, "fuse": fuse, "mutation-fifo": mutation_fifo}


async def _monitor_loop(stop: asyncio.Event, metrics: dict[str, Any], pool: Any) -> None:
    loop = asyncio.get_running_loop()
    interval = 0.02
    target = loop.time() + interval
    activity_counter = 0
    while not stop.is_set():
        await asyncio.sleep(interval)
        now = loop.time()
        metrics["max_event_loop_lag_ms"] = max(
            metrics["max_event_loop_lag_ms"], max(0.0, now - target) * 1000
        )
        target = now + interval
        activity_counter += 1
        if activity_counter % 10 == 0:
            async with pool.acquire() as connection:
                active = int(
                    await connection.fetchval(
                        "SELECT COUNT(*) FROM pg_stat_activity WHERE datname = current_database()"
                    )
                    or 0
                )
            metrics["max_pg_connections"] = max(metrics["max_pg_connections"], active)


def _operation_started() -> float:
    return time.perf_counter()


def _operation_elapsed(started: float) -> float:
    return time.perf_counter() - started


async def test_10k_addressable_clients_keep_the_control_plane_bounded() -> None:
    accept_latencies: list[float] = []
    status_latencies: list[float] = []
    cancel_latencies: list[float] = []
    cursor_latencies: list[float] = []
    query_ids: list[str] = []
    mutation_ids_by_workspace: dict[str, list[str]] = {}
    cancel_outcomes: dict[str, int] = {}
    workload_started = time.perf_counter()
    rss_idle = _rss_bytes()
    metrics: dict[str, Any] = {
        "max_event_loop_lag_ms": 0.0,
        "max_pg_connections": 0,
    }
    plans: dict[str, str] = {}

    async with isolated_run_runtime(
        "load_runtime",
        query_max_active_runs=_QUERY_ACTIVE,
        query_max_nonterminal_runs=_QUERY_FUSE,
        corpus_mutation_max_active_runs=_MUTATION_ACTIVE,
        corpus_mutation_max_nonterminal_runs=_MUTATION_FUSE,
        pool_max_size=32,
    ) as (store, pool):
        # Hold fake expensive work long enough to exercise both accepted active
        # ceilings while control-plane requests continue against the backlog.
        query_executor = TrackingExecutor(delay_seconds=0.1)
        mutation_executor = TrackingExecutor(delay_seconds=0.02)
        stop_monitor = asyncio.Event()
        monitor = asyncio.create_task(_monitor_loop(stop_monitor, metrics, pool))

        async def submit_query(index: int) -> str:
            shape = query_submission(index)
            envelope = run_envelope(
                shape.kind,
                key=f"client-{index:05d}",
                mode=shape.mode or "fast",
                workspace_count=shape.workspace_count,
            )
            started = _operation_started()
            creation = await store.accept_run(envelope=envelope, run_id=str(uuid.uuid7()))
            accept_latencies.append(_operation_elapsed(started))
            assert creation.replayed is False
            query_ids.append(creation.run.run_id)
            return creation.run.run_id

        async def submit_mutation(index: int) -> str:
            workspace = f"workspace-{index % 100:03d}"
            started = _operation_started()
            creation = await store.accept_run(
                envelope=run_envelope(
                    "corpus_mutation",
                    key=f"mutation-{index:04d}",
                    workspace=workspace,
                    action=mutation_action(index),
                ),
                run_id=str(uuid.uuid7()),
            )
            accept_latencies.append(_operation_elapsed(started))
            mutation_ids_by_workspace.setdefault(workspace, []).append(creation.run.run_id)
            return creation.run.run_id

        # Seed a reproducible backlog before workers start. This is large enough
        # to exercise plans and queues without misrepresenting the 30k fuse as a target.
        for index in range(_SEED_QUERY):
            run_id = await submit_query(index)
            if index % 20 == 0:
                started = _operation_started()
                outcome = await store.request_cancellation(owner_id="load-owner", run_id=run_id)
                cancel_latencies.append(_operation_elapsed(started))
                cancel_outcomes[outcome.outcome] = cancel_outcomes.get(outcome.outcome, 0) + 1
                assert outcome.outcome == "cancelled"
        for index in range(_SEED_MUTATION):
            await submit_mutation(index)
        async with pool.acquire() as connection:
            mutation_rows_before_rejection = int(
                await connection.fetchval(
                    "SELECT COUNT(*) FROM dlightrag_runs WHERE lane = 'corpus_mutation'"
                )
                or 0
            )
        with pytest.raises(RunCapacityExceededError):
            await store.accept_run(
                envelope=run_envelope(
                    "corpus_mutation",
                    key="mutation-over-fuse",
                    workspace="workspace-000",
                ),
                run_id=str(uuid.uuid7()),
            )
        async with pool.acquire() as connection:
            mutation_rows_after_rejection = int(
                await connection.fetchval(
                    "SELECT COUNT(*) FROM dlightrag_runs WHERE lane = 'corpus_mutation'"
                )
                or 0
            )
        mutation_fuse_rejected_before_insert = (
            mutation_rows_before_rejection == mutation_rows_after_rejection == _MUTATION_FUSE
        )
        plans = await _capture_backlog_plans(pool)

        coordinator = RunCoordinator(
            store=store,
            executors={
                "answer": query_executor,
                "retrieval": query_executor,
                "corpus_mutation": mutation_executor,
            },
            query_worker_concurrency=_QUERY_ACTIVE,
            corpus_mutation_worker_concurrency=_MUTATION_ACTIVE,
            sweep_seconds=0.02,
        )
        await coordinator.start()
        coordinator.wake()
        drain_started = 0.0
        try:
            next_mutation = _SEED_MUTATION
            for index in range(_SEED_QUERY, _QUERY_SUBMISSIONS):
                run_id = await submit_query(index)
                if index % 25 == 0:
                    started = _operation_started()
                    record = await store.get_run(owner_id="load-owner", run_id=run_id)
                    status_latencies.append(_operation_elapsed(started))
                    assert record is not None
                    assert await store.get_run(owner_id="unauthorized", run_id=run_id) is None
                if index % 500 == 0:
                    started = _operation_started()
                    outcome = await store.request_cancellation(owner_id="load-owner", run_id=run_id)
                    cancel_latencies.append(_operation_elapsed(started))
                    cancel_outcomes[outcome.outcome] = cancel_outcomes.get(outcome.outcome, 0) + 1
                    assert outcome.outcome in {
                        "cancelled",
                        "pending",
                        "already_terminal",
                    }
                if index % 50 == 0 and next_mutation < _MUTATION_SUBMISSIONS:
                    await submit_mutation(next_mutation)
                    next_mutation += 1
                if index % 100 == 0:
                    coordinator.wake()
            while next_mutation < _MUTATION_SUBMISSIONS:
                await submit_mutation(next_mutation)
                next_mutation += 1
            coordinator.wake()
            drain_started = time.perf_counter()
            async with asyncio.timeout(300):
                while True:
                    async with pool.acquire() as connection:
                        nonterminal = int(
                            await connection.fetchval(
                                "SELECT COUNT(*) FROM dlightrag_runs "
                                "WHERE status IN ('queued', 'running')"
                            )
                            or 0
                        )
                    if nonterminal == 0:
                        break
                    await asyncio.sleep(0.05)
            drain_seconds = time.perf_counter() - drain_started
        finally:
            await coordinator.aclose()
            stop_monitor.set()
            await monitor

        # Status and cursor reconnect are sampled after drain as well as during load.
        for run_id in query_ids[::50]:
            started = _operation_started()
            record = await store.get_run(owner_id="load-owner", run_id=run_id)
            status_latencies.append(_operation_elapsed(started))
            assert record is not None and record.terminal
            started = _operation_started()
            first_page = await store.read_event_page(
                owner_id="load-owner", run_id=run_id, after_sequence=0
            )
            cursor_latencies.append(_operation_elapsed(started))
            assert first_page
            cursor = first_page[0].sequence
            started = _operation_started()
            resumed = await store.read_event_page(
                owner_id="load-owner", run_id=run_id, after_sequence=cursor
            )
            cursor_latencies.append(_operation_elapsed(started))
            assert all(event.sequence > cursor for event in resumed)

        event_run_id = query_ids[-1]
        async with pool.acquire() as connection:
            plans["events"] = await _explain(
                connection,
                "SELECT event_sequence, event_type, payload, created_at "
                "FROM dlightrag_run_events "
                "WHERE owner_id = $1 AND run_id = $2 AND event_sequence > $3 "
                "ORDER BY event_sequence LIMIT 500",
                "load-owner",
                uuid.UUID(event_run_id),
                0,
            )
            await connection.execute("ANALYZE dlightrag_runs")
            totals = await connection.fetchrow(
                "SELECT COUNT(*) AS total, "
                "COUNT(*) FILTER (WHERE status IN ('queued', 'running')) AS nonterminal, "
                "COUNT(*) FILTER (WHERE status = 'cancelled') AS cancelled "
                "FROM dlightrag_runs"
            )
            duplicate_terminal = int(
                await connection.fetchval(
                    "SELECT COUNT(*) FROM ("
                    "SELECT run_id FROM dlightrag_run_events "
                    "WHERE event_type IN ('done', 'error') "
                    "GROUP BY run_id HAVING COUNT(*) > 1) duplicates"
                )
                or 0
            )
            queue_percentiles = await connection.fetchrow(
                "SELECT "
                "percentile_cont(0.50) WITHIN GROUP "
                "(ORDER BY EXTRACT(EPOCH FROM started_at - created_at)) AS p50, "
                "percentile_cont(0.95) WITHIN GROUP "
                "(ORDER BY EXTRACT(EPOCH FROM started_at - created_at)) AS p95, "
                "percentile_cont(0.99) WITHIN GROUP "
                "(ORDER BY EXTRACT(EPOCH FROM started_at - created_at)) AS p99 "
                "FROM dlightrag_runs WHERE started_at IS NOT NULL"
            )
            terminal_event_count = int(
                await connection.fetchval(
                    "SELECT COUNT(*) FROM dlightrag_run_events "
                    "WHERE event_type IN ('done', 'error')"
                )
                or 0
            )

        total_duration = time.perf_counter() - workload_started
        rss_peak = _rss_bytes()
        expected_total = _QUERY_SUBMISSIONS + _MUTATION_SUBMISSIONS
        shape_counts = {
            "retrieval": sum(
                query_submission(index).kind == "retrieval" for index in range(_QUERY_SUBMISSIONS)
            ),
            "fast_answer": sum(
                query_submission(index).mode == "fast" for index in range(_QUERY_SUBMISSIONS)
            ),
            "research_answer": sum(
                query_submission(index).mode == "research" for index in range(_QUERY_SUBMISSIONS)
            ),
        }
        mutation_counts = {
            action: sum(mutation_action(index) == action for index in range(_MUTATION_SUBMISSIONS))
            for action in ("ingest", "replace", "delete", "retry", "reset")
        }
        fifo_ok = all(
            mutation_executor.starts_by_workspace.get(workspace, []) == accepted_ids
            for workspace, accepted_ids in mutation_ids_by_workspace.items()
        )
        latency = {
            "accept": latency_summary(accept_latencies),
            "status": latency_summary(status_latencies),
            "cancel": latency_summary(cancel_latencies),
            "event_cursor": latency_summary(cursor_latencies),
        }
        hard_gates = {
            "all_accepted_rows_present": int(totals["total"]) == expected_total,
            "eventual_drain": int(totals["nonterminal"]) == 0,
            "single_terminal_event_per_run": duplicate_terminal == 0
            and terminal_event_count == expected_total,
            "query_active_ceiling_exercised": query_executor.max_active == _QUERY_ACTIVE,
            "mutation_active_ceiling_exercised": mutation_executor.max_active == _MUTATION_ACTIVE,
            "workspace_fifo": fifo_ok,
            "mutation_fuse_rejected_before_insert": mutation_fuse_rejected_before_insert,
            "retrieval_workspace_shapes": query_executor.retrieval_workspace_counts
            >= {1, 10, 50, 100},
            "duration_broad_ceiling": total_duration < 900,
            "drain_broad_ceiling": drain_seconds < 300,
            "control_plane_broad_ceiling": all(
                float(operation["p99_ms"]) < 30_000 for operation in latency.values()
            ),
            "event_loop_broad_ceiling": metrics["max_event_loop_lag_ms"] < 5_000,
            "memory_broad_ceiling": rss_peak - rss_idle < 2 * 1024**3,
        }
        summary = {
            "schema_version": 1,
            "captured_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "command": "make load-runtime",
            "environment": {
                "os": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "postgres": str(await pool.fetchval("SELECT current_setting('server_version')")),
                "cpu_count": os.cpu_count(),
                "memory_note": "ru_maxrss process high-water mark; platform-normalized to bytes",
            },
            "traffic": {
                "logical_clients": _QUERY_SUBMISSIONS,
                "query_submissions": _QUERY_SUBMISSIONS,
                "mutation_submissions": _MUTATION_SUBMISSIONS,
                "total_run_rows": expected_total,
                "query_mix": shape_counts,
                "mutation_mix": mutation_counts,
                "workspaces": 100,
                "retrieval_workspace_counts": [1, 10, 50, 100],
                "providers": "deterministic in-process fakes; no network or paid calls",
            },
            "capacity": {
                "query_worker_concurrency": _QUERY_ACTIVE,
                "query_max_active_runs": _QUERY_ACTIVE,
                "query_max_nonterminal_runs": _QUERY_FUSE,
                "corpus_mutation_worker_concurrency": _MUTATION_ACTIVE,
                "corpus_mutation_max_active_runs": _MUTATION_ACTIVE,
                "corpus_mutation_max_nonterminal_runs": _MUTATION_FUSE,
            },
            "results": {
                "passed": all(hard_gates.values()),
                "duration_seconds": round(total_duration, 3),
                "throughput_submissions_per_second": round(expected_total / total_duration, 3),
                "drain_seconds_after_offer_stopped": round(drain_seconds, 3),
                "latency": latency,
                "queue_residence_seconds": {
                    key: round(float(queue_percentiles[key] or 0), 6)
                    for key in ("p50", "p95", "p99")
                },
                "rss_idle_bytes": rss_idle,
                "rss_peak_bytes": rss_peak,
                "rss_growth_bytes": max(0, rss_peak - rss_idle),
                "max_event_loop_lag_ms": round(metrics["max_event_loop_lag_ms"], 3),
                "max_pg_connections": metrics["max_pg_connections"],
                "query_executor_max_in_flight": query_executor.max_active,
                "mutation_executor_max_in_flight": mutation_executor.max_active,
                "cancelled_runs": int(totals["cancelled"]),
                "cancel_outcomes": cancel_outcomes,
                "duplicate_terminal_runs": duplicate_terminal,
                "accepted_runs_lost": expected_total - int(totals["total"]),
            },
            "hard_gates": hard_gates,
            "thresholds": {
                "duration_seconds": 900,
                "drain_seconds": 300,
                "control_operation_p99_ms": 30_000,
                "event_loop_lag_ms": 5_000,
                "rss_growth_bytes": 2 * 1024**3,
                "note": "Broad survival ceilings are regression tripwires, not production SLOs.",
            },
            "limitations": [
                "One local process and one fresh PostgreSQL database; no infrastructure capacity claim.",
                "Addressable users are deterministic client submissions under one trusted organization, not 10,000 simultaneous expensive operations.",
                "Fake executors validate the control plane, lane bounds, FIFO, and drain; they do not benchmark LightRAG, parsers, providers, networking, or storage provisioning.",
                "Queue residence is measured and is not a pass/fail SLO.",
            ],
        }
        _RESULTS.mkdir(parents=True, exist_ok=True)
        (_RESULTS / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for name, plan in plans.items():
            (_RESULTS / f"explain-{name}.txt").write_text(plan, encoding="utf-8")
        (_RESULTS / "invariants.log").write_text(
            "\n".join(
                f"{name}={'PASS' if passed else 'FAIL'}" for name, passed in hard_gates.items()
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "RUN_RUNTIME_LOAD "
            f"{'PASS' if summary['results']['passed'] else 'FAIL'} "
            f"runs={expected_total} duration={total_duration:.3f}s "
            f"throughput={expected_total / total_duration:.3f}/s "
            f"drain={drain_seconds:.3f}s"
        )
        assert all(hard_gates.values()), {
            name: passed for name, passed in hard_gates.items() if not passed
        }
