# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Representative RunRuntime claim, fuse, FIFO, and cursor query plans."""

from __future__ import annotations

import uuid

import asyncpg
import pytest

from dlightrag.engine.runtime import MAX_RECLAIMS_WITHOUT_PROGRESS
from tests.integration.pg_conn import PG_CONN_KWARGS
from tests.integration.run_runtime_pg_harness import isolated_run_runtime, run_envelope

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def _pg_available() -> bool:
    try:
        connection = await asyncpg.connect(**PG_CONN_KWARGS)
        await connection.fetchval("SELECT 1")
        await connection.close()
        return True
    except Exception:
        return False


async def _plan(connection, sql: str, *args) -> str:
    rows = await connection.fetch(f"EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) {sql}", *args)
    return "\n".join(str(row["QUERY PLAN"]) for row in rows)


async def test_runtime_backlog_queries_use_bounded_indexes() -> None:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")
    async with isolated_run_runtime("runtime_plans") as (store, pool):
        for index in range(40):
            await store.accept_run(
                envelope=run_envelope("retrieval", key=f"query-{index:03d}"),
                run_id=str(uuid.uuid7()),
            )
        for index in range(20):
            await store.accept_run(
                envelope=run_envelope(
                    "corpus_mutation",
                    key=f"mutation-{index:03d}",
                    workspace=f"workspace-{index % 5:03d}",
                ),
                run_id=str(uuid.uuid7()),
            )
        claimed = await store.claim_next(
            worker_id="event-writer", run_kinds=("retrieval",), lanes=("query",)
        )
        assert claimed is not None
        await store.append_event(
            owner_id=claimed.run.owner_id,
            run_id=claimed.run.run_id,
            worker_id="event-writer",
            fencing_epoch=claimed.run.fencing_epoch,
            phase="searching",
            event_type="progress",
            payload={"step": 1},
        )

        async with pool.acquire() as connection, connection.transaction():
            await connection.execute("ANALYZE dlightrag_runs")
            await connection.execute("ANALYZE dlightrag_run_events")
            await connection.execute("SET LOCAL enable_seqscan = off")
            fuse = await _plan(
                connection,
                "SELECT COUNT(*) FROM dlightrag_runs "
                "WHERE lane = $1 AND status IN ('queued', 'running')",
                "corpus_mutation",
            )
            active = await _plan(
                connection,
                "SELECT COUNT(*) FROM dlightrag_runs "
                "WHERE lane = $1 AND active_permit = TRUE "
                "AND status = 'running' AND lease_expires_at > NOW()",
                "query",
            )
            mutation_fifo = await _plan(
                connection,
                "SELECT r.owner_id, r.run_id FROM dlightrag_runs r "
                "WHERE r.run_kind = ANY($2::text[]) AND r.lane = ANY($3::text[]) "
                "AND r.cancel_requested_at IS NULL "
                "AND (r.next_attempt_at IS NULL OR r.next_attempt_at <= NOW()) "
                "AND (r.status = 'queued' OR "
                "(r.status = 'running' AND r.lease_expires_at < NOW() "
                "AND r.reclaims_without_progress < $1)) "
                "AND (r.lane <> 'corpus_mutation' OR NOT EXISTS ("
                "SELECT 1 FROM dlightrag_runs earlier "
                "WHERE earlier.lane = 'corpus_mutation' "
                "AND earlier.owner_id = r.owner_id "
                "AND earlier.status IN ('queued', 'running') "
                "AND (earlier.created_at, earlier.run_id) < (r.created_at, r.run_id))) "
                "ORDER BY r.created_at, r.run_id LIMIT 1",
                MAX_RECLAIMS_WITHOUT_PROGRESS,
                ["corpus_mutation"],
                ["corpus_mutation"],
            )
            events = await _plan(
                connection,
                "SELECT event_sequence, event_type, payload, created_at "
                "FROM dlightrag_run_events "
                "WHERE owner_id = $1 AND run_id = $2 AND event_sequence > $3 "
                "ORDER BY event_sequence LIMIT 500",
                claimed.run.owner_id,
                uuid.UUID(claimed.run.run_id),
                0,
            )

        assert "idx_dlightrag_runs_claim" in fuse
        # Active count is bounded by the accepted ceiling; it may share the
        # partial claim index rather than requiring a write-amplifying index.
        assert "idx_dlightrag_runs_claim" in active
        assert "idx_dlightrag_runs_claim" in mutation_fifo
        assert "idx_dlightrag_runs_mutation_fifo" in mutation_fifo
        assert "dlightrag_run_events_pkey" in events
