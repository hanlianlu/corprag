# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Controlled Corpus Mutation phase faults on the shared PostgreSQL RunRuntime."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import asyncpg
import pytest

from dlightrag.application.corpus_admin.mutations import CorpusMutationExecutor
from dlightrag.engine.dependencies import ParserUnavailableError, TransientDependencyError
from dlightrag.engine.runtime import RunCoordinator
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


@pytest.fixture(autouse=True)
async def _require_postgres() -> None:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")


class _Maintenance:
    @asynccontextmanager
    async def workspace_write_gate(self, _workspace: str):
        yield


class _WorkspacePool:
    def __init__(self, runtimes: dict[str, Any]) -> None:
        self.runtimes = runtimes
        self.evict = AsyncMock()

    async def acquire(self, workspace: str) -> Any:
        return self.runtimes[workspace]


def _projection_store() -> Any:
    return SimpleNamespace(record_corpus_window=AsyncMock(return_value=True))


def _runtime(*, tracked: dict[str, Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        lightrag=SimpleNamespace(
            aget_docs_by_track_id=AsyncMock(return_value=tracked or {}),
            apipeline_process_enqueue_documents=AsyncMock(),
        ),
        aingest=AsyncMock(
            return_value={
                "processed": 1,
                "errors": [],
                "results": [{"doc_id": "doc-1", "chunks": ["chunk-1"]}],
            }
        ),
        adelete_files=AsyncMock(
            return_value=[
                {
                    "identifier": "doc-1",
                    "status": "deleted",
                    "matched_doc_ids": ["doc-1"],
                }
            ]
        ),
        aretryable_document_ids=AsyncMock(return_value=("doc-1",)),
        aretry_failed_docs=AsyncMock(
            return_value={
                "retried": 1,
                "succeeded": 1,
                "failed": 0,
                "succeeded_docs": [{"doc_id": "doc-1"}],
                "failed_docs": [],
            }
        ),
        areset=AsyncMock(return_value={"documents_deleted": 1, "errors": []}),
    )


async def _wait_for(store: Any, owner: str, run_id: str, predicate: Any) -> Any:
    async with asyncio.timeout(10):
        while True:
            record = await store.get_run(owner_id=owner, run_id=run_id)
            if record is not None and predicate(record):
                return record
            await asyncio.sleep(0.02)


async def test_ingest_owned_phase_faults_defer_without_leaking_capacity_or_diagnostics() -> None:
    # These are the public WorkspaceRag legs that aggregate enqueue/parser,
    # LightRAG processing, and all required DlightRAG finalizers. Exact
    # finalizer ordering and marker safety are covered by the engine unit gate.
    faults = {
        "phase-enqueue": ParserUnavailableError(),
        "phase-processing": TransientDependencyError(
            "corpus_storage", "LightRAG processing temporarily unavailable /private/path"
        ),
        "phase-visual": TransientDependencyError(
            "providers", "visual finalizer temporarily unavailable token=private"
        ),
        "phase-bm25": TransientDependencyError(
            "corpus_storage", "BM25 finalizer temporarily unavailable"
        ),
        "phase-metadata-source": TransientDependencyError(
            "corpus_storage", "metadata/source locator temporarily unavailable"
        ),
        "phase-readiness": TransientDependencyError(
            "corpus_storage", "readiness promotion temporarily unavailable"
        ),
    }
    runtimes: dict[str, Any] = {}
    async with isolated_run_runtime("mutation_phases") as (store, _pool):
        accepted = []
        for index, (phase, fault) in enumerate(faults.items()):
            workspace = f"phase_{index}"
            runtime = _runtime()
            runtime.aingest.side_effect = fault
            runtimes[workspace] = runtime
            envelope = run_envelope(
                "corpus_mutation",
                key=phase,
                workspace=workspace,
                action="ingest",
            )
            creation = await store.accept_run(envelope=envelope, run_id=str(uuid.uuid7()))
            accepted.append((workspace, creation.run.run_id, envelope.payload["track_id"], phase))

        executor = CorpusMutationExecutor(
            pool=cast(Any, _WorkspacePool(runtimes)),
            maintenance=cast(Any, _Maintenance()),
            store=cast(Any, _projection_store()),
        )
        coordinator = RunCoordinator(
            store=store,
            executors={"corpus_mutation": executor},
            query_worker_concurrency=1,
            corpus_mutation_worker_concurrency=2,
            sweep_seconds=0.02,
        )
        await coordinator.start()
        coordinator.wake()
        try:
            rows = await asyncio.gather(
                *(
                    _wait_for(
                        store,
                        workspace,
                        run_id,
                        lambda row: row.status == "queued" and row.phase == "deferred",
                    )
                    for workspace, run_id, _track, _phase in accepted
                )
            )
        finally:
            await coordinator.aclose()

        for row, (workspace, _run_id, track_id, _phase) in zip(rows, accepted, strict=True):
            assert row.active_permit is False and row.lease_owner is None
            assert row.handoff_started_at is not None
            assert row.next_attempt_at is not None
            assert row.checkpoint is not None
            assert row.checkpoint["track_id"] == track_id
            assert "/private/path" not in str(row.checkpoint)
            assert "token=private" not in str(row.checkpoint)
            assert await store.get_run(owner_id="other", run_id=row.run_id) is None
            assert runtimes[workspace].aingest.await_count == 1


async def test_transient_ingest_recovers_the_same_track_with_exponential_checkpoint() -> None:
    runtime = _runtime()
    runtime.aingest.side_effect = [
        ParserUnavailableError(),
        {
            "processed": 1,
            "errors": [],
            "results": [{"doc_id": "doc-recovered", "chunks": []}],
        },
    ]
    pool = _WorkspacePool({"recovery": runtime})
    async with isolated_run_runtime("mutation_recovery") as (store, database_pool):
        envelope = run_envelope(
            "corpus_mutation",
            key="transient-recovery",
            workspace="recovery",
            action="ingest",
        )
        accepted = await store.accept_run(envelope=envelope, run_id=str(uuid.uuid7()))
        executor = CorpusMutationExecutor(
            pool=cast(Any, pool),
            maintenance=cast(Any, _Maintenance()),
            store=cast(Any, _projection_store()),
        )
        coordinator = RunCoordinator(
            store=store,
            executors={"corpus_mutation": executor},
            query_worker_concurrency=1,
            corpus_mutation_worker_concurrency=1,
            sweep_seconds=0.02,
        )
        await coordinator.start()
        coordinator.wake()
        try:
            deferred = await _wait_for(
                store,
                "recovery",
                accepted.run.run_id,
                lambda row: row.status == "queued" and row.phase == "deferred",
            )
            assert deferred.checkpoint is not None
            assert deferred.checkpoint["parser_unavailable_attempt"] == 1
            async with database_pool.acquire() as connection:
                await connection.execute(
                    "UPDATE dlightrag_runs SET next_attempt_at = NOW() - INTERVAL '1 second' "
                    "WHERE run_id = $1",
                    uuid.UUID(accepted.run.run_id),
                )
            coordinator.wake()
            final = await _wait_for(
                store,
                "recovery",
                accepted.run.run_id,
                lambda row: row.terminal,
            )
        finally:
            await coordinator.aclose()

        assert final.status == "succeeded"
        assert runtime.aingest.await_count == 2
        assert {call.kwargs["_track_id"] for call in runtime.aingest.await_args_list} == {
            envelope.payload["track_id"]
        }
        events = await store.read_event_page(owner_id="recovery", run_id=accepted.run.run_id)
        assert [event.sequence for event in events] == list(range(1, len(events) + 1))
        assert sum(event.event_type in {"done", "error"} for event in events) == 1


@pytest.mark.parametrize("action", ["replace", "delete", "retry", "reset"])
async def test_ambiguous_destructive_phase_waits_for_repair_and_keeps_fifo(
    action: str,
) -> None:
    workspace = f"ambiguous_{action}"
    runtime = _runtime()
    secret_fault = RuntimeError("ambiguous upstream failure at /private/file token=secret")
    if action == "replace":
        runtime.aingest.side_effect = secret_fault
    elif action == "delete":
        runtime.adelete_files.side_effect = [
            [{"identifier": "doc-1", "matched_doc_ids": ["doc-1"]}],
            secret_fault,
        ]
    elif action == "retry":
        runtime.aretry_failed_docs.side_effect = secret_fault
    else:
        runtime.areset.side_effect = secret_fault

    async with isolated_run_runtime(f"mutation_{action}") as (store, _pool):
        accepted = await store.accept_run(
            envelope=run_envelope(
                "corpus_mutation", key=f"fault-{action}", workspace=workspace, action=action
            ),
            run_id=str(uuid.uuid7()),
        )
        executor = CorpusMutationExecutor(
            pool=cast(Any, _WorkspacePool({workspace: runtime})),
            maintenance=cast(Any, _Maintenance()),
            store=cast(Any, _projection_store()),
        )
        coordinator = RunCoordinator(
            store=store,
            executors={"corpus_mutation": executor},
            query_worker_concurrency=1,
            corpus_mutation_worker_concurrency=1,
            sweep_seconds=0.02,
        )
        await coordinator.start()
        coordinator.wake()
        try:
            waiting = await _wait_for(
                store,
                workspace,
                accepted.run.run_id,
                lambda row: row.phase == "waiting_for_repair",
            )
        finally:
            await coordinator.aclose()

        assert waiting.status == "running"
        assert waiting.active_permit is False and waiting.lease_owner is None
        assert waiting.handoff_started_at is not None
        assert waiting.checkpoint is not None
        assert waiting.checkpoint["repair_reason"]
        assert waiting.checkpoint["repair_remedy"]
        assert "/private/file" not in str(waiting.checkpoint)
        assert "token=secret" not in str(waiting.checkpoint)
        assert (
            await store.request_cancellation(owner_id=workspace, run_id=waiting.run_id)
        ).outcome == "rejected"

        behind = await store.accept_run(
            envelope=run_envelope("corpus_mutation", key=f"behind-{action}", workspace=workspace),
            run_id=str(uuid.uuid7()),
        )
        free = await store.accept_run(
            envelope=run_envelope(
                "corpus_mutation", key=f"free-{action}", workspace=f"free_{action}"
            ),
            run_id=str(uuid.uuid7()),
        )
        claim = await store.claim_next(
            worker_id="other-writer",
            run_kinds=("corpus_mutation",),
            lanes=("corpus_mutation",),
        )
        assert claim is not None and claim.run.run_id == free.run.run_id
        assert claim.run.run_id != behind.run.run_id


@pytest.mark.parametrize("action", ["ingest", "replace"])
async def test_public_track_reconciliation_does_not_repeat_source_admission(action: str) -> None:
    workspace = f"tracked_{action}"
    runtime = _runtime(tracked={"doc-tracked": {"status": "processed"}})
    runtime.aretry_failed_docs.return_value = {
        "retried": 1,
        "succeeded": 1,
        "failed": 0,
        "succeeded_docs": [{"doc_id": "doc-tracked"}],
        "failed_docs": [],
    }
    async with isolated_run_runtime(f"mutation_track_{action}") as (store, _pool):
        envelope = run_envelope(
            "corpus_mutation", key=f"tracked-{action}", workspace=workspace, action=action
        )
        accepted = await store.accept_run(envelope=envelope, run_id=str(uuid.uuid7()))
        crashed = await store.claim_next(
            worker_id="crashed-writer",
            run_kinds=("corpus_mutation",),
            lanes=("corpus_mutation",),
        )
        assert crashed is not None
        assert await store.start_handoff(
            owner_id=workspace,
            run_id=accepted.run.run_id,
            worker_id="crashed-writer",
            fencing_epoch=crashed.run.fencing_epoch,
            checkpoint={"phase": "handoff_started", "track_id": envelope.payload["track_id"]},
        )
        assert (
            await store.release_for_shutdown(
                owner_id=workspace,
                run_id=accepted.run.run_id,
                worker_id="crashed-writer",
                fencing_epoch=crashed.run.fencing_epoch,
            )
            == "requeued"
        )

        executor = CorpusMutationExecutor(
            pool=cast(Any, _WorkspacePool({workspace: runtime})),
            maintenance=cast(Any, _Maintenance()),
            store=cast(Any, _projection_store()),
        )
        coordinator = RunCoordinator(
            store=store,
            executors={"corpus_mutation": executor},
            query_worker_concurrency=1,
            corpus_mutation_worker_concurrency=1,
            sweep_seconds=0.02,
        )
        await coordinator.start()
        coordinator.wake()
        try:
            final = await _wait_for(store, workspace, accepted.run.run_id, lambda row: row.terminal)
        finally:
            await coordinator.aclose()

        assert final.status == "succeeded"
        runtime.aingest.assert_not_awaited()
        runtime.aretry_failed_docs.assert_awaited_once_with(
            cohort_doc_ids=("doc-tracked",), track_id=envelope.payload["track_id"]
        )


async def test_missing_staged_source_fails_terminally_without_public_path() -> None:
    workspace = "missing_source"
    runtime = _runtime()
    async with isolated_run_runtime("mutation_stage") as (store, _pool):
        envelope = run_envelope(
            "corpus_mutation", key="missing-stage", workspace=workspace, action="ingest"
        )
        payload = dict(envelope.payload)
        missing_path = str((Path("missing") / "source.pdf").resolve())
        payload["source"] = {
            "source_type": "local",
            "path": missing_path,
            "replace": False,
        }
        payload["staged_sources"] = [
            {
                "path": missing_path,
                "content_sha256": "0" * 64,
                "size_bytes": 1,
            }
        ]
        envelope = replace(envelope, payload=payload)
        accepted = await store.accept_run(envelope=envelope, run_id=str(uuid.uuid7()))
        executor = CorpusMutationExecutor(
            pool=cast(Any, _WorkspacePool({workspace: runtime})),
            maintenance=cast(Any, _Maintenance()),
            store=cast(Any, _projection_store()),
        )
        coordinator = RunCoordinator(
            store=store,
            executors={"corpus_mutation": executor},
            query_worker_concurrency=1,
            corpus_mutation_worker_concurrency=1,
            sweep_seconds=0.02,
        )
        await coordinator.start()
        coordinator.wake()
        try:
            final = await _wait_for(store, workspace, accepted.run.run_id, lambda row: row.terminal)
        finally:
            await coordinator.aclose()

        assert final.status == "failed"
        assert final.error_kind == "corpus_source_unavailable"
        assert "missing" not in str(final.result)
        assert "source.pdf" not in str(final.error_message)
        runtime.aingest.assert_not_awaited()
