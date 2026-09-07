# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL coverage for Retrieval on the shared durable Run runtime."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

import asyncpg
import pytest

from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.application.retrieval import (
    PinnedRetrievalModel,
    RetrievalExecutor,
    RetrievalRunInput,
    canonical_retrieval_result,
)
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.rag.retrieval import RetrievalOptions, RetrievalResult
from dlightrag.engine.runtime import (
    RETRIEVAL_RUN_RETENTION_SECONDS,
    IdempotencyKeyConflict,
    PreparedRunEnvelope,
    RunAccessScope,
    RunCoordinator,
    run_request_fingerprint,
)
from tests.integration.pg_conn import PG_CONN_KWARGS

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_OWNER = "retrieval-owner"
_FINGERPRINT = ModelFingerprint("openai", "extract-model", None)
_PROFILE = ModelProfile(context_window_tokens=128_000, supports_images=True)


async def _pg_available() -> bool:
    try:
        connection = await asyncpg.connect(**PG_CONN_KWARGS)
        await connection.fetchval("SELECT 1")
        await connection.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def retrieval_pg() -> AsyncIterator[tuple[PGRunStore, Any]]:
    """Use a fresh database so no stale local schema can affect validation."""
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")
    database = f"dlightrag_retrieval_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(
        **{**PG_CONN_KWARGS, "database": database}, min_size=1, max_size=8
    )
    try:
        store = PGRunStore(pool=pool)
        await store.initialize()
        yield store, pool
    finally:
        await pool.close()
        admin = await asyncpg.connect(**PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
        finally:
            await admin.close()


def _prepared(*, query: str = "quarterly report") -> dict[str, Any]:
    return RetrievalRunInput(
        query=query,
        workspaces=("finance",),
        retrieval=RetrievalOptions(top_k=40, chunk_top_k=20),
        bm25_query=None,
        filters=None,
        query_images=(),
        pinned_models=(
            PinnedRetrievalModel(
                role="extract",
                fingerprint=_FINGERPRINT,
                profile=_PROFILE,
            ),
        ),
        context_policy_revision=CONTEXT_POLICY_REVISION,
        model_catalog_revision=current_model_catalog_revision(),
        idempotency_fingerprint=run_request_fingerprint(
            {"query": query, "workspaces": ["finance"]}
        ),
    ).as_request()


def _envelope(*, key: str, query: str = "quarterly report") -> PreparedRunEnvelope:
    prepared = _prepared(query=query)
    return PreparedRunEnvelope(
        run_kind="retrieval",
        lane="query",
        submitted_by=_OWNER,
        access_scope=RunAccessScope(kind="owner", scope_id=_OWNER),
        submission_key=key,
        request_fingerprint=prepared["idempotency_fingerprint"],
        payload=prepared,
        accepted_input={
            "query": query,
            "workspaces": ["finance"],
            "query_image_count": 0,
            "query_image_digests": [],
            "result_projection": "retrieval_v1",
            "retention_seconds": RETRIEVAL_RUN_RETENTION_SECONDS,
        },
        retention_seconds=RETRIEVAL_RUN_RETENTION_SECONDS,
    )


async def _accept(store: PGRunStore, *, key: str, query: str = "quarterly report") -> Any:
    return await store.accept_run(
        envelope=_envelope(key=key, query=query),
        run_id=str(uuid.uuid4()),
    )


async def _expire_lease(pool: Any, run_id: str) -> None:
    async with pool.acquire() as connection:
        await connection.execute(
            "UPDATE dlightrag_runs SET lease_expires_at = NOW() - INTERVAL '1 second' "
            "WHERE run_id = $1",
            uuid.UUID(run_id),
        )


async def test_retrieval_acceptance_claim_and_terminal_result_use_the_generic_store(
    retrieval_pg: tuple[PGRunStore, Any],
) -> None:
    store, pool = retrieval_pg
    first = await _accept(store, key="same-key")
    replay = await store.accept_run(envelope=_envelope(key="same-key"), run_id=str(uuid.uuid4()))

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.run.run_id == first.run.run_id
    assert first.run.run_kind == "retrieval"
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            "SELECT retention_seconds, purge_after, created_at FROM dlightrag_runs "
            "WHERE run_id = $1",
            uuid.UUID(first.run.run_id),
        )
        assert row is not None
        assert row["retention_seconds"] == 7 * 24 * 3600
        assert row["purge_after"] is None
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM dlightrag_answer_run_routing WHERE run_id = $1",
                uuid.UUID(first.run.run_id),
            )
            == 0
        )
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM dlightrag_agent_sessions WHERE lease_run_id = $1",
                uuid.UUID(first.run.run_id),
            )
            == 0
        )

    with pytest.raises(IdempotencyKeyConflict):
        await _accept(store, key="same-key", query="changed input")

    claim = await store.claim_next(
        worker_id="retrieval-worker",
        run_kinds=("retrieval",),
        lanes=("query",),
    )
    assert claim is not None
    assert claim.run.run_id == first.run.run_id
    with pytest.raises(RuntimeError, match="no Agent Session repository"):
        _ = claim.execution.session_repository
    with pytest.raises(RuntimeError, match="no Answer progress store"):
        _ = claim.execution.progress_store
    assert claim.execution.workspace_store is None

    assert (
        await store.record_phase(
            owner_id=_OWNER,
            run_id=first.run.run_id,
            worker_id="retrieval-worker",
            fencing_epoch=claim.run.fencing_epoch,
            phase="searching",
        )
        == 1
    )
    result = canonical_retrieval_result(
        RetrievalResult(
            contexts={
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "content": "revenue grew",
                        "_workspace": "finance",
                    }
                ],
                "entities": [],
                "relationships": [],
            },
            trace={"planner": "done"},
            image_descriptions=[],
        )
    )
    terminal = await store.finish_success(
        owner_id=_OWNER,
        run_id=first.run.run_id,
        worker_id="retrieval-worker",
        fencing_epoch=claim.run.fencing_epoch,
        result=result,
    )
    assert terminal.committed is True
    assert terminal.event_sequence == 2
    final = await store.get_run(owner_id=_OWNER, run_id=first.run.run_id)
    assert final is not None
    assert final.status == "succeeded"
    assert final.result == result
    assert final.prepared_input is None
    events = await store.read_event_page(owner_id=_OWNER, run_id=first.run.run_id)
    assert [(event.sequence, event.event_type) for event in events] == [
        (1, "progress"),
        (2, "done"),
    ]
    assert events[-1].payload == {"status": "succeeded", "result": result}


async def test_retrieval_reclaim_and_both_cancellation_states_use_the_common_lifecycle(
    retrieval_pg: tuple[PGRunStore, Any],
) -> None:
    store, pool = retrieval_pg
    recovering = await _accept(store, key="recover")
    first_claim = await store.claim_next(
        worker_id="crashed-worker", run_kinds=("retrieval",), lanes=("query",)
    )
    assert first_claim is not None
    await store.record_phase(
        owner_id=_OWNER,
        run_id=recovering.run.run_id,
        worker_id="crashed-worker",
        fencing_epoch=first_claim.run.fencing_epoch,
        phase="planning",
    )
    await _expire_lease(pool, recovering.run.run_id)
    reclaimed = await store.claim_next(
        worker_id="recovery-worker", run_kinds=("retrieval",), lanes=("query",)
    )
    assert reclaimed is not None
    assert reclaimed.run.run_id == recovering.run.run_id
    assert reclaimed.run.fencing_epoch == first_claim.run.fencing_epoch + 1
    assert reclaimed.run.prepared_input == _prepared()
    settled = await store.finish_success(
        owner_id=_OWNER,
        run_id=recovering.run.run_id,
        worker_id="recovery-worker",
        fencing_epoch=reclaimed.run.fencing_epoch,
        result={
            "contexts": {"chunks": [], "entities": [], "relationships": []},
            "trace": {},
            "image_descriptions": [],
        },
    )
    assert settled.committed is True

    queued = await _accept(store, key="queued-cancel")
    queued_cancel = await store.request_cancellation(owner_id=_OWNER, run_id=queued.run.run_id)
    assert queued_cancel.outcome == "cancelled"
    assert queued_cancel.run is not None and queued_cancel.run.status == "cancelled"

    running = await _accept(store, key="running-cancel")
    running_claim = await store.claim_next(
        worker_id="cancel-worker", run_kinds=("retrieval",), lanes=("query",)
    )
    assert running_claim is not None and running_claim.run.run_id == running.run.run_id
    pending = await store.request_cancellation(owner_id=_OWNER, run_id=running.run.run_id)
    assert pending.outcome == "pending"
    renewal = await store.heartbeat(
        owner_id=_OWNER,
        run_id=running.run.run_id,
        worker_id="cancel-worker",
        fencing_epoch=running_claim.run.fencing_epoch,
    )
    assert renewal.renewed is True and renewal.cancel_requested is True
    cancelled = await store.finish_cancelled(
        owner_id=_OWNER,
        run_id=running.run.run_id,
        worker_id="cancel-worker",
        fencing_epoch=running_claim.run.fencing_epoch,
    )
    assert cancelled.committed is True
    final = await store.get_run(owner_id=_OWNER, run_id=running.run.run_id)
    assert final is not None and final.status == "cancelled"
    assert [
        event.event_type
        for event in await store.read_event_page(owner_id=_OWNER, run_id=running.run.run_id)
    ] == ["done"]


class _RetrievalOperation:
    def __init__(self) -> None:
        self.calls = 0

    def warm(self, workspaces: Sequence[str]) -> None:
        assert tuple(workspaces) == ("finance",)

    async def prepare_query_images(self, images: Sequence[Mapping[str, Any]]) -> list[str]:
        assert not images
        return []

    async def retrieve_result(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.calls += 1
        assert query == "quarterly report"
        assert kwargs["workspaces"] == ("finance",)
        return RetrievalResult(
            contexts={
                "chunks": [{"chunk_id": "c1", "content": "coordinator result"}],
                "entities": [],
                "relationships": [],
            },
            trace={"executor": "retrieval"},
            image_descriptions=[],
        )


async def test_registered_retrieval_executor_runs_through_the_real_pg_coordinator(
    retrieval_pg: tuple[PGRunStore, Any],
) -> None:
    store, _pool = retrieval_pg
    operation = _RetrievalOperation()
    executor = RetrievalExecutor(
        operation=cast(Any, operation),
        timeout_seconds=30,
        model_fingerprint_for_role=lambda _role: _FINGERPRINT,
    )
    coordinator = RunCoordinator(
        store=store,
        executors={"retrieval": executor},
        query_worker_concurrency=1,
    )
    accepted = await _accept(store, key="coordinator")
    await coordinator.start()
    coordinator.wake()
    try:
        async with asyncio.timeout(10):
            while True:
                final = await store.get_run(owner_id=_OWNER, run_id=accepted.run.run_id)
                if final is not None and final.terminal:
                    break
                await asyncio.sleep(0.02)
    finally:
        await coordinator.aclose()

    assert final is not None
    assert final.status == "succeeded"
    assert final.result is not None
    assert final.result["trace"] == {"executor": "retrieval"}
    assert operation.calls == 1


async def test_active_requirement_scan_includes_retrieval_prepared_schema(
    retrieval_pg: tuple[PGRunStore, Any],
) -> None:
    store, _pool = retrieval_pg
    accepted = await _accept(store, key="startup-scan")

    requirements = [item async for item in store.iter_active_run_requirements(page_size=1)]

    assert requirements == [
        {
            "run_kind": "retrieval",
            "prepared_input": _prepared(),
        }
    ]
    assert accepted.run.status == "queued"
