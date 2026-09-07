# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The durable Answer REST contract over a live PostgreSQL run store.

Exercises what an in-memory fake cannot prove: that a created run and its
uploaded bytes are persisted in one transaction, that idempotency replay and
conflict are owner-scoped, that another owner's run is indistinguishable from an
unknown one, that a reconnecting subscriber replays the durable sequence without
gaps or duplicates, and that cancellation and event trimming return the exact
documented statuses.

Every test runs inside a throwaway database, so the developer's ``dlightrag``
database is never mutated. Requires PostgreSQL at localhost:5432; skipped
otherwise.
"""

import json
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import asyncpg
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.server import create_app
from dlightrag.adapters.postgres.runtime import PGRunBlobStore
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.application.access import UserContext, owner_id_from_user
from dlightrag.application.answer_runs import AnswerService
from dlightrag.application.answer_runs.capabilities import AnswerCapabilities, RequestModelContext
from dlightrag.application.config import DlightragConfig, set_config
from dlightrag.application.runs import RunService
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.settings import (
    MODEL_ROLE_NAMES,
    EmbeddingSettings,
    ModelRole,
    ModelRoleSettings,
    ModelSettings,
)
from tests.integration.pg_conn import PG_CONN_KWARGS

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_PG_CONN_KWARGS: dict[str, Any] = PG_CONN_KWARGS

_ANON = UserContext(user_id="anonymous", auth_mode="none")
_ALICE = UserContext(user_id="alice", auth_mode="jwt", claims={"iss": "https://issuer.test"})
_BOB = UserContext(user_id="bob", auth_mode="jwt", claims={"iss": "https://issuer.test"})
_PROFILE = ModelProfile(
    context_window_tokens=200_000,
    max_input_tokens=180_000,
    max_output_tokens=16_000,
)


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")

    db_name = f"dlightrag_run_api_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    created = await asyncpg.create_pool(
        **{**_PG_CONN_KWARGS, "database": db_name}, min_size=1, max_size=8
    )
    try:
        yield created
    finally:
        await created.close()
        admin = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


@pytest.fixture
async def store(pool: Any) -> PGRunStore:
    created = PGRunStore(pool=pool)
    await created.initialize()
    return created


class _StoreScheduler:
    """Wake-free scheduler whose subscriptions replay the real PG event page."""

    def __init__(self, store: PGRunStore) -> None:
        self._store = store
        self.is_started = True

    def wake(self) -> None:
        return None

    def cancel_local(self, owner_id: str, run_id: str) -> None:
        return None

    @asynccontextmanager
    async def admission(self) -> AsyncIterator[bool]:
        yield self.is_started

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncIterator[Any]:
        async def _iterate() -> AsyncIterator[Any]:
            events = await self._store.read_event_page(
                owner_id=owner_id,
                run_id=run_id,
                after_sequence=after_sequence,
            )
            for event in events:
                yield event

        return _iterate()


class _Resources:
    async def pin_current_image_links(
        self, request: Any, attachment_bytes: Any
    ) -> tuple[Any, list[bytes]]:
        return request, list(attachment_bytes)

    async def resolve(
        self,
        _resources: Any,
        /,
        *,
        models: RequestModelContext,
        text_window_budget: Any,
        confirm_image_context: Any,
        resolved_mode: str,
    ) -> Any:
        del text_window_budget, confirm_image_context, resolved_mode
        return SimpleNamespace(
            models=models,
            registry=None,
            current_images=(),
            resource_tools=(),
            resource_manifest=(),
            web_sources=None,
            image_budget=None,
            query_images=(),
        )


class _Capabilities:
    async def refresh_vlm(self) -> AnswerCapabilities:
        return AnswerCapabilities(answer=None, vlm_status="unknown")

    def current_profiles(self) -> dict[ModelRole, ModelProfile]:
        return {role: _PROFILE for role in MODEL_ROLE_NAMES}

    def request_model_context(
        self, profiles: dict[ModelRole, ModelProfile] | None, /
    ) -> RequestModelContext:
        selected = profiles or self.current_profiles()
        return RequestModelContext(
            extract=selected["extract"],
            query=selected["query"],
            vlm=selected["vlm"],
        )

    def answer_image_policy(self, _profile: ModelProfile, /) -> Any:
        return MagicMock()

    async def confirmed_live_answer_context(
        self, models: RequestModelContext, /
    ) -> tuple[RequestModelContext, None]:
        return models, None


class _Retrieval:
    def planner_for(self, _profile: ModelProfile | None = None) -> Any:
        planner = MagicMock()
        planner.history_input_measure.return_value = lambda messages, _summary="": (
            10 * len(messages)
        )
        return planner

    def warm(self, _workspaces: Any) -> None:
        return None

    async def schema_for(self, _workspaces: Any) -> dict[str, Any]:
        return {}


class _CapabilityView:
    async def read(self) -> AnswerCapabilities:
        return AnswerCapabilities(answer=None, vlm_status="unknown")


def _fingerprint(role: ModelRole) -> ModelFingerprint:
    return ModelFingerprint(provider="test", model=f"model-{role}", endpoint_fingerprint=None)


class _StoreBackedApplication:
    """Application shell with a real AnswerService wired to the real store."""

    def __init__(self, store: PGRunStore, config: DlightragConfig) -> None:
        self._store = store
        self.config = config
        self.corpora = SimpleNamespace(
            alist_workspace_records=self._alist_workspace_records,
        )
        scheduler = _StoreScheduler(store)
        self.runs = RunService(store=store, scheduler=scheduler)
        self.answers = AnswerService(
            store=store,
            blob_store=PGRunBlobStore(pool=store._operation_pool),  # noqa: SLF001
            coordinator=cast(Any, scheduler),
            retrieval=cast(Any, _Retrieval()),
            capabilities=cast(Any, _Capabilities()),
            capability_view=cast(Any, _CapabilityView()),
            models=cast(Any, SimpleNamespace(query_image_describer=lambda: MagicMock())),
            resources=cast(Any, _Resources()),
            model_fingerprint_for_role=_fingerprint,
            child_roster_cursor_secret=b"answer-run-api-child-roster-test",
        )

    @staticmethod
    async def _alist_workspace_records() -> list[dict[str, str]]:
        return [{"workspace": "default"}]


@pytest.fixture
def app(store: PGRunStore, tmp_path) -> Iterator[FastAPI]:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        deployment={
            "working_dir": str(tmp_path / "dlightrag_storage"),
        },
        models={
            "chat": ModelRoleSettings(default=ModelSettings(model="gpt-5.4-mini", api_key="test")),
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="test",
                startup_probe=False,
            ),
        },
    )
    set_config(config)
    application = create_app(include_web_app=False)
    application.state.application = _StoreBackedApplication(store, config)
    application.dependency_overrides[get_current_user] = lambda: _ANON
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as api:
        yield api


def _as_user(app: FastAPI, user: UserContext) -> None:
    app.dependency_overrides[get_current_user] = lambda: user


async def _claim(store: PGRunStore, owner: str, run_id: str) -> Any:
    claimed = await store.claim_next(worker_id="worker-1")
    assert claimed is not None
    assert claimed.run.run_id == run_id
    assert claimed.run.owner_id == owner
    return claimed


async def test_create_persists_the_run_and_its_uploaded_bytes(
    client: AsyncClient, store: PGRunStore, pool: Any
) -> None:
    response = await client.post(
        "/answer",
        data={"request": json.dumps({"query": "summarize"})},
        files=[("attachments", ("notes.txt", b"durable-bytes", "text/plain"))],
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    owner = owner_id_from_user(_ANON)
    record = await store.get_run(owner_id=owner, run_id=run_id)
    assert record is not None
    assert record.status == "queued"
    assert (record.prepared_input or {})["query"] == "summarize"
    references = await store.list_run_artifacts(owner_id=owner, run_id=run_id)
    assert [reference.filename for reference in references] == ["notes.txt"]
    assert (
        await PGRunBlobStore(pool=pool).read(owner_id=owner, digest=references[0].digest)
        == b"durable-bytes"
    )


async def test_idempotency_replays_and_conflicts_within_one_owner(
    client: AsyncClient, app: FastAPI
) -> None:
    headers = {"Idempotency-Key": "key-1"}
    first = await client.post("/answer", json={"query": "q"}, headers=headers)
    replay = await client.post("/answer", json={"query": "q"}, headers=headers)
    conflict = await client.post("/answer", json={"query": "different"}, headers=headers)

    assert first.status_code == replay.status_code == 202
    assert first.json()["run_id"] == replay.json()["run_id"]
    assert conflict.status_code == 409

    # The same key belongs to a different owner's namespace.
    _as_user(app, _ALICE)
    other = await client.post("/answer", json={"query": "different"}, headers=headers)
    assert other.status_code == 202
    assert other.json()["run_id"] != first.json()["run_id"]


@pytest.mark.parametrize("key", ["", "   "])
async def test_a_blank_idempotency_key_never_replays_or_conflicts(
    client: AsyncClient, key: str
) -> None:
    headers = {"Idempotency-Key": key}
    first = await client.post("/answer", json={"query": "one"}, headers=headers)
    second = await client.post("/answer", json={"query": "two"}, headers=headers)

    assert first.status_code == second.status_code == 202
    assert first.json()["run_id"] != second.json()["run_id"]


async def test_another_owner_cannot_read_cancel_or_follow_a_run(
    client: AsyncClient, app: FastAPI
) -> None:
    _as_user(app, _ALICE)
    run_id = (await client.post("/answer", json={"query": "q"})).json()["run_id"]

    _as_user(app, _BOB)
    assert (await client.get(f"/runs/{run_id}")).status_code == 404
    assert (await client.get(f"/runs/{run_id}/events")).status_code == 404
    assert (await client.delete(f"/runs/{run_id}")).status_code == 404


async def test_reconnect_replays_the_durable_sequence_without_gaps(
    client: AsyncClient, store: PGRunStore
) -> None:
    run_id = (await client.post("/answer", json={"query": "q"})).json()["run_id"]
    owner = owner_id_from_user(_ANON)
    claimed = await _claim(store, owner, run_id)
    worker = str(claimed.run.lease_owner)
    epoch = int(claimed.run.fencing_epoch)
    await store.record_phase(
        owner_id=owner, run_id=run_id, worker_id=worker, fencing_epoch=epoch, phase="planning"
    )
    await store.append_event(
        owner_id=owner,
        run_id=run_id,
        worker_id=worker,
        fencing_epoch=epoch,
        phase=None,
        event_type="token",
        payload={"text": "first"},
    )
    await store.append_event(
        owner_id=owner,
        run_id=run_id,
        worker_id=worker,
        fencing_epoch=epoch,
        phase=None,
        event_type="token",
        payload={"text": "second"},
    )

    full = await client.get(f"/runs/{run_id}/events")
    resumed = await client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": "1"})

    assert [line for line in full.text.splitlines() if line.startswith("id: ")] == [
        "id: 1",
        "id: 2",
        "id: 3",
    ]
    assert [line for line in resumed.text.splitlines() if line.startswith("id: ")] == [
        "id: 2",
        "id: 3",
    ]


async def test_cancellation_status_matrix(client: AsyncClient, store: PGRunStore) -> None:
    queued = (await client.post("/answer", json={"query": "queued"})).json()["run_id"]
    running = (await client.post("/answer", json={"query": "running"})).json()["run_id"]
    owner = owner_id_from_user(_ANON)

    cancelled = await client.delete(f"/runs/{queued}")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert (await client.delete(f"/runs/{queued}")).status_code == 200

    await _claim(store, owner, running)
    pending = await client.delete(f"/runs/{running}")
    assert pending.status_code == 202
    assert pending.json()["cancel_requested"] is True


async def test_a_trimmed_event_log_is_gone_but_the_result_remains(
    client: AsyncClient, store: PGRunStore, pool: Any
) -> None:
    run_id = (await client.post("/answer", json={"query": "q"})).json()["run_id"]
    owner = owner_id_from_user(_ANON)
    claimed = await _claim(store, owner, run_id)
    await store.finish_success(
        owner_id=owner,
        run_id=run_id,
        worker_id=str(claimed.run.lease_owner),
        fencing_epoch=int(claimed.run.fencing_epoch),
        result={"answer": "durable", "contexts": {"chunks": []}, "sources": []},
    )
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs SET finished_at = NOW() - INTERVAL '370 days', "
            "purge_after = NOW() - INTERVAL '5 days' WHERE run_id = $1",
            uuid.UUID(run_id),
        )
    assert await store.trim_expired_event_logs() == 1

    events = await client.get(f"/runs/{run_id}/events")
    status = await client.get(f"/runs/{run_id}")

    assert events.status_code == 410
    assert status.status_code == 200
    assert status.json()["result"]["answer"] == "durable"
