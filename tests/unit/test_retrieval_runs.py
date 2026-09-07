# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Focused contracts for durable top-level Retrieval acceptance and execution."""

import asyncio
import datetime
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from dlightrag.application.retrieval import (
    PinnedRetrievalModel,
    ProjectedRetrieval,
    RetrievalExecutor,
    RetrievalInputError,
    RetrievalService,
    RetrievalSettings,
    RetrieveRequest,
)
from dlightrag.application.runs import IdempotencyKeyConflict
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.telemetry import NoopTelemetry
from dlightrag.engine.dependencies import ProviderUnavailableError
from dlightrag.engine.rag.retrieval import RetrievalResult
from dlightrag.engine.runtime import (
    Deferred,
    Failed,
    RunAccessScope,
    RunCreation,
    RunExecutionContext,
    RunExecutionError,
    RunRecord,
    Succeeded,
)
from dlightrag.engine.runtime import (
    IdempotencyKeyConflict as RuntimeIdempotencyKeyConflict,
)

_OWNER = "owner-1"
_FINGERPRINT = ModelFingerprint(provider="openai", model="query-model", endpoint_fingerprint=None)
_PROFILE = ModelProfile(context_window_tokens=128_000, supports_images=True)
_NOW = datetime.datetime(2026, 8, 14, tzinfo=datetime.UTC)


class _Planner:
    async def plan(self, query: str, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            standalone_query=query,
            metadata_filter=None,
            metadata_filter_source=None,
            bm25_query=None,
            outcome="planned",
        )


class _Planners:
    def planner_for(self, model_profile: ModelProfile | None = None) -> Any:
        del model_profile
        return _Planner()

    async def aclose(self) -> None:
        return None


class _Coordinator:
    is_started = True

    def __init__(self) -> None:
        self.wakes = 0
        self.events: list[Any] = []

    @asynccontextmanager
    async def admission(self):
        yield self.is_started

    def wake(self) -> None:
        self.wakes += 1

    async def subscribe(self, **_kwargs: Any):
        for event in self.events:
            yield event


class _Store:
    def __init__(self) -> None:
        self.envelope = None
        self.run_id = ""
        self.replay: RunCreation | None = None
        self.final: RunRecord | None = None

    async def replay_run(self, **_kwargs: Any) -> RunCreation | None:
        return self.replay

    async def accept_run(self, *, envelope, run_id: str) -> RunCreation:
        self.envelope = envelope
        self.run_id = run_id
        record = _record(
            run_id=run_id,
            prepared=dict(envelope.payload),
            accepted=dict(envelope.accepted_input),
        )
        return RunCreation(run=record, replayed=False)

    async def get_run(self, **_kwargs: Any) -> RunRecord | None:
        return self.final


def test_retrieval_claim_exposes_no_answer_owned_execution_facilities() -> None:
    context = RunExecutionContext(
        owner_id=_OWNER,
        run_id="0199a0a0-0000-7000-8000-0000000000aa",
        worker_id="worker-1",
        lease_owner="worker-1",
        fencing_epoch=1,
    )

    with pytest.raises(RuntimeError, match="no Agent Session repository"):
        _ = context.session_repository
    with pytest.raises(RuntimeError, match="no Answer progress store"):
        _ = context.progress_store


def _record(
    *,
    run_id: str = "0199a0a0-0000-7000-8000-0000000000aa",
    status: str = "queued",
    prepared: dict[str, Any] | None = None,
    accepted: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> RunRecord:
    terminal = status in {"succeeded", "failed", "cancelled"}
    return RunRecord(
        run_id=run_id,
        run_kind="retrieval",
        lane="query",
        submitted_by=_OWNER,
        access_scope=RunAccessScope(kind="owner", scope_id=_OWNER),
        submission_key="key-1",
        request_fingerprint="fingerprint",
        prepared_input=prepared,
        accepted_input=accepted,
        status=status,  # type: ignore[arg-type]
        phase=None,
        stop_reason=None,
        cancel_requested_at=None,
        lease_owner=None,
        lease_expires_at=None,
        fencing_epoch=0,
        durable_progress_version=0,
        last_reclaim_progress_version=0,
        reclaims_without_progress=0,
        next_event_sequence=1,
        events_trimmed_at=None,
        result=result,
        error_kind=None,
        error_message=None,
        created_at=_NOW,
        updated_at=_NOW,
        started_at=None,
        finished_at=_NOW if terminal else None,
    )


def _service(*, store: Any | None = None, coordinator: Any | None = None) -> RetrievalService:
    return RetrievalService(
        pool=AsyncMock(),
        planners=_Planners(),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(
            return_value=ProjectedRetrieval(
                contexts={"chunks": [], "entities": [], "relationships": []},
                sources=(),
            )
        ),
        settings=RetrievalSettings(
            default_top_k=40,
            default_chunk_top_k=20,
            timeout_seconds=300,
            query_image_limit=3,
        ),
        telemetry=NoopTelemetry(),
        store=store,
        coordinator=coordinator,
        model_profile_for_role=lambda _role: _PROFILE,
        model_fingerprint_for_role=lambda _role: _FINGERPRINT,
    )


async def test_create_pins_normalized_recovery_input_and_seven_day_retention() -> None:
    store = _Store()
    coordinator = _Coordinator()
    service = _service(store=store, coordinator=coordinator)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}

    creation = await service.create(
        request=RetrieveRequest(
            query="report",
            workspaces=("finance",),
            bm25_query="  quarterly  ",
            query_images=(image,),
        ),
        owner_id=_OWNER,
        idempotency_key="key-1",
    )

    assert creation.run.run_kind == "retrieval"
    assert creation.run.lane == "query"
    assert coordinator.wakes == 1
    envelope = store.envelope
    assert envelope is not None
    assert envelope.retention_seconds == 7 * 24 * 3600
    assert envelope.payload["top_k"] == 40
    assert envelope.payload["chunk_top_k"] == 20
    assert envelope.payload["bm25_query"] == "quarterly"
    assert envelope.payload["query_images"] == [image]
    assert envelope.payload["context_policy_revision"] == CONTEXT_POLICY_REVISION
    assert envelope.payload["model_catalog_revision"] == current_model_catalog_revision()
    assert "capability_facts" not in envelope.payload
    assert {item["role"] for item in envelope.payload["pinned_models"]} == {"extract", "vlm"}
    assert envelope.accepted_input["query_image_count"] == 1
    assert len(envelope.accepted_input["query_image_digests"][0]) == 64
    assert "query_images" not in envelope.accepted_input
    assert envelope.accepted_input["result_projection"] == "retrieval_v1"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("top_k", 401, "top_k"), ("chunk_top_k", 201, "chunk_top_k")],
)
async def test_create_validates_config_relative_k_limits_before_acceptance(
    field: str, value: int, message: str
) -> None:
    store = _Store()
    request = {"query": "q", "workspaces": ("finance",), field: value}

    with pytest.raises(ValueError, match=message):
        await _service(store=store, coordinator=_Coordinator()).create(
            request=RetrieveRequest(**request),  # type: ignore[arg-type]
            owner_id=_OWNER,
        )

    assert store.envelope is None


async def test_create_rejects_oversized_query_images_before_storage() -> None:
    store = _Store()
    image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + "A" * (8 * 1024 * 1024)},
    }

    with pytest.raises(RetrievalInputError, match="prepared_input_too_large"):
        await _service(store=store, coordinator=_Coordinator()).create(
            request=RetrieveRequest(query="q", workspaces=("finance",), query_images=(image,)),
            owner_id=_OWNER,
        )

    assert store.envelope is None


async def test_create_translates_atomic_changed_input_conflict() -> None:
    store = _Store()
    store.replay_run = AsyncMock(side_effect=RuntimeIdempotencyKeyConflict("changed"))

    with pytest.raises(IdempotencyKeyConflict):
        await _service(store=store, coordinator=_Coordinator()).create(
            request=RetrieveRequest(query="changed", workspaces=("finance",)),
            owner_id=_OWNER,
            idempotency_key="key-1",
        )


class _Session:
    def __init__(self, prepared: dict[str, Any], checkpoint: dict[str, Any] | None = None) -> None:
        self.prepared_input = prepared
        self.checkpoint = checkpoint
        self.phases: list[str] = []

    async def enter_phase(self, phase: str) -> None:
        self.phases.append(phase)

    async def check_cancelled(self) -> None:
        return None


def _prepared(*, query_images: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    pin = PinnedRetrievalModel(
        role="extract",
        fingerprint=_FINGERPRINT,
        profile=_PROFILE,
    )
    pins = [pin.as_json()]
    if query_images:
        pins.append(
            PinnedRetrievalModel(role="vlm", fingerprint=_FINGERPRINT, profile=_PROFILE).as_json()
        )
    return {
        "query": "report",
        "workspaces": ["finance"],
        "top_k": 40,
        "chunk_top_k": 20,
        "federated_rerank": False,
        "bm25_query": None,
        "filters": None,
        "query_images": query_images or [],
        "pinned_models": pins,
        "context_policy_revision": CONTEXT_POLICY_REVISION,
        "model_catalog_revision": current_model_catalog_revision(),
        "idempotency_fingerprint": "fingerprint",
    }


async def test_executor_uses_two_phases_and_stores_no_projection_or_image_bytes() -> None:
    operation = SimpleNamespace(
        warm=Mock(),
        prepare_query_images=AsyncMock(return_value=["Image 1: chart"]),
        retrieve_result=AsyncMock(
            return_value=RetrievalResult(
                contexts={
                    "chunks": [
                        {
                            "chunk_id": "figure-1",
                            "file_path": "report.pdf",
                            "content": "chart",
                            "_workspace": "finance",
                            "image_data": "raw-bytes",
                            "image_url": "/stale/image",
                            "download_url": "/stale/download",
                        }
                    ]
                },
                trace={"thumbnail_url": "/stale/thumb", "count": 1},
                image_descriptions=["Image 1: chart"],
            )
        ),
    )
    session = _Session(
        _prepared(
            query_images=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
        )
    )
    executor = RetrievalExecutor(
        operation=cast(Any, operation),
        timeout_seconds=30,
        model_fingerprint_for_role=lambda _role: _FINGERPRINT,
    )

    outcome = await executor.execute(session)  # type: ignore[arg-type]

    assert isinstance(outcome, Succeeded)
    assert session.phases == ["planning", "searching"]
    chunk = outcome.result["contexts"]["chunks"][0]
    assert chunk["_has_visual_asset"] is True
    assert {"image_data", "image_url", "download_url"}.isdisjoint(chunk)
    assert "thumbnail_url" not in outcome.result["trace"]
    operation.retrieve_result.assert_awaited_once()


async def test_executor_rejects_model_catalog_drift_before_operation() -> None:
    operation = SimpleNamespace(
        warm=Mock(),
        prepare_query_images=AsyncMock(return_value=[]),
        retrieve_result=AsyncMock(),
    )
    prepared = _prepared()
    prepared["model_catalog_revision"] = "stale-catalog"
    executor = RetrievalExecutor(
        operation=cast(Any, operation),
        timeout_seconds=30,
        model_fingerprint_for_role=lambda _role: _FINGERPRINT,
    )

    with pytest.raises(RunExecutionError) as raised:
        await executor.execute(_Session(prepared))  # type: ignore[arg-type]

    assert raised.value.kind == "retrieval_model_catalog_changed"
    operation.warm.assert_not_called()


async def test_executor_timeout_is_a_terminal_public_failure() -> None:
    async def block(*_args: Any, **_kwargs: Any) -> RetrievalResult:
        await asyncio.Event().wait()
        raise AssertionError

    executor = RetrievalExecutor(
        operation=cast(
            Any,
            SimpleNamespace(
                warm=Mock(),
                prepare_query_images=AsyncMock(return_value=[]),
                retrieve_result=block,
            ),
        ),
        timeout_seconds=0.001,
        model_fingerprint_for_role=lambda _role: _FINGERPRINT,
    )

    with pytest.raises(RunExecutionError) as raised:
        await executor.execute(_Session(_prepared()))  # type: ignore[arg-type]

    assert raised.value.kind == "retrieval_timeout"


async def test_executor_defers_provider_unavailability_with_bounded_backoff() -> None:
    operation = SimpleNamespace(
        warm=Mock(),
        prepare_query_images=AsyncMock(return_value=[]),
        retrieve_result=AsyncMock(side_effect=ProviderUnavailableError()),
    )
    executor = RetrievalExecutor(
        operation=cast(Any, operation),
        timeout_seconds=30,
        model_fingerprint_for_role=lambda _role: _FINGERPRINT,
        now=lambda: _NOW,
    )

    outcome = await executor.execute(
        _Session(_prepared(), checkpoint={"providers_unavailable_attempt": 3})  # type: ignore[arg-type]
    )

    assert isinstance(outcome, Deferred)
    assert outcome.checkpoint == {"providers_unavailable_attempt": 4}
    assert (outcome.next_attempt_at - _NOW).total_seconds() == 40


async def test_executor_keeps_unknown_failure_terminal() -> None:
    operation = SimpleNamespace(
        warm=Mock(),
        prepare_query_images=AsyncMock(return_value=[]),
        retrieve_result=AsyncMock(side_effect=RuntimeError("unknown")),
    )
    executor = RetrievalExecutor(
        operation=cast(Any, operation),
        timeout_seconds=30,
        model_fingerprint_for_role=lambda _role: _FINGERPRINT,
    )

    outcome = await executor.execute(_Session(_prepared()))  # type: ignore[arg-type]

    assert isinstance(outcome, Failed)
    assert outcome.error_kind == "retrieval_failed"


async def test_executor_defers_corpus_unavailability_with_bounded_backoff() -> None:
    from dlightrag.application.errors import CorpusUnavailableError

    operation = SimpleNamespace(
        warm=Mock(),
        prepare_query_images=AsyncMock(return_value=[]),
        retrieve_result=AsyncMock(side_effect=CorpusUnavailableError("offline")),
    )
    executor = RetrievalExecutor(
        operation=cast(Any, operation),
        timeout_seconds=30,
        model_fingerprint_for_role=lambda _role: _FINGERPRINT,
        now=lambda: _NOW,
    )

    outcome = await executor.execute(
        _Session(_prepared(), checkpoint={"corpus_unavailable_attempt": 4})  # type: ignore[arg-type]
    )

    assert isinstance(outcome, Deferred)
    assert outcome.checkpoint == {"corpus_unavailable_attempt": 5}
    assert outcome.next_attempt_at == _NOW + datetime.timedelta(seconds=60)
