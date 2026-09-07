# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for the durable Answer application service."""

import asyncio
import contextlib
import datetime
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.application.answer_runs import (
    AnswerHistoryResource,
    AnswerRequest,
    AnswerRuntimeUnavailableError,
    AnswerService,
    ChildRosterCursor,
    ChildRosterPageRequest,
    ChildRosterRowPage,
)
from dlightrag.application.answer_runs.capabilities import AnswerCapabilities, RequestModelContext
from dlightrag.application.answer_runs.capability import AnswerImageCapability
from dlightrag.application.answer_runs.errors import (
    AnswerInputOverflowError,
    CurrentImagePayloadError,
    UnsupportedAnswerModeError,
)
from dlightrag.application.answer_runs.execution import AnswerRunInput, AnswerRunRequest
from dlightrag.application.answer_runs.routing import decide_resolved_mode
from dlightrag.application.runs import (
    RunCancelledError,
    RunFailedError,
)
from dlightrag.application.runs import (
    RunEvent as ApplicationRunEvent,
)
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import MODEL_CATALOG_REVISION
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.settings import MODEL_ROLE_NAMES, ModelRole
from dlightrag.engine.answer.execution import AnswerResourceResolver, AnswerResourceSettings
from dlightrag.engine.answer.resources.models import ResourceInput
from dlightrag.engine.runtime import (
    CancellationOutcome,
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
    RunAccessScope,
    RunArtifactReference,
    RunCreation,
    RunEvent,
    RunRecord,
)
from tests.unit.conftest import answer_image_policy

_OWNER = "owner-1"
_NOW = datetime.datetime(2026, 8, 17, tzinfo=datetime.UTC)
_PROFILE = ModelProfile(
    context_window_tokens=200_000,
    max_input_tokens=180_000,
)


def _record(
    *,
    run_id: str = "run-1",
    run_kind: str = "answer",
    status: str = "queued",
    result: Mapping[str, Any] | None = None,
    error_kind: str | None = None,
    error_message: str | None = None,
    accepted_input: Mapping[str, Any] | None = None,
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        run_kind=run_kind,  # type: ignore[arg-type]
        lane="query",
        submitted_by=_OWNER,
        access_scope=RunAccessScope(kind="owner", scope_id=_OWNER),
        submission_key=None or str(run_id),
        request_fingerprint="test-fingerprint",
        prepared_input={"query": "q"},
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
        error_kind=error_kind,
        error_message=error_message,
        created_at=_NOW,
        updated_at=_NOW,
        started_at=None,
        finished_at=None,
        accepted_input=accepted_input,
    )


class _Store:
    """The owner-scoped durable operations, recorded in memory."""

    def __init__(
        self,
        *,
        replay: RunCreation | None = None,
        run: RunRecord | None = None,
        references: tuple[RunArtifactReference, ...] = (),
        blobs: Mapping[str, bytes] | None = None,
    ) -> None:
        self._replay = replay
        self._run = run or _record()
        self._references = references
        self._blobs = dict(blobs or {})
        self.created: list[dict[str, Any]] = []
        self.replay_calls = 0
        self.cancellations: list[tuple[str, str]] = []
        self.artifact_reads: list[tuple[str, str]] = []
        self.controls: list[dict[str, Any]] = []
        self.child_rows: tuple[Mapping[str, Any], ...] = ()
        self.child_page_rows: tuple[Mapping[str, Any], ...] = ()
        self.child_page_has_more = False
        self.transcript_rows: tuple[Mapping[str, Any], ...] = ()

    async def create_run(
        self,
        *,
        envelope: PreparedRunEnvelope,
        run_id: str,
        resources: Sequence[Mapping[str, Any]] = (),
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: object | None = None,
    ) -> RunCreation:
        self.created.append(
            {
                "owner_id": envelope.submitted_by,
                "run_id": run_id,
                "envelope": envelope,
                "prepared_input": dict(envelope.payload),
                "idempotency_fingerprint": envelope.request_fingerprint,
                "resources": resources,
                "idempotency_key": envelope.submission_key,
                "artifacts": [artifact.content for artifact in artifacts],
                "references": list(references),
                "routing": routing,
            }
        )
        return RunCreation(run=self._run, replayed=False)

    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        run_kind: str,
    ) -> RunCreation | None:
        del owner_id, idempotency_key, idempotency_fingerprint
        assert run_kind == "answer"
        self.replay_calls += 1
        return self._replay

    async def get_run(self, *, owner_id: str, run_id: str) -> RunRecord | None:
        if owner_id != _OWNER or run_id != self._run.run_id:
            return None
        return self._run

    async def list_runs(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> tuple[RunRecord, ...]:
        if owner_id != _OWNER:
            return ()
        return (self._run,)

    async def request_cancellation(self, *, owner_id: str, run_id: str) -> CancellationOutcome:
        self.cancellations.append((owner_id, run_id))
        return CancellationOutcome(outcome="cancelled", run=self._run)

    async def enqueue_agent_control(
        self, *, owner_id: str, run_id: str, kind: str, content: str
    ) -> Mapping[str, Any] | None:
        if owner_id != _OWNER or run_id != self._run.run_id or self._run.terminal:
            return None
        row = {
            "run_id": run_id,
            "control_sequence": len(self.controls) + 1,
            "kind": kind,
            "content": content,
        }
        self.controls.append(row)
        return row

    async def list_child_sessions(
        self, *, owner_id: str, run_id: str
    ) -> tuple[Mapping[str, Any], ...]:
        if owner_id != _OWNER or run_id != self._run.run_id:
            return ()
        return self.child_rows

    async def list_child_sessions_page(
        self,
        *,
        owner_id: str,
        run_id: str,
        page: ChildRosterPageRequest,
    ) -> ChildRosterRowPage:
        del page
        if owner_id != _OWNER or run_id != self._run.run_id:
            return ChildRosterRowPage(children=(), has_more=False, fetched_rows=0)
        return ChildRosterRowPage(
            children=self.child_page_rows,
            has_more=self.child_page_has_more,
            fetched_rows=len(self.child_page_rows),
        )

    async def load_agent_transcript(
        self, *, owner_id: str, run_id: str, session_id: str, limit: int
    ) -> tuple[Mapping[str, Any], ...]:
        del session_id
        if owner_id != _OWNER or run_id != self._run.run_id:
            return ()
        return self.transcript_rows[-limit:]

    async def list_run_artifacts(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunArtifactReference, ...]:
        if owner_id != _OWNER or run_id != self._run.run_id:
            return ()
        return self._references

    async def stream(
        self,
        *,
        owner_id: str,
        digest: str,
        offset: int = 0,
        length: int | None = None,
    ) -> AsyncIterator[bytes]:
        self.artifact_reads.append((owner_id, digest))
        if owner_id != _OWNER:
            return
        blob = self._blobs.get(digest)
        if blob is not None:
            yield blob[max(0, offset) :]

    async def size(self, *, owner_id: str, digest: str) -> int | None:
        if owner_id != _OWNER:
            return None
        blob = self._blobs.get(digest)
        return None if blob is None else len(blob)


class _Coordinator:
    """The started coordinator, replaying a scripted event stream."""

    def __init__(self, events: Sequence[RunEvent] = (), *, block: bool = False) -> None:
        self._events = tuple(events)
        self._block = block
        self.is_started = True
        self._admission_lock = asyncio.Lock()
        self.wakes = 0
        self.cancelled_local: list[str] = []
        self.subscriptions: list[tuple[str, str, int]] = []
        self.attached = asyncio.Event()

    def cancel_local(self, owner_id: str, run_id: str) -> None:
        self.cancelled_local.append(run_id)

    @contextlib.asynccontextmanager
    async def admission(self) -> AsyncIterator[bool]:
        async with self._admission_lock:
            yield self.is_started

    async def stop(self) -> None:
        async with self._admission_lock:
            self.is_started = False

    def wake(self) -> None:
        self.wakes += 1

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncGenerator[RunEvent]:
        self.subscriptions.append((owner_id, run_id, after_sequence))

        async def _events() -> AsyncGenerator[RunEvent]:
            for event in self._events:
                yield event
            self.attached.set()
            if self._block:
                await asyncio.Event().wait()

        return _events()


class _Capabilities:
    def __init__(self) -> None:
        self.vlm_refreshes = 0

    async def refresh_vlm(self) -> AnswerCapabilities:
        self.vlm_refreshes += 1
        return AnswerCapabilities(answer=None, vlm_status="unknown")

    def current_profiles(self) -> dict[ModelRole, ModelProfile]:
        return {role: _PROFILE for role in MODEL_ROLE_NAMES}

    def request_model_context(
        self, pinned: Mapping[ModelRole, ModelProfile] | None, /
    ) -> RequestModelContext:
        profiles = pinned or self.current_profiles()
        return RequestModelContext(
            extract=profiles["extract"],
            query=profiles["query"],
            vlm=profiles["vlm"],
        )

    def answer_image_policy(self, profile: ModelProfile, /) -> Any:
        del profile
        return MagicMock()

    async def confirmed_live_answer_context(
        self, models: RequestModelContext, /
    ) -> tuple[RequestModelContext, Any]:
        return models, None


class _TextQueryInspectCapabilities(_Capabilities):
    def __init__(self, *, configured_ceiling: int = 3, query_status: str = "unsupported") -> None:
        super().__init__()
        self._answer = AnswerImageCapability(
            status=query_status,  # type: ignore[arg-type]
            configured_ceiling=configured_ceiling,
            effective_max_images=configured_ceiling if query_status == "supported" else 0,
            provider="test",
            base_url=None,
            model="text-query",
            failure_kind=None,
        )

    async def refresh_answer(self) -> AnswerCapabilities:
        return AnswerCapabilities(answer=self._answer, vlm_status="supported")

    async def refresh_vlm(self) -> AnswerCapabilities:
        self.vlm_refreshes += 1
        return AnswerCapabilities(answer=self._answer, vlm_status="supported")

    def current_profiles(self) -> dict[ModelRole, ModelProfile]:
        profiles = super().current_profiles()
        profiles["query"] = ModelProfile(
            context_window_tokens=_PROFILE.context_window_tokens,
            max_input_tokens=_PROFILE.max_input_tokens,
            supports_images=False,
        )
        profiles["vlm"] = ModelProfile(
            context_window_tokens=_PROFILE.context_window_tokens,
            max_input_tokens=_PROFILE.max_input_tokens,
            supports_images=True,
        )
        return profiles

    def answer_image_policy(self, profile: ModelProfile, /) -> Any:
        return answer_image_policy(
            max_images=self._answer.configured_ceiling if profile.supports_images else 0
        )

    def vlm_image_policy(self, profile: ModelProfile, /) -> Any:
        return answer_image_policy(
            max_images=self._answer.configured_ceiling if profile.supports_images else 0
        )

    async def confirmed_live_answer_context(
        self, models: RequestModelContext, /
    ) -> tuple[RequestModelContext, AnswerImageCapability]:
        return models, self._answer


class _SmallProfileCapabilities(_Capabilities):
    """A profile whose physical input cannot preserve Fast's full 40K reserve."""

    def current_profiles(self) -> dict[ModelRole, ModelProfile]:
        profile = ModelProfile(context_window_tokens=30_000)
        return {role: profile for role in MODEL_ROLE_NAMES}


class _FastMemoryBoundaryCapabilities(_Capabilities):
    """A profile with exactly 100 tokens before Fast's full dynamic reserve."""

    def current_profiles(self) -> dict[ModelRole, ModelProfile]:
        profile = ModelProfile(context_window_tokens=57_508)
        return {role: profile for role in MODEL_ROLE_NAMES}


class _Registry:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _Resources:
    """Link pinning and resolution, recording the order acceptance ran in."""

    def __init__(
        self, *, registry: _Registry | None = None, calls: list[str] | None = None
    ) -> None:
        self._registry = registry
        self.calls = calls if calls is not None else []

    async def pin_current_image_links(
        self, request: AnswerRunRequest, attachment_bytes: Sequence[bytes], /
    ) -> tuple[AnswerRunRequest, list[bytes]]:
        self.calls.append("pin_current_image_links")
        return request, list(attachment_bytes)

    async def resolve(
        self,
        resources: list[ResourceInput] | None,
        /,
        *,
        models: RequestModelContext,
        text_window_budget: Any,
        confirm_image_context: Any,
        resolved_mode: str = "fast",
    ) -> Any:
        del resources, text_window_budget, confirm_image_context, resolved_mode
        self.calls.append("resolve")
        return MagicMock(
            models=models,
            registry=self._registry,
            current_images=[],
            resource_tools=[],
            resource_manifest=(),
            web_sources=None,
            image_budget=None,
            query_images=None,
        )


def _planner() -> Any:
    planner = MagicMock()
    planner.history_input_measure.return_value = lambda messages, _summary="": 10 * len(messages)
    return planner


class _Retrieval:
    def __init__(self, calls: list[str] | None = None) -> None:
        self.calls = calls if calls is not None else []
        self.warmed: list[list[str]] = []

    def planner_for(self, model_profile: ModelProfile | None = None) -> Any:
        del model_profile
        return _planner()

    def warm(self, workspaces: Sequence[str]) -> None:
        self.warmed.append(list(workspaces))

    async def schema_for(self, workspaces: Sequence[str]) -> dict[str, Any]:
        del workspaces
        self.calls.append("schema_for")
        return {}


class _CapabilityView:
    def __init__(self, snapshot: AnswerCapabilities) -> None:
        self._snapshot = snapshot
        self.reads = 0

    async def read(self) -> AnswerCapabilities:
        self.reads += 1
        return self._snapshot


def _fingerprint(role: ModelRole) -> ModelFingerprint:
    return ModelFingerprint(provider="test", model=f"model-{role}", endpoint_fingerprint=None)


def _service(
    *,
    store: Any = None,
    coordinator: Any = None,
    retrieval: Any = None,
    capabilities: Any = None,
    capability_view: Any = None,
    resources: Any = None,
    memory_capability: Any = None,
) -> AnswerService:
    selected_store = store or _Store()
    return AnswerService(
        store=selected_store,
        blob_store=selected_store,
        coordinator=coordinator or _Coordinator(),
        retrieval=retrieval or _Retrieval(),
        capabilities=capabilities or _Capabilities(),
        capability_view=capability_view or _CapabilityView(AnswerCapabilities(None, "unknown")),
        models=MagicMock(query_image_describer=MagicMock(return_value=MagicMock())),
        resources=resources or _Resources(),
        model_fingerprint_for_role=_fingerprint,
        memory_capability=memory_capability,
        child_roster_cursor_secret=b"answer-service-child-roster-test",
    )


def _request(**overrides: Any) -> AnswerRequest:
    values: dict[str, Any] = {"query": "why?", "workspaces": ("finance",)}
    values.update(overrides)
    return AnswerRequest(**values)


async def test_acceptance_pins_disabled_profile_memory_without_reserving_its_capability() -> None:
    store = _Store()

    async def disabled(**_kwargs: Any) -> tuple[bool, int]:
        return False, 7

    service = _service(store=store, memory_capability=disabled)
    await service.create(
        request=_request(mode="research"),
        owner_id=_OWNER,
        auth_mode="jwt",
    )

    prepared = store.created[0]["prepared_input"]
    assert prepared["profile_memory_enabled"] is False
    assert prepared["profile_memory_epoch"] == 7


async def test_fast_acceptance_never_enters_profile_memory_capability() -> None:
    store = _Store()

    async def unexpected(**_kwargs: Any) -> tuple[bool, int]:
        raise AssertionError("Fast mode must not read Profile Memory settings")

    service = _service(store=store, memory_capability=unexpected)
    await service.create(
        request=_request(mode="fast"),
        owner_id=_OWNER,
        auth_mode="jwt",
    )

    prepared = store.created[0]["prepared_input"]
    assert prepared["profile_memory_enabled"] is False
    assert prepared["profile_memory_epoch"] == 0


async def test_explicit_fast_with_text_query_and_current_image_creates_no_run() -> None:
    store = _Store()
    resources = _Resources()
    service = _service(
        store=store,
        capabilities=_TextQueryInspectCapabilities(),
        resources=resources,
    )

    with pytest.raises(UnsupportedAnswerModeError):
        await service.create(
            request=_request(
                mode="fast",
                resources=(
                    ResourceInput(
                        filename="chart.png",
                        content=b"image",
                        declared_mime="image/png",
                    ),
                ),
            ),
            owner_id=_OWNER,
        )

    assert store.created == []
    assert resources.calls == ["pin_current_image_links"]


async def test_auto_text_query_with_inspect_persists_research_only() -> None:
    store = _Store()
    service = _service(
        store=store,
        capabilities=_TextQueryInspectCapabilities(),
    )

    await service.create(
        request=_request(
            mode="auto",
            resources=(
                ResourceInput(
                    filename="chart.png",
                    content=b"image",
                    declared_mime="image/png",
                ),
            ),
        ),
        owner_id=_OWNER,
    )

    assert store.created[0]["routing"].valid_modes == ("research",)


@pytest.mark.parametrize("mode", ["auto", "fast"])
async def test_capability_recovered_while_pinning_is_not_rejected_early(mode: str) -> None:
    class RecoveringCapabilities(_Capabilities):
        def __init__(self) -> None:
            super().__init__()
            self.recovered = False

        def current_profiles(self) -> dict[ModelRole, ModelProfile]:
            profiles = super().current_profiles()
            profiles["query"] = ModelProfile(
                context_window_tokens=_PROFILE.context_window_tokens,
                max_input_tokens=_PROFILE.max_input_tokens,
                supports_images=self.recovered,
            )
            profiles["vlm"] = ModelProfile(
                context_window_tokens=_PROFILE.context_window_tokens,
                max_input_tokens=_PROFILE.max_input_tokens,
                supports_images=self.recovered,
            )
            return profiles

    class RecoveringResources(_Resources):
        async def pin_current_image_links(
            self, request: AnswerRunRequest, attachment_bytes: Sequence[bytes], /
        ) -> tuple[AnswerRunRequest, list[bytes]]:
            self.calls.append("pin_current_image_links")
            capabilities.recovered = True
            return request, list(attachment_bytes)

    store = _Store()
    capabilities = RecoveringCapabilities()
    resources = RecoveringResources()
    service = _service(store=store, capabilities=capabilities, resources=resources)

    await service.create(
        request=_request(
            mode=mode,
            resources=(
                ResourceInput(
                    filename="chart.png",
                    content=b"image",
                    declared_mime="image/png",
                ),
            ),
        ),
        owner_id=_OWNER,
    )

    assert store.created[0]["routing"].valid_modes == ("fast", "research")
    assert resources.calls[0] == "pin_current_image_links"
    assert capabilities.vlm_refreshes == 1


@pytest.mark.parametrize("mode", ["auto", "fast"])
async def test_capability_narrowed_while_pinning_recomputes_valid_modes(mode: str) -> None:
    class NarrowingCapabilities(_Capabilities):
        def __init__(self) -> None:
            super().__init__()
            self.query_narrowed = False

        def current_profiles(self) -> dict[ModelRole, ModelProfile]:
            profiles = super().current_profiles()
            profiles["query"] = ModelProfile(
                context_window_tokens=_PROFILE.context_window_tokens,
                max_input_tokens=_PROFILE.max_input_tokens,
                supports_images=not self.query_narrowed,
            )
            profiles["vlm"] = ModelProfile(
                context_window_tokens=_PROFILE.context_window_tokens,
                max_input_tokens=_PROFILE.max_input_tokens,
                supports_images=True,
            )
            return profiles

    class NarrowingResources(_Resources):
        async def pin_current_image_links(
            self, request: AnswerRunRequest, attachment_bytes: Sequence[bytes], /
        ) -> tuple[AnswerRunRequest, list[bytes]]:
            self.calls.append("pin_current_image_links")
            capabilities.query_narrowed = True
            return request, list(attachment_bytes)

    store = _Store()
    capabilities = NarrowingCapabilities()
    resources = NarrowingResources()
    service = _service(store=store, capabilities=capabilities, resources=resources)
    request = _request(
        mode=mode,
        resources=(
            ResourceInput(
                filename="chart.png",
                content=b"image",
                declared_mime="image/png",
            ),
        ),
    )

    if mode == "fast":
        with pytest.raises(UnsupportedAnswerModeError):
            await service.create(request=request, owner_id=_OWNER)
        assert store.created == []
    else:
        await service.create(request=request, owner_id=_OWNER)
        assert store.created[0]["routing"].valid_modes == ("research",)

    assert resources.calls[0] == "pin_current_image_links"
    assert capabilities.vlm_refreshes == 1


async def test_service_enforces_image_ceiling_on_text_query_inspect_path() -> None:
    store = _Store()
    capabilities = _TextQueryInspectCapabilities(configured_ceiling=1)
    runtime_models = MagicMock()
    resolver = AnswerResourceResolver(
        settings=AnswerResourceSettings(
            max_attachments=6,
            max_attachment_bytes=10_000,
            max_total_attachment_bytes=20_000,
            image_max_bytes=5_000,
            image_max_pixels=4_000_000,
        ),
        models=runtime_models,
        capabilities=capabilities,  # type: ignore[arg-type]
    )
    service = _service(store=store, capabilities=capabilities, resources=resolver)

    with pytest.raises(CurrentImagePayloadError, match="at most 1"):
        await service.create(
            request=_request(
                mode="auto",
                resources=(
                    ResourceInput(
                        filename="one.png",
                        content=b"one",
                        declared_mime="image/png",
                    ),
                    ResourceInput(
                        filename="two.png",
                        content=b"two",
                        declared_mime="image/png",
                    ),
                ),
            ),
            owner_id=_OWNER,
        )

    assert store.created == []


async def test_auto_removes_fast_before_persisting_routing_when_40k_cannot_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dlightrag.application.answer_runs.service as service_module

    monkeypatch.setattr(
        service_module,
        "research_history_input_measure",
        lambda **_kwargs: lambda _history, _summary="": 0,
    )
    store = _Store()
    service = _service(store=store, capabilities=_SmallProfileCapabilities())

    await service.create(request=_request(mode="auto"), owner_id=_OWNER)

    routing = store.created[0]["routing"]
    assert routing.valid_modes == ("research",)
    assert (
        decide_resolved_mode(
            requested_mode=routing.requested_mode,
            valid_modes=frozenset(routing.valid_modes),
        )
        == "research"
    )


async def test_auto_fast_capacity_excludes_research_only_profile_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dlightrag.application.answer_runs.service as service_module

    observed_fast_memory: list[str] = []

    def history_measure(
        _self: object,
        _query: str,
        memory_text: str = "",
        episodic_summary: str = "",
    ) -> Any:
        del episodic_summary
        observed_fast_memory.append(memory_text)
        return lambda _history, _summary="": 50 + len(memory_text)

    monkeypatch.setattr(service_module, "standing_memory_for_acceptance", lambda _auth: "m" * 100)
    monkeypatch.setattr(
        service_module.AnswerSynthesizer,
        "history_input_measure",
        history_measure,
    )
    monkeypatch.setattr(
        service_module,
        "research_history_input_measure",
        lambda **_kwargs: lambda _history, _summary="": 0,
    )
    store = _Store()
    service = _service(store=store, capabilities=_FastMemoryBoundaryCapabilities())

    await service.create(
        request=_request(mode="auto"),
        owner_id=_OWNER,
        auth_mode="jwt",
    )

    assert observed_fast_memory == [""]
    assert store.created[0]["routing"].valid_modes == ("fast", "research")


async def test_explicit_fast_rejects_when_40k_cannot_fit() -> None:
    store = _Store()
    service = _service(store=store, capabilities=_SmallProfileCapabilities())

    with pytest.raises(AnswerInputOverflowError, match="fast_planner fixed input"):
        await service.create(request=_request(mode="fast"), owner_id=_OWNER)

    assert store.created == []


async def test_explicit_fast_with_pdf_creates_no_run() -> None:
    store = _Store()
    service = _service(store=store)
    with pytest.raises(UnsupportedAnswerModeError):
        await service.create(
            request=_request(
                mode="fast",
                resources=(ResourceInput(filename="brief.pdf", content=b"%PDF"),),
            ),
            owner_id=_OWNER,
        )
    assert store.created == []


async def test_explicit_fast_with_history_pdf_creates_no_run() -> None:
    store = _Store()
    service = _service(store=store)
    with pytest.raises(UnsupportedAnswerModeError):
        await service.create(
            request=_request(
                mode="fast",
                history_resources=(
                    AnswerHistoryResource(
                        run_id="run-prior",
                        source_ordinal=0,
                        digest="a" * 64,
                        filename="old.pdf",
                        mime_type="application/pdf",
                        byte_size=4,
                    ),
                ),
            ),
            owner_id=_OWNER,
        )
    assert store.created == []


async def test_idempotent_replay_returns_before_preparation_and_materialization() -> None:
    replayed = RunCreation(run=_record(status="running"), replayed=True)
    store = _Store(replay=replayed)
    resources = _Resources()
    retrieval = _Retrieval()
    service = _service(store=store, resources=resources, retrieval=retrieval)

    creation = await service.create(request=_request(), owner_id=_OWNER, idempotency_key="key-1")

    assert creation.replayed is True
    assert creation.run.run_id == replayed.run.run_id
    assert resources.calls == []
    assert retrieval.calls == []
    assert store.created == []
    assert store.replay_calls == 1


async def test_idempotent_replay_precedes_live_multimodal_capability_validation() -> None:
    replayed = RunCreation(run=_record(status="running"), replayed=True)
    store = _Store(replay=replayed)
    resources = _Resources()
    service = _service(
        store=store,
        capabilities=_TextQueryInspectCapabilities(),
        resources=resources,
    )

    creation = await service.create(
        request=_request(
            mode="fast",
            resources=(
                ResourceInput(
                    filename="chart.png",
                    content=b"image",
                    declared_mime="image/png",
                ),
            ),
        ),
        owner_id=_OWNER,
        idempotency_key="key-image",
    )

    assert creation.replayed is True
    assert creation.run.run_id == replayed.run.run_id
    assert resources.calls == []
    assert store.created == []
    assert store.replay_calls == 1


async def test_accepted_run_stores_input_artifacts_and_wakes_the_coordinator() -> None:
    store = _Store()
    coordinator = _Coordinator()
    service = _service(store=store, coordinator=coordinator)

    await service.create(
        request=_request(resources=(ResourceInput(filename="a.txt", content=b"hello"),)),
        owner_id=_OWNER,
        idempotency_key="key-1",
    )

    accepted = store.created[0]
    assert accepted["owner_id"] == _OWNER
    assert accepted["idempotency_key"] == "key-1"
    assert accepted["artifacts"] == [b"hello"]
    assert accepted["routing"] is not None
    assert accepted["routing"].requested_mode == "auto"
    assert "research" in accepted["routing"].valid_modes
    assert [
        (reference.reference_kind, reference.ordinal, reference.filename)
        for reference in accepted["references"]
    ] == [("current_attachment", 0, "a.txt")]
    assert coordinator.wakes == 1


async def test_carried_history_resource_loads_from_the_run_that_accepted_it() -> None:
    loaded: list[bytes] = []

    class LoadingResources(_Resources):
        async def resolve(self, resources, /, **kwargs):
            for resource in resources or ():
                if resource.loader is not None:
                    loaded.append(await resource.loader())
            return await super().resolve(resources, **kwargs)

    store = _Store(
        references=(
            _reference(
                kind="current_attachment",
                ordinal=3,
                digest="b" * 64,
                filename="prior.txt",
                mime_type="text/plain",
            ),
        ),
        blobs={"b" * 64: b"prior-bytes"},
    )
    service = _service(store=store, resources=LoadingResources())

    await service.create(
        request=_request(
            history_resources=(
                AnswerHistoryResource(
                    run_id="run-1",
                    source_ordinal=3,
                    digest="b" * 64,
                    filename="prior.txt",
                    mime_type="text/plain",
                    byte_size=11,
                ),
            )
        ),
        owner_id=_OWNER,
    )

    assert loaded == [b"prior-bytes"]
    accepted = store.created[0]
    assert [
        (reference.reference_kind, reference.ordinal, reference.digest)
        for reference in accepted["references"]
    ] == [("history_attachment", 0, "b" * 64)]

    run_input = AnswerRunInput.from_request(accepted["prepared_input"])
    assert run_input.workspaces == ("finance",)
    assert run_input.context_policy_revision == CONTEXT_POLICY_REVISION
    assert run_input.model_catalog_revision == MODEL_CATALOG_REVISION
    assert {pinned.role for pinned in run_input.pinned_models} == set(MODEL_ROLE_NAMES)
    assert run_input.agent_run_plan is not None
    assert run_input.agent_run_plan.context_policy_revision == CONTEXT_POLICY_REVISION
    assert run_input.agent_run_plan.digest


async def test_preparation_runs_after_link_materialization() -> None:
    calls: list[str] = []
    service = _service(
        resources=_Resources(calls=calls),
        retrieval=_Retrieval(calls),
    )

    await service.create(request=_request(), owner_id=_OWNER)

    assert calls == ["pin_current_image_links", "resolve", "schema_for"]


@pytest.mark.parametrize(
    "workspaces",
    [(), ("Finance Reports",), ("*",), ("finance,legal",), ("finance", "Legal")],
)
async def test_create_rejects_non_canonical_or_empty_scope(workspaces: tuple[str, ...]) -> None:
    store = _Store()
    service = _service(store=store)

    with pytest.raises(ValueError):
        await service.create(request=_request(workspaces=workspaces), owner_id=_OWNER)

    assert store.created == []


async def test_resource_registry_closes_when_preparation_fails() -> None:
    registry = _Registry()
    retrieval = _Retrieval()
    retrieval.schema_for = AsyncMock(side_effect=RuntimeError("schema unavailable"))  # type: ignore[method-assign]
    service = _service(resources=_Resources(registry=registry), retrieval=retrieval)

    with pytest.raises(RuntimeError, match="schema unavailable"):
        await service.create(request=_request(), owner_id=_OWNER)

    assert registry.closed is True


async def test_create_rejects_an_unstarted_runtime_before_persisting_a_run() -> None:
    store = _Store()
    coordinator = _Coordinator()
    coordinator.is_started = False
    service = _service(store=store, coordinator=coordinator)

    with pytest.raises(AnswerRuntimeUnavailableError, match="runtime is unavailable"):
        await service.create(request=_request(), owner_id=_OWNER, idempotency_key="key-1")

    assert store.replay_calls == 1
    assert store.created == []
    assert coordinator.wakes == 0


async def test_runtime_stopping_during_preparation_prevents_persistence() -> None:
    store = _Store()
    coordinator = _Coordinator()
    preparation_started = asyncio.Event()
    release_preparation = asyncio.Event()
    resources = _Resources()

    async def pin_then_wait(
        request: AnswerRunRequest,
        attachment_bytes: Sequence[bytes],
        /,
    ) -> tuple[AnswerRunRequest, list[bytes]]:
        preparation_started.set()
        await release_preparation.wait()
        return request, list(attachment_bytes)

    resources.pin_current_image_links = pin_then_wait  # type: ignore[method-assign]
    service = _service(store=store, coordinator=coordinator, resources=resources)
    acceptance = asyncio.create_task(service.create(request=_request(), owner_id=_OWNER))
    await preparation_started.wait()

    await coordinator.stop()
    release_preparation.set()

    with pytest.raises(AnswerRuntimeUnavailableError, match="runtime is unavailable"):
        await acceptance
    assert store.created == []
    assert coordinator.wakes == 0


async def test_idempotent_replay_survives_local_runtime_unavailability() -> None:
    replayed = RunCreation(run=_record(status="running"), replayed=True)
    store = _Store(replay=replayed)
    coordinator = _Coordinator()
    coordinator.is_started = False
    service = _service(store=store, coordinator=coordinator)

    creation = await service.create(
        request=_request(),
        owner_id=_OWNER,
        idempotency_key="key-1",
    )

    assert creation.replayed is True
    assert creation.run.run_id == replayed.run.run_id
    assert store.replay_calls == 1
    assert store.created == []
    assert coordinator.wakes == 0


async def test_wait_projects_a_succeeded_run_into_its_canonical_result() -> None:
    store = _Store(run=_record(status="succeeded", result={"answer": "42"}))
    service = _service(store=store, coordinator=_Coordinator())

    result = await service.wait(owner_id=_OWNER, run_id="run-1")

    assert result.answer == "42"


async def test_wait_raises_the_typed_cancellation_of_a_cancelled_run() -> None:
    service = _service(store=_Store(run=_record(status="cancelled")))

    with pytest.raises(RunCancelledError):
        await service.wait(owner_id=_OWNER, run_id="run-1")


async def test_wait_raises_the_public_failure_of_a_failed_run() -> None:
    store = _Store(
        run=_record(status="failed", error_kind="model_unavailable", error_message="no model")
    )
    service = _service(store=store)

    with pytest.raises(RunFailedError) as failure:
        await service.wait(owner_id=_OWNER, run_id="run-1")

    assert failure.value.error_kind == "model_unavailable"
    assert failure.value.public_message == "no model"


async def test_observer_cancellation_never_requests_run_cancellation() -> None:
    store = _Store(run=_record(status="running"))
    coordinator = _Coordinator(block=True)
    service = _service(store=store, coordinator=coordinator)

    waiter = asyncio.create_task(service.wait(owner_id=_OWNER, run_id="run-1"))
    await coordinator.attached.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert store.cancellations == []


def test_answer_service_does_not_own_the_common_run_lifecycle() -> None:
    service = _service()

    assert not hasattr(service, "get")
    assert not hasattr(service, "list")
    assert not hasattr(service, "cancel")
    assert not hasattr(service, "resume")
    assert not hasattr(service, "subscribe")


async def test_answer_creates_a_durable_run_and_waits_for_its_result() -> None:
    store = _Store(run=_record(status="succeeded", result={"answer": "durable"}))
    coordinator = _Coordinator()
    service = _service(store=store, coordinator=coordinator)

    result = await service.answer(_request(), owner_id=_OWNER)

    assert result.answer == "durable"
    assert len(store.created) == 1
    assert coordinator.subscriptions == [(_OWNER, "run-1", 0)]


async def test_answer_stream_creates_a_durable_run_and_follows_its_events() -> None:
    event = RunEvent(
        sequence=1,
        event_type="token",
        payload={"text": "hi"},
        created_at=_NOW,
    )
    store = _Store()
    coordinator = _Coordinator([event])
    service = _service(store=store, coordinator=coordinator)

    streamed = [item async for item in service.answer_stream(_request(), owner_id=_OWNER)]

    assert streamed == [
        ApplicationRunEvent(
            sequence=event.sequence,
            event_type=event.event_type,
            payload=event.payload,
            created_at=event.created_at,
        )
    ]
    assert len(store.created) == 1
    assert coordinator.subscriptions == [(_OWNER, "run-1", 0)]


def _reference(
    *,
    kind: str,
    ordinal: int,
    digest: str,
    filename: str,
    mime_type: str = "text/plain",
) -> RunArtifactReference:
    return RunArtifactReference(
        resource_id=f"{kind}-{ordinal}",
        reference_kind=kind,  # type: ignore[arg-type]
        ordinal=ordinal,
        digest=digest,
        filename=filename,
        mime_type=mime_type,
        transform_locator={},
        created_at=_NOW,
    )


async def test_read_input_artifact_returns_accepted_input_metadata_and_bytes() -> None:
    store = _Store(
        references=(
            _reference(kind="current_attachment", ordinal=0, digest="d0", filename="a.txt"),
            _reference(kind="history_attachment", ordinal=1, digest="d1", filename="old.txt"),
        ),
        blobs={"d0": b"current", "d1": b"history"},
    )
    service = _service(store=store)

    current = await service.read_input_artifact(owner_id=_OWNER, run_id="run-1", ordinal=0)
    history = await service.read_input_artifact(owner_id=_OWNER, run_id="run-1", ordinal=1)

    assert current is not None
    assert (current.reference_kind, current.filename, current.content) == (
        "current_attachment",
        "a.txt",
        b"current",
    )
    assert history is not None
    assert (history.reference_kind, history.filename, history.content) == (
        "history_attachment",
        "old.txt",
        b"history",
    )


async def test_read_input_artifact_never_serves_fetched_research_artifacts() -> None:
    store = _Store(
        references=(
            _reference(kind="fetched_resource", ordinal=0, digest="d0", filename="w.html"),
        ),
        blobs={"d0": b"fetched"},
    )
    service = _service(store=store)

    assert await service.read_input_artifact(owner_id=_OWNER, run_id="run-1", ordinal=0) is None
    assert store.artifact_reads == []


async def test_read_input_artifact_is_owner_scoped() -> None:
    store = _Store(
        references=(
            _reference(kind="current_attachment", ordinal=0, digest="d0", filename="a.txt"),
        ),
        blobs={"d0": b"current"},
    )
    service = _service(store=store)

    assert await service.read_input_artifact(owner_id="intruder", run_id="run-1", ordinal=0) is None
    assert store.artifact_reads == []


async def test_retrieval_id_is_unknown_to_every_answer_only_interface() -> None:
    store = _Store(
        run=_record(
            run_kind="retrieval",
            status="succeeded",
            result={"answer": "fabricated retrieval answer"},
            accepted_input={
                "query": "retrieval query",
                "workspaces": ["finance"],
                "agent_session_id": "0199a0a0-0000-7000-8000-000000000099",
            },
        ),
        references=(
            _reference(kind="published_artifact", ordinal=0, digest="d0", filename="report.md"),
        ),
        blobs={"d0": b"retrieval bytes"},
    )
    store.transcript_rows = ({"role": "assistant", "content": "fabricated"},)
    store.child_page_rows = ({"child_session_id": "fabricated"},)
    service = _service(store=store)
    service.create = AsyncMock()  # type: ignore[method-assign]

    assert await service.steer(owner_id=_OWNER, run_id="run-1", instruction="continue") is None
    assert await service.transcript_tail(owner_id=_OWNER, run_id="run-1") is None
    assert await service.children(owner_id=_OWNER, run_id="run-1") is None
    assert await service.continuation_workspaces(owner_id=_OWNER, run_id="run-1") is None
    assert await service.list_artifacts(owner_id=_OWNER, run_id="run-1") is None
    assert (
        await service.read_artifact(
            owner_id=_OWNER, run_id="run-1", resource_id="published_artifact-0"
        )
        is None
    )
    assert (
        await service.open_artifact(
            owner_id=_OWNER, run_id="run-1", resource_id="published_artifact-0"
        )
        is None
    )
    assert (
        await service.artifact_size(
            owner_id=_OWNER, run_id="run-1", resource_id="published_artifact-0"
        )
        is None
    )
    assert await service.read_input_artifact(owner_id=_OWNER, run_id="run-1", ordinal=0) is None
    assert (
        await service.follow_up(
            owner_id=_OWNER,
            run_id="run-1",
            query="fabricate answer",
            authorized_workspaces=("finance",),
        )
        is None
    )
    assert (
        await service.fork(
            owner_id=_OWNER,
            run_id="run-1",
            query="fabricate branch",
            authorized_workspaces=("finance",),
        )
        is None
    )
    service.create.assert_not_awaited()
    assert store.controls == []
    assert store.artifact_reads == []


async def test_capabilities_exposes_the_public_immutable_snapshot() -> None:
    snapshot = AnswerCapabilities(answer=None, vlm_status="supported")
    view = _CapabilityView(snapshot)
    service = _service(capability_view=view)

    assert await service.capabilities() is snapshot
    assert view.reads == 1


async def test_agent_controls_share_ordered_service_interface() -> None:
    store = _Store(run=_record(status="running"))
    service = _service(store=store)

    first = await service.steer(owner_id=_OWNER, run_id="run-1", instruction="focus on risks")
    second = await service.steer(owner_id=_OWNER, run_id="run-1", instruction="compare dates")

    assert first is not None and first.control_sequence == 1
    assert second is not None and second.control_sequence == 2
    assert [item["content"] for item in store.controls] == ["focus on risks", "compare dates"]


async def test_transcript_and_child_roster_are_owner_scoped() -> None:
    record = _record(
        status="succeeded",
        result={"answer": "final answer"},
        accepted_input={
            "query": "parent question",
            "workspaces": ["finance"],
            "agent_session_id": "0199a0a0-0000-7000-8000-000000000099",
            "agent_lane_id": "main",
        },
    )
    store = _Store(run=record)
    store.transcript_rows = (
        {"role": "assistant", "content": "checking", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "found"},
        {"role": "user", "content": "steer: compare dates"},
    )
    store.child_rows = ({"child_session_id": "child-1", "status": "succeeded"},)
    service = _service(store=store)

    transcript = await service.transcript_tail(owner_id=_OWNER, run_id="run-1")
    store.child_page_rows = store.child_rows
    children = await service.children(owner_id=_OWNER, run_id="run-1")

    assert transcript is not None
    assert transcript.messages == store.transcript_rows
    assert children is not None
    assert children.children == store.child_rows
    assert children.next_cursor is None
    assert await service.children(owner_id="other", run_id="run-1") is None


async def test_child_roster_page_derives_a_run_bound_cursor_from_the_last_row() -> None:
    run_id = "0199a0a0-0000-7000-8000-000000000077"
    child_id = "0199a0a0-0000-7000-8000-000000000088"
    store = _Store(run=_record(run_id=run_id, status="succeeded"))
    store.child_page_rows = (
        {
            "child_session_id": child_id,
            "created_at": datetime.datetime(2026, 3, 4, 5, 6, 7, 123456, tzinfo=datetime.UTC),
            "status": "succeeded",
        },
    )
    store.child_page_has_more = True
    service = _service(store=store)

    page = await service.children(
        owner_id=_OWNER, run_id=run_id, page=ChildRosterPageRequest(limit=1)
    )

    assert page is not None
    assert page.next_cursor is not None
    assert str(page.next_cursor.run_id) == run_id
    assert str(page.next_cursor.child_session_id) == child_id
    assert page.next_cursor.created_at == datetime.datetime(
        2026, 3, 4, 5, 6, 7, 123456, tzinfo=datetime.UTC
    )
    token = service.child_roster_cursor_codec.encode(page.next_cursor)
    assert service.child_roster_cursor_codec.decode(token) == page.next_cursor


async def test_child_roster_rejects_a_cursor_from_another_run_before_storage() -> None:
    from uuid import UUID as _UUID

    run_id = "0199a0a0-0000-7000-8000-000000000077"
    other_run = "0199a0a0-0000-7000-8000-000000000078"
    store = _Store(run=_record(run_id=run_id, status="succeeded"))
    service = _service(store=store)
    cursor = ChildRosterCursor(
        run_id=_UUID(other_run),
        created_at=datetime.datetime(2026, 3, 4, tzinfo=datetime.UTC),
        child_session_id=_UUID("0199a0a0-0000-7000-8000-000000000088"),
    )

    with pytest.raises(ValueError, match="another run"):
        await service.children(
            owner_id=_OWNER,
            run_id=run_id,
            page=ChildRosterPageRequest(cursor=cursor),
        )


async def test_child_roster_empty_page_has_no_continuation() -> None:
    service = _service(store=_Store(run=_record(status="succeeded")))

    page = await service.children(owner_id=_OWNER, run_id="run-1")

    assert page is not None
    assert page.children == ()
    assert page.next_cursor is None


async def test_continuation_content_limit_is_transport_neutral() -> None:
    terminal = _record(
        status="succeeded",
        accepted_input={"query": "parent", "workspaces": ["finance"]},
    )
    service = _service(store=_Store(run=terminal))

    with pytest.raises(ValueError, match="20000"):
        await service.continuation_request(
            owner_id=_OWNER,
            run_id="run-1",
            query="x" * 20_001,
            include_answer=True,
            authorized_workspaces=("finance",),
        )


async def test_follow_up_and_fork_reenter_one_acceptance_interface() -> None:
    terminal = _record(
        status="succeeded",
        result={"answer": "parent answer"},
        accepted_input={
            "query": "parent question",
            "workspaces": ["finance"],
            "history": [
                {"role": "user", "content": "ancestor question"},
                {"role": "assistant", "content": "ancestor answer"},
            ],
            "episodic_summary": "Older accepted context.",
            "top_k": 7,
            "chunk_top_k": 11,
            "filters": {"author": "Ada"},
            "semantic_highlights": True,
            "history_attachments": [
                {
                    "ordinal": 0,
                    "digest": "a" * 64,
                    "filename": "ancestor.txt",
                    "mime_type": "text/plain",
                    "byte_size": 8,
                }
            ],
            "attachments": [
                {
                    "ordinal": 0,
                    "digest": "b" * 64,
                    "filename": "parent.txt",
                    "mime_type": "text/plain",
                    "byte_size": 6,
                }
            ],
            "mode": "research",
            "agent_session_id": "0199a0a0-0000-7000-8000-000000000099",
            "agent_lane_id": "main",
        },
    )
    service = _service(store=_Store(run=terminal))
    created = RunCreation(run=_record(run_id="next"), replayed=False)
    service.create = AsyncMock(return_value=created)  # type: ignore[method-assign]

    follow = await service.follow_up(
        owner_id=_OWNER,
        run_id="run-1",
        query="next question",
        authorized_workspaces=("finance",),
    )
    assert service.create.await_args is not None
    follow_request = service.create.await_args.kwargs["request"]
    fork = await service.fork(
        owner_id=_OWNER,
        run_id="run-1",
        query="other branch",
        authorized_workspaces=("finance",),
    )
    assert service.create.await_args is not None
    fork_request = service.create.await_args.kwargs["request"]

    assert follow == created and fork == created
    assert follow_request.history == (
        {"role": "user", "content": "ancestor question"},
        {"role": "assistant", "content": "ancestor answer"},
        {"role": "user", "content": "parent question"},
        {"role": "assistant", "content": "parent answer"},
    )
    assert fork_request.history == (
        {"role": "user", "content": "ancestor question"},
        {"role": "assistant", "content": "ancestor answer"},
        {"role": "user", "content": "parent question"},
    )
    assert follow_request.episodic_summary == "Older accepted context."
    assert follow_request.retrieval.top_k == 7
    assert follow_request.retrieval.chunk_top_k == 11
    assert follow_request.semantic_highlights is True
    assert [item.reference_kind for item in follow_request.history_resources] == [
        "history_attachment",
        "current_attachment",
    ]
    assert follow_request.workspaces == ("finance",)
    assert follow_request.parent_run_id == "run-1"
    assert follow_request.continuation_kind == "follow_up"
    assert fork_request.parent_run_id == "run-1"
    assert fork_request.continuation_kind == "fork"
    assert follow_request.agent_session_id == "0199a0a0-0000-7000-8000-000000000099"
    assert follow_request.agent_lane_id == "main"
    assert follow_request.source_lane_id is None
    assert fork_request.agent_session_id == follow_request.agent_session_id
    assert fork_request.agent_lane_id != "main"
    assert fork_request.source_lane_id == "main"
