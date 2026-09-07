# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for the shared raw Retrieval execution seam."""

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest

from dlightrag.application.retrieval import (
    CorpusUnavailableError,
    ProjectedRetrieval,
    RetrievalService,
    RetrievalSettings,
    RetrieveRequest,
)
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.telemetry import NoopTelemetry
from dlightrag.engine.rag.retrieval import MetadataFilter, RetrievalOptions, RetrievalResult
from dlightrag.engine.rag.retrieval.runtime import RetrievalPlannerRuntime
from dlightrag.engine.rag.workspace.pool import WorkspacePool
from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag


class _Planner:
    def __init__(self, plan=None) -> None:
        self._plan = plan

    async def plan(self, query: str, **_kwargs):
        return self._plan or SimpleNamespace(
            standalone_query=query,
            metadata_filter=None,
            metadata_filter_source=None,
            bm25_query=None,
            outcome="planned",
        )


class _Planners:
    def __init__(self, planner=None) -> None:
        self._planner = planner or _Planner()

    def planner_for(self, model_profile: Any | None = None) -> Any:
        del model_profile
        return self._planner

    async def aclose(self) -> None:
        return None


async def _execute_request(service: RetrievalService, request: RetrieveRequest) -> RetrievalResult:
    """Exercise the shared raw stage without restoring the removed inline API."""
    if len(request.query_images) > service._settings.query_image_limit:
        raise ValueError(
            f"at most {service._settings.query_image_limit} current images are allowed"
        )
    service.warm(request.workspaces)
    images = tuple(dict(image) for image in request.query_images)
    descriptions = await service.prepare_query_images(images)
    return await service.retrieve_result(
        request.query,
        workspaces=request.workspaces,
        retrieval=RetrievalOptions(
            top_k=request.top_k,
            chunk_top_k=request.chunk_top_k,
            federated_rerank=request.federated_rerank,
        ),
        bm25_query=request.bm25_query,
        filters=request.filters,
        query_images=images,
        image_descriptions=descriptions,
    )


async def test_planner_history_measure_uses_execution_schema_images_and_mode() -> None:
    planner = Mock()
    measure = Mock(return_value=17)
    planner.history_input_measure.return_value = measure
    schema_lookup = AsyncMock(return_value={"author": {"type": "string"}})
    service = RetrievalService(
        pool=AsyncMock(),
        planners=_Planners(planner),
        schema_lookup=schema_lookup,
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )
    profile = ModelProfile(context_window_tokens=64_000)

    returned = await service.planner_history_input_measure(
        query="report",
        workspaces=("finance",),
        model_profile=profile,
        current_image_descriptions=("Image 1: chart",),
        preserve_query=None,
    )

    assert returned is measure
    planner.history_input_measure.assert_called_once_with(
        "report",
        schema={"author": {"type": "string"}},
        current_image_descriptions=["Image 1: chart"],
        preserve_query=None,
    )


async def test_explicit_filters_and_bm25_override_planner_inference() -> None:
    inferred = MetadataFilter(author="Planner")
    explicit = MetadataFilter(author="Caller")
    planner = _Planner(
        SimpleNamespace(
            standalone_query="standalone",
            metadata_filter=inferred,
            metadata_filter_source="llm_inferred",
            bm25_query="inferred lexical",
            outcome="planned",
        )
    )
    runtime = AsyncMock()
    runtime.aretrieve.return_value = RetrievalResult()
    pool = AsyncMock()
    pool.acquire.return_value = runtime
    projector = Mock()
    projector.return_value = ProjectedRetrieval(contexts={}, sources=())
    service = RetrievalService(
        pool=pool,
        planners=_Planners(planner),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=projector,
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    await _execute_request(
        service,
        RetrieveRequest(
            query="report",
            workspaces=("finance",),
            top_k=9,
            chunk_top_k=6,
            bm25_query="caller lexical",
            filters=explicit,
        ),
    )

    assert runtime.aretrieve.await_args.args == ("standalone",)
    kwargs = runtime.aretrieve.await_args.kwargs
    assert kwargs["top_k"] == 9
    assert kwargs["chunk_top_k"] == 6
    assert kwargs["filters"] is explicit
    assert kwargs["filter_source"] == "explicit"
    assert kwargs["bm25_query"] == "caller lexical"


async def test_retrieve_projects_lightrag_mix_trace_vocabulary() -> None:
    runtime = AsyncMock()
    runtime.aretrieve.return_value = RetrievalResult(trace={"lightrag_mix_chunk_count": 2})
    pool = AsyncMock()
    pool.acquire.return_value = runtime
    projector = Mock(
        return_value=ProjectedRetrieval(
            contexts={"chunks": [], "entities": [], "relationships": []},
            sources=(),
        )
    )
    service = RetrievalService(
        pool=pool,
        planners=_Planners(),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=projector,
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    response = await _execute_request(
        service, RetrieveRequest(query="report", workspaces=("finance",))
    )

    assert response.trace == {"lightrag_mix_chunk_count": 2, "query_image_description_count": 0}
    assert "semantic_chunk_count" not in response.trace


async def test_schema_cache_is_set_keyed_bounded_and_uses_stale_on_refresh_failure() -> None:
    now = 0.0
    lookup = AsyncMock(return_value={"revision": "current"})
    service = RetrievalService(
        pool=AsyncMock(),
        planners=_Planners(),
        schema_lookup=lookup,
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
        clock=lambda: now,
    )

    first = await service.schema_for(("legal", "finance"))
    reordered = await service.schema_for(("finance", "legal"))
    assert first == reordered == {"revision": "current"}
    lookup.assert_awaited_once_with(("finance", "legal"))

    now = 16.0
    lookup.side_effect = RuntimeError("unavailable")
    assert await service.schema_for(("legal", "finance")) == {"revision": "current"}

    lookup.side_effect = None
    lookup.return_value = {}
    for index in range(129):
        await service.schema_for((f"workspace_{index}",))
    lookup.reset_mock()
    await service.schema_for(("workspace_0",))
    lookup.assert_awaited_once_with(("workspace_0",))


async def test_cold_schema_failure_is_retried_and_recovered() -> None:
    lookup = AsyncMock(side_effect=RuntimeError("database unavailable"))
    service = RetrievalService(
        pool=AsyncMock(),
        planners=_Planners(),
        schema_lookup=lookup,
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    assert await service.schema_for(("reports",)) == {}
    lookup.side_effect = None
    lookup.return_value = {"custom_keys": ["department"]}

    assert await service.schema_for(("reports",)) == {"custom_keys": ["department"]}
    assert lookup.await_count == 2


async def test_schema_lookup_is_single_flight_and_cache_is_not_mutable_by_callers() -> None:
    release = asyncio.Event()
    lookup_started = asyncio.Event()

    async def lookup(_workspaces):
        lookup_started.set()
        await release.wait()
        return {"custom_keys": ["department"]}

    service = RetrievalService(
        pool=AsyncMock(),
        planners=_Planners(),
        schema_lookup=lookup,
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    first = asyncio.create_task(service.schema_for(("reports",)))
    second = asyncio.create_task(service.schema_for(("reports",)))
    await lookup_started.wait()
    release.set()
    first_schema, second_schema = await asyncio.gather(first, second)

    assert first_schema == second_schema == {"custom_keys": ["department"]}
    first_schema["poisoned"] = True
    first_schema["custom_keys"].append("poisoned")
    assert await service.schema_for(("reports",)) == {"custom_keys": ["department"]}


async def test_schema_ttl_starts_when_successful_refresh_enters_cache() -> None:
    now = 0.0

    async def slow_lookup(_workspaces):
        nonlocal now
        now = 14.0
        return {"revision": "current"}

    lookup = AsyncMock(side_effect=slow_lookup)
    service = RetrievalService(
        pool=AsyncMock(),
        planners=_Planners(),
        schema_lookup=lookup,
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
        clock=lambda: now,
    )

    assert await service.schema_for(("reports",)) == {"revision": "current"}
    now = 15.0
    assert await service.schema_for(("reports",)) == {"revision": "current"}
    lookup.assert_awaited_once()


async def test_raw_retrieval_uses_history_profile_without_inline_projection_or_timeout() -> None:
    profile = ModelProfile(context_window_tokens=10_000)
    planner = AsyncMock()
    planner.plan.return_value = SimpleNamespace(
        standalone_query="standalone",
        metadata_filter=None,
        metadata_filter_source=None,
        bm25_query=None,
        outcome="planned",
    )
    planners = Mock()
    planners.planner_for.return_value = planner
    planners.aclose = AsyncMock()
    runtime = AsyncMock()
    runtime.aretrieve.return_value = RetrievalResult()
    pool = AsyncMock()
    pool.acquire.return_value = runtime
    projector = Mock()
    projector.side_effect = AssertionError("raw retrieval must not project")
    service = RetrievalService(
        pool=pool,
        planners=planners,
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(side_effect=AssertionError("images are already prepared")),
        projector=projector,
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=0.0001,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )
    history = ({"role": "user", "content": "earlier"},)

    result = await service.retrieve_result(
        "query",
        workspaces=("reports",),
        conversation_history=history,
        image_descriptions=("Image 1: chart",),
        preserve_query=True,
        model_profile=profile,
    )

    assert result is runtime.aretrieve.return_value
    planners.planner_for.assert_called_once_with(profile)
    assert planner.plan.await_args is not None
    assert planner.plan.await_args.kwargs["conversation_history"] == history
    assert planner.plan.await_args.kwargs["preserve_query"] is True
    projector.assert_not_called()


async def test_requested_workspace_schema_is_passed_to_planner() -> None:
    schemas = {
        ("reports",): {"custom_keys": ["department"]},
        ("legal",): {"custom_keys": ["jurisdiction"]},
    }
    lookup = AsyncMock(side_effect=lambda workspaces: schemas[workspaces])
    planner = AsyncMock()
    planner.plan.return_value = SimpleNamespace(
        standalone_query="query",
        metadata_filter=None,
        metadata_filter_source=None,
        bm25_query=None,
        outcome="planned",
    )
    runtime = AsyncMock()
    runtime.aretrieve.return_value = RetrievalResult()
    pool = AsyncMock()
    pool.acquire.return_value = runtime
    projector = Mock()
    projector.return_value = ProjectedRetrieval(contexts={}, sources=())
    service = RetrievalService(
        pool=pool,
        planners=_Planners(planner),
        schema_lookup=lookup,
        image_preparer=AsyncMock(return_value=[]),
        projector=projector,
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    await _execute_request(service, RetrieveRequest(query="query", workspaces=("reports",)))
    await _execute_request(service, RetrieveRequest(query="query", workspaces=("legal",)))

    assert planner.plan.await_args_list[0].kwargs["schema"] == schemas[("reports",)]
    assert planner.plan.await_args_list[1].kwargs["schema"] == schemas[("legal",)]


async def test_retrieve_starts_workspace_warmup_before_planning() -> None:
    warm_started = asyncio.Event()
    release_warm = asyncio.Event()
    plan_started = asyncio.Event()

    async def warm(_workspaces) -> None:
        warm_started.set()
        await release_warm.wait()

    async def plan(query: str, **_kwargs):
        await warm_started.wait()
        plan_started.set()
        return SimpleNamespace(
            standalone_query=query,
            metadata_filter=None,
            metadata_filter_source=None,
            bm25_query=None,
            outcome="planned",
        )

    planner = AsyncMock()
    planner.plan.side_effect = plan
    runtime = AsyncMock()
    runtime.aretrieve.return_value = RetrievalResult()
    pool = AsyncMock()
    pool.warm.side_effect = warm
    pool.acquire.return_value = runtime
    projector = Mock()
    projector.return_value = ProjectedRetrieval(contexts={}, sources=())
    service = RetrievalService(
        pool=pool,
        planners=_Planners(planner),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=projector,
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    task = asyncio.create_task(
        _execute_request(service, RetrieveRequest(query="query", workspaces=("reports",)))
    )
    try:
        await plan_started.wait()
    finally:
        release_warm.set()

    await task
    pool.warm.assert_awaited_once_with(("reports",))


async def test_one_thousand_identical_service_warms_share_one_outer_task() -> None:
    warm_started = asyncio.Event()
    release_warm = asyncio.Event()

    async def warm(_workspaces) -> None:
        warm_started.set()
        await release_warm.wait()

    pool = AsyncMock()
    pool.warm.side_effect = warm
    service = RetrievalService(
        pool=pool,
        planners=_Planners(),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    for index in range(1000):
        workspaces = ("legal", "reports", "reports") if index % 2 else ("reports", "legal")
        service.warm(workspaces)
    await warm_started.wait()

    pool.warm.assert_awaited_once_with(("legal", "reports"))
    assert len(service._warmups) == 1

    release_warm.set()
    while service._warmups:
        await asyncio.sleep(0)


async def test_coalesced_warm_failure_is_observed_once() -> None:
    release_warm = asyncio.Event()

    async def warm(_workspaces) -> None:
        await release_warm.wait()
        raise RuntimeError("warm failed")

    pool = AsyncMock()
    pool.warm.side_effect = warm
    service = RetrievalService(
        pool=pool,
        planners=_Planners(),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    with patch("dlightrag.application.retrieval.service.logger.debug") as log_debug:
        for _ in range(1000):
            service.warm(("reports",))
        release_warm.set()
        while service._warmups:
            await asyncio.sleep(0)

    pool.warm.assert_awaited_once_with(("reports",))
    log_debug.assert_called_once()
    assert log_debug.call_args.args[0] == "Workspace warm-up failed"


async def test_service_close_cancels_only_its_waiter_not_the_pool_flight() -> None:
    runtime = cast(WorkspaceRag, AsyncMock())
    build_started = asyncio.Event()
    release_build = asyncio.Event()
    build_cancelled = False

    async def build(*_args: Any) -> WorkspaceRag:
        nonlocal build_cancelled
        build_started.set()
        try:
            await release_build.wait()
        except asyncio.CancelledError:
            build_cancelled = True
            raise
        return runtime

    pool = WorkspacePool(build=build)
    service = RetrievalService(
        pool=pool,
        planners=_Planners(),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    service.warm(("reports",))
    await build_started.wait()
    other_consumer = asyncio.create_task(pool.acquire("reports"))
    await asyncio.sleep(0)
    await service.aclose()

    assert not build_cancelled
    assert not other_consumer.done()

    release_build.set()
    assert await other_consumer is runtime
    await pool.aclose()
    cast(AsyncMock, runtime.aclose).assert_awaited_once()


async def test_close_cancels_warmups_and_closed_service_starts_no_new_warmup() -> None:
    warm_started = asyncio.Event()
    warm_cancelled = asyncio.Event()

    async def warm(_workspaces) -> None:
        warm_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            warm_cancelled.set()

    pool = AsyncMock()
    pool.warm.side_effect = warm
    planners = _Planners()
    service = RetrievalService(
        pool=pool,
        planners=planners,
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    service.warm(("reports",))
    await warm_started.wait()
    await service.aclose()
    await warm_cancelled.wait()
    service.warm(("legal",))

    pool.warm.assert_awaited_once_with(("reports",))

    request = RetrieveRequest(query="closed", workspaces=("reports",))
    with pytest.raises(CorpusUnavailableError, match="Retrieval service is closed"):
        await _execute_request(service, request)
    with pytest.raises(CorpusUnavailableError, match="Retrieval service is closed"):
        await service.retrieve_result("closed", workspaces=("reports",))
    with pytest.raises(CorpusUnavailableError, match="Retrieval service is closed"):
        service.planner_for()


async def test_concurrent_service_close_callers_join_the_same_cleanup() -> None:
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    planners = _Planners()

    async def close_planners() -> None:
        close_started.set()
        await release_close.wait()

    planners.aclose = close_planners  # type: ignore[method-assign]
    service = RetrievalService(
        pool=AsyncMock(),
        planners=planners,
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    first = asyncio.create_task(service.aclose())
    await close_started.wait()
    second = asyncio.create_task(service.aclose())
    await asyncio.sleep(0)
    assert not second.done()
    release_close.set()

    await asyncio.gather(first, second)


async def test_visual_disabled_does_not_receive_raw_query_images() -> None:
    images = ({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},)
    image_preparer = AsyncMock(return_value=["Image 1: chart"])
    runtime = AsyncMock()
    runtime.aretrieve.return_value = RetrievalResult()
    pool = AsyncMock()
    pool.acquire.return_value = runtime
    projector = Mock()
    projector.return_value = ProjectedRetrieval(contexts={}, sources=())
    service = RetrievalService(
        pool=pool,
        planners=_Planners(),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=image_preparer,
        projector=projector,
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=1,
        ),
        telemetry=NoopTelemetry(),
    )

    response = await _execute_request(
        service,
        RetrieveRequest(
            query="query",
            workspaces=("reports",),
            query_images=images,
        ),
    )

    image_preparer.assert_awaited_once_with(images)
    assert "query_image_blocks" not in runtime.aretrieve.await_args.kwargs
    assert "prepared_visual_query" not in runtime.aretrieve.await_args.kwargs
    assert response.image_descriptions == ["Image 1: chart"]
    with pytest.raises(ValueError, match="at most 1 current images"):
        await _execute_request(
            service,
            RetrieveRequest(
                query="query",
                workspaces=("reports",),
                query_images=images * 2,
            ),
        )


async def test_multiple_workspaces_use_federated_retrieval() -> None:
    pool = AsyncMock()
    projector = Mock()
    projector.return_value = ProjectedRetrieval(contexts={}, sources=())
    service = RetrievalService(
        pool=pool,
        planners=_Planners(),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=projector,
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
            workspace_fanout_concurrency=3,
        ),
        telemetry=NoopTelemetry(),
    )

    with patch(
        "dlightrag.application.retrieval.service.federated_retrieve",
        new=AsyncMock(return_value=RetrievalResult()),
    ) as federated:
        await _execute_request(
            service,
            RetrieveRequest(
                query="query",
                workspaces=("reports", "legal"),
            ),
        )

    federated.assert_awaited_once()
    assert federated.await_args is not None
    assert federated.await_args.args[:2] == ("query", ["reports", "legal"])
    acquire = federated.await_args.args[2]
    assert acquire is not pool.acquire  # the service wraps acquire to translate pool errors
    assert federated.await_args.kwargs["max_concurrency"] == 3
    policy = federated.await_args.kwargs["policy"]
    assert policy.chunk_top_k == 5
    assert policy.min_chunks_per_workspace == 7
    assert federated.await_args.kwargs["reranker"] is None
    pool.acquire.assert_not_awaited()


_FEDERATION_SETTINGS = RetrievalSettings(
    default_top_k=8,
    default_chunk_top_k=5,
    timeout_seconds=30,
    query_image_limit=4,
)


def _federated_service(
    *,
    factory: Any = None,
) -> RetrievalService:
    """One federation-test service with a mock pool, projector, and planner."""
    projector = Mock()
    projector.return_value = ProjectedRetrieval(contexts={}, sources=())
    return RetrievalService(
        pool=AsyncMock(),
        planners=_Planners(),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=projector,
        settings=_FEDERATION_SETTINGS,
        telemetry=NoopTelemetry(),
        federated_reranker_factory=factory,
    )


@contextmanager
def _patched_federated_retrieve(trace: Mapping[str, Any] | None = None) -> Iterator[Any]:
    """Patch the service's federated_retrieve import; yields the mock."""
    with patch(
        "dlightrag.application.retrieval.service.federated_retrieve",
        new=AsyncMock(return_value=RetrievalResult(trace=dict(trace or {}))),
    ) as federated:
        yield federated


def _flagged_request(
    chunk_top_k: int | None = None,
    *,
    federated_rerank: bool = True,
) -> RetrieveRequest:
    return RetrieveRequest(
        query="query",
        workspaces=("reports", "legal"),
        chunk_top_k=chunk_top_k,
        federated_rerank=federated_rerank,
    )


async def test_federated_policy_reuses_the_requested_chunk_budget() -> None:
    service = _federated_service()

    with _patched_federated_retrieve() as federated:
        await _execute_request(service, _flagged_request(chunk_top_k=3, federated_rerank=False))

    assert federated.await_args is not None
    policy = federated.await_args.kwargs["policy"]
    assert policy.chunk_top_k == 3


async def test_federated_rerank_flag_resolves_the_injected_reranker_once() -> None:
    reranker = AsyncMock()
    factory = Mock(return_value=reranker)
    service = _federated_service(factory=factory)

    with _patched_federated_retrieve() as federated:
        for _ in range(2):
            await _execute_request(service, _flagged_request())

    factory.assert_called_once()
    assert all(call.kwargs["reranker"] is reranker for call in federated.await_args_list)


async def test_federated_rerank_flag_without_a_reranker_marks_unavailable() -> None:
    service = _federated_service()

    with _patched_federated_retrieve(trace={"federated": True}) as federated:
        response = await _execute_request(service, _flagged_request())

    assert federated.await_args is not None
    assert federated.await_args.kwargs["reranker"] is None
    assert response.trace["federated_rerank_unavailable"] is True


async def test_federated_rerank_factory_failure_is_remembered_not_retried() -> None:
    factory = Mock(side_effect=RuntimeError("build boom"))
    service = _federated_service(factory=factory)

    with _patched_federated_retrieve(trace={"federated": True}):
        for _ in range(2):
            await _execute_request(service, _flagged_request())

    factory.assert_called_once()


async def test_federated_rerank_factory_returning_none_marks_unavailable_once() -> None:
    """rerank.enabled=false makes the factory yield None; the flag degrades to
    the default path with an unavailable marker, memoized across requests."""
    factory = Mock(return_value=None)
    service = _federated_service(factory=factory)

    with _patched_federated_retrieve(trace={"federated": True}) as federated:
        for _ in range(2):
            await _execute_request(service, _flagged_request())

    factory.assert_called_once()
    assert all(call.kwargs["reranker"] is None for call in federated.await_args_list)


async def test_planner_runtime_caches_by_profile_and_closes_its_model_once() -> None:
    default_profile = ModelProfile(context_window_tokens=200_000)
    pinned_profile = ModelProfile(context_window_tokens=125_000)
    model_settings = Mock()
    scheduler = Mock()
    model = AsyncMock()

    with patch(
        "dlightrag.engine.rag.retrieval.runtime.CompletionModel",
        return_value=model,
    ) as create_model:
        runtime = RetrievalPlannerRuntime(
            model_settings=model_settings,
            default_profile=lambda: default_profile,
            scheduler=scheduler,
            telemetry=NoopTelemetry(),
        )
        default_planner = runtime.planner_for()
        pinned_planner = runtime.planner_for(pinned_profile)

        assert runtime.planner_for() is default_planner
        assert runtime.planner_for(pinned_profile) is pinned_planner
        assert pinned_planner is not default_planner
        create_model.assert_called_once_with(
            model_settings,
            scheduler=scheduler,
            telemetry=ANY,
        )

        await runtime.aclose()
        await runtime.aclose()

    model.aclose.assert_awaited_once()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.planner_for()


async def test_concurrent_planner_runtime_close_callers_join_model_cleanup() -> None:
    default_profile = ModelProfile(context_window_tokens=200_000)
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    model = AsyncMock()

    async def close_model() -> None:
        close_started.set()
        await release_close.wait()

    model.aclose.side_effect = close_model
    with patch("dlightrag.engine.rag.retrieval.runtime.CompletionModel", return_value=model):
        runtime = RetrievalPlannerRuntime(
            model_settings=Mock(),
            default_profile=lambda: default_profile,
            scheduler=Mock(),
            telemetry=NoopTelemetry(),
        )
        runtime.planner_for()
        first = asyncio.create_task(runtime.aclose())
        await close_started.wait()
        second = asyncio.create_task(runtime.aclose())
        await asyncio.sleep(0)
        assert not second.done()
        release_close.set()

        await asyncio.gather(first, second)

    model.aclose.assert_awaited_once()
