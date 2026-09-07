# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable top-level Retrieval and the shared raw Retrieval Stage."""

import asyncio
import copy
import hashlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, aclosing
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import uuid7

from dlightrag.application.errors import CorpusUnavailableError
from dlightrag.application.runs import (
    IdempotencyKeyConflict,
    RunCancelledError,
    RunCapacityExceededError,
    RunCreation,
    RunEvent,
    RunFailedError,
    RunRuntimeUnavailableError,
)
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.settings import ModelRole
from dlightrag.engine.ai.telemetry import Telemetry
from dlightrag.engine.rag.retrieval import (
    MetadataFilter,
    RetrievalContexts,
    RetrievalOptions,
    RetrievalResult,
)
from dlightrag.engine.rag.retrieval.federation import (
    FederatedReranker,
    FederationMergePolicy,
    WorkspaceRetriever,
    federated_retrieve,
)
from dlightrag.engine.rag.retrieval.planner import RetrievalPlan, RetrievalPlanner
from dlightrag.engine.rag.retrieval.visual import PreparedVisualQuery, VisualEmbeddingDomain
from dlightrag.engine.rag.workspace.lifecycle import await_shared_cleanup
from dlightrag.engine.rag.workspace.pool import WorkspacePool
from dlightrag.engine.rag.workspace.ports import (
    CorpusUnavailableError as _EngineCorpusUnavailableError,
)
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id
from dlightrag.engine.runtime import (
    RETRIEVAL_RUN_RETENTION_SECONDS,
    PreparedInputTooLargeError,
    PreparedRunEnvelope,
    RunAccessScope,
    RunKind,
    RunRecord,
    require_prepared_input_bounds,
    run_request_fingerprint,
)
from dlightrag.engine.runtime import (
    IdempotencyKeyConflict as RuntimeIdempotencyKeyConflict,
)
from dlightrag.engine.runtime import (
    RunCapacityExceededError as RuntimeRunCapacityExceededError,
)
from dlightrag.engine.runtime import (
    RunCreation as RuntimeRunCreation,
)
from dlightrag.engine.runtime import (
    RunEvent as RuntimeRunEvent,
)

from .execution import PinnedRetrievalModel, RetrievalRunInput, restore_retrieval_result

logger = logging.getLogger(__name__)

type SchemaLookup = Callable[[Sequence[str]], Awaitable[dict[str, Any]]]
type QueryImagePreparer = Callable[[Sequence[Mapping[str, Any]]], Awaitable[list[str]]]
type RetrievalProjection = Callable[[RetrievalResult, "RetrieveProjection"], "ProjectedRetrieval"]
type _VisualPreparationStatus = Literal["cache_hit", "singleflight_hit", "started"]

_VISUAL_QUERY_CACHE_SIZE = 32


@dataclass(frozen=True, slots=True)
class _VisualQueryCacheKey:
    domain: VisualEmbeddingDomain
    payload_digest: bytes


class RetrievalTimeoutError(RuntimeError):
    """Legacy name for a terminal Retrieval timeout classification."""


class RetrievalInputError(ValueError):
    """A top-level Retrieval request failed pre-acceptance validation."""


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    default_top_k: int
    default_chunk_top_k: int
    timeout_seconds: float
    query_image_limit: int
    federation_min_chunks_per_workspace: int = 7
    workspace_fanout_concurrency: int = 8


@dataclass(frozen=True, slots=True)
class RetrieveProjection:
    """Already-authorized reader scope for client-safe projection."""

    downloadable_workspaces: frozenset[str] | None
    visual_workspaces: frozenset[str] | None
    include_download_links: bool = False
    image_url_prefix: str | None = "/images"


@dataclass(frozen=True, slots=True)
class RetrieveRequest:
    """One authorized Retrieval request with concrete canonical workspaces."""

    query: str
    workspaces: tuple[str, ...]
    top_k: int | None = None
    chunk_top_k: int | None = None
    bm25_query: str | None = None
    filters: MetadataFilter | None = None
    query_images: tuple[Mapping[str, Any], ...] = ()
    federated_rerank: bool = False


@dataclass(frozen=True, slots=True)
class ProjectedRetrieval:
    contexts: RetrievalContexts
    sources: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class RetrieveResponse:
    contexts: RetrievalContexts
    sources: tuple[Mapping[str, Any], ...]
    trace: Mapping[str, Any]
    image_descriptions: tuple[str, ...]


class PlannerProvider(Protocol):
    def planner_for(self, model_profile: ModelProfile | None = None) -> RetrievalPlanner: ...

    async def aclose(self) -> None: ...


class RetrievalRunRepository(Protocol):
    """Owner-scoped generic Run operations used by Retrieval acceptance."""

    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        run_kind: RunKind,
    ) -> RuntimeRunCreation | None: ...

    async def accept_run(
        self, *, envelope: PreparedRunEnvelope, run_id: str
    ) -> RuntimeRunCreation: ...

    async def get_run(self, *, owner_id: str, run_id: str) -> RunRecord | None: ...


class RetrievalRunScheduler(Protocol):
    @property
    def is_started(self) -> bool: ...

    def admission(self) -> AbstractAsyncContextManager[bool]: ...

    def wake(self) -> None: ...

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncGenerator[RuntimeRunEvent]: ...


class RetrievalService:
    """Accept durable Retrieval and expose the raw stage Answer reuses directly."""

    def __init__(
        self,
        *,
        pool: WorkspacePool,
        planners: PlannerProvider,
        schema_lookup: SchemaLookup,
        image_preparer: QueryImagePreparer,
        projector: RetrievalProjection,
        settings: RetrievalSettings,
        telemetry: Telemetry,
        store: RetrievalRunRepository | None = None,
        coordinator: RetrievalRunScheduler | None = None,
        model_profile_for_role: Callable[[ModelRole], ModelProfile] | None = None,
        model_fingerprint_for_role: Callable[[ModelRole], ModelFingerprint] | None = None,
        run_retention_seconds: int = RETRIEVAL_RUN_RETENTION_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        federated_reranker_factory: Callable[[], FederatedReranker | None] | None = None,
    ) -> None:
        self._pool = pool
        self._planners = planners
        self._schema_lookup = schema_lookup
        self._image_preparer = image_preparer
        self._projector = projector
        self._settings = settings
        self._telemetry = telemetry
        self._store = store
        self._coordinator = coordinator
        self._model_profile_for_role = model_profile_for_role
        self._model_fingerprint_for_role = model_fingerprint_for_role
        self._run_retention_seconds = int(run_retention_seconds)
        self._clock = clock
        self._federated_reranker_factory = federated_reranker_factory
        self._federated_reranker: FederatedReranker | None = None
        self._federated_reranker_unavailable = False
        self._schema_cache: dict[tuple[str, ...], tuple[float, dict[str, Any]]] = {}
        self._schema_refreshes: dict[tuple[str, ...], asyncio.Task[dict[str, Any]]] = {}
        self._visual_query_cache: OrderedDict[_VisualQueryCacheKey, PreparedVisualQuery] = (
            OrderedDict()
        )
        self._visual_query_flights: dict[
            _VisualQueryCacheKey, asyncio.Task[PreparedVisualQuery | None]
        ] = {}
        self._warmups: dict[tuple[str, ...], asyncio.Task[None]] = {}
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    async def _resolve_federated_reranker(self, requested: bool) -> FederatedReranker | None:
        """Build or return the shared federation reranker once, on first request.

        Builds lazily so the composition-time vision probe result is available.
        A missing factory or a failed build degrades to the capped interleave
        and is remembered instead of retried per request.
        """
        if not requested or self._federated_reranker_factory is None:
            return None
        if self._federated_reranker is None and not self._federated_reranker_unavailable:
            try:
                self._federated_reranker = self._federated_reranker_factory()
            except Exception:
                logger.warning("Federated reranker build failed", exc_info=True)
            if self._federated_reranker is None:
                self._federated_reranker_unavailable = True
        return self._federated_reranker

    @property
    def closed(self) -> bool:
        return self._closed

    def bind_runtime(
        self, *, store: RetrievalRunRepository, coordinator: RetrievalRunScheduler
    ) -> None:
        """Complete the single composition cycle shared with this service's executor."""
        if self._store is not None or self._coordinator is not None:
            raise RuntimeError("Retrieval runtime is already bound")
        self._store = store
        self._coordinator = coordinator

    async def _acquire(self, workspace: str) -> Any:
        """Acquire one workspace and translate Engine availability errors."""
        try:
            return await self._pool.acquire(workspace)
        except _EngineCorpusUnavailableError as exc:
            raise CorpusUnavailableError(str(exc)) from exc

    def planner_for(self, model_profile: ModelProfile | None = None) -> RetrievalPlanner:
        if self._closed:
            raise CorpusUnavailableError("Retrieval service is closed")
        return self._planners.planner_for(model_profile)

    async def planner_history_input_measure(
        self,
        *,
        query: str,
        workspaces: Sequence[str],
        model_profile: ModelProfile,
        current_image_descriptions: Sequence[str] = (),
        preserve_query: bool | None = None,
    ) -> Callable[..., int]:
        """Return the exact planner serializer for the schema used by retrieval."""
        planner = self.planner_for(model_profile)
        schema = await self.schema_for(workspaces)
        return planner.history_input_measure(
            query,
            schema=schema,
            current_image_descriptions=list(current_image_descriptions) or None,
            preserve_query=preserve_query,
        )

    async def schema_for(self, workspaces: Sequence[str]) -> dict[str, Any]:
        key = tuple(sorted(workspaces))
        now = self._clock()
        cached = self._schema_cache.get(key)
        if cached is not None and now - cached[0] < 15.0:
            return copy.deepcopy(cached[1])
        refresh = self._schema_refreshes.get(key)
        if refresh is None:
            refresh = asyncio.create_task(self._refresh_schema(key))
            self._schema_refreshes[key] = refresh
            refresh.add_done_callback(
                lambda task, refresh_key=key: self._finish_schema_refresh(refresh_key, task)
            )
        try:
            schema = await asyncio.shield(refresh)
        except Exception:
            logger.debug("Schema lookup failed for workspaces %s", key, exc_info=True)
            return copy.deepcopy(cached[1]) if cached is not None else {}
        if key not in self._schema_cache and len(self._schema_cache) >= 128:
            oldest = min(self._schema_cache, key=lambda item: self._schema_cache[item][0])
            self._schema_cache.pop(oldest, None)
        cached_schema = copy.deepcopy(schema)
        self._schema_cache[key] = (self._clock(), cached_schema)
        return copy.deepcopy(cached_schema)

    async def _refresh_schema(self, key: tuple[str, ...]) -> dict[str, Any]:
        return await self._schema_lookup(key)

    def _finish_schema_refresh(
        self,
        key: tuple[str, ...],
        task: asyncio.Task[dict[str, Any]],
    ) -> None:
        if self._schema_refreshes.get(key) is task:
            self._schema_refreshes.pop(key, None)
        if not task.cancelled() and task.exception() is not None:
            logger.debug("Schema refresh failed for workspaces %s", key, exc_info=task.exception())

    async def _prepared_visual_query(
        self,
        service: WorkspaceRetriever,
        domain: VisualEmbeddingDomain,
        blocks: Sequence[Mapping[str, Any]],
    ) -> tuple[PreparedVisualQuery | None, _VisualPreparationStatus]:
        if self._closed:
            raise CorpusUnavailableError("Retrieval service is closed")
        key = _VisualQueryCacheKey(domain=domain, payload_digest=_query_image_digest(blocks))
        cached = self._visual_query_cache.get(key)
        if cached is not None:
            self._visual_query_cache.move_to_end(key)
            return cached, "cache_hit"

        task = self._visual_query_flights.get(key)
        status: _VisualPreparationStatus = "singleflight_hit"
        if task is None:
            status = "started"
            # The owned task retains raw blocks only for the duration of decode
            # and embedding. Durable cache state contains only the digest and vectors.
            copied_blocks = [dict(block) for block in blocks]
            task = asyncio.create_task(
                self._run_visual_preparation(service, domain, copied_blocks),
                name="retrieval-visual-prepare",
            )
            self._visual_query_flights[key] = task
            task.add_done_callback(
                lambda completed, cache_key=key: self._finish_visual_preparation(
                    cache_key, completed
                )
            )
        return await asyncio.shield(task), status

    async def _run_visual_preparation(
        self,
        service: WorkspaceRetriever,
        domain: VisualEmbeddingDomain,
        blocks: list[dict[str, Any]],
    ) -> PreparedVisualQuery | None:
        try:
            prepared = await service.prepare_visual_query(blocks)
        except Exception:
            logger.warning("Visual query preparation failed", exc_info=True)
            return None
        if prepared is None or not prepared.vectors:
            return None
        if prepared.domain != domain:
            logger.warning("Visual query preparation returned a mismatched embedding domain")
            return None
        return prepared

    def _finish_visual_preparation(
        self,
        key: _VisualQueryCacheKey,
        task: asyncio.Task[PreparedVisualQuery | None],
    ) -> None:
        if self._visual_query_flights.get(key) is task:
            self._visual_query_flights.pop(key, None)
        if task.cancelled() or self._closed:
            return
        try:
            prepared = task.result()
        except Exception:
            logger.debug("Visual query preparation task failed", exc_info=True)
            return
        if prepared is None or not prepared.vectors:
            return
        self._visual_query_cache[key] = prepared
        self._visual_query_cache.move_to_end(key)
        while len(self._visual_query_cache) > _VISUAL_QUERY_CACHE_SIZE:
            self._visual_query_cache.popitem(last=False)

    async def create(
        self,
        *,
        request: RetrieveRequest,
        owner_id: str,
        idempotency_key: str | None = None,
    ) -> RunCreation:
        """Accept one durable Retrieval and return without waiting for execution."""
        run_input, fingerprint, accepted_input = self._normalized_run_input(request)
        store = self._store
        coordinator = self._coordinator
        if store is None or coordinator is None:
            raise RunRuntimeUnavailableError("Retrieval runtime is unavailable")
        try:
            if idempotency_key is not None:
                replay = await store.replay_run(
                    owner_id=owner_id,
                    idempotency_key=idempotency_key,
                    idempotency_fingerprint=fingerprint,
                    run_kind="retrieval",
                )
                if replay is not None:
                    return RunCreation.from_runtime(replay)
            if not coordinator.is_started:
                raise RunRuntimeUnavailableError("Retrieval runtime is unavailable")
            prepared_input = run_input.as_request()
            try:
                require_prepared_input_bounds(prepared_input)
            except PreparedInputTooLargeError as exc:
                raise RetrievalInputError(str(exc)) from exc
            async with coordinator.admission() as runtime_available:
                if not runtime_available:
                    raise RunRuntimeUnavailableError("Retrieval runtime is unavailable")
                run_id = str(uuid7())
                creation = await store.accept_run(
                    envelope=PreparedRunEnvelope(
                        run_kind="retrieval",
                        lane="query",
                        submitted_by=owner_id,
                        access_scope=RunAccessScope(kind="owner", scope_id=owner_id),
                        submission_key=idempotency_key or run_id,
                        request_fingerprint=fingerprint,
                        payload=prepared_input,
                        accepted_input=accepted_input,
                        retention_seconds=self._run_retention_seconds,
                    ),
                    run_id=run_id,
                )
                coordinator.wake()
        except RuntimeIdempotencyKeyConflict as exc:
            raise IdempotencyKeyConflict(str(exc)) from exc
        except RuntimeRunCapacityExceededError as exc:
            raise RunCapacityExceededError(str(exc)) from exc
        return RunCreation.from_runtime(creation)

    def _normalized_run_input(
        self, request: RetrieveRequest
    ) -> tuple[RetrievalRunInput, str, dict[str, Any]]:
        if self._closed:
            raise CorpusUnavailableError("Retrieval service is closed")
        if not request.workspaces:
            raise RetrievalInputError("At least one canonical workspace is required")
        workspaces = tuple(
            require_canonical_workspace_id(workspace) for workspace in request.workspaces
        )
        images = tuple(dict(image) for image in request.query_images)
        if len(images) > self._settings.query_image_limit:
            raise RetrievalInputError(
                f"at most {self._settings.query_image_limit} current images are allowed"
            )
        top_k = request.top_k or self._settings.default_top_k
        chunk_top_k = request.chunk_top_k or self._settings.default_chunk_top_k
        if top_k < 1 or top_k > self._settings.default_top_k * 10:
            raise RetrievalInputError(
                f"top_k must be between 1 and {self._settings.default_top_k * 10}"
            )
        if chunk_top_k < 1 or chunk_top_k > self._settings.default_chunk_top_k * 10:
            raise RetrievalInputError(
                f"chunk_top_k must be between 1 and {self._settings.default_chunk_top_k * 10}"
            )
        filters = (
            request.filters.model_dump(exclude_none=True, mode="json")
            if request.filters is not None
            else None
        )
        normalized_request = {
            "query": request.query,
            "workspaces": list(workspaces),
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
            "federated_rerank": bool(request.federated_rerank),
            "filters": filters,
            "bm25_query": (request.bm25_query or "").strip() or None,
            "query_images": [dict(image) for image in images],
        }
        fingerprint = run_request_fingerprint(normalized_request)
        if self._model_profile_for_role is None or self._model_fingerprint_for_role is None:
            raise RunRuntimeUnavailableError("Retrieval model pinning is unavailable")
        roles: tuple[ModelRole, ...] = ("extract", "vlm") if images else ("extract",)
        pinned_models = tuple(
            PinnedRetrievalModel(
                role=role,
                fingerprint=self._model_fingerprint_for_role(role),
                profile=self._model_profile_for_role(role),
            )
            for role in roles
        )
        run_input = RetrievalRunInput(
            query=request.query,
            workspaces=workspaces,
            retrieval=RetrievalOptions(
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                federated_rerank=bool(request.federated_rerank),
            ),
            bm25_query=normalized_request["bm25_query"],
            filters=filters,
            query_images=images,
            pinned_models=pinned_models,
            context_policy_revision=CONTEXT_POLICY_REVISION,
            model_catalog_revision=current_model_catalog_revision(),
            idempotency_fingerprint=fingerprint,
        )
        accepted_input = {
            **{key: value for key, value in normalized_request.items() if key != "query_images"},
            "query_image_count": len(images),
            "query_image_digests": [_query_image_hexdigest((image,)) for image in images],
            "result_projection": "retrieval_v1",
            "retention_seconds": self._run_retention_seconds,
        }
        return run_input, fingerprint, accepted_input

    async def wait(
        self,
        *,
        owner_id: str,
        run_id: str,
        projection: RetrieveProjection,
    ) -> RetrieveResponse:
        """Wait for one owned run and apply the caller's current reader scope."""
        store = self._store
        coordinator = self._coordinator
        if store is None or coordinator is None:
            raise RunRuntimeUnavailableError("Retrieval runtime is unavailable")
        async with aclosing(coordinator.subscribe(owner_id=owner_id, run_id=run_id)) as events:
            async for _event in events:
                pass
        final = await store.get_run(owner_id=owner_id, run_id=run_id)
        if final is None:
            raise RunFailedError(
                "retrieval_run_missing",
                "Retrieval disappeared before it finished.",
            )
        if final.status == "succeeded":
            return self.project_stored(final.result or {}, projection)
        if final.status == "cancelled":
            raise RunCancelledError(final.run_id)
        raise RunFailedError(
            final.error_kind or "retrieval_failed",
            final.error_message or "Retrieval failed.",
        )

    def project_stored(
        self,
        stored: Mapping[str, Any],
        projection: RetrieveProjection,
    ) -> RetrieveResponse:
        """Apply current reader scope to canonical Retrieval output."""
        result = restore_retrieval_result(stored)
        projected = self._projector(result, projection)
        return RetrieveResponse(
            contexts=projected.contexts,
            sources=projected.sources,
            trace=dict(result.trace),
            image_descriptions=tuple(result.image_descriptions),
        )

    async def retrieve(
        self,
        request: RetrieveRequest,
        *,
        owner_id: str,
        projection: RetrieveProjection,
        idempotency_key: str | None = None,
    ) -> RetrieveResponse:
        """Create one durable Retrieval and wait through the common lifecycle."""
        creation = await self.create(
            request=request,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
        )
        return await self.wait(
            owner_id=owner_id,
            run_id=creation.run.run_id,
            projection=projection,
        )

    async def stream(
        self,
        *,
        request: RetrieveRequest,
        owner_id: str,
        idempotency_key: str | None = None,
    ) -> AsyncGenerator[RunEvent]:
        """Create one durable Retrieval and follow its common ordered events."""
        creation = await self.create(
            request=request,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
        )
        coordinator = self._coordinator
        if coordinator is None:
            raise RunRuntimeUnavailableError("Retrieval runtime is unavailable")
        async with aclosing(
            coordinator.subscribe(owner_id=owner_id, run_id=creation.run.run_id)
        ) as events:
            async for event in events:
                yield RunEvent.from_runtime(event)

    async def prepare_query_images(self, images: Sequence[Mapping[str, Any]]) -> list[str]:
        """Describe accepted query images during owned run execution."""
        return await self._image_preparer(images) if images else []

    def warm(self, workspaces: Sequence[str]) -> None:
        """Start or join one owned warm waiter for an identical workspace set."""
        if self._closed:
            return
        key = tuple(sorted(set(workspaces)))
        if key in self._warmups:
            return
        warmup = asyncio.create_task(
            self._pool.warm(key),
            name=f"retrieval-warm:{','.join(key)}",
        )
        self._warmups[key] = warmup
        warmup.add_done_callback(
            lambda completed, warm_key=key: self._observe_warmup(warm_key, completed)
        )

    async def retrieve_result(
        self,
        query: str,
        *,
        workspaces: Sequence[str],
        conversation_history: Sequence[Mapping[str, object]] | None = None,
        retrieval: RetrievalOptions = RetrievalOptions(),
        bm25_query: str | None = None,
        filters: MetadataFilter | None = None,
        query_images: Sequence[Mapping[str, Any]] = (),
        image_descriptions: Sequence[str] = (),
        preserve_query: bool | None = None,
        model_profile: ModelProfile | None = None,
        planner: RetrievalPlanner | None = None,
    ) -> RetrievalResult:
        """Plan and execute raw retrieval without inline timeout or reader projection."""
        if self._closed:
            raise CorpusUnavailableError("Retrieval service is closed")
        if not workspaces:
            raise ValueError("At least one canonical workspace is required")
        top_k = retrieval.top_k
        chunk_top_k = retrieval.chunk_top_k
        federated_rerank = retrieval.federated_rerank
        active_planner = planner or self.planner_for(model_profile)
        async with self._telemetry.observe(
            "retrieval_planning",
            as_type="chain",
            input={"query": query},
            metadata={
                "workspaces": list(workspaces),
                "history_messages": len(conversation_history or ()),
            },
        ) as planning_observation:
            schema = await self.schema_for(workspaces)
            plan: RetrievalPlan = await active_planner.plan(
                query,
                conversation_history=conversation_history,
                schema=schema,
                current_image_descriptions=list(image_descriptions) or None,
                preserve_query=preserve_query,
            )
            planning_observation.update(
                output={
                    "standalone_query": plan.standalone_query,
                    "has_metadata_filter": plan.metadata_filter is not None,
                    "planning_outcome": plan.outcome,
                }
            )

        effective_top_k = _positive_int_or_none(top_k) or self._settings.default_top_k
        effective_chunk_top_k = (
            _positive_int_or_none(chunk_top_k) or self._settings.default_chunk_top_k
        )
        kwargs: dict[str, Any] = {
            "top_k": effective_top_k,
            "chunk_top_k": effective_chunk_top_k,
        }
        visual_stats = {
            "visual_preparation_domain_count": 0,
            "visual_preparation_started_count": 0,
            "visual_preparation_cache_hit_count": 0,
            "visual_preparation_singleflight_hit_count": 0,
            "visual_preparation_failed_count": 0,
        }
        visual_blocks = tuple(dict(image) for image in query_images)

        async def _prepare_visual(
            runtime: WorkspaceRetriever,
            domain: VisualEmbeddingDomain,
            blocks: Sequence[Mapping[str, Any]],
        ) -> PreparedVisualQuery | None:
            visual_stats["visual_preparation_domain_count"] += 1
            prepared, status = await self._prepared_visual_query(runtime, domain, blocks)
            visual_stats[f"visual_preparation_{status}_count"] += 1
            if prepared is None:
                visual_stats["visual_preparation_failed_count"] += 1
            return prepared

        effective_filters = filters if filters is not None else plan.metadata_filter
        if effective_filters is not None:
            kwargs["filters"] = effective_filters
        filter_source = "explicit" if filters is not None else plan.metadata_filter_source
        if filter_source is not None:
            kwargs["filter_source"] = filter_source
        effective_bm25_query = (bm25_query or "").strip() or plan.bm25_query
        if effective_bm25_query is not None:
            kwargs["bm25_query"] = effective_bm25_query

        async with self._telemetry.observe(
            "retrieve",
            as_type="retriever",
            input={"query": query},
            metadata={
                "workspaces": list(workspaces),
                "top_k": effective_top_k,
                "chunk_top_k": effective_chunk_top_k,
                "has_filters": effective_filters is not None,
                "federated_rerank": federated_rerank,
            },
        ) as observation:
            if len(workspaces) == 1:
                runtime = await self._acquire(workspaces[0])
                domain = getattr(runtime, "visual_embedding_domain", None)
                if visual_blocks and isinstance(domain, VisualEmbeddingDomain):
                    kwargs["prepared_visual_query"] = await _prepare_visual(
                        runtime, domain, visual_blocks
                    )
                result = await runtime.aretrieve(plan.standalone_query, **kwargs)
            else:
                reranker = await self._resolve_federated_reranker(federated_rerank)
                policy = FederationMergePolicy(
                    chunk_top_k=effective_chunk_top_k,
                    min_chunks_per_workspace=(self._settings.federation_min_chunks_per_workspace),
                )
                result = await federated_retrieve(
                    plan.standalone_query,
                    list(workspaces),
                    self._acquire,
                    policy=policy,
                    reranker=reranker,
                    max_concurrency=self._settings.workspace_fanout_concurrency,
                    query_image_blocks=visual_blocks,
                    prepare_visual_query=_prepare_visual if visual_blocks else None,
                    **{key: value for key, value in kwargs.items() if key != "chunk_top_k"},
                )
                if federated_rerank and reranker is None:
                    result.trace["federated_rerank_unavailable"] = True
            result.image_descriptions = list(image_descriptions)
            result.trace["query_image_description_count"] = len(image_descriptions)
            if visual_blocks:
                result.trace.update(visual_stats)
            observation.update(
                output={
                    **_context_output(result.contexts),
                    "standalone_query": plan.standalone_query,
                    "query_image_description_count": len(image_descriptions),
                }
            )
            return result

    async def aclose(self) -> None:
        close_task = self._close_task
        if close_task is None:
            self._closed = True
            close_task = asyncio.create_task(self._close_resources())
            self._close_task = close_task
        await await_shared_cleanup(close_task)

    async def _close_resources(self) -> None:
        warmups = list(self._warmups.values())
        for warmup in warmups:
            warmup.cancel()
        for refresh in self._schema_refreshes.values():
            refresh.cancel()
        visual_flights = list(self._visual_query_flights.values())
        for flight in visual_flights:
            flight.cancel()
        if warmups:
            await asyncio.gather(*warmups, return_exceptions=True)
        self._warmups.clear()
        if self._schema_refreshes:
            await asyncio.gather(*self._schema_refreshes.values(), return_exceptions=True)
            self._schema_refreshes.clear()
        if visual_flights:
            await asyncio.gather(*visual_flights, return_exceptions=True)
        self._visual_query_flights.clear()
        if self._federated_reranker is not None:
            close = getattr(self._federated_reranker, "aclose", None)
            if close is not None:
                await close()
            self._federated_reranker = None
        self._visual_query_cache.clear()
        await self._planners.aclose()

    def _observe_warmup(
        self,
        key: tuple[str, ...],
        task: asyncio.Task[None],
    ) -> None:
        if self._warmups.get(key) is task:
            self._warmups.pop(key, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.debug("Workspace warm-up failed", exc_info=error)


def retrieval_response_payload(response: RetrieveResponse) -> dict[str, Any]:
    """Serialize the shared reader projection for REST and MCP."""
    return {
        "contexts": response.contexts,
        "sources": [dict(source) for source in response.sources],
        "trace": dict(response.trace),
        "image_descriptions": list(response.image_descriptions),
    }


def _query_image_digest(blocks: Sequence[Mapping[str, Any]]) -> bytes:
    payload = json.dumps(
        [dict(block) for block in blocks],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _query_image_hexdigest(blocks: Sequence[Mapping[str, Any]]) -> str:
    return _query_image_digest(blocks).hex()


def _positive_int_or_none(value: int | None) -> int | None:
    return value if value is not None and value > 0 else None


def _context_output(contexts: RetrievalContexts) -> dict[str, int]:
    return {
        "chunk_count": len(contexts.get("chunks", [])),
        "entity_count": len(contexts.get("entities", [])),
        "relationship_count": len(contexts.get("relationships", [])),
    }


__all__ = [
    "CorpusUnavailableError",
    "ProjectedRetrieval",
    "QueryImagePreparer",
    "RetrieveProjection",
    "RetrieveRequest",
    "RetrieveResponse",
    "RetrievalInputError",
    "RetrievalOptions",
    "RetrievalRunRepository",
    "RetrievalRunScheduler",
    "RetrievalService",
    "RetrievalSettings",
    "RetrievalTimeoutError",
    "SchemaLookup",
    "retrieval_response_payload",
]
