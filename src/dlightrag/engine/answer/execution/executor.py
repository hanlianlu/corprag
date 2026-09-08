# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer execution over durable Runtime sessions and RAG workspaces."""

import asyncio
import base64
import datetime
import hashlib
import hmac
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from dlightrag_memory import Memory, MemoryStore

from dlightrag.application.answer_runs.capabilities import (
    AnswerCapabilityCoordinator,
    RequestModelContext,
)
from dlightrag.application.answer_runs.capability import (
    AnswerImageCapability,
    check_answer_image_capability,
    check_answer_image_count,
)
from dlightrag.application.answer_runs.errors import (
    AnswerInputError,
    AnswerResourceAdmissionError,
    CurrentImagePayloadError,
    InvalidToolConfigurationError,
    classify_answer_error,
)
from dlightrag.application.answer_runs.execution import (
    AnswerRunInput,
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    build_current_answer_resources,
)
from dlightrag.application.answer_runs.mode import ModeResource, ResolvedMode, resource_role
from dlightrag.application.answer_runs.results import store_answer_result
from dlightrag.application.answer_runs.routing import AnswerRoutingStore, decide_resolved_mode
from dlightrag.application.answer_runs.sources import project_contexts_for_client
from dlightrag.engine.agent.environment import (
    ExecutionEnvironment,
    SearchToolchain,
    resolve_execution_adapter,
)
from dlightrag.engine.agent.session.effects import (
    canonical_json,
)
from dlightrag.engine.agent.session.entries import CompactionEntry, UserMessageEntry
from dlightrag.engine.agent.session.fold import (
    PriorTurns,
    host_turn_starts,
    project_session_messages,
)
from dlightrag.engine.agent.session.ids import (
    EntryId,
    LaneId,
    OperationId,
    SessionId,
)
from dlightrag.engine.agent.session.operation import (
    OperationCompleted,
)
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.projection import ContextProjection
from dlightrag.engine.agent.session.registers import (
    HostTurnReservation,
    LaneHead,
    LaneState,
    RegisterRef,
    SetRegister,
)
from dlightrag.engine.agent.session.repository import validate_snapshot_refresh
from dlightrag.engine.agent.session.runtime import (
    AgentSessionRuntime,
    AgentSessionSnapshotSeed,
    FollowUpCommand,
    OperationConflictError,
    SessionLeaseLostError,
)
from dlightrag.engine.agent.session.transactions import (
    RegisterConflict,
    RegisterExpectation,
    SessionTransaction,
    TransactionLeaseLost,
)
from dlightrag.engine.agent.skills import SkillsBundle, SkillsBundleFactory
from dlightrag.engine.agent.tools import (
    AgentTool,
)
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.scheduler import model_call_scope
from dlightrag.engine.ai.settings import MODEL_ROLE_NAMES, ModelRole
from dlightrag.engine.ai.telemetry import Telemetry, safe_log_text
from dlightrag.engine.answer.citations.finalization import finalize_answer
from dlightrag.engine.answer.citations.streaming import aclose_answer_stream
from dlightrag.engine.answer.compaction import CompactionCoordinator
from dlightrag.engine.answer.fast import FastRunBoundaries, FastSessionHost, ensure_session_lane
from dlightrag.engine.answer.highlights import SemanticHighlightSettings, enrich_semantic_highlights
from dlightrag.engine.answer.history import HistoryInputMeasure, HistoryProjectionTarget
from dlightrag.engine.answer.images import AnswerImageBudget
from dlightrag.engine.answer.media import evidence_images_from_sources
from dlightrag.engine.answer.memory import memory_owner_allowed, render_auto_recall
from dlightrag.engine.answer.model_runtime import AnswerModelRuntime
from dlightrag.engine.answer.orchestration import AnswerOrchestrator
from dlightrag.engine.answer.publication import (
    ArtifactAttachment,
    PublicationLimits,
    PublicationPlan,
    is_empty_answer,
    validate_publication,
)
from dlightrag.engine.answer.research.runtime import (
    AnswerRuntimeControls,
    FetchedResourceBuffer,
    ResearchRuntimeEffects,
    _answer_runtime_event_sink,
    _async_store_method,
    _bound_child_runner,
    _buffered_fetched_bytes_sink,
    _drive_answer_operation,
    _durable_child_usage,
    _fenced_child_writer,
    _fenced_control_ack,
    _fenced_control_reader,
    _oldest_pending_input,
    _restore_durable_evidence,
    _usage_from_snapshot_entries,
)
from dlightrag.engine.answer.resources import ResourceInput, ResourceRegistry
from dlightrag.engine.answer.resources.models import (
    ResourceManifestEntry,
    ResourceRegistryError,
    TextWindowBudget,
)
from dlightrag.engine.answer.resources.registry import (
    FetchedBytesSink,
)
from dlightrag.engine.answer.resources.visual import ResourceInspector
from dlightrag.engine.answer.router import AnswerModeRouter
from dlightrag.engine.answer.tools.memory import MemoryHost
from dlightrag.engine.answer.tools.resources import build_resource_tools, make_resource_reader
from dlightrag.engine.answer.tools.subagents import (
    SubagentHost,
)
from dlightrag.engine.answer.web_sources import WebSourceService
from dlightrag.engine.answer.workspace import (
    WorkspaceIntegrityError,
    WorkspaceRecoveryFailed,
    bind_run_workspace,
)
from dlightrag.engine.dependencies import (
    DependencyComponent,
    classify_transient_dependency,
    dependency_component_from_checkpoint,
    next_dependency_retry,
)
from dlightrag.engine.public_http import fetch_public_http
from dlightrag.engine.rag.corpus.sources.source_contract import safe_source_filename
from dlightrag.engine.rag.retrieval import (
    MetadataFilter,
    RetrievalContexts,
    RetrievalOptions,
    RetrievalResult,
)
from dlightrag.engine.rag.workspace.lifecycle import defer_cancellation
from dlightrag.engine.rag.workspace.pool import WorkspacePool
from dlightrag.engine.runtime import (
    AlreadyCommittedTerminal,
    Deferred,
    IncompatibleActiveRunError,
    LeaseLostError,
    RunCancellationObserved,
    RunExecutionError,
    RunExecutionOutcome,
    RunSession,
    Succeeded,
)
from dlightrag.engine.runtime.records import (
    PendingPublication,
    RunFetchedResource,
    artifact_digest,
)
from dlightrag.engine.runtime.settlements import (
    ArtifactAttachmentUpdate,
    EffectHostUpdate,
)

logger = logging.getLogger(__name__)
_FAST_COMPACTION_ATTEMPT_LIMIT = 3
_DEPENDENCY_DEFER_BASE_SECONDS = 5
_DEPENDENCY_DEFER_MAX_SECONDS = 60

type DependencyStateCallback = Callable[[DependencyComponent], None]


def _scoped_secret(secret: bytes | None, scope: str | None) -> bytes | None:
    if secret is None or scope is None:
        return secret
    return hmac.new(secret, scope.encode("utf-8"), hashlib.sha256).digest()


class RunBlobReader(Protocol):
    """Stream one owner-scoped opaque Run blob by digest."""

    def stream(
        self,
        *,
        owner_id: str,
        digest: str,
        offset: int = 0,
        length: int | None = None,
    ) -> AsyncIterator[bytes]: ...


class ArtifactReader(Protocol):
    """Read Answer-owned artifact metadata without owning blob bytes."""

    async def list_run_artifacts(self, *, owner_id: str, run_id: str) -> tuple[Any, ...]: ...
    async def list_artifact_attachments(
        self, *, owner_id: str, run_id: str
    ) -> tuple[ArtifactAttachmentUpdate, ...]: ...
    async def list_fetched_resources(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunFetchedResource, ...]: ...


class AnswerExecutionStore(ArtifactReader, AnswerRoutingStore, Protocol):
    """Executor store: artifacts plus the lease-fenced Routing Record."""


type PlannerHistoryInputMeasureFactory = Callable[..., Awaitable[HistoryInputMeasure]]
type WorkspaceWarmer = Callable[[Sequence[str]], None]


class RawRetrieval(Protocol):
    async def __call__(
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
    ) -> RetrievalResult: ...


@dataclass(frozen=True, slots=True)
class AnswerResourceSettings:
    max_attachments: int
    max_attachment_bytes: int
    max_total_attachment_bytes: int
    image_max_bytes: int
    image_max_pixels: int


@dataclass(frozen=True, slots=True)
class AnswerExecutorSettings:
    default_top_k: int
    default_chunk_top_k: int
    semantic_highlights: SemanticHighlightSettings
    publication: PublicationLimits = PublicationLimits()


@dataclass
class OrchestratorRun:
    """One durable request resolved into an orchestrator and its exact inputs."""

    orchestrator: AnswerOrchestrator
    image_descriptions: list[str]
    query_images: list[dict[str, Any]] | None
    history: PriorTurns
    fast_history_targets: tuple[HistoryProjectionTarget, ...]
    current_image_count: int
    workspaces: list[str]
    registry: ResourceRegistry | None


@dataclass(frozen=True, slots=True)
class ResolvedAnswerResources:
    models: RequestModelContext
    web_sources: WebSourceService | None
    registry: ResourceRegistry | None
    resource_tools: list[AgentTool]
    resource_manifest: tuple[ResourceManifestEntry, ...]
    current_images: list[dict[str, Any]]
    current_image_count: int
    image_budget: AnswerImageBudget | None
    query_images: list[dict[str, Any]] | None


class AnswerResourceResolver:
    """Resolve request resources, visual policy, and peer tools exactly once."""

    def __init__(
        self,
        *,
        settings: AnswerResourceSettings,
        models: AnswerModelRuntime,
        capabilities: AnswerCapabilityCoordinator,
        resource_identity_secret: bytes | None = None,
        resource_cursor_secret: bytes | None = None,
    ) -> None:
        self._settings = settings
        self._models = models
        self._capabilities = capabilities
        self._resource_identity_secret = resource_identity_secret
        self._resource_cursor_secret = resource_cursor_secret

    async def pin_current_image_links(
        self,
        request: AnswerRunRequest,
        attachment_bytes: Sequence[bytes],
    ) -> tuple[AnswerRunRequest, list[bytes]]:
        """Materialize declared image links once for acceptance and durable replay."""
        if len(request.attachments) != len(attachment_bytes):
            raise ValueError("current attachment references and bytes must have equal length")
        image_count = sum(
            resource_role(filename=attachment.filename, mime_type=attachment.mime_type) == "image"
            for attachment in request.attachments
        ) + sum(
            resource_role(filename=link.filename or link.url, mime_type=link.mime_type) == "image"
            for link in request.links
        )
        if image_count:
            capabilities = await self._capabilities.refresh_answer()
            check_answer_image_count(
                image_count=image_count,
                configured_ceiling=(
                    capabilities.answer.configured_ceiling if capabilities.answer is not None else 0
                ),
            )
        links: list[LinkReference] = []
        pinned_link_attachments: list[AttachmentReference] = []
        pinned_link_bytes: list[bytes] = []
        for link in request.links:
            if (
                resource_role(
                    filename=link.filename or link.url,
                    mime_type=link.mime_type,
                )
                != "image"
            ):
                links.append(link)
                continue
            data = await self.materialize_link_image(link.url)
            if data is None:
                raise CurrentImagePayloadError(
                    f"current image {link.filename or link.url} could not be fetched and verified"
                )
            try:
                mime_type, _data_uri = await asyncio.to_thread(
                    _verified_current_image_data_uri,
                    data,
                    max_pixels=self._settings.image_max_pixels,
                )
            except ValueError as exc:
                raise CurrentImagePayloadError(
                    f"current image {link.filename or link.url} {exc}"
                ) from exc
            pinned_link_attachments.append(
                AttachmentReference(
                    digest=artifact_digest(data),
                    filename=safe_source_filename(link.filename),
                    mime_type=mime_type,
                    ordinal=len(pinned_link_attachments),
                    byte_size=len(data),
                )
            )
            pinned_link_bytes.append(data)
        offset = len(pinned_link_attachments)
        attachments = [
            *pinned_link_attachments,
            *(
                replace(attachment, ordinal=offset + index)
                for index, attachment in enumerate(request.attachments)
            ),
        ]
        return (
            replace(
                request,
                links=tuple(links),
                attachments=tuple(attachments),
            ),
            [*pinned_link_bytes, *attachment_bytes],
        )

    async def resolve(
        self,
        resources: list[ResourceInput] | None,
        *,
        models: RequestModelContext,
        text_window_budget: TextWindowBudget,
        confirm_image_context: Callable[
            [RequestModelContext],
            Awaitable[tuple[RequestModelContext, AnswerImageCapability | None]],
        ],
        fetched_bytes_sink: FetchedBytesSink | None = None,
        resolved_mode: ResolvedMode,
        resource_scope: str | None = None,
    ) -> ResolvedAnswerResources:
        """Resolve resource capabilities, manifests, tools, and image transport."""
        declared_image_count = sum(
            1
            for resource in resources or ()
            if resource.loader is None
            and resource_role(
                filename=resource.filename or resource.url,
                mime_type=resource.declared_mime,
            )
            == "image"
        )
        image_capability: AnswerImageCapability | None = None
        if declared_image_count:
            models, image_capability = await confirm_image_context(models)
            self._check_current_image_admission(
                image_count=declared_image_count,
                capability=image_capability,
                models=models,
                resolved_mode=resolved_mode,
            )
        (
            current_images,
            remaining_resources,
            current_image_resources,
        ) = await self.prepare_current_images(resources)
        if current_images and not declared_image_count:
            models, image_capability = await confirm_image_context(models)
        self._check_current_image_admission(
            image_count=len(current_images),
            capability=image_capability,
            models=models,
            resolved_mode=resolved_mode,
        )

        web_sources = self._models.web_sources()
        registry, resource_tools = self.build_resource_context(
            remaining_resources,
            text_window_budget=text_window_budget,
            web_sources=web_sources,
            fetched_bytes_sink=fetched_bytes_sink,
            vlm_profile=models.vlm,
            resource_scope=resource_scope,
        )
        try:
            current_image_resource_ids = (
                tuple(registry.register(resource) for resource in current_image_resources)
                if registry is not None
                else ()
            )
            resource_manifest = registry.manifest() if registry is not None else ()
            image_budget: AnswerImageBudget | None = None
            query_images: list[dict[str, Any]] | None = current_images or None
            if resolved_mode == "research":
                image_budget = self._capabilities.answer_image_policy(models.query).new_budget()
                if image_capability is not None and image_capability.status == "supported":
                    query_images = (
                        await self.budget_agent_images(
                            current_images,
                            image_budget,
                            current_image_resource_ids,
                        )
                        or None
                    )
                else:
                    inspect_budget = self._capabilities.vlm_image_policy(models.vlm).new_budget()
                    await self.budget_agent_images(current_images, inspect_budget)
                    query_images = None
            return ResolvedAnswerResources(
                models=models,
                web_sources=web_sources,
                registry=registry,
                resource_tools=resource_tools,
                resource_manifest=resource_manifest,
                current_images=current_images,
                current_image_count=len(current_images),
                image_budget=image_budget,
                query_images=query_images,
            )
        except BaseException:
            if registry is not None:
                await registry.aclose()
            raise

    async def prepare_current_images(
        self,
        resources: list[ResourceInput] | None,
    ) -> tuple[list[dict[str, Any]], list[ResourceInput], list[ResourceInput]]:
        """Build verified current-image blocks while retaining attachments as resources."""
        if not resources:
            return [], [], []
        images: list[dict[str, Any]] = []
        remaining: list[ResourceInput] = []
        image_resources: list[ResourceInput] = []
        for resource in resources:
            data: bytes | None = None
            if resource.loader is not None:
                remaining.append(resource)
                continue
            if resource.content is not None:
                data = resource.content
            elif (
                resource.url is not None
                and resource_role(
                    filename=resource.filename or resource.url,
                    mime_type=resource.declared_mime,
                )
                == "image"
            ):
                data = await self.materialize_link_image(resource.url)
            if data is None:
                if (
                    resource.url is not None
                    and resource_role(
                        filename=resource.filename or resource.url,
                        mime_type=resource.declared_mime,
                    )
                    == "image"
                ):
                    raise CurrentImagePayloadError(
                        f"current image {resource.filename or resource.url} "
                        "could not be fetched and verified"
                    )
                remaining.append(resource)
                continue
            try:
                mime, data_uri = await asyncio.to_thread(
                    _verified_current_image_data_uri,
                    data,
                    max_pixels=self._settings.image_max_pixels,
                )
            except ValueError as exc:
                if (
                    resource_role(
                        filename=resource.filename or resource.url,
                        mime_type=resource.declared_mime,
                    )
                    == "image"
                ):
                    raise CurrentImagePayloadError(
                        f"current image {resource.filename or len(images) + 1} {exc}"
                    ) from exc
                remaining.append(resource)
                continue
            images.append({"type": "image_url", "image_url": {"url": data_uri}})
            image_resource = ResourceInput(
                filename=resource.filename,
                content=data,
                declared_mime=mime,
            )
            remaining.append(image_resource)
            image_resources.append(image_resource)
        return images, remaining, image_resources

    @staticmethod
    def _check_current_image_admission(
        *,
        image_count: int,
        capability: AnswerImageCapability | None,
        models: RequestModelContext,
        resolved_mode: ResolvedMode,
    ) -> None:
        if image_count <= 0:
            return
        if capability is None:
            check_answer_image_capability(image_count=image_count, capability=None)
            return
        check_answer_image_count(
            image_count=image_count,
            configured_ceiling=capability.configured_ceiling,
        )
        if resolved_mode == "fast" or not models.vlm.supports_images:
            check_answer_image_capability(
                image_count=image_count,
                capability=capability,
            )

    async def materialize_link_image(self, url: str) -> bytes | None:
        """Fetch one current-image link under SSRF revalidation."""
        try:
            result = await fetch_public_http(
                url,
                max_bytes=self._settings.image_max_bytes,
                timeout=120.0,
            )
            return result.content
        except Exception:
            logger.warning("Failed to materialize current image link", exc_info=True)
            return None

    def build_resource_context(
        self,
        resources: list[ResourceInput] | None,
        *,
        text_window_budget: TextWindowBudget,
        web_sources: WebSourceService | None = None,
        fetched_bytes_sink: FetchedBytesSink | None = None,
        vlm_profile: ModelProfile,
        resource_scope: str | None = None,
    ) -> tuple[ResourceRegistry | None, list[AgentTool]]:
        """Register resources and compose their text and visual peer tools.

        The registry always exists in Research-capable composition so ``read(url=...)``
        does not depend on an Execution Environment or a configured provider.
        """
        registry = ResourceRegistry(
            max_attachments=self._settings.max_attachments,
            max_attachment_bytes=self._settings.max_attachment_bytes,
            max_total_attachment_bytes=self._settings.max_total_attachment_bytes,
            url_text_fallback=(web_sources.extract if web_sources is not None else None),
            fetched_bytes_sink=fetched_bytes_sink,
            resource_secret=_scoped_secret(self._resource_identity_secret, resource_scope),
            cursor_secret=_scoped_secret(self._resource_cursor_secret, resource_scope),
        )
        try:
            for resource in resources or []:
                registry.register(resource)
        except (ValueError, ResourceRegistryError) as exc:
            raise AnswerResourceAdmissionError() from exc

        vlm_policy = self._capabilities.vlm_image_policy(vlm_profile)
        visual_supported = vlm_profile.supports_images and vlm_policy.max_images > 0
        inspector = (
            ResourceInspector(
                registry,
                vlm_func=self._models.vlm_func(),
                image_policy=vlm_policy,
            )
            if visual_supported
            else None
        )
        tools = build_resource_tools(
            registry,
            text_window_budget=text_window_budget,
            inspector=inspector,
            visual_supported=visual_supported,
        )
        return registry, tools

    @staticmethod
    async def budget_agent_images(
        current_images: list[dict[str, Any]],
        budget: AnswerImageBudget,
        resource_ids: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        def build() -> list[dict[str, Any]]:
            blocks: list[dict[str, Any]] = []
            for index, image in enumerate(current_images, start=1):
                block = budget.add_user_image(image, label=f"query_image_{index}")
                if block is None:
                    raise CurrentImagePayloadError(
                        f"current image query_image_{index} could not fit the answer image budget"
                    )
                if index <= len(resource_ids):
                    blocks.append(
                        {
                            "type": "text",
                            "text": (
                                f"[current image {index} | resource: {resource_ids[index - 1]}]"
                            ),
                        }
                    )
                blocks.append(block)
            return blocks

        return await asyncio.to_thread(build)


async def _memory_recall_allowed(
    checker: Callable[..., Awaitable[bool]] | None, *, owner_id: str
) -> bool:
    """Whether answer injection may use this owner's memory.

    A missing checker means the composition has no settings store and memory
    stays enabled — the historical default.
    """
    return checker is None or await checker(owner_id=owner_id)


class AnswerExecutor:
    """Execute durable Answer runs without composition or storage dependencies."""

    def __init__(
        self,
        *,
        store: AnswerExecutionStore,
        blob_store: RunBlobReader,
        pool: WorkspacePool,
        warm: WorkspaceWarmer,
        retrieve: RawRetrieval,
        planner_history_input_measure: PlannerHistoryInputMeasureFactory,
        models: AnswerModelRuntime,
        capabilities: AnswerCapabilityCoordinator,
        resources: AnswerResourceResolver,
        settings: AnswerExecutorSettings,
        telemetry: Telemetry,
        model_fingerprint_for_role: Callable[[ModelRole], ModelFingerprint],
        execution_environment: str = "trust",
        workspace_root: str | None = None,
        search_toolchain: SearchToolchain | None = None,
        working_dir: str = "./dlightrag_storage",
        memory_store: MemoryStore | None = None,
        memory_recall_enabled: Callable[..., Awaitable[bool]] | None = None,
        memory_capability_current: Callable[..., Awaitable[bool]] | None = None,
        external_tools: tuple[AgentTool, ...] = (),
        skills_bundle_factory: SkillsBundleFactory | None = None,
        now: Callable[[], datetime.datetime] | None = None,
        on_dependency_unavailable: DependencyStateCallback | None = None,
        on_dependency_recovered: DependencyStateCallback | None = None,
    ) -> None:
        self._store = store
        self._blob_store = blob_store
        self._pool = pool
        self._warm = warm
        self._retrieve_result = retrieve
        self._planner_history_input_measure = planner_history_input_measure
        self._models = models
        self._capabilities = capabilities
        self._resources = resources
        self._settings = settings
        self._telemetry = telemetry
        self._model_fingerprint_for_role = model_fingerprint_for_role
        self._execution_environment = execution_environment
        self._workspace_root_setting = workspace_root
        self._search_toolchain = search_toolchain
        self._working_dir = working_dir
        self._memory_store = memory_store
        self._memory = Memory(memory_store) if memory_store is not None else None
        self._memory_recall_enabled = memory_recall_enabled
        self._memory_capability_current = memory_capability_current
        self._external_tools = external_tools
        self._skills_bundle_factory = skills_bundle_factory
        self._now = now or (lambda: datetime.datetime.now(datetime.UTC))
        self._on_dependency_unavailable = on_dependency_unavailable
        self._on_dependency_recovered = on_dependency_recovered
        if execution_environment not in {"disabled", "trust", "sandbox"}:
            raise ValueError(f"unknown agent execution mode: {execution_environment}")
        self._execution_adapter = resolve_execution_adapter(
            execution_environment,  # type: ignore[arg-type]
        )

    async def aclose(self) -> None:
        """Close process execution before the durable coordinator drains runs."""
        if self._execution_adapter is not None:
            await self._execution_adapter.aclose()

    def acceptance_research_tools(self) -> tuple[AgentTool, ...]:
        """Return non-resource definitions execution may expose to Research.

        Acceptance combines these exact factories with request-specific search
        and resource tools. The execute closures are never invoked here.
        """
        from dlightrag.engine.agent.environment import AccessScheduler
        from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
        from dlightrag.engine.agent.tools.files import path_tools, read_tool
        from dlightrag.engine.agent.tools.registry import ToolRegistry
        from dlightrag.engine.answer.tools.artifacts import attach_artifact_tool
        from dlightrag.engine.answer.tools.memory import (
            forget_tool,
            recall_memory_tool,
            remember_tool,
        )
        from dlightrag.engine.answer.tools.subagents import subagent_tools

        access = AccessScheduler()

        async def unused_resource_reader(_request: Any, _runtime: Any) -> Any:
            raise RuntimeError("acceptance tool definitions are never executed")

        # Research always creates a ResourceRegistry so direct public-URL reads
        # and later resource cursors have one stable accepted contract.
        tools: list[AgentTool] = [
            read_tool(None, access, resource_reader=unused_resource_reader),
            *self._external_tools,
        ]
        if self._execution_adapter is not None:
            tools.extend(
                tool
                for tool in path_tools(
                    LocalExecutionEnvironment(Path.cwd()),
                    scheduler=access,
                )
                if tool.name != "read"
            )
            tools.append(
                attach_artifact_tool(
                    Path.cwd() / "artifacts",
                    scheduler=access,
                    limits=self._settings.publication,
                )
            )
        tools.extend(subagent_tools(host=SubagentHost()))
        if self._memory_store is not None:
            host = MemoryHost()
            tools.extend(
                (remember_tool(host=host), forget_tool(host=host), recall_memory_tool(host=host))
            )
        if self._skills_bundle_factory is not None:
            # Acceptance needs schemas only: a sentinel owner produces the same
            # tool membership (load always; publish/delete for parents) without
            # touching any owner directory.
            tools.extend(self._skills_bundle_factory("__acceptance__").tools(child=False))
        return ToolRegistry(tools).resolve()

    async def execute(self, session: RunSession) -> RunExecutionOutcome:
        with model_call_scope((session.owner_id, session.run_id)):
            try:
                outcome = await self._execute(session)
            except (
                asyncio.CancelledError,
                RunCancellationObserved,
                LeaseLostError,
                RunExecutionError,
            ):
                raise
            except Exception as exc:
                component = classify_transient_dependency(exc, component_hint="providers")
                if component is not None:
                    await session.check_cancelled()
                    # Clear any durable optimistic draft before releasing the
                    # Query permit. The same Run and Agent Session resume from
                    # their fenced durable state after the bounded delay.
                    await session.reset_output()
                    checkpoint, delay = next_dependency_retry(
                        session.checkpoint,
                        component,
                        base_seconds=_DEPENDENCY_DEFER_BASE_SECONDS,
                        max_seconds=_DEPENDENCY_DEFER_MAX_SECONDS,
                    )
                    self._notify_dependency(self._on_dependency_unavailable, component)
                    return Deferred(
                        checkpoint=checkpoint,
                        next_attempt_at=self._now() + datetime.timedelta(seconds=delay),
                    )
                logger.warning(
                    "Answer run %s execution failed",
                    session.run_id,
                    extra={"error_type": type(exc).__name__},
                )
                message = (
                    exc.public_message
                    if isinstance(exc, AnswerInputError | InvalidToolConfigurationError)
                    and exc.public_message
                    else "Answer run failed."
                )
                raise RunExecutionError(classify_answer_error(exc), message) from exc
            recovered = dependency_component_from_checkpoint(session.checkpoint)
            if recovered is not None:
                self._notify_dependency(self._on_dependency_recovered, recovered)
            return outcome

    @staticmethod
    def _notify_dependency(
        callback: DependencyStateCallback | None,
        component: DependencyComponent,
    ) -> None:
        if callback is not None:
            callback(component)

    async def _ensure_resolved_mode(
        self,
        session: RunSession,
        request: AnswerRunInput,
        *,
        history: Sequence[Mapping[str, Any]],
    ) -> ResolvedMode:
        record = await self._store.load_routing(owner_id=session.owner_id, run_id=session.run_id)
        if record is None:
            raise RunExecutionError("routing_failed", "Routing record is missing.")
        if record.resolved_mode:
            return _require_resolved_mode(record.resolved_mode)
        try:
            decided = decide_resolved_mode(
                requested_mode=record.requested_mode,
                valid_modes=frozenset(record.valid_modes),
            )
        except ValueError as exc:
            raise RunExecutionError("routing_failed", "Answer mode routing failed.") from exc
        if decided is None:
            decided = _require_resolved_mode(
                await self._route_with_model(
                    request,
                    history=history,
                    valid_modes=record.valid_modes,
                )
            )
        written = await self._store.resolve(
            owner_id=session.owner_id,
            run_id=session.run_id,
            worker_id=session.worker_id,
            fencing_epoch=session.fencing_epoch,
            resolved_mode=decided,
        )
        return _require_resolved_mode(written or decided)

    async def _route_with_model(
        self,
        request: AnswerRunInput,
        *,
        history: Sequence[Mapping[str, Any]],
        valid_modes: tuple[str, ...],
    ) -> str:
        model, _telemetry = self._models.new_highlight_model()
        router = AnswerModeRouter(model)
        resources = tuple(
            ModeResource(role=resource_role(filename=item.filename, mime_type=item.mime_type))
            for item in (*request.attachments, *request.history_attachments)
        )
        tools = ["search_knowledge_base"]
        web_sources = self._models.web_sources()
        if web_sources is not None and web_sources.search_enabled:
            tools.append("search_web")
        try:
            return await router.choose(
                query=request.query,
                history=history,
                resources=resources,
                tool_categories=tools,
                has_images=any(item.role == "image" for item in resources),
                valid_modes=valid_modes,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "auto router failed; defaulting to research",
                extra={"valid_modes": list(valid_modes)},
                exc_info=True,
            )
            if "research" in valid_modes:
                return "research"
            raise RunExecutionError("routing_failed", "Answer mode routing failed.") from exc

    async def _compact_fast_history_if_needed(
        self,
        *,
        host: FastSessionHost,
        session_id: SessionId,
        lane_id: LaneId,
        reservation_id: str,
        accepted_user_entry_id: EntryId,
        targets: Sequence[HistoryProjectionTarget],
        compaction_model_profile: ModelProfile,
    ) -> tuple[PriorTurns, dict[str, Any], bool]:
        """Commit one canonical projection satisfying every reachable Fast serializer."""
        snapshot = await host.snapshot(session_id, selected_lane_id=lane_id)
        coordinator = CompactionCoordinator(
            model_profile=compaction_model_profile,
            context_policy=CONTEXT_POLICY,
            stream_model=self._models.query_tool_model().stream_text,
            exchange_starts_func=host_turn_starts,
        )
        failures: list[dict[str, Any]] = []
        refreshed_after_commit_error = False
        tail_reductions = 0
        for attempt in range(1, _FAST_COMPACTION_ATTEMPT_LIMIT + 1):
            history = _project_fast_history_before_current_user(
                snapshot,
                lane_id=lane_id,
                projection=snapshot.active_projection,
                accepted_user_entry_id=accepted_user_entry_id,
            )
            before = _measure_fast_history_targets(history, targets)
            if _fast_history_targets_fit(before):
                recovered_projection = (
                    refreshed_after_commit_error and snapshot.active_projection is not None
                )
                trace = _durable_fast_compaction_trace(snapshot) if recovered_projection else {}
                return history, trace, recovered_projection

            attempt_trace: dict[str, Any] = {}
            tail = CONTEXT_POLICY.retained_tail_target(compaction_model_profile) // (
                2**tail_reductions
            )
            try:
                projection, _outcome = await coordinator.prepare(
                    snapshot,
                    tail_target_tokens=tail,
                    accounted_before=max(item["input_tokens"] for item in before.values()),
                    trace=attempt_trace,
                )
                candidate = _project_fast_history_before_current_user(
                    snapshot,
                    lane_id=lane_id,
                    projection=projection,
                    accepted_user_entry_id=accepted_user_entry_id,
                )
                after = _measure_fast_history_targets(candidate, targets)
                if not _fast_history_targets_fit(after):
                    overflowing = ", ".join(
                        name
                        for name, item in after.items()
                        if item["input_tokens"] > item["input_limit_tokens"]
                    )
                    raise ValueError(
                        f"prepared Fast projection still exceeds targets: {overflowing}"
                    )
            except asyncio.CancelledError, SessionLeaseLostError:
                raise
            except Exception as exc:
                failures.append(_fast_compaction_failure(attempt, "prepare", exc))
                tail_reductions += 1
                continue

            try:
                await host.commit_compaction(
                    snapshot=snapshot,
                    session_id=session_id,
                    lane_id=lane_id,
                    reservation_id=reservation_id,
                    projection=projection,
                )
            except asyncio.CancelledError, SessionLeaseLostError:
                raise
            except Exception as exc:
                failures.append(_fast_compaction_failure(attempt, "commit", exc))
                authoritative = await host.snapshot(
                    session_id,
                    selected_lane_id=lane_id,
                    force_reload=True,
                )
                if _active_fast_compaction(authoritative, projection) is not None:
                    recovered = _project_fast_history_before_current_user(
                        authoritative,
                        lane_id=lane_id,
                        projection=authoritative.active_projection,
                        accepted_user_entry_id=accepted_user_entry_id,
                    )
                    return recovered, _durable_fast_compaction_trace(authoritative), True
                _require_fast_turn_reservation(
                    authoritative,
                    lane_id=lane_id,
                    reservation_id=reservation_id,
                    accepted_user_entry_id=accepted_user_entry_id,
                )
                snapshot = authoritative
                refreshed_after_commit_error = True
                continue

            trace = dict(attempt_trace)
            trace["fast_compaction_attempt"] = attempt
            if failures:
                trace["fast_compaction_retries"] = list(failures)
            trace["fast_compaction_targets"] = {
                name: {
                    "input_tokens_before": before[name]["input_tokens"],
                    "input_tokens_after": after[name]["input_tokens"],
                    "input_limit_tokens": after[name]["input_limit_tokens"],
                }
                for name in before
            }
            return candidate, trace, True

        failure_trace = {"compaction_failed": {"attempts": failures}}
        logger.warning(
            "Fast Session compaction failed",
            extra={"trace": failure_trace, "session_id": session_id.value},
        )
        raise RunExecutionError(
            "compaction_failed",
            "Fast Answer could not compact the conversation within model capacity.",
        )

    async def _execute(self, session: RunSession) -> RunExecutionOutcome:
        request = AnswerRunInput.from_prepared_input(session.prepared_input)
        model_profiles = self.validate_pinned_model_profiles(request)
        agent_session_id = SessionId(request.agent_session_id)
        agent_lane_id = LaneId(request.agent_lane_id)
        repository = session.execution.session_repository
        progress_store = session.execution.progress_store
        workspace_store = session.execution.workspace_store
        if repository is None or progress_store is None:
            raise RunExecutionError(
                "run_execution_failed",
                "Answer execution state is unavailable.",
            )
        loaded_snapshot = await repository.load(agent_session_id)
        canonical_snapshot = await _reserve_agent_session_boundary(
            repository,
            session_id=agent_session_id,
            fencing_epoch=session.execution.fencing_epoch,
            previous=loaded_snapshot,
        )
        lane_ids = {lane.lane_id for lane in canonical_snapshot.tree.lanes}
        source_lane_id = LaneId(request.source_lane_id) if request.source_lane_id else None
        history_lane_id = (
            agent_lane_id
            if agent_lane_id in lane_ids
            else source_lane_id
            if source_lane_id in lane_ids
            else LaneId.main()
        )
        selected_snapshot = replace(
            canonical_snapshot,
            selected_lane_id=history_lane_id,
        )
        authoritative_messages = project_session_messages(
            canonical_snapshot.tree.ancestry(history_lane_id),
            selected_snapshot.active_projection,
        )
        has_agent_history = bool(authoritative_messages)
        if has_agent_history:
            routing_history = PriorTurns(authoritative_messages)
        else:
            routing_history = PriorTurns(
                [dict(message) for message in request.history],
                episodic_summary=request.episodic_summary,
            )
        await session.enter_phase("routing")
        resolved_mode = await self._ensure_resolved_mode(
            session,
            request,
            history=routing_history.messages,
        )
        await session.enter_phase("planning")
        projected_history = (
            PriorTurns(authoritative_messages)
            if has_agent_history and resolved_mode == "fast"
            else PriorTurns()
            if has_agent_history
            else routing_history
        )

        fast_boundaries: FastRunBoundaries | None = None
        fast_session_host: FastSessionHost | None = None
        fast_compaction_trace: dict[str, Any] = {}
        fast_reservation_active = False
        agent_runtime: AgentSessionRuntime[EffectHostUpdate] | None = None
        research_operation_id: OperationId | None = None
        research_plan: AgentRunPlan | None = None
        agent_operations: list[dict[str, Any]] = []

        fetched_buffer = FetchedResourceBuffer()

        run = await self.prepare_orchestrated_run(
            query=request.query,
            workspaces=list(request.workspaces),
            retrieval=request.retrieval,
            filters=MetadataFilter.model_validate(request.filters) if request.filters else None,
            resources=await self._answer_run_resources(request, owner_id=session.owner_id),
            fetched_bytes_sink=_buffered_fetched_bytes_sink(fetched_buffer),
            resolved_mode=resolved_mode,
            resource_scope=f"{session.owner_id}\0{session.run_id}",
            pinned_image_descriptions=request.image_descriptions,
            projected_history=projected_history,
            model_profiles=model_profiles,
            skills=(
                self._skills_bundle_factory(session.owner_id, request.requested_skill)
                if self._skills_bundle_factory is not None
                else None
            ),
        )
        attachment_snapshots: dict[str, bytes] = {}
        if run.registry is not None:
            attachment_snapshots = await self._restore_registry_fetches(
                run.registry,
                owner_id=session.owner_id,
                run_id=str(session.run_id),
            )
        attachment_admissions = run.orchestrator.admit_durable_attachments(
            authoritative_messages,
            attachment_snapshots,
        )
        auth_mode = str((session.prepared_input or {}).get("auth_mode") or "none")
        prepared_input = session.prepared_input or {}
        recall_allowed = resolved_mode == "research" and bool(
            prepared_input.get("profile_memory_enabled", True)
        )
        memory_epoch = int(prepared_input.get("profile_memory_epoch") or 0)
        memory_recall_record_count = 0
        memory_recall_chars = 0
        if self._memory is None or not memory_owner_allowed(auth_mode):
            recall_allowed = False
        elif recall_allowed and self._memory_capability_current is not None:
            recall_allowed = await self._memory_capability_current(
                owner_id=session.owner_id, epoch=memory_epoch
            )
        elif recall_allowed:
            recall_allowed = await _memory_recall_allowed(
                self._memory_recall_enabled, owner_id=session.owner_id
            )
        stream: AsyncIterator[str] | None = None
        try:
            await ensure_session_lane(
                repository=repository,
                snapshot=canonical_snapshot,
                fencing_epoch=session.execution.fencing_epoch,
                session_id=agent_session_id,
                lane_id=agent_lane_id,
                source_lane_id=(
                    LaneId(request.source_lane_id) if request.source_lane_id is not None else None
                ),
            )
            prepared_early: Any = None
            if resolved_mode == "research":
                from dlightrag.engine.answer.execution_settings import validate_agent_execution

                root = validate_agent_execution(
                    execution_environment=self._execution_environment,
                    workspace_root=self._workspace_root_setting,
                    working_dir=self._working_dir,
                    sandbox_adapter=(
                        self._execution_adapter
                        if self._execution_environment == "sandbox"
                        else None
                    ),
                )
                if root is not None:
                    try:
                        bound = await bind_run_workspace(
                            workspace_root=root,
                            owner_id=session.owner_id,
                            run_id=session.run_id,
                            fencing_epoch=session.execution.fencing_epoch,
                            recorded_epoch=session.workspace_epoch,
                            store=workspace_store,
                            execution_adapter=self._execution_adapter,
                        )
                    except WorkspaceRecoveryFailed as exc:
                        raise RunExecutionError("workspace_recovery_failed", str(exc)) from exc
                    except WorkspaceIntegrityError as exc:
                        raise RunExecutionError("workspace_integrity_error", str(exc)) from exc
                    run.orchestrator.bind_workspace(bound)
                session_id = agent_session_id
                store = self._store
                run.orchestrator.bind_memory(
                    owner_id=session.owner_id,
                    auth_mode=str((session.prepared_input or {}).get("auth_mode") or "none"),
                    run_id=session.run_id,
                    session_id=session_id.value,
                    store=self._memory_store,
                    enabled=recall_allowed,
                    epoch=memory_epoch,
                    capability_current=self._memory_capability_current,
                )
                persist_child_runtime = _fenced_child_writer(store, "upsert_child_session", session)
                claim_child = _fenced_child_writer(store, "claim_child_session", session)
                renew_child = _fenced_child_writer(store, "heartbeat_child_session", session)
                if persist_child_runtime is None or claim_child is None or renew_child is None:
                    raise RunExecutionError(
                        "run_execution_failed",
                        "Child Session persistence is unavailable.",
                    )
                run.orchestrator.bind_subagents(
                    parent_session_id=session_id,
                    run_id=session.run_id,
                    owner_id=session.owner_id,
                    persist=_fenced_child_writer(store, "upsert_child_session", session),
                    load_child=_async_store_method(store, "load_child_session"),
                    finish_child=_fenced_child_writer(store, "finish_child_session", session),
                    run_child=_bound_child_runner(
                        orchestrator=run.orchestrator,
                        repository=repository,
                        session=session,
                        fetched_buffer=fetched_buffer,
                        parent_session_id=session_id,
                        persist_child_runtime=persist_child_runtime,
                        claim_child=claim_child,
                        renew_child=renew_child,
                    ),
                    check_cancelled=session.check_cancelled,
                )
                # Resolve and compare every accepted execution pin before the
                # first Session mutation. The later post-recall comparison is
                # a second guard immediately before provider/tool effects.
                pin_probe = run.orchestrator.prepare_run(
                    request.query,
                    conversation_history=run.history,
                    query_images=run.query_images,
                    registry=run.registry,
                )
                self.validate_pinned_model_profiles(request)
                self.validate_pinned_agent_run_plan(request, pin_probe.tools)
                snapshot = await repository.refresh(
                    session_id,
                    previous=canonical_snapshot,
                )
                validate_snapshot_refresh(
                    session_id,
                    previous=canonical_snapshot,
                    snapshot=snapshot,
                )
                is_new_session = snapshot.commit_sequence == 0
                memory_text = ""
                if self._memory is not None and recall_allowed:
                    recalled = await self._memory.recall(
                        owner_id=session.owner_id,
                        query=request.query,
                    )
                    memory_text = render_auto_recall(recalled.records)
                    memory_recall_record_count = len(recalled.records)
                    memory_recall_chars = recalled.content_chars
                run.orchestrator.bind_recall(memory_text)
                prepared_early = run.orchestrator.prepare_run(
                    request.query,
                    conversation_history=run.history,
                    query_images=run.query_images,
                    registry=run.registry,
                    attachment_snapshots=attachment_snapshots,
                    attachment_admissions=attachment_admissions,
                )
                self.validate_pinned_model_profiles(request)
                self.validate_pinned_agent_run_plan(request, prepared_early.tools)
                if not is_new_session:
                    await _restore_durable_evidence(prepared_early, repository, session_id)
                    if run.registry is not None:
                        run.registry.restore_discovered_resources(prepared_early.evidence.contexts)
                plan = request.agent_run_plan
                if plan is None:
                    raise RunExecutionError(
                        "run_execution_failed",
                        "Research answer run is missing its accepted Agent Plan",
                    )
                research_plan = plan

                def validate_research_pins() -> None:
                    self.validate_pinned_model_profiles(request)
                    self.validate_pinned_agent_run_plan(request, prepared_early.tools)

                effects = ResearchRuntimeEffects(
                    orchestrator=run.orchestrator,
                    prepared=prepared_early,
                    session=session,
                    session_id=session_id,
                    fetched_buffer=fetched_buffer,
                    persist_child_intent=_fenced_child_writer(
                        store, "upsert_child_session", session
                    ),
                    validate_pins=validate_research_pins,
                    publish_provider_text=True,
                )
                control_reader = _fenced_control_reader(store, session)
                control_ack = _fenced_control_ack(store, session)
                controls = (
                    AnswerRuntimeControls(reader=control_reader, acknowledge=control_ack)
                    if control_reader is not None and control_ack is not None
                    else None
                )
                agent_runtime = AgentSessionRuntime(
                    repository=repository,
                    effects=effects,
                    tools=prepared_early.tools,
                    fencing_epoch=session.execution.fencing_epoch,
                    provider_attempt_limit=plan.provider_attempt_limit,
                    event_sink=_answer_runtime_event_sink(session),
                    controls=controls,
                    initial_snapshot=AgentSessionSnapshotSeed(
                        repository=repository,
                        session_id=session_id,
                        snapshot=snapshot,
                    ),
                )
                accepted = await agent_runtime.accept(
                    session_id=session_id,
                    lane_id=agent_lane_id,
                    idempotency_key=f"answer-run:{session.run_id}",
                    content=request.query,
                    plan=plan,
                )
                research_operation_id = accepted.operation_id
                await session.enter_phase("researching")
                while True:
                    usage_floor = accepted.cursor.last_entry_sequence
                    operation = await _drive_answer_operation(
                        agent_runtime,
                        session=session,
                        session_id=session_id,
                        operation_id=accepted.operation_id,
                    )
                    if not isinstance(operation.state, OperationCompleted):
                        raise RunExecutionError(
                            "run_execution_failed",
                            f"Research Agent operation ended as {operation.state.state_type}.",
                        )
                    snapshot = operation.context.snapshot
                    operation_usage = (
                        _usage_from_snapshot_entries(
                            snapshot_entries=(
                                entry for entry in snapshot.entries if entry.sequence > usage_floor
                            )
                        )
                        or {}
                    )
                    agent_operations.append(
                        {
                            "operation_id": accepted.operation_id.value,
                            "purpose": "research" if not agent_operations else "follow_up",
                            "status": "completed",
                            "usage": operation_usage,
                        }
                    )
                    next_input = _oldest_pending_input(snapshot, agent_lane_id)
                    command_ids: tuple[str, ...] = ()
                    if next_input is None and controls is not None:
                        commands = await controls.poll(operation.context)
                        if commands:
                            command = commands[0]
                            command_ids = (command.command_id,)
                            if isinstance(command, FollowUpCommand):
                                next_input = (
                                    command.idempotency_key,
                                    command.content,
                                )
                            else:
                                next_input = (command.command_id, command.content)
                    if next_input is None:
                        break
                    if len(agent_operations) >= 1 + plan.max_pending_follow_ups:
                        raise RunExecutionError(
                            "run_execution_failed",
                            "Research linked-operation bound was exhausted.",
                        )
                    validate_research_pins()
                    if prepared_early.streamed_terminal_text is not None:
                        await session.reset_output()
                        prepared_early.streamed_terminal_text = None
                        await session.enter_phase("researching")
                    accepted = await agent_runtime.accept(
                        session_id=session_id,
                        lane_id=agent_lane_id,
                        idempotency_key=next_input[0],
                        content=next_input[1],
                        plan=plan,
                    )
                    research_operation_id = accepted.operation_id
                    if command_ids and controls is not None:
                        if not await controls.acknowledge(command_ids):
                            raise LeaseLostError
                run.orchestrator.restore_runtime_snapshot(prepared_early, snapshot)
            else:
                fast_boundaries = FastRunBoundaries(
                    session=session,
                    progress=progress_store,
                    run_id=session.run_id,
                    initial_progress_version=session.durable_progress_version,
                    plan={
                        "query": request.query,
                        "workspaces": list(request.workspaces),
                        "top_k": request.retrieval.top_k,
                        "chunk_top_k": request.retrieval.chunk_top_k,
                    },
                )
                fast_session_host = FastSessionHost(
                    repository=repository,
                    initial_snapshot=canonical_snapshot,
                    load_settled_result=fast_boundaries.load_settled_result,
                    fencing_epoch=session.execution.fencing_epoch,
                )
                fast_turn = await fast_session_host.accept(
                    session_id=agent_session_id,
                    lane_id=agent_lane_id,
                    reservation_id=session.run_id,
                    idempotency_key=request.idempotency_fingerprint,
                    content=request.query,
                )
                if fast_turn.progress_advanced:
                    fast_boundaries.observe_session_progress()
                if fast_turn.settled_payload is not None:
                    stored = dict(fast_turn.settled_payload)
                    terminal = await fast_boundaries.settle_final(
                        result=stored,
                        result_digest=canonical_json(stored),
                    )
                    return AlreadyCommittedTerminal(terminal)
                fast_reservation_active = True
                if not fast_turn.created:
                    replay_snapshot = await fast_session_host.snapshot(
                        agent_session_id,
                        selected_lane_id=agent_lane_id,
                    )
                    run.history = _project_fast_history_before_current_user(
                        replay_snapshot,
                        lane_id=agent_lane_id,
                        projection=replay_snapshot.active_projection,
                        accepted_user_entry_id=fast_turn.user_entry_id,
                    )
                    if replay_snapshot.active_projection is not None:
                        fast_compaction_trace.update(
                            _durable_fast_compaction_trace(replay_snapshot)
                        )
                if has_agent_history:
                    (
                        compacted_history,
                        compaction_trace,
                        compacted,
                    ) = await self._compact_fast_history_if_needed(
                        host=fast_session_host,
                        session_id=agent_session_id,
                        lane_id=agent_lane_id,
                        reservation_id=session.run_id,
                        accepted_user_entry_id=fast_turn.user_entry_id,
                        targets=run.fast_history_targets,
                        compaction_model_profile=model_profiles["query"],
                    )
                    run.history = compacted_history
                    if "fast_compaction_attempt" in compaction_trace:
                        fast_compaction_trace.clear()
                    fast_compaction_trace.update(compaction_trace)
                    if compacted:
                        fast_boundaries.observe_session_progress()
                await fast_boundaries.settle_planner()

            async with self._telemetry.observe(
                "answer_orchestration",
                as_type="chain",
                input={"query": request.query},
                metadata={
                    "run_id": session.run_id,
                    "resolved_mode": resolved_mode,
                    "workspaces": run.workspaces,
                    "history_turns": len(run.history or []),
                    "query_image_count": run.current_image_count,
                    "semantic_highlights": request.semantic_highlights,
                },
            ) as pipeline_trace:
                prepared = prepared_early
                stream: AsyncIterator[str] | None = None
                if resolved_mode == "research":
                    if prepared is None:
                        raise RunExecutionError(
                            "run_execution_failed",
                            "Research Runtime lost its prepared Host state.",
                        )
                    contexts, answer_text, already_streamed = run.orchestrator.runtime_answer(
                        prepared
                    )
                    if not already_streamed:
                        await session.enter_phase("generating")
                        await session.emit_token(answer_text)
                else:
                    contexts, stream = await run.orchestrator.answer_stream(
                        request.query,
                        conversation_history=run.history,
                        query_images=run.query_images,
                        boundaries=fast_boundaries,
                    )
                    answer_parts: list[str] = []
                    if stream is not None:
                        async for chunk in stream:
                            answer_parts.append(chunk)
                            await session.emit_token(chunk)
                    answer_text = getattr(stream, "answer", "") or "".join(answer_parts)
                emitted_answer_text = answer_text
                await session.flush_tokens()
                finalized = finalize_answer(answer_text, contexts)
                artifact_root = run.orchestrator.artifact_root()
                artifact_attachments = (
                    _publication_attachments(
                        await self._store.list_artifact_attachments(
                            owner_id=session.owner_id,
                            run_id=session.run_id,
                        )
                    )
                    if artifact_root is not None
                    else ()
                )
                publication = _publication_plan(
                    artifact_root,
                    answer=finalized.answer,
                    attachments=artifact_attachments,
                    limits=self._settings.publication,
                )
                finalized.answer = publication.answer
                if (
                    publication.issues
                    and agent_runtime is not None
                    and research_plan is not None
                    and prepared_early is not None
                ):
                    await session.reset_output()
                    prepared_early.streamed_terminal_text = None
                    await session.enter_phase("researching")
                    correction = await agent_runtime.accept(
                        session_id=agent_session_id,
                        lane_id=agent_lane_id,
                        idempotency_key=f"publication-correction:{session.run_id}",
                        content=publication.correction_feedback(),
                        plan=research_plan,
                    )
                    correction_usage_floor = correction.cursor.last_entry_sequence
                    corrected = await _drive_answer_operation(
                        agent_runtime,
                        session=session,
                        session_id=agent_session_id,
                        operation_id=correction.operation_id,
                    )
                    if not isinstance(corrected.state, OperationCompleted):
                        raise RunExecutionError(
                            "run_execution_failed",
                            "Publication correction Agent operation did not complete.",
                        )
                    corrected_snapshot = corrected.context.snapshot
                    correction_usage = (
                        _usage_from_snapshot_entries(
                            snapshot_entries=(
                                entry
                                for entry in corrected_snapshot.entries
                                if entry.sequence > correction_usage_floor
                            )
                        )
                        or {}
                    )
                    correction_record = {
                        "operation_id": correction.operation_id.value,
                        "purpose": "publication_correction",
                        "status": "completed",
                        "usage": correction_usage,
                    }
                    agent_operations.append(correction_record)
                    research_operation_id = correction.operation_id
                    run.orchestrator.restore_runtime_snapshot(
                        prepared_early,
                        corrected_snapshot,
                    )
                    contexts, answer_text, already_streamed = run.orchestrator.runtime_answer(
                        prepared_early
                    )
                    if not already_streamed:
                        await session.enter_phase("generating")
                        await session.emit_token(answer_text)
                    emitted_answer_text = answer_text
                    await session.flush_tokens()
                    finalized = finalize_answer(answer_text, contexts)
                    artifact_root = run.orchestrator.artifact_root()
                    artifact_attachments = (
                        _publication_attachments(
                            await self._store.list_artifact_attachments(
                                owner_id=session.owner_id,
                                run_id=session.run_id,
                            )
                        )
                        if artifact_root is not None
                        else ()
                    )
                    publication = _publication_plan(
                        artifact_root,
                        answer=finalized.answer,
                        attachments=artifact_attachments,
                        limits=self._settings.publication,
                    )
                    finalized.answer = publication.answer
                    correction_record["publication_outcome"] = publication.outcome
                if resolved_mode == "research" and finalized.answer != emitted_answer_text:
                    await session.reset_output()
                    if prepared_early is not None:
                        prepared_early.streamed_terminal_text = None
                    await session.enter_phase("generating")
                    await session.emit_token(finalized.answer)
                    await session.flush_tokens()
                    emitted_answer_text = finalized.answer
                if request.semantic_highlights:
                    finalized.sources = await enrich_semantic_highlights(
                        finalized.sources,
                        answer_text=finalized.answer,
                        settings=self._settings.semantic_highlights,
                        model_factory=self._models.new_highlight_model,
                    )
                trace = dict(
                    prepared.trace
                    if resolved_mode == "research" and prepared is not None
                    else getattr(stream, "trace", None) or {}
                )
                if fast_compaction_trace:
                    trace.update(fast_compaction_trace)
                if agent_runtime is not None and research_operation_id is not None:
                    root_usage: dict[str, int] = {}
                    for item in agent_operations:
                        for key, value in item["usage"].items():
                            root_usage[key] = root_usage.get(key, 0) + int(value)
                    child_usage = await _durable_child_usage(
                        self._store,
                        owner_id=session.owner_id,
                        run_id=session.run_id,
                    )
                    if not child_usage:
                        child_usage = {
                            str(key): int(value)
                            for key, value in (trace.get("child_usage") or {}).items()
                            if isinstance(value, int)
                        }
                    inclusive = dict(root_usage)
                    for key, value in child_usage.items():
                        inclusive[key] = inclusive.get(key, 0) + value
                    trace["usage"] = {
                        "usage_details": root_usage,
                        "child_usage_details": child_usage,
                        "inclusive_usage_details": inclusive,
                    }
                    trace["agent_operations"] = list(agent_operations)
                trace.setdefault("agent_operations", list(agent_operations))
                trace["query_image_description_count"] = len(run.image_descriptions)
                trace["memory_recall_record_count"] = memory_recall_record_count
                trace["memory_recall_chars"] = memory_recall_chars
                images = evidence_images_from_sources(finalized.sources, contexts=contexts)
                pipeline_trace.update(
                    output=answer_trace_output(
                        finalized.answer,
                        finalized.sources,
                        contexts,
                        capture_sensitive_data=self._telemetry.capture_sensitive_data,
                    )
                )
                publications, artifact_descriptors, artifact_sources = _stage_publications(
                    plan=publication,
                    answer=finalized.answer,
                    contexts=contexts,
                    require_answer=getattr(prepared_early, "stop_reason", None) == "model_stop",
                )
                # Fast terminal settlement has no publication channel; Research
                # leaves publication ownership with the coordinator.
                session.pending_publications = publications if fast_boundaries is None else []
                stored = store_answer_result(
                    answer=finalized.answer,
                    contexts=project_contexts_for_client(contexts),
                    sources=finalized.sources,
                    evidence_images=images,
                    trace=trace,
                    image_descriptions=run.image_descriptions,
                    artifacts=artifact_descriptors,
                    artifact_outcome=publication.outcome,
                    artifact_sources=artifact_sources,
                )
                if fast_boundaries is not None:
                    await fast_boundaries.settle_retrieval(contexts)
                    await fast_boundaries.stage_result(
                        result=stored,
                        result_digest=canonical_json(stored),
                    )
                if fast_session_host is not None:
                    fast_commit = await fast_session_host.complete(
                        session_id=agent_session_id,
                        lane_id=agent_lane_id,
                        reservation_id=session.run_id,
                        content=finalized.answer,
                        usage=(
                            trace.get("usage") if isinstance(trace.get("usage"), Mapping) else None
                        ),
                    )
                    fast_reservation_active = False
                    if fast_boundaries is not None and fast_commit is not None:
                        fast_boundaries.observe_session_progress()
                if fast_boundaries is not None:
                    terminal = await fast_boundaries.settle_final(
                        result=stored,
                        result_digest=canonical_json(stored),
                    )
                    return AlreadyCommittedTerminal(terminal)
                return Succeeded(stored)
        except BaseException:
            if fast_session_host is not None and fast_reservation_active:
                try:
                    await fast_session_host.fail(
                        session_id=agent_session_id,
                        lane_id=agent_lane_id,
                        reservation_id=session.run_id,
                    )
                except Exception:
                    logger.exception("Failed to clear Fast Host turn reservation")
            raise
        finally:
            await _close_execution_resources(stream, run.registry)

    async def prepare_orchestrated_run(
        self,
        *,
        query: str,
        workspaces: list[str],
        retrieval: RetrievalOptions,
        filters: MetadataFilter | None,
        resources: list[ResourceInput] | None,
        fetched_bytes_sink: FetchedBytesSink | None = None,
        pinned_image_descriptions: tuple[str, ...],
        projected_history: PriorTurns,
        model_profiles: Mapping[ModelRole, ModelProfile],
        environment: ExecutionEnvironment | None = None,
        resolved_mode: ResolvedMode,
        resource_scope: str,
        skills: SkillsBundle | None = None,
    ) -> OrchestratorRun:
        history = projected_history
        models = self._capabilities.request_model_context(model_profiles)
        query_profile = models.query
        if not workspaces:
            raise ValueError("an Answer run requires at least one workspace")
        self._warm(workspaces)
        text_window_budget = TextWindowBudget(CONTEXT_POLICY.hard_input_limit(query_profile))
        resolved = await self._resources.resolve(
            resources,
            models=models,
            text_window_budget=text_window_budget,
            confirm_image_context=self._capabilities.pinned_answer_context,
            fetched_bytes_sink=fetched_bytes_sink,
            resolved_mode=resolved_mode,
            resource_scope=resource_scope,
        )
        try:
            models = resolved.models
            query_profile = models.query
            image_descriptions = list(pinned_image_descriptions)
            fast_history_targets: tuple[HistoryProjectionTarget, ...] = ()
            if resolved_mode == "fast":
                planner_measure = await self._planner_history_input_measure(
                    query=query,
                    workspaces=tuple(workspaces),
                    model_profile=models.extract,
                    current_image_descriptions=image_descriptions,
                    preserve_query=None,
                )
                synthesizer = self._models.answer_synthesizer(models.query)
                generation_measure = (
                    synthesizer.history_input_measure(
                        query,
                        current_images=resolved.current_images,
                    )
                    if resolved.current_images
                    else synthesizer.history_input_measure(query)
                )
                fast_history_targets = (
                    HistoryProjectionTarget(
                        "planner",
                        models.extract,
                        planner_measure,
                        proactive_compaction=True,
                        require_full_dynamic_reserve=True,
                    ),
                    HistoryProjectionTarget(
                        "fast_generation",
                        models.query,
                        generation_measure,
                        proactive_compaction=True,
                        require_full_dynamic_reserve=True,
                    ),
                )
            orchestrated_run: OrchestratorRun | None = None

            async def retrieve_knowledge_base(search_query: str) -> RetrievalResult:
                active_history = (
                    orchestrated_run.history if orchestrated_run is not None else history
                )
                return await self._retrieve_result(
                    search_query,
                    workspaces=workspaces,
                    conversation_history=active_history.messages,
                    retrieval=retrieval,
                    filters=filters,
                    query_images=resolved.current_images,
                    image_descriptions=image_descriptions,
                    preserve_query=True if resolved_mode == "research" else None,
                    model_profile=models.extract,
                )

            model_func: Callable[..., Any] | None = None
            stream_model_func: Callable[..., AsyncIterator[str]] | None = None
            if resolved_mode == "research":
                tool_model = self._models.query_tool_model()
                model_func = tool_model
                stream_model_func = tool_model.stream_text

            def resolve_child_model(
                role: str,
            ) -> tuple[Callable[..., Any], Callable[..., AsyncIterator[str]], ModelProfile]:
                if role not in {"query", "extract"}:
                    raise ValueError(f"unknown child model role: {role}")
                selected_role: ModelRole = role  # type: ignore[assignment]
                profile = models.query if role == "query" else models.extract
                selected = self._models.tool_model(selected_role)
                return selected, selected.stream_text, profile

            orchestrator = AnswerOrchestrator(
                synthesizer=self._models.answer_synthesizer(query_profile),
                retrieve_knowledge_base=retrieve_knowledge_base,
                search_web=(
                    resolved.web_sources.search
                    if resolved.web_sources is not None and resolved.web_sources.search_enabled
                    else None
                ),
                model_func=model_func,
                stream_model_func=stream_model_func,
                resource_tools=[*resolved.resource_tools, *self._external_tools],
                resource_manifest=resolved.resource_manifest,
                register_web_source=(
                    resolved.registry.register_discovered_link
                    if resolved.registry is not None and resolved.web_sources is not None
                    else None
                ),
                image_budget=resolved.image_budget,
                text_window_budget=text_window_budget,
                model_profile=query_profile,
                context_policy=CONTEXT_POLICY,
                publication_limits=self._settings.publication,
                telemetry=self._telemetry,
                environment=environment,
                search_toolchain=self._search_toolchain,
                resolved_mode=resolved_mode,
                subagent_host=SubagentHost() if resolved_mode == "research" else None,
                memory_host=(
                    MemoryHost()
                    if resolved_mode == "research" and self._memory_store is not None
                    else None
                ),
                resource_reader=(
                    make_resource_reader(resolved.registry, text_window_budget)
                    if resolved.registry is not None
                    else None
                ),
                child_model_resolver=resolve_child_model,
                skills=skills,
            )
            orchestrated_run = OrchestratorRun(
                orchestrator=orchestrator,
                image_descriptions=image_descriptions,
                query_images=resolved.query_images,
                history=history,
                fast_history_targets=fast_history_targets,
                current_image_count=resolved.current_image_count,
                workspaces=workspaces,
                registry=resolved.registry,
            )
            return orchestrated_run
        except BaseException:
            if resolved.registry is not None:
                await resolved.registry.aclose()
            raise

    async def _restore_registry_fetches(
        self,
        registry: ResourceRegistry,
        *,
        owner_id: str,
        run_id: str,
    ) -> dict[str, bytes]:
        attachment_snapshots: dict[str, bytes] = {}
        for resource in await self._store.list_fetched_resources(
            owner_id=owner_id,
            run_id=run_id,
        ):
            pieces: list[bytes] = []
            async for piece in self._blob_store.stream(
                owner_id=owner_id,
                digest=resource.digest,
            ):
                pieces.append(piece)
            if not pieces:
                raise RunExecutionError(
                    "run_execution_failed",
                    "A durable Web resource representation no longer exists.",
                )
            content = b"".join(pieces)
            capabilities = resource.capabilities
            attachment_snapshots[resource.resource_id] = content
            if capabilities.get("resource_kind") == "tool_attachment":
                continue
            raw_aliases = capabilities.get("resource_aliases", [])
            if (
                not isinstance(raw_aliases, list)
                or len(raw_aliases) > 64
                or not all(isinstance(alias, str) for alias in raw_aliases)
            ):
                raise RunExecutionError(
                    "run_execution_failed",
                    "A durable Web resource catalog entry is invalid.",
                )
            origin = str(capabilities.get("admission_origin") or "")
            if origin not in {"caller", "search", "agent"}:
                raise RunExecutionError(
                    "run_execution_failed",
                    "A durable Web resource catalog entry is invalid.",
                )
            acquisition = str(capabilities.get("acquisition") or "")
            if acquisition not in {"direct_http", "exa_extract", "tavily_extract"}:
                raise RunExecutionError(
                    "run_execution_failed",
                    "A durable Web resource catalog entry is invalid.",
                )
            try:
                registry.restore_fetched_resource(
                    resource_id=resource.resource_id,
                    ordinal=resource.ordinal,
                    filename=resource.filename,
                    mime_type=resource.mime_type,
                    url=resource.source_locator.decode("utf-8"),
                    content=content,
                    admission_origin=origin,  # type: ignore[arg-type]
                    acquisition=acquisition,
                    aliases=tuple(raw_aliases),
                )
            except (UnicodeError, ValueError, ResourceRegistryError) as exc:
                raise RunExecutionError(
                    "run_execution_failed",
                    "A durable Web resource catalog entry is invalid.",
                ) from exc
        return attachment_snapshots

    async def _answer_run_resources(
        self,
        request: AnswerRunInput,
        *,
        owner_id: str,
    ) -> list[ResourceInput] | None:
        if not request.links and not request.attachments and not request.history_attachments:
            return None

        async def load(digest: str) -> bytes:
            pieces: list[bytes] = []
            async for piece in self._blob_store.stream(owner_id=owner_id, digest=digest):
                pieces.append(piece)
            if not pieces:
                raise RunExecutionError(
                    "run_execution_failed",
                    "Answer run attachment bytes no longer exist.",
                )
            return b"".join(pieces)

        def loader(digest: str) -> Callable[[], Awaitable[bytes]]:
            async def read() -> bytes:
                return await load(digest)

            return read

        resources = await build_current_answer_resources(
            links=request.links,
            attachments=request.attachments,
            attachment_loaders=[loader(attachment.digest) for attachment in request.attachments],
        )
        resources.extend(
            ResourceInput(
                filename=attachment.filename,
                declared_mime=attachment.mime_type,
                loader=loader(attachment.digest),
            )
            for attachment in request.history_attachments
        )
        return resources

    async def _load_corpus_image(self, workspace: str, chunk_id: str) -> str | None:
        try:
            runtime = await self._pool.acquire(workspace)
            asset = await runtime.aget_visual_asset(chunk_id, size="full")
        except Exception:
            logger.info(
                "Knowledge-base visual for '%s' no longer resolves; dropping the image block",
                safe_log_text(chunk_id),
            )
            return None
        content = getattr(asset, "data", None)
        return base64.b64encode(content).decode("ascii") if content else None

    @staticmethod
    def validate_pinned_agent_run_plan(
        request: AnswerRunInput,
        tools: Sequence[AgentTool],
    ) -> None:
        """Reject execution when runtime tools differ from acceptance."""
        pinned = request.agent_run_plan
        if pinned is None:
            raise IncompatibleActiveRunError(
                "Research answer run is missing its accepted Agent Plan"
            )
        actual = AgentRunPlan.from_tools(
            tools,
            model_role="query",
            context_policy_revision=request.context_policy_revision,
        )
        if (
            actual.model_role != pinned.model_role
            or actual.context_policy_revision != pinned.context_policy_revision
            or actual.tools != pinned.tools
        ):
            raise IncompatibleActiveRunError(
                "answer run Agent Plan differs from its accepted tool contracts"
            )

    def validate_pinned_model_profiles(
        self,
        request: AnswerRunInput,
    ) -> dict[ModelRole, ModelProfile]:
        # Capacity is recalculated from the pinned model facts for each segment.
        # A global arithmetic revision is not a reason to strand an otherwise
        # replayable run.
        pinned = {item.role: item for item in request.pinned_models}
        if len(request.pinned_models) != len(MODEL_ROLE_NAMES) or set(pinned) != set(
            MODEL_ROLE_NAMES
        ):
            raise IncompatibleActiveRunError(
                "answer run does not contain the complete pinned model role set"
            )
        if request.context_policy_revision != CONTEXT_POLICY_REVISION:
            raise IncompatibleActiveRunError("answer run uses another context policy revision")
        if request.model_catalog_revision != current_model_catalog_revision():
            raise IncompatibleActiveRunError("answer run uses another model catalog revision")
        if any(
            pinned[role].fingerprint != self._model_fingerprint_for_role(role)
            for role in MODEL_ROLE_NAMES
        ):
            raise IncompatibleActiveRunError(
                "answer run targets another model endpoint configuration"
            )
        return {role: pinned[role].profile for role in MODEL_ROLE_NAMES}


def _measure_fast_history_targets(
    history: PriorTurns,
    targets: Sequence[HistoryProjectionTarget],
) -> dict[str, dict[str, int]]:
    """Measure the authoritative history with every exact Fast serializer."""
    measured: dict[str, dict[str, int]] = {}
    for target in targets:
        if target.name in measured:
            raise ValueError(f"duplicate Fast history target: {target.name}")
        limit = (
            CONTEXT_POLICY.compaction_trigger(
                target.profile,
                require_full_dynamic_reserve=target.require_full_dynamic_reserve,
            )
            if target.proactive_compaction
            else CONTEXT_POLICY.hard_input_limit(target.profile)
        )
        measured[target.name] = {
            "input_tokens": target.measure_input(
                history.messages,
                history.episodic_summary,
            ),
            "input_limit_tokens": limit,
        }
    return measured


def _fast_history_targets_fit(measured: Mapping[str, Mapping[str, int]]) -> bool:
    return all(item["input_tokens"] <= item["input_limit_tokens"] for item in measured.values())


def _fast_compaction_failure(
    attempt: int,
    stage: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "stage": stage,
        "error_type": type(error).__name__,
        "detail": safe_log_text(str(error)),
    }


def _active_fast_compaction(
    snapshot: Any,
    projection: ContextProjection,
) -> CompactionEntry | None:
    """Return the active checkpoint only when it exactly materializes a projection."""
    if snapshot.active_projection != projection:
        return None
    ancestry = snapshot.graph.ancestry()
    latest = ancestry[-1] if ancestry else None
    if not isinstance(latest, CompactionEntry):
        return None
    if (
        latest.projection_id != projection.projection_id
        or latest.summary != projection.summary
        or latest.covered_through_sequence != projection.covered_through_sequence
        or latest.first_retained_sequence != projection.first_retained_sequence
        or latest.covered_through_entry_id != projection.covered_through_entry_id
        or latest.first_retained_entry_id != projection.first_retained_entry_id
        or latest.source_digest != projection.source_digest
    ):
        return None
    return latest


def _durable_fast_compaction_trace(snapshot: Any) -> dict[str, Any]:
    """Reconstruct only coverage facts that the durable checkpoint actually stores."""
    projection = snapshot.active_projection
    if projection is None:
        return {}
    entry = _active_fast_compaction(snapshot, projection)
    if entry is None:
        return {}
    return {
        "fast_compaction_recovered": True,
        "fast_compaction_coverage": {
            "projection_id": projection.projection_id.value,
            "covered_through_sequence": projection.covered_through_sequence,
            "first_retained_sequence": projection.first_retained_sequence,
        },
    }


async def _reserve_agent_session_boundary(
    repository: Any,
    *,
    session_id: SessionId,
    fencing_epoch: int,
    previous: Any,
) -> Any:
    """Claim one Host-neutral Session boundary before history-sensitive routing.

    The durable repository binds Session writes to the active Answer run. An
    empty transaction establishes that lease without choosing Fast or Research,
    then the refresh supplies the exact authoritative history boundary that no
    earlier run can advance while this run routes and accepts its operation.
    """
    snapshot = previous
    while True:
        lanes = tuple(snapshot.tree.lanes)
        if lanes:
            lane = next(
                (item for item in lanes if item.lane_id == snapshot.selected_lane_id),
                lanes[0],
            )
            lane_state = lane.state
            if not isinstance(lane_state.value, LaneState):
                raise TypeError("Lane State register has the wrong value type")
            transaction = SessionTransaction.from_parts(
                register_writes=[SetRegister(lane_state.value)],
                expectations=[RegisterExpectation(lane_state.ref, lane_state.sequence)],
            )
        else:
            lane_id = LaneId.main()
            head = LaneHead(lane_id, None)
            state = LaneState(lane_id)
            transaction = SessionTransaction.from_parts(
                register_writes=[SetRegister(head), SetRegister(state)],
                expectations=[
                    RegisterExpectation(head.ref, None),
                    RegisterExpectation(state.ref, None),
                ],
            )
        outcome = await repository.transact(
            session_id=session_id,
            fencing_epoch=fencing_epoch,
            transaction=transaction,
        )
        if isinstance(outcome, TransactionLeaseLost):
            raise SessionLeaseLostError(session_id.value)
        refreshed = await repository.refresh(session_id, previous=snapshot)
        validate_snapshot_refresh(
            session_id,
            previous=snapshot,
            snapshot=refreshed,
        )
        if isinstance(outcome, RegisterConflict):
            snapshot = refreshed
            continue
        return refreshed


def _require_fast_turn_reservation(
    snapshot: Any,
    *,
    lane_id: LaneId,
    reservation_id: str,
    accepted_user_entry_id: EntryId,
) -> None:
    record = next(
        (
            item
            for item in snapshot.registers
            if item.ref == RegisterRef("host_turn_reservation", lane_id.value)
        ),
        None,
    )
    if record is None or not isinstance(record.value, HostTurnReservation):
        raise OperationConflictError("Fast Host turn reservation is not active")
    reservation = record.value
    if (
        reservation.reservation_id != reservation_id
        or reservation.user_entry_id != accepted_user_entry_id
    ):
        raise OperationConflictError("Fast Host turn reservation identity changed")


def _project_fast_history_before_current_user(
    snapshot: Any,
    *,
    lane_id: LaneId,
    projection: ContextProjection | None,
    accepted_user_entry_id: EntryId,
) -> PriorTurns:
    """Fold a prepared projection while leaving the separately serialized query out."""
    selected = replace(snapshot, selected_lane_id=lane_id)
    ancestry = selected.graph.ancestry()
    semantic_entries = [entry for entry in ancestry if not isinstance(entry, CompactionEntry)]
    latest = semantic_entries[-1] if semantic_entries else None
    if not isinstance(latest, UserMessageEntry) or latest.entry_id != accepted_user_entry_id:
        raise ValueError("Fast compaction lost the current accepted User Entry")
    messages = project_session_messages(
        ancestry,
        projection,
        included_incomplete_host_user_entry_id=accepted_user_entry_id,
    )
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("Fast compaction projection did not retain the current User query")
    return PriorTurns(messages[:-1])


async def _close_execution_resources(
    stream: AsyncIterator[str] | None,
    registry: ResourceRegistry | None,
) -> None:
    cancellation: asyncio.CancelledError | None = None
    try:
        await aclose_answer_stream(stream)
    except asyncio.CancelledError as exc:
        cancellation = defer_cancellation(cancellation, exc)
    except Exception:
        logger.warning("Failed to close Answer stream", exc_info=True)
    if registry is not None:
        try:
            await registry.aclose()
        except asyncio.CancelledError as exc:
            cancellation = defer_cancellation(cancellation, exc)
        except Exception:
            logger.warning("Failed to close Answer resource registry", exc_info=True)
    if cancellation is not None:
        raise cancellation


def _require_resolved_mode(value: str | None) -> ResolvedMode:
    if value == "fast" or value == "research":
        return value
    raise RunExecutionError("routing_failed", "Answer mode routing failed.")


def _verified_current_image_data_uri(data: bytes, *, max_pixels: int) -> tuple[str, str]:
    from dlightrag.engine.ai.media import image_bytes_to_data_uri, verify_web_image_bytes

    mime = verify_web_image_bytes(data, max_pixels=max_pixels)
    return mime, image_bytes_to_data_uri(data, fallback_mime=mime)


def _context_count(contexts: RetrievalContexts, key: str) -> int:
    items = contexts.get(key, [])
    return len(items) if isinstance(items, list) else 0


def _publication_attachments(
    records: Sequence[ArtifactAttachmentUpdate],
) -> tuple[ArtifactAttachment, ...]:
    return tuple(
        ArtifactAttachment(
            relative_path=record.relative_path,
            label=record.label,
            content_digest=record.content_digest,
            size_bytes=record.size_bytes,
            presentation=record.presentation,  # type: ignore[arg-type]
        )
        for record in records
    )


def _publication_plan(
    root: Path | None,
    *,
    answer: str,
    attachments: Sequence[ArtifactAttachment],
    limits: PublicationLimits,
) -> PublicationPlan:
    if not isinstance(root, Path):
        return PublicationPlan(answer=answer)
    return validate_publication(
        root,
        answer=answer,
        attachments=attachments,
        limits=limits,
    )


def _stage_publications(
    *,
    plan: PublicationPlan,
    answer: str,
    contexts: RetrievalContexts,
    require_answer: bool = False,
) -> tuple[list[PendingPublication], list[dict[str, Any]], dict[str, list[Any]]]:
    if require_answer and is_empty_answer(answer=answer, has_artifacts=bool(plan.artifacts)):
        raise RunExecutionError("empty_answer", "The run produced no answer.")
    publications: list[PendingPublication] = []
    artifact_sources: dict[str, list[Any]] = {}
    descriptors = [dict(item) for item in plan.descriptors]
    for item in plan.artifacts:
        payload = item.content
        if item.media_type == "text/markdown":
            cleaned = finalize_answer(payload.decode("utf-8"), contexts)
            payload = cleaned.answer.encode("utf-8")
            artifact_sources[item.resource_id] = list(cleaned.sources)
            for descriptor in descriptors:
                if descriptor.get("resource_id") == item.resource_id:
                    descriptor["byte_size"] = len(payload)
                    descriptor["digest"] = artifact_digest(payload)
                    break
        publications.append(
            PendingPublication(
                resource_id=item.resource_id,
                reference_kind="published_artifact",
                filename=item.filename,
                mime_type=item.media_type,
                content=payload,
            )
        )
    return publications, descriptors, artifact_sources


def answer_trace_output(
    answer: str | None,
    sources: Sequence[Any] | None,
    contexts: RetrievalContexts,
    *,
    capture_sensitive_data: bool = False,
) -> dict[str, Any]:
    """Shape what a pipeline observation reports for one Answer result."""
    output: dict[str, Any] = {
        "answer_len": len(answer or ""),
        "source_count": len(sources or []),
        "context_chunk_count": _context_count(contexts, "chunks"),
    }
    if capture_sensitive_data:
        output["answer"] = answer or ""
    return output


__all__ = [
    "AnswerExecutionStore",
    "AnswerExecutor",
    "AnswerExecutorSettings",
    "AnswerResourceResolver",
    "AnswerResourceSettings",
    "OrchestratorRun",
    "ResolvedAnswerResources",
    "answer_trace_output",
]
