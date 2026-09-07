# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable answer runs over already-authorized canonical workspaces."""

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, aclosing
from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast
from uuid import UUID, uuid7

from dlightrag.application.answer_runs.capabilities import AnswerCapabilities, RequestModelContext
from dlightrag.application.answer_runs.capability import AnswerImageCapability
from dlightrag.application.answer_runs.envelope import accepted_input_envelope
from dlightrag.application.answer_runs.errors import (
    AnswerInputOverflowError,
    InvalidToolConfigurationError,
    UnsupportedAnswerModeError,
)
from dlightrag.application.answer_runs.execution import (
    AnswerRunInput,
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    PinnedModelProfile,
    build_current_answer_resources,
    in_memory_attachment_loader,
)
from dlightrag.application.answer_runs.mode import (
    AnswerMode,
    ModeCapability,
    ModeResource,
    ResolvedMode,
    canonical_answer_mode,
    require_supported_mode,
    resource_role,
    valid_modes,
)
from dlightrag.application.answer_runs.results import AnswerResult, restore_answer_result
from dlightrag.application.answer_runs.routing import RoutingAcceptance
from dlightrag.application.runs import (
    IdempotencyKeyConflict,
    RunCancelledError,
    RunCapacityExceededError,
    RunCreation,
    RunEvent,
    RunFailedError,
    RunRuntimeUnavailableError,
)
from dlightrag.engine.agent.session.fold import PriorTurns
from dlightrag.engine.agent.session.ids import LaneId, SessionId
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.tools import AgentTool
from dlightrag.engine.agent.tools.registry import DuplicateToolError, ToolRegistry
from dlightrag.engine.ai.capacity import (
    CONTEXT_POLICY,
    CONTEXT_POLICY_REVISION,
    ModelProfile,
)
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.settings import MODEL_ROLE_NAMES, ModelRole
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.execution import (
    ResolvedAnswerResources,
    research_history_input_measure,
)
from dlightrag.engine.answer.history import (
    HistoryProjectionOverflowError,
    HistoryProjectionTarget,
    project_history,
)
from dlightrag.engine.answer.images import AnswerImagePolicy
from dlightrag.engine.answer.memory import memory_owner_allowed, standing_memory_for_acceptance
from dlightrag.engine.answer.resources.images import QueryImageDescriber, prepare_query_images
from dlightrag.engine.answer.resources.models import ResourceInput, TextWindowBudget
from dlightrag.engine.answer.synthesizer import AnswerSynthesizer
from dlightrag.engine.answer.tools import compose_research_tools
from dlightrag.engine.rag.corpus.sources.source_contract import safe_source_filename
from dlightrag.engine.rag.retrieval import MetadataFilter, RetrievalOptions, RetrievalResult
from dlightrag.engine.rag.retrieval.planner import RetrievalPlanner
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id
from dlightrag.engine.runtime import (
    ArtifactReferenceKind,
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
    RunAccessScope,
    RunArtifactReference,
    RunKind,
    RunRecord,
    artifact_digest,
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

from .child_roster import (
    ChildRosterCursor,
    ChildRosterCursorCodec,
    ChildRosterPage,
    ChildRosterPageRequest,
    ChildRosterRowPage,
)

#: Accepted input uploads, in the precedence one ordinal resolves against.
_INPUT_REFERENCE_KINDS: tuple[ArtifactReferenceKind, ...] = (
    "current_attachment",
    "history_attachment",
)
_AGENT_CONTROL_CONTENT_LIMIT = 20_000


@dataclass(frozen=True, slots=True)
class AnswerHistoryResource:
    """One accepted upload carried from an owned prior run into this request."""

    run_id: str
    source_ordinal: int
    digest: str
    filename: str
    mime_type: str
    byte_size: int
    reference_kind: ArtifactReferenceKind = "current_attachment"


@dataclass(frozen=True, slots=True)
class AnswerRequest:
    """One authorized answer request over concrete canonical workspaces.

    Authorization happens before this contract exists: the workspace set is the
    already-expanded canonical result, never a policy wildcard, a token claim,
    or a user-visible display name.
    """

    query: str
    workspaces: tuple[str, ...]
    history: tuple[Mapping[str, Any], ...] = ()
    episodic_summary: str = ""
    retrieval: RetrievalOptions = RetrievalOptions()
    filters: MetadataFilter | None = None
    semantic_highlights: bool = False
    resources: tuple[ResourceInput, ...] = ()
    history_resources: tuple[AnswerHistoryResource, ...] = ()
    mode: str | None = None
    parent_run_id: str | None = None
    continuation_kind: str | None = None
    agent_session_id: str = ""
    agent_lane_id: str = "main"
    source_lane_id: str | None = None
    requested_skill: str | None = None


@dataclass(frozen=True, slots=True)
class AnswerInputArtifact:
    """One accepted run input upload, read back with its stored bytes."""

    reference_kind: ArtifactReferenceKind
    ordinal: int
    filename: str
    mime_type: str
    digest: str
    content: bytes


@dataclass(frozen=True, slots=True)
class AgentControlReceipt:
    """One ordered control accepted for a live Research session."""

    run_id: str
    control_sequence: int
    kind: str


@dataclass(frozen=True, slots=True)
class AgentTranscriptTail:
    """Bounded application projection shared by every transport."""

    run_id: str
    status: str
    messages: tuple[Mapping[str, Any], ...]


class HistoryResolver(Protocol):
    """In-process durable history projection invoked after exact targets exist."""

    def __call__(self, targets: Sequence[HistoryProjectionTarget]) -> Awaitable[PriorTurns]: ...


class AnswerRuntimeUnavailableError(RunRuntimeUnavailableError):
    """Answer-specific compatibility name for common runtime unavailability."""


class AnswerRunAcceptor[T](Protocol):
    """Persist or replay one prepared run, optionally with an atomic domain link."""

    async def create_run(
        self,
        *,
        envelope: PreparedRunEnvelope,
        run_id: str,
        resources: Sequence[Mapping[str, Any]] = (),
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
    ) -> T | None: ...

    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        run_kind: RunKind,
    ) -> T | None: ...


class _AnswerRunRepository(AnswerRunAcceptor[RuntimeRunCreation], Protocol):
    """The owner-scoped durable operations this service performs."""

    async def get_run(self, *, owner_id: str, run_id: str) -> RunRecord | None: ...

    async def enqueue_agent_control(
        self, *, owner_id: str, run_id: str, kind: str, content: str
    ) -> Mapping[str, Any] | None: ...

    async def list_child_sessions(
        self, *, owner_id: str, run_id: str
    ) -> tuple[Mapping[str, Any], ...]: ...

    async def list_child_sessions_page(
        self,
        *,
        owner_id: str,
        run_id: str,
        page: ChildRosterPageRequest,
    ) -> ChildRosterRowPage: ...

    async def load_agent_transcript(
        self, *, owner_id: str, run_id: str, session_id: str, limit: int
    ) -> tuple[Mapping[str, Any], ...]: ...

    async def list_run_artifacts(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunArtifactReference, ...]: ...


class _RunBlobReader(Protocol):
    """The opaque bytes seam; Answer metadata never owns blob persistence."""

    def stream(
        self,
        *,
        owner_id: str,
        digest: str,
        offset: int = 0,
        length: int | None = None,
    ) -> AsyncIterator[bytes]: ...

    async def size(self, *, owner_id: str, digest: str) -> int | None: ...


class _RunScheduler(Protocol):
    """The started coordinator accepted runs execute and stream through."""

    @property
    def is_started(self) -> bool: ...

    def admission(self) -> AbstractAsyncContextManager[bool]: ...

    def wake(self) -> None: ...

    def cancel_local(self, owner_id: str, run_id: str) -> None: ...

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncGenerator[RuntimeRunEvent]: ...


class _RetrievalPlanning(Protocol):
    """The retrieval facts acceptance needs before a run is scheduled."""

    def planner_for(self, model_profile: ModelProfile | None = None) -> RetrievalPlanner: ...

    def warm(self, workspaces: Sequence[str]) -> None: ...

    async def schema_for(self, workspaces: Sequence[str]) -> dict[str, Any]: ...


class _AnswerCapabilityReader(Protocol):
    """The public immutable capability snapshot callers may read."""

    async def read(self) -> AnswerCapabilities: ...


class _AnswerCapabilityPlanner(Protocol):
    """Role profiles and image policy one acceptance pins itself against."""

    async def refresh_vlm(self) -> AnswerCapabilities: ...

    def current_profiles(self) -> dict[ModelRole, ModelProfile]: ...

    def request_model_context(
        self, pinned: Mapping[ModelRole, ModelProfile] | None, /
    ) -> RequestModelContext: ...

    def answer_image_policy(self, profile: ModelProfile, /) -> AnswerImagePolicy: ...

    async def confirmed_live_answer_context(
        self, models: RequestModelContext, /
    ) -> tuple[RequestModelContext, AnswerImageCapability | None]: ...


class _QueryImageRuntime(Protocol):
    """The model runtime acceptance describes current-turn images with."""

    def query_image_describer(self) -> QueryImageDescriber: ...


class _AnswerResourcePreparer(Protocol):
    """Resource materialization and resolution shared with run execution."""

    async def pin_current_image_links(
        self, request: AnswerRunRequest, attachment_bytes: Sequence[bytes], /
    ) -> tuple[AnswerRunRequest, list[bytes]]: ...

    async def resolve(
        self,
        resources: list[ResourceInput] | None,
        /,
        *,
        models: RequestModelContext,
        text_window_budget: TextWindowBudget,
        confirm_image_context: Callable[
            [RequestModelContext],
            Awaitable[tuple[RequestModelContext, AnswerImageCapability | None]],
        ],
        resolved_mode: ResolvedMode,
    ) -> ResolvedAnswerResources: ...


@dataclass(frozen=True, slots=True)
class _AcceptanceProjection:
    history: tuple[Mapping[str, Any], ...]
    episodic_summary: str
    image_descriptions: tuple[str, ...]
    pinned_models: tuple[PinnedModelProfile, ...]
    agent_run_plan: AgentRunPlan | None
    valid_modes: frozenset[ResolvedMode]


def _attachment_bytes(resources: Sequence[ResourceInput]) -> list[bytes]:
    """Return the inline bytes an accepted run must persist with its input."""
    return [resource.content for resource in resources if resource.content is not None]


def _prepared_input_payload(
    run_input: Any, *, requested_mode: str, auth_mode: str = "none"
) -> dict[str, Any]:
    """Encode one accepted run with its canonical Session/Lane mapping."""
    payload = dict(run_input.as_request())
    payload["auth_mode"] = auth_mode
    payload["mode"] = requested_mode
    return payload


def _accepted_resource_payloads(
    run_input: Any, *, attachment_bytes: Sequence[bytes]
) -> list[dict[str, Any]]:
    import hashlib

    payloads: list[dict[str, Any]] = []
    for ordinal, attachment in enumerate(run_input.attachments):
        content = attachment_bytes[ordinal] if ordinal < len(attachment_bytes) else b""
        payloads.append(
            {
                "resource_id": attachment.resource_id,
                "safe_name": attachment.filename,
                "media_type": attachment.mime_type or "application/octet-stream",
                "capabilities": {},
                "ordinal": ordinal,
                "blob_digest": hashlib.sha256(content).hexdigest(),
            }
        )
    return payloads


def _normalized_request(request: AnswerRequest) -> AnswerRunRequest:
    """Project one public request into durable acceptance input, without I/O."""
    if not request.workspaces:
        raise ValueError("at least one canonical workspace is required")
    workspaces = tuple(
        require_canonical_workspace_id(workspace) for workspace in request.workspaces
    )
    links: list[LinkReference] = []
    attachments: list[AttachmentReference] = []
    for resource in request.resources:
        if resource.url is not None:
            links.append(
                LinkReference(
                    url=resource.url,
                    filename=resource.filename,
                    ordinal=len(links),
                    mime_type=resource.declared_mime,
                )
            )
            continue
        if resource.content is None:
            raise ValueError("durable answer resources need inline bytes or an HTTPS link")
        attachments.append(
            AttachmentReference(
                digest=artifact_digest(resource.content),
                filename=safe_source_filename(resource.filename),
                mime_type=resource.declared_mime or "application/octet-stream",
                ordinal=len(attachments),
                byte_size=len(resource.content),
            )
        )
    return AnswerRunRequest(
        query=request.query,
        workspaces=workspaces,
        history=tuple(dict(message) for message in request.history),
        episodic_summary=request.episodic_summary,
        retrieval=request.retrieval,
        filters=(
            request.filters.model_dump(exclude_none=True, mode="json") if request.filters else None
        ),
        semantic_highlights=request.semantic_highlights,
        links=tuple(links),
        attachments=tuple(attachments),
        mode=request.mode or "auto",
        parent_run_id=request.parent_run_id,
        continuation_kind=request.continuation_kind,
        agent_session_id=request.agent_session_id,
        agent_lane_id=request.agent_lane_id,
        source_lane_id=request.source_lane_id,
        requested_skill=request.requested_skill,
        history_attachments=tuple(
            AttachmentReference(
                digest=resource.digest,
                filename=safe_source_filename(resource.filename),
                mime_type=resource.mime_type,
                ordinal=ordinal,
                byte_size=resource.byte_size,
            )
            for ordinal, resource in enumerate(request.history_resources)
        ),
    )


def _artifact_references(request: AnswerRunInput) -> list[PendingArtifactReference]:
    """Describe every accepted input upload's durable replay slot."""
    references = [
        PendingArtifactReference(
            resource_id=attachment.resource_id,
            reference_kind="current_attachment",
            ordinal=attachment.ordinal,
            digest=attachment.digest,
            filename=attachment.filename,
            mime_type=attachment.mime_type,
        )
        for attachment in request.attachments
    ]
    references.extend(
        PendingArtifactReference(
            resource_id=attachment.history_resource_id,
            reference_kind="history_attachment",
            ordinal=attachment.ordinal,
            digest=attachment.digest,
            filename=attachment.filename,
            mime_type=attachment.mime_type,
        )
        for attachment in request.history_attachments
    )
    return references


class AnswerService:
    """Accept, follow, and read back this deployment's durable answer runs."""

    def __init__(
        self,
        *,
        store: _AnswerRunRepository,
        blob_store: _RunBlobReader,
        coordinator: _RunScheduler,
        retrieval: _RetrievalPlanning,
        capabilities: _AnswerCapabilityPlanner,
        capability_view: _AnswerCapabilityReader,
        models: _QueryImageRuntime,
        resources: _AnswerResourcePreparer,
        model_fingerprint_for_role: Callable[[ModelRole], ModelFingerprint],
        child_roster_cursor_secret: bytes,
        research_tool_supplements: Callable[[], Sequence[AgentTool]] | None = None,
        memory_capability: Callable[..., Awaitable[tuple[bool, int]]] | None = None,
        run_retention_seconds: int = 365 * 24 * 3600,
    ) -> None:
        self._store = store
        self._blob_store = blob_store
        self._coordinator = coordinator
        self._retrieval = retrieval
        self._capabilities = capabilities
        self._capability_view = capability_view
        self._models = models
        self._resources = resources
        self._model_fingerprint_for_role = model_fingerprint_for_role
        self._research_tool_supplements = research_tool_supplements or (lambda: ())
        self._memory_capability = memory_capability
        self._run_retention_seconds = int(run_retention_seconds)
        self._child_roster_codec = ChildRosterCursorCodec(child_roster_cursor_secret)

    async def create(
        self,
        *,
        request: AnswerRequest,
        owner_id: str,
        idempotency_key: str | None = None,
        auth_mode: str = "none",
    ) -> RunCreation:
        """Accept one durable run and return its descriptor without waiting.

        A keyed replay is resolved before any link is materialized or any input
        is prepared, so a retried submission never repeats acceptance work. The
        accepted run outlives this call and is read back through :meth:`get`,
        :meth:`subscribe`, and :meth:`cancel`.
        """
        try:
            creation = await self._accept(
                request=request,
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                idempotency_fingerprint=None,
                acceptor=self._store,
                auth_mode=auth_mode,
            )
        except RuntimeIdempotencyKeyConflict as exc:
            raise IdempotencyKeyConflict(str(exc)) from exc
        except RuntimeRunCapacityExceededError as exc:
            raise RunCapacityExceededError(str(exc)) from exc
        if creation is None:
            raise RuntimeError("Answer run acceptance returned no descriptor")
        return RunCreation.from_runtime(creation)

    async def accept[T](
        self,
        *,
        request: AnswerRequest,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        acceptor: AnswerRunAcceptor[T],
        auth_mode: str = "none",
        history_resolver: HistoryResolver | None = None,
    ) -> T | None:
        """Accept through a typed atomic linker while preserving one run pipeline."""
        try:
            return await self._accept(
                request=request,
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                idempotency_fingerprint=idempotency_fingerprint,
                acceptor=acceptor,
                auth_mode=auth_mode,
                history_resolver=history_resolver,
            )
        except RuntimeIdempotencyKeyConflict as exc:
            raise IdempotencyKeyConflict(str(exc)) from exc
        except RuntimeRunCapacityExceededError as exc:
            raise RunCapacityExceededError(str(exc)) from exc

    async def _accept[T](
        self,
        *,
        request: AnswerRequest,
        owner_id: str,
        idempotency_key: str | None,
        idempotency_fingerprint: str | None,
        acceptor: AnswerRunAcceptor[T],
        auth_mode: str = "none",
        history_resolver: HistoryResolver | None = None,
    ) -> T | None:
        run_request = _normalized_request(request)
        fingerprint = idempotency_fingerprint or run_request_fingerprint(run_request.as_request())
        # Canonical mode syntax is capability-independent and may be checked
        # before replay. Live capability validation must wait until after the
        # answer/VLM refreshes below because an unknown probe can recover.
        requested_mode = canonical_answer_mode(run_request.mode)
        if idempotency_key is not None:
            replay = await acceptor.replay_run(
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                idempotency_fingerprint=fingerprint,
                run_kind="answer",
            )
            if replay is not None:
                return replay
        if not self._coordinator.is_started:
            raise AnswerRuntimeUnavailableError("Answer runtime is unavailable")
        run_request, attachment_bytes = await self._resources.pin_current_image_links(
            run_request,
            _attachment_bytes(request.resources),
        )
        if run_request.links or run_request.attachments or run_request.history_attachments:
            await self._capabilities.refresh_vlm()
        # Pinning may narrow the live query profile, and the VLM refresh may
        # remove inspect. This is the authoritative Valid Mode Set persisted
        # with the run; the earlier check only avoids needless acceptance I/O.
        requested_mode, allowed_modes = self._reject_unsupported_mode(run_request)
        acceptance_resources = await build_current_answer_resources(
            links=run_request.links,
            attachments=run_request.attachments,
            attachment_loaders=[
                in_memory_attachment_loader(content) for content in attachment_bytes
            ],
        )
        acceptance_resources.extend(
            self._history_resource_input(owner_id, resource)
            for resource in request.history_resources
        )
        memory_enabled = (
            requested_mode != "fast"
            and "research" in allowed_modes
            and memory_owner_allowed(auth_mode)
        )
        memory_epoch = 0
        if memory_enabled and self._memory_capability is not None:
            memory_enabled, memory_epoch = await self._memory_capability(owner_id=owner_id)
        run_input, allowed_modes = await self._prepare_input(
            run_request,
            resources=acceptance_resources or None,
            idempotency_fingerprint=fingerprint,
            requested_mode=requested_mode,
            allowed_modes=allowed_modes,
            auth_mode=auth_mode,
            memory_enabled=memory_enabled,
            history_resolver=history_resolver,
        )
        prepared_input = _prepared_input_payload(
            run_input, requested_mode=requested_mode, auth_mode=auth_mode
        )
        prepared_input["profile_memory_enabled"] = memory_enabled
        prepared_input["profile_memory_epoch"] = memory_epoch
        require_prepared_input_bounds(prepared_input)
        resources_payload = _accepted_resource_payloads(
            run_input, attachment_bytes=attachment_bytes
        )
        async with self._coordinator.admission() as runtime_available:
            if not runtime_available:
                raise AnswerRuntimeUnavailableError("Answer runtime is unavailable")
            run_id = str(uuid7())
            accepted = await acceptor.create_run(
                envelope=PreparedRunEnvelope(
                    run_kind="answer",
                    lane="query",
                    submitted_by=owner_id,
                    access_scope=RunAccessScope(kind="owner", scope_id=owner_id),
                    submission_key=idempotency_key or run_id,
                    request_fingerprint=fingerprint,
                    payload=prepared_input,
                    accepted_input=accepted_input_envelope(prepared_input),
                    retention_seconds=self._run_retention_seconds,
                ),
                run_id=run_id,
                resources=resources_payload,
                artifacts=[PendingArtifact(content=content) for content in attachment_bytes],
                references=_artifact_references(run_input),
                routing=RoutingAcceptance(
                    requested_mode=requested_mode,
                    valid_modes=tuple(sorted(allowed_modes)),
                    context_policy_revision=CONTEXT_POLICY_REVISION,
                    model_fingerprints={
                        item.role: {
                            "provider": item.fingerprint.provider,
                            "model": item.fingerprint.model,
                            "endpoint_fingerprint": item.fingerprint.endpoint_fingerprint,
                        }
                        for item in run_input.pinned_models
                    },
                    agent_session_id=run_input.agent_session_id,
                    agent_lane_id=run_input.agent_lane_id,
                    source_lane_id=run_input.source_lane_id,
                ),
            )
            if accepted is not None:
                self._coordinator.wake()
        return accepted

    def _reject_unsupported_mode(
        self, request: AnswerRunRequest
    ) -> tuple[AnswerMode, frozenset[ResolvedMode]]:
        """Fail closed before a run row exists when the requested mode cannot resolve."""
        profiles = self._capabilities.current_profiles()
        query = profiles["query"]
        vlm = profiles["vlm"]
        resources: list[ModeResource] = []
        for attachment in (*request.attachments, *request.history_attachments):
            resources.append(
                ModeResource(
                    role=resource_role(filename=attachment.filename, mime_type=attachment.mime_type)
                )
            )
        for link in request.links:
            role = resource_role(filename=link.filename or link.url, mime_type=link.mime_type)
            if role == "other":
                role = "document"
            resources.append(ModeResource(role=role))
        allowed = valid_modes(
            resources=tuple(resources),
            capability=ModeCapability(
                query_supports_images=query.supports_images,
                inspect_available=vlm.supports_images,
            ),
        )
        requested = require_supported_mode(requested=request.mode, valid=allowed)
        return requested, allowed

    def _history_resource_input(
        self,
        owner_id: str,
        resource: AnswerHistoryResource,
    ) -> ResourceInput:
        async def load() -> bytes:
            artifact = await self.read_input_artifact(
                owner_id=owner_id,
                run_id=resource.run_id,
                ordinal=resource.source_ordinal,
                reference_kind=resource.reference_kind,
            )
            if artifact is None:
                raise AnswerRuntimeUnavailableError("Accepted answer input artifact is unavailable")
            return artifact.content

        return ResourceInput(
            filename=resource.filename,
            declared_mime=resource.mime_type,
            loader=load,
        )

    async def list_artifacts(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunArtifactReference, ...] | None:
        """List artifacts for one owned Answer; every other id is unknown."""
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            return None
        return await self._store.list_run_artifacts(owner_id=owner_id, run_id=run_id)

    async def read_artifact(
        self,
        *,
        owner_id: str,
        run_id: str,
        resource_id: str,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes | None:
        """Read a bounded Answer artifact; every other id returns ``None``."""
        stream = await self.open_artifact(
            owner_id=owner_id,
            run_id=run_id,
            resource_id=resource_id,
            offset=offset,
            length=length,
        )
        if stream is None:
            return None
        pieces = [piece async for piece in stream]
        return b"".join(pieces)

    async def open_artifact(
        self,
        *,
        owner_id: str,
        run_id: str,
        resource_id: str,
        offset: int = 0,
        length: int | None = None,
    ) -> AsyncIterator[bytes] | None:
        """Open an Answer artifact; every other id returns ``None``.

        Callers that serve large published artifacts stream these chunks; no
        complete-blob materialization happens on this path.
        """
        refs = await self.list_artifacts(owner_id=owner_id, run_id=run_id)
        if refs is None:
            return None
        match = next((item for item in refs if item.resource_id == resource_id), None)
        if match is None:
            return None
        return self._blob_store.stream(
            owner_id=owner_id,
            digest=match.digest,
            offset=max(0, offset),
            length=length,
        )

    async def artifact_size(self, *, owner_id: str, run_id: str, resource_id: str) -> int | None:
        """Return an Answer artifact size; every other id returns ``None``."""
        refs = await self.list_artifacts(owner_id=owner_id, run_id=run_id)
        if refs is None:
            return None
        match = next((item for item in refs if item.resource_id == resource_id), None)
        if match is None:
            return None
        return await self._blob_store.size(owner_id=owner_id, digest=match.digest)

    async def _get_answer_run(self, *, owner_id: str, run_id: str) -> RunRecord | None:
        """Return one owned Answer; unknown, foreign, and wrong-kind ids are identical."""
        record = await self._store.get_run(owner_id=owner_id, run_id=run_id)
        return record if record is not None and record.run_kind == "answer" else None

    async def steer(
        self, *, owner_id: str, run_id: str, instruction: str
    ) -> AgentControlReceipt | None:
        """Queue one ordered steering instruction for a live Research session."""
        text = instruction.strip()
        if not text:
            raise ValueError("steer instruction cannot be empty")
        if len(text) > _AGENT_CONTROL_CONTENT_LIMIT:
            raise ValueError("steer instruction exceeds 20000 characters")
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            return None
        row = await self._store.enqueue_agent_control(
            owner_id=owner_id,
            run_id=run_id,
            kind="steer",
            content=text,
        )
        if row is None:
            return None
        return AgentControlReceipt(
            run_id=run_id,
            control_sequence=int(row["control_sequence"]),
            kind=str(row["kind"]),
        )

    async def children(
        self,
        *,
        owner_id: str,
        run_id: str,
        page: ChildRosterPageRequest | None = None,
    ) -> ChildRosterPage | None:
        """Return one bounded newest-first child-roster page, or None if unknown."""
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            return None
        requested = page or ChildRosterPageRequest()
        if requested.cursor is not None and str(requested.cursor.run_id) != run_id:
            raise ValueError("child-roster cursor belongs to another run")
        result = await self._store.list_child_sessions_page(
            owner_id=owner_id,
            run_id=run_id,
            page=requested,
        )
        next_cursor = None
        if result.has_more:
            if not result.children:
                raise RuntimeError("child-roster store reported more rows after an empty page")
            last = result.children[-1]
            next_cursor = ChildRosterCursor(
                run_id=UUID(run_id),
                created_at=last["created_at"],
                child_session_id=UUID(str(last["child_session_id"])),
            )
        return ChildRosterPage(
            children=result.children,
            next_cursor=next_cursor,
            fetched_rows=result.fetched_rows,
        )

    @property
    def child_roster_cursor_codec(self) -> ChildRosterCursorCodec:
        """Return the codec shared with the answer-runs HTTP adapters."""
        return self._child_roster_codec

    async def transcript_tail(
        self, *, owner_id: str, run_id: str, limit: int = 20
    ) -> AgentTranscriptTail | None:
        """Return a bounded transport-neutral transcript projection."""
        record = await self._get_answer_run(owner_id=owner_id, run_id=run_id)
        if record is None:
            return None
        request = record.request_input()
        cap = max(1, min(int(limit), 100))
        session_id = str(request.get("agent_session_id") or "")
        load_transcript = getattr(self._store, "load_agent_transcript", None)
        if session_id and callable(load_transcript):
            loader = cast(
                Callable[..., Awaitable[Sequence[Mapping[str, Any]]]],
                load_transcript,
            )
            canonical = await loader(
                owner_id=owner_id,
                run_id=run_id,
                session_id=session_id,
                limit=cap,
            )
            if canonical:
                return AgentTranscriptTail(
                    run_id=run_id,
                    status=record.status,
                    messages=tuple(dict(message) for message in canonical),
                )
        # Fast has no Agent Session. Project its accepted invocation and final
        # result through the same transport-neutral message shape.
        messages = [
            dict(message)
            for message in request.get("history") or ()
            if isinstance(message, Mapping)
        ]
        query = str(request.get("query") or "")
        if query:
            messages.append({"role": "user", "content": query})
        result = record.result or {}
        answer = str(result.get("answer") or "")
        if answer:
            messages.append({"role": "assistant", "content": answer})
        return AgentTranscriptTail(
            run_id=run_id,
            status=record.status,
            messages=tuple(messages[-cap:]),
        )

    async def continuation_workspaces(
        self, *, owner_id: str, run_id: str
    ) -> tuple[str, ...] | None:
        """Return a terminal Answer's workspace set for current authorization."""
        record = await self._get_answer_run(owner_id=owner_id, run_id=run_id)
        if record is None or not record.terminal:
            return None
        return tuple(str(item) for item in record.request_input().get("workspaces") or ())

    async def follow_up(
        self,
        *,
        owner_id: str,
        run_id: str,
        query: str,
        idempotency_key: str | None = None,
        auth_mode: str = "none",
        authorized_workspaces: Sequence[str] | None,
    ) -> RunCreation | None:
        """Start a continuation from one terminal result through normal acceptance."""
        request = await self.continuation_request(
            owner_id=owner_id,
            run_id=run_id,
            query=query,
            include_answer=True,
            authorized_workspaces=authorized_workspaces,
        )
        if request is None:
            return None
        return await self.create(
            request=request,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
            auth_mode=auth_mode,
        )

    async def fork(
        self,
        *,
        owner_id: str,
        run_id: str,
        query: str,
        idempotency_key: str | None = None,
        auth_mode: str = "none",
        authorized_workspaces: Sequence[str] | None,
    ) -> RunCreation | None:
        """Start a sibling branch from the selected run's accepted context."""
        request = await self.continuation_request(
            owner_id=owner_id,
            run_id=run_id,
            query=query,
            include_answer=False,
            authorized_workspaces=authorized_workspaces,
        )
        if request is None:
            return None
        return await self.create(
            request=request,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
            auth_mode=auth_mode,
        )

    async def continuation_request(
        self,
        *,
        owner_id: str,
        run_id: str,
        query: str,
        include_answer: bool,
        authorized_workspaces: Sequence[str] | None,
    ) -> AnswerRequest | None:
        """Build the selected accepted context after transport authorization.

        The parent's question always joins the history; the parent's answer
        joins only for a follow-up. A fork therefore branches from the same
        starting point without carrying the answer it is meant to redo.
        """
        text = query.strip()
        if not text:
            raise ValueError("continuation query cannot be empty")
        if len(text) > _AGENT_CONTROL_CONTENT_LIMIT:
            raise ValueError("continuation query exceeds 20000 characters")
        record = await self._get_answer_run(owner_id=owner_id, run_id=run_id)
        if record is None or not record.terminal:
            return None
        if authorized_workspaces is None:
            raise ValueError("continuation requires a currently authorized workspace set")
        accepted = record.request_input()
        history: list[Mapping[str, Any]] = [
            dict(message)
            for message in accepted.get("history") or ()
            if isinstance(message, Mapping)
        ]
        parent_query = str(accepted.get("query") or "")
        if parent_query:
            history.append({"role": "user", "content": parent_query})
        if include_answer:
            parent_answer = str((record.result or {}).get("answer") or "")
            if parent_answer:
                history.append({"role": "assistant", "content": parent_answer})

        history_resources: list[AnswerHistoryResource] = []
        for reference_kind, items in (
            ("history_attachment", accepted.get("history_attachments") or ()),
            ("current_attachment", accepted.get("attachments") or ()),
        ):
            history_resources.extend(
                AnswerHistoryResource(
                    run_id=run_id,
                    source_ordinal=int(item.get("ordinal") or 0),
                    digest=str(item.get("digest") or ""),
                    filename=str(item.get("filename") or "attachment"),
                    mime_type=str(item.get("mime_type") or "application/octet-stream"),
                    byte_size=int(item.get("byte_size") or 0),
                    reference_kind=reference_kind,  # type: ignore[arg-type]
                )
                for item in items
                if isinstance(item, Mapping) and item.get("digest")
            )
        link_resources = tuple(
            ResourceInput(
                url=str(item.get("url") or ""),
                filename=(str(item["filename"]) if item.get("filename") else None),
                declared_mime=(str(item["mime_type"]) if item.get("mime_type") else None),
            )
            for item in accepted.get("links") or ()
            if isinstance(item, Mapping) and item.get("url")
        )
        filters = accepted.get("filters")
        agent_session_id = str(accepted.get("agent_session_id") or "") or SessionId.new().value
        parent_lane_id = str(accepted.get("agent_lane_id") or LaneId.main().value)
        continuation_kind = "follow_up" if include_answer else "fork"
        agent_lane_id = parent_lane_id if include_answer else LaneId.new().value
        return AnswerRequest(
            query=text,
            workspaces=tuple(str(item) for item in authorized_workspaces),
            history=tuple(history),
            episodic_summary=str(accepted.get("episodic_summary") or ""),
            retrieval=RetrievalOptions(
                top_k=(int(accepted["top_k"]) if accepted.get("top_k") is not None else None),
                chunk_top_k=(
                    int(accepted["chunk_top_k"])
                    if accepted.get("chunk_top_k") is not None
                    else None
                ),
                federated_rerank=bool(accepted.get("federated_rerank")),
            ),
            filters=(
                MetadataFilter.model_validate(filters) if isinstance(filters, Mapping) else None
            ),
            semantic_highlights=bool(accepted.get("semantic_highlights")),
            resources=link_resources,
            history_resources=tuple(history_resources),
            mode=str(accepted.get("mode") or "auto"),
            parent_run_id=run_id,
            continuation_kind=continuation_kind,
            agent_session_id=agent_session_id,
            agent_lane_id=agent_lane_id,
            source_lane_id=(parent_lane_id if not include_answer else None),
        )

    async def wait(self, *, owner_id: str, run_id: str) -> AnswerResult:
        """Follow one owned run to its terminal state and project its result.

        Cancelling this wait detaches this observer only; use the common Run
        service to stop the run itself.
        """
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            raise RunFailedError(
                "answer_run_missing",
                "Answer run disappeared before it finished.",
            )
        async with aclosing(
            self._coordinator.subscribe(owner_id=owner_id, run_id=run_id)
        ) as events:
            async for _event in events:
                pass
        final = await self._store.get_run(owner_id=owner_id, run_id=run_id)
        if final is None:
            raise RunFailedError(
                "answer_run_missing",
                "Answer run disappeared before it finished.",
            )
        if final.status == "succeeded":
            return restore_answer_result(final.result or {})
        if final.status == "cancelled":
            raise RunCancelledError(final.run_id)
        raise RunFailedError(
            final.error_kind or "answer_stream_failed",
            final.error_message or "Answer run failed.",
        )

    async def answer(
        self,
        request: AnswerRequest,
        *,
        owner_id: str,
        idempotency_key: str | None = None,
    ) -> AnswerResult:
        """Create one durable answer run and wait for its canonical result."""
        creation = await self.create(
            request=request,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
        )
        return await self.wait(owner_id=owner_id, run_id=creation.run.run_id)

    async def answer_stream(
        self,
        request: AnswerRequest,
        *,
        owner_id: str,
        idempotency_key: str | None = None,
    ) -> AsyncGenerator[RunEvent]:
        """Create one durable answer run and follow its events until it ends."""
        creation = await self.create(
            request=request,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
        )
        run = creation.run
        async with aclosing(
            self._coordinator.subscribe(owner_id=owner_id, run_id=run.run_id)
        ) as events:
            async for event in events:
                yield RunEvent.from_runtime(event)

    async def capabilities(self) -> AnswerCapabilities:
        """Return the public image-capability snapshot after its allowed re-probe."""
        return await self._capability_view.read()

    async def read_input_artifact(
        self,
        *,
        owner_id: str,
        run_id: str,
        ordinal: int,
        reference_kind: ArtifactReferenceKind | None = None,
    ) -> AnswerInputArtifact | None:
        """Read one owned run's accepted input upload by its ordinal.

        Resources a worker fetched mid-run are run state, never accepted input,
        so they are not readable here; a current-turn upload takes precedence
        over a history upload sharing its ordinal.
        """
        references = await self.list_artifacts(owner_id=owner_id, run_id=run_id)
        if references is None:
            return None
        reference = next(
            (
                item
                for kind in (
                    (reference_kind,) if reference_kind is not None else _INPUT_REFERENCE_KINDS
                )
                for item in references
                if item.reference_kind == kind and item.ordinal == ordinal
            ),
            None,
        )
        if reference is None:
            return None
        pieces = [
            piece
            async for piece in self._blob_store.stream(owner_id=owner_id, digest=reference.digest)
        ]
        if not pieces:
            return None
        content = b"".join(pieces)
        return AnswerInputArtifact(
            reference_kind=reference.reference_kind,
            ordinal=reference.ordinal,
            filename=reference.filename,
            mime_type=reference.mime_type,
            digest=reference.digest,
            content=content,
        )

    async def _prepare_input(
        self,
        request: AnswerRunRequest,
        *,
        resources: list[ResourceInput] | None,
        idempotency_fingerprint: str,
        requested_mode: AnswerMode,
        allowed_modes: frozenset[ResolvedMode],
        auth_mode: str = "none",
        memory_enabled: bool = True,
        history_resolver: HistoryResolver | None = None,
    ) -> tuple[AnswerRunInput, frozenset[ResolvedMode]]:
        """Resolve one normalized request and its capacity-narrowed mode set."""
        projection = await self._project_acceptance(
            request,
            resources=resources,
            requested_mode=requested_mode,
            allowed_modes=allowed_modes,
            auth_mode=auth_mode,
            memory_enabled=memory_enabled,
            history_resolver=history_resolver,
        )
        return AnswerRunInput(
            query=request.query,
            workspaces=request.workspaces,
            history=projection.history,
            episodic_summary=projection.episodic_summary,
            retrieval=request.retrieval,
            filters=request.filters,
            semantic_highlights=request.semantic_highlights,
            links=request.links,
            attachments=request.attachments,
            history_attachments=request.history_attachments,
            pinned_models=projection.pinned_models,
            context_policy_revision=CONTEXT_POLICY_REVISION,
            model_catalog_revision=current_model_catalog_revision(),
            idempotency_fingerprint=idempotency_fingerprint,
            agent_run_plan=projection.agent_run_plan,
            image_descriptions=projection.image_descriptions,
            parent_run_id=request.parent_run_id,
            continuation_kind=request.continuation_kind,
            agent_session_id=request.agent_session_id or SessionId.new().value,
            agent_lane_id=request.agent_lane_id,
            source_lane_id=request.source_lane_id,
        ), projection.valid_modes

    async def _project_acceptance(
        self,
        request: AnswerRunRequest,
        *,
        resources: list[ResourceInput] | None,
        requested_mode: AnswerMode,
        allowed_modes: frozenset[ResolvedMode],
        auth_mode: str = "none",
        memory_enabled: bool = True,
        history_resolver: HistoryResolver | None = None,
    ) -> _AcceptanceProjection:
        """Resolve the exact shared-history envelopes without building the run rig."""
        model_profiles = self._capabilities.current_profiles()
        models = self._capabilities.request_model_context(model_profiles)
        planner = self._retrieval.planner_for(models.extract)
        text_window_budget = TextWindowBudget(CONTEXT_POLICY.hard_input_limit(models.query))
        resolved = await self._resources.resolve(
            resources,
            models=models,
            text_window_budget=text_window_budget,
            confirm_image_context=self._capabilities.confirmed_live_answer_context,
            resolved_mode=("research" if "research" in allowed_modes else "fast"),
        )
        agent_run_plan: AgentRunPlan | None = None
        try:
            workspaces = list(request.workspaces)
            self._retrieval.warm(workspaces)
            models = resolved.models
            model_profiles["extract"] = models.extract
            model_profiles["query"] = models.query
            model_profiles["vlm"] = models.vlm
            image_descriptions = tuple(
                await prepare_query_images(
                    query_images=resolved.current_images,
                    describer=self._models.query_image_describer(),
                )
                if resolved.current_images
                else ()
            )
            schema = await self._retrieval.schema_for(workspaces)
            memory_text = standing_memory_for_acceptance(auth_mode) if memory_enabled else ""
            effective_modes = allowed_modes
            fast_targets: list[HistoryProjectionTarget] = []
            if "fast" in effective_modes:
                fast_targets.append(
                    HistoryProjectionTarget(
                        "fast_planner",
                        models.extract,
                        planner.history_input_measure(
                            request.query,
                            schema=schema,
                            current_image_descriptions=list(image_descriptions) or None,
                            preserve_query=None,
                        ),
                        proactive_compaction=True,
                        require_full_dynamic_reserve=True,
                    )
                )
                synthesizer = AnswerSynthesizer(
                    image_policy=self._capabilities.answer_image_policy(models.query),
                    model_profile=models.query,
                    context_policy=CONTEXT_POLICY,
                    model_func=None,
                )
                fast_generation_measure = (
                    synthesizer.history_input_measure(
                        request.query,
                        memory_text="",
                        episodic_summary=request.episodic_summary,
                        current_images=resolved.current_images,
                    )
                    if resolved.current_images
                    else synthesizer.history_input_measure(
                        request.query,
                        memory_text="",
                        episodic_summary=request.episodic_summary,
                    )
                )
                fast_targets.append(
                    HistoryProjectionTarget(
                        "fast_generation",
                        models.query,
                        fast_generation_measure,
                        proactive_compaction=True,
                        require_full_dynamic_reserve=True,
                    )
                )
                try:
                    project_history([], targets=fast_targets)
                except HistoryProjectionOverflowError as exc:
                    if requested_mode == "fast":
                        raise AnswerInputOverflowError(str(exc)) from exc
                    effective_modes = cast(
                        frozenset[ResolvedMode],
                        frozenset(mode for mode in effective_modes if mode != "fast"),
                    )
                    if not effective_modes:
                        raise UnsupportedAnswerModeError(requested_mode) from exc

            targets: list[HistoryProjectionTarget] = []
            if "research" in effective_modes:
                targets.append(
                    HistoryProjectionTarget(
                        "research_planner",
                        models.extract,
                        planner.history_input_measure(
                            request.query,
                            schema=schema,
                            current_image_descriptions=list(image_descriptions) or None,
                            preserve_query=True,
                        ),
                    )
                )
                evidence = EvidenceLedger(image_budget=resolved.image_budget)

                async def unused_retrieve(_query: str) -> RetrievalResult:
                    raise RuntimeError("acceptance tool definitions are never executed")

                tools = compose_research_tools(
                    evidence=evidence,
                    trace={},
                    retrieve_knowledge_base=unused_retrieve,
                    search_web=(
                        resolved.web_sources.search
                        if resolved.web_sources is not None and resolved.web_sources.search_enabled
                        else None
                    ),
                    resource_tools=resolved.resource_tools,
                    register_web_source=(
                        resolved.registry.register_discovered_link
                        if resolved.registry is not None and resolved.web_sources is not None
                        else None
                    ),
                )
                supplements = list(self._research_tool_supplements())
                if not memory_enabled:
                    supplements = [
                        tool
                        for tool in supplements
                        if tool.name not in {"remember", "forget", "recall_memory"}
                    ]
                try:
                    tools = list(ToolRegistry([*tools, *supplements]).resolve())
                except DuplicateToolError as exc:
                    raise InvalidToolConfigurationError(exc.names) from exc
                agent_run_plan = AgentRunPlan.from_tools(
                    tools,
                    model_role="query",
                    context_policy_revision=CONTEXT_POLICY_REVISION,
                    model_identity=asdict(self._model_fingerprint_for_role("query")),
                    model_profile=asdict(models.query),
                )
                measure = research_history_input_measure(
                    model_profile=models.query,
                    context_policy=CONTEXT_POLICY,
                    query=request.query,
                    query_images=resolved.query_images,
                    resource_manifest=resolved.resource_manifest,
                    image_budget=resolved.image_budget,
                    tools=tools,
                    retained_tail_tokens=CONTEXT_POLICY.retained_tail_target(models.query),
                    memory_text=memory_text,
                    episodic_summary=request.episodic_summary,
                )
                targets.append(
                    HistoryProjectionTarget(
                        "research_seed",
                        models.query,
                        measure,
                        proactive_compaction=True,
                    )
                )
            if "fast" in effective_modes:
                targets.extend(fast_targets)
            if requested_mode == "auto" and effective_modes >= {"fast", "research"}:
                from dlightrag.engine.answer.router import AnswerModeRouter

                async def _unused_router(**_kwargs: Any) -> str:
                    raise RuntimeError("acceptance router measure never calls the model")

                router = AnswerModeRouter(_unused_router)
                mode_resources = tuple(
                    ModeResource(
                        role=resource_role(filename=item.filename, mime_type=item.mime_type)
                    )
                    for item in (*request.attachments, *request.history_attachments)
                )
                targets.append(
                    HistoryProjectionTarget(
                        "router",
                        models.query,
                        router.history_input_measure(
                            request.query,
                            resources=mode_resources,
                            valid_modes=tuple(sorted(effective_modes)),
                        ),
                    )
                )
            try:
                history = (
                    await history_resolver(targets)
                    if history_resolver is not None
                    else project_history(
                        [dict(message) for message in request.history],
                        targets=targets,
                    )
                )
            except HistoryProjectionOverflowError as exc:
                if exc.target == "router":
                    raise UnsupportedAnswerModeError("auto") from exc
                raise AnswerInputOverflowError(str(exc)) from exc
            episodic_parts = [
                item.strip()
                for item in (request.episodic_summary, history.episodic_summary)
                if item.strip()
            ]
            return _AcceptanceProjection(
                history=tuple(dict(message) for message in history.messages),
                episodic_summary="\n\n".join(dict.fromkeys(episodic_parts)),
                image_descriptions=image_descriptions,
                pinned_models=self._pin_model_profiles(model_profiles),
                agent_run_plan=agent_run_plan,
                valid_modes=effective_modes,
            )
        finally:
            if resolved.registry is not None:
                await resolved.registry.aclose()

    def _pin_model_profiles(
        self,
        profiles: Mapping[ModelRole, ModelProfile],
    ) -> tuple[PinnedModelProfile, ...]:
        return tuple(
            PinnedModelProfile(
                role=role,
                fingerprint=self._model_fingerprint_for_role(role),
                profile=profiles[role],
            )
            for role in MODEL_ROLE_NAMES
        )


__all__ = [
    "AnswerHistoryResource",
    "AnswerInputArtifact",
    "HistoryResolver",
    "AnswerRequest",
    "AnswerRunAcceptor",
    "AnswerRuntimeUnavailableError",
    "AnswerService",
]
