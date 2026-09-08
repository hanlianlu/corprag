# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Capability-driven answer orchestrator.

One owner routes every answer. Resolved Fast mode takes the standard-RAG path: fixed knowledge-base
retrieval and one final answer generation, with no control turn. Resolved
Research mode enters the agent loop: the model selects from the available peer
tools and writes the answer when it stops calling tools.
"""

import base64
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from dlightrag_memory import Memory

from dlightrag.engine.agent.environment.access import AccessScheduler
from dlightrag.engine.agent.environment.errors import TOOL_RESULT_MAX_BYTES, TOOL_RESULT_MAX_LINES
from dlightrag.engine.agent.environment.execution import ExecutionEnvironment
from dlightrag.engine.agent.environment.toolchain import SearchToolchain
from dlightrag.engine.agent.events import AgentEvent
from dlightrag.engine.agent.session.entries import AssistantMessageEntry, CompactionEntry
from dlightrag.engine.agent.session.fold import (
    PriorTurns,
    WorkingContextProjection,
    project_session_messages,
)
from dlightrag.engine.agent.session.ids import EntryId, SessionId
from dlightrag.engine.agent.session.projection import (
    AgentInputOverflowError,
    require_compactable,
    should_compact,
)
from dlightrag.engine.agent.session.registers import (
    RequestSnapshot,
)
from dlightrag.engine.agent.session.runtime import (
    CompactionRequired,
    CompactionResult,
    RuntimeContext,
)
from dlightrag.engine.agent.skills import SkillsBundle
from dlightrag.engine.agent.tools import (
    AgentTool,
    ExecutedTurn,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.agent.tools.contracts import ToolModelFunc
from dlightrag.engine.agent.tools.files import (
    PreparedImageAttachment,
    ResourceReader,
    ResourceReadRequest,
)
from dlightrag.engine.agent.tools.registry import DuplicateToolError, ToolRegistry
from dlightrag.engine.ai.capacity import (
    CONTEXT_POLICY,
    ContextPolicy,
    ModelProfile,
)
from dlightrag.engine.ai.media import decode_image_base64, detect_image_mime
from dlightrag.engine.ai.messages import AssistantTurn, ToolDefinition
from dlightrag.engine.ai.telemetry import Telemetry
from dlightrag.engine.ai.tokens import estimate_tokens
from dlightrag.engine.answer.compaction import CompactionCoordinator
from dlightrag.engine.answer.errors import (
    AnswerInputOverflowError,
    InvalidToolConfigurationError,
)
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.images import AnswerImageBudget
from dlightrag.engine.answer.mode import ResolvedMode
from dlightrag.engine.answer.publication import PublicationLimits
from dlightrag.engine.answer.research.context import ContextAssembler
from dlightrag.engine.answer.resources.models import ResourceManifestEntry, TextWindowBudget
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.synthesizer import AnswerSynthesizer
from dlightrag.engine.answer.tools import KnowledgeRetrieval, WebSearch, compose_research_tools
from dlightrag.engine.answer.tools.memory import MemoryHost
from dlightrag.engine.answer.tools.subagents import ChildContextSnapshot, ChildRequest, SubagentHost
from dlightrag.engine.answer.workspace import RunWorkspace
from dlightrag.engine.rag.retrieval import RetrievalContexts

logger = logging.getLogger(__name__)

ToolModel = ToolModelFunc
StreamModel = Callable[..., AsyncIterator[str]]
EventSink = Callable[[AgentEvent], Awaitable[None]]
ProviderTextSink = Callable[[str], Awaitable[None]]


class PhaseBoundaries(Protocol):
    """The narrow phase/cancellation boundary shared by Fast and Research."""

    async def enter_phase(self, phase: str) -> None: ...

    async def check_cancelled(self) -> None: ...


class _NoPhaseBoundaries:
    async def enter_phase(self, phase: str) -> None:
        del phase

    async def check_cancelled(self) -> None:
        return None


@dataclass(slots=True)
class PreparedRun:
    """One run's live memory plus the wiring that executes it here.

    The working and evidence are request-local materializers rebuilt from the
    durable Session on recovery; they carry no export/restore interface.
    """

    context: ContextAssembler
    tools: list[AgentTool]
    evidence: EvidenceLedger
    working: WorkingContextProjection
    registry: ResourceRegistry | None
    trace: dict[str, Any]
    model_func: ToolModelFunc
    stream_model_func: StreamModel | None
    model_profile: ModelProfile
    attachment_snapshots: dict[str, bytes] = field(default_factory=dict)
    attachment_admissions: dict[str, int] = field(default_factory=dict)
    model_role: str = "query"
    agent_turn_count: int = 0
    stop_reason: str = "model_stop"
    last_turn: ExecutedTurn | None = None
    compaction_overflow_retried: bool = False
    streamed_terminal_text: str | None = None


class AnswerOrchestrator:
    """Route every answer through one fast or research path and one final answer."""

    def __init__(
        self,
        *,
        synthesizer: AnswerSynthesizer,
        retrieve_knowledge_base: KnowledgeRetrieval,
        search_web: WebSearch | None = None,
        model_func: ToolModel | None = None,
        stream_model_func: StreamModel | None = None,
        resource_tools: list[AgentTool] | None = None,
        resource_manifest: tuple[ResourceManifestEntry, ...] = (),
        register_web_source: Callable[[str], str | None] | None = None,
        image_budget: AnswerImageBudget | None = None,
        text_window_budget: TextWindowBudget,
        model_profile: ModelProfile,
        context_policy: ContextPolicy = CONTEXT_POLICY,
        publication_limits: PublicationLimits | None = None,
        telemetry: Telemetry,
        environment: ExecutionEnvironment | None = None,
        search_toolchain: SearchToolchain | None = None,
        resource_reader: ResourceReader | None = None,
        resolved_mode: ResolvedMode,
        subagent_host: SubagentHost | None = None,
        memory_host: MemoryHost | None = None,
        skills: SkillsBundle | None = None,
        child_model_resolver: Callable[[str], tuple[ToolModelFunc, StreamModel, ModelProfile]]
        | None = None,
    ) -> None:
        self._synthesizer = synthesizer
        self._retrieve_knowledge_base = retrieve_knowledge_base
        self._search_web = search_web
        self._model_func = model_func
        self._stream_model_func = stream_model_func
        self._resource_tools = list(resource_tools or [])
        self._resource_manifest = tuple(resource_manifest)
        self._register_web_source = register_web_source
        self._image_budget = image_budget
        self._text_window_budget = text_window_budget
        self._model_profile = model_profile
        self._context_policy = context_policy
        self._publication_limits = publication_limits or PublicationLimits()
        self._telemetry = telemetry
        self._environment = environment
        self._search_toolchain = search_toolchain
        self._resource_reader = resource_reader
        self._workspace: RunWorkspace | None = None
        self._resolved_mode: ResolvedMode = resolved_mode
        self._subagent_host = subagent_host
        self._memory_host = memory_host
        self._memory_text = ""
        self._parent_query = ""
        self._parent_history = PriorTurns()
        self._skills = skills
        self._child_model_resolver = child_model_resolver
        self._access = AccessScheduler()
        self._compaction: dict[str, CompactionCoordinator] = {}

    def bind_subagents(
        self,
        *,
        parent_session_id: SessionId,
        run_id: str,
        owner_id: str,
        persist: Any = None,
        load_child: Any = None,
        finish_child: Any = None,
        run_child: Any = None,
        check_cancelled: Any = None,
    ) -> None:
        if self._subagent_host is None:
            return
        self._subagent_host.parent_session_id = parent_session_id
        self._subagent_host.run_id = run_id
        self._subagent_host.owner_id = owner_id
        self._subagent_host.persist = persist
        self._subagent_host.load_child = load_child
        self._subagent_host.finish_child = finish_child
        self._subagent_host.run_child = run_child
        self._subagent_host.check_cancelled = check_cancelled

    def bind_child_context(
        self,
        run: PreparedRun,
        runtime_context: RuntimeContext,
    ) -> None:
        """Capture the exact parent ancestry handed to a spawned Child."""
        if self._subagent_host is None:
            return
        lane = runtime_context.snapshot.tree.lane(runtime_context.lane_id)
        parent_entry_id = lane.head_entry_id
        if parent_entry_id is None:
            raise RuntimeError("Child ContextSnapshot requires a parent Entry")
        selected = replace(
            runtime_context.snapshot,
            selected_lane_id=runtime_context.lane_id,
        )
        messages = project_session_messages(
            selected.tree.ancestry(runtime_context.lane_id),
            selected.active_projection,
        )
        _hydrate_attachment_messages(
            messages,
            run.attachment_snapshots,
            admissions=run.attachment_admissions,
        )
        self._subagent_host.context_snapshot = ChildContextSnapshot.from_values(
            parent_session_id=runtime_context.session_id,
            parent_entry_id=parent_entry_id,
            depth=self._subagent_host.depth,
            messages=messages,
            evidence_state=run.evidence.durable_state(),
        )

    def admit_durable_attachments(
        self,
        messages: list[dict[str, Any]],
        snapshots: Mapping[str, bytes],
    ) -> dict[str, int]:
        """Reserve retained attachment payloads once in this run's shared budget."""
        return _admit_durable_attachment_messages(
            messages,
            snapshots,
            self._image_budget,
        )

    def bind_memory(
        self,
        *,
        owner_id: str,
        auth_mode: str,
        run_id: str,
        session_id: str,
        store: Any,
        enabled: bool = True,
        epoch: int = 0,
        capability_current: Any = None,
    ) -> None:
        if not enabled:
            self._memory_host = None
            self._memory_text = ""
            return
        if self._memory_host is None:
            return
        self._memory_host.owner_id = owner_id
        self._memory_host.auth_mode = auth_mode
        self._memory_host.run_id = run_id
        self._memory_host.session_id = session_id
        self._memory_host.memory = Memory(store)
        self._memory_host.enabled = enabled
        self._memory_host.epoch = epoch
        self._memory_host.capability_current = capability_current

    def bind_recall(self, text: str) -> None:
        """Attach the non-citable auto-recall block for this run."""
        self._memory_text = text

    def bind_workspace(self, workspace: RunWorkspace) -> None:
        """Attach the claimed run workspace used for tools, spill, and publication."""
        self._workspace = workspace
        self._environment = workspace.environment

    def artifact_root(self) -> Path | None:
        """Return this run's request-local Artifact root, when execution owns one."""
        return None if self._workspace is None else self._workspace.workspace / "artifacts"

    @property
    def resolved_mode(self) -> ResolvedMode:
        """The durable Fast or Research path this orchestrator was built for."""
        return self._resolved_mode

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def answer_stream(
        self,
        query: str,
        *,
        conversation_history: PriorTurns | None = None,
        query_images: list[dict[str, Any]] | None = None,
        boundaries: PhaseBoundaries | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        if self._resolved_mode != "fast":
            raise RuntimeError("Research must be driven by AgentSessionRuntime")
        limits = boundaries or _NoPhaseBoundaries()
        return await self._fast_answer_stream(
            query,
            conversation_history=conversation_history,
            query_images=query_images,
            boundaries=limits,
        )

    # ------------------------------------------------------------------
    # Fast path
    # ------------------------------------------------------------------

    async def _fast_answer_stream(
        self,
        query: str,
        *,
        conversation_history: PriorTurns | None,
        query_images: list[dict[str, Any]] | None,
        boundaries: PhaseBoundaries,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        await boundaries.enter_phase("searching")
        retrieval = await self._retrieve_knowledge_base(query)
        await boundaries.check_cancelled()
        await boundaries.enter_phase("generating")
        contexts, stream = await self._synthesizer.generate_stream(
            query,
            retrieval.contexts,
            conversation_history=conversation_history,
            memory_text=self._memory_text,
            current_images=query_images,
        )
        if stream is not None:
            existing = getattr(stream, "trace", None)
            merged = (
                {**retrieval.trace, **existing} if isinstance(existing, dict) else retrieval.trace
            )
            stream.trace = merged  # type: ignore[attr-defined]
        return contexts, stream

    # ------------------------------------------------------------------
    # Research path
    # ------------------------------------------------------------------

    def runtime_answer(self, run: PreparedRun) -> tuple[RetrievalContexts, str, bool]:
        """Return a completed Research answer and whether it is already visible."""
        run.trace["agent_stop_reason"] = run.stop_reason
        text = run.last_turn.assistant.text if run.last_turn is not None else ""
        return run.evidence.contexts, text, run.streamed_terminal_text == text

    async def assemble_runtime_request(
        self,
        run: PreparedRun,
        runtime_context: RuntimeContext,
    ) -> RequestSnapshot | CompactionRequired:
        """Build one exact provider request without executing an external effect."""
        self._record_working_fold(run, runtime_context.snapshot)
        tool_schema_tokens = _tool_schema_tokens(run.tools)
        estimated = (
            run.context.measure_control_input(evidence=run.evidence, working=run.working)
            + tool_schema_tokens
        )
        if should_compact(
            run.model_profile,
            input_tokens=estimated,
            context_policy=self._context_policy,
        ):
            self._require_compactable_floor(run, tool_schema_tokens)
            return CompactionRequired()
        messages = await run.context.control_turn(
            evidence=run.evidence,
            working=run.working,
            tool_schema_tokens=tool_schema_tokens,
        )
        max_tokens = run.context.control_output_allowance(
            messages,
            tool_schema_tokens=tool_schema_tokens,
        )
        return RequestSnapshot.from_values(
            operation_id=runtime_context.operation_id,
            turn_number=getattr(runtime_context.state, "turn_count", 0) + 1,
            plan_digest=runtime_context.meta.plan_digest,
            model_role=run.model_role,
            messages=messages,
            tools=[asdict(tool.definition) for tool in run.tools],
            tool_choice="auto",
            max_tokens=max_tokens,
        )

    async def call_runtime_provider(
        self,
        request: RequestSnapshot,
        *,
        model_profile: ModelProfile | None = None,
        emit_text: ProviderTextSink | None = None,
    ) -> AssistantTurn:
        """Execute one already-persisted exact provider Request Snapshot."""
        definitions = [ToolDefinition(**definition) for definition in request.tools]
        kwargs: dict[str, Any] = {
            "messages": request.messages,
            "tools": definitions,
            "tool_choice": request.tool_choice,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        kwargs["model_profile"] = model_profile or self._model_profile
        stream_turn = getattr(self._model_func, "stream_turn", None)
        if emit_text is not None and callable(stream_turn):
            kwargs["emit_text"] = emit_text
            return await cast(Callable[..., Awaitable[AssistantTurn]], stream_turn)(**kwargs)
        return await cast(ToolModelFunc, self._model_func)(**kwargs)

    async def compact_runtime_context(
        self,
        run: PreparedRun,
        runtime_context: RuntimeContext,
        attempt: int,
    ) -> CompactionResult:
        """Prepare one automatic checkpoint compaction for Runtime settlement."""
        snapshot = runtime_context.snapshot
        coordinator = self._compaction_coordinator(run)
        tail = self._context_policy.retained_tail_target(run.model_profile) // (
            2 ** max(0, attempt - 1)
        )
        projection, _outcome = await coordinator.prepare(
            snapshot,
            tail_target_tokens=tail,
            accounted_before=(
                run.context.measure_control_input(evidence=run.evidence, working=run.working)
                + _tool_schema_tokens(run.tools)
            ),
            trace=run.trace,
        )
        return CompactionResult(
            entry=CompactionEntry(
                entry_id=EntryId.new(),
                session_id=runtime_context.session_id,
                timestamp=datetime.now(UTC),
                projection_id=projection.projection_id,
                summary=projection.summary,
                covered_through_sequence=projection.covered_through_sequence,
                first_retained_sequence=projection.first_retained_sequence,
                covered_through_entry_id=projection.covered_through_entry_id,
                first_retained_entry_id=projection.first_retained_entry_id,
                source_digest=projection.source_digest,
            ),
            projection=projection,
        )

    def restore_runtime_snapshot(self, run: PreparedRun, snapshot: Any) -> None:
        """Project a terminal Runtime snapshot into the product's live result cache."""
        self._record_working_fold(run, snapshot)
        assistants = [
            entry for entry in snapshot.graph.ancestry() if isinstance(entry, AssistantMessageEntry)
        ]
        run.agent_turn_count = len(assistants)
        run.trace["agent_turns"] = run.agent_turn_count
        if assistants:
            entry = assistants[-1]
            run.last_turn = ExecutedTurn(
                assistant=AssistantTurn(
                    text=entry.content,
                    tool_calls=entry.tool_calls,
                    stop_reason=entry.stop_reason,
                    reasoning=entry.reasoning,
                    usage_details=(dict(entry.usage) if isinstance(entry.usage, Mapping) else None),
                    cost_details=(dict(entry.cost) if isinstance(entry.cost, Mapping) else None),
                    provider_state=(
                        dict(entry.provider_state)
                        if isinstance(entry.provider_state, Mapping)
                        else None
                    ),
                ),
                results=(),
                messages=[],
            )
        run.stop_reason = "model_stop"

    # ------------------------------------------------------------------
    # Research helpers
    # ------------------------------------------------------------------

    def prepare_run(
        self,
        query: str,
        *,
        conversation_history: PriorTurns | None = None,
        query_images: list[dict[str, Any]] | None = None,
        registry: ResourceRegistry | None = None,
        attachment_snapshots: Mapping[str, bytes] | None = None,
        attachment_admissions: Mapping[str, int] | None = None,
        agent_turn_count: int = 0,
    ) -> PreparedRun:
        """Build one run's memory and the tools bound to it, before any restore."""
        if self._model_func is None:
            raise RuntimeError("Research answer requires a tool-capable model")
        self._parent_query = query
        self._parent_history = conversation_history or PriorTurns()
        evidence = EvidenceLedger(image_budget=self._image_budget)
        retained_tail_tokens = self._context_policy.retained_tail_target(self._model_profile)
        trace = _fresh_research_trace()
        skills = self._skills
        tools = self._compose_tools(
            evidence,
            trace,
            child=False,
            skill_tools=[] if skills is None else skills.tools(child=False),
        )
        return PreparedRun(
            context=ContextAssembler(
                model_profile=self._model_profile,
                context_policy=self._context_policy,
                query=query,
                history=conversation_history or PriorTurns(),
                query_images=query_images,
                resource_manifest=self._resource_manifest,
                memory_text=self._memory_text,
                contributions=() if skills is None else skills.context_contributions(),
                tool_guidance=_tool_guidance(tools),
                profile_memory_write=any(tool.name == "remember" for tool in tools),
                artifact_publication=any(tool.name == "attach_artifact" for tool in tools),
            ),
            tools=tools,
            evidence=evidence,
            working=WorkingContextProjection(retained_tail_tokens=retained_tail_tokens),
            registry=registry,
            trace=trace,
            attachment_snapshots=dict(attachment_snapshots or {}),
            attachment_admissions=dict(attachment_admissions or {}),
            model_func=self._model_func,
            stream_model_func=self._stream_model_func,
            model_profile=self._model_profile,
            agent_turn_count=agent_turn_count,
        )

    def prepare_child_session(
        self,
        request: ChildRequest,
        *,
        context_snapshot: ChildContextSnapshot,
        child_session_id: str = "",
    ) -> PreparedRun:
        """Build a bounded child with selected context and inherited tool subset."""
        if self._child_model_resolver is None:
            if request.model_role != "query" or self._model_func is None:
                raise ValueError(f"child model role is not tool-capable: {request.model_role}")
            child_model = self._model_func
            child_stream = self._stream_model_func
            child_profile = self._model_profile
        else:
            child_model, child_stream, child_profile = self._child_model_resolver(
                request.model_role
            )
        evidence = EvidenceLedger(image_budget=self._image_budget)
        if request.context == "parent" and context_snapshot.evidence_state:
            evidence.restore_ledger_state(context_snapshot.evidence_state)
        retained_tail_tokens = self._context_policy.retained_tail_target(child_profile)
        trace = _fresh_research_trace()
        trace["child_context"] = request.context
        trace["child_model_role"] = request.model_role
        skills = self._skills
        tools = self._compose_tools(
            evidence,
            trace,
            child=True,
            skill_tools=[] if skills is None else skills.tools(child=True),
            tool_names=request.tools,
            child_session_id=child_session_id,
        )
        history = PriorTurns()
        if request.context == "parent":
            history = PriorTurns(context_snapshot.messages)
        trace["child_depth"] = context_snapshot.depth + 1
        trace["parent_entry_id"] = context_snapshot.parent_entry_id.value
        return PreparedRun(
            context=ContextAssembler(
                model_profile=child_profile,
                context_policy=self._context_policy,
                query=child_question(request.objective),
                history=history,
                query_images=None,
                resource_manifest=self._resource_manifest,
                memory_text=self._memory_text,
                contributions=() if skills is None else skills.context_contributions(),
                tool_guidance=_tool_guidance(tools),
            ),
            tools=tools,
            evidence=evidence,
            working=WorkingContextProjection(retained_tail_tokens=retained_tail_tokens),
            registry=None,
            trace=trace,
            model_func=child_model,
            stream_model_func=child_stream,
            model_profile=child_profile,
            model_role=request.model_role,
        )

    def _record_working_fold(self, run: PreparedRun, snapshot: Any) -> None:
        """Replace the working with the projection-retained fold plus its summary."""
        projection = snapshot.active_projection
        graph = getattr(snapshot, "graph", None)
        entries = graph.ancestry() if graph is not None else snapshot.entries
        messages = project_session_messages(entries, projection)
        _hydrate_attachment_messages(
            messages,
            run.attachment_snapshots,
            admissions=run.attachment_admissions,
        )
        working = WorkingContextProjection(
            retained_tail_tokens=self._context_policy.retained_tail_target(run.model_profile)
        )
        self._record_exchanges(working, messages)
        run.working = working

    @staticmethod
    def _record_exchanges(
        working: WorkingContextProjection, messages: list[dict[str, Any]]
    ) -> None:
        exchanges: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls") and current:
                exchanges.append(current)
                current = [message]
            else:
                current.append(message)
        if current:
            exchanges.append(current)
        for exchange in exchanges:
            working.record(exchange)

    def _compose_tools(
        self,
        evidence: EvidenceLedger,
        trace: dict[str, Any],
        *,
        child: bool,
        skill_tools: list[AgentTool],
        tool_names: tuple[str, ...] | None = None,
        child_session_id: str = "",
    ) -> list[AgentTool]:
        subagent_host = self._subagent_host
        if subagent_host is not None and not child:

            def merge_child(state: Any, child_id: str, call_id: str) -> tuple[str, ...]:
                before = len(evidence.contexts["chunks"])
                evidence.merge_child_state(
                    state,
                    child_session_id=child_id,
                    parent_call_id=call_id,
                )
                return tuple(evidence.citation_handles(after_chunk_count=before))

            subagent_host.merge_evidence = merge_child

            def record_child_usage(usage: Mapping[str, int]) -> None:
                inclusive = trace.setdefault("child_usage", {})
                for key, value in usage.items():
                    inclusive[key] = int(inclusive.get(key, 0)) + int(value)

            subagent_host.record_usage = record_child_usage
        memory_host = self._memory_host
        if child and child_session_id and memory_host is not None:
            memory_host = replace(memory_host, session_id=child_session_id)
        composed = compose_research_tools(
            evidence=evidence,
            trace=trace,
            retrieve_knowledge_base=self._retrieve_knowledge_base,
            search_web=self._search_web,
            resource_tools=self._resource_tools,
            register_web_source=self._register_web_source,
            resource_reader=self._resource_reader_for_run(),
            environment=self._environment,
            scheduler=self._access,
            search_toolchain=self._search_toolchain,
            image_preparer=self._prepare_local_image,
            spill=(None if self._workspace is None else self._spill_writer()),
            output_stage_factory=(
                None if self._workspace is None else self._output_stage_factory()
            ),
            artifacts_root=self.artifact_root(),
            publication_limits=self._publication_limits,
            subagent_host=subagent_host,
            memory_host=memory_host,
            skill_tools=skill_tools,
            child=child,
        )
        try:
            registry = ToolRegistry(composed)
        except DuplicateToolError as exc:
            raise InvalidToolConfigurationError(exc.names) from exc
        return list(
            registry.resolve(
                tool_names,
                exclude={
                    "spawn_agent",
                    "subagent_status",
                    "wait_subagent",
                    "cancel_subagent",
                }
                if child
                else (),
            )
        )

    def _prepare_local_image(
        self,
        data: bytes,
        path: str,
    ) -> PreparedImageAttachment | None:
        budget = self._image_budget
        if budget is None:
            return None
        block = budget.add_base64(base64.b64encode(data).decode("ascii"), label=path)
        if block is None:
            return None
        value = block.get("image_url")
        if not isinstance(value, dict) or not isinstance(value.get("url"), str):
            return None
        prepared, declared_media_type = decode_image_base64(value["url"])
        return PreparedImageAttachment(
            data=prepared,
            media_type=detect_image_mime(prepared, fallback=declared_media_type),
            transformed=prepared != data,
        )

    def _resource_reader_for_run(self) -> ResourceReader | None:
        base_reader = self._resource_reader
        if base_reader is None and self._workspace is None:
            return None

        async def read(
            request: ResourceReadRequest,
            runtime: ToolRuntime,
        ) -> ToolResult:
            workspace = self._workspace
            resource_id = request.resource_id or ""
            if workspace is not None and resource_id.startswith("spill_"):
                return _read_committed_spill(workspace.spill_dir, resource_id, request.cursor)
            if base_reader is None:
                return ToolResult.text("resource read is not available", is_error=True)
            return await base_reader(request, runtime)

        return read

    def _output_stage_factory(self) -> Any:
        from dlightrag.engine.answer.workspace import FileOutputStage

        workspace = self._workspace
        if workspace is None:
            raise RuntimeError("output staging requires a bound workspace")

        def create(tool: str) -> FileOutputStage:
            return FileOutputStage(
                workspace.spill_dir,
                f"spill_{tool}_{uuid4().hex}",
            )

        return create

    def _spill_writer(self) -> Any:
        from dlightrag.engine.answer.workspace import spill_receipt, write_spill_file

        workspace = self._workspace
        if workspace is None:
            raise RuntimeError("spill requires a bound workspace")

        async def write(text: str) -> Any:
            resource_id = f"spill_{uuid4().hex}"
            write_spill_file(workspace.spill_dir, resource_id, text)
            return spill_receipt(resource_id, text)

        return write

    def _compaction_coordinator(self, run: PreparedRun) -> CompactionCoordinator:
        if run.model_role not in self._compaction:
            if run.stream_model_func is None:
                raise RuntimeError("Research compaction requires a streaming model")
            self._compaction[run.model_role] = CompactionCoordinator(
                model_profile=run.model_profile,
                context_policy=self._context_policy,
                stream_model=run.stream_model_func,
            )
        return self._compaction[run.model_role]

    def _require_compactable_floor(self, run: PreparedRun, tool_schema_tokens: int) -> None:
        """Fail before any compaction or model call when the fixed envelope alone
        cannot fit the hard limit — shrinking branch ancestry can never help."""
        fixed = (
            run.context.measure_control_input(
                evidence=EvidenceLedger(),
                working=WorkingContextProjection(retained_tail_tokens=0),
            )
            + tool_schema_tokens
        )
        try:
            require_compactable(
                run.model_profile,
                input_tokens=fixed,
                fixed_input_tokens=fixed,
                context_policy=self._context_policy,
            )
        except AgentInputOverflowError as exc:
            raise AnswerInputOverflowError(str(exc)) from exc


def _admit_durable_attachment_messages(
    messages: list[dict[str, Any]],
    snapshots: Mapping[str, bytes],
    budget: AnswerImageBudget | None,
) -> dict[str, int]:
    """Rebuild one run's image budget from retained durable attachment occurrences."""
    admitted: dict[str, int] = {}
    for message in messages:
        attachments = message.get("attachments")
        if not isinstance(attachments, list):
            continue
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            resource_id = str(attachment.get("resource_id") or "")
            content = snapshots.get(resource_id)
            if content is None:
                continue
            _validate_attachment_snapshot(attachment, content)
            media_type = str(attachment.get("media_type") or "application/octet-stream")
            if media_type.startswith("image/"):
                if budget is None or not budget.reserve_prepared(
                    content,
                    label=f"durable_attachment:{resource_id}",
                ):
                    continue
            admitted[resource_id] = admitted.get(resource_id, 0) + 1
    return admitted


def _hydrate_attachment_messages(
    messages: list[dict[str, Any]],
    snapshots: Mapping[str, bytes],
    *,
    admissions: Mapping[str, int] | None = None,
) -> None:
    """Restore transport-private attachment bytes after durable Session decode."""
    occurrences: dict[str, int] = {}
    for message in messages:
        attachments = message.get("attachments")
        if not isinstance(attachments, list):
            continue
        for attachment in attachments:
            if not isinstance(attachment, dict) or attachment.get("data_url"):
                continue
            resource_id = str(attachment.get("resource_id") or "")
            content = snapshots.get(resource_id)
            if content is None:
                continue
            occurrence = occurrences.get(resource_id, 0) + 1
            occurrences[resource_id] = occurrence
            if admissions is not None and occurrence > admissions.get(resource_id, 0):
                continue
            _validate_attachment_snapshot(attachment, content)
            media_type = str(attachment.get("media_type") or "application/octet-stream")
            encoded = base64.b64encode(content).decode("ascii")
            attachment["data_url"] = f"data:{media_type};base64,{encoded}"


def _validate_attachment_snapshot(attachment: Mapping[str, Any], content: bytes) -> None:
    digest = str(attachment.get("content_digest") or "")
    size = int(attachment.get("size_bytes") or 0)
    if hashlib.sha256(content).hexdigest() != digest or len(content) != size:
        raise ValueError("durable tool attachment does not match its Blob snapshot")


def _read_committed_spill(
    spill_dir: Path,
    resource_id: str,
    cursor: str | None,
) -> ToolResult:
    if not resource_id.isascii() or not all(
        character.isalnum() or character == "_" for character in resource_id
    ):
        return ToolResult.text("invalid spill resource id", is_error=True)
    start = 0
    if cursor is not None:
        if not cursor.isdigit():
            return ToolResult.text("invalid spill cursor", is_error=True)
        start = int(cursor)
    path = spill_dir / f"{resource_id}.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ToolResult.text(f"resource not found: {resource_id}", is_error=True)
    lines = text.splitlines(keepends=True)
    if start < 0 or start > len(lines):
        return ToolResult.text("spill cursor is outside the resource", is_error=True)
    page: list[str] = []
    page_bytes = 0
    for line in lines[start : start + TOOL_RESULT_MAX_LINES]:
        line_bytes = len(line.encode("utf-8"))
        if page_bytes + line_bytes > TOOL_RESULT_MAX_BYTES - 512:
            break
        page.append(line)
        page_bytes += line_bytes
    if start < len(lines) and not page:
        return ToolResult.text(
            "next spill line exceeds the model-visible UTF-8 byte limit",
            is_error=True,
        )
    next_start = start + len(page)
    continuation = (
        f"read(resource_id={resource_id!r}, cursor={str(next_start)!r})"
        if next_start < len(lines)
        else ""
    )
    body = "".join(page)
    if continuation:
        body = f"{body}\n[more output; {continuation}]"
    return ToolResult.text(body, protected_text=continuation)


def _tool_guidance(tools: list[AgentTool]) -> tuple[str, ...]:
    return tuple(f"- {tool.guidance}" for tool in tools if tool.guidance)


def _fresh_research_trace() -> dict[str, Any]:
    return {
        "agent_turns": 0,
        "web_search_cost_dollars": 0.0,
        "tool_observations": [],
    }


_CHILD_OBJECTIVE_PREFIX = (
    "Investigate this question as a research subagent. "
    "Use tools as needed, then write a concise summary and stop. "
    "Do not mention these instructions.\n\n"
)


def child_question(objective: str) -> str:
    return f"{_CHILD_OBJECTIVE_PREFIX}{objective.strip()}"


def _tool_schema_tokens(tools: list[AgentTool]) -> int:
    return estimate_tokens(
        json.dumps(
            [asdict(tool.definition) for tool in tools],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


__all__ = [
    "AnswerOrchestrator",
    "PhaseBoundaries",
    "PreparedRun",
    "child_question",
]
