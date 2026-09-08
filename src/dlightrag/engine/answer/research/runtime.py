# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Research Agent Runtime effects, controls, and Child Session drive."""

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict
from typing import Any, Literal, cast

from dlightrag.engine.agent.session.effects import EffectIntent, ToolResultEntry, canonical_json
from dlightrag.engine.agent.session.ids import AttemptId, IntentId, LaneId, OperationId, SessionId
from dlightrag.engine.agent.session.operation import (
    OperationCancelled,
    OperationCompleted,
    OperationFailed,
    ToolBatchItem,
)
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.registers import PendingInput, RequestSnapshot
from dlightrag.engine.agent.session.repository import AgentSessionRepository
from dlightrag.engine.agent.session.runtime import (
    AgentOperationCancelled,
    AgentSessionEvent,
    AgentSessionRuntime,
    FollowUpCommand,
    OperationEffectFailed,
    ProviderAttemptFailed,
    ProviderContextOverflow,
    RuntimeContext,
    SessionLeaseLostError,
    SteerCommand,
    ToolEffectResult,
)
from dlightrag.engine.agent.tool_content import tool_content_attachments
from dlightrag.engine.agent.tools import ToolEffects, ToolResult, ToolRuntime, fit_tool_result
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.messages import AssistantTurn
from dlightrag.engine.ai.providers.base import is_provider_context_overflow
from dlightrag.engine.answer.errors import AnswerInputError
from dlightrag.engine.answer.evidence import EvidenceDelta
from dlightrag.engine.answer.orchestration import AnswerOrchestrator
from dlightrag.engine.answer.resources.registry import (
    FetchedBytesSink,
    FetchedResourceBytes,
    ResourceEffectOwner,
)
from dlightrag.engine.answer.tools.subagents import (
    ChildContextSnapshot,
    ChildOutcome,
    ChildRequest,
    SpawnAgentInput,
)
from dlightrag.engine.dependencies import classify_transient_dependency
from dlightrag.engine.runtime.blob_chunks import blob_digest, plan_blob
from dlightrag.engine.runtime.coordinator import (
    LeaseLostError,
    RunCancellationObserved,
    RunSession,
)
from dlightrag.engine.runtime.errors import (
    IncompatibleActiveRunError,
    RunExecutionError,
)
from dlightrag.engine.runtime.policy import RUN_LEASE_SECONDS
from dlightrag.engine.runtime.settlements import (
    ArtifactAttachmentUpdate,
    CommittedSpillUpdate,
    CompleteBlobDescriptor,
    EffectHostUpdate,
    FetchedResourceSettlementUpdate,
    InventoryPathRecord,
    MemoryOperationSettlement,
    OpaqueEvidenceWrite,
    OpaqueFetchedResourceWrite,
    WorkspaceInventoryUpdate,
)

logger = logging.getLogger(__name__)
_CHILD_LEASE_HEARTBEAT_SECONDS = RUN_LEASE_SECONDS / 3


def _research_dynamic_context_reserve(profile: ModelProfile) -> int:
    """Return the pinned profile's effective Research observation capacity."""
    hard_limit = CONTEXT_POLICY.hard_input_limit(profile)
    trigger = CONTEXT_POLICY.compaction_trigger(profile)
    return max(0, hard_limit - trigger)


class FetchedResourceBuffer:
    """Fetched bytes partitioned by explicit Session and model tool call."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], dict[str, FetchedResourceBytes]] = {}

    def append(
        self,
        fetched: FetchedResourceBytes,
        owner: ResourceEffectOwner | None,
    ) -> None:
        key = (owner.execution_scope, owner.intent_id.value) if owner is not None else ("", "")
        self._items.setdefault(key, {})[fetched.resource_id] = fetched

    def drain(self, *, scope: str, intent_id: IntentId) -> tuple[FetchedResourceBytes, ...]:
        fetched = list(self._items.pop((scope, intent_id.value), {}).values())
        # Acceptance-time fetches happen before a tool task has a scope. Bind
        # them to the first durable settlement rather than sharing a live list.
        fetched.extend(self._items.pop(("", ""), {}).values())
        return tuple(fetched)


def _buffered_fetched_bytes_sink(buffer: FetchedResourceBuffer) -> FetchedBytesSink:
    """Buffer fetched bytes until their explicit owning intent settles."""

    async def persist(
        fetched: FetchedResourceBytes,
        owner: ResourceEffectOwner | None,
    ) -> None:
        buffer.append(fetched, owner)

    return persist


def _memory_operation_settlement(
    details: Mapping[str, Any] | None,
) -> MemoryOperationSettlement | None:
    """Decode the product-owned Memory receipt envelope; never infer from text."""
    if not isinstance(details, Mapping):
        return None
    raw = details.get("memory_operation")
    if not isinstance(raw, Mapping):
        return None
    operation = str(raw.get("operation") or "")
    outcome = str(raw.get("outcome") or "")
    if operation not in {"remember", "forget", "undo"}:
        return None
    if outcome not in {"changed", "unchanged", "rejected", "conflict"}:
        return None
    memory_ids = raw.get("memory_ids")
    return MemoryOperationSettlement(
        operation=operation,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        change_id=str(raw["change_id"]) if raw.get("change_id") else None,
        memory_ids=(
            tuple(str(memory_id) for memory_id in memory_ids)
            if isinstance(memory_ids, list | tuple)
            else ()
        ),
        kind=str(raw["kind"]) if raw.get("kind") else None,
        body=str(raw.get("body") or ""),
        supersedes_id=str(raw["supersedes_id"]) if raw.get("supersedes_id") else None,
        target_change_id=(str(raw["target_change_id"]) if raw.get("target_change_id") else None),
    )


def _artifact_attachment_update(
    details: Mapping[str, Any] | None,
    *,
    session_id: SessionId,
    intent_id: IntentId,
) -> ArtifactAttachmentUpdate | None:
    """Decode the product-owned attachment receipt; never infer it from text."""
    if not isinstance(details, Mapping):
        return None
    raw = details.get("artifact_attachment")
    if not isinstance(raw, Mapping):
        return None
    try:
        return ArtifactAttachmentUpdate(
            relative_path=str(raw["relative_path"]),
            label=str(raw.get("label") or ""),
            content_digest=str(raw["content_digest"]),
            size_bytes=int(raw["size_bytes"]),
            presentation=str(raw["presentation"]),
            session_id=session_id.value,
            intent_id=intent_id.value,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid Artifact attachment receipt") from exc


def _build_effect_host_update(
    *,
    session_id: SessionId,
    intent: EffectIntent,
    ledger_state: Callable[[], str],
    fetched_buffer: FetchedResourceBuffer,
    execution_scope: str,
    tool_effects: ToolEffects,
    details: Mapping[str, Any] | None = None,
) -> EffectHostUpdate:
    """Convert one product Tool result into its typed atomic HostDelta."""
    evidence: tuple[OpaqueEvidenceWrite, ...] = ()
    encoded_ledger = ledger_state()
    if encoded_ledger != "{}":
        content = encoded_ledger.encode("utf-8")
        evidence = (
            OpaqueEvidenceWrite(
                session_id=session_id.value,
                intent_id=intent.intent_id.value,
                result_ordinal=0,
                content_digest=blob_digest(content),
                locator_digest=blob_digest(b"{}"),
                content=content,
                locator=b"{}",
            ),
        )
    fetched_updates: list[FetchedResourceSettlementUpdate] = []
    fetched_digests: dict[str, str] = {}
    for fetched in fetched_buffer.drain(
        scope=execution_scope,
        intent_id=intent.intent_id,
    ):
        digest = blob_digest(fetched.content)
        fetched_digests[fetched.resource_id] = digest
        plan = plan_blob(fetched.content)
        fetched_updates.append(
            FetchedResourceSettlementUpdate(
                resource=OpaqueFetchedResourceWrite(
                    resource_id=fetched.resource_id,
                    ordinal=fetched.ordinal,
                    safe_name=fetched.filename,
                    media_type=fetched.mime_type,
                    capabilities={
                        "resource_kind": "web",
                        "admission_origin": fetched.admission_origin,
                        "acquisition": fetched.acquisition,
                        "resource_aliases": list(fetched.aliases),
                    },
                    blob_digest=plan.digest,
                    source_locator_digest=blob_digest(fetched.url.encode("utf-8")),
                    source_locator=fetched.url.encode("utf-8"),
                    session_id=session_id.value,
                    intent_id=intent.intent_id.value,
                ),
                complete_blob=CompleteBlobDescriptor(
                    digest=plan.digest,
                    total_bytes=plan.total_bytes,
                    chunks=tuple(
                        plan.chunk(fetched.content, index) for index in range(plan.chunk_count)
                    ),
                ),
            )
        )
    inventory = tool_effects.workspace_inventory

    def blob_descriptor(content: bytes) -> CompleteBlobDescriptor:
        plan = plan_blob(content)
        return CompleteBlobDescriptor(
            digest=plan.digest,
            total_bytes=plan.total_bytes,
            chunks=tuple(plan.chunk(content, index) for index in range(plan.chunk_count)),
        )

    attached_updates: list[FetchedResourceSettlementUpdate] = []
    for attached in tool_effects.attached_resources:
        digest = blob_digest(attached.content)
        fetched_digest = fetched_digests.get(attached.resource_id)
        if fetched_digest is not None:
            if fetched_digest != digest:
                raise ValueError("Web snapshot and tool attachment bytes disagree")
            continue
        attached_updates.append(
            FetchedResourceSettlementUpdate(
                resource=OpaqueFetchedResourceWrite(
                    resource_id=attached.resource_id,
                    ordinal=0,
                    safe_name=attached.filename,
                    media_type=attached.mime_type,
                    capabilities={"resource_kind": "tool_attachment"},
                    blob_digest=digest,
                    source_locator_digest=blob_digest(attached.source_locator.encode("utf-8")),
                    source_locator=attached.source_locator.encode("utf-8"),
                    session_id=session_id.value,
                    intent_id=intent.intent_id.value,
                ),
                complete_blob=blob_descriptor(attached.content),
            )
        )
    return EffectHostUpdate(
        evidence=evidence,
        fetched=(*fetched_updates, *attached_updates),
        committed_outputs=tuple(
            CommittedSpillUpdate(
                resource_id=output.resource_id,
                content_digest=output.content_digest,
                size_bytes=output.size_bytes,
                session_id=session_id.value,
                intent_id=intent.intent_id.value,
            )
            for output in tool_effects.committed_outputs
        ),
        workspace_inventory=(
            None
            if inventory is None
            else WorkspaceInventoryUpdate(
                upserts=tuple(
                    InventoryPathRecord(
                        relative_path=record.relative_path,
                        entry_type=record.entry_type,
                        size_bytes=record.size_bytes,
                        mode=record.mode,
                        content_digest=record.content_digest,
                    )
                    for record in inventory.upserts
                ),
                deletes=inventory.deletes,
                replace_all=inventory.replace_all,
            )
        ),
        artifact_attachment=(
            _artifact_attachment_update(
                details,
                session_id=session_id,
                intent_id=intent.intent_id,
            )
            if intent.tool_name == "attach_artifact"
            else None
        ),
        memory_operation=_memory_operation_settlement(details),
    )


class AnswerRuntimeControls:
    """Translate ordered Answer control rows into typed Runtime controls."""

    def __init__(
        self,
        *,
        reader: Callable[[], Awaitable[tuple[Mapping[str, Any], ...]]],
        acknowledge: Callable[[tuple[int, ...]], Awaitable[bool]],
    ) -> None:
        self._reader = reader
        self._acknowledge = acknowledge
        self._sequences: dict[str, int] = {}

    async def poll(self, context: RuntimeContext) -> tuple[Any, ...]:
        del context
        commands: list[Any] = []
        for row in await self._reader():
            sequence = int(row.get("control_sequence") or 0)
            kind = str(row.get("kind") or "steer")
            content = str(row.get("content") or "")
            command_id = f"answer-control:{sequence}"
            self._sequences[command_id] = sequence
            if kind == "follow_up":
                commands.append(
                    FollowUpCommand(
                        command_id=command_id,
                        input_id=command_id,
                        idempotency_key=command_id,
                        content=content,
                    )
                )
            else:
                commands.append(SteerCommand(command_id=command_id, content=content))
        return tuple(commands)

    async def acknowledge(self, command_ids: tuple[str, ...]) -> bool:
        sequences = tuple(self._sequences[command_id] for command_id in command_ids)
        return await self._acknowledge(sequences)


class ResearchRuntimeEffects:
    """Answer Host adapters for the product-neutral AgentSessionRuntime effects."""

    def __init__(
        self,
        *,
        orchestrator: AnswerOrchestrator,
        prepared: Any,
        session: RunSession,
        session_id: SessionId,
        fetched_buffer: FetchedResourceBuffer,
        persist_child_intent: Callable[..., Awaitable[Any]] | None,
        validate_pins: Callable[[], None] | None = None,
        publish_provider_text: bool = False,
    ) -> None:
        self._orchestrator = orchestrator
        self._prepared = prepared
        self._session = session
        self._session_id = session_id
        self._fetched_buffer = fetched_buffer
        self._persist_child_intent = persist_child_intent
        self._validate_pins = validate_pins
        self._publish_provider_text = publish_provider_text
        self._tools = {tool.name: tool for tool in prepared.tools}

    async def assemble_request(self, context: RuntimeContext) -> RequestSnapshot | Any:
        await self._check_cancelled()
        self._check_pins()
        try:
            return await self._orchestrator.assemble_runtime_request(self._prepared, context)
        except AnswerInputError as exc:
            raise OperationEffectFailed("context_overflow", str(exc)) from exc

    async def call_provider(
        self,
        context: RuntimeContext,
        request: RequestSnapshot,
        attempt_id: AttemptId,
        emit_ephemeral: Any,
    ) -> AssistantTurn:
        await self._check_cancelled()
        self._check_pins()
        await emit_ephemeral(
            AgentSessionEvent(
                kind="model_start",
                session_id=context.session_id,
                lane_id=context.lane_id,
                operation_id=context.operation_id,
                commit_sequence=None,
                data={"attempt_id": attempt_id.value},
                ephemeral=True,
            )
        )
        emitted: list[str] = []

        async def emit_text(text: str) -> None:
            if not text:
                return
            if not emitted:
                await self._session.enter_phase("generating")
            emitted.append(text)
            await self._session.emit_token(text)
            await emit_ephemeral(
                AgentSessionEvent(
                    kind="provider_delta",
                    session_id=context.session_id,
                    lane_id=context.lane_id,
                    operation_id=context.operation_id,
                    commit_sequence=None,
                    data={"text": text},
                    ephemeral=True,
                )
            )

        try:
            assistant = await self._orchestrator.call_runtime_provider(
                request,
                model_profile=self._prepared.model_profile,
                emit_text=emit_text if self._publish_provider_text else None,
            )
        except asyncio.CancelledError:
            raise
        except RunCancellationObserved as exc:
            if emitted:
                await self._session.reset_output()
            raise AgentOperationCancelled(exc) from exc
        except Exception as exc:
            if emitted:
                await self._session.reset_output()
            if is_provider_context_overflow(exc):
                raise ProviderContextOverflow from exc
            retryable = (
                classify_transient_dependency(exc, component_hint="providers") == "providers"
            )
            raise ProviderAttemptFailed(
                "Model provider is temporarily unavailable"
                if retryable
                else "Model provider rejected the request",
                retryable=retryable,
            ) from exc

        streamed_text = "".join(emitted)
        if assistant.tool_calls or streamed_text != assistant.text:
            if emitted:
                await self._session.reset_output()
            if assistant.tool_calls and emitted:
                await self._session.enter_phase("researching")
            self._prepared.streamed_terminal_text = None
        elif emitted:
            self._prepared.streamed_terminal_text = streamed_text

        await emit_ephemeral(
            AgentSessionEvent(
                kind="model_end",
                session_id=context.session_id,
                lane_id=context.lane_id,
                operation_id=context.operation_id,
                commit_sequence=None,
                data={
                    "attempt_id": attempt_id.value,
                    "stop_reason": assistant.stop_reason,
                    "tool_calls": len(assistant.tool_calls),
                },
                ephemeral=True,
            )
        )
        return assistant

    async def execute_tool(
        self,
        context: RuntimeContext,
        item: ToolBatchItem,
        arguments: Mapping[str, Any],
        attempt_id: AttemptId,
        emit_ephemeral: Any,
    ) -> ToolEffectResult[EffectHostUpdate]:
        await self._check_cancelled()
        self._check_pins()
        self._orchestrator.bind_child_context(self._prepared, context)
        if item.intent_id is None:
            raise RuntimeError("executable Tool item lost its IntentId")
        tool = self._tools.get(item.tool_name)
        if tool is None:
            raise RuntimeError(f"Tool contract unavailable: {item.tool_name}")
        await self._bind_subagent_intent(item, arguments)
        validated = tool.input_model.model_validate(arguments)

        async def update(result: ToolResult) -> None:
            details = result.details if isinstance(result.details, dict) else {}
            object_label = details.get("object_label")
            await emit_ephemeral(
                AgentSessionEvent(
                    kind="tool_update",
                    session_id=context.session_id,
                    lane_id=context.lane_id,
                    operation_id=context.operation_id,
                    commit_sequence=None,
                    data={
                        "tool_name": item.tool_name,
                        "call_id": item.call_id,
                        "source_position": item.source_index,
                        "text_chars": len(result.text_content),
                        **(
                            {"object_label": object_label}
                            if isinstance(object_label, str) and object_label
                            else {}
                        ),
                    },
                    ephemeral=True,
                )
            )

        await emit_ephemeral(
            AgentSessionEvent(
                kind="tool_start",
                session_id=context.session_id,
                lane_id=context.lane_id,
                operation_id=context.operation_id,
                commit_sequence=None,
                data={
                    "tool_name": item.tool_name,
                    "call_id": item.call_id,
                    "source_position": item.source_index,
                    "attempt_id": attempt_id.value,
                },
                ephemeral=True,
            )
        )
        runtime = ToolRuntime(
            call_id=item.call_id,
            tool_name=item.tool_name,
            intent_id=item.intent_id,
            execution_scope=self._session_id.value,
            _update_sink=update,
        )
        result = await tool.execute(validated, runtime)
        observation_capacity = _research_dynamic_context_reserve(self._prepared.model_profile)
        fitted = fit_tool_result(
            result,
            max_tokens=observation_capacity,
        )
        outcome = "failed" if fitted.is_error else "succeeded"
        durable = ToolResultEntry(
            tool_name=item.tool_name,
            call_id=item.call_id,
            outcome=outcome,
            parts=fitted.parts,
            details=fitted.details,
            cached=fitted.cached,
        )
        self._prepared.trace["tool_observations"].append(
            {
                "tool": item.tool_name,
                "call_id": item.call_id,
                "outcome": outcome,
                "cached": fitted.cached,
                "is_error": fitted.is_error,
                "content_chars": len(fitted.text_content),
                "capacity_tokens": observation_capacity,
            }
        )
        intent = EffectIntent(
            intent_id=item.intent_id,
            tool_name=item.tool_name,
            replay_policy=item.replay_policy,
            contract_version=item.contract_version,
            input_schema_digest=item.input_schema_digest,
            canonical_input=canonical_json(arguments),
            source_call_id=item.call_id,
        )
        for attachment in tool_content_attachments(durable.parts):
            if attachment.data:
                self._prepared.attachment_snapshots[attachment.resource_id] = attachment.data
                admissions = self._prepared.attachment_admissions
                admissions[attachment.resource_id] = admissions.get(attachment.resource_id, 0) + 1
        delta = _build_effect_host_update(
            session_id=self._session_id,
            intent=intent,
            ledger_state=lambda: self._prepared.evidence.ledger_state_json(),
            fetched_buffer=self._fetched_buffer,
            execution_scope=self._session_id.value,
            tool_effects=fitted.effects,
            details=fitted.details,
        )
        return ToolEffectResult(durable, delta)

    async def compact(self, context: RuntimeContext, attempt: int) -> Any:
        return await self._orchestrator.compact_runtime_context(
            self._prepared,
            context,
            attempt,
        )

    async def _bind_subagent_intent(
        self,
        item: ToolBatchItem,
        arguments: Mapping[str, Any],
    ) -> None:
        if (
            self._persist_child_intent is None
            or item.tool_name != "spawn_agent"
            or item.intent_id is None
        ):
            return
        from dlightrag.engine.answer.tools.subagents import child_session_id

        spawn = SpawnAgentInput.model_validate(arguments)
        for position, request in enumerate(spawn.children):
            child_id = child_session_id(
                run_id=self._session.run_id,
                parent_session_id=self._session_id,
                parent_intent_id=item.intent_id,
                position=position,
            )
            await self._persist_child_intent(
                owner_id=self._session.owner_id,
                run_id=self._session.run_id,
                child_session_id=child_id.value,
                parent_session_id=self._session_id.value,
                parent_call_id=item.call_id,
                parent_intent_id=item.intent_id.value,
                objective=request.objective,
                context_mode=request.context,
                model_role=request.model_role,
                tools=request.tools,
            )

    def _check_pins(self) -> None:
        if self._validate_pins is None:
            return
        try:
            self._validate_pins()
        except IncompatibleActiveRunError as exc:
            raise OperationEffectFailed("plan_unavailable", str(exc)) from exc

    async def _check_cancelled(self) -> None:
        try:
            await self._session.check_cancelled()
        except RunCancellationObserved as exc:
            raise AgentOperationCancelled(exc) from exc


def _answer_runtime_event_sink(
    session: RunSession,
) -> Callable[[AgentSessionEvent], Awaitable[None]]:
    async def publish(event: AgentSessionEvent) -> None:
        event_type = {
            "tool_start": "tool_start",
            "tool_update": "tool_progress",
            "tool_result_committed": "tool_end",
        }.get(event.kind)
        if event_type is None:
            return
        allowed = {
            "tool_name",
            "call_id",
            "source_position",
            "source_index",
            "outcome",
            "text_chars",
            "attempt_id",
            "object_label",
        }
        payload = {
            key: value
            for key, value in event.data.items()
            if key in allowed and isinstance(value, (str, int, float, bool))
        }
        if event.commit_sequence is not None:
            payload["session_commit_sequence"] = event.commit_sequence
        await session.emit_tool_event(event_type, payload)

    return publish


async def _drive_answer_operation(
    runtime: AgentSessionRuntime[EffectHostUpdate],
    *,
    session: RunSession,
    session_id: SessionId,
    operation_id: OperationId,
) -> Any:
    try:
        return await runtime.drive(session_id=session_id, operation_id=operation_id)
    except (
        asyncio.CancelledError,
        RunCancellationObserved,
        AgentOperationCancelled,
    ) as exc:
        await runtime.cancel(session_id=session_id, operation_id=operation_id)
        await runtime.close(session_id=session_id, operation_id=operation_id)
        if isinstance(exc, AgentOperationCancelled):
            raise exc.reason from exc
        raise
    except SessionLeaseLostError as exc:
        raise LeaseLostError from exc


def _oldest_pending_input(snapshot: Any, lane_id: LaneId) -> tuple[str, Any] | None:
    for record in snapshot.registers:
        if isinstance(record.value, PendingInput) and record.value.lane_id == lane_id:
            if not record.value.items:
                return None
            item = record.value.items[0]
            return item.idempotency_key, item.content
    return None


def _bound_child_runner(
    *,
    orchestrator: AnswerOrchestrator,
    repository: AgentSessionRepository[EffectHostUpdate],
    session: RunSession,
    fetched_buffer: FetchedResourceBuffer,
    parent_session_id: SessionId,
    persist_child_runtime: Callable[..., Awaitable[Any]],
    claim_child: Callable[..., Awaitable[Any]],
    renew_child: Callable[..., Awaitable[Any]],
) -> Callable[[SessionId, ChildRequest, str, ChildContextSnapshot], Awaitable[ChildOutcome]]:
    async def run_child(
        child_id: SessionId,
        request: ChildRequest,
        parent_call_id: str,
        context_snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        return await run_child_session(
            orchestrator=orchestrator,
            repository=repository,
            session=session,
            fetched_buffer=fetched_buffer,
            child_id=child_id,
            request=request,
            parent_call_id=parent_call_id,
            parent_session_id=parent_session_id,
            context_snapshot=context_snapshot,
            persist_child_runtime=persist_child_runtime,
            claim_child=claim_child,
            renew_child=renew_child,
        )

    return run_child


async def run_child_session(
    *,
    orchestrator: AnswerOrchestrator,
    repository: AgentSessionRepository[EffectHostUpdate],
    session: RunSession,
    fetched_buffer: FetchedResourceBuffer,
    child_id: SessionId,
    request: ChildRequest,
    parent_call_id: str,
    parent_session_id: SessionId,
    context_snapshot: ChildContextSnapshot,
    persist_child_runtime: Callable[..., Awaitable[Any]],
    claim_child: Callable[..., Awaitable[Any]],
    renew_child: Callable[..., Awaitable[Any]] | None = None,
) -> ChildOutcome:
    """Run or restore one Child through the same deep AgentSessionRuntime."""
    if context_snapshot.parent_session_id != parent_session_id:
        raise RunExecutionError(
            "run_execution_failed",
            "Child ContextSnapshot parent identity changed.",
        )
    prepared = orchestrator.prepare_child_session(
        request,
        context_snapshot=context_snapshot,
        child_session_id=child_id.value,
    )
    plan = AgentRunPlan.from_tools(
        prepared.tools,
        model_role=request.model_role,
        context_policy_revision=CONTEXT_POLICY_REVISION,
        model_identity={"role": request.model_role, "scope": "child"},
        model_profile=asdict(prepared.model_profile),
    )
    await persist_child_runtime(
        owner_id=session.owner_id,
        run_id=session.run_id,
        child_session_id=child_id.value,
        parent_session_id=parent_session_id.value,
        parent_call_id=parent_call_id,
        objective=request.objective,
        context_mode=request.context,
        model_role=request.model_role,
        tools=request.tools,
        depth=context_snapshot.depth + 1,
        context_snapshot=context_snapshot.canonical_payload(),
        plan=plan.canonical_payload(),
        budget={
            "provider_attempt_limit": plan.provider_attempt_limit,
            "compaction_attempt_limit": plan.compaction_attempt_limit,
            "model_profile": asdict(prepared.model_profile),
        },
        host_state={"inherits_parent_evidence": bool(context_snapshot.evidence_state)},
    )
    child_epoch = await claim_child(
        owner_id=session.owner_id,
        run_id=session.run_id,
        child_session_id=child_id.value,
    )
    if not isinstance(child_epoch, int):
        raise LeaseLostError
    child_repository = repository
    bind_child = getattr(repository, "for_child", None)
    if callable(bind_child):
        child_repository = cast(
            AgentSessionRepository[EffectHostUpdate],
            bind_child(child_id, fencing_epoch=child_epoch),
        )
    effects = ResearchRuntimeEffects(
        orchestrator=orchestrator,
        prepared=prepared,
        session=session,
        session_id=child_id,
        fetched_buffer=fetched_buffer,
        persist_child_intent=None,
    )
    runtime = AgentSessionRuntime(
        repository=child_repository,
        effects=effects,
        tools=prepared.tools,
        fencing_epoch=child_epoch,
        provider_attempt_limit=plan.provider_attempt_limit,
        event_sink=_answer_runtime_event_sink(session),
    )
    accepted = await runtime.accept(
        session_id=child_id,
        lane_id=LaneId.main(),
        idempotency_key=f"child-session:{child_id.value}",
        content=request.objective,
        plan=plan,
    )
    try:
        operation = await _drive_child_with_lease_renewal(
            runtime,
            session_id=child_id,
            operation_id=accepted.operation_id,
            child_fencing_epoch=child_epoch,
            renew_child=renew_child,
        )
    except (
        asyncio.CancelledError,
        RunCancellationObserved,
        AgentOperationCancelled,
    ) as exc:
        await runtime.cancel(session_id=child_id, operation_id=accepted.operation_id)
        await runtime.close(session_id=child_id, operation_id=accepted.operation_id)
        if isinstance(exc, AgentOperationCancelled):
            raise exc.reason from exc
        raise
    except SessionLeaseLostError as exc:
        raise LeaseLostError from exc
    snapshot = await child_repository.load(child_id)
    orchestrator.restore_runtime_snapshot(prepared, snapshot)
    await _restore_durable_evidence(prepared, child_repository, child_id)
    if isinstance(operation.state, OperationCompleted):
        status: Literal["succeeded", "failed", "cancelled"] = "succeeded"
    elif isinstance(operation.state, OperationCancelled):
        status = "cancelled"
    elif isinstance(operation.state, OperationFailed):
        status = "failed"
    else:
        raise RunExecutionError(
            "run_execution_failed",
            "Child Agent Runtime returned a non-terminal operation.",
        )
    summary = _child_summary(prepared, status)
    return ChildOutcome(
        status=status,
        summary=summary,
        handles=tuple(prepared.evidence.citation_handles()),
        usage=_usage_from_snapshot_entries(snapshot_entries=snapshot.entries),
        delta=_delta_from_ledger(prepared.evidence),
        child_session_id=child_id.value,
        evidence_state=prepared.evidence.durable_state(),
    )


async def _drive_child_with_lease_renewal(
    runtime: AgentSessionRuntime[EffectHostUpdate],
    *,
    session_id: SessionId,
    operation_id: OperationId,
    child_fencing_epoch: int,
    renew_child: Callable[..., Awaitable[Any]] | None,
) -> Any:
    if renew_child is None:
        return await runtime.drive(session_id=session_id, operation_id=operation_id)

    async def renew_forever() -> None:
        while True:
            await asyncio.sleep(_CHILD_LEASE_HEARTBEAT_SECONDS)
            try:
                renewed = await renew_child(
                    child_session_id=session_id.value,
                    child_fencing_epoch=child_fencing_epoch,
                )
            except LeaseLostError:
                raise
            except Exception:
                logger.warning(
                    "Child Session %s lease heartbeat failed; retrying next cadence",
                    session_id.value,
                    exc_info=True,
                )
                continue
            if renewed is not True:
                raise LeaseLostError

    drive_task = asyncio.create_task(
        runtime.drive(session_id=session_id, operation_id=operation_id)
    )
    heartbeat_task = asyncio.create_task(renew_forever())
    try:
        done, _pending = await asyncio.wait(
            (drive_task, heartbeat_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if drive_task in done:
            return await drive_task
        drive_task.cancel()
        await asyncio.gather(drive_task, return_exceptions=True)
        exception = heartbeat_task.exception()
        if exception is not None:
            raise exception
        raise LeaseLostError
    finally:
        if not drive_task.done():
            drive_task.cancel()
        heartbeat_task.cancel()
        await asyncio.gather(drive_task, heartbeat_task, return_exceptions=True)


def _child_summary(prepared: Any, status: str) -> str:
    text = prepared.last_turn.assistant.text if prepared.last_turn is not None else ""
    return text.strip() or f"Child session {status}."


def _usage_from_snapshot_entries(*, snapshot_entries: Any) -> dict[str, int] | None:
    total: dict[str, int] = {}
    for entry in snapshot_entries or ():
        usage = getattr(entry, "usage", None)
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                total[key] = total.get(key, 0) + value
    return total or None


def _delta_from_ledger(evidence: Any) -> Any:
    contexts = getattr(evidence, "contexts", {}) or {}
    return EvidenceDelta(
        new_chunks=len(contexts.get("chunks") or ()),
        new_entities=len(contexts.get("entities") or ()),
        new_relationships=len(contexts.get("relationships") or ()),
    )


def _async_store_method(store: object, name: str) -> Any | None:
    method = getattr(store, name, None)
    if inspect.iscoroutinefunction(method):
        return method
    return None


async def _durable_child_usage(
    store: object,
    *,
    owner_id: str,
    run_id: str,
) -> dict[str, int]:
    """Aggregate settled child usage so recovery matches uninterrupted execution."""
    method = _async_store_method(store, "list_child_sessions")
    if method is None:
        return {}
    rows = await method(owner_id=owner_id, run_id=run_id)
    aggregate: dict[str, int] = {}
    for row in rows or ():
        usage = row.get("usage") if isinstance(row, Mapping) else None
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                name = str(key)
                aggregate[name] = aggregate.get(name, 0) + value
    return aggregate


def _fenced_control_reader(
    store: object, session: RunSession
) -> Callable[[], Awaitable[tuple[Mapping[str, Any], ...]]] | None:
    method = _async_store_method(store, "load_pending_agent_controls")
    if method is None:
        return None

    async def read() -> tuple[Mapping[str, Any], ...]:
        controls = await method(
            owner_id=session.owner_id,
            run_id=session.run_id,
            worker_id=session.worker_id,
            fencing_epoch=session.fencing_epoch,
        )
        if controls is None:
            raise LeaseLostError
        return tuple(controls)

    return read


def _fenced_control_ack(
    store: object, session: RunSession
) -> Callable[[tuple[int, ...]], Awaitable[bool]] | None:
    method = _async_store_method(store, "acknowledge_agent_controls")
    if method is None:
        return None

    async def acknowledge(sequences: tuple[int, ...]) -> bool:
        held = await method(
            owner_id=session.owner_id,
            run_id=session.run_id,
            control_sequences=sequences,
            worker_id=session.worker_id,
            fencing_epoch=session.fencing_epoch,
        )
        return bool(held)

    return acknowledge


def _fenced_child_writer(store: object, name: str, session: RunSession) -> Any | None:
    method = _async_store_method(store, name)
    if method is None:
        return None

    async def write(**kwargs: Any) -> Any:
        held = await method(
            **kwargs,
            worker_id=session.worker_id,
            fencing_epoch=session.fencing_epoch,
        )
        if held is False:
            raise LeaseLostError
        return held

    return write


async def _restore_durable_evidence(prepared: Any, repository: Any, session_id: SessionId) -> None:
    """Restore the latest durable Evidence state into the live ledger."""
    loader = getattr(repository, "load_evidence", None)
    if loader is None or prepared is None:
        return
    writes = await loader(session_id)
    if not writes:
        return
    import json as _json

    latest = writes[-1]
    prepared.evidence.restore_ledger_state(_json.loads(latest.content.decode("utf-8")))
