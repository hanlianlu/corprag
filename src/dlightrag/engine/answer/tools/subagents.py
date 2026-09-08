# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""First-class foreground child Agent tools.

A spawn call may launch one or many ordinary child Agent Sessions in parallel,
but it does not return until every child has settled. There is no detached
scheduler: host task references exist only so concurrent status/wait/cancel
calls can address foreground work in the same parent run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlightrag.engine.agent.session.effects import canonical_json
from dlightrag.engine.agent.session.ids import EntryId, IntentId, SessionId
from dlightrag.engine.agent.tools import AgentTool, ToolResult, ToolRuntime
from dlightrag.engine.answer.evidence import EvidenceDelta
from dlightrag.engine.runtime.coordinator import RunCancellationObserved
from dlightrag.engine.runtime.errors import RunCancelledError

type ChildStatus = Literal["running", "succeeded", "failed", "cancelled"]
type ChildContextMode = Literal["isolated", "parent"]
type ChildModelRole = Literal["query", "extract"]


class _ParentRunCancelled(asyncio.CancelledError):
    """A cooperative parent cancellation crossing the Tool execution seam."""


class ChildRequest(BaseModel):
    """One bounded child invocation selected by the parent model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    objective: str = Field(min_length=1, description="One concrete child objective.")
    context: ChildContextMode = Field(
        default="isolated",
        description="isolated starts from the objective; parent also receives parent context.",
    )
    model_role: ChildModelRole = Field(
        default="query", description="Configured tool-capable model role for the child."
    )
    tools: tuple[str, ...] | None = Field(
        default=None,
        description="Optional inherited tool-name subset; spawn tools are always removed.",
    )


class SpawnAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    children: tuple[ChildRequest, ...] = Field(
        min_length=1,
        max_length=8,
        description="One or more foreground child requests, run in parallel when possible.",
    )

    @model_validator(mode="after")
    def _unique_objectives(self) -> SpawnAgentInput:
        if len({child.objective for child in self.children}) != len(self.children):
            raise ValueError("child objectives must be unique within one spawn call")
        return self


class ChildControlInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    child_session_id: str = Field(min_length=1, description="Child session id from spawn_agent.")


@dataclass(frozen=True, slots=True)
class ChildContextSnapshot:
    """Bounded immutable parent context explicitly handed to one Child Session."""

    parent_session_id: SessionId
    parent_entry_id: EntryId
    depth: int
    messages_json: str
    evidence_state_json: str = "{}"

    def __post_init__(self) -> None:
        import json

        if self.depth < 0:
            raise ValueError("Child context depth cannot be negative")
        if not isinstance(json.loads(self.messages_json), list):
            raise ValueError("Child context messages must be an array")
        if not isinstance(json.loads(self.evidence_state_json), dict):
            raise ValueError("Child context evidence state must be an object")

    @classmethod
    def from_values(
        cls,
        *,
        parent_session_id: SessionId,
        parent_entry_id: EntryId,
        depth: int,
        messages: list[dict[str, Any]],
        evidence_state: Mapping[str, Any] | None = None,
    ) -> ChildContextSnapshot:
        return cls(
            parent_session_id=parent_session_id,
            parent_entry_id=parent_entry_id,
            depth=depth,
            messages_json=canonical_json(messages),
            evidence_state_json=canonical_json(dict(evidence_state or {})),
        )

    @property
    def messages(self) -> list[dict[str, Any]]:
        import json

        return json.loads(self.messages_json)

    @property
    def evidence_state(self) -> dict[str, Any]:
        import json

        return json.loads(self.evidence_state_json)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "parent_session_id": self.parent_session_id.value,
            "parent_entry_id": self.parent_entry_id.value,
            "depth": self.depth,
            "messages": self.messages,
            "evidence_state": self.evidence_state,
        }


@dataclass(frozen=True, slots=True)
class ChildOutcome:
    """Distilled Child result returned to the parent ToolResult."""

    status: ChildStatus
    summary: str
    handles: tuple[str, ...] = ()
    usage: Mapping[str, int] | None = None
    delta: EvidenceDelta | None = None
    child_session_id: str = ""
    evidence_state: Mapping[str, Any] | None = None

    def durable_payload(self) -> dict[str, Any]:
        """Return the exact parent-visible outcome needed for replay."""
        return {
            "status": self.status,
            "summary": self.summary,
            "handles": list(self.handles),
            "usage": dict(self.usage or {}),
            "child_session_id": self.child_session_id,
            "evidence_state": (
                dict(self.evidence_state) if self.evidence_state is not None else None
            ),
        }

    @classmethod
    def from_durable_payload(cls, payload: Mapping[str, Any]) -> ChildOutcome:
        status = payload.get("status")
        if status not in {"running", "succeeded", "failed", "cancelled"}:
            raise ValueError("persisted Child outcome has an invalid status")
        handles = payload.get("handles")
        usage = payload.get("usage")
        evidence_state = payload.get("evidence_state")
        if not isinstance(handles, list) or not all(isinstance(item, str) for item in handles):
            raise ValueError("persisted Child outcome has invalid handles")
        if not isinstance(usage, Mapping) or not all(
            isinstance(key, str) and isinstance(value, int) for key, value in usage.items()
        ):
            raise ValueError("persisted Child outcome has invalid usage")
        if "evidence_state" not in payload or (
            evidence_state is not None and not isinstance(evidence_state, Mapping)
        ):
            raise ValueError("persisted Child outcome has invalid evidence state")
        return cls(
            status=cast(ChildStatus, status),
            summary=str(payload.get("summary") or ""),
            handles=tuple(handles),
            usage={str(key): int(value) for key, value in usage.items()},
            child_session_id=str(payload.get("child_session_id") or ""),
            evidence_state=(dict(evidence_state) if isinstance(evidence_state, Mapping) else None),
        )


@dataclass
class SubagentHost:
    """Late-bound lineage, persistence, and foreground-task state."""

    parent_session_id: SessionId | None = None
    run_id: str = ""
    owner_id: str = ""
    max_concurrency: int = 4
    check_cancelled: Callable[[], Awaitable[None]] | None = None
    persist: Callable[..., Awaitable[Any]] | None = None
    load_child: Callable[..., Awaitable[Any]] | None = None
    finish_child: Callable[..., Awaitable[Any]] | None = None
    run_child: (
        Callable[[SessionId, ChildRequest, str, ChildContextSnapshot], Awaitable[ChildOutcome]]
        | None
    ) = None
    context_snapshot: ChildContextSnapshot | None = None
    depth: int = 0
    merge_evidence: Callable[[Mapping[str, Any], str, str], tuple[str, ...]] | None = None
    record_usage: Callable[[Mapping[str, int]], None] | None = None
    tasks: dict[str, asyncio.Task[ChildOutcome]] = field(default_factory=dict)
    outcomes: dict[str, ChildOutcome] = field(default_factory=dict)


def subagent_tools(*, host: SubagentHost) -> tuple[AgentTool, ...]:
    """Return spawn/status/wait/cancel tools over one foreground roster."""

    async def spawn(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(SpawnAgentInput, raw)
        return await _spawn(host, args, runtime)

    async def status(raw: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        args = cast(ChildControlInput, raw)
        await _check_cancelled(host)
        outcome = await _status(host, args.child_session_id)
        return _single_result(outcome)

    async def wait(raw: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        args = cast(ChildControlInput, raw)
        await _check_cancelled(host)
        task = host.tasks.get(args.child_session_id)
        if task is not None:
            outcome = await asyncio.shield(task)
        else:
            outcome = await _status(host, args.child_session_id)
        return _single_result(outcome)

    async def cancel(raw: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        args = cast(ChildControlInput, raw)
        await _check_cancelled(host)
        task = host.tasks.get(args.child_session_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            outcome = ChildOutcome(
                status="cancelled",
                summary="Child session cancelled.",
                child_session_id=args.child_session_id,
            )
            host.outcomes[args.child_session_id] = outcome
        else:
            outcome = await _status(host, args.child_session_id)
        return _single_result(outcome)

    return (
        AgentTool(
            "spawn_agent",
            "Run one or many foreground child Agent Sessions and wait for all results.",
            SpawnAgentInput,
            spawn,
            replay_policy="replayable",
        ),
        AgentTool(
            "subagent_status",
            "Read one foreground or completed child session status.",
            ChildControlInput,
            status,
            replay_policy="replayable",
        ),
        AgentTool(
            "wait_subagent",
            "Wait for one known foreground child session.",
            ChildControlInput,
            wait,
            replay_policy="replayable",
        ),
        AgentTool(
            "cancel_subagent",
            "Cancel one known foreground child session.",
            ChildControlInput,
            cancel,
            replay_policy="never",
        ),
    )


async def _spawn(
    host: SubagentHost,
    args: SpawnAgentInput,
    runtime: ToolRuntime,
) -> ToolResult:
    if host.parent_session_id is None or not host.run_id:
        raise RuntimeError("spawn_agent is not bound to a parent session")
    if host.run_child is None:
        raise RuntimeError("spawn_agent has no child runner")
    await _check_cancelled(host)
    parent_session_id = host.parent_session_id
    run_child = host.run_child
    call_id = runtime.call_id
    context_snapshot = host.context_snapshot
    if context_snapshot is None:
        raise RuntimeError("spawn_agent has no explicit parent ContextSnapshot")
    semaphore = asyncio.Semaphore(max(1, host.max_concurrency))
    child_ids = [
        child_session_id(
            run_id=host.run_id,
            parent_session_id=parent_session_id,
            parent_intent_id=runtime.intent_id,
            position=position,
        )
        for position in range(len(args.children))
    ]

    async def run_one(child_id: SessionId, request: ChildRequest) -> ChildOutcome:
        await _check_cancelled(host)
        persisted = await _load_terminal_child(host, child_id.value)
        if persisted is not None:
            if persisted.evidence_state is not None and host.merge_evidence is not None:
                host.merge_evidence(persisted.evidence_state, child_id.value, call_id)
            if persisted.usage is not None and host.record_usage is not None:
                host.record_usage(persisted.usage)
            host.outcomes[child_id.value] = persisted
            return persisted
        if host.persist is not None:
            await host.persist(
                owner_id=host.owner_id,
                run_id=host.run_id,
                child_session_id=child_id.value,
                parent_session_id=parent_session_id.value,
                parent_call_id=call_id,
                objective=request.objective,
                context_mode=request.context,
                model_role=request.model_role,
                tools=request.tools,
                depth=context_snapshot.depth + 1,
                context_snapshot=context_snapshot.canonical_payload(),
            )
        try:
            async with semaphore:
                await _check_cancelled(host)
                outcome = await run_child(child_id, request, call_id, context_snapshot)
        except (RunCancellationObserved, RunCancelledError) as exc:
            await _finish_cancelled_child(host, child_id.value)
            raise _ParentRunCancelled from exc
        except _ParentRunCancelled:
            await _finish_cancelled_child(host, child_id.value)
            raise
        except asyncio.CancelledError:
            return await _finish_cancelled_child(host, child_id.value)
        if outcome.evidence_state is not None and host.merge_evidence is not None:
            outcome = replace(
                outcome,
                handles=host.merge_evidence(outcome.evidence_state, child_id.value, call_id),
            )
        if outcome.usage is not None and host.record_usage is not None:
            host.record_usage(outcome.usage)
        if host.finish_child is not None:
            await host.finish_child(
                owner_id=host.owner_id,
                run_id=host.run_id,
                child_session_id=child_id.value,
                status=outcome.status,
                summary=outcome.summary,
                usage=outcome.usage,
                outcome=outcome.durable_payload(),
            )
        host.outcomes[child_id.value] = outcome
        return outcome

    tasks = [
        asyncio.create_task(
            run_one(child_id, request),
            name=f"agent-child:{child_id.value}",
        )
        for child_id, request in zip(child_ids, args.children, strict=True)
    ]
    host.tasks.update(
        (child_id.value, task) for child_id, task in zip(child_ids, tasks, strict=True)
    )
    try:
        outcomes = await asyncio.gather(*tasks)
    finally:
        for child_id in child_ids:
            task = host.tasks.pop(child_id.value, None)
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return _many_result(tuple(outcomes))


async def _check_cancelled(host: SubagentHost) -> None:
    if host.check_cancelled is None:
        return
    try:
        await host.check_cancelled()
    except (RunCancellationObserved, RunCancelledError) as exc:
        raise _ParentRunCancelled from exc


async def _finish_cancelled_child(host: SubagentHost, child_id: str) -> ChildOutcome:
    cancelled = ChildOutcome(
        status="cancelled",
        summary="Child session cancelled.",
        child_session_id=child_id,
    )
    host.outcomes[child_id] = cancelled
    if host.finish_child is not None:
        await host.finish_child(
            owner_id=host.owner_id,
            run_id=host.run_id,
            child_session_id=child_id,
            status="cancelled",
            summary=cancelled.summary,
            usage=None,
            outcome=cancelled.durable_payload(),
        )
    return cancelled


async def _load_terminal_child(host: SubagentHost, child_id: str) -> ChildOutcome | None:
    if host.load_child is None:
        return None
    row = await host.load_child(
        owner_id=host.owner_id,
        run_id=host.run_id,
        child_session_id=child_id,
    )
    if row is None or str(row.get("status") or "running") == "running":
        return None
    return _terminal_outcome_from_row(row, child_id)


async def _status(host: SubagentHost, child_id: str) -> ChildOutcome:
    task = host.tasks.get(child_id)
    if task is not None:
        if not task.done():
            return ChildOutcome(
                status="running",
                summary="Child session is running.",
                child_session_id=child_id,
            )
        return task.result()
    if child_id in host.outcomes:
        return host.outcomes[child_id]
    if host.load_child is not None:
        row = await host.load_child(
            owner_id=host.owner_id,
            run_id=host.run_id,
            child_session_id=child_id,
        )
        if row is not None:
            status = str(row.get("status") or "failed")
            if status != "running":
                return _terminal_outcome_from_row(row, child_id)
            return ChildOutcome(
                status="running",
                summary=str(row.get("summary") or ""),
                child_session_id=child_id,
            )
    return ChildOutcome(
        status="failed",
        summary="Unknown child session.",
        child_session_id=child_id,
    )


def _terminal_outcome_from_row(row: Mapping[str, Any], child_id: str) -> ChildOutcome:
    host_state = row.get("host_state")
    payload = host_state.get("terminal_outcome") if isinstance(host_state, Mapping) else None
    if not isinstance(payload, Mapping):
        raise RuntimeError("terminal Child session lost its durable outcome")
    outcome = ChildOutcome.from_durable_payload(payload)
    if outcome.child_session_id != child_id or outcome.status != str(row.get("status")):
        raise RuntimeError("terminal Child session outcome identity changed")
    return outcome


def child_session_id(
    *,
    run_id: str,
    parent_session_id: SessionId,
    parent_intent_id: IntentId,
    position: int = 0,
) -> SessionId:
    """Deterministic child identity owned by a durable parent Effect intent."""
    suffix = f":{position}" if position else ""
    return SessionId.deterministic(
        run_id=run_id,
        name=f"child:{parent_session_id.value}:{parent_intent_id.value}{suffix}",
    )


def _single_result(outcome: ChildOutcome) -> ToolResult:
    return _many_result((outcome,))


def _many_result(outcomes: tuple[ChildOutcome, ...]) -> ToolResult:
    lines: list[str] = []
    children: list[dict[str, Any]] = []
    inclusive_usage: dict[str, int] = {}
    for outcome in outcomes:
        lines.append(
            f"Child {outcome.child_session_id} [{outcome.status}]: "
            f"{outcome.summary.strip() or '(no summary)'}"
        )
        if outcome.handles:
            lines.extend(f"- merged {item}" for item in outcome.handles)
        if outcome.usage:
            for key, value in outcome.usage.items():
                inclusive_usage[key] = inclusive_usage.get(key, 0) + int(value)
        children.append(
            {
                "child_session_id": outcome.child_session_id,
                "status": outcome.status,
                "evidence_handles": list(outcome.handles),
                "usage": dict(outcome.usage or {}),
            }
        )
    return ToolResult.text(
        "\n".join(lines),
        details={"children": children, "inclusive_usage": inclusive_usage},
    )


__all__ = [
    "ChildContextMode",
    "ChildContextSnapshot",
    "ChildControlInput",
    "ChildModelRole",
    "ChildOutcome",
    "ChildRequest",
    "ChildStatus",
    "SpawnAgentInput",
    "SubagentHost",
    "child_session_id",
    "subagent_tools",
]
