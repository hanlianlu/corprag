# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Foreground subagent composition, controls, replay, and durable children."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.engine.agent.session.fold import PriorTurns, WorkingContextProjection
from dlightrag.engine.agent.session.ids import EntryId, IntentId, SessionId
from dlightrag.engine.ai.capacity import CONTEXT_POLICY
from dlightrag.engine.ai.messages import AssistantTurn
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.orchestration import AnswerOrchestrator
from dlightrag.engine.answer.research.runtime import (
    FetchedResourceBuffer,
    run_child_session,
)
from dlightrag.engine.answer.resources.models import TextWindowBudget
from dlightrag.engine.answer.synthesizer import AnswerSynthesizer
from dlightrag.engine.answer.tools.composition import compose_research_tools
from dlightrag.engine.answer.tools.subagents import (
    ChildContextSnapshot,
    ChildControlInput,
    ChildOutcome,
    ChildRequest,
    SpawnAgentInput,
    SubagentHost,
    child_session_id,
    subagent_tools,
)
from dlightrag.engine.runtime.coordinator import RunCancellationObserved
from tests.in_memory_session_repository import InMemoryAgentSessionRepository
from tests.tool_helpers import tool_runtime
from tests.unit.conftest import answer_image_policy, answer_model_profile


def _spawn_input(objective: str) -> SpawnAgentInput:
    return SpawnAgentInput(children=(ChildRequest(objective=objective),))


def _context_snapshot(parent_id: SessionId | None = None) -> ChildContextSnapshot:
    return ChildContextSnapshot.from_values(
        parent_session_id=parent_id or SessionId.new(),
        parent_entry_id=EntryId.new(),
        depth=0,
        messages=[{"role": "user", "content": "parent question"}],
    )


async def _retrieve(_query: str) -> object:
    raise RuntimeError("unused")


def test_child_identity_uses_durable_intent_not_provider_call_id() -> None:
    run_id = SessionId.new().value
    parent_id = SessionId.new()
    first = child_session_id(
        run_id=run_id,
        parent_session_id=parent_id,
        parent_intent_id=IntentId.new(),
    )
    second = child_session_id(
        run_id=run_id,
        parent_session_id=parent_id,
        parent_intent_id=IntentId.new(),
    )
    assert first != second


def test_subagent_cancel_is_never_replayed_without_durable_reconciliation() -> None:
    tools = {tool.name: tool for tool in subagent_tools(host=SubagentHost())}
    assert tools["cancel_subagent"].replay_policy == "never"


def test_child_outcome_durable_payload_round_trips_evidence_state() -> None:
    evidence_state = {
        "contexts": {
            "chunks": [{"chunk_id": "c1", "content": "finding"}],
            "entities": [],
            "relationships": [],
        }
    }
    outcome = ChildOutcome(
        status="succeeded",
        summary="finding",
        handles=("[1] report.pdf",),
        usage={"input_tokens": 8},
        child_session_id="child-1",
        evidence_state=evidence_state,
    )

    assert ChildOutcome.from_durable_payload(outcome.durable_payload()) == outcome
    invalid = outcome.durable_payload()
    invalid["evidence_state"] = []
    with pytest.raises(ValueError, match="evidence state"):
        ChildOutcome.from_durable_payload(invalid)


def test_parent_tools_include_spawn_and_child_omits_it() -> None:
    host = SubagentHost()
    parent = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        subagent_host=host,
    )
    child = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        subagent_host=host,
        child=True,
    )
    controls = {"subagent_status", "wait_subagent", "cancel_subagent"}
    assert {"spawn_agent", *controls} <= {tool.name for tool in parent}
    assert {"search_knowledge_base"} == {tool.name for tool in child}
    assert not ({"spawn_agent"} | controls) & {tool.name for tool in child}


async def test_spawn_many_runs_in_parallel_and_aggregates_usage() -> None:
    started = 0
    both_started = asyncio.Event()

    async def run_child(
        child_id: SessionId,
        request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        return ChildOutcome(
            status="succeeded",
            summary=request.objective,
            usage={"input_tokens": 2},
            child_session_id=child_id.value,
        )

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=SessionId.new().value,
        run_child=run_child,
        context_snapshot=_context_snapshot(),
    )
    result = await subagent_tools(host=host)[0].execute(
        SpawnAgentInput(
            children=(
                ChildRequest(objective="one"),
                ChildRequest(objective="two"),
            )
        ),
        tool_runtime(call_id="parallel", tool_name="spawn_agent"),
    )

    assert result.details is not None
    assert len(result.details["children"]) == 2
    assert result.details["inclusive_usage"] == {"input_tokens": 4}
    assert not host.tasks


async def test_spawn_checks_parent_cancellation_before_starting_children() -> None:
    cancelled = AsyncMock(side_effect=RunCancellationObserved)
    runner = AsyncMock()
    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=SessionId.new().value,
        check_cancelled=cancelled,
        run_child=runner,
    )

    with pytest.raises(asyncio.CancelledError):
        await subagent_tools(host=host)[0].execute(
            _spawn_input("must not start"), tool_runtime(tool_name="spawn_agent")
        )

    cancelled.assert_awaited()
    runner.assert_not_awaited()


async def test_spawn_propagates_parent_cancel_and_finishes_persisted_child() -> None:
    finish = AsyncMock()

    async def run_child(
        _child_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        raise RunCancellationObserved

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=SessionId.new().value,
        persist=AsyncMock(),
        finish_child=finish,
        run_child=run_child,
        context_snapshot=_context_snapshot(),
    )

    with pytest.raises(asyncio.CancelledError):
        await subagent_tools(host=host)[0].execute(
            _spawn_input("cancel in flight"), tool_runtime(tool_name="spawn_agent")
        )

    assert finish.await_args is not None
    assert finish.await_args.kwargs["status"] == "cancelled"
    assert not host.tasks


async def test_cancel_tool_joins_a_known_foreground_child() -> None:
    started = asyncio.Event()

    async def sleeper() -> ChildOutcome:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    task = asyncio.create_task(sleeper())
    await started.wait()
    host = SubagentHost(tasks={"child-1": task})

    result = await subagent_tools(host=host)[3].execute(
        ChildControlInput(child_session_id="child-1"),
        tool_runtime(tool_name="cancel_subagent"),
    )

    assert task.cancelled()
    assert "[cancelled]" in result.text_content
    assert host.outcomes["child-1"].status == "cancelled"


async def test_terminal_persisted_spawn_replay_never_reenters_child_execution() -> None:
    persist = AsyncMock()
    finish = AsyncMock()
    run_child = AsyncMock()
    ledger = EvidenceLedger()
    evidence_state = {
        "contexts": {
            "chunks": [{"chunk_id": "c1", "content": "persisted finding"}],
            "entities": [],
            "relationships": [],
        }
    }

    def remerge_evidence(state: Any, child_id: str, call_id: str) -> tuple[str, ...]:
        before = len(ledger.contexts["chunks"])
        ledger.merge_child_state(
            state,
            child_session_id=child_id,
            parent_call_id=call_id,
        )
        return tuple(ledger.citation_handles(after_chunk_count=before))

    merge_evidence = MagicMock(side_effect=remerge_evidence)

    async def load_child(**kwargs: Any) -> dict[str, Any]:
        child_id = kwargs["child_session_id"]
        return {
            "status": "succeeded",
            "summary": "Persisted child finding.",
            "host_state": {
                "terminal_outcome": {
                    "status": "succeeded",
                    "summary": "Persisted child finding.",
                    "handles": ["[1] report.pdf"],
                    "usage": {"input_tokens": 8},
                    "child_session_id": child_id,
                    "evidence_state": evidence_state,
                }
            },
        }

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=str(SessionId.new().value),
        owner_id="owner",
        load_child=load_child,
        persist=persist,
        finish_child=finish,
        run_child=run_child,
        context_snapshot=_context_snapshot(),
        merge_evidence=merge_evidence,
    )
    tool = subagent_tools(host=host)[0]
    runtime = tool_runtime(call_id="call-1", tool_name="spawn_agent")
    result = await tool.execute(_spawn_input("what happened?"), runtime)
    replayed = await tool.execute(_spawn_input("what happened?"), runtime)

    assert "Persisted child finding." in result.text_content
    assert "[1] report.pdf" in result.text_content
    assert result.details is not None
    assert result.details["inclusive_usage"] == {"input_tokens": 8}
    assert result.details["children"][0]["evidence_handles"] == ["[1] report.pdf"]
    assert replayed.details is not None
    assert replayed.details["children"][0]["evidence_handles"] == ["[1] report.pdf"]
    assert len(ledger.contexts["chunks"]) == 1
    assert merge_evidence.call_count == 2
    assert all(
        call.args
        == (
            evidence_state,
            result.details["children"][0]["child_session_id"],
            "call-1",
        )
        for call in merge_evidence.call_args_list
    )
    persist.assert_not_awaited()
    finish.assert_not_awaited()
    run_child.assert_not_awaited()


async def test_spawn_reports_child_outcome_and_usage() -> None:
    persist = AsyncMock()
    finish = AsyncMock()

    async def run_child(
        _child_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        return ChildOutcome(
            status="succeeded",
            summary="Child summary.",
            handles=("[1] Page A [resource: res-a]",),
            usage={"input_tokens": 12, "output_tokens": 4},
            child_session_id="child",
        )

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=str(SessionId.new().value),
        owner_id="owner",
        persist=persist,
        load_child=AsyncMock(return_value=None),
        finish_child=finish,
        run_child=run_child,
        context_snapshot=_context_snapshot(),
    )
    tool = subagent_tools(host=host)[0]
    result = await tool.execute(
        _spawn_input("summarize filings"),
        tool_runtime(call_id="call-9", tool_name="spawn_agent"),
    )
    assert "Child summary." in result.text_content
    assert "[1] Page A [resource: res-a]" in result.text_content
    assert result.details is not None
    assert result.details["inclusive_usage"] == {"input_tokens": 12, "output_tokens": 4}
    assert result.details["children"][0]["status"] == "succeeded"
    persist.assert_awaited()
    finish.assert_awaited()
    assert finish.call_args is not None
    assert finish.call_args.kwargs["status"] == "succeeded"


async def test_spawn_adopts_child_evidence_before_returning_result() -> None:
    adopted = MagicMock(return_value=("[2] child source",))
    child_state = {
        "contexts": {
            "chunks": [{"chunk_id": "c1", "content": "finding"}],
            "entities": [],
            "relationships": [],
        }
    }

    async def run_child(
        _child_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        return ChildOutcome(
            status="succeeded",
            summary="Child summary.",
            child_session_id="child",
            evidence_state=child_state,
        )

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=str(SessionId.new().value),
        owner_id="owner",
        run_child=run_child,
        context_snapshot=_context_snapshot(),
        merge_evidence=adopted,
    )
    tool = subagent_tools(host=host)[0]
    result = await tool.execute(
        _spawn_input("find source"),
        tool_runtime(call_id="call-adopt", tool_name="spawn_agent"),
    )

    assert "[2] child source" in result.text_content
    adopted.assert_called_once()
    assert adopted.call_args.args[0] == child_state
    assert adopted.call_args.args[2] == "call-adopt"


async def test_failed_child_is_recorded_failed() -> None:
    finish = AsyncMock()

    async def run_child(
        _child_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        return ChildOutcome(status="failed", summary="provider down", child_session_id="child")

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=str(SessionId.new().value),
        owner_id="owner",
        persist=AsyncMock(),
        load_child=AsyncMock(return_value=None),
        finish_child=finish,
        run_child=run_child,
        context_snapshot=_context_snapshot(),
    )
    tool = subagent_tools(host=host)[0]
    result = await tool.execute(
        _spawn_input("x"),
        tool_runtime(call_id="call-err", tool_name="spawn_agent"),
    )
    assert result.details is not None
    assert result.details["children"][0]["status"] == "failed"
    assert finish.call_args is not None
    assert finish.call_args.kwargs["status"] == "failed"


@dataclass
class _FakeSession:
    run_id: str
    owner_id: str = "owner"
    execution: Any = field(default_factory=lambda: SimpleNamespace(fencing_epoch=1))

    async def check_cancelled(self) -> None:
        return None

    async def enter_phase(self, _phase: str) -> None:
        return None

    async def emit_tool_event(self, _event_type: str, _payload: object) -> None:
        return None


def _child_orchestrator(
    model_func: Any,
    *,
    environment: Any = None,
    retrieve_func: Any = None,
) -> AnswerOrchestrator:
    profile = answer_model_profile()

    async def retrieve(_query: str) -> Any:
        raise RuntimeError("child should not search in this test")

    return AnswerOrchestrator(
        synthesizer=AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=profile,
        ),
        retrieve_knowledge_base=retrieve_func or retrieve,
        search_web=None,
        model_func=model_func,
        telemetry=NOOP_TELEMETRY,
        model_profile=profile,
        text_window_budget=TextWindowBudget(CONTEXT_POLICY.hard_input_limit(profile)),
        subagent_host=SubagentHost(),
        resolved_mode="research",
        environment=environment,
    )


async def test_child_session_persists_and_replays_without_rerun() -> None:
    calls = {"n": 0}

    async def model(**_kwargs: object) -> AssistantTurn:
        calls["n"] += 1
        return AssistantTurn(
            text="Persisted child summary.",
            tool_calls=(),
            stop_reason="stop",
            usage_details={"input_tokens": 3, "output_tokens": 2},
        )

    orchestrator = _child_orchestrator(model)
    repository = InMemoryAgentSessionRepository()
    parent_id = SessionId.new()
    child_id = SessionId.deterministic(run_id=str(parent_id.value), name="child:test:1")
    session = _FakeSession(run_id=str(parent_id.value))

    first = await run_child_session(
        orchestrator=orchestrator,
        repository=repository,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child_id,
        request=ChildRequest(objective="summarize filings"),
        parent_call_id="call-1",
        parent_session_id=parent_id,
        context_snapshot=_context_snapshot(parent_id),
        persist_child_runtime=AsyncMock(),
        claim_child=AsyncMock(return_value=1),
    )
    assert first.status == "succeeded"
    assert first.summary == "Persisted child summary."
    assert first.usage == {"input_tokens": 3, "output_tokens": 2}
    assert calls["n"] == 1
    snapshot = await repository.load(child_id)
    assert snapshot.commit_sequence >= 2
    assert [entry.entry_type for entry in snapshot.entries] == [
        "user_message",
        "assistant_message",
    ]
    assert any(record.ref.kind == "operation_state" for record in snapshot.registers)

    second = await run_child_session(
        orchestrator=orchestrator,
        repository=repository,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child_id,
        request=ChildRequest(objective="summarize filings"),
        parent_call_id="call-1",
        parent_session_id=parent_id,
        context_snapshot=_context_snapshot(parent_id),
        persist_child_runtime=AsyncMock(),
        claim_child=AsyncMock(return_value=1),
    )
    assert second.summary == first.summary
    assert second.usage == first.usage
    assert second.handles == first.handles
    assert second.status == first.status
    assert calls["n"] == 1


async def test_child_renews_its_lease_while_a_provider_call_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dlightrag.engine.answer.research.runtime._CHILD_LEASE_HEARTBEAT_SECONDS", 0.005
    )

    async def model(**_kwargs: object) -> AssistantTurn:
        await asyncio.sleep(0.03)
        return AssistantTurn(text="renewed child", tool_calls=(), stop_reason="stop")

    parent_id = SessionId.new()
    child_id = SessionId.deterministic(run_id=str(parent_id.value), name="child:renew")
    renew_child = AsyncMock(return_value=True)
    outcome = await run_child_session(
        orchestrator=_child_orchestrator(model),
        repository=InMemoryAgentSessionRepository(),  # type: ignore[arg-type]
        session=_FakeSession(run_id=str(parent_id.value)),  # type: ignore[arg-type]
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child_id,
        request=ChildRequest(objective="wait for a slow provider"),
        parent_call_id="call-renew",
        parent_session_id=parent_id,
        context_snapshot=_context_snapshot(parent_id),
        persist_child_runtime=AsyncMock(),
        claim_child=AsyncMock(return_value=1),
        renew_child=renew_child,
    )

    assert outcome.status == "succeeded"
    assert renew_child.await_count >= 1
    assert renew_child.await_args is not None
    assert renew_child.await_args.kwargs == {
        "child_session_id": child_id.value,
        "child_fencing_epoch": 1,
    }


async def test_child_selects_parent_context_and_an_inherited_tool_subset() -> None:
    async def model(**_kwargs: object) -> AssistantTurn:
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    orchestrator = _child_orchestrator(model)
    orchestrator.prepare_run(
        "parent question",
        conversation_history=PriorTurns(
            [
                {"role": "user", "content": "older question"},
                {"role": "assistant", "content": "older answer"},
            ]
        ),
    )
    child = orchestrator.prepare_child_session(
        ChildRequest(
            objective="focused",
            context="parent",
            tools=("search_knowledge_base",),
        ),
        context_snapshot=ChildContextSnapshot.from_values(
            parent_session_id=SessionId.new(),
            parent_entry_id=EntryId.new(),
            depth=0,
            messages=[
                {"role": "user", "content": "older question"},
                {"role": "assistant", "content": "older answer"},
                {"role": "user", "content": "parent question"},
            ],
        ),
    )

    messages = await child.context.control_turn(
        evidence=child.evidence,
        working=WorkingContextProjection(retained_tail_tokens=1000),
        tool_schema_tokens=0,
    )

    assert [tool.name for tool in child.tools] == ["search_knowledge_base"]
    assert any(message.get("content") == "older answer" for message in messages)
    assert any(message.get("content") == "parent question" for message in messages)


def test_child_inherits_parent_path_tools_except_spawn() -> None:
    child = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        environment=MagicMock(),
        artifacts_root=Path("/unused/artifacts"),
        child=True,
    )
    names = {tool.name for tool in child}
    assert names >= {"search_knowledge_base", "read", "write", "edit", "grep", "bash"}
    assert "spawn_agent" not in names
    assert "attach_artifact" not in names


async def test_cancelled_child_closes_pending_intent_before_terminal() -> None:
    from dlightrag.engine.agent.session.entries import ToolResultMessageEntry
    from dlightrag.engine.agent.session.operation import OperationCancelled
    from dlightrag.engine.agent.session.registers import OperationStateRegister
    from dlightrag.engine.ai.messages import ToolCall

    async def model(**_kwargs: object) -> AssistantTurn:
        return AssistantTurn(
            text="",
            tool_calls=(
                ToolCall(
                    id="search-cancelled",
                    name="search_knowledge_base",
                    arguments={"query": "cancel me"},
                ),
            ),
            stop_reason="tool_use",
        )

    async def cancel_during_search(_query: str) -> Any:
        raise asyncio.CancelledError

    parent_id = SessionId.new()
    child_id = SessionId.deterministic(run_id=str(parent_id.value), name="pending-cancel")
    repository = InMemoryAgentSessionRepository()
    with pytest.raises(asyncio.CancelledError):
        await run_child_session(
            orchestrator=_child_orchestrator(model, retrieve_func=cancel_during_search),
            repository=repository,  # type: ignore[arg-type]
            session=_FakeSession(run_id=str(parent_id.value)),  # type: ignore[arg-type]
            fetched_buffer=FetchedResourceBuffer(),
            child_id=child_id,
            request=ChildRequest(objective="cancel while searching"),
            parent_call_id="call-pending",
            parent_session_id=parent_id,
            context_snapshot=_context_snapshot(parent_id),
            persist_child_runtime=AsyncMock(),
            claim_child=AsyncMock(return_value=1),
        )

    snapshot = await repository.load(child_id)
    result = next(entry for entry in snapshot.entries if isinstance(entry, ToolResultMessageEntry))
    assert result.result.call_id == "search-cancelled"
    assert result.result.outcome == "outcome_unknown"
    state = next(
        record.value.state
        for record in snapshot.registers
        if isinstance(record.value, OperationStateRegister)
    )
    assert isinstance(state, OperationCancelled)


async def test_parent_cancel_marks_the_child_cancelled() -> None:
    async def model(**_kwargs: object) -> AssistantTurn:
        raise AssertionError("cancelled child must not call the model")

    class _CancelSession(_FakeSession):
        async def check_cancelled(self) -> None:
            raise RunCancellationObserved

    parent_id = SessionId.new()
    with pytest.raises(RunCancellationObserved):
        await run_child_session(
            orchestrator=_child_orchestrator(model),
            repository=InMemoryAgentSessionRepository(),  # type: ignore[arg-type]
            session=_CancelSession(run_id=str(parent_id.value)),  # type: ignore[arg-type]
            fetched_buffer=FetchedResourceBuffer(),
            child_id=SessionId.deterministic(run_id=str(parent_id.value), name="c"),
            request=ChildRequest(objective="stop"),
            parent_call_id="call-c",
            parent_session_id=parent_id,
            context_snapshot=_context_snapshot(parent_id),
            persist_child_runtime=AsyncMock(),
            claim_child=AsyncMock(return_value=1),
        )
