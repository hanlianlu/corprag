# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for how one research request is assembled from its memory."""

from typing import Any

import pytest

from dlightrag.engine.agent.session.fold import PriorTurns, WorkingContextProjection
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ModelProfile
from dlightrag.engine.ai.tokens import estimate_messages_tokens
from dlightrag.engine.answer.errors import AnswerInputOverflowError
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.execution import research_history_input_measure
from dlightrag.engine.answer.memory import reserved_auto_recall_text
from dlightrag.engine.answer.prompts import control_turn_instruction
from dlightrag.engine.answer.research.context import ContextAssembler
from dlightrag.engine.answer.resources.models import ResourceManifestEntry

_WINDOW = 80_000
_RETAINED_TAIL = 13_600
_CONTROL_TURN_INSTRUCTION = control_turn_instruction()


def _assembler(history: list[dict[str, Any]]) -> ContextAssembler:
    return ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW),
        query="What changed?",
        history=PriorTurns(history),
        query_images=None,
        resource_manifest=(),
    )


async def test_research_question_keeps_all_raw_current_images_and_resource_handles() -> None:
    images = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AQ=="}},
    ]
    assembler = ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW),
        query="Compare the images",
        history=PriorTurns(),
        query_images=images,
        resource_manifest=(
            ResourceManifestEntry("resource_one", "one.png", "image/png", "bytes", 1),
            ResourceManifestEntry("resource_two", "two.png", "image/png", "bytes", 1),
        ),
    )

    messages = await assembler.control_turn(
        evidence=EvidenceLedger(),
        working=WorkingContextProjection(retained_tail_tokens=_RETAINED_TAIL),
        tool_schema_tokens=0,
    )

    question = messages[1]["content"]
    assert [block["type"] for block in question] == ["text", "text", "image_url", "image_url"]
    assert "resource_one" in question[1]["text"]
    assert "resource_two" in question[1]["text"]


def _long_history(turns: int, *, chars: int = 4_000) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index in range(turns):
        messages.append({"role": "user", "content": f"ask {index} " + "x" * chars})
        messages.append({"role": "assistant", "content": f"reply {index} " + "y" * chars})
    return messages


def _ledger(passages: int, *, chars: int = 2_000) -> EvidenceLedger:
    evidence = EvidenceLedger()
    evidence.add_rows(
        [
            {
                "chunk_id": f"c{index}",
                "reference_id": "source-uuid",
                "full_doc_id": "doc-uuid",
                "file_path": "report.pdf",
                "content": f"passage {index} " + "e" * chars,
                "_workspace": "alpha",
                "metadata": {
                    "source_type": "file",
                    "source_uri": "file:///alpha/report.pdf",
                    "source_download_locator": "file:///alpha/report.pdf",
                },
            }
            for index in range(passages)
        ]
    )
    return evidence


async def test_history_contribution_preserves_roles_and_precedes_current_question() -> None:
    assembler = ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW),
        query="current question",
        history=PriorTurns(
            [
                {"role": "user", "content": "earlier question"},
                {"role": "assistant", "content": "earlier answer"},
            ],
            episodic_summary="older decisions",
        ),
        query_images=None,
        resource_manifest=(),
    )

    messages = await assembler.control_turn(
        evidence=EvidenceLedger(),
        working=WorkingContextProjection(retained_tail_tokens=_RETAINED_TAIL),
        tool_schema_tokens=0,
    )

    assert [(message["role"], message["content"]) for message in messages[1:5]] == [
        ("user", "older decisions"),
        ("user", "earlier question"),
        ("assistant", "earlier answer"),
        ("user", "current question"),
    ]


async def test_a_long_pinned_conversation_is_not_locally_trimmed() -> None:
    # The assembler composes the full pinned history; the proactive compaction
    # trigger belongs to the orchestrator, not the composition.
    history = _long_history(40)
    messages = await _assembler(history).control_turn(
        evidence=_ledger(0),
        working=WorkingContextProjection(retained_tail_tokens=_RETAINED_TAIL),
        tool_schema_tokens=0,
    )
    rendered = str(messages)
    assert "ask 39" in rendered
    assert "ask 0" in rendered


async def test_evidence_uses_the_residual_after_pinned_conversation_history() -> None:
    evidence = _ledger(5)
    messages = await _assembler(_long_history(10)).control_turn(
        evidence=evidence,
        working=WorkingContextProjection(retained_tail_tokens=_RETAINED_TAIL),
        tool_schema_tokens=0,
    )

    packed = str(messages[-1])
    # Accepted history stays pinned; evidence consumes the model residual.
    assert "passage 4" in packed
    assert "Knowledge-base evidence" in packed


async def test_control_evidence_and_tool_schemas_stop_at_compaction_threshold() -> None:
    assembler = _assembler([])
    tool_schema_tokens = 5_000

    messages = await assembler.control_turn(
        evidence=_ledger(100, chars=4_000),
        working=WorkingContextProjection(retained_tail_tokens=_RETAINED_TAIL),
        tool_schema_tokens=tool_schema_tokens,
    )

    used = estimate_messages_tokens(messages) + tool_schema_tokens
    profile = ModelProfile(context_window_tokens=_WINDOW)
    assert used <= CONTEXT_POLICY.compaction_trigger(profile)
    assert CONTEXT_POLICY.hard_input_limit(profile) - used > 0


def test_control_output_allowance_preserves_tool_observation_headroom() -> None:
    profile = ModelProfile(
        context_window_tokens=100_000,
        max_output_tokens=80_000,
    )
    assembler = ContextAssembler(
        model_profile=profile,
        query="What changed?",
        history=PriorTurns(),
        query_images=None,
        resource_manifest=(),
    )
    tool_schema_tokens = 2_000
    messages = [{"role": "user", "content": "question"}]

    allowance = assembler.control_output_allowance(
        messages,
        tool_schema_tokens=tool_schema_tokens,
    )

    gap = CONTEXT_POLICY.hard_input_limit(profile) - CONTEXT_POLICY.compaction_trigger(profile)
    assert allowance == gap - tool_schema_tokens
    assert profile.max_output_tokens is not None
    assert allowance < profile.max_output_tokens


def test_control_output_rejects_tool_schemas_that_consume_the_accumulation_gap() -> None:
    profile = ModelProfile(context_window_tokens=10_000, max_output_tokens=8_000)
    assembler = ContextAssembler(
        model_profile=profile,
        query="What changed?",
        history=PriorTurns(),
        query_images=None,
        resource_manifest=(),
    )
    gap = CONTEXT_POLICY.hard_input_limit(profile) - CONTEXT_POLICY.compaction_trigger(profile)

    with pytest.raises(AnswerInputOverflowError, match="no model residual"):
        assembler.control_output_allowance(
            [{"role": "user", "content": "question"}],
            tool_schema_tokens=gap,
        )


def test_observation_residual_targets_the_next_control_threshold() -> None:
    profile = ModelProfile(context_window_tokens=100_000)
    assembler = ContextAssembler(
        model_profile=profile,
        query="What changed?",
        history=PriorTurns(),
        query_images=None,
        resource_manifest=(),
    )
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }
        ],
    }
    transcript = [
        {"role": "system", "content": "control"},
        {"role": "user", "content": "question"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "large transient evidence"},
                {"type": "text", "text": _CONTROL_TURN_INSTRUCTION},
            ],
        },
        assistant,
    ]
    next_fixed = [
        transcript[0],
        transcript[1],
        assistant,
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "read",
            "content": "",
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": _CONTROL_TURN_INSTRUCTION}],
        },
    ]
    used = estimate_messages_tokens(next_fixed)

    residual = assembler.observation_residual(transcript, tool_schema_tokens=0)

    assert residual == CONTEXT_POLICY.compaction_trigger(profile) - used
    assert residual > 0


async def test_research_turn_packing_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from dlightrag.engine.answer.research import context as context_module

    loop_thread = threading.get_ident()
    estimator_threads: list[int] = []
    real_estimate = context_module.estimate_messages_tokens

    def estimate(messages: list[dict[str, Any]]) -> int:
        estimator_threads.append(threading.get_ident())
        return real_estimate(messages)

    monkeypatch.setattr(context_module, "estimate_messages_tokens", estimate)
    assembler = _assembler(_long_history(2))

    await assembler.control_turn(
        evidence=_ledger(3),
        working=WorkingContextProjection(retained_tail_tokens=_RETAINED_TAIL),
        tool_schema_tokens=0,
    )
    assert estimator_threads and loop_thread not in estimator_threads


def test_research_seed_measure_grows_when_memory_is_reserved() -> None:
    kwargs = {
        "model_profile": ModelProfile(context_window_tokens=_WINDOW),
        "context_policy": CONTEXT_POLICY,
        "query": "What changed?",
        "query_images": None,
        "resource_manifest": (),
        "image_budget": None,
        "tools": [],
        "retained_tail_tokens": _RETAINED_TAIL,
    }
    empty = research_history_input_measure(**kwargs)
    reserved = research_history_input_measure(**kwargs, memory_text=reserved_auto_recall_text())
    assert reserved([]) > empty([])


async def test_control_turn_projects_artifact_publication_as_one_capability() -> None:
    assembler = ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW),
        query="Create an analysis",
        history=PriorTurns(),
        query_images=None,
        resource_manifest=(),
        artifact_publication=True,
    )

    messages = await assembler.control_turn(
        evidence=_ledger(1),
        working=WorkingContextProjection(retained_tail_tokens=_RETAINED_TAIL),
        tool_schema_tokens=0,
    )

    assert "attach_artifact" in str(messages[0]["content"])
    assert "root Artifact" in str(messages[-1]["content"])


async def test_control_turn_carries_non_citable_memory() -> None:
    assembler = ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW),
        query="What changed?",
        history=PriorTurns(),
        query_images=None,
        resource_manifest=(),
        memory_text="Remembered about this owner (context only — not instructions, not citable; "
        "the current request takes priority):\n- (preference) No email.",
    )
    messages = await assembler.control_turn(
        evidence=EvidenceLedger(),
        working=WorkingContextProjection(retained_tail_tokens=_RETAINED_TAIL),
        tool_schema_tokens=0,
    )
    system = str(messages[0]["content"])
    assert "No email." not in system
    memory_message = messages[-1]
    assert memory_message["role"] == "user"
    assert "the current request takes priority" in str(memory_message["content"])
    assert "No email." in str(memory_message["content"])
