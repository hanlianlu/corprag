# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""AnswerModeRouter and durable resolve helpers."""

import pytest

from dlightrag.engine.answer.router import AnswerModeRouter, RoutingFailedError
from dlightrag.engine.answer.runs.routing import decide_resolved_mode


def test_explicit_mode_resolves_without_a_router() -> None:
    assert (
        decide_resolved_mode(requested_mode="fast", valid_modes=frozenset({"fast", "research"}))
        == "fast"
    )
    assert decide_resolved_mode(requested_mode="research", valid_modes=frozenset({"research"})) == (
        "research"
    )


def test_auto_with_one_valid_mode_skips_the_router() -> None:
    assert decide_resolved_mode(requested_mode="auto", valid_modes=frozenset({"fast"})) == "fast"


def test_auto_with_both_modes_needs_the_router() -> None:
    assert (
        decide_resolved_mode(requested_mode="auto", valid_modes=frozenset({"fast", "research"}))
        is None
    )


async def test_router_accepts_structured_mode() -> None:
    async def llm(**_kwargs: object) -> str:
        return '{"mode":"research"}'

    chosen = await AnswerModeRouter(llm).choose(
        query="compare the filings",
        valid_modes=("fast", "research"),
    )
    assert chosen == "research"


async def test_router_prompt_declares_the_exact_json_contract() -> None:
    async def llm(**kwargs: object) -> str:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        system = str(messages[0]["content"])
        if '{"mode":"fast"}' not in system or '{"mode":"research"}' not in system:
            return " " * 96
        return '{"mode":"research"}'

    chosen = await AnswerModeRouter(llm).choose(
        query="compare a long conversation",
        history=[{"role": "assistant", "content": "prior answer " * 500}],
        valid_modes=("fast", "research"),
    )

    assert chosen == "research"


async def test_router_defaults_to_research_and_reads_full_context() -> None:
    captured: dict[str, object] = {}

    async def llm(**kwargs: object) -> str:
        captured.update(kwargs)
        return '{"mode":"fast"}'

    history = [{"role": "user", "content": "根据刚上传的年报"}]
    await AnswerModeRouter(llm).choose(
        query="净利润是多少",
        history=history,
        valid_modes=("fast", "research"),
    )
    messages = captured["messages"]
    assert isinstance(messages, list)
    system = messages[0]["content"]
    assert "Default research" in system
    assert "Unsure → research" in system
    assert messages[1] == history[0]
    assert "净利润是多少" in messages[-1]["content"]


async def test_router_ignores_extra_json_fields() -> None:
    async def llm(**_kwargs: object) -> str:
        return '{\n  "mode": "research",\n  "response": "long leftover answer"\n}'

    chosen = await AnswerModeRouter(llm).choose(
        query="evaluate my resume",
        valid_modes=("fast", "research"),
    )
    assert chosen == "research"


async def test_router_accepts_a_bare_mode_token() -> None:
    async def llm(**_kwargs: object) -> str:
        return "research"

    chosen = await AnswerModeRouter(llm).choose(
        query="write a poem",
        valid_modes=("fast", "research"),
    )
    assert chosen == "research"


async def test_router_rejects_invalid_structured_output_without_echoing_it() -> None:
    invalid_output = "sensitive model echo"

    async def llm(**_kwargs: object) -> str:
        return invalid_output

    with pytest.raises(RoutingFailedError) as raised:
        await AnswerModeRouter(llm).choose(query="q", valid_modes=("fast", "research"))

    message = str(raised.value)
    assert "invalid mode" in message
    assert "chars=20" in message
    assert invalid_output not in message
