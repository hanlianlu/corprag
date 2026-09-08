# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Structured AnswerModeRouter for auto when both Fast and Research are valid."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from dlightrag.engine.ai.structured import StructuredOutput
from dlightrag.engine.ai.tokens import estimate_messages_tokens
from dlightrag.engine.answer.mode import ModeResource, ResolvedMode

_ROUTER_SYSTEM = (
    "Pick one allowed mode. Default research. "
    "fast: one-shot KB retrieve and generate — only if history plus this turn "
    "asks the corpus or continues corpus-grounded work. "
    "Otherwise research. Unsure → research."
)


class _ModeDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: Literal["fast", "research"]


ROUTER_STRUCTURED_OUTPUT = StructuredOutput(name="answer_mode", schema=_ModeDecision)


def _coerce_mode(raw: Any, valid_modes: Sequence[str]) -> ResolvedMode:
    """Accept {\"mode\": ...} JSON or a bare fast/research token."""
    try:
        parsed = ROUTER_STRUCTURED_OUTPUT.parse(raw)
        mode = getattr(parsed, "mode", None)
        if mode in {"fast", "research"} and mode in valid_modes:
            return mode
    except ValueError, TypeError:
        pass
    text = str(raw)
    token = text.strip().strip("`\"'").casefold()
    if token in {"fast", "research"} and token in valid_modes:
        return token  # type: ignore[return-value]
    raise RoutingFailedError(
        "router returned an invalid mode "
        f"(type={type(raw).__name__}, chars={len(text)}, "
        f"non_whitespace={bool(text.strip())}, allowed={list(valid_modes)})"
    )


class RoutingFailedError(RuntimeError):
    """The router did not return a legal structured mode."""


class AnswerModeRouter:
    """One structured call that picks fast or research."""

    def __init__(self, llm: Callable[..., Awaitable[Any]]) -> None:
        self._llm = llm

    def history_input_measure(
        self,
        query: str,
        *,
        resources: Sequence[ModeResource] = (),
        valid_modes: Sequence[str] = ("fast", "research"),
    ) -> Callable[..., int]:
        def measure(
            history: list[dict[str, Any]],
            projected_summary: str = "",
        ) -> int:
            # Routing receives the pinned recent-message projection only.
            del projected_summary
            return estimate_messages_tokens(
                self._messages(
                    query,
                    history=history,
                    resources=resources,
                    valid_modes=valid_modes,
                    tool_categories=(),
                    has_images=False,
                )
            )

        return measure

    async def choose(
        self,
        *,
        query: str,
        history: Sequence[Mapping[str, Any]] = (),
        resources: Sequence[ModeResource] = (),
        tool_categories: Sequence[str] = (),
        has_images: bool = False,
        valid_modes: Sequence[str],
    ) -> ResolvedMode:
        raw = await self._llm(
            messages=self._messages(
                query,
                history=[dict(item) for item in history],
                resources=resources,
                valid_modes=valid_modes,
                tool_categories=tool_categories,
                has_images=has_images,
            ),
            structured_output=ROUTER_STRUCTURED_OUTPUT,
        )
        try:
            return _coerce_mode(raw, valid_modes)
        except RoutingFailedError:
            raise
        except (ValueError, TypeError) as exc:
            raise RoutingFailedError("router output was not a valid mode") from exc

    def _messages(
        self,
        query: str,
        *,
        history: list[dict[str, Any]],
        resources: Sequence[ModeResource],
        valid_modes: Sequence[str],
        tool_categories: Sequence[str],
        has_images: bool,
    ) -> list[dict[str, Any]]:
        roles = ",".join(resource.role for resource in resources) or "none"
        tools = ",".join(tool_categories) or "none"
        allowed = ",".join(valid_modes)
        allowed_outputs = " or ".join(f'{{"mode":"{mode}"}}' for mode in valid_modes)
        system = (
            f"{_ROUTER_SYSTEM} Return exactly one JSON object: {allowed_outputs}. "
            "No other keys or text."
        )
        user = (
            f"query: {query}\n"
            f"resources: {roles}\n"
            f"images: {has_images}\n"
            f"tools: {tools}\n"
            f"allowed: {allowed}"
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user})
        return messages


__all__ = ["AnswerModeRouter", "ROUTER_STRUCTURED_OUTPUT", "RoutingFailedError"]
