# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Typed browser-facing SSE event payloads.

One durable run event becomes one browser frame. Nothing here is stored: the
rendered answer, its images, and its sources are derived from the run's
canonical result when the terminal event is projected for this reader.
"""

from typing import Any, Literal

from pydantic import Field

from dlightrag.adapters.http.browser.presentation import AnswerPresentation
from dlightrag.application.runs import RunPhase
from dlightrag.engine.answer.client_contracts import ClientContractModel


class AnswerProgressEvent(ClientContractModel):
    phase: RunPhase


class AnswerDoneEvent(ClientContractModel):
    status: Literal["succeeded", "cancelled"]
    presentation: AnswerPresentation | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)


class AnswerErrorEvent(ClientContractModel):
    kind: str
    message: str


__all__ = [
    "AnswerDoneEvent",
    "AnswerErrorEvent",
    "AnswerProgressEvent",
]
