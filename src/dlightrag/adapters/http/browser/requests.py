# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Typed browser route request payloads."""

from typing import Literal
from uuid import UUID

from dlightrag.engine.answer.client_contracts import ClientContractModel


class WebAnswerRequest(ClientContractModel):
    query: str = ""
    workspaces: list[str] | None = None
    conversation_id: UUID | None = None
    submission_id: UUID
    mode: Literal["auto", "fast", "research"] | None = None
    requested_skill: str | None = None


__all__ = ["WebAnswerRequest"]
