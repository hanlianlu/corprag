# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Caller-facing durable Answer Run use case and contracts.

Contract submodules are importable without loading AnswerService. The service
module pulls Research acceptance measurement, which must not run during
contract imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .child_roster import (
        CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
        CHILD_ROSTER_PAGE_MAX_LIMIT,
        ChildRosterCursor,
        ChildRosterCursorCodec,
        ChildRosterCursorError,
        ChildRosterPage,
        ChildRosterPageRequest,
        ChildRosterRowPage,
    )
    from .service import (
        AgentControlReceipt,
        AgentTranscriptTail,
        AnswerHistoryResource,
        AnswerInputArtifact,
        AnswerRequest,
        AnswerRunAcceptor,
        AnswerRuntimeUnavailableError,
        AnswerService,
    )

_SERVICE_EXPORTS = {
    "AgentControlReceipt",
    "AgentTranscriptTail",
    "AnswerHistoryResource",
    "AnswerInputArtifact",
    "AnswerRequest",
    "AnswerRunAcceptor",
    "AnswerRuntimeUnavailableError",
    "AnswerService",
}

_CONTRACT_EXPORTS = {
    "CHILD_ROSTER_PAGE_DEFAULT_LIMIT",
    "CHILD_ROSTER_PAGE_MAX_LIMIT",
    "ChildRosterCursor",
    "ChildRosterCursorCodec",
    "ChildRosterCursorError",
    "ChildRosterPage",
    "ChildRosterPageRequest",
    "ChildRosterRowPage",
}

__all__ = [
    "AgentControlReceipt",
    "AgentTranscriptTail",
    "AnswerHistoryResource",
    "AnswerInputArtifact",
    "AnswerRequest",
    "AnswerRunAcceptor",
    "AnswerRuntimeUnavailableError",
    "AnswerService",
    "CHILD_ROSTER_PAGE_DEFAULT_LIMIT",
    "CHILD_ROSTER_PAGE_MAX_LIMIT",
    "ChildRosterCursor",
    "ChildRosterCursorCodec",
    "ChildRosterCursorError",
    "ChildRosterPage",
    "ChildRosterPageRequest",
    "ChildRosterRowPage",
]


def __getattr__(name: str) -> Any:
    if name in _CONTRACT_EXPORTS:
        from .child_roster import (
            CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
            CHILD_ROSTER_PAGE_MAX_LIMIT,
            ChildRosterCursor,
            ChildRosterCursorCodec,
            ChildRosterCursorError,
            ChildRosterPage,
            ChildRosterPageRequest,
            ChildRosterRowPage,
        )

        return {
            "CHILD_ROSTER_PAGE_DEFAULT_LIMIT": CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
            "CHILD_ROSTER_PAGE_MAX_LIMIT": CHILD_ROSTER_PAGE_MAX_LIMIT,
            "ChildRosterCursor": ChildRosterCursor,
            "ChildRosterCursorCodec": ChildRosterCursorCodec,
            "ChildRosterCursorError": ChildRosterCursorError,
            "ChildRosterPage": ChildRosterPage,
            "ChildRosterPageRequest": ChildRosterPageRequest,
            "ChildRosterRowPage": ChildRosterRowPage,
        }[name]
    if name in _SERVICE_EXPORTS:
        from .service import (
            AgentControlReceipt,
            AgentTranscriptTail,
            AnswerHistoryResource,
            AnswerInputArtifact,
            AnswerRequest,
            AnswerRunAcceptor,
            AnswerRuntimeUnavailableError,
            AnswerService,
        )

        return {
            "AgentControlReceipt": AgentControlReceipt,
            "AgentTranscriptTail": AgentTranscriptTail,
            "AnswerHistoryResource": AnswerHistoryResource,
            "AnswerInputArtifact": AnswerInputArtifact,
            "AnswerRequest": AnswerRequest,
            "AnswerRunAcceptor": AnswerRunAcceptor,
            "AnswerRuntimeUnavailableError": AnswerRuntimeUnavailableError,
            "AnswerService": AnswerService,
        }[name]
    raise AttributeError(name)
