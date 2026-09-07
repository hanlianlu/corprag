# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Web Conversation records and persistence port."""

import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import UUID

from dlightrag.application.answer_runs.routing import RoutingAcceptance
from dlightrag.application.opaque_cursor import OpaqueCursorEnvelope
from dlightrag.application.runs import RunView
from dlightrag.engine.runtime import PendingArtifact, PendingArtifactReference, PreparedRunEnvelope


@dataclass(frozen=True, slots=True)
class LinkedTurn:
    """One conversation entry and the authoritative run state behind it."""

    turn_id: str
    turn_number: int
    submission_id: str
    created_at: datetime.datetime
    run: RunView
    conversation_id: str = ""

    @property
    def answer_run_id(self) -> str:
        return self.run.run_id


@dataclass(frozen=True, slots=True)
class ConversationHead:
    """Owned conversation identity and durable execution mapping, without turns."""

    principal_id: str
    conversation_id: str
    content_revision: int
    title: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    agent_session_id: str
    agent_lane_id: str


@dataclass(frozen=True, slots=True)
class CarriedAttachment:
    """Bounded metadata for one successful prior current attachment."""

    run_id: str
    source_ordinal: int
    digest: str
    filename: str
    mime_type: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class SubmissionSeed:
    """Conversation mapping plus bounded attachment carry-forward metadata."""

    head: ConversationHead
    attachments: tuple[CarriedAttachment, ...] = ()


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    """Application projection of one durable conversation row."""

    conversation_id: str
    title: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    forked_from_conversation_id: str | None = None
    forked_from_title: str | None = None


CONVERSATION_PAGE_DEFAULT_LIMIT = 50
CONVERSATION_PAGE_MAX_LIMIT = 100
CONVERSATION_HISTORY_PAGE_DEFAULT_LIMIT = 40
CONVERSATION_HISTORY_PAGE_MAX_LIMIT = 100
RECOVERY_PAGE_MAX_LIMIT = 128


class ConversationCursorError(ValueError):
    """An opaque conversation page cursor is malformed or fails integrity checking."""


@dataclass(frozen=True, slots=True)
class ConversationCursor:
    """The complete ordering key for the next newest-first page."""

    updated_at: datetime.datetime
    conversation_id: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.updated_at, datetime.datetime):
            raise ValueError("conversation cursor timestamp must be a datetime")
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("conversation cursor timestamp must include a timezone")
        if not isinstance(self.conversation_id, UUID):
            raise ValueError("conversation cursor id must be a UUID")


@dataclass(frozen=True, slots=True)
class ConversationPageRequest:
    """One hard-bounded application request for conversation summaries."""

    limit: int = CONVERSATION_PAGE_DEFAULT_LIMIT
    cursor: ConversationCursor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("conversation page limit must be an integer")
        if not 1 <= self.limit <= CONVERSATION_PAGE_MAX_LIMIT:
            raise ValueError(
                f"conversation page limit must be between 1 and {CONVERSATION_PAGE_MAX_LIMIT}"
            )
        if self.cursor is not None and not isinstance(self.cursor, ConversationCursor):
            raise ValueError("conversation cursor must contain paired ordering fields")


@dataclass(frozen=True, slots=True)
class ConversationRowPage:
    """Bounded persistence result, including the measured physical fetch size."""

    items: tuple[Mapping[str, Any], ...]
    has_more: bool
    fetched_rows: int


@dataclass(frozen=True, slots=True)
class ConversationPage:
    """Newest-first application page with a paired continuation key."""

    items: tuple[ConversationSummary, ...]
    next_cursor: ConversationCursor | None
    fetched_rows: int


@dataclass(frozen=True, slots=True)
class ConversationHistoryCursor:
    """Immutable boundary for an older page in one conversation."""

    conversation_id: UUID
    before_turn_number: int

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_id, UUID):
            raise ValueError("conversation history cursor id must be a UUID")
        if isinstance(self.before_turn_number, bool) or not isinstance(
            self.before_turn_number, int
        ):
            raise ValueError("conversation history cursor turn number must be an integer")
        if self.before_turn_number < 1:
            raise ValueError("conversation history cursor turn number must be positive")


@dataclass(frozen=True, slots=True)
class ConversationHistoryPageRequest:
    """One hard-bounded recent or older turn-page request."""

    limit: int = CONVERSATION_HISTORY_PAGE_DEFAULT_LIMIT
    cursor: ConversationHistoryCursor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("conversation history page limit must be an integer")
        if not 1 <= self.limit <= CONVERSATION_HISTORY_PAGE_MAX_LIMIT:
            raise ValueError(
                "conversation history page limit must be between 1 and "
                f"{CONVERSATION_HISTORY_PAGE_MAX_LIMIT}"
            )
        if self.cursor is not None and not isinstance(self.cursor, ConversationHistoryCursor):
            raise ValueError("conversation history cursor is invalid")


@dataclass(frozen=True, slots=True)
class ConversationHistoryPage:
    """Chronological presentation page with an older-page continuation."""

    conversation: ConversationHead
    turns: tuple[LinkedTurn, ...]
    next_cursor: ConversationHistoryCursor | None
    fetched_rows: int


@dataclass(frozen=True, slots=True)
class RecoveryPageRequest:
    """Bounded physical turn scan used only for durable recovery projection."""

    direction: Literal["newest", "oldest"]
    limit: int
    before_turn_number: int | None = None
    after_turn_number: int | None = None
    upper_turn_number: int | None = None

    def __post_init__(self) -> None:
        if self.direction not in {"newest", "oldest"}:
            raise ValueError("recovery direction is invalid")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("recovery page limit must be an integer")
        if not 1 <= self.limit <= RECOVERY_PAGE_MAX_LIMIT:
            raise ValueError(f"recovery page limit must be between 1 and {RECOVERY_PAGE_MAX_LIMIT}")
        for value in (
            self.before_turn_number,
            self.after_turn_number,
            self.upper_turn_number,
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError("recovery turn boundaries must be positive integers")


@dataclass(frozen=True, slots=True)
class RecoveryTurnBatch:
    """One bounded physical recovery read; turns follow the requested direction."""

    turns: tuple[LinkedTurn, ...]
    has_more: bool
    fetched_rows: int


class ConversationCursorCodec:
    """Encode paired ordering facts as an opaque, integrity-checked token.

    The cursor deliberately carries no principal. Ownership remains a mandatory
    query predicate derived from authentication on every page. Pre-governance
    tokens without scope/version pins are intentionally invalidated because
    continuation cursors are short-lived state.
    """

    def __init__(self, secret: bytes) -> None:
        self._envelope = OpaqueCursorEnvelope(
            secret,
            domain="conversation-list",
            scope="conversation-list",
            fields_by_version={1: {"conversation_id", "updated_at"}},
            current_version=1,
        )

    def encode(self, cursor: ConversationCursor) -> str:
        return self._envelope.encode(
            {
                "conversation_id": str(cursor.conversation_id),
                "updated_at": _canonical_cursor_timestamp(cursor.updated_at),
            }
        )

    def decode(self, token: str) -> ConversationCursor:
        try:
            decoded = self._envelope.decode(token)
            conversation_id_text = decoded["conversation_id"]
            timestamp_text = decoded["updated_at"]
            if not isinstance(conversation_id_text, str) or not isinstance(timestamp_text, str):
                raise ValueError
            conversation_id = UUID(conversation_id_text)
            if str(conversation_id) != conversation_id_text:
                raise ValueError
            updated_at = datetime.datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
            if _canonical_cursor_timestamp(updated_at) != timestamp_text:
                raise ValueError
            return ConversationCursor(
                updated_at=updated_at,
                conversation_id=conversation_id,
            )
        except ValueError as exc:
            raise ConversationCursorError("invalid conversation page cursor") from exc


class ConversationHistoryCursorCodec:
    """Signed, opaque, conversation-bound turn cursor with canonical decoding."""

    def __init__(self, secret: bytes) -> None:
        self._envelope = OpaqueCursorEnvelope(
            secret,
            domain="conversation-history",
            scope="conversation-history",
            fields_by_version={1: {"before_turn_number", "conversation_id"}},
            current_version=1,
        )

    def encode(self, cursor: ConversationHistoryCursor) -> str:
        return self._envelope.encode(
            {
                "before_turn_number": cursor.before_turn_number,
                "conversation_id": str(cursor.conversation_id),
            }
        )

    def decode(self, token: str) -> ConversationHistoryCursor:
        try:
            decoded = self._envelope.decode(token)
            conversation_text = decoded["conversation_id"]
            before_turn_number = decoded["before_turn_number"]
            if not isinstance(conversation_text, str):
                raise ValueError
            conversation_id = UUID(conversation_text)
            if str(conversation_id) != conversation_text:
                raise ValueError
            return ConversationHistoryCursor(
                conversation_id=conversation_id,
                before_turn_number=before_turn_number,
            )
        except ValueError as exc:
            raise ConversationCursorError("invalid conversation history cursor") from exc


def _canonical_cursor_timestamp(value: datetime.datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("conversation cursor timestamp must include a timezone")
    return value.astimezone(datetime.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class AnswerTurnCreation:
    """The run and conversation entry one submission durably created."""

    turn: LinkedTurn
    summary: dict[str, Any]
    replayed: bool


class ConversationSubmissionConflict(RuntimeError):
    """One principal reused a submission id for different accepted input."""


class WebConversationUnavailableError(RuntimeError):
    """Durable Web Conversation storage cannot currently be reached."""

    detail = "Web conversation storage is unavailable"


class WebConversationSchemaError(RuntimeError):
    """The durable Web Conversation schema is incompatible with this revision."""


class WebConversationStore(Protocol):
    """Persistence operations owned by Web Conversations."""

    async def prune_empty_conversations(self, *, batch_size: int = 500) -> int: ...
    async def create_conversation(self, principal_id: str) -> dict[str, Any]: ...
    async def list_conversations(
        self,
        principal_id: str,
        *,
        page: ConversationPageRequest,
    ) -> ConversationRowPage: ...
    async def rename_conversation(
        self, principal_id: str, conversation_id: str, *, title: str
    ) -> dict[str, Any] | None: ...
    async def delete_conversation(self, principal_id: str, conversation_id: str) -> bool: ...
    async def delete_all_conversations(self, principal_id: str) -> int: ...
    async def history_page(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        page: ConversationHistoryPageRequest,
    ) -> ConversationHistoryPage | None: ...
    async def submission_seed(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        attachment_limit: int,
    ) -> SubmissionSeed | None: ...
    async def recovery_page(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        page: RecoveryPageRequest,
    ) -> RecoveryTurnBatch: ...
    async def find_turn_by_run(self, principal_id: str, run_id: str) -> LinkedTurn | None: ...
    async def find_answer_turn_by_submission(
        self, principal_id: str, submission_id: str
    ) -> AnswerTurnCreation | None: ...
    async def replay_answer_turn(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        idempotency_fingerprint: str,
    ) -> AnswerTurnCreation | None: ...
    async def create_answer_turn(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        envelope: PreparedRunEnvelope,
        run_id: str,
        artifacts: Sequence[PendingArtifact],
        references: Sequence[PendingArtifactReference],
        title_hint: str | None,
        routing: RoutingAcceptance | None = None,
        create_conversation: bool = False,
        forked_from_conversation_id: str | None = None,
    ) -> AnswerTurnCreation | None: ...


__all__ = [
    "AnswerTurnCreation",
    "CarriedAttachment",
    "CONVERSATION_HISTORY_PAGE_DEFAULT_LIMIT",
    "CONVERSATION_HISTORY_PAGE_MAX_LIMIT",
    "CONVERSATION_PAGE_DEFAULT_LIMIT",
    "CONVERSATION_PAGE_MAX_LIMIT",
    "ConversationCursor",
    "ConversationCursorCodec",
    "ConversationCursorError",
    "ConversationHead",
    "ConversationHistoryCursor",
    "ConversationHistoryCursorCodec",
    "ConversationHistoryPage",
    "ConversationHistoryPageRequest",
    "ConversationPage",
    "ConversationPageRequest",
    "ConversationRowPage",
    "ConversationSummary",
    "ConversationSubmissionConflict",
    "LinkedTurn",
    "RECOVERY_PAGE_MAX_LIMIT",
    "RecoveryPageRequest",
    "RecoveryTurnBatch",
    "SubmissionSeed",
    "WebConversationSchemaError",
    "WebConversationStore",
    "WebConversationUnavailableError",
]
