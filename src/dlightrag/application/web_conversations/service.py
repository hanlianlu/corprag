# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Application-owned persistence lifecycle for durable Web Conversations."""

import asyncio
import datetime
import logging
from collections.abc import Awaitable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar
from uuid import UUID, uuid5

from dlightrag.application.access import UserContext, owner_id_from_user
from dlightrag.application.answer_runs import (
    AnswerHistoryResource,
    AnswerInputArtifact,
    AnswerRequest,
    AnswerRunAcceptor,
    AnswerService,
)
from dlightrag.application.answer_runs.execution import AnswerRunRequest
from dlightrag.application.answer_runs.routing import RoutingAcceptance
from dlightrag.application.runs import RunView
from dlightrag.engine.agent.session.fold import PriorTurns
from dlightrag.engine.ai.media import thumbnail_bytes
from dlightrag.engine.answer.history import (
    HistoryProjectionTarget,
    IncrementalHistoryProjector,
)
from dlightrag.engine.answer.resources.models import ResourceInput
from dlightrag.engine.runtime import (
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
    RunKind,
    parse_run_id,
    run_request_fingerprint,
)

from .models import (
    AnswerTurnCreation,
    ConversationCursor,
    ConversationCursorCodec,
    ConversationHead,
    ConversationHistoryCursor,
    ConversationHistoryCursorCodec,
    ConversationHistoryPage,
    ConversationHistoryPageRequest,
    ConversationPage,
    ConversationPageRequest,
    ConversationSummary,
    LinkedTurn,
    RecoveryPageRequest,
    SubmissionSeed,
    WebConversationStore,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


class WebAttachment(Protocol):
    """Admitted browser attachment fields consumed by Application acceptance."""

    @property
    def filename(self) -> str: ...

    @property
    def mime_type(self) -> str: ...

    @property
    def ordinal(self) -> int: ...

    @property
    def byte_size(self) -> int: ...

    @property
    def content_sha256(self) -> str: ...

    @property
    def attachment_bytes(self) -> bytes: ...


_HISTORY_THUMBNAIL_MAX_PX = 320
_HISTORY_THUMBNAIL_MAX_BYTES = 128 * 1024
_HISTORY_THUMBNAIL_QUALITY = 82
_HISTORY_THUMBNAIL_MIN_QUALITY = 50
_HISTORY_THUMBNAIL_MIN_PX = 64
_PRUNE_INTERVAL_SECONDS = 60 * 60
_PRUNE_BATCH_SIZE = 500
_RECOVERY_BATCH_SIZE = 64
_NEW_CONVERSATION_NAMESPACE = UUID("9c0e62a5-a12c-45b2-8aeb-474fc2237cdf")


def _is_image_mime(mime_type: str | None) -> bool:
    return bool(mime_type) and mime_type.lower().startswith("image/")


@dataclass(frozen=True, slots=True)
class WebAnswerSubmission:
    """The run and conversation entry one accepted browser submission created."""

    run: RunView
    turn_id: str
    turn_number: int
    conversation: ConversationSummary
    submission_id: str = ""
    created_at: datetime.datetime | None = None


@dataclass(frozen=True, slots=True)
class _PreparedSubmission:
    """One browser submission projected into the Answer service contract."""

    request: AnswerRequest


@dataclass(frozen=True, slots=True)
class _WebAnswerAcceptor(AnswerRunAcceptor[WebAnswerSubmission]):
    """Atomically link AnswerService acceptance to one browser conversation."""

    store: WebConversationStore
    conversation_id: str
    title_hint: str | None
    create_conversation: bool = False
    forked_from_conversation_id: str | None = None

    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        run_kind: RunKind,
    ) -> WebAnswerSubmission | None:
        if run_kind != "answer":
            raise ValueError("Web Answer replay requires an Answer run kind")
        creation = await self.store.replay_answer_turn(
            principal_id=owner_id,
            conversation_id=self.conversation_id,
            submission_id=idempotency_key,
            idempotency_fingerprint=idempotency_fingerprint,
        )
        return None if creation is None else _submission(creation)

    async def create_run(
        self,
        *,
        envelope: PreparedRunEnvelope,
        run_id: str,
        resources: Sequence[Mapping[str, Any]] = (),
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
    ) -> WebAnswerSubmission | None:
        creation = await self.store.create_answer_turn(
            principal_id=envelope.submitted_by,
            conversation_id=self.conversation_id,
            submission_id=envelope.submission_key,
            envelope=envelope,
            run_id=run_id,
            artifacts=artifacts,
            references=references,
            title_hint=self.title_hint,
            routing=routing,
            create_conversation=self.create_conversation,
            forked_from_conversation_id=self.forked_from_conversation_id,
        )
        return None if creation is None else _submission(creation)


class WebConversationService:
    """Map authenticated browser operations onto the scoped persistence store."""

    def __init__(
        self,
        *,
        store: WebConversationStore,
        answers: AnswerService,
        max_attachments: int,
        cursor_secret: bytes,
    ) -> None:
        self._store = store
        self._answers = answers
        self._max_attachments = max_attachments
        self._cursor_codec = ConversationCursorCodec(cursor_secret)
        self._history_cursor_codec = ConversationHistoryCursorCodec(cursor_secret)
        self._prune_task: asyncio.Task[None] | None = None

    @property
    def cursor_codec(self) -> ConversationCursorCodec:
        """Return the codec shared by this service and its HTTP adapter."""
        return self._cursor_codec

    @property
    def history_cursor_codec(self) -> ConversationHistoryCursorCodec:
        """Return the owned HTTP adapter's conversation-turn cursor codec."""
        return self._history_cursor_codec

    async def start_retention(self) -> None:
        """Start the empty-conversation sweep after schema composition."""
        await self._prune_empty_batch()
        if self._prune_task is None:
            self._prune_task = asyncio.create_task(self._prune_empty_loop())

    async def _prune_empty_batch(self) -> int:
        try:
            return await self._store.prune_empty_conversations(batch_size=_PRUNE_BATCH_SIZE)
        except Exception:
            logger.exception("Failed to prune empty Web conversations")
            return 0

    async def _prune_empty_loop(self) -> None:
        while True:
            await asyncio.sleep(_PRUNE_INTERVAL_SECONDS)
            while await self._prune_empty_batch() >= _PRUNE_BATCH_SIZE:
                await asyncio.sleep(0)

    async def aclose(self) -> None:
        """Stop periodic retention. Safe to call more than once."""
        task = self._prune_task
        self._prune_task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    # ------------------------------------------------------------------
    # Conversation lifecycle
    # ------------------------------------------------------------------

    async def create(self, user: UserContext | None) -> ConversationSummary:
        principal_id = owner_id_from_user(user)
        row = await self._store_call(self._store.create_conversation(principal_id))
        return _conversation_summary(row)

    async def list(
        self,
        user: UserContext | None,
        *,
        page: ConversationPageRequest | None = None,
    ) -> ConversationPage:
        """Return one newest-first keyset page scoped to the authenticated principal.

        This is standard keyset pagination rather than a snapshot: conversations
        created or touched above the supplied cursor after an earlier page are not
        pulled into the older traversal. Already returned rows never repeat.
        """
        principal_id = owner_id_from_user(user)
        requested_page = page or ConversationPageRequest()
        result = await self._store_call(
            self._store.list_conversations(principal_id, page=requested_page)
        )
        items = tuple(_conversation_summary(row) for row in result.items)
        next_cursor = None
        if result.has_more:
            if not items:
                raise RuntimeError("conversation store reported more rows after an empty page")
            last = items[-1]
            next_cursor = ConversationCursor(
                updated_at=last.updated_at,
                conversation_id=UUID(last.conversation_id),
            )
        return ConversationPage(
            items=items,
            next_cursor=next_cursor,
            fetched_rows=result.fetched_rows,
        )

    async def history(
        self,
        user: UserContext | None,
        conversation_id: str,
        *,
        page: ConversationHistoryPageRequest | None = None,
    ) -> ConversationHistoryPage | None:
        """Return one chronological recent or older presentation page."""
        principal_id = owner_id_from_user(user)
        requested = page or ConversationHistoryPageRequest()
        if (
            requested.cursor is not None
            and str(requested.cursor.conversation_id) != conversation_id
        ):
            raise ValueError("conversation history cursor belongs to another conversation")
        result = await self._store_call(
            self._store.history_page(
                principal_id,
                conversation_id,
                page=requested,
            )
        )
        if result is None:
            return None
        next_cursor = None
        if result.next_cursor is not None:
            next_cursor = ConversationHistoryCursor(
                conversation_id=UUID(conversation_id),
                before_turn_number=result.next_cursor.before_turn_number,
            )
        return ConversationHistoryPage(
            conversation=result.conversation,
            turns=result.turns,
            next_cursor=next_cursor,
            fetched_rows=result.fetched_rows,
        )

    async def rename(
        self,
        user: UserContext | None,
        conversation_id: str,
        title: str,
    ) -> ConversationSummary | None:
        principal_id = owner_id_from_user(user)
        row = await self._store_call(
            self._store.rename_conversation(
                principal_id,
                conversation_id,
                title=title,
            )
        )
        return _conversation_summary(row) if row is not None else None

    async def delete(self, user: UserContext | None, conversation_id: str) -> bool:
        principal_id = owner_id_from_user(user)
        return await self._store_call(
            self._store.delete_conversation(principal_id, conversation_id)
        )

    async def delete_all(self, user: UserContext | None) -> int:
        principal_id = owner_id_from_user(user)
        return await self._store_call(self._store.delete_all_conversations(principal_id))

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def turn_for_run(self, user: UserContext | None, run_id: str) -> LinkedTurn | None:
        """Return the conversation entry an owned run belongs to, if any.

        An unparseable identifier is simply unknown here: every run route reads
        through this projection, so a malformed id is the same opaque miss as a
        pruned or foreign one and never reaches storage.
        """
        principal_id = owner_id_from_user(user)
        if parse_run_id(run_id) is None:
            return None
        return await self._store_call(self._store.find_turn_by_run(principal_id, run_id))

    async def submission(
        self, user: UserContext | None, submission_id: str
    ) -> WebAnswerSubmission | None:
        """Return one owner-scoped accepted browser submission, if it exists."""
        principal_id = owner_id_from_user(user)
        try:
            UUID(submission_id)
        except ValueError:
            return None
        creation = await self._store_call(
            self._store.find_answer_turn_by_submission(principal_id, submission_id)
        )
        return None if creation is None else _submission(creation)

    async def attachment(
        self,
        user: UserContext | None,
        run_id: str,
        ordinal: int,
    ) -> AnswerInputArtifact | None:
        """Load one owned run's uploaded attachment bytes by its ordinal."""
        principal_id = owner_id_from_user(user)
        turn = await self.turn_for_run(user, run_id)
        if turn is None:
            return None
        return await self._store_call(
            self._answers.read_input_artifact(
                owner_id=principal_id,
                run_id=run_id,
                ordinal=ordinal,
            )
        )

    async def thumbnail(
        self,
        user: UserContext | None,
        run_id: str,
        ordinal: int,
    ) -> tuple[bytes, str] | None:
        """Derive one bounded UI thumbnail for an image attachment."""
        stored = await self.attachment(user, run_id, ordinal)
        if stored is None or not _is_image_mime(stored.mime_type):
            return None
        try:
            payload, mime_type = await asyncio.to_thread(
                thumbnail_bytes,
                stored.content,
                max_px=_HISTORY_THUMBNAIL_MAX_PX,
                max_bytes=_HISTORY_THUMBNAIL_MAX_BYTES,
                quality=_HISTORY_THUMBNAIL_QUALITY,
                min_quality=_HISTORY_THUMBNAIL_MIN_QUALITY,
                min_px=_HISTORY_THUMBNAIL_MIN_PX,
            )
        except Exception:
            logger.warning("Failed to derive Web conversation thumbnail", exc_info=True)
            return None
        return payload, mime_type

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def start_answer(
        self,
        user: UserContext | None,
        *,
        conversation_id: str | None,
        submission_id: str,
        query: str,
        workspaces: Sequence[str],
        attachments: Sequence[WebAttachment] = (),
        mode: str | None = None,
        requested_skill: str | None = None,
    ) -> WebAnswerSubmission | None:
        """Create or replay one submission's run and its conversation entry.

        An existing conversation is validated and locked. When ``conversation_id``
        is absent, a stable server-generated id is derived from this owner-wide
        submission key and the conversation is inserted in the same transaction
        as its first run, uploaded bytes, and linked turn.
        """
        principal_id = owner_id_from_user(user)
        create_conversation = conversation_id is None
        if create_conversation:
            conversation_id = _new_conversation_id(principal_id, submission_id)
        idempotency_fingerprint = _web_answer_request_fingerprint(
            conversation_id=conversation_id,
            query=query,
            workspaces=workspaces,
            attachments=attachments,
            mode=mode,
            requested_skill=requested_skill,
        )
        replay = await self._store_call(
            self._store.replay_answer_turn(
                principal_id=principal_id,
                conversation_id=conversation_id,
                submission_id=submission_id,
                idempotency_fingerprint=idempotency_fingerprint,
            )
        )
        if replay is not None:
            return _submission(replay)

        if create_conversation:
            seed = SubmissionSeed(head=_empty_head(principal_id, conversation_id))
        else:
            seed = await self._store_call(
                self._store.submission_seed(
                    principal_id,
                    conversation_id,
                    attachment_limit=max(0, self._max_attachments - len(attachments)),
                )
            )
            if seed is None:
                return None
        prepared = _prepare_submission(
            query=query,
            workspaces=workspaces,
            seed=seed,
            attachments=attachments,
            mode=mode,
            requested_skill=requested_skill,
        )

        async def resolve_history(
            targets: Sequence[HistoryProjectionTarget],
        ) -> PriorTurns:
            if create_conversation:
                return IncrementalHistoryProjector(targets=targets).finish()
            return await self._resolve_recovery_history(
                principal_id,
                conversation_id,
                targets=targets,
            )

        try:
            return await self._answers.accept(
                request=prepared.request,
                owner_id=principal_id,
                idempotency_key=submission_id,
                idempotency_fingerprint=idempotency_fingerprint,
                auth_mode=(user.auth_mode if user is not None else "none"),
                acceptor=_WebAnswerAcceptor(
                    store=self._store,
                    conversation_id=conversation_id,
                    title_hint=_auto_title(query),
                    create_conversation=create_conversation,
                ),
                history_resolver=resolve_history,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # A concurrent identical acceptance may win while recovery is being
            # projected. Never turn that durable replay into a recovery failure.
            replay = await self._store_call(
                self._store.replay_answer_turn(
                    principal_id=principal_id,
                    conversation_id=conversation_id,
                    submission_id=submission_id,
                    idempotency_fingerprint=idempotency_fingerprint,
                )
            )
            if replay is not None:
                return _submission(replay)
            raise

    async def continue_answer(
        self,
        user: UserContext | None,
        *,
        parent_run_id: str,
        submission_id: str,
        query: str,
        kind: str,
        authorized_workspaces: Sequence[str] | None,
    ) -> WebAnswerSubmission | None:
        """Start a linked follow-up or a new conversation branch."""
        if kind not in {"follow_up", "fork"}:
            raise ValueError(f"unsupported continuation kind: {kind}")
        principal_id = owner_id_from_user(user)
        parent = await self.turn_for_run(user, parent_run_id)
        if parent is None or not parent.conversation_id:
            return None
        request = await self._answers.continuation_request(
            owner_id=principal_id,
            run_id=parent_run_id,
            query=query,
            include_answer=kind == "follow_up",
            authorized_workspaces=authorized_workspaces,
        )
        if request is None:
            return None
        create_conversation = kind == "fork"
        conversation_id = (
            _new_conversation_id(principal_id, submission_id)
            if create_conversation
            else parent.conversation_id
        )
        fingerprint = run_request_fingerprint(
            {
                "conversation_id": conversation_id,
                "parent_run_id": parent_run_id,
                "continuation_kind": kind,
                "query": query.strip(),
            }
        )
        return await self._answers.accept(
            request=request,
            owner_id=principal_id,
            idempotency_key=submission_id,
            idempotency_fingerprint=fingerprint,
            auth_mode=(user.auth_mode if user is not None else "none"),
            acceptor=_WebAnswerAcceptor(
                store=self._store,
                conversation_id=conversation_id,
                title_hint=_auto_title(query),
                create_conversation=create_conversation,
                forked_from_conversation_id=(parent.conversation_id if kind == "fork" else None),
            ),
        )

    async def _resolve_recovery_history(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        targets: Sequence[HistoryProjectionTarget],
    ) -> PriorTurns:
        """Project durable succeeded pairs with bounded physical keyset reads."""
        projector = IncrementalHistoryProjector(targets=targets)
        if not projector.accepts_history:
            return projector.finish()

        before: int | None = None
        rejected_turn_number: int | None = None
        while True:
            batch = await self._store_call(
                self._store.recovery_page(
                    principal_id,
                    conversation_id,
                    page=RecoveryPageRequest(
                        direction="newest",
                        limit=_RECOVERY_BATCH_SIZE,
                        before_turn_number=before,
                    ),
                )
            )
            if not batch.turns:
                break
            for turn in batch.turns:
                pair = _successful_pair(turn)
                if pair is None:
                    continue
                if not projector.offer_newest_pair(*pair):
                    rejected_turn_number = turn.turn_number
                    break
            if rejected_turn_number is not None or not batch.has_more:
                break
            before = batch.turns[-1].turn_number

        if rejected_turn_number is None or not projector.needs_omitted_pairs:
            return projector.finish()

        after: int | None = None
        while projector.needs_omitted_pairs:
            batch = await self._store_call(
                self._store.recovery_page(
                    principal_id,
                    conversation_id,
                    page=RecoveryPageRequest(
                        direction="oldest",
                        limit=_RECOVERY_BATCH_SIZE,
                        after_turn_number=after,
                        upper_turn_number=rejected_turn_number + 1,
                    ),
                )
            )
            if not batch.turns:
                break
            for turn in batch.turns:
                pair = _successful_pair(turn)
                if pair is not None and not projector.offer_oldest_omitted_pair(*pair):
                    break
            if not batch.has_more:
                break
            after = batch.turns[-1].turn_number
        return projector.finish()

    async def _store_call(self, operation: Awaitable[T]) -> T:
        return await operation


def _new_conversation_id(principal_id: str, submission_id: str) -> str:
    """Return the stable server-owned conversation id for a first submission."""
    return str(
        uuid5(
            _NEW_CONVERSATION_NAMESPACE,
            f"{principal_id}\0{submission_id}",
        )
    )


def _empty_head(principal_id: str, conversation_id: str) -> ConversationHead:
    """Supply execution identity while the first row awaits atomic acceptance."""
    now = datetime.datetime.now(datetime.UTC)
    return ConversationHead(
        principal_id=principal_id,
        conversation_id=conversation_id,
        content_revision=0,
        title=None,
        created_at=now,
        updated_at=now,
        agent_session_id=conversation_id,
        agent_lane_id="main",
    )


def _auto_title(query: str) -> str | None:
    return " ".join(query.split())[:120] or None


def _submission(creation: AnswerTurnCreation) -> WebAnswerSubmission:
    return WebAnswerSubmission(
        run=creation.turn.run,
        turn_id=creation.turn.turn_id,
        turn_number=creation.turn.turn_number,
        submission_id=creation.turn.submission_id,
        created_at=creation.turn.created_at,
        conversation=_conversation_summary(creation.summary),
    )


def _prepare_submission(
    *,
    query: str,
    workspaces: Sequence[str],
    seed: SubmissionSeed,
    attachments: Sequence[WebAttachment],
    mode: str | None = None,
    requested_skill: str | None = None,
) -> _PreparedSubmission:
    """Normalize a browser submission without coupling it to a UI history page."""
    request = AnswerRequest(
        query=query,
        workspaces=tuple(workspaces),
        history=(),
        semantic_highlights=True,
        mode=mode,
        requested_skill=requested_skill,
        resources=tuple(
            ResourceInput(
                filename=attachment.filename,
                declared_mime=attachment.mime_type,
                content=attachment.attachment_bytes,
            )
            for attachment in attachments
        ),
        history_resources=tuple(
            AnswerHistoryResource(
                run_id=item.run_id,
                source_ordinal=item.source_ordinal,
                digest=item.digest,
                filename=item.filename,
                mime_type=item.mime_type,
                byte_size=item.byte_size,
            )
            for item in seed.attachments
        ),
        agent_session_id=seed.head.agent_session_id,
        agent_lane_id=seed.head.agent_lane_id,
    )
    return _PreparedSubmission(request=request)


def _successful_pair(
    turn: LinkedTurn,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Project only complete succeeded durable turns into model history."""
    if turn.run.status != "succeeded":
        return None
    request = AnswerRunRequest.from_request(turn.run.request_input())
    return (
        {"role": "user", "content": request.query},
        {
            "role": "assistant",
            "content": str((turn.run.result or {}).get("answer") or ""),
        },
    )


def _web_answer_request_fingerprint(
    *,
    conversation_id: str,
    query: str,
    workspaces: Sequence[str],
    attachments: Sequence[WebAttachment],
    mode: str | None = None,
    requested_skill: str | None = None,
) -> str:
    """Hash only the stable browser submission, before conversation enrichment."""
    return run_request_fingerprint(
        {
            "conversation_id": conversation_id,
            "query": query,
            "mode": mode or "auto",
            "requested_skill": requested_skill or "",
            "workspaces": list(workspaces),
            "attachments": [
                {
                    "digest": attachment.content_sha256,
                    "filename": attachment.filename,
                    "mime_type": attachment.mime_type,
                    "ordinal": attachment.ordinal,
                    "byte_size": attachment.byte_size,
                }
                for attachment in attachments
            ],
        }
    )


def _conversation_summary(row: Mapping[str, Any]) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=str(row["conversation_id"]),
        title=(str(row["title"]) if row.get("title") is not None else None),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        forked_from_conversation_id=(
            str(row["forked_from_conversation_id"])
            if row.get("forked_from_conversation_id") is not None
            else None
        ),
        forked_from_title=(
            str(row["forked_from_title"]) if row.get("forked_from_title") is not None else None
        ),
    )


__all__ = ["WebAnswerSubmission", "WebAttachment", "WebConversationService"]
