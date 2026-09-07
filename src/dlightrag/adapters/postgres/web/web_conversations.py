# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL persistence for principal-scoped Web conversations.

A conversation is navigation and history; it owns no execution state. Each turn
is one row pointing at the durable Answer run that owns the request input, the
uploaded bytes, the streamed events, and the canonical result. The turn is
created inside the run's own creation transaction and keyed by the browser's
submission id, so history exists before the 202 response and no subscriber,
finalizer, or reconnect has to commit it afterwards.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any
from uuid import UUID, uuid4

import asyncpg

from dlightrag.adapters.postgres.core._errors import is_postgres_unavailable
from dlightrag.adapters.postgres.core._migrations import (
    ForeignKeyRequirement,
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)
from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.adapters.postgres.runtime.run_store import (
    RUN_MIGRATION_SCOPE,
    RUN_MIGRATIONS,
    RUN_SCHEMA_TABLES,
    PGRunStore,
    run_columns,
    run_record,
)
from dlightrag.application.answer_runs.envelope import accepted_input_envelope
from dlightrag.application.answer_runs.routing import RoutingAcceptance
from dlightrag.application.runs import RunView
from dlightrag.application.web_conversations import (
    AnswerTurnCreation,
    CarriedAttachment,
    ConversationCursor,
    ConversationHead,
    ConversationHistoryCursor,
    ConversationHistoryPage,
    ConversationHistoryPageRequest,
    ConversationPageRequest,
    ConversationRowPage,
    ConversationSubmissionConflict,
    LinkedTurn,
    RecoveryPageRequest,
    RecoveryTurnBatch,
    SubmissionSeed,
    WebConversationSchemaError,
    WebConversationUnavailableError,
)
from dlightrag.engine.runtime import (
    IdempotencyKeyConflict,
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
    RunDeletion,
    RunSchemaError,
    parse_run_id,
)

_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS web_conversations (
    principal_id TEXT NOT NULL,
    conversation_id UUID NOT NULL,
    agent_session_id UUID NOT NULL,
    agent_lane_id TEXT NOT NULL,
    title TEXT,
    content_revision BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    forked_from_conversation_id UUID,
    PRIMARY KEY (principal_id, conversation_id),
    CHECK (title IS NULL OR char_length(title) BETWEEN 1 AND 120)
)
"""

# The turn carries conversation order and the run link only. Request content,
# answer text, sources, and uploaded bytes all live in the run it references, so
# nothing about one answer is stored twice. Deleting the run deletes the turn:
# pruning a failed or cancelled run removes its visible terminal entry, and
# deleting a conversation deletes its runs in the same transaction.
_CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS web_conversation_turns (
    turn_id UUID NOT NULL,
    principal_id TEXT NOT NULL,
    conversation_id UUID NOT NULL,
    turn_number INTEGER NOT NULL,
    submission_id UUID NOT NULL,
    answer_run_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (principal_id, conversation_id, turn_id),
    UNIQUE (principal_id, conversation_id, turn_number),
    FOREIGN KEY (principal_id, conversation_id)
      REFERENCES web_conversations (principal_id, conversation_id)
      ON DELETE CASCADE,
    FOREIGN KEY (principal_id, answer_run_id)
      REFERENCES dlightrag_runs (owner_id, run_id)
      ON DELETE CASCADE
)
"""

_CREATE_CONVERSATION_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_web_conversations_principal_updated "
    "ON web_conversations (principal_id, updated_at DESC, conversation_id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_web_conversations_updated "
    "ON web_conversations (updated_at, principal_id, conversation_id)",
)

_CREATE_TURN_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_web_conversation_turns_principal_conversation "
    "ON web_conversation_turns (principal_id, conversation_id, turn_number DESC)",
    # The submission id is the owner-wide idempotency key of the run itself, so
    # its turn must be unique in the same namespace rather than per conversation.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_web_conversation_turns_submission "
    "ON web_conversation_turns (principal_id, submission_id)",
    # One turn per run, and the reverse lookup run retention uses to recognize a
    # conversation-linked run.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_web_conversation_turns_run "
    "ON web_conversation_turns (principal_id, answer_run_id)",
)

# One baseline. Fresh installs get the final table; there is no ALTER history.
WEB_CONVERSATION_MIGRATIONS = (
    Migration(
        "web_conversations",
        "Create Web conversations linked to durable Answer runs",
        (
            _CREATE_CONVERSATIONS,
            _CREATE_TURNS,
            *_CREATE_CONVERSATION_INDEXES,
            *_CREATE_TURN_INDEXES,
        ),
    ),
)

WEB_CONVERSATION_SCHEMA_TABLES = (
    TableRequirement(
        name="web_conversations",
        columns=(
            "principal_id",
            "conversation_id",
            "agent_session_id",
            "agent_lane_id",
            "title",
            "content_revision",
            "created_at",
            "updated_at",
            "forked_from_conversation_id",
        ),
        primary_key=("principal_id", "conversation_id"),
        indexes=(
            "idx_web_conversations_principal_updated",
            "idx_web_conversations_updated",
        ),
    ),
    TableRequirement(
        name="web_conversation_turns",
        columns=(
            "turn_id",
            "principal_id",
            "conversation_id",
            "turn_number",
            "submission_id",
            "answer_run_id",
            "created_at",
        ),
        primary_key=("principal_id", "conversation_id", "turn_id"),
        unique=(("principal_id", "conversation_id", "turn_number"),),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("principal_id", "conversation_id"), references="web_conversations"
            ),
            ForeignKeyRequirement(
                columns=("principal_id", "answer_run_id"), references="dlightrag_runs"
            ),
        ),
        indexes=("idx_web_conversation_turns_principal_conversation",),
        unique_indexes=(
            "idx_web_conversation_turns_submission",
            "idx_web_conversation_turns_run",
        ),
    ),
)

_SUMMARY_COLUMNS = """
conversation_id::text AS conversation_id,
agent_session_id::text AS agent_session_id,
agent_lane_id,
title,
content_revision,
created_at,
updated_at,
forked_from_conversation_id::text AS forked_from_conversation_id,
(
    SELECT parent.title FROM web_conversations parent
    WHERE parent.principal_id = web_conversations.principal_id
      AND parent.conversation_id = web_conversations.forked_from_conversation_id
) AS forked_from_title
"""

_CREATE_CONVERSATION = f"""
INSERT INTO web_conversations (
    principal_id, conversation_id, agent_session_id, agent_lane_id)
VALUES ($1, $2::text::uuid, $2::text::uuid, 'main')
RETURNING {_SUMMARY_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_CREATE_CONVERSATION_IF_MISSING = f"""
INSERT INTO web_conversations (
    principal_id, conversation_id, forked_from_conversation_id,
    agent_session_id, agent_lane_id)
VALUES ($1, $2::text::uuid, $3::text::uuid, $4::text::uuid, $5)
ON CONFLICT (principal_id, conversation_id) DO NOTHING
RETURNING {_SUMMARY_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_LIST_CONVERSATIONS_FIRST_PAGE = f"""
SELECT {_SUMMARY_COLUMNS}
FROM web_conversations
WHERE principal_id = $1
ORDER BY updated_at DESC, conversation_id DESC
LIMIT $2
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_LIST_CONVERSATIONS_AFTER = f"""
SELECT {_SUMMARY_COLUMNS}
FROM web_conversations
WHERE principal_id = $1
  AND (updated_at, conversation_id) < ($2::timestamptz, $3::uuid)
ORDER BY updated_at DESC, conversation_id DESC
LIMIT $4
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_RENAME_CONVERSATION = f"""
UPDATE web_conversations
SET title = $3,
    updated_at = NOW()
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
RETURNING {_SUMMARY_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_LOCK_CONVERSATION = f"""
SELECT {_SUMMARY_COLUMNS}
FROM web_conversations
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
FOR UPDATE
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_LOCK_AGENT_SESSION_IF_PRESENT = """
SELECT 1
FROM dlightrag_agent_sessions
WHERE owner_id = $1 AND session_id = $2::text::uuid
FOR UPDATE
"""

_REBASE_EMPTY_CONVERSATION = f"""
UPDATE web_conversations
SET agent_lane_id = 'main',
    content_revision = content_revision + 1,
    updated_at = NOW()
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
RETURNING {_SUMMARY_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_DELETE_BATCH_SIZE = 128

_SELECT_CONVERSATION_RUN_BATCH = """
SELECT answer_run_id::text AS answer_run_id
FROM web_conversation_turns
WHERE principal_id = $1 AND conversation_id = $2::text::uuid
ORDER BY answer_run_id
LIMIT $3
"""

# A delete-all caller waits rather than skipping locked rows: returning while a
# pre-existing owned conversation survives would be a false success. Every
# caller locks the same UUID order, which also keeps two delete-all callers from
# deadlocking. Deleted rows fall out of the next first-page query, so no cursor,
# OFFSET, or transaction-wide identity set is needed.
_LOCK_PRINCIPAL_CONVERSATION_BATCH = """
SELECT conversations.conversation_id::text AS conversation_id,
       conversations.agent_session_id::text AS agent_session_id
FROM web_conversations AS conversations
WHERE conversations.principal_id = $1
ORDER BY conversations.conversation_id
LIMIT $2
FOR UPDATE OF conversations
"""

# This runs as a new READ COMMITTED statement after the conversation locks are
# acquired. It therefore sees a turn committed by a submitter that held a row
# lock before delete-all, unlike a projection in the blocking lock statement.
_SELECT_LINKED_CONVERSATION_BATCH = """
SELECT DISTINCT conversation_id::text AS conversation_id
FROM web_conversation_turns
WHERE principal_id = $1 AND conversation_id = ANY($2::uuid[])
ORDER BY conversation_id
LIMIT $3
"""

_DELETE_CONVERSATION = """
DELETE FROM web_conversations
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
RETURNING agent_session_id::text
"""

_DELETE_CONVERSATION_BATCH = """
WITH deleted AS (
    DELETE FROM web_conversations AS conversations
    WHERE conversations.principal_id = $1
      AND conversations.conversation_id = ANY($2::uuid[])
    RETURNING conversations.conversation_id, conversations.agent_session_id
)
SELECT conversation_id::text AS conversation_id,
       agent_session_id::text AS agent_session_id
FROM deleted
ORDER BY conversation_id
LIMIT $3
"""

_DELETE_AGENT_SESSION_IF_UNREFERENCED = """
DELETE FROM dlightrag_agent_sessions AS sessions
WHERE sessions.owner_id = $1 AND sessions.session_id = $2::text::uuid
  AND NOT EXISTS (
      SELECT 1 FROM dlightrag_answer_run_routing AS routing
      WHERE routing.owner_id = sessions.owner_id
        AND routing.agent_session_id = sessions.session_id
  )
"""

_DELETE_AGENT_SESSION_BATCH_IF_UNREFERENCED = """
DELETE FROM dlightrag_agent_sessions AS sessions
WHERE sessions.owner_id = $1 AND sessions.session_id = ANY($2::uuid[])
  AND NOT EXISTS (
      SELECT 1 FROM dlightrag_answer_run_routing AS routing
      WHERE routing.owner_id = sessions.owner_id
        AND routing.agent_session_id = sessions.session_id
  )
"""

_GET_CONVERSATION = """
SELECT
    principal_id,
    conversation_id::text AS conversation_id,
    agent_session_id::text AS agent_session_id,
    agent_lane_id,
    content_revision,
    title,
    created_at,
    updated_at
FROM web_conversations
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
"""

_TURN_COLUMNS = """
t.turn_id::text AS turn_id,
t.turn_number,
t.submission_id::text AS submission_id,
t.answer_run_id::text AS answer_run_id,
t.conversation_id::text AS turn_conversation_id,
t.created_at AS turn_created_at
"""

_TURN_CONVERSATION_SUMMARY_COLUMNS = """
c.conversation_id::text AS conversation_id,
c.title,
c.content_revision,
c.created_at,
c.updated_at
"""

_GET_TURNS_PAGE = f"""
WITH selected_turns AS (
    SELECT t.*
    FROM web_conversation_turns AS t
    WHERE t.principal_id = $1
      AND t.conversation_id = $2::text::uuid
      AND ($3::integer IS NULL OR t.turn_number < $3)
    ORDER BY t.turn_number DESC
    LIMIT $4
)
SELECT
{_TURN_COLUMNS},
{run_columns("r")}
FROM selected_turns AS t
JOIN dlightrag_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
ORDER BY t.turn_number DESC
"""  # noqa: S608 - interpolates only trusted column-projection constants

_GET_RECOVERY_NEWEST = _GET_TURNS_PAGE

_GET_RECOVERY_OLDEST = f"""
WITH selected_turns AS (
    SELECT t.*
    FROM web_conversation_turns AS t
    WHERE t.principal_id = $1
      AND t.conversation_id = $2::text::uuid
      AND ($3::integer IS NULL OR t.turn_number > $3)
      AND ($4::integer IS NULL OR t.turn_number < $4)
    ORDER BY t.turn_number ASC
    LIMIT $5
)
SELECT
{_TURN_COLUMNS},
{run_columns("r")}
FROM selected_turns AS t
JOIN dlightrag_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
ORDER BY t.turn_number ASC
"""  # noqa: S608 - interpolates only trusted column-projection constants

_GET_CARRIED_ATTACHMENTS = """
SELECT
    t.answer_run_id::text AS run_id,
    (attachment.value->>'ordinal')::integer AS source_ordinal,
    attachment.value->>'digest' AS digest,
    attachment.value->>'filename' AS filename,
    attachment.value->>'mime_type' AS mime_type,
    COALESCE((attachment.value->>'byte_size')::bigint, 0) AS byte_size
FROM web_conversation_turns AS t
JOIN dlightrag_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
CROSS JOIN LATERAL jsonb_array_elements(
    COALESCE(r.prepared_input_json, r.accepted_input_json)->'attachments'
) AS attachment(value)
WHERE t.principal_id = $1
  AND t.conversation_id = $2::text::uuid
  AND r.status = 'succeeded'
  AND attachment.value->>'ordinal' IS NOT NULL
  AND attachment.value->>'digest' IS NOT NULL
  AND attachment.value->>'filename' IS NOT NULL
  AND attachment.value->>'mime_type' IS NOT NULL
ORDER BY t.turn_number DESC, source_ordinal ASC
LIMIT $3
"""

_GET_TURN_BY_SUBMISSION = f"""
SELECT
{_TURN_COLUMNS},
r.request_fingerprint,
{run_columns("r")},
{_TURN_CONVERSATION_SUMMARY_COLUMNS}
FROM web_conversation_turns AS t
JOIN dlightrag_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
JOIN web_conversations AS c
  ON c.principal_id = t.principal_id
 AND c.conversation_id = t.conversation_id
WHERE t.principal_id = $1
  AND t.submission_id = $2::text::uuid
"""  # noqa: S608 - interpolates only trusted column-projection constants

_GET_TURN_BY_RUN = f"""
SELECT
{_TURN_COLUMNS},
{run_columns("r")}
FROM web_conversation_turns AS t
JOIN dlightrag_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
WHERE t.principal_id = $1
  AND t.answer_run_id = $2::text::uuid
"""  # noqa: S608 - interpolates only trusted column-projection constants

_TOUCH_CONVERSATION = f"""
UPDATE web_conversations
SET content_revision = content_revision + 1,
    title = COALESCE(title, $3),
    updated_at = NOW()
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
RETURNING {_SUMMARY_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_INSERT_TURN = """
INSERT INTO web_conversation_turns (
    turn_id,
    principal_id,
    conversation_id,
    turn_number,
    submission_id,
    answer_run_id
)
VALUES (
    $1::text::uuid,
    $2,
    $3::text::uuid,
    $4,
    $5::text::uuid,
    $6::text::uuid
)
RETURNING turn_id::text AS turn_id, turn_number
"""

_SELECT_EMPTY_CONVERSATIONS = """
SELECT principal_id, conversation_id, agent_session_id
FROM web_conversations AS conversations
WHERE NOT EXISTS (
    SELECT 1
    FROM web_conversation_turns AS turns
    WHERE turns.principal_id = conversations.principal_id
      AND turns.conversation_id = conversations.conversation_id
)
ORDER BY updated_at, principal_id, conversation_id
LIMIT $1
FOR UPDATE SKIP LOCKED
"""

_DELETE_CONVERSATIONS = """
WITH deleted AS (
    DELETE FROM web_conversations AS conversations
    WHERE (conversations.principal_id, conversations.conversation_id)
          IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
    RETURNING 1
)
SELECT count(*)::int AS count FROM deleted
"""


#: The two indexes that make one submission id one turn in the owner's namespace.
#: Only these mean "this key is already accepted"; any other violation is a bug.
_SUBMISSION_KEY_INDEXES = frozenset(
    {"idx_web_conversation_turns_submission", "idx_web_conversation_turns_run"}
)


def _row_dict(row: Any) -> dict[str, Any]:
    return dict(row)


def _conversation_head(row: Any) -> ConversationHead:
    return ConversationHead(
        principal_id=str(row["principal_id"]),
        conversation_id=str(row["conversation_id"]),
        content_revision=int(row["content_revision"]),
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        agent_session_id=str(row["agent_session_id"]),
        agent_lane_id=str(row["agent_lane_id"]),
    )


def _linked_turn(row: Any) -> LinkedTurn:
    return LinkedTurn(
        turn_id=str(row["turn_id"]),
        turn_number=int(row["turn_number"]),
        submission_id=str(row["submission_id"]),
        created_at=row["turn_created_at"],
        run=RunView.from_runtime(run_record(row)),
        conversation_id=str(row["turn_conversation_id"]),
    )


class PGWebConversationStore(PostgresOperationRunner):
    """Durable PostgreSQL store for server-owned Web conversations."""

    def __init__(
        self,
        *,
        pool: ConnectionPool | None = None,
        run_store: PGRunStore | None = None,
    ) -> None:
        super().__init__(pool=pool)
        self._run_store = run_store or PGRunStore(pool=pool)
        self._initialized = False

    async def _run_read[T](self, operation: Callable[[Any], Awaitable[T]]) -> T:
        try:
            return await self._run(operation)
        except Exception as exc:
            if is_postgres_unavailable(exc):
                raise WebConversationUnavailableError from exc
            raise

    async def _run_write[T](self, operation: Callable[[Any], Awaitable[T]]) -> T:
        try:
            return await self._run_once(operation)
        except Exception as exc:
            if is_postgres_unavailable(exc):
                raise WebConversationUnavailableError from exc
            raise

    async def initialize(self, *, validate_only: bool = False) -> None:
        """Create the Web conversation schema, or validate it (reader).

        A turn's foreign key targets the durable Answer run table, so the run
        schema is established first in the same connection.
        """
        if self._initialized:
            return

        async def _operation(conn: Any) -> None:
            if validate_only:
                await verify_migrations(
                    conn,
                    scope=RUN_MIGRATION_SCOPE,
                    migrations=RUN_MIGRATIONS,
                    tables=RUN_SCHEMA_TABLES,
                    schema_error=RunSchemaError,
                )
                await verify_migrations(
                    conn,
                    scope="web_conversations",
                    migrations=WEB_CONVERSATION_MIGRATIONS,
                    tables=WEB_CONVERSATION_SCHEMA_TABLES,
                    schema_error=WebConversationSchemaError,
                )
                return
            await apply_migrations(
                conn,
                scope=RUN_MIGRATION_SCOPE,
                migrations=RUN_MIGRATIONS,
                schema_error=RunSchemaError,
            )
            await apply_migrations(
                conn,
                scope="web_conversations",
                migrations=WEB_CONVERSATION_MIGRATIONS,
                schema_error=WebConversationSchemaError,
            )

        await self._run_write(_operation)
        self._initialized = True

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def create_conversation(self, principal_id: str) -> dict[str, Any]:
        """Create and return an empty conversation owned by ``principal_id``."""
        await self._ensure_initialized()
        conversation_id = str(uuid4())

        async def _operation(conn: Any) -> dict[str, Any]:
            row = await conn.fetchrow(_CREATE_CONVERSATION, principal_id, conversation_id)
            if row is None:
                raise RuntimeError("conversation insert returned no row")
            return _row_dict(row)

        return await self._run_write(_operation)

    async def list_conversations(
        self,
        principal_id: str,
        *,
        page: ConversationPageRequest,
    ) -> ConversationRowPage:
        """Fetch one hard-bounded principal page through the covering order index."""
        if not isinstance(page, ConversationPageRequest):
            raise ValueError("conversation page request is required")
        validated_page = ConversationPageRequest(limit=page.limit, cursor=page.cursor)
        cursor = validated_page.cursor
        if cursor is not None and not isinstance(cursor, ConversationCursor):
            raise ValueError("conversation cursor must contain paired ordering fields")
        await self._ensure_initialized()
        fetch_limit = validated_page.limit + 1

        async def _select(conn: Any) -> ConversationRowPage:
            if cursor is None:
                rows = await conn.fetch(
                    _LIST_CONVERSATIONS_FIRST_PAGE,
                    principal_id,
                    fetch_limit,
                )
            else:
                rows = await conn.fetch(
                    _LIST_CONVERSATIONS_AFTER,
                    principal_id,
                    cursor.updated_at,
                    cursor.conversation_id,
                    fetch_limit,
                )
            fetched_rows = len(rows)
            return ConversationRowPage(
                items=tuple(_row_dict(row) for row in rows[: validated_page.limit]),
                has_more=fetched_rows > validated_page.limit,
                fetched_rows=fetched_rows,
            )

        return await self._run_read(_select)

    async def rename_conversation(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        title: str,
    ) -> dict[str, Any] | None:
        """Rename an owned conversation."""
        await self._ensure_initialized()
        normalized_title = title.strip()
        if not normalized_title or len(normalized_title) > 120:
            raise ValueError("conversation title must contain 1 to 120 characters")

        async def _operation(conn: Any) -> dict[str, Any] | None:
            row = await conn.fetchrow(
                _RENAME_CONVERSATION,
                principal_id,
                conversation_id,
                normalized_title,
            )
            return _row_dict(row) if row is not None else None

        return await self._run_write(_operation)

    @staticmethod
    def _canonical_delete_id(value: Any, *, kind: str) -> str:
        if not isinstance(value, str):
            raise RuntimeError(f"{kind} deletion row did not contain a text UUID")
        parsed = parse_run_id(value)
        if parsed is None or str(parsed) != value:
            raise RuntimeError(f"{kind} deletion row contained a non-canonical UUID")
        return value

    async def _delete_linked_runs_in_batches(
        self,
        conn: Any,
        *,
        principal_id: str,
        conversation_id: str,
    ) -> None:
        """Delete one locked conversation's run links with bounded metadata.

        Run deletion cascades each selected turn, so repeatedly reading the
        first ordered page advances without OFFSET. Re-reading a selected id is
        a fail-closed progress violation: it catches a malformed or partial run
        store result before the conversation row could cascade a leftover turn.
        """
        previous_ids: frozenset[str] = frozenset()
        while True:
            rows = await conn.fetch(
                _SELECT_CONVERSATION_RUN_BATCH,
                principal_id,
                conversation_id,
                _DELETE_BATCH_SIZE,
            )
            if len(rows) > _DELETE_BATCH_SIZE:
                raise RuntimeError("conversation run deletion exceeded its batch bound")
            run_ids = [self._canonical_delete_id(row["answer_run_id"], kind="run") for row in rows]
            if not run_ids:
                return
            if run_ids != sorted(run_ids) or len(set(run_ids)) != len(run_ids):
                raise RuntimeError("conversation run deletion returned unstable identities")
            if previous_ids.intersection(run_ids):
                raise RuntimeError("conversation run deletion made no forward progress")

            deletion = await self._run_store.delete_runs_in(
                conn,
                owner_id=principal_id,
                run_ids=run_ids,
            )
            if (
                not isinstance(deletion, RunDeletion)
                or type(deletion.runs) is not int
                or type(deletion.artifacts) is not int
                or not 0 <= deletion.runs <= len(run_ids)
                or deletion.artifacts < 0
            ):
                raise RuntimeError("run store returned an invalid deletion result")
            previous_ids = frozenset(run_ids)

    async def delete_conversation(
        self,
        principal_id: str,
        conversation_id: str,
    ) -> bool:
        """Delete one owned conversation, its turns, and the runs they linked.

        The conversation is locked before its run ids are read, so a submission
        committing its own turn concurrently is either already visible here or
        still waiting behind this transaction; neither leaves an orphan run.
        Deleting the runs in the same transaction stops any lease-fenced worker
        from appending to state the conversation no longer owns, cascades the
        runs' events, turns, and artifact references, and releases blobs no
        surviving run still references. Runs are deleted before the conversation
        so this path takes the run lock before the turn lock, exactly as run
        retention does; the reverse order deadlocks against a concurrent prune.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                summary_row = await conn.fetchrow(_LOCK_CONVERSATION, principal_id, conversation_id)
                if summary_row is None:
                    return False
                agent_session_id = self._canonical_delete_id(
                    summary_row["agent_session_id"], kind="Agent Session"
                )
                await self._delete_linked_runs_in_batches(
                    conn,
                    principal_id=principal_id,
                    conversation_id=conversation_id,
                )
                deleted_session_id = await conn.fetchval(
                    _DELETE_CONVERSATION, principal_id, conversation_id
                )
                if str(deleted_session_id) != agent_session_id:
                    raise RuntimeError("locked conversation was not deleted exactly once")
                await conn.execute(
                    _DELETE_AGENT_SESSION_IF_UNREFERENCED,
                    principal_id,
                    agent_session_id,
                )
                return True

        return await self._run_write(_operation)

    async def delete_all_conversations(self, principal_id: str) -> int:
        """Delete every owned conversation in bounded locked batches, atomically.

        The outer transaction deliberately spans every batch. A submission to
        an existing conversation takes the same conversation row lock, so it is
        either included before that row is drained or observes its deletion;
        no accepted turn is orphaned. Concurrent owner-wide deleters wait on the
        first UUID rather than skipping it and all acquire UUIDs in one order.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> int:
            async with conn.transaction():
                deleted_count = 0
                while True:
                    rows = await conn.fetch(
                        _LOCK_PRINCIPAL_CONVERSATION_BATCH,
                        principal_id,
                        _DELETE_BATCH_SIZE,
                    )
                    if len(rows) > _DELETE_BATCH_SIZE:
                        raise RuntimeError("conversation deletion exceeded its batch bound")
                    conversations = [
                        (
                            self._canonical_delete_id(row["conversation_id"], kind="conversation"),
                            self._canonical_delete_id(
                                row["agent_session_id"], kind="Agent Session"
                            ),
                        )
                        for row in rows
                    ]
                    if not conversations:
                        return deleted_count
                    conversation_ids = [conversation_id for conversation_id, _ in conversations]
                    if conversation_ids != sorted(conversation_ids) or len(
                        set(conversation_ids)
                    ) != len(conversation_ids):
                        raise RuntimeError("conversation deletion returned unstable identities")
                    linked_rows = await conn.fetch(
                        _SELECT_LINKED_CONVERSATION_BATCH,
                        principal_id,
                        conversation_ids,
                        _DELETE_BATCH_SIZE,
                    )
                    if len(linked_rows) > _DELETE_BATCH_SIZE:
                        raise RuntimeError("linked conversation scan exceeded its batch bound")
                    linked_conversation_ids = [
                        self._canonical_delete_id(row["conversation_id"], kind="conversation")
                        for row in linked_rows
                    ]
                    if (
                        linked_conversation_ids != sorted(linked_conversation_ids)
                        or len(set(linked_conversation_ids)) != len(linked_conversation_ids)
                        or not set(linked_conversation_ids).issubset(conversation_ids)
                    ):
                        raise RuntimeError("linked conversation scan returned unstable identities")

                    for conversation_id in linked_conversation_ids:
                        await self._delete_linked_runs_in_batches(
                            conn,
                            principal_id=principal_id,
                            conversation_id=conversation_id,
                        )
                    deleted_rows = await conn.fetch(
                        _DELETE_CONVERSATION_BATCH,
                        principal_id,
                        conversation_ids,
                        _DELETE_BATCH_SIZE,
                    )
                    if len(deleted_rows) > _DELETE_BATCH_SIZE:
                        raise RuntimeError("conversation deletion exceeded its batch bound")
                    deleted_conversations = [
                        (
                            self._canonical_delete_id(row["conversation_id"], kind="conversation"),
                            self._canonical_delete_id(
                                row["agent_session_id"], kind="Agent Session"
                            ),
                        )
                        for row in deleted_rows
                    ]
                    expected_conversations = conversations
                    if deleted_conversations != expected_conversations:
                        raise RuntimeError("locked conversations were not deleted exactly once")
                    await conn.execute(
                        _DELETE_AGENT_SESSION_BATCH_IF_UNREFERENCED,
                        principal_id,
                        [session_id for _, session_id in deleted_conversations],
                    )
                    deleted_count += len(deleted_conversations)

        return await self._run_write(_operation)

    async def history_page(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        page: ConversationHistoryPageRequest,
    ) -> ConversationHistoryPage | None:
        """Read one chronological limit+1 keyset page under a coherent head."""
        validated = ConversationHistoryPageRequest(limit=page.limit, cursor=page.cursor)
        if (
            validated.cursor is not None
            and str(validated.cursor.conversation_id) != conversation_id
        ):
            raise ValueError("conversation history cursor belongs to another conversation")
        await self._ensure_initialized()

        async def _operation(conn: Any) -> ConversationHistoryPage | None:
            async with conn.transaction(isolation="repeatable_read", readonly=True):
                row = await conn.fetchrow(_GET_CONVERSATION, principal_id, conversation_id)
                if row is None:
                    return None
                turn_rows = await conn.fetch(
                    _GET_TURNS_PAGE,
                    principal_id,
                    conversation_id,
                    (validated.cursor.before_turn_number if validated.cursor is not None else None),
                    validated.limit + 1,
                )
                fetched_rows = len(turn_rows)
                retained = turn_rows[: validated.limit]
                next_cursor = None
                if fetched_rows > validated.limit:
                    if not retained:
                        raise RuntimeError("history page reported more rows after an empty page")
                    next_cursor = ConversationHistoryCursor(
                        conversation_id=validated.cursor.conversation_id
                        if validated.cursor is not None
                        else UUID(conversation_id),
                        before_turn_number=int(retained[-1]["turn_number"]),
                    )
                return ConversationHistoryPage(
                    conversation=_conversation_head(row),
                    turns=tuple(_linked_turn(turn) for turn in reversed(retained)),
                    next_cursor=next_cursor,
                    fetched_rows=fetched_rows,
                )

        return await self._run_read(_operation)

    async def submission_seed(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        attachment_limit: int,
    ) -> SubmissionSeed | None:
        """Read the execution mapping and newest successful attachment metadata."""
        if isinstance(attachment_limit, bool) or attachment_limit < 0:
            raise ValueError("attachment limit must be non-negative")
        await self._ensure_initialized()

        async def _operation(conn: Any) -> SubmissionSeed | None:
            async with conn.transaction(isolation="repeatable_read", readonly=True):
                row = await conn.fetchrow(_GET_CONVERSATION, principal_id, conversation_id)
                if row is None:
                    return None
                attachment_rows = (
                    await conn.fetch(
                        _GET_CARRIED_ATTACHMENTS,
                        principal_id,
                        conversation_id,
                        attachment_limit,
                    )
                    if attachment_limit
                    else ()
                )
                attachments = tuple(
                    CarriedAttachment(
                        run_id=str(item["run_id"]),
                        source_ordinal=int(item["source_ordinal"]),
                        digest=str(item["digest"]),
                        filename=str(item["filename"]),
                        mime_type=str(item["mime_type"]),
                        byte_size=int(item["byte_size"]),
                    )
                    for item in attachment_rows
                )
                return SubmissionSeed(
                    head=_conversation_head(row),
                    attachments=attachments,
                )

        return await self._run_read(_operation)

    async def recovery_page(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        page: RecoveryPageRequest,
    ) -> RecoveryTurnBatch:
        """Read one bounded physical recovery page across every run status."""
        validated = RecoveryPageRequest(
            direction=page.direction,
            limit=page.limit,
            before_turn_number=page.before_turn_number,
            after_turn_number=page.after_turn_number,
            upper_turn_number=page.upper_turn_number,
        )
        await self._ensure_initialized()

        async def _operation(conn: Any) -> RecoveryTurnBatch:
            if validated.direction == "newest":
                rows = await conn.fetch(
                    _GET_RECOVERY_NEWEST,
                    principal_id,
                    conversation_id,
                    validated.before_turn_number,
                    validated.limit + 1,
                )
            else:
                rows = await conn.fetch(
                    _GET_RECOVERY_OLDEST,
                    principal_id,
                    conversation_id,
                    validated.after_turn_number,
                    validated.upper_turn_number,
                    validated.limit + 1,
                )
            fetched_rows = len(rows)
            return RecoveryTurnBatch(
                turns=tuple(_linked_turn(row) for row in rows[: validated.limit]),
                has_more=fetched_rows > validated.limit,
                fetched_rows=fetched_rows,
            )

        return await self._run_read(_operation)

    async def find_turn_by_run(self, principal_id: str, run_id: str) -> LinkedTurn | None:
        """Return the conversation entry one owned run belongs to, if any."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> LinkedTurn | None:
            row = await conn.fetchrow(_GET_TURN_BY_RUN, principal_id, run_id)
            return _linked_turn(row) if row is not None else None

        return await self._run_read(_operation)

    async def find_answer_turn_by_submission(
        self, principal_id: str, submission_id: str
    ) -> AnswerTurnCreation | None:
        """Return one owner-scoped accepted submission without needing its conversation."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> AnswerTurnCreation | None:
            row = await conn.fetchrow(_GET_TURN_BY_SUBMISSION, principal_id, submission_id)
            if row is None:
                return None
            return AnswerTurnCreation(
                turn=_linked_turn(row),
                summary=_row_dict(row),
                replayed=True,
            )

        return await self._run_read(_operation)

    async def create_answer_turn(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        envelope: PreparedRunEnvelope,
        run_id: str,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        title_hint: str | None,
        routing: RoutingAcceptance | None = None,
        create_conversation: bool = False,
        forked_from_conversation_id: str | None = None,
    ) -> AnswerTurnCreation | None:
        """Accept one submission as a run plus its conversation entry, atomically.

        The owned conversation is locked, or inserted when this is its first
        submission. The run, uploaded bytes, and linked turn are committed in
        that same transaction. Replaying the same submission returns the
        authoritative run and turn; reusing it for a different conversation or
        different normalized input is a conflict. ``None`` means a required
        existing conversation does not exist for this principal.
        """
        await self._ensure_initialized()
        turn_id = str(uuid4())
        request = envelope.payload
        fingerprint = envelope.request_fingerprint

        async def _operation(conn: Any) -> AnswerTurnCreation | None:
            async with conn.transaction():
                summary_row = await conn.fetchrow(_LOCK_CONVERSATION, principal_id, conversation_id)
                if summary_row is None and create_conversation:
                    summary_row = await conn.fetchrow(
                        _CREATE_CONVERSATION_IF_MISSING,
                        principal_id,
                        conversation_id,
                        forked_from_conversation_id,
                        str(request["agent_session_id"]),
                        str(request.get("agent_lane_id") or "main"),
                    )
                    if summary_row is None:
                        summary_row = await conn.fetchrow(
                            _LOCK_CONVERSATION,
                            principal_id,
                            conversation_id,
                        )
                if summary_row is None:
                    return None
                if str(summary_row["agent_session_id"]) != str(
                    request.get("agent_session_id") or ""
                ) or str(summary_row["agent_lane_id"]) != str(
                    request.get("agent_lane_id") or "main"
                ):
                    raise ConversationSubmissionConflict(
                        "conversation Agent Session/Lane mapping changed"
                    )
                existing = await conn.fetchrow(_GET_TURN_BY_SUBMISSION, principal_id, submission_id)
                if existing is not None:
                    if str(existing["turn_conversation_id"]) != str(conversation_id):
                        raise ConversationSubmissionConflict(
                            "submission id was reused in a different conversation"
                        )
                    if str(existing["request_fingerprint"]) != fingerprint:
                        raise ConversationSubmissionConflict(
                            "submission id was reused with different request input"
                        )
                    return AnswerTurnCreation(
                        turn=_linked_turn(existing),
                        summary=_row_dict(existing),
                        replayed=True,
                    )
                accepted_request = request
                accepted_routing = routing
                session_exists = await conn.fetchval(
                    _LOCK_AGENT_SESSION_IF_PRESENT,
                    principal_id,
                    str(summary_row["agent_session_id"]),
                )
                if session_exists is None:
                    accepted_request = {
                        **request,
                        "agent_lane_id": "main",
                        "source_lane_id": None,
                    }
                    accepted_routing = replace(
                        routing or RoutingAcceptance.fallback(request),
                        agent_lane_id="main",
                        source_lane_id=None,
                    )
                    if str(summary_row["agent_lane_id"]) != "main":
                        rebased = await conn.fetchrow(
                            _REBASE_EMPTY_CONVERSATION,
                            principal_id,
                            conversation_id,
                        )
                        if rebased is None:
                            raise ConversationSubmissionConflict(
                                "conversation Agent Session/Lane mapping changed"
                            )
                        summary_row = rebased
                try:
                    creation = await self._run_store.create_run_in(
                        conn,
                        envelope=replace(
                            envelope,
                            payload=accepted_request,
                            accepted_input=accepted_input_envelope(accepted_request),
                        ),
                        run_id=run_id,
                        artifacts=artifacts,
                        references=references,
                        routing=accepted_routing,
                    )
                except IdempotencyKeyConflict as exc:
                    accepted = await conn.fetchrow(
                        _GET_TURN_BY_SUBMISSION,
                        principal_id,
                        submission_id,
                    )
                    if accepted is not None and str(accepted["turn_conversation_id"]) != str(
                        conversation_id
                    ):
                        raise ConversationSubmissionConflict(
                            "submission id was reused in a different conversation"
                        ) from exc
                    raise
                touched = await conn.fetchrow(
                    _TOUCH_CONVERSATION, principal_id, conversation_id, title_hint
                )
                if touched is None:
                    raise RuntimeError("conversation touch returned no row")
                turn_number = int(touched["content_revision"])
                try:
                    turn_row = await conn.fetchrow(
                        _INSERT_TURN,
                        turn_id,
                        principal_id,
                        conversation_id,
                        turn_number,
                        submission_id,
                        creation.run.run_id,
                    )
                except asyncpg.UniqueViolationError as exc:
                    # A concurrent submission committed this key first: both saw
                    # no turn and both replayed the one run the key owns, so the
                    # loser is a reuse conflict rather than a server fault. The
                    # transaction unwinds, leaving that run and its bytes alone.
                    if getattr(exc, "constraint_name", None) not in _SUBMISSION_KEY_INDEXES:
                        raise
                    raise ConversationSubmissionConflict(
                        "submission id was accepted concurrently for another conversation"
                    ) from None
                if turn_row is None:
                    raise RuntimeError("conversation turn insert returned no row")
                if int(turn_row["turn_number"]) != turn_number:
                    raise RuntimeError("conversation turn insert changed its allocated number")
                return AnswerTurnCreation(
                    turn=LinkedTurn(
                        turn_id=str(turn_row["turn_id"]),
                        turn_number=turn_number,
                        submission_id=submission_id,
                        created_at=creation.run.created_at,
                        run=RunView.from_runtime(creation.run),
                        conversation_id=conversation_id,
                    ),
                    summary=_row_dict(touched if touched is not None else summary_row),
                    replayed=creation.replayed,
                )

        return await self._run_write(_operation)

    async def replay_answer_turn(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        idempotency_fingerprint: str,
    ) -> AnswerTurnCreation | None:
        """Return an accepted browser submission before resolved input is rebuilt."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> AnswerTurnCreation | None:
            existing = await conn.fetchrow(_GET_TURN_BY_SUBMISSION, principal_id, submission_id)
            if existing is None:
                return None
            if str(existing["turn_conversation_id"]) != str(conversation_id):
                raise ConversationSubmissionConflict(
                    "submission id was reused in a different conversation"
                )
            if str(existing["request_fingerprint"]) != idempotency_fingerprint:
                raise ConversationSubmissionConflict(
                    "submission id was reused with different request input"
                )
            return AnswerTurnCreation(
                turn=_linked_turn(existing),
                summary=_row_dict(existing),
                replayed=True,
            )

        return await self._run_read(_operation)

    async def prune_empty_conversations(self, *, batch_size: int = 500) -> int:
        """Delete one skip-locked batch of conversations with no turns left.

        Turns live and die with their Answer runs: run retention deletes a
        terminal run after its retention floor and the turn cascade empties the
        conversation. A conversation row with no turns carries no content, so
        this sweep reclaims it.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        await self._ensure_initialized()

        async def _operation(conn: Any) -> int:
            async with conn.transaction():
                rows = await conn.fetch(_SELECT_EMPTY_CONVERSATIONS, batch_size)
                if not rows:
                    return 0
                principals = [str(row["principal_id"]) for row in rows]
                conversation_ids = [row["conversation_id"] for row in rows]
                session_mappings = {
                    (str(row["principal_id"]), str(row["agent_session_id"])) for row in rows
                }
                deleted = await conn.fetchrow(_DELETE_CONVERSATIONS, principals, conversation_ids)
                for principal, session_id in session_mappings:
                    await conn.execute(
                        _DELETE_AGENT_SESSION_IF_UNREFERENCED,
                        principal,
                        session_id,
                    )
                return int(deleted["count"]) if deleted is not None else 0

        return await self._run_write(_operation)


__all__ = [
    "WEB_CONVERSATION_MIGRATIONS",
    "WEB_CONVERSATION_SCHEMA_TABLES",
    "AnswerTurnCreation",
    "ConversationHead",
    "ConversationHistoryPage",
    "ConversationSubmissionConflict",
    "LinkedTurn",
    "PGWebConversationStore",
]
