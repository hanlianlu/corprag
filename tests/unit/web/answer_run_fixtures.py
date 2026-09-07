# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared builders for durable Answer run state in Web tests."""

import datetime
from dataclasses import asdict, replace
from typing import Any
from uuid import uuid7

from dlightrag.application.answer_runs import (
    AnswerInputArtifact,
    AnswerRequest,
    AnswerRunAcceptor,
    AnswerService,
)
from dlightrag.application.answer_runs.envelope import accepted_input_envelope
from dlightrag.application.runs import RunStatus, RunView
from dlightrag.application.web_conversations import (
    AnswerTurnCreation,
    ConversationSummary,
    LinkedTurn,
    WebAnswerSubmission,
)
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.tokens import estimate_messages_tokens, estimate_tokens
from dlightrag.engine.answer.history import HistoryProjectionTarget
from dlightrag.engine.runtime import (
    PendingArtifact,
    PreparedRunEnvelope,
    RunAccessScope,
)

NOW = datetime.datetime(2026, 8, 12, tzinfo=datetime.UTC)
RUN_ID = "019893f4-0000-7000-8000-000000000001"
TURN_ID = "00000000-0000-0000-0000-000000000010"
SUBMISSION_ID = "00000000-0000-0000-0000-0000000000aa"


class FakeAnswers(AnswerService):
    """The durable Answer operations one Web conversation depends on."""

    def __init__(self, artifacts: dict[tuple[str, int], AnswerInputArtifact] | None = None) -> None:
        self.artifacts = dict(artifacts or {})
        self.prepared: list[AnswerRequest] = []
        self.reads: list[tuple[str, str, int]] = []

    async def accept[T](
        self,
        *,
        request: AnswerRequest,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        acceptor: AnswerRunAcceptor[T],
        auth_mode: str = "none",
        history_resolver: Any | None = None,
    ) -> T | None:
        del auth_mode
        replay = await acceptor.replay_run(
            owner_id=owner_id,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=idempotency_fingerprint,
            run_kind="answer",
        )
        if replay is not None:
            return replay
        if history_resolver is not None:
            profile = ModelProfile(
                context_window_tokens=1_000_000,
                max_input_tokens=None,
                max_output_tokens=1_000,
                supports_images=True,
            )

            def measure_history(messages: list[dict[str, Any]], projected_summary: str = "") -> int:
                return estimate_messages_tokens(messages) + estimate_tokens(projected_summary)

            projected = await history_resolver(
                (HistoryProjectionTarget("fake", profile, measure_history),)
            )
            request = replace(
                request,
                history=tuple(projected.messages),
                episodic_summary=projected.episodic_summary,
            )
        self.prepared.append(request)
        prepared_input = {
            "query": request.query,
            "workspaces": list(request.workspaces),
            "agent_session_id": request.agent_session_id,
            "agent_lane_id": request.agent_lane_id,
            "source_lane_id": request.source_lane_id,
        }
        run_id = str(uuid7())
        return await acceptor.create_run(
            envelope=PreparedRunEnvelope(
                run_kind="answer",
                lane="query",
                submitted_by=owner_id,
                access_scope=RunAccessScope(kind="owner", scope_id=owner_id),
                submission_key=idempotency_key or run_id,
                request_fingerprint=idempotency_fingerprint,
                payload=prepared_input,
                accepted_input=accepted_input_envelope(prepared_input),
                retention_seconds=365 * 24 * 60 * 60,
            ),
            run_id=run_id,
            artifacts=tuple(
                PendingArtifact(content=resource.content)
                for resource in request.resources
                if resource.content is not None
            ),
            references=(),
        )

    async def read_input_artifact(
        self, *, owner_id: str, run_id: str, ordinal: int
    ) -> AnswerInputArtifact | None:
        self.reads.append((owner_id, run_id, ordinal))
        return self.artifacts.get((run_id, ordinal))


def input_artifact(
    *,
    content: bytes,
    ordinal: int = 0,
    filename: str = "notes.txt",
    mime_type: str = "text/plain",
    digest: str = "a" * 64,
    reference_kind: str = "current_attachment",
) -> AnswerInputArtifact:
    return AnswerInputArtifact(
        reference_kind=reference_kind,  # type: ignore[arg-type]
        ordinal=ordinal,
        filename=filename,
        mime_type=mime_type,
        digest=digest,
        content=content,
    )


def run_request(**overrides: Any) -> dict[str, Any]:
    return {
        "query": "What changed?",
        "workspaces": ["default"],
        "history": [],
        "top_k": None,
        "chunk_top_k": None,
        "filters": None,
        "semantic_highlights": True,
        "links": [],
        "attachments": [],
        "history_attachments": [],
        "pinned_models": [
            {
                "role": "query",
                "fingerprint": {
                    "provider": "openai",
                    "model": "test-model",
                    "endpoint_fingerprint": None,
                },
                "profile": {
                    "context_window_tokens": 1_000_000,
                    "max_input_tokens": None,
                    "max_output_tokens": 128_000,
                    "supports_images": True,
                    "reasoning": None,
                },
            }
        ],
        "context_policy_revision": "m1-v1",
        "model_catalog_revision": "2026-08-14",
        "idempotency_fingerprint": "test-public-request-hash",
        "image_descriptions": [],
        **overrides,
    }


def answer_run(
    *,
    status: RunStatus = "queued",
    request: dict[str, Any] | None = None,
    accepted: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    cancel_requested_at: datetime.datetime | None = None,
    error_kind: str | None = None,
    error_message: str | None = None,
    events_trimmed_at: datetime.datetime | None = None,
    run_id: str = RUN_ID,
    owner_id: str = "owner-1",
) -> RunView:
    terminal = status in ("succeeded", "failed", "cancelled")
    return RunView(
        run_id=run_id,
        run_kind="answer",
        lane="query",
        submitted_by=owner_id,
        access_scope_kind="owner",
        access_scope_id=owner_id,
        status=status,
        phase=None,
        durable_progress_version=0,
        next_event_sequence=1,
        events_trimmed_at=events_trimmed_at,
        cancel_requested=cancel_requested_at is not None,
        result=result,
        error_kind=error_kind,
        error_message=error_message,
        created_at=NOW,
        started_at=None,
        finished_at=NOW if terminal else None,
        request=accepted if accepted is not None else request or run_request(),
    )


def linked_turn(
    run: RunView | None = None, *, turn_number: int = 1, conversation_id: str = ""
) -> LinkedTurn:
    return LinkedTurn(
        turn_id=TURN_ID,
        turn_number=turn_number,
        submission_id=SUBMISSION_ID,
        created_at=NOW,
        run=run if run is not None else answer_run(),
        conversation_id=conversation_id,
    )


def conversation_summary(conversation_id: str) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=conversation_id,
        title=None,
        created_at=NOW,
        updated_at=NOW,
    )


def answer_turn_creation(
    *,
    conversation_id: str,
    run: RunView | None = None,
    replayed: bool = False,
) -> AnswerTurnCreation:
    return AnswerTurnCreation(
        turn=linked_turn(run, conversation_id=conversation_id),
        summary=asdict(conversation_summary(conversation_id)),
        replayed=replayed,
    )


def web_answer_submission(
    *,
    conversation_id: str,
    run: RunView | None = None,
) -> WebAnswerSubmission:
    return WebAnswerSubmission(
        run=run if run is not None else answer_run(),
        turn_id=TURN_ID,
        turn_number=1,
        conversation=conversation_summary(conversation_id),
        submission_id=SUBMISSION_ID,
        created_at=NOW,
    )


def stored_result(answer: str = "Revenue increased [1].") -> dict[str, Any]:
    return {
        "answer": answer,
        "contexts": {"chunks": []},
        "sources": [
            {
                "id": "1",
                "title": "Report",
                "type": "document",
                "source_uri": "local://report.pdf",
                "workspace": "default",
                "document_id": "report",
                "chunks": [],
            }
        ],
        "evidence_images": [],
        "artifacts": [],
        "artifact_outcome": {"status": "complete", "issues": []},
        "trace": {},
        "usage": {"usage_details": {"total_tokens": 42}},
        "evidence": {"chunks": 1, "entities": 0, "relationships": 0, "sources": 1},
        "image_descriptions": [],
    }
