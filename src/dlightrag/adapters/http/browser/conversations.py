# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser presentation for Application-owned Web Conversation records."""

from collections.abc import Mapping
from typing import Any

from dlightrag.adapters.http.browser.conversation_models import (
    ConversationAttachmentReference,
    ConversationHistory,
    ConversationSummary,
    ConversationTurn,
)
from dlightrag.adapters.http.browser.presentation import build_answer_presentation
from dlightrag.application.web_conversations import (
    ConversationHead,
    ConversationHistoryPage,
    LinkedTurn,
)
from dlightrag.application.web_conversations import (
    ConversationSummary as ApplicationConversationSummary,
)
from dlightrag.engine.answer.citations.sources import SourceDownloadLinkBuilder
from dlightrag.engine.answer.execution.input import AnswerRunRequest, AttachmentReference
from dlightrag.engine.answer.results import project_answer_result

WEB_SOURCE_DOWNLOAD_BASE = "/web/api/files/raw"
WEB_IMAGE_URL_BASE = "/web/api/images"


def _is_image_mime(mime_type: str | None) -> bool:
    return bool(mime_type) and mime_type.lower().startswith("image/")


def project_conversation_summary(
    summary: ApplicationConversationSummary | Mapping[str, Any],
) -> ConversationSummary:
    """Project an Application conversation summary into its browser model."""
    if isinstance(summary, Mapping):
        return ConversationSummary.model_validate(summary)
    return ConversationSummary(
        conversation_id=summary.conversation_id,
        title=summary.title,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        forked_from_conversation_id=summary.forked_from_conversation_id,
        forked_from_title=summary.forked_from_title,
    )


def project_conversation_history(
    page: ConversationHistoryPage,
    *,
    next_cursor: str | None = None,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
) -> ConversationHistory:
    """Project one bounded durable page into browser presentation models."""
    return ConversationHistory(
        conversation=_head_summary(page.conversation),
        turns=[
            project_conversation_turn(
                turn,
                downloadable_workspaces=downloadable_workspaces,
                visual_workspaces=visual_workspaces,
            )
            for turn in page.turns
        ],
        next_cursor=next_cursor,
    )


def project_conversation_turn(
    turn: LinkedTurn,
    *,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
) -> ConversationTurn:
    """Render one linked turn from the authoritative state of its run.

    A queued or running turn is a pending entry carrying its run id and
    cancellation state, so a reloaded browser resubscribes without remembering
    the original 202 response. A failed or cancelled turn stays visible with its
    public terminal state until its run prunes. Only a succeeded turn carries an
    answer, and that answer is projected from the run's canonical result.
    """
    run = turn.run
    request = AnswerRunRequest.from_request(run.request_input())
    answer = ""
    presentation = None
    if run.status == "succeeded" and run.result is not None:
        projected = project_answer_result(
            run.result,
            source_link_builder=SourceDownloadLinkBuilder(base_url=WEB_SOURCE_DOWNLOAD_BASE),
            downloadable_workspaces=downloadable_workspaces,
            visual_workspaces=visual_workspaces,
            image_url_prefix=WEB_IMAGE_URL_BASE,
            run_id=run.run_id,
            artifact_url_prefix="/web/api/answer",
        )
        answer = str(projected["answer"])
        presentation = build_answer_presentation(
            answer=answer,
            sources=projected["sources"],
            evidence_images=projected["evidence_images"],
            artifacts=projected["artifacts"],
            artifact_outcome=projected["artifact_outcome"],
        )
    return ConversationTurn(
        turn_id=turn.turn_id,
        turn_number=turn.turn_number,
        answer_run_id=run.run_id,
        submission_id=turn.submission_id,
        status=run.status,
        cancel_requested=run.cancel_requested,
        user_text=request.query,
        assistant_text=answer,
        user_attachments=[
            _attachment_reference(run.run_id, turn.turn_number, attachment)
            for attachment in request.attachments
        ],
        presentation=presentation,
        usage=dict((run.result or {}).get("usage") or {}),
        evidence=dict((run.result or {}).get("evidence") or {}),
        error_kind=run.error_kind,
        error_message=run.error_message,
        created_at=turn.created_at,
    )


def _head_summary(head: ConversationHead) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=head.conversation_id,
        title=head.title,
        created_at=head.created_at,
        updated_at=head.updated_at,
    )


def _attachment_reference(
    run_id: str,
    turn_number: int,
    attachment: AttachmentReference,
) -> ConversationAttachmentReference:
    is_image = _is_image_mime(attachment.mime_type)
    url = f"/web/api/runs/{run_id}/attachments/{attachment.ordinal}"
    return ConversationAttachmentReference(
        attachment_id=f"{run_id}:{attachment.ordinal}",
        ordinal=attachment.ordinal,
        kind="image" if is_image else "document",
        filename=attachment.filename,
        mime_type=attachment.mime_type,
        byte_size=attachment.byte_size,
        url=url,
        thumbnail_url=(url + "/thumbnail") if is_image else None,
        label=f"Turn {turn_number}, attachment {attachment.ordinal}",
    )


__all__ = [
    "WEB_IMAGE_URL_BASE",
    "WEB_SOURCE_DOWNLOAD_BASE",
    "project_conversation_history",
    "project_conversation_summary",
    "project_conversation_turn",
]
