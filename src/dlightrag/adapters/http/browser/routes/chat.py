# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for the chat interface and durable answer runs."""

import logging
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from dlightrag.adapters.http.artifact_delivery import (
    artifact_descriptor,
    artifact_presentation_available,
    artifact_range,
    artifact_response,
)
from dlightrag.adapters.http.browser.answer_events import browser_frame
from dlightrag.adapters.http.browser.app_shell import app_html_response
from dlightrag.adapters.http.browser.attachment_requests import parse_web_answer_request
from dlightrag.adapters.http.browser.conversation_models import (
    AcceptedAnswer,
    ConversationTurn,
    WebCommandError,
    WebCommandErrorKind,
)
from dlightrag.adapters.http.browser.conversations import (
    WEB_IMAGE_URL_BASE,
    WEB_SOURCE_DOWNLOAD_BASE,
    project_conversation_summary,
    project_conversation_turn,
)
from dlightrag.adapters.http.browser.deps import (
    enforce_web_access,
    get_application,
    get_web_access_gate,
    get_web_conversation_service,
    get_workspace,
)
from dlightrag.adapters.http.browser.presentation import (
    AnswerPresentation,
    build_answer_presentation,
)
from dlightrag.adapters.http.browser.routes.skills import require_known_skill
from dlightrag.adapters.http.streaming.answer_stream import follow_run_frames, resume_cursor
from dlightrag.application.access import AccessAction, owner_id_from_user
from dlightrag.application.answer_runs import (
    CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
    CHILD_ROSTER_PAGE_MAX_LIMIT,
    AnswerRuntimeUnavailableError,
    ChildRosterCursorError,
    ChildRosterPageRequest,
)
from dlightrag.application.answer_runs.results import (
    project_answer_result,
    project_artifact_sources,
)
from dlightrag.application.answer_runs.sources import SourceDownloadLinkBuilder
from dlightrag.application.corpus_admin import normalize_workspace_ids
from dlightrag.application.runs import IdempotencyKeyConflict, RunCapacityExceededError
from dlightrag.application.web_conversations import (
    ConversationSubmissionConflict,
    LinkedTurn,
    WebAnswerSubmission,
    WebConversationService,
    WebConversationUnavailableError,
)

logger = logging.getLogger(__name__)

router = APIRouter()
page_router = APIRouter()


class _WebAgentControl(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=20_000)


class _WebContinuation(_WebAgentControl):
    submission_id: UUID


@page_router.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    """Serve the Vite-owned application document."""
    return app_html_response("index.html")


@page_router.get("/conversations/{conversation_id}", response_class=FileResponse)
async def conversation_page(
    conversation_id: str,  # noqa: ARG001 - the browser router owns selection
) -> FileResponse:
    """Serve the same application document for one explicit client route."""
    return app_html_response("index.html")


@page_router.get("/design-system", response_class=FileResponse)
async def design_system_page() -> FileResponse:
    """Serve the isolated Vite design-system entry document."""
    return app_html_response("design-system.html")


@page_router.get("/product-showcase", response_class=FileResponse)
async def product_showcase_page() -> FileResponse:
    """Serve the product-component showcase entry document."""
    return app_html_response("product-showcase.html")


@router.post("/answer", status_code=202, response_model=AcceptedAnswer)
async def start_answer_run(
    request: Request,
    workspace: str = Depends(get_workspace),
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> AcceptedAnswer:
    """Accept one submission as a durable run linked to its conversation entry.

    The run, its uploaded bytes, and the conversation turn are committed in one
    transaction before this result is returned, so the browser can discard its
    local File and Blob resources at the acceptance seam.
    """
    application = get_application(request)
    cfg = application.config
    # Enforce the probed answer capability at admission (pre-acceptance 4xx).
    capability = (await application.answers.capabilities()).answer
    try:
        body = await parse_web_answer_request(
            request,
            max_attachments=cfg.answer.generation.max_attachments,
            max_attachment_bytes=cfg.answer.generation.max_attachment_bytes,
            max_total_attachment_bytes=cfg.answer.generation.max_total_attachment_bytes,
            image_max_pixels=cfg.answer.generation.image_max_pixels,
            answer_image_capability=capability,
        )
    except HTTPException as exc:
        kind = "attachment_rejected" if exc.status_code == 413 else "invalid_request"
        raise _command_error(exc.status_code, kind, str(exc.detail)) from exc
    query = body.query.strip()
    if not query:
        raise _command_error(422, "invalid_request", "A question is required")

    requested_skill = body.requested_skill
    if requested_skill is not None:
        try:
            owner_id = owner_id_from_user(getattr(request.state, "user_context", None))
            requested_skill = require_known_skill(application, owner_id, requested_skill)
        except ValueError as exc:
            raise _command_error(422, "invalid_request", str(exc)) from exc
    mode = body.mode
    if requested_skill is not None and mode != "research":
        # Skills (load_skill tool + metadata) exist only in Research runs.
        mode = "research"

    target_workspaces = normalize_workspace_ids(body.workspaces or [workspace])
    try:
        for ws in target_workspaces:
            await enforce_web_access(request, AccessAction.WORKSPACE_QUERY, ws)
    except HTTPException as exc:
        if exc.status_code != 403:
            raise
        raise _command_error(403, "scope_forbidden", str(exc.detail)) from exc

    try:
        submission = await conversation_service.start_answer(
            getattr(request.state, "user_context", None),
            conversation_id=(
                str(body.conversation_id) if body.conversation_id is not None else None
            ),
            submission_id=str(body.submission_id),
            query=query,
            workspaces=target_workspaces,
            attachments=body.attachments,
            mode=mode,
            requested_skill=requested_skill,
        )
    except ConversationSubmissionConflict, IdempotencyKeyConflict:
        raise _command_error(
            409,
            "submission_conflict",
            "This submission id was already used for a different request",
        ) from None
    except (
        AnswerRuntimeUnavailableError,
        RunCapacityExceededError,
        WebConversationUnavailableError,
    ):
        raise _command_error(
            503, "service_unavailable", "Answer submission is temporarily unavailable"
        ) from None
    if submission is None:
        raise _command_error(404, "conversation_missing", "Conversation not found")
    return await accepted_answer(request, submission)


@router.get("/answer-submissions/{submission_id}", response_model=AcceptedAnswer)
async def accepted_answer_submission(
    submission_id: UUID,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> AcceptedAnswer:
    """Recover one owner-scoped accepted command after an ambiguous POST result."""
    try:
        submission = await conversation_service.submission(
            getattr(request.state, "user_context", None), str(submission_id)
        )
    except WebConversationUnavailableError:
        raise _command_error(
            503,
            "service_unavailable",
            "Answer submission lookup is temporarily unavailable",
        ) from None
    if submission is None:
        raise _command_error(404, "invalid_request", "Answer submission not found")
    return await accepted_answer(request, submission)


@router.get("/runs/{run_id}", response_model=ConversationTurn)
async def answer_run_status(
    run_id: str,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationTurn:
    """Return one owned run's conversation entry and authoritative state."""
    turn = await conversation_service.turn_for_run(
        getattr(request.state, "user_context", None), run_id
    )
    if turn is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    downloadable, visual = await _projection_workspaces(request, turn.run.request_input())
    return project_conversation_turn(
        turn, downloadable_workspaces=downloadable, visual_workspaces=visual
    )


@router.post("/answer/{run_id}/steer", status_code=202)
async def steer_answer_run(
    run_id: str,
    body: _WebAgentControl,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> dict[str, Any]:
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    receipt = await get_application(request).answers.steer(
        owner_id=owner_id_from_user(user), run_id=run_id, instruction=body.content
    )
    if receipt is None:
        raise HTTPException(status_code=409, detail="Run is not a live Research session")
    return {
        "run_id": receipt.run_id,
        "control_sequence": receipt.control_sequence,
        "kind": receipt.kind,
    }


@router.get("/answer/{run_id}/children")
async def answer_run_children(
    run_id: str,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
    limit: Annotated[
        int,
        Query(ge=1, le=CHILD_ROSTER_PAGE_MAX_LIMIT),
    ] = CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=1024)] = None,
) -> dict[str, Any]:
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    answers = get_application(request).answers
    try:
        decoded_cursor = (
            answers.child_roster_cursor_codec.decode(cursor) if cursor is not None else None
        )
        if decoded_cursor is not None and str(decoded_cursor.run_id) != run_id:
            raise ChildRosterCursorError("child-roster cursor belongs to another run")
        page_request = ChildRosterPageRequest(limit=limit, cursor=decoded_cursor)
    except (ChildRosterCursorError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    page = await answers.children(
        owner_id=owner_id_from_user(user),
        run_id=run_id,
        page=page_request,
    )
    if page is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    return {
        "run_id": run_id,
        "children": [dict(child) for child in page.children],
        "next_cursor": (
            answers.child_roster_cursor_codec.encode(page.next_cursor)
            if page.next_cursor is not None
            else None
        ),
    }


async def _continue_answer_run(
    *,
    kind: str,
    run_id: str,
    body: _WebContinuation,
    request: Request,
    conversation_service: WebConversationService,
) -> AcceptedAnswer:
    user = getattr(request.state, "user_context", None)
    parent = await conversation_service.turn_for_run(user, run_id)
    authorized_workspaces: list[str] | None = None
    if parent is not None and parent.run.terminal:
        authorized_workspaces = [
            str(item) for item in parent.run.request_input().get("workspaces") or ()
        ]
        for workspace_id in authorized_workspaces:
            await enforce_web_access(request, AccessAction.WORKSPACE_QUERY, workspace_id)
    try:
        submission = await conversation_service.continue_answer(
            user,
            parent_run_id=run_id,
            submission_id=str(body.submission_id),
            query=body.content,
            kind=kind,
            authorized_workspaces=authorized_workspaces,
        )
    except ConversationSubmissionConflict, IdempotencyKeyConflict:
        raise _command_error(
            409,
            "submission_conflict",
            "This submission id was already used for a different continuation",
        ) from None
    except RunCapacityExceededError:
        raise _command_error(
            503, "service_unavailable", "Answer submission is temporarily unavailable"
        ) from None
    if submission is None:
        raise _command_error(
            409,
            "invalid_request",
            "Continuation requires a terminal answer",
        )
    return await accepted_answer(request, submission)


@router.post("/answer/{run_id}/follow-up", status_code=202)
async def follow_up_answer_run(
    run_id: str,
    body: _WebContinuation,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> AcceptedAnswer:
    return await _continue_answer_run(
        kind="follow_up",
        run_id=run_id,
        body=body,
        request=request,
        conversation_service=conversation_service,
    )


@router.post("/answer/{run_id}/fork", status_code=202)
async def fork_answer_run(
    run_id: str,
    body: _WebContinuation,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> AcceptedAnswer:
    return await _continue_answer_run(
        kind="fork",
        run_id=run_id,
        body=body,
        request=request,
        conversation_service=conversation_service,
    )


@router.delete("/runs/{run_id}", response_model=ConversationTurn)
async def cancel_answer_run(
    run_id: str,
    request: Request,
    response: Response,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationTurn:
    """Request cancellation of one owned run; repeating it is a no-op."""
    user = getattr(request.state, "user_context", None)
    turn = await conversation_service.turn_for_run(user, run_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    outcome = await get_application(request).runs.cancel(
        owner_id=owner_id_from_user(user), run_id=run_id
    )
    if outcome.run is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    # 202 only while a running worker still has to observe the request.
    response.status_code = 202 if outcome.outcome == "pending" else 200
    downloadable, visual = await _projection_workspaces(request, outcome.run.request_input())
    return project_conversation_turn(
        replace(turn, run=outcome.run),
        downloadable_workspaces=downloadable,
        visual_workspaces=visual,
    )


@router.get("/answer/{run_id}/artifacts/{resource_id}")
async def answer_artifact_data(
    run_id: str,
    resource_id: str,
    request: Request,
    download: bool = False,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> StreamingResponse:
    """Stream one owned Artifact as inert data with Range support."""
    user = getattr(request.state, "user_context", None)
    turn = await conversation_service.turn_for_run(user, run_id)
    result = turn.run.result if turn is not None else None
    descriptor = artifact_descriptor(result, resource_id)
    if descriptor is None or descriptor.get("status") != "available":
        raise HTTPException(status_code=404, detail="Artifact not found")
    owner = owner_id_from_user(user)
    application = get_application(request)
    total = await application.answers.artifact_size(
        owner_id=owner, run_id=run_id, resource_id=resource_id
    )
    if total is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    offset, length, status_code, content_range = artifact_range(
        request.headers.get("range", "").strip(), total
    )
    stream = await application.answers.open_artifact(
        owner_id=owner,
        run_id=run_id,
        resource_id=resource_id,
        offset=offset,
        length=length,
    )
    if stream is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    media_type, headers = artifact_response(
        descriptor,
        download=download,
        content_range=content_range,
    )
    return StreamingResponse(
        stream,
        media_type=media_type,
        status_code=status_code,
        headers=headers,
    )


@router.get(
    "/answer/{run_id}/artifacts/{resource_id}/presentation",
    response_model=AnswerPresentation,
)
async def answer_artifact_presentation(
    run_id: str,
    resource_id: str,
    request: Request,
    response: Response,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> AnswerPresentation:
    """Return a safe AnswerPresentation for any Markdown Artifact."""
    user = getattr(request.state, "user_context", None)
    turn = await conversation_service.turn_for_run(user, run_id)
    if turn is None or turn.run.status != "succeeded" or turn.run.result is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    descriptor = artifact_descriptor(turn.run.result, resource_id)
    if not artifact_presentation_available(descriptor):
        raise HTTPException(status_code=404, detail="Artifact presentation not found")
    blob = await get_application(request).answers.read_artifact(
        owner_id=owner_id_from_user(user), run_id=run_id, resource_id=resource_id
    )
    if blob is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        markdown = blob.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="Artifact is not UTF-8") from exc
    downloadable, visual = await _projection_workspaces(request, turn.run.request_input())
    projected = project_answer_result(
        turn.run.result,
        source_link_builder=SourceDownloadLinkBuilder(base_url=WEB_SOURCE_DOWNLOAD_BASE),
        downloadable_workspaces=downloadable,
        visual_workspaces=visual,
        image_url_prefix=WEB_IMAGE_URL_BASE,
        run_id=run_id,
        artifact_url_prefix="/web/api/answer",
    )
    sources = project_artifact_sources(
        turn.run.result,
        resource_id=resource_id,
        source_link_builder=SourceDownloadLinkBuilder(base_url=WEB_SOURCE_DOWNLOAD_BASE),
        downloadable_workspaces=downloadable,
        visual_workspaces=visual,
        image_url_prefix=WEB_IMAGE_URL_BASE,
    )
    response.headers["Cache-Control"] = "private, no-store"
    return build_answer_presentation(
        answer=markdown,
        sources=sources,
        evidence_images=[],
        artifacts=projected["artifacts"],
        artifact_outcome=projected["artifact_outcome"],
    )


@router.get("/runs/{run_id}/events")
async def answer_run_events(
    run_id: str,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> StreamingResponse:
    """Replay this run's durable events from a cursor, then follow it live."""
    user = getattr(request.state, "user_context", None)
    turn = await conversation_service.turn_for_run(user, run_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    if turn.run.events_trimmed_at is not None:
        raise HTTPException(
            status_code=410,
            detail="Answer run events expired; read its result from the conversation",
        )
    downloadable, visual = await _projection_workspaces(request, turn.run.request_input())
    events = get_application(request).runs.subscribe(
        owner_id=owner_id_from_user(user),
        run_id=run_id,
        after_sequence=resume_cursor(request),
    )
    return StreamingResponse(
        follow_run_frames(
            events,
            partial(
                browser_frame,
                downloadable_workspaces=downloadable,
                visual_workspaces=visual,
                live_after=turn.run.next_event_sequence - 1,
                run_id=run_id,
            ),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def accepted_answer(
    request: Request,
    submission: WebAnswerSubmission,
) -> AcceptedAnswer:
    """Project one accepted command through the same canonical history model."""
    downloadable, visual = await _projection_workspaces(request, submission.run.request_input())
    linked = LinkedTurn(
        turn_id=submission.turn_id,
        turn_number=submission.turn_number,
        submission_id=submission.submission_id,
        created_at=submission.created_at or submission.run.created_at,
        run=submission.run,
        conversation_id=submission.conversation.conversation_id,
    )
    return AcceptedAnswer(
        conversation=project_conversation_summary(submission.conversation),
        turn=project_conversation_turn(
            linked,
            downloadable_workspaces=downloadable,
            visual_workspaces=visual,
        ),
    )


def _command_error(status_code: int, kind: WebCommandErrorKind, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=WebCommandError(kind=kind, message=message).model_dump(),
    )


async def _projection_workspaces(
    request: Request, run_request: Mapping[str, Any]
) -> tuple[set[str], set[str]]:
    """Authorize this reader's source downloads and visuals for one run."""
    workspace_ids = [str(value) for value in run_request.get("workspaces") or ()]
    gate = get_web_access_gate(request)
    return (
        await gate.authorized_workspace_ids(
            AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
            workspace_ids,
        ),
        await gate.authorized_workspace_ids(
            AccessAction.WORKSPACE_READ_VISUAL_ASSET,
            workspace_ids,
        ),
    )
