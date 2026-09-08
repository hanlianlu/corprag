# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Create durable Answers and expose Answer-specific continuation and Artifacts.

``POST /answer`` always accepts, persists, and returns a 202 descriptor; the run
outlives its creating request. Common status, events, and cancellation live under
``/runs``. Every read is owner-scoped, so an unknown run and another owner's run
are indistinguishable. Stored results carry transport-neutral identities only, and
each authenticated read projects fresh URLs from them.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from dlightrag.adapters.http.artifact_delivery import (
    artifact_descriptor,
    artifact_presentation_available,
    artifact_range,
    artifact_response,
)
from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.rest.models import (
    ANSWER_REQUEST_PART_MAX_BYTES,
    AnswerRequest,
    AnswerResponse,
    RunDescriptor,
)
from dlightrag.application.access import AccessAction, UserContext, owner_id_from_user
from dlightrag.application.answer_runs import (
    CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
    CHILD_ROSTER_PAGE_MAX_LIMIT,
    ChildRosterCursorError,
    ChildRosterPageRequest,
)
from dlightrag.application.answer_runs import AnswerRequest as ServiceAnswerRequest
from dlightrag.application.answer_runs.client_contracts import conversation_history_as_dicts
from dlightrag.application.answer_runs.execution import ResourceInput
from dlightrag.application.answer_runs.resource_links import answer_link_resources
from dlightrag.application.answer_runs.results import (
    answer_parts_from_markdown,
    project_answer_result,
    project_artifact_sources,
)
from dlightrag.application.answer_runs.sources import SourceDownloadLinkBuilder
from dlightrag.application.config import AnswerConfig
from dlightrag.application.corpus_admin import safe_source_filename
from dlightrag.application.retrieval import RetrievalOptions
from dlightrag.application.runs import IdempotencyKeyConflict, RunAdmissionLimitExceededError

from .deps import (
    authorized_workspaces,
    get_application,
    idempotency_key,
    resolve_authorized_query_workspaces,
)
from .runs import run_descriptor

logger = logging.getLogger(__name__)
router = APIRouter()

_ALLOWED_ANSWER_PARTS = {"request", "attachments"}
_MAX_ANSWER_FORM_FIELDS = 8


class _AgentControlBody(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=20_000)


@dataclass(frozen=True, slots=True)
class _UploadedAttachment:
    """One multipart file admitted before the run-creation transaction."""

    filename: str
    mime_type: str
    content: bytes


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


def _enforce_answer_attachment_count(count: int, max_attachments: int) -> None:
    """Reject over-limit attachment counts with a stable 413 and the safe limit."""
    if count > max_attachments:
        raise HTTPException(
            status_code=413,
            detail=f"Too many attachments; at most {max_attachments} are allowed",
        )


async def _parse_answer_body(
    request: Request, answer_cfg: AnswerConfig
) -> tuple[AnswerRequest, list[_UploadedAttachment]]:
    """Parse a JSON or multipart answer request bounded by attachment limits.

    JSON bodies carry the complete request with optional HTTPS link descriptors.
    Multipart bodies carry exactly one JSON ``request`` part plus repeated
    ``attachments`` file parts; uploaded files and JSON links may mix. Count,
    per-attachment, and total-byte admission are enforced here, before the run is
    accepted, without buffering an unbounded body.
    """
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" not in content_type:
        try:
            body = AnswerRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        _enforce_answer_attachment_count(len(body.attachments or []), answer_cfg.max_attachments)
        return body, []

    max_attachments = answer_cfg.max_attachments
    max_item = max(1, answer_cfg.max_attachment_bytes)
    max_total = answer_cfg.max_total_attachment_bytes
    try:
        form = await request.form(
            max_files=max_attachments + 2,
            max_fields=_MAX_ANSWER_FORM_FIELDS,
            max_part_size=ANSWER_REQUEST_PART_MAX_BYTES,
        )
    except StarletteHTTPException as exc:
        detail = str(exc.detail)
        if exc.status_code == 400 and detail.startswith(
            ("Too many files.", "Too many fields.", "Part exceeded maximum size")
        ):
            raise HTTPException(status_code=413, detail=detail) from exc
        raise
    try:
        unexpected = sorted({key for key, _ in form.multi_items()} - _ALLOWED_ANSWER_PARTS)
        if unexpected:
            raise HTTPException(
                status_code=400,
                detail=f"Unexpected multipart field(s): {', '.join(unexpected)}",
            )
        request_parts = form.getlist("request")
        if len(request_parts) != 1:
            raise HTTPException(
                status_code=400,
                detail="multipart answer requires exactly one 'request' part",
            )
        raw_request = request_parts[0]
        if isinstance(raw_request, StarletteUploadFile):
            if raw_request.size is not None and raw_request.size > ANSWER_REQUEST_PART_MAX_BYTES:
                raise HTTPException(status_code=413, detail="Answer request part is too large")
            request_json = await raw_request.read(ANSWER_REQUEST_PART_MAX_BYTES + 1)
            if len(request_json) > ANSWER_REQUEST_PART_MAX_BYTES:
                raise HTTPException(status_code=413, detail="Answer request part is too large")
        else:
            request_json = raw_request
        try:
            body = AnswerRequest.model_validate_json(request_json)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

        uploads: list[_UploadedAttachment] = []
        total = 0
        for part in form.getlist("attachments"):
            if not isinstance(part, StarletteUploadFile):
                raise HTTPException(
                    status_code=400, detail="'attachments' parts must be uploaded files"
                )
            if part.size is not None and part.size > max_item:
                raise HTTPException(
                    status_code=413, detail="An attachment exceeds the per-attachment size limit"
                )
            data = await part.read(max_item + 1)
            if len(data) > max_item:
                raise HTTPException(
                    status_code=413, detail="An attachment exceeds the per-attachment size limit"
                )
            total += len(data)
            if total > max_total:
                raise HTTPException(
                    status_code=413, detail="Attachments exceed the total size limit"
                )
            uploads.append(
                _UploadedAttachment(
                    filename=safe_source_filename(part.filename),
                    mime_type=part.content_type or "application/octet-stream",
                    content=data,
                )
            )
        _enforce_answer_attachment_count(
            len(body.attachments or []) + len(uploads), max_attachments
        )
        return body, uploads
    finally:
        await form.close()


def _service_request(
    body: AnswerRequest,
    uploads: list[_UploadedAttachment],
    *,
    workspaces: list[str],
) -> ServiceAnswerRequest:
    """Project one validated wire request into the Answer application contract."""
    from dlightrag.application.retrieval import MetadataFilter

    resources = answer_link_resources(body.attachments)
    resources.extend(
        ResourceInput(
            filename=upload.filename,
            content=upload.content,
            declared_mime=upload.mime_type,
        )
        for upload in uploads
    )
    return ServiceAnswerRequest(
        query=body.query,
        workspaces=tuple(workspaces),
        history=tuple(conversation_history_as_dicts(body.history) or ()),
        retrieval=RetrievalOptions(
            top_k=body.top_k,
            chunk_top_k=body.chunk_top_k,
            federated_rerank=body.federated_rerank,
        ),
        filters=(
            MetadataFilter.model_validate(body.filters.model_dump(exclude_none=True, mode="json"))
            if body.filters
            else None
        ),
        semantic_highlights=body.semantic_highlights,
        resources=tuple(resources),
        mode=body.mode,
    )


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/answer/{run_id}/artifacts")
async def list_answer_artifacts(
    run_id: str, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    application = get_application(request)
    owner_id = owner_id_from_user(user)
    if await application.answers.list_artifacts(owner_id=owner_id, run_id=run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    record = await application.runs.get(owner_id=owner_id, run_id=run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    if record.result is None:
        raise HTTPException(
            status_code=409,
            detail="Answer artifacts are not available until the run has a stored result",
        )
    projected = project_answer_result(
        record.result,
        run_id=run_id,
        artifact_url_prefix="/answer",
    )
    return {"artifacts": projected["artifacts"], "artifact_outcome": projected["artifact_outcome"]}


@router.get(
    "/answer/{run_id}/artifacts/{resource_id}/presentation",
    response_model=AnswerResponse,
)
async def read_answer_artifact_presentation(
    run_id: str,
    resource_id: str,
    request: Request,
    response: Response,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Return one authenticated Markdown Artifact as a typed Answer presentation."""
    application = get_application(request)
    owner_id = owner_id_from_user(user)
    record = await application.runs.get(owner_id=owner_id, run_id=run_id)
    descriptor = artifact_descriptor(record.result if record else None, resource_id)
    if (
        record is None
        or record.status != "succeeded"
        or not artifact_presentation_available(descriptor)
    ):
        raise HTTPException(status_code=404, detail="artifact presentation not found")
    blob = await application.answers.read_artifact(
        owner_id=owner_id, run_id=run_id, resource_id=resource_id
    )
    if blob is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    try:
        markdown = blob.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="artifact is not UTF-8") from exc

    workspaces = [str(value) for value in record.request_input().get("workspaces") or ()]
    downloadable = await authorized_workspaces(
        request, user, workspaces, AccessAction.WORKSPACE_DOWNLOAD_SOURCE
    )
    visual = await authorized_workspaces(
        request, user, workspaces, AccessAction.WORKSPACE_READ_VISUAL_ASSET
    )
    projected = project_answer_result(
        record.result or {},
        source_link_builder=SourceDownloadLinkBuilder(),
        downloadable_workspaces=downloadable,
        visual_workspaces=visual,
        run_id=run_id,
        artifact_url_prefix="/answer",
    )
    artifact_sources = project_artifact_sources(
        record.result or {},
        resource_id=resource_id,
        source_link_builder=SourceDownloadLinkBuilder(),
        downloadable_workspaces=downloadable,
        visual_workspaces=visual,
    )
    projected.update(
        answer=markdown,
        parts=answer_parts_from_markdown(
            markdown,
            artifacts=projected["artifacts"],
            evidence_images=[],
        ),
        contexts={},
        references=[
            {"id": source.id, "title": source.title or "Source"} for source in artifact_sources
        ],
        sources=[source.model_dump() for source in artifact_sources],
        evidence_images=[],
    )
    response.headers["Cache-Control"] = "private, no-store"
    return projected


@router.get("/answer/{run_id}/artifacts/{resource_id}")
async def read_answer_artifact(
    run_id: str,
    resource_id: str,
    request: Request,
    download: bool = False,
    user: UserContext = Depends(get_current_user),
) -> StreamingResponse:
    application = get_application(request)
    owner_id = owner_id_from_user(user)
    record = await application.runs.get(owner_id=owner_id, run_id=run_id)
    descriptor = artifact_descriptor(record.result if record else None, resource_id)
    if descriptor is None or descriptor.get("status") != "available":
        raise HTTPException(status_code=404, detail="artifact not found")
    header = request.headers.get("range", "").strip()
    total = await application.answers.artifact_size(
        owner_id=owner_id,
        run_id=run_id,
        resource_id=resource_id,
    )
    if total is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    offset, length, status_code, content_range = artifact_range(header, total)
    stream = await application.answers.open_artifact(
        owner_id=owner_id,
        run_id=run_id,
        resource_id=resource_id,
        offset=offset,
        length=length,
    )
    if stream is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    media_type, headers = artifact_response(
        descriptor,
        download=download,
        content_range=content_range,
    )
    return StreamingResponse(
        stream,
        media_type=media_type,
        headers=headers,
        status_code=status_code,
    )


@router.post("/answer", response_model=RunDescriptor, status_code=202)
async def create_answer_run(
    request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Accept one durable answer run and return its owner-scoped descriptor.

    Accepts ``application/json`` (link descriptors only) or ``multipart/form-data``
    with one JSON ``request`` part plus repeated ``attachments`` files. Uploaded
    bytes and their references are committed with the run itself.
    """
    application = get_application(request)
    body, uploads = await _parse_answer_body(request, application.config.answer.generation)
    workspaces = await resolve_authorized_query_workspaces(
        request,
        user,
        workspaces=body.workspaces,
        all_workspaces=body.all_workspaces,
    )
    try:
        creation = await application.answers.create(
            request=_service_request(body, uploads, workspaces=workspaces),
            owner_id=owner_id_from_user(user),
            idempotency_key=idempotency_key(request),
            auth_mode=user.auth_mode,
        )
    except IdempotencyKeyConflict:
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key was reused with a different answer request",
        ) from None
    except RunAdmissionLimitExceededError:
        raise HTTPException(
            status_code=503,
            detail="Deployment-wide nonterminal admission limit reached",
        ) from None
    return run_descriptor(creation.run)


@router.post("/answer/{run_id}/steer", status_code=202)
async def steer_answer_run(
    run_id: str,
    body: _AgentControlBody,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    receipt = await get_application(request).answers.steer(
        owner_id=owner_id_from_user(user),
        run_id=run_id,
        instruction=body.content,
    )
    if receipt is None:
        raise HTTPException(status_code=409, detail="Run is not a live Research session")
    return {
        "run_id": receipt.run_id,
        "control_sequence": receipt.control_sequence,
        "kind": receipt.kind,
    }


async def _continue_answer_run(
    *,
    operation: str,
    run_id: str,
    body: _AgentControlBody,
    request: Request,
    user: UserContext,
) -> dict[str, Any]:
    application = get_application(request)
    answers = application.answers
    owner_id = owner_id_from_user(user)
    parent_workspaces = await answers.continuation_workspaces(owner_id=owner_id, run_id=run_id)
    authorized_workspaces: Sequence[str] | None = None
    if parent_workspaces is not None:
        authorized_workspaces = await resolve_authorized_query_workspaces(
            request,
            user,
            workspaces=list(parent_workspaces),
            all_workspaces=False,
        )
    method = answers.follow_up if operation == "follow-up" else answers.fork
    try:
        creation = await method(
            owner_id=owner_id,
            run_id=run_id,
            query=body.content,
            idempotency_key=idempotency_key(request),
            auth_mode=user.auth_mode,
            authorized_workspaces=authorized_workspaces,
        )
    except IdempotencyKeyConflict:
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key was reused with a different continuation",
        ) from None
    except RunAdmissionLimitExceededError:
        raise HTTPException(
            status_code=503,
            detail="Deployment-wide nonterminal admission limit reached",
        ) from None
    if creation is None:
        raise HTTPException(status_code=409, detail="Continuation requires a terminal owned run")
    return run_descriptor(creation.run)


@router.post("/answer/{run_id}/follow-up", status_code=202)
async def follow_up_answer_run(
    run_id: str,
    body: _AgentControlBody,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    return await _continue_answer_run(
        operation="follow-up", run_id=run_id, body=body, request=request, user=user
    )


@router.post("/answer/{run_id}/fork", status_code=202)
async def fork_answer_run(
    run_id: str,
    body: _AgentControlBody,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    return await _continue_answer_run(
        operation="fork", run_id=run_id, body=body, request=request, user=user
    )


@router.get("/answer/{run_id}/transcript")
async def answer_run_transcript(
    run_id: str,
    request: Request,
    limit: int = 20,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    transcript = await get_application(request).answers.transcript_tail(
        owner_id=owner_id_from_user(user), run_id=run_id, limit=limit
    )
    if transcript is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    return {
        "run_id": transcript.run_id,
        "status": transcript.status,
        "messages": list(transcript.messages),
    }


@router.get("/answer/{run_id}/children")
async def answer_run_children(
    run_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
    limit: Annotated[
        int,
        Query(ge=1, le=CHILD_ROSTER_PAGE_MAX_LIMIT),
    ] = CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=1024)] = None,
) -> dict[str, Any]:
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
        owner_id=owner_id_from_user(user), run_id=run_id, page=page_request
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


__all__ = ["router"]
