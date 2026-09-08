# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Common owner-scoped REST lifecycle for every durable run kind."""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.rest.models import RunStatusResponse
from dlightrag.adapters.http.streaming.answer_stream import (
    follow_run_frames,
    resume_cursor,
    sse_frame,
)
from dlightrag.application.access import (
    AccessAction,
    UserContext,
    corpus_mutation_access_action,
    owner_id_from_user,
)
from dlightrag.application.retrieval import (
    RetrieveProjection,
    RetrieveResponse,
    retrieval_response_payload,
)
from dlightrag.application.runs import RunEvent, RunView
from dlightrag.engine.answer.citations.sources import SourceDownloadLinkBuilder
from dlightrag.engine.answer.results import project_answer_result

from .deps import authorized_workspaces, enforce_access, get_application

router = APIRouter()


def run_descriptor(record: RunView) -> dict[str, Any]:
    """Project the common lifecycle URLs plus operation-owned Answer lineage."""
    accepted = record.request_input()
    return {
        "run_id": record.run_id,
        "run_kind": record.run_kind,
        "lane": record.lane,
        "status": record.status,
        "status_url": f"/runs/{record.run_id}",
        "events_url": f"/runs/{record.run_id}/events",
        "cancel_url": f"/runs/{record.run_id}",
        "parent_run_id": accepted.get("parent_run_id"),
        "continuation_kind": accepted.get("continuation_kind"),
    }


async def run_status_payload(
    request: Request, user: UserContext, record: RunView
) -> dict[str, Any]:
    application = get_application(request)
    result: dict[str, Any] | None = None
    if record.result is not None:
        workspaces = [str(value) for value in record.request_input().get("workspaces") or ()]
        downloadable = await authorized_workspaces(
            request, user, workspaces, AccessAction.WORKSPACE_DOWNLOAD_SOURCE
        )
        visual = await authorized_workspaces(
            request, user, workspaces, AccessAction.WORKSPACE_READ_VISUAL_ASSET
        )
        if record.run_kind == "retrieval":
            result = retrieval_response_payload(
                application.retrieval.project_stored(
                    record.result,
                    RetrieveProjection(
                        downloadable_workspaces=frozenset(downloadable),
                        visual_workspaces=frozenset(visual),
                        include_download_links=True,
                    ),
                )
            )
        elif record.run_kind == "answer":
            result = project_answer_result(
                record.result,
                source_link_builder=SourceDownloadLinkBuilder(),
                downloadable_workspaces=downloadable,
                visual_workspaces=visual,
                run_id=record.run_id,
                artifact_url_prefix="/answer",
            )
        else:
            result = dict(record.result)
    return {
        **run_descriptor(record),
        "phase": record.phase,
        "durable_progress_version": record.durable_progress_version,
        "cancel_requested": record.cancel_requested,
        "result": result,
        "error_kind": record.error_kind,
        "error_message": record.error_message,
        "repair_reason": record.repair_reason,
        "repair_remedy": record.repair_remedy,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
    }


@router.get("/runs")
async def list_runs(
    request: Request,
    user: UserContext = Depends(get_current_user),
    after: str | None = None,
    limit: int = 50,
    workspace: str | None = None,
) -> dict[str, Any]:
    owner = owner_id_from_user(user)
    if workspace is not None:
        from dlightrag.application.corpus_admin import normalize_workspace

        try:
            owner = normalize_workspace(workspace)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid workspace") from exc
        if not owner:
            raise HTTPException(status_code=400, detail="Invalid workspace")
        await enforce_access(request, user, AccessAction.WORKSPACE_LIST_FILES, workspace=owner)
    rows = await get_application(request).runs.list(
        owner_id=owner,
        after_run_id=after,
        limit=min(max(limit, 1), 100),
    )
    if workspace is not None:
        rows = tuple(row for row in rows if row.access_scope_kind == "workspace")
    return {"runs": [run_descriptor(record) for record in rows]}


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run(
    run_id: str, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    record = await _authorized_run(request, user, run_id, cancel=False)
    return await run_status_payload(request, user, record)


@router.delete(
    "/runs/{run_id}",
    response_model=RunStatusResponse,
    responses={202: {"model": RunStatusResponse, "description": "Cancellation pending."}},
)
async def cancel_run(
    run_id: str,
    request: Request,
    response: Response,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    record = await _authorized_run(request, user, run_id, cancel=True)
    outcome = await get_application(request).runs.cancel(
        owner_id=record.access_scope_id, run_id=run_id
    )
    if outcome.outcome == "unknown" or outcome.run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if outcome.outcome == "rejected":
        raise HTTPException(
            status_code=409,
            detail="Run cancellation is closed after the upstream handoff started",
        )
    response.status_code = 202 if outcome.outcome == "pending" else 200
    return await run_status_payload(request, user, outcome.run)


@router.post("/runs/{run_id}/resume", response_model=RunStatusResponse, status_code=202)
async def resume_run(
    run_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Requeue the same authorized mutation after explicit operator repair."""
    record = await _authorized_run(request, user, run_id, cancel=True)
    if record.run_kind != "corpus_mutation" or record.phase != "waiting_for_repair":
        raise HTTPException(status_code=409, detail="Run is not waiting for repair")
    if not await get_application(request).runs.resume_repair(
        owner_id=record.access_scope_id, run_id=run_id
    ):
        raise HTTPException(status_code=409, detail="Run could not be resumed")
    updated = await get_application(request).runs.get(
        owner_id=record.access_scope_id, run_id=run_id
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return await run_status_payload(request, user, updated)


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str, request: Request, user: UserContext = Depends(get_current_user)
) -> StreamingResponse:
    application = get_application(request)
    record = await _authorized_run(request, user, run_id, cancel=False)
    owner_id = record.access_scope_id
    if record.events_trimmed_at is not None:
        raise HTTPException(
            status_code=410,
            detail="Run events expired; read its result from the status endpoint",
        )
    workspaces = [str(value) for value in record.request_input().get("workspaces") or ()]
    downloadable = await authorized_workspaces(
        request, user, workspaces, AccessAction.WORKSPACE_DOWNLOAD_SOURCE
    )
    visual = await authorized_workspaces(
        request, user, workspaces, AccessAction.WORKSPACE_READ_VISUAL_ASSET
    )
    events = application.runs.subscribe(
        owner_id=owner_id, run_id=run_id, after_sequence=resume_cursor(request)
    )
    return StreamingResponse(
        follow_run_frames(
            events,
            partial(
                run_frame,
                downloadable_workspaces=downloadable,
                visual_workspaces=visual,
                run_id=run_id,
                run_kind=record.run_kind,
                retrieval_projector=(
                    application.retrieval.project_stored if record.run_kind == "retrieval" else None
                ),
            ),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _authorized_run(
    request: Request,
    user: UserContext,
    run_id: str,
    *,
    cancel: bool,
) -> RunView:
    """Resolve globally, then fail closed through the Run's declared scope."""
    record = await get_application(request).runs.get_global(run_id=run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if record.access_scope_kind == "owner":
        if record.access_scope_id != owner_id_from_user(user):
            raise HTTPException(status_code=404, detail="Run not found")
        return record
    action = AccessAction.WORKSPACE_LIST_FILES
    if cancel:
        action = corpus_mutation_access_action(record.request_input().get("action"))
    try:
        await enforce_access(request, user, action, workspace=record.access_scope_id)
    except HTTPException:
        # Unknown and unauthorized are deliberately indistinguishable.
        raise HTTPException(status_code=404, detail="Run not found") from None
    return record


def run_frame(
    event: RunEvent,
    *,
    downloadable_workspaces: set[str] | None,
    visual_workspaces: set[str] | None,
    run_id: str,
    run_kind: str,
    retrieval_projector: Callable[[Mapping[str, Any], RetrieveProjection], RetrieveResponse]
    | None = None,
) -> str:
    payload = dict(event.payload)
    stored = payload.get("result")
    if run_kind == "retrieval" and isinstance(stored, dict):
        if retrieval_projector is None:
            raise RuntimeError("Retrieval event projection is unavailable")
        payload["result"] = retrieval_response_payload(
            retrieval_projector(
                stored,
                RetrieveProjection(
                    downloadable_workspaces=(
                        frozenset(downloadable_workspaces)
                        if downloadable_workspaces is not None
                        else None
                    ),
                    visual_workspaces=(
                        frozenset(visual_workspaces) if visual_workspaces is not None else None
                    ),
                    include_download_links=True,
                ),
            )
        )
    elif run_kind == "answer" and isinstance(stored, dict):
        payload["result"] = project_answer_result(
            stored,
            source_link_builder=SourceDownloadLinkBuilder(),
            downloadable_workspaces=downloadable_workspaces,
            visual_workspaces=visual_workspaces,
            run_id=run_id,
            artifact_url_prefix="/answer",
        )
    return sse_frame(sequence=event.sequence, event_type=event.event_type, payload=payload)


__all__ = ["router", "run_descriptor", "run_frame", "run_status_payload"]
