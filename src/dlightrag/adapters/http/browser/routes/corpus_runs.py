# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Same-origin browser observation for durable Corpus Mutation Runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from dlightrag.adapters.http.browser.deps import enforce_web_access, get_application
from dlightrag.adapters.http.browser.file_models import WebCorpusRunReceipt, WebCorpusRunStatus
from dlightrag.adapters.http.streaming.answer_stream import (
    follow_run_frames,
    resume_cursor,
    sse_frame,
)
from dlightrag.application.access import AccessAction, corpus_mutation_access_action
from dlightrag.application.runs import RunEvent, RunView

router = APIRouter(prefix="/corpus-runs")


def corpus_run_receipt(
    record: Any,
    *,
    workspace: str,
    file_count: int | None = None,
) -> WebCorpusRunReceipt:
    """Project browser-owned lifecycle URLs for one accepted mutation."""
    base = f"/web/api/corpus-runs/{record.run_id}"
    return WebCorpusRunReceipt(
        run_id=record.run_id,
        run_kind=record.run_kind,
        lane=record.lane,
        status=record.status,
        status_url=base,
        events_url=f"{base}/events",
        cancel_url=base,
        resume_url=f"{base}/resume",
        workspace=workspace,
        file_count=file_count,
    )


def _status(record: RunView) -> WebCorpusRunStatus:
    receipt = corpus_run_receipt(record, workspace=record.access_scope_id)
    return WebCorpusRunStatus(
        **receipt.model_dump(),
        phase=record.phase,
        durable_progress_version=record.durable_progress_version,
        cancel_requested=record.cancel_requested,
        result=dict(record.result) if record.result is not None else None,
        error_kind=record.error_kind,
        error_message=record.error_message,
        repair_reason=record.repair_reason,
        repair_remedy=record.repair_remedy,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
    )


async def _authorized_corpus_run(
    request: Request,
    run_id: str,
    *,
    cancel: bool,
) -> RunView:
    record = await get_application(request).runs.get_global(run_id=run_id)
    if (
        record is None
        or record.run_kind != "corpus_mutation"
        or record.access_scope_kind != "workspace"
    ):
        raise HTTPException(status_code=404, detail="Corpus Mutation Run not found")
    action = (
        corpus_mutation_access_action(record.request_input().get("action"))
        if cancel
        else AccessAction.WORKSPACE_LIST_FILES
    )
    try:
        await enforce_web_access(request, action, record.access_scope_id)
    except HTTPException:
        raise HTTPException(status_code=404, detail="Corpus Mutation Run not found") from None
    return record


@router.get("/{run_id}", response_model=WebCorpusRunStatus)
async def get_corpus_run(run_id: str, request: Request) -> WebCorpusRunStatus:
    record = await _authorized_corpus_run(request, run_id, cancel=False)
    return _status(record)


@router.delete(
    "/{run_id}",
    response_model=WebCorpusRunStatus,
    responses={202: {"model": WebCorpusRunStatus, "description": "Cancellation pending."}},
)
async def cancel_corpus_run(
    run_id: str,
    request: Request,
    response: Response,
) -> WebCorpusRunStatus:
    record = await _authorized_corpus_run(request, run_id, cancel=True)
    outcome = await get_application(request).runs.cancel(
        owner_id=record.access_scope_id,
        run_id=run_id,
    )
    if outcome.outcome == "unknown" or outcome.run is None:
        raise HTTPException(status_code=404, detail="Corpus Mutation Run not found")
    if outcome.outcome == "rejected":
        raise HTTPException(
            status_code=409,
            detail="Run cancellation is closed after the upstream handoff started",
        )
    response.status_code = 202 if outcome.outcome == "pending" else 200
    return _status(outcome.run)


def _corpus_frame(event: RunEvent) -> str:
    payload = dict(event.payload)
    stored = payload.get("result")
    if isinstance(stored, Mapping):
        payload["result"] = dict(stored)
    return sse_frame(sequence=event.sequence, event_type=event.event_type, payload=payload)


@router.post("/{run_id}/resume", response_model=WebCorpusRunStatus, status_code=202)
async def resume_corpus_run(run_id: str, request: Request) -> WebCorpusRunStatus:
    """Resume the same mutation only after an authorized explicit repair action."""
    record = await _authorized_corpus_run(request, run_id, cancel=True)
    if record.phase != "waiting_for_repair":
        raise HTTPException(status_code=409, detail="Run is not waiting for repair")
    application = get_application(request)
    if not await application.runs.resume_repair(
        owner_id=record.access_scope_id,
        run_id=run_id,
    ):
        raise HTTPException(status_code=409, detail="Run could not be resumed")
    updated = await application.runs.get(
        owner_id=record.access_scope_id,
        run_id=run_id,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Corpus Mutation Run not found")
    return _status(updated)


@router.get("/{run_id}/events")
async def stream_corpus_run_events(run_id: str, request: Request) -> StreamingResponse:
    record = await _authorized_corpus_run(request, run_id, cancel=False)
    if record.events_trimmed_at is not None:
        raise HTTPException(
            status_code=410,
            detail="Run events expired; read its result from the status endpoint",
        )
    events = get_application(request).runs.subscribe(
        owner_id=record.access_scope_id,
        run_id=run_id,
        after_sequence=resume_cursor(request),
    )
    return StreamingResponse(
        follow_run_frames(events, _corpus_frame),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["corpus_run_receipt", "router"]
