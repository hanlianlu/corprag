# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""File operations API routes."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.responses import FileResponse, RedirectResponse

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.rest.models import (
    FailedFilesResponse,
    FileListResponse,
)
from dlightrag.adapters.http.source_download import source_download_response
from dlightrag.application.access import AccessAction, UserContext
from dlightrag.application.corpus_admin import (
    FILE_PANEL_PAGE_DEFAULT_LIMIT,
    FILE_PANEL_PAGE_MAX_LIMIT,
    FilePanelCursorError,
    FilePanelPageRequest,
    safe_log_text,
)

from .deps import enforce_access, get_application, resolve_workspace

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/files", response_model=FileListResponse)
async def list_files(
    request: Request,
    workspace: str | None = Query(default=None),
    limit: Annotated[
        int,
        Query(ge=1, le=FILE_PANEL_PAGE_MAX_LIMIT),
    ] = FILE_PANEL_PAGE_DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=1024)] = None,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """List one bounded page of ingested documents."""
    application = get_application(request)
    ws = resolve_workspace(workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_LIST_FILES, workspace=ws)
    page = _file_page_request(
        application,
        ws,
        view="processed",
        limit=limit,
        cursor=cursor,
    )
    snapshot = await application.corpora.file_panel_snapshot(ws, page=page)
    files = snapshot["files"]
    return {
        "files": files,
        "count": len(files),
        "workspace": ws,
        "next_cursor": _encode_file_cursor(application, snapshot["next_cursor"]),
        "fetched_rows": snapshot["fetched_rows"],
    }


@router.get("/files/failed", response_model=FailedFilesResponse)
async def list_failed_files(
    request: Request,
    workspace: str | None = Query(default=None),
    limit: Annotated[
        int,
        Query(ge=1, le=FILE_PANEL_PAGE_MAX_LIMIT),
    ] = FILE_PANEL_PAGE_DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=1024)] = None,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """List one bounded page of documents in DocStatus.FAILED."""
    application = get_application(request)
    ws = resolve_workspace(workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_LIST_FILES, workspace=ws)
    page = _file_page_request(
        application,
        ws,
        view="failed",
        limit=limit,
        cursor=cursor,
    )
    snapshot = await application.corpora.failed_file_snapshot(ws, page=page)
    failed = snapshot["failed"]
    return {
        "failed": failed,
        "count": len(failed),
        "workspace": ws,
        "next_cursor": _encode_file_cursor(application, snapshot["next_cursor"]),
        "fetched_rows": snapshot["fetched_rows"],
    }


@router.get("/files/raw/{document_id:path}", response_model=None)
async def serve_file(
    document_id: str,
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> FileResponse | RedirectResponse:
    """Download one source document through the REST Bearer boundary."""
    safe_workspace = resolve_workspace(workspace, request)
    await _enforce_source_download_access(
        request,
        user,
        workspace=safe_workspace,
    )
    return await source_download_response(
        get_application(request).corpora,
        workspace=safe_workspace,
        document_id=document_id,
    )


def _file_page_request(
    application: Any,
    workspace: str,
    *,
    view: str,
    limit: int,
    cursor: str | None,
) -> FilePanelPageRequest:
    try:
        decoded = (
            application.corpora.file_panel_cursor_codec.decode(cursor)
            if cursor is not None
            else None
        )
        if decoded is not None and decoded.workspace != workspace:
            raise FilePanelCursorError("file-panel cursor belongs to another workspace")
        if decoded is not None and decoded.view != view:
            raise FilePanelCursorError("file-panel cursor belongs to another view")
        return FilePanelPageRequest(limit=limit, cursor=decoded)
    except (FilePanelCursorError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


def _encode_file_cursor(application: Any, cursor: Any | None) -> str | None:
    if cursor is None:
        return None
    return application.corpora.file_panel_cursor_codec.encode(cursor)


async def _enforce_source_download_access(
    request: Request,
    user: UserContext,
    *,
    workspace: str,
) -> None:
    try:
        await enforce_access(
            request,
            user,
            AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
            workspace=workspace,
        )
    except HTTPException as exc:
        if exc.status_code == 403:
            logger.info(
                "source_download_projection_outcome",
                extra={"outcome": "unauthorized", "workspace": safe_log_text(workspace)},
            )
        raise
