# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Workspace lifecycle API routes."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.rest.models import (
    WorkspaceCreateRequest,
    WorkspaceCreateResponse,
    WorkspacesResponse,
)
from dlightrag.application.access import AccessAction, UserContext
from dlightrag.application.corpus_admin import (
    WORKSPACE_CATALOG_PAGE_DEFAULT_LIMIT,
    WORKSPACE_CATALOG_PAGE_MAX_LIMIT,
    WorkspaceCatalogCursorError,
    WorkspaceCatalogPageRequest,
    normalize_workspace,
    validate_workspace_name,
)

from .deps import (
    enforce_access,
    filter_workspace_records,
    get_application,
)

router = APIRouter()


def _normalize_create_body(body: WorkspaceCreateRequest) -> tuple[str, str]:
    """Return internal workspace id and display name for a create request."""
    try:
        label = validate_workspace_name(body.workspace)
        display_name = validate_workspace_name(body.display_name or label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return normalize_workspace(label), display_name


@router.get("/workspaces", response_model=WorkspacesResponse)
async def list_workspaces(
    request: Request,
    user: UserContext = Depends(get_current_user),
    limit: Annotated[
        int,
        Query(ge=1, le=WORKSPACE_CATALOG_PAGE_MAX_LIMIT),
    ] = WORKSPACE_CATALOG_PAGE_DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=1024)] = None,
) -> dict[str, Any]:
    """Return one bounded page of registered workspaces.

    The catalog is paged over its full ascending workspace ordering, and the
    caller's access gate filters the returned page afterwards. The gate is
    per-request user-dependent, so paging over a pre-filtered set would leak
    ordering state across principals; paging the full catalog keeps the cursor
    a pure ordering fact.
    """
    application = get_application(request)
    try:
        decoded_cursor = (
            application.corpora.workspace_catalog_cursor_codec.decode(cursor)
            if cursor is not None
            else None
        )
        page_request = WorkspaceCatalogPageRequest(limit=limit, cursor=decoded_cursor)
    except (WorkspaceCatalogCursorError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    page = await application.corpora.list_workspace_records_page(page=page_request)
    records = await filter_workspace_records(
        request, user, AccessAction.WORKSPACE_QUERY, list(page.items)
    )
    return {
        "workspaces": [row["workspace"] for row in records],
        "records": records,
        "next_cursor": (
            application.corpora.workspace_catalog_cursor_codec.encode(page.next_cursor)
            if page.next_cursor is not None
            else None
        ),
    }


@router.post(
    "/workspaces",
    status_code=status.HTTP_201_CREATED,
    response_model=WorkspaceCreateResponse,
)
async def create_workspace(
    body: WorkspaceCreateRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Create an empty workspace in the durable registry."""
    application = get_application(request)
    workspace, display_name = _normalize_create_body(body)
    await enforce_access(request, user, AccessAction.WORKSPACE_CREATE, workspace=workspace)
    existing = await application.corpora.list_workspaces()
    if workspace in existing:
        raise HTTPException(status_code=409, detail=f"Workspace '{display_name}' already exists")

    await application.corpora.create_workspace(workspace, display_name=display_name)
    return {
        "workspace": workspace,
        "display_name": display_name,
        "created": True,
    }


@router.get("/workspaces/{workspace}/storage")
async def get_workspace_storage(
    workspace: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Operator-facing storage/promotion facts for one workspace (admin only).

    Ordinary readers/editors are never granted ``workspace.storage_status``:
    tier, promotion state, bounded counters, last error, and retry time are
    operations data, not answer-user data.
    """
    application = get_application(request)
    try:
        label = validate_workspace_name(workspace)
        normalized = normalize_workspace(label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await enforce_access(request, user, AccessAction.WORKSPACE_STORAGE_STATUS, workspace=normalized)
    status = await application.corpora.get_workspace_storage_status(normalized)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{normalized}' not found")
    return status
