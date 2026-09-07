# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for workspace management."""

import logging
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from dlightrag.adapters.http.browser.deps import (
    enforce_web_access,
    filter_web_workspace_records,
    get_application,
)
from dlightrag.adapters.http.browser.file_models import WebCorpusRunReceipt
from dlightrag.adapters.http.browser.routes.corpus_runs import corpus_run_receipt
from dlightrag.adapters.http.browser.workspace_models import (
    WebBootstrapWorkspace,
    project_workspace_record,
)
from dlightrag.application.access import AccessAction, WorkspaceRecord, owner_id_from_user
from dlightrag.application.answer_runs.client_contracts import ClientContractModel
from dlightrag.application.corpus_admin import (
    WORKSPACE_CATALOG_PAGE_DEFAULT_LIMIT,
    WORKSPACE_CATALOG_PAGE_MAX_LIMIT,
    WorkspaceCatalogCursorError,
    WorkspaceCatalogPageRequest,
    normalize_workspace,
)

if TYPE_CHECKING:
    from dlightrag.application import Application

logger = logging.getLogger(__name__)

router = APIRouter()


def _ordered_unique(workspaces: list[str]) -> list[str]:
    result: list[str] = []
    for workspace in workspaces:
        if workspace and workspace not in result:
            result.append(workspace)
    return result


async def _visible_workspace_names(request: Request, application: Application) -> list[str]:
    records: list[WorkspaceRecord] = [
        {"workspace": workspace} for workspace in await application.corpora.list_workspaces()
    ]
    visible = await filter_web_workspace_records(request, AccessAction.WORKSPACE_QUERY, records)
    return [str(row["workspace"]) for row in visible]


def _default_workspace(workspaces: list[str]) -> str:
    if not workspaces:
        return ""
    return "default" if "default" in workspaces else workspaces[0]


def _cookie_active_workspaces(request: Request, visible_workspaces: list[str]) -> list[str]:
    visible = set(visible_workspaces)
    raw = request.cookies.get("dlightrag_workspace_ids", "")
    active = [normalize_workspace(item.strip()) for item in raw.split(",") if item.strip()]
    return _ordered_unique([workspace for workspace in active if workspace in visible])


def _set_workspace_cookies(
    response: Response,
    request: Request,
    visible_workspaces: list[str],
    *,
    active_workspaces: list[str] | None = None,
    primary_workspace: str | None = None,
) -> None:
    """Persist selector state using only canonical workspace names.

    All workspace values are server-trusted (sourced from the DB or
    normalized via ``normalize_workspace`` before reaching this function),
    so cookie values are set directly without runtime sanitization.
    """
    canonical_visible = _ordered_unique(
        [normalize_workspace(w) for w in visible_workspaces if normalize_workspace(w)]
    )
    visible = set(canonical_visible)
    if not visible:
        response.delete_cookie("dlightrag_workspace", path="/")
        response.delete_cookie("dlightrag_workspace_ids", path="/")
        return

    canonical_active = _ordered_unique(
        [
            normalize_workspace(w)
            for w in (active_workspaces or [])
            if normalize_workspace(w) in visible
        ]
    )
    active = canonical_active
    if not active:
        fallback = (
            normalize_workspace(primary_workspace)
            if primary_workspace and normalize_workspace(primary_workspace) in visible
            else _default_workspace(canonical_visible)
        )
        active = [fallback] if fallback else []
    primary = (
        normalize_workspace(primary_workspace)
        if primary_workspace and normalize_workspace(primary_workspace) in active
        else active[0]
    )
    joined = ",".join(active)
    secure = request.url.scheme == "https"
    response.set_cookie(
        key="dlightrag_workspace",
        value=primary,
        httponly=False,
        samesite="lax",
        secure=secure,
        path="/",
    )
    response.set_cookie(
        key="dlightrag_workspace_ids",
        value=joined,
        httponly=False,
        samesite="lax",
        secure=secure,
        path="/",
    )


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


class WebWorkspacesPage(ClientContractModel):
    workspaces: list[WebBootstrapWorkspace]
    next_cursor: str | None = None


@router.get("/workspaces", response_model=WebWorkspacesPage)
async def list_workspaces_page(
    request: Request,
    limit: Annotated[
        int,
        Query(ge=1, le=WORKSPACE_CATALOG_PAGE_MAX_LIMIT),
    ] = WORKSPACE_CATALOG_PAGE_DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=1024)] = None,
) -> dict[str, Any]:
    """Return one bounded ascending workspace-catalog page for the picker."""
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
    records = await filter_web_workspace_records(
        request,
        AccessAction.WORKSPACE_QUERY,
        list(page.items),
    )
    return {
        "workspaces": [project_workspace_record(record) for record in records],
        "next_cursor": (
            application.corpora.workspace_catalog_cursor_codec.encode(page.next_cursor)
            if page.next_cursor is not None
            else None
        ),
    }


@router.post("/workspaces/create")
async def create_workspace(
    request: Request,
    workspace_name: str = Form(default=""),
):
    """Create a new workspace and return updated workspace list."""
    from dlightrag.application.corpus_admin import validate_workspace_name

    application = get_application(request)

    try:
        name = validate_workspace_name(workspace_name)
    except ValueError as exc:
        return _error(str(exc))

    ws = normalize_workspace(name)
    await enforce_web_access(request, AccessAction.WORKSPACE_CREATE, ws)

    # Duplicate check
    existing = await application.corpora.list_workspaces()
    if ws in existing:
        return _error(f"Workspace '{name}' already exists", status_code=409)

    # Initialize workspace (creates the WorkspaceRag)
    try:
        await application.corpora.create_workspace(ws, display_name=name)
    except Exception:
        logger.exception("Workspace creation failed")
        return _error(
            "Failed to create workspace; see server logs for details.",
            status_code=500,
        )

    response = JSONResponse({"workspace": ws, "display_name": name})
    visible_workspaces = await _visible_workspace_names(request, application)
    _set_workspace_cookies(
        response,
        request,
        visible_workspaces,
        active_workspaces=[ws],
        primary_workspace=ws,
    )
    return response


@router.post("/workspaces/reset", response_model=WebCorpusRunReceipt, status_code=202)
async def reset_workspace(
    request: Request,
    workspace_name: str = Form(default=""),
    confirm_name: str = Form(default=""),
):
    """Accept a full Corpus Reset after type-to-confirm verification."""
    application = get_application(request)
    name = workspace_name.strip()
    confirm = confirm_name.strip()

    if not name:
        return _error("Workspace name cannot be empty")
    if normalize_workspace(name) != normalize_workspace(confirm):
        return _error("Confirmation name does not match")

    ws = normalize_workspace(name)
    await enforce_web_access(request, AccessAction.WORKSPACE_RESET, ws)

    try:
        creation = await application.corpus_mutations.create_reset(
            workspace=ws,
            submitted_by=owner_id_from_user(getattr(request.state, "user_context", None)),
        )
    except Exception:
        logger.exception("Workspace reset Run acceptance failed")
        return _error(
            "Failed to accept Corpus Reset; see server logs for details.",
            status_code=503,
        )
    return corpus_run_receipt(creation.run, workspace=ws)
