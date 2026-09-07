# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Common dependencies for API routes."""

from collections.abc import Sequence

from fastapi import HTTPException, Request

from dlightrag.adapters.http.application import get_application
from dlightrag.application.access import (
    AccessControl,
    AccessDeniedError,
    AccessGate,
    AccessSubject,
    NoQueryableWorkspacesError,
    WorkspaceRecord,
    WorkspaceSelectionConflictError,
    access_control_from_settings,
)
from dlightrag.application.corpus_admin import normalize_workspace, normalize_workspace_ids
from dlightrag.application.errors import WorkspaceWriteFencedError
from dlightrag.application.settings import access_settings


def raise_fenced_http(exc: WorkspaceWriteFencedError) -> HTTPException:
    """Translate a promotion fence refusal into retryable 409 + Retry-After.

    Retry-After rounds the remaining fence duration UP: a client retrying
    early must still meet the fence.
    """
    import math

    return HTTPException(
        status_code=409,
        detail=str(exc),
        headers={"Retry-After": str(max(1, math.ceil(exc.retry_after_seconds)))},
    )


def idempotency_key(request: Request) -> str | None:
    """Return a meaningful caller replay key, preserving its exact value."""
    value = request.headers.get("Idempotency-Key")
    return value if value and value.strip() else None


def resolve_workspace(ws: str | None, request: Request) -> str:
    workspace = get_application(request).config.deployment.workspace
    return normalize_workspace(ws or workspace)


def get_access_control(request: Request) -> AccessControl:
    return getattr(request.app.state, "access_control", None) or access_control_from_settings(
        access_settings(get_application(request).config)
    )


def get_access_gate(request: Request, subject: AccessSubject) -> AccessGate:
    return AccessGate(get_access_control(request), subject)


async def enforce_access(
    request: Request,
    user: AccessSubject,
    action: str,
    *,
    workspace: str | None = None,
) -> None:
    try:
        await get_access_gate(request, user).check(action, workspace=workspace)
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None


async def filter_workspace_records(
    request: Request,
    user: AccessSubject,
    action: str,
    records: Sequence[WorkspaceRecord],
) -> list[WorkspaceRecord]:
    return await get_access_gate(request, user).filter_workspace_records(action, records)


async def authorized_workspaces(
    request: Request,
    user: AccessSubject,
    workspaces: list[str],
    action: str,
) -> set[str]:
    """Return the subset of workspaces this caller may use for one action."""
    return await get_access_gate(request, user).authorized_workspace_ids(action, workspaces)


async def resolve_authorized_query_workspaces(
    request: Request,
    user: AccessSubject,
    *,
    workspaces: list[str] | None,
    all_workspaces: bool,
) -> list[str]:
    """Resolve query targets after applying the caller's existing ACL."""
    try:
        return await get_access_gate(request, user).resolve_query_workspaces(
            get_application(request).corpora,
            default_workspace=normalize_workspace(
                get_application(request).config.deployment.workspace
            ),
            workspaces=normalize_workspace_ids(workspaces) if workspaces is not None else None,
            all_workspaces=all_workspaces,
        )
    except NoQueryableWorkspacesError:
        raise HTTPException(
            status_code=403,
            detail="No workspaces are available for query",
        ) from None
    except WorkspaceSelectionConflictError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
