# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Authenticated browser bootstrap contract shared by the page and Lit app."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request

from dlightrag.adapters.http.browser.attachment_models import SUPPORTED_DOCUMENT_EXTENSIONS
from dlightrag.adapters.http.browser.deps import (
    filter_web_workspace_records,
    get_application,
    get_workspace,
)
from dlightrag.adapters.http.browser.workspace_models import (
    WebBootstrapWorkspace,
    project_workspace_record,
)
from dlightrag.application.access import AccessAction, WorkspaceRecord
from dlightrag.application.corpus_admin import (
    WORKSPACE_CATALOG_PAGE_MAX_LIMIT,
    WorkspaceCatalogPageRequest,
    normalize_workspace,
)
from dlightrag.engine.answer.client_contracts import ClientContractModel
from dlightrag.engine.answer.image_capability import ImageCapabilityStatus

router = APIRouter()


class WebBootstrapUnavailableError(RuntimeError):
    """The workspace inventory required for browser startup is unavailable."""


class WebAttachmentBootstrap(ClientContractModel):
    count_limit: int
    image_max_bytes: int
    document_max_bytes: int
    extensions: list[str]
    image_capability: ImageCapabilityStatus
    image_limit: int
    accept: str


class WebBootstrap(ClientContractModel):
    contract_version: Literal[1] = 1
    workspaces: list[WebBootstrapWorkspace]
    workspaces_next_cursor: str | None = None
    primary_workspace: str
    active_workspaces: list[str]
    known_workspaces: list[str]
    answer_attachments: WebAttachmentBootstrap
    active_html_preview_enabled: bool


async def build_web_bootstrap(
    request: Request,
    workspace: str,
) -> WebBootstrap:
    """Build the one authorized startup snapshot consumed by the browser."""
    application = get_application(request)
    capabilities = await application.answers.capabilities()
    records: list[WorkspaceRecord]
    try:
        records = await application.corpora.alist_workspace_records()
    except Exception as exc:
        raise WebBootstrapUnavailableError from exc
    records = await filter_web_workspace_records(
        request,
        AccessAction.WORKSPACE_QUERY,
        records,
    )

    # Authorization inputs (known/active/primary) keep the full catalog. Only
    # the user-visible array is a bounded first page over the full catalog
    # ordering; its continuation cursor is minted from the full page so a later
    # load-more page never re-delivers rows the first page already displayed.
    try:
        catalog_page = await application.corpora.list_workspace_records_page(
            page=WorkspaceCatalogPageRequest(limit=WORKSPACE_CATALOG_PAGE_MAX_LIMIT)
        )
        page_records = await filter_web_workspace_records(
            request,
            AccessAction.WORKSPACE_QUERY,
            list(catalog_page.items),
        )
        workspaces = [project_workspace_record(record) for record in page_records]
        next_cursor = (
            application.corpora.workspace_catalog_cursor_codec.encode(catalog_page.next_cursor)
            if catalog_page.next_cursor is not None
            else None
        )
    except Exception as exc:
        if isinstance(exc, WebBootstrapUnavailableError):
            raise
        raise WebBootstrapUnavailableError from exc

    known = set(record["workspace"] for record in records)
    if next_cursor is None:
        # Degraded catalog fallback: a synthetic default record the full
        # authorization list carries but the registry page could not may only
        # be appended when the catalog traversal is exhausted, preserving order.
        shown = {workspace.workspace for workspace in workspaces}
        for record in records:
            if record["workspace"] not in shown:
                workspaces.append(project_workspace_record(record))

    # Active/primary computation keeps the full authorized catalog; the bounded
    # workspaces array above is presentation-only.
    authorized_full = [record["workspace"] for record in records]
    active_raw = request.cookies.get("dlightrag_workspace_ids", "")
    active = [normalize_workspace(item.strip()) for item in active_raw.split(",") if item.strip()]
    active = [item for item in active if item in known]

    primary = normalize_workspace(request.cookies.get("dlightrag_workspace", workspace))
    if not active:
        active = authorized_full
    if primary not in known:
        primary = (
            "default" if "default" in known else (authorized_full[0] if authorized_full else "")
        )

    capability = capabilities.answer
    if capability is None:
        capability_status = "unknown"
        effective_current_upload_limit = 0
    else:
        capability_status = capability.status
        effective_current_upload_limit = capability.effective_max_images

    extensions = sorted(SUPPORTED_DOCUMENT_EXTENSIONS)
    attachment_limit = application.config.answer.generation.max_attachment_bytes
    return WebBootstrap(
        workspaces=workspaces,
        workspaces_next_cursor=next_cursor,
        primary_workspace=primary,
        active_workspaces=active,
        known_workspaces=authorized_full,
        answer_attachments=WebAttachmentBootstrap(
            count_limit=application.config.answer.generation.max_attachments,
            image_max_bytes=attachment_limit,
            document_max_bytes=attachment_limit,
            extensions=extensions,
            image_capability=capability_status,
            image_limit=effective_current_upload_limit,
            accept=",".join(["image/*", *(f".{extension}" for extension in extensions)]),
        ),
        active_html_preview_enabled=(
            application.config.answer.conversations.active_html_preview_enabled
        ),
    )


@router.get("/bootstrap", response_model=WebBootstrap)
async def browser_bootstrap(
    request: Request,
    workspace: str = Depends(get_workspace),
) -> WebBootstrap:
    try:
        return await build_web_bootstrap(request, workspace)
    except WebBootstrapUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Web application bootstrap is unavailable",
        ) from None


__all__ = [
    "WebAttachmentBootstrap",
    "WebBootstrap",
    "WebBootstrapUnavailableError",
    "build_web_bootstrap",
    "router",
]
