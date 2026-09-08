# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for file management."""

import logging
from pathlib import Path
from typing import Annotated, Any, NoReturn
from uuid import uuid7

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from dlightrag.adapters.http.browser.deps import enforce_web_access, get_application, get_workspace
from dlightrag.adapters.http.browser.file_models import (
    WebCorpusRunReceipt,
    WebFailedFileItem,
    WebFailedFilesPage,
    WebFileItem,
    WebFilePanelSnapshot,
)
from dlightrag.adapters.http.browser.routes.corpus_runs import corpus_run_receipt
from dlightrag.adapters.http.source_download import source_download_response
from dlightrag.application.access import AccessAction, owner_id_from_user
from dlightrag.application.corpus_admin import (
    FILE_PANEL_PAGE_DEFAULT_LIMIT,
    FILE_PANEL_PAGE_MAX_LIMIT,
    FilePanelCursorError,
    FilePanelPageRequest,
    UnsafeUploadNameError,
    UploadTooLargeError,
    safe_log_text,
)
from dlightrag.application.runs import RunAdmissionLimitExceededError

logger = logging.getLogger(__name__)

router = APIRouter()
_FAILED_PAGE_DEFAULT_LIMIT = 5
_MAX_BROWSER_UPLOAD_FILES = 100


@router.get("/files/raw/{document_id:path}", response_model=None)
async def download_source(
    document_id: str,
    request: Request,
    workspace: str | None = Query(default=None),
) -> FileResponse | RedirectResponse:
    """Download one source document through the Web session boundary."""
    from dlightrag.application.corpus_admin import normalize_workspace

    safe_workspace = normalize_workspace(
        workspace or get_application(request).config.deployment.workspace
    )
    try:
        await enforce_web_access(
            request,
            AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
            safe_workspace,
        )
    except HTTPException as exc:
        if exc.status_code == 403:
            logger.info(
                "source_download_projection_outcome",
                extra={"outcome": "unauthorized", "workspace": safe_log_text(safe_workspace)},
            )
        raise

    return await source_download_response(
        get_application(request).corpora,
        workspace=safe_workspace,
        document_id=document_id,
    )


def _resolve_workspace(requested: str | None, cookie_workspace: str) -> str:
    from dlightrag.application.corpus_admin import normalize_workspace

    if not requested:
        return cookie_workspace
    normalized = normalize_workspace(requested)
    return normalized or cookie_workspace


async def _resolve_registered_workspace(
    request: Request,
    workspace: str,
) -> str | None:
    """Return the requested workspace after one bounded registry lookup."""
    return workspace if await _workspace_is_registered(request, workspace) else None


async def _workspace_is_registered(request: Request, workspace: str) -> bool:
    """Return whether a workspace is registered; fail open on registry outages."""
    try:
        return bool(await get_application(request).corpora.workspace_exists(workspace))
    except Exception:
        return True


def _stale_workspace() -> NoReturn:
    raise HTTPException(
        status_code=409,
        detail="Workspace no longer exists. Refresh and choose an existing workspace.",
    )


def _file_view_models(files: list[dict[str, Any]]) -> list[WebFileItem]:
    rows: list[WebFileItem] = []
    for item in files:
        file_path = str(item.get("file_path") or "")
        file_name = str(item.get("file_name") or item.get("filename") or "")
        if not file_name and file_path:
            file_name = Path(file_path).name
        if not file_name:
            file_name = str(item.get("doc_id") or "Untitled file")
        rows.append(WebFileItem(file_name=file_name, file_path=file_path))
    return rows


def _failed_file_view_models(files: list[dict[str, Any]]) -> list[WebFailedFileItem]:
    rows: list[WebFailedFileItem] = []
    for item in files:
        file_path = str(item.get("file_path") or "")
        rows.append(
            WebFailedFileItem(
                document_id=str(item.get("doc_id") or ""),
                file_name=Path(file_path).name or "Untitled file",
                error=str(item.get("error") or ""),
                updated_at=str(item.get("updated_at") or ""),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# GET /web/api/files — file list panel content
# ---------------------------------------------------------------------------


@router.get("/files", response_model=WebFilePanelSnapshot)
async def file_list(
    request: Request,
    workspace: str = Depends(get_workspace),
    workspace_name: str | None = Query(default=None, alias="workspace"),
    limit: Annotated[int, Query(ge=1, le=FILE_PANEL_PAGE_MAX_LIMIT)] = (
        FILE_PANEL_PAGE_DEFAULT_LIMIT
    ),
    cursor: Annotated[str | None, Query(min_length=1, max_length=1024)] = None,
) -> WebFilePanelSnapshot:
    """Return one typed Files panel snapshot."""
    selected_workspace = _resolve_workspace(workspace_name, workspace)
    selected_workspace = await _resolve_registered_workspace(request, selected_workspace)
    if selected_workspace is None:
        _stale_workspace()
    await enforce_web_access(request, AccessAction.WORKSPACE_LIST_FILES, selected_workspace)
    application = get_application(request)
    try:
        decoded_cursor = (
            application.corpora.file_panel_cursor_codec.decode(cursor)
            if cursor is not None
            else None
        )
        if decoded_cursor is not None and decoded_cursor.workspace != selected_workspace:
            raise FilePanelCursorError("file-panel cursor belongs to another workspace")
        if decoded_cursor is not None and decoded_cursor.view != "processed":
            raise FilePanelCursorError("file-panel cursor belongs to another view")
        page = FilePanelPageRequest(limit=limit, cursor=decoded_cursor)
    except (FilePanelCursorError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return await _file_panel_snapshot(request, selected_workspace, page=page)


async def _file_panel_snapshot(
    request: Request,
    workspace: str,
    *,
    page: FilePanelPageRequest | None = None,
) -> WebFilePanelSnapshot:
    application = get_application(request)
    try:
        snapshot = await application.corpora.file_panel_snapshot(workspace, page=page)
    except Exception:
        logger.exception(
            "Could not read Files panel snapshot for workspace %s",
            safe_log_text(workspace),
        )
        raise HTTPException(status_code=503, detail="Files are temporarily unavailable") from None
    next_cursor = snapshot.get("next_cursor")
    return WebFilePanelSnapshot(
        workspace=workspace,
        files=_file_view_models(list(snapshot.get("files") or [])),
        next_cursor=(
            application.corpora.file_panel_cursor_codec.encode(next_cursor)
            if next_cursor is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Failed-document recovery — bounded listing + durable Corpus Mutation Run
# ---------------------------------------------------------------------------


@router.get("/files/failed", response_model=WebFailedFilesPage)
async def failed_file_list(
    request: Request,
    workspace: str = Depends(get_workspace),
    workspace_name: str | None = Query(default=None, alias="workspace"),
    limit: Annotated[int, Query(ge=1, le=FILE_PANEL_PAGE_MAX_LIMIT)] = (_FAILED_PAGE_DEFAULT_LIMIT),
    cursor: Annotated[str | None, Query(min_length=1, max_length=1024)] = None,
) -> WebFailedFilesPage:
    selected_workspace = _resolve_workspace(workspace_name, workspace)
    selected_workspace = await _resolve_registered_workspace(request, selected_workspace)
    if selected_workspace is None:
        _stale_workspace()
    await enforce_web_access(request, AccessAction.WORKSPACE_LIST_FILES, selected_workspace)
    application = get_application(request)
    try:
        decoded_cursor = (
            application.corpora.file_panel_cursor_codec.decode(cursor)
            if cursor is not None
            else None
        )
        if decoded_cursor is not None and decoded_cursor.workspace != selected_workspace:
            raise FilePanelCursorError("file-panel cursor belongs to another workspace")
        if decoded_cursor is not None and decoded_cursor.view != "failed":
            raise FilePanelCursorError("file-panel cursor belongs to another view")
        page = FilePanelPageRequest(limit=limit, cursor=decoded_cursor)
    except (FilePanelCursorError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    try:
        snapshot = await application.corpora.failed_file_snapshot(
            selected_workspace,
            page=page,
        )
    except Exception:
        logger.exception(
            "Could not read failed files for workspace %s",
            safe_log_text(selected_workspace),
        )
        raise HTTPException(
            status_code=503,
            detail="Failed-document status is temporarily unavailable",
        ) from None

    next_cursor = snapshot.get("next_cursor")
    return WebFailedFilesPage(
        workspace=selected_workspace,
        failed=_failed_file_view_models(list(snapshot.get("failed") or [])),
        next_cursor=(
            application.corpora.file_panel_cursor_codec.encode(next_cursor)
            if next_cursor is not None
            else None
        ),
    )


@router.post("/files/retry", response_model=WebCorpusRunReceipt, status_code=202)
async def start_failed_file_retry(
    request: Request,
    workspace: str = Depends(get_workspace),
    workspace_name: str | None = Query(default=None, alias="workspace"),
) -> WebCorpusRunReceipt:
    selected_workspace = _resolve_workspace(workspace_name, workspace)
    if not await _workspace_is_registered(request, selected_workspace):
        _stale_workspace()
    await enforce_web_access(request, AccessAction.WORKSPACE_INGEST, selected_workspace)
    try:
        creation = await get_application(request).corpus_mutations.create_retry(
            workspace=selected_workspace,
            selector="all_retryable",
            submitted_by=owner_id_from_user(getattr(request.state, "user_context", None)),
        )
    except RunAdmissionLimitExceededError:
        raise HTTPException(
            status_code=503,
            detail="Deployment-wide nonterminal admission limit reached",
        ) from None
    except Exception:
        logger.exception(
            "Could not accept failed-document retry for workspace %s",
            safe_log_text(selected_workspace),
        )
        raise HTTPException(
            status_code=503, detail="Document recovery could not be accepted"
        ) from None
    return corpus_run_receipt(creation.run, workspace=selected_workspace)


# ---------------------------------------------------------------------------
# POST /web/api/files/upload — upload staging + durable Corpus Mutation Run
# ---------------------------------------------------------------------------


@router.post("/files/upload", response_model=WebCorpusRunReceipt, status_code=202)
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    workspace_name: str | None = Form(default=None, alias="workspace"),
    content_sha256: str | None = Form(default=None),
    workspace: str = Depends(get_workspace),
):
    """Stage uploaded files and accept one durable Corpus Mutation Run."""
    application = get_application(request)
    cfg = application.config
    # Per-file document cap is the single shared limit used by every ingest
    # path (Run upload, URL, web upload): one document may not exceed it.
    # The larger per-request cap is a temp-directory guard for multi-file
    # (folder) uploads.
    per_file_max_bytes = cfg.corpus.ingestion.max_upload_bytes
    batch_max_bytes = cfg.max_upload_batch_bytes
    per_file_max_mb = per_file_max_bytes // (1024 * 1024)

    selected_workspace = _resolve_workspace(workspace_name, workspace)
    if not await _workspace_is_registered(request, selected_workspace):
        _stale_workspace()
    await enforce_web_access(request, AccessAction.WORKSPACE_INGEST, selected_workspace)

    run_id = str(uuid7())
    stage_owned = True
    staged = []
    total_bytes = 0
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No valid files selected")
        if len(files) > _MAX_BROWSER_UPLOAD_FILES:
            raise HTTPException(status_code=413, detail="Too many upload files")
        if content_sha256 is not None and len(files) != 1:
            raise HTTPException(
                status_code=400,
                detail="content_sha256 is supported only for a single upload",
            )
        for upload in files:
            remaining = batch_max_bytes - total_bytes
            if remaining <= 0:
                raise UploadTooLargeError("upload batch exceeds configured maximum")
            item = await application.corpus_mutations.stage_upload(
                run_id=run_id,
                workspace=selected_workspace,
                filename=upload.filename or "",
                reader=upload,
                max_bytes=min(per_file_max_bytes, remaining),
                content_sha256=content_sha256,
            )
            staged.append(item)
            total_bytes += item.size_bytes
        if total_bytes > batch_max_bytes:
            raise UploadTooLargeError("upload batch exceeds configured maximum")
        try:
            creation = await application.corpus_mutations.create_staged_batch(
                workspace=selected_workspace,
                staged=staged,
                submitted_by=owner_id_from_user(getattr(request.state, "user_context", None)),
            )
        except RunAdmissionLimitExceededError:
            raise HTTPException(
                status_code=503,
                detail="Deployment-wide nonterminal admission limit reached",
            ) from None
        except Exception:
            logger.exception(
                "Failed to accept ingest Run for workspace %s",
                safe_log_text(selected_workspace),
            )
            raise HTTPException(
                status_code=503,
                detail="Upload could not be accepted. Please retry.",
            ) from None
        stage_owned = False
        return corpus_run_receipt(
            creation.run,
            workspace=selected_workspace,
            file_count=len(staged),
        )
    except UnsafeUploadNameError as exc:
        logger.warning("Rejected upload with unsafe filename: %s", exc)
        raise HTTPException(status_code=400, detail="Upload contains an unsafe filename") from None
    except UploadTooLargeError:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload exceeds limit ({per_file_max_mb} MB per file, "
                f"{cfg.interfaces.max_upload_size_mb} MB per request)"
            ),
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except HTTPException:
        raise
    except Exception:
        logger.exception("Upload staging failed")
        raise HTTPException(status_code=500, detail="Upload failed. Please try again.") from None
    finally:
        if stage_owned:
            try:
                await application.corpus_mutations.discard_staged_run(
                    workspace=selected_workspace,
                    run_id=run_id,
                )
            except Exception:
                logger.exception("Failed to discard rejected browser upload stage")


# ---------------------------------------------------------------------------
# DELETE /web/api/files
# ---------------------------------------------------------------------------


@router.delete("/files", response_model=WebCorpusRunReceipt, status_code=202)
async def delete_files(
    request: Request,
    workspace: str = Depends(get_workspace),
):
    """Accept an exact file deletion as a durable Corpus Mutation Run."""
    file_path = request.query_params.get("file_path", "")
    file_paths = [file_path] if file_path else []
    application = get_application(request)
    selected_workspace = _resolve_workspace(request.query_params.get("workspace"), workspace)
    if not await _workspace_is_registered(request, selected_workspace):
        _stale_workspace()
    await enforce_web_access(request, AccessAction.WORKSPACE_DELETE_FILES, selected_workspace)

    if not file_paths:
        raise HTTPException(status_code=422, detail="file_path is required")
    try:
        creation = await application.corpus_mutations.create_delete(
            workspace=selected_workspace,
            file_paths=file_paths,
            submitted_by=owner_id_from_user(getattr(request.state, "user_context", None)),
        )
    except RunAdmissionLimitExceededError:
        raise HTTPException(
            status_code=503,
            detail="Deployment-wide nonterminal admission limit reached",
        ) from None
    except Exception:
        logger.exception("Delete Run acceptance failed")
        raise HTTPException(status_code=503, detail="Delete could not be accepted") from None

    return corpus_run_receipt(creation.run, workspace=selected_workspace)
