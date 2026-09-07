# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Run-native Corpus Mutation action and upload acceptance routes."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, cast
from uuid import uuid7

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.datastructures import FormData
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.rest.models import (
    DeleteRequest,
    IngestRequest,
    ResetRequest,
    RetryRequest,
    RunDescriptor,
)
from dlightrag.application.access import (
    UserContext,
    corpus_mutation_access_action,
    owner_id_from_user,
)
from dlightrag.application.corpus_admin import (
    UnsafeUploadNameError,
    UploadTooLargeError,
    ingest_spec_from_payload,
    managed_local_ingest_documents,
    managed_local_ingest_path,
)
from dlightrag.application.runs import (
    IdempotencyKeyConflict,
    RunCapacityExceededError,
    RunCreation,
    RunRuntimeUnavailableError,
)

from .deps import enforce_access, get_application, idempotency_key, resolve_workspace
from .runs import run_descriptor

router = APIRouter(prefix="/runs/corpus", tags=["Corpus Mutation Runs"])

_MAX_UPLOAD_FILES = 100
_MAX_FORM_FIELDS = 8
_FORM_PART_MAX_BYTES = 1024 * 1024
_SINGLE_UPLOAD_FIELDS = frozenset(
    {"file", "workspace", "title", "author", "metadata", "content_sha256"}
)
_BATCH_UPLOAD_FIELDS = frozenset({"file", "workspace"})


def _required_idempotency_key(request: Request) -> str:
    value = idempotency_key(request)
    if value is None:
        raise HTTPException(status_code=400, detail="Idempotency-Key header is required")
    return value


async def _accept(call: Callable[[], Awaitable[RunCreation]]) -> dict[str, Any]:
    try:
        creation = await call()
    except IdempotencyKeyConflict:
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key was reused with a different Corpus Mutation request",
        ) from None
    except RunCapacityExceededError:
        raise HTTPException(status_code=503, detail="Run admission capacity is full") from None
    except RunRuntimeUnavailableError:
        raise HTTPException(status_code=503, detail="Run runtime is unavailable") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return run_descriptor(creation.run)


async def _authorize(
    request: Request,
    user: UserContext,
    workspace: str | None,
    action: str,
) -> tuple[Any, str, str]:
    application = get_application(request)
    canonical = resolve_workspace(workspace, request)
    await enforce_access(
        request,
        user,
        corpus_mutation_access_action(action),
        workspace=canonical,
    )
    return application, canonical, owner_id_from_user(user)


async def _ingest_action(
    body: IngestRequest,
    request: Request,
    user: UserContext,
    *,
    replace: bool,
) -> dict[str, Any]:
    application, workspace, submitted_by = await _authorize(
        request, user, body.workspace, "replace" if replace else "ingest"
    )
    spec = ingest_spec_from_payload(body).model_copy(update={"replace": replace})
    if spec.source_type == "local":
        try:
            path = managed_local_ingest_path(
                source_type=spec.source_type,
                path=spec.path,
                input_dir=application.config.input_dir_path,
                workspace=workspace,
            )
            documents = managed_local_ingest_documents(
                source_type=spec.source_type,
                documents=spec.documents,
                input_dir=application.config.input_dir_path,
                workspace=workspace,
            )
            spec = spec.model_copy(update={"path": path, "documents": documents})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
    key = _required_idempotency_key(request)
    return await _accept(
        lambda: application.corpus_mutations.create_ingest(
            workspace=workspace,
            spec=spec,
            submitted_by=submitted_by,
            idempotency_key=key,
        )
    )


@router.post("/ingest", response_model=RunDescriptor, status_code=202)
async def ingest(
    body: IngestRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    return await _ingest_action(body, request, user, replace=False)


@router.post("/replace", response_model=RunDescriptor, status_code=202)
async def replace(
    body: IngestRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    return await _ingest_action(body, request, user, replace=True)


@router.post("/delete", response_model=RunDescriptor, status_code=202)
async def delete(
    body: DeleteRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    application, workspace, submitted_by = await _authorize(request, user, body.workspace, "delete")
    if not any((body.file_paths, body.filenames, body.document_ids)):
        raise HTTPException(status_code=400, detail="At least one exact identifier is required")
    key = _required_idempotency_key(request)
    return await _accept(
        lambda: application.corpus_mutations.create_delete(
            workspace=workspace,
            submitted_by=submitted_by,
            file_paths=body.file_paths or (),
            filenames=body.filenames or (),
            document_ids=body.document_ids or (),
            idempotency_key=key,
        )
    )


@router.post("/retry", response_model=RunDescriptor, status_code=202)
async def retry(
    body: RetryRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    application, workspace, submitted_by = await _authorize(request, user, body.workspace, "retry")
    selector = None if body.document_ids else body.selector
    key = _required_idempotency_key(request)
    return await _accept(
        lambda: application.corpus_mutations.create_retry(
            workspace=workspace,
            submitted_by=submitted_by,
            document_ids=body.document_ids or (),
            selector=selector,
            idempotency_key=key,
        )
    )


@router.post("/reset", response_model=RunDescriptor, status_code=202)
async def reset(
    body: ResetRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    application, workspace, submitted_by = await _authorize(request, user, body.workspace, "reset")
    key = _required_idempotency_key(request)
    return await _accept(
        lambda: application.corpus_mutations.create_reset(
            workspace=workspace,
            submitted_by=submitted_by,
            supersedes_run_id=body.supersedes_run_id,
            idempotency_key=key,
        )
    )


def _text_field(form: FormData, name: str) -> str | None:
    value = form.get(name)
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"{name} must be a text field")
    return value


async def _parse_upload_form(
    request: Request,
    *,
    batch: bool,
) -> FormData:
    try:
        form = await request.form(
            max_files=_MAX_UPLOAD_FILES if batch else 1,
            max_fields=_MAX_FORM_FIELDS,
            max_part_size=_FORM_PART_MAX_BYTES,
        )
    except StarletteHTTPException as exc:
        detail = str(exc.detail)
        if exc.status_code == 400 and detail.startswith(
            ("Too many files.", "Too many fields.", "Part exceeded maximum size")
        ):
            raise HTTPException(status_code=413, detail=detail) from exc
        raise
    allowed = _BATCH_UPLOAD_FIELDS if batch else _SINGLE_UPLOAD_FIELDS
    unexpected = sorted({key for key, _ in form.multi_items()} - allowed)
    if unexpected:
        await form.close()
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected multipart field(s): {', '.join(unexpected)}",
        )
    return form


async def _upload_action(
    request: Request,
    user: UserContext,
    *,
    replace: bool,
    batch: bool,
) -> dict[str, Any]:
    form = await _parse_upload_form(request, batch=batch)
    key = _required_idempotency_key(request)
    run_id = str(uuid7())
    application: Any | None = None
    staged_workspace: str | None = None

    async def _discard_unaccepted_stage() -> None:
        if application is None or staged_workspace is None:
            return
        await application.corpus_mutations.discard_staged_run(
            workspace=staged_workspace,
            run_id=run_id,
        )

    try:
        files = form.getlist("file")
        if not files or any(not isinstance(item, StarletteUploadFile) for item in files):
            raise HTTPException(status_code=400, detail="At least one file is required")
        if not batch and len(files) != 1:
            raise HTTPException(status_code=400, detail="Exactly one file is required")
        uploads = [cast(StarletteUploadFile, item) for item in files]
        if any(not item.filename for item in uploads):
            raise HTTPException(status_code=400, detail="Every file requires a filename")
        authorized_application, workspace, submitted_by = await _authorize(
            request,
            user,
            _text_field(form, "workspace"),
            "replace" if replace else "ingest",
        )
        application = authorized_application
        staged_workspace = workspace
        max_bytes = (
            authorized_application.config.max_upload_batch_bytes
            if batch
            else authorized_application.config.corpus.ingestion.max_upload_bytes
        )
        staged = []
        for item in uploads:
            staged.append(
                await authorized_application.corpus_mutations.stage_upload(
                    workspace=workspace,
                    run_id=run_id,
                    filename=str(item.filename),
                    reader=item,
                    max_bytes=max_bytes,
                    content_sha256=None if batch else _text_field(form, "content_sha256"),
                )
            )
        if sum(item.size_bytes for item in staged) > max_bytes:
            raise UploadTooLargeError(f"upload exceeds {max_bytes} bytes")

        if batch:
            return await _accept(
                lambda: authorized_application.corpus_mutations.create_staged_batch(
                    workspace=workspace,
                    staged=staged,
                    submitted_by=submitted_by,
                    idempotency_key=key,
                    replace=replace,
                )
            )
        metadata_text = _text_field(form, "metadata")
        metadata: dict[str, Any] | None = None
        if metadata_text is not None:
            try:
                decoded = json.loads(metadata_text)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON") from None
            if not isinstance(decoded, dict):
                raise HTTPException(status_code=400, detail="metadata must be a JSON object")
            metadata = decoded
        return await _accept(
            lambda: authorized_application.corpus_mutations.create_staged_ingest(
                workspace=workspace,
                staged=staged[0],
                submitted_by=submitted_by,
                idempotency_key=key,
                title=_text_field(form, "title"),
                author=_text_field(form, "author"),
                metadata=metadata,
                replace=replace,
            )
        )
    except UnsafeUploadNameError as exc:
        await _discard_unaccepted_stage()
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except UploadTooLargeError as exc:
        await _discard_unaccepted_stage()
        raise HTTPException(status_code=413, detail=str(exc)) from None
    except ValueError as exc:
        await _discard_unaccepted_stage()
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except BaseException:
        await _discard_unaccepted_stage()
        raise
    finally:
        await form.close()


@router.post("/ingest/upload", response_model=RunDescriptor, status_code=202)
async def ingest_upload(
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    return await _upload_action(request, user, replace=False, batch=False)


@router.post("/replace/upload", response_model=RunDescriptor, status_code=202)
async def replace_upload(
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    return await _upload_action(request, user, replace=True, batch=False)


@router.post("/ingest/uploads", response_model=RunDescriptor, status_code=202)
async def ingest_uploads(
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    return await _upload_action(request, user, replace=False, batch=True)


@router.post("/replace/uploads", response_model=RunDescriptor, status_code=202)
async def replace_uploads(
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    return await _upload_action(request, user, replace=True, batch=True)
