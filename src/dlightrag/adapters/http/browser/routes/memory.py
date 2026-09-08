# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web Adapter for owner Profile Memory settings and mutations."""

from typing import Annotated, Any, Literal

from dlightrag_memory import MemoryProvenance
from dlightrag_memory.errors import MemoryUnavailableError, MemoryWriteRejectedError
from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from dlightrag.adapters.http.browser.deps import get_application
from dlightrag.application.access import owner_id_from_user
from dlightrag.application.memory import MemoryDisabledError, MemorySettings
from dlightrag.application.memory.projections import memory_receipt_payload

router = APIRouter()
IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)]


def _user(request: Request) -> Any:
    return getattr(request.state, "user_context", None)


class MemorySettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(description="Whether this owner's Profile Memory capability is active.")


class RememberMemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["preference", "fact"]
    body: str = Field(min_length=1, max_length=500)
    supersedes_id: str | None = None


@router.get("/memory/settings")
async def memory_settings(request: Request) -> dict[str, object]:
    application = get_application(request)
    user = _user(request)
    try:
        settings = await application.memory.settings(
            owner_id=owner_id_from_user(user), auth_mode=user.auth_mode
        )
    except MemoryUnavailableError as exc:
        raise HTTPException(status_code=403, detail=exc.public_message) from exc
    return _settings(settings)


@router.put("/memory/settings")
async def update_memory_settings(request: Request, body: MemorySettingsInput) -> dict[str, object]:
    application = get_application(request)
    user = _user(request)
    try:
        settings = await application.memory.set_enabled(
            owner_id=owner_id_from_user(user), auth_mode=user.auth_mode, enabled=body.enabled
        )
    except MemoryUnavailableError as exc:
        raise HTTPException(status_code=403, detail=exc.public_message) from exc
    return _settings(settings)


@router.post("/memory")
async def remember_memory(
    request: Request, body: RememberMemoryInput, idempotency_key: IdempotencyKey
) -> dict[str, Any]:
    application = get_application(request)
    user = _user(request)
    try:
        receipt = await application.memory.remember(
            owner_id=owner_id_from_user(user),
            auth_mode=user.auth_mode,
            kind=body.kind,
            body=body.body,
            supersedes_id=body.supersedes_id,
            provenance=MemoryProvenance(origin_kind="management", origin_id=idempotency_key),
            idempotency_key=f"web:{idempotency_key}",
        )
    except (MemoryUnavailableError, MemoryDisabledError) as exc:
        raise _capability_error(exc) from exc
    except MemoryWriteRejectedError as exc:
        raise HTTPException(status_code=409, detail=exc.public_message) from exc
    return memory_receipt_payload(receipt)


@router.delete("/memory/{memory_id}")
async def forget_memory(
    memory_id: str, request: Request, idempotency_key: IdempotencyKey
) -> dict[str, Any]:
    application = get_application(request)
    user = _user(request)
    try:
        receipt = await application.memory.forget(
            owner_id=owner_id_from_user(user),
            auth_mode=user.auth_mode,
            memory_id=memory_id,
            provenance=MemoryProvenance(origin_kind="management", origin_id=idempotency_key),
            idempotency_key=f"web:{idempotency_key}",
        )
    except (MemoryUnavailableError, MemoryDisabledError) as exc:
        raise _capability_error(exc) from exc
    except MemoryWriteRejectedError as exc:
        raise HTTPException(status_code=409, detail=exc.public_message) from exc
    return memory_receipt_payload(receipt)


@router.post("/memory/changes/{change_id}/undo")
async def undo_memory_change(
    change_id: str, request: Request, idempotency_key: IdempotencyKey
) -> dict[str, Any]:
    application = get_application(request)
    user = _user(request)
    try:
        receipt = await application.memory.undo(
            owner_id=owner_id_from_user(user),
            auth_mode=user.auth_mode,
            change_id=change_id,
            provenance=MemoryProvenance(origin_kind="undo", origin_id=idempotency_key),
            idempotency_key=f"web:{idempotency_key}",
        )
    except (MemoryUnavailableError, MemoryDisabledError) as exc:
        raise _capability_error(exc) from exc
    except MemoryWriteRejectedError as exc:
        raise HTTPException(status_code=409, detail=exc.public_message) from exc
    return memory_receipt_payload(receipt)


@router.post("/memory/clear", status_code=status.HTTP_204_NO_CONTENT)
async def clear_memory(request: Request) -> None:
    application = get_application(request)
    user = _user(request)
    try:
        await application.memory.clear(owner_id=owner_id_from_user(user), auth_mode=user.auth_mode)
    except (MemoryUnavailableError, MemoryDisabledError) as exc:
        raise _capability_error(exc) from exc


def _settings(settings: MemorySettings) -> dict[str, object]:
    return {"enabled": settings.enabled, "active_count": settings.active_count}


def _capability_error(exc: Exception) -> HTTPException:
    status_code = 409 if isinstance(exc, MemoryDisabledError) else 403
    return HTTPException(status_code=status_code, detail=getattr(exc, "public_message", str(exc)))


__all__ = ["router"]
