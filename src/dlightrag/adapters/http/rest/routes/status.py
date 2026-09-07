# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""System status and health API routes.

``/health`` is liveness only: it answers from in-process state and never touches
PostgreSQL, so an unauthenticated poll loop cannot turn it into database load.
``/ready`` projects the application-owned readiness verdict. Composition injects
the database/corpus probe; this transport imports no storage implementation.
"""

from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from dlightrag.adapters.http.application import get_application
from dlightrag.application.config import ServiceRole
from dlightrag.application.health import ApplicationHealth

router = APIRouter()


class _StatusModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class HealthStorageResponse(_StatusModel):
    vector: str
    graph: str
    kv: str
    doc_status: str


class HealthComponentResponse(_StatusModel):
    status: Literal["healthy", "degraded", "starting", "unknown", "stopped"]
    detail: str | None = None


class AnswerImageCapabilityResponse(_StatusModel):
    status: str
    effective_max_images: int
    configured_ceiling: int
    model: str | None = None


class HealthResponse(_StatusModel):
    status: Literal["healthy", "degraded"]
    rag_initialized: bool
    service_role: ServiceRole
    crafted_by: str
    maintained_by: str
    storage: HealthStorageResponse
    components: dict[str, HealthComponentResponse]
    warnings: list[str] | None = None
    answer_image_capability: AnswerImageCapabilityResponse | None = None


class ReadinessResponse(_StatusModel):
    status: Literal["ready", "not_ready"]
    service_role: ServiceRole
    detail: str | None = None


def _application_health(request: Request) -> ApplicationHealth:
    return get_application(request).health


def _not_ready(*, service_role: ServiceRole, detail: str) -> JSONResponse:
    payload = ReadinessResponse(
        status="not_ready",
        service_role=service_role,
        detail=detail,
    )
    return JSONResponse(status_code=503, content=payload.model_dump(exclude_none=True))


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
async def health(request: Request) -> dict[str, object]:
    """Report process liveness and the capabilities this build exposes."""
    config = get_application(request).config
    application_health = _application_health(request)

    warnings = application_health.warnings
    status: dict[str, object] = {
        "status": "degraded" if application_health.is_degraded else "healthy",
        "rag_initialized": application_health.is_ready,
        "service_role": config.deployment.service_role,
        "crafted_by": "hllyu",
        "maintained_by": "HanlianLyu",
        "storage": {
            "vector": config.storage.lightrag.vector_storage,
            "graph": config.storage.lightrag.graph_storage,
            "kv": config.storage.lightrag.kv_storage,
            "doc_status": config.storage.lightrag.doc_status_storage,
        },
        "components": application_health.components,
        "answer_image_capability": application_health.answer_image_capability,
    }
    if warnings:
        status["warnings"] = warnings
    return status


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    response_model_exclude_none=True,
    responses={503: {"model": ReadinessResponse}},
)
async def readiness(request: Request) -> ReadinessResponse | JSONResponse:
    """Return whether this process can accept query traffic."""
    config = get_application(request).config
    detail = await _application_health(request).readiness_detail()
    if detail is not None:
        return _not_ready(service_role=config.deployment.service_role, detail=detail)

    return ReadinessResponse(status="ready", service_role=config.deployment.service_role)
