# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval Run acceptance route."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.rest.models import RetrieveRequest, RunDescriptor
from dlightrag.adapters.http.rest.payloads import metadata_filter_from_payload
from dlightrag.application.access import UserContext, owner_id_from_user
from dlightrag.application.retrieval import RetrieveRequest as ServiceRequest
from dlightrag.application.runs import IdempotencyKeyConflict, RunCapacityExceededError

from .deps import (
    get_application,
    idempotency_key,
    resolve_authorized_query_workspaces,
)

router = APIRouter()


@router.post("/retrieve", response_model=RunDescriptor, status_code=202)
async def retrieve(
    body: RetrieveRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Accept durable Retrieval and return its common Run descriptor."""
    application = get_application(request)
    resolved_workspaces = await resolve_authorized_query_workspaces(
        request,
        user,
        workspaces=body.workspaces,
        all_workspaces=body.all_workspaces,
    )
    try:
        creation = await application.retrieval.create(
            request=ServiceRequest(
                query=body.query,
                workspaces=tuple(resolved_workspaces),
                top_k=body.top_k,
                chunk_top_k=body.chunk_top_k,
                federated_rerank=body.federated_rerank,
                bm25_query=body.bm25_query,
                filters=metadata_filter_from_payload(body.filters),
                query_images=tuple(
                    image.model_dump(exclude_none=True) for image in body.query_images or ()
                ),
            ),
            owner_id=owner_id_from_user(user),
            idempotency_key=idempotency_key(request),
        )
    except IdempotencyKeyConflict:
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key was reused with a different retrieval request",
        ) from None
    except RunCapacityExceededError:
        raise HTTPException(status_code=503, detail="Run admission capacity is full") from None
    from .runs import run_descriptor

    return run_descriptor(creation.run)
