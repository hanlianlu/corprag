# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP creation tool for durable Retrieval runs."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

from dlightrag.adapters.mcp import server as mcp_server
from dlightrag.adapters.mcp.contracts import (
    RetrieveInput,
)
from dlightrag.adapters.mcp.server import (
    FederatedRerankParam,
    IdempotencyKeyParam,
    QueryImagesParam,
    mcp_app,
)
from dlightrag.application.retrieval import MetadataFilter
from dlightrag.application.retrieval import RetrieveRequest as ServiceRequest
from dlightrag.application.runs import IdempotencyKeyConflict, RunAdmissionLimitExceededError


@mcp_app.tool(
    name="retrieve",
    description=(
        "Start durable Retrieval over the default or selected workspaces. Returns "
        "immediately with a run_id, not contexts. Poll get_run until the common status "
        "is terminal, or call cancel_run. The run survives this call and server restarts."
    ),
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=False),
)
async def retrieve_tool(
    query: Annotated[str, Field(description="The search query")],
    top_k: Annotated[
        int | None,
        Field(default=None, description="Number of top results to return"),
    ] = None,
    chunk_top_k: Annotated[
        int | None,
        Field(default=None, description="Vector chunk candidate count override."),
    ] = None,
    federated_rerank: FederatedRerankParam = False,
    bm25_query: Annotated[
        str | None,
        Field(
            default=None,
            max_length=1024,
            description=(
                "Optional lexical/BM25 query override. When omitted, BM25 uses the main query."
            ),
        ),
    ] = None,
    workspaces: Annotated[
        list[str] | None,
        Field(default=None, description="Workspace names to search. Omit for default."),
    ] = None,
    all_workspaces: Annotated[
        bool,
        Field(
            default=False,
            description="Search all workspaces visible to the current caller.",
        ),
    ] = False,
    filters: Annotated[
        dict[str, Any] | None,
        Field(default=None, description="Metadata filters for structured queries."),
    ] = None,
    query_images: QueryImagesParam = Field(default_factory=list),
    idempotency_key: IdempotencyKeyParam = None,
) -> dict[str, Any]:
    args = RetrieveInput.model_validate(locals())
    application = await mcp_server._ensure_application()
    resolved_workspaces = await mcp_server._resolve_authorized_query_workspaces(
        application,
        workspaces=args.workspaces,
        all_workspaces=args.all_workspaces,
    )
    try:
        creation = await application.retrieval.create(
            request=ServiceRequest(
                query=args.query,
                workspaces=tuple(resolved_workspaces),
                top_k=args.top_k,
                chunk_top_k=args.chunk_top_k,
                federated_rerank=args.federated_rerank,
                bm25_query=args.bm25_query,
                filters=MetadataFilter.model_validate(args.filters) if args.filters else None,
                query_images=tuple(
                    image.model_dump(exclude_none=True) for image in args.query_images or ()
                ),
            ),
            owner_id=mcp_server._owner_id(),
            idempotency_key=args.idempotency_key,
        )
    except IdempotencyKeyConflict:
        raise ValueError(
            "idempotency_key was already used for a different retrieval request"
        ) from None
    except RunAdmissionLimitExceededError:
        raise ValueError("Deployment-wide nonterminal admission limit reached") from None
    return mcp_server._run_descriptor(creation.run)
