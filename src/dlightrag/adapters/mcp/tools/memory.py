# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP tools for Profile Memory."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from dlightrag_memory import MemoryProvenance
from dlightrag_memory.errors import MemoryUnavailableError, MemoryWriteRejectedError
from mcp.types import ToolAnnotations
from pydantic import Field

from dlightrag.adapters.mcp import server as mcp_server
from dlightrag.adapters.mcp.server import (
    mcp_app,
)
from dlightrag.application.access import current_request_scope
from dlightrag.application.memory import MemoryDisabledError
from dlightrag.application.memory.projections import memory_receipt_payload


@mcp_app.tool(
    name="list_memories",
    description=(
        "List this caller's active Profile Memories, newest first, bounded to the "
        "first 50. `has_more` marks additional older memories, available only "
        "through the REST memory endpoint with cursor paging. Not evidence."
    ),
    annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
)
async def list_memories_tool() -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    try:
        page = await application.memory.list_active_page(
            owner_id=mcp_server._owner_id(), auth_mode=current_request_scope().auth_mode
        )
    except (MemoryUnavailableError, MemoryDisabledError) as exc:
        raise ValueError(exc.public_message) from exc
    return {
        "memories": [
            {"memory_id": row.memory_id, "kind": row.kind, "body": row.body} for row in page.records
        ],
        "has_more": page.next_cursor is not None,
    }


@mcp_app.tool(
    name="remember_memory",
    description="Store one durable owner preference or fact. Not evidence.",
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=True),
)
async def remember_memory_tool(
    kind: Annotated[Literal["preference", "fact"], Field(description="Memory kind")],
    body: Annotated[str, Field(min_length=1, max_length=500, description="Memory body")],
    idempotency_key: Annotated[str, Field(min_length=1, max_length=255)],
    supersedes_id: Annotated[str | None, Field(default=None)] = None,
) -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    try:
        receipt = await application.memory.remember(
            owner_id=mcp_server._owner_id(),
            auth_mode=current_request_scope().auth_mode,
            kind=kind,
            body=body,
            supersedes_id=supersedes_id,
            provenance=MemoryProvenance(origin_kind="mcp", origin_id=idempotency_key),
            idempotency_key=f"mcp:{idempotency_key}",
        )
    except (MemoryUnavailableError, MemoryDisabledError, MemoryWriteRejectedError) as exc:
        raise ValueError(exc.public_message) from exc
    return memory_receipt_payload(receipt)


@mcp_app.tool(
    name="forget_memory",
    description="Idempotently forget one active Profile Memory by id.",
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=True),
)
async def forget_memory_tool(
    memory_id: Annotated[str, Field(description="Memory id")],
    idempotency_key: Annotated[str, Field(min_length=1, max_length=255)],
) -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    try:
        receipt = await application.memory.forget(
            owner_id=mcp_server._owner_id(),
            auth_mode=current_request_scope().auth_mode,
            memory_id=memory_id,
            provenance=MemoryProvenance(origin_kind="mcp", origin_id=idempotency_key),
            idempotency_key=f"mcp:{idempotency_key}",
        )
    except (MemoryUnavailableError, MemoryDisabledError, MemoryWriteRejectedError) as exc:
        raise ValueError(exc.public_message) from exc
    return memory_receipt_payload(receipt)


@mcp_app.tool(
    name="undo_memory_change",
    description="Compensate one still-current Profile Memory change.",
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=True),
)
async def undo_memory_change_tool(
    change_id: Annotated[str, Field(description="Change id")],
    idempotency_key: Annotated[str, Field(min_length=1, max_length=255)],
) -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    try:
        receipt = await application.memory.undo(
            owner_id=mcp_server._owner_id(),
            auth_mode=current_request_scope().auth_mode,
            change_id=change_id,
            provenance=MemoryProvenance(origin_kind="undo", origin_id=idempotency_key),
            idempotency_key=f"mcp:{idempotency_key}",
        )
    except (MemoryUnavailableError, MemoryDisabledError, MemoryWriteRejectedError) as exc:
        raise ValueError(exc.public_message) from exc
    return memory_receipt_payload(receipt)


@mcp_app.tool(
    name="get_memory_settings",
    description="Read this caller's Profile Memory activation state.",
    annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
)
async def get_memory_settings_tool() -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    settings = await application.memory.settings(
        owner_id=mcp_server._owner_id(), auth_mode=current_request_scope().auth_mode
    )
    return {"enabled": settings.enabled, "active_count": settings.active_count}


@mcp_app.tool(
    name="set_memory_enabled",
    description="Activate or deactivate this caller's complete Profile Memory capability.",
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=True),
)
async def set_memory_enabled_tool(enabled: bool) -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    settings = await application.memory.set_enabled(
        owner_id=mcp_server._owner_id(),
        auth_mode=current_request_scope().auth_mode,
        enabled=enabled,
    )
    return {"enabled": settings.enabled, "active_count": settings.active_count}


@mcp_app.tool(
    name="clear_memory",
    description="Physically clear this caller's active Profile Memory schema state.",
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=True),
)
async def clear_memory_tool() -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    try:
        removed = await application.memory.clear(
            owner_id=mcp_server._owner_id(), auth_mode=current_request_scope().auth_mode
        )
    except (MemoryUnavailableError, MemoryDisabledError) as exc:
        raise ValueError(exc.public_message) from exc
    return {"removed": removed}
