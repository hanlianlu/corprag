# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP server for agent integration (stdio + streamable-http).

Entry point: dlightrag-mcp
Primarily used by DeerFlow and other MCP-compatible agents for
retrieve() + lightweight ingest().
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from mcp import MCPError
from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.mcpserver import Context
from mcp.types import CallToolResult, InputRequiredResult, TextContent
from pydantic import Field
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

import dlightrag
from dlightrag import create_application
from dlightrag.adapters.mcp.auth import DlightRAGTokenVerifier
from dlightrag.adapters.mcp.contracts import (
    ConversationMessage,
    CreateWorkspaceInput,
)
from dlightrag.application import Application, ApplicationClosedError
from dlightrag.application.access import (
    AccessAction,
    AccessDeniedError,
    AccessGate,
    NoQueryableWorkspacesError,
    RequestScope,
    WorkspaceRecord,
    access_control_from_settings,
    current_request_scope,
    owner_id_from_principal,
    request_scope_context,
)
from dlightrag.application.answer_runs import AnswerRuntimeUnavailableError
from dlightrag.application.config import DlightragConfig, get_config
from dlightrag.application.corpus_admin import (
    normalize_workspace,
    normalize_workspace_ids,
)
from dlightrag.application.retrieval import CorpusUnavailableError
from dlightrag.application.runs import RunRuntimeUnavailableError, RunView
from dlightrag.application.settings import access_settings
from dlightrag.engine.answer.client_contracts import (
    MAX_HISTORY_MESSAGES,
    AnswerAttachmentLink,
    QueryImage,
)
from dlightrag.engine.answer.errors import (
    AnswerInputError,
    InvalidToolConfigurationError,
)

logger = logging.getLogger(__name__)

QueryImagesParam = Annotated[
    list[QueryImage],
    Field(
        max_length=3,
        description="User-attached image URLs or data URI blocks (max 3)",
    ),
]
AttachmentsParam = Annotated[
    list[AnswerAttachmentLink],
    Field(
        description=(
            "Answer attachments as HTTPS link descriptors ({url, filename?}). MCP accepts "
            "links only; local paths, raw bytes, and base64 are rejected."
        ),
    ),
]
HistoryParam = Annotated[
    list[ConversationMessage] | None,
    Field(
        max_length=MAX_HISTORY_MESSAGES,
        description=(
            "Prior conversation turns as role/content messages. Independent requests "
            "re-send the desired turns; accepted runs pin the bounded history for recovery."
        ),
    ),
]
FederatedRerankParam = Annotated[
    bool,
    Field(
        description=(
            "Apply one cross-workspace rerank pass over the merged federation pool "
            "before selecting the top chunks. Defaults to off; the round-robin "
            "fairness cap is used instead."
        ),
    ),
]
IdempotencyKeyParam = Annotated[
    str | None,
    Field(
        default=None,
        max_length=255,
        description=(
            "Optional replay key scoped to the calling identity. Repeating it with the "
            "same request returns the same run instead of starting a second one; "
            "repeating it with a different request is rejected."
        ),
    ),
]


def _owner_id() -> str:
    """Project the current MCP principal into the shared run owner namespace."""
    scope = current_request_scope()
    return owner_id_from_principal(
        auth_mode=scope.auth_mode,
        user_id=scope.user_id,
        issuer=str(scope.claims.get("iss") or "") or None,
    )


def _run_descriptor(record: RunView) -> dict[str, Any]:
    """Project one run's lifecycle and continuation lineage."""
    accepted = record.request_input()
    return {
        "run_id": record.run_id,
        "run_kind": record.run_kind,
        "lane": record.lane,
        "status": record.status,
        "cancel_requested": record.cancel_requested,
        "parent_run_id": accepted.get("parent_run_id"),
        "continuation_kind": accepted.get("continuation_kind"),
        "created_at": record.created_at.isoformat(),
    }


class DlightRAGMCPServer(MCPServer):
    """MCPServer with DlightRAG's strict input and text-error contract."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Context | None = None,
    ) -> CallToolResult | InputRequiredResult:
        try:
            await self._reject_unknown_arguments(name, arguments or {})
            return await super().call_tool(name, arguments or {}, context=context)
        except MCPError:
            raise
        except Exception as exc:
            # MCPServer wraps tool-body errors as ToolError(...) from the original, so
            # inspect __cause__ as well. Server misconfiguration is a server failure;
            # user-facing validation/authorization messages are surfaced as rejections;
            # unexpected internals hide behind a generic message.
            surfaced = (
                ValueError,
                PermissionError,
                InvalidToolConfigurationError,
                ApplicationClosedError,
                CorpusUnavailableError,
                AnswerRuntimeUnavailableError,
                RunRuntimeUnavailableError,
            )
            inner = exc if isinstance(exc, surfaced) else exc.__cause__
            if isinstance(inner, InvalidToolConfigurationError):
                logger.exception("MCP tool '%s' failed: %s", name, inner)
                text = f"Error [{inner.error_kind}]: {inner.public_message}"
            elif isinstance(
                inner,
                ValueError
                | PermissionError
                | ApplicationClosedError
                | CorpusUnavailableError
                | AnswerRuntimeUnavailableError
                | RunRuntimeUnavailableError,
            ):
                logger.warning("MCP tool '%s' rejected: %s", name, inner)
                text = (
                    f"Error [{inner.error_kind}]: {inner.public_message}"
                    if isinstance(inner, AnswerInputError)
                    else f"Error: {inner}"
                )
            else:
                logger.exception("MCP tool '%s' failed", name)
                text = "Error: internal tool failure"
            return CallToolResult(
                content=[TextContent(type="text", text=text)],
                is_error=True,
            )

    async def _reject_unknown_arguments(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        tool = next((tool for tool in await self.list_tools() if tool.name == name), None)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        allowed = set(tool.input_schema.get("properties", {}))
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ValueError(f"Unexpected argument(s) for {name}: {', '.join(unknown)}")


class DlightRAGRequestScopeMiddleware:
    """Project an MCP OAuth principal into DlightRAG's request scope."""

    async def __call__(
        self,
        ctx: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> HandlerResult:
        access_token = get_access_token()
        scope = current_request_scope()
        if access_token is not None:
            scope = RequestScope(
                user_id=access_token.subject or access_token.client_id,
                auth_mode=_get_config().access.auth_mode,
                claims=dict(access_token.claims or {}),
            )
        with request_scope_context(scope):
            return await call_next(ctx)


def _get_config() -> DlightragConfig:
    return get_config()


_application: Application | None = None


async def _ensure_application() -> Application:
    """Return the lifespan-bound Application; never compose a fallback service."""
    if _application is None:
        raise ApplicationClosedError("Application is not bound to this MCP transport")
    return _application


async def _close_application() -> None:
    global _application
    application, _application = _application, None
    if application is not None:
        await application.aclose()


@asynccontextmanager
async def _mcp_lifespan(_: MCPServer[Any]) -> AsyncIterator[None]:
    global _application
    if _application is not None:
        raise RuntimeError("Application is already bound to this MCP transport")
    _application = await create_application()
    try:
        yield
    finally:
        await _close_application()


def _http_auth(
    config: DlightragConfig,
) -> tuple[AuthSettings | None, DlightRAGTokenVerifier | None]:
    if config.interfaces.mcp.transport != "streamable-http" or config.access.auth_mode == "none":
        return None, None
    resource = config.interfaces.mcp.resource_server_url
    auth = AuthSettings.model_validate(
        {
            "issuer_url": config.access.jwt_issuer or "http://localhost",
            "resource_server_url": resource,
        }
    )
    return auth, DlightRAGTokenVerifier(config, resource=resource)


_auth, _token_verifier = _http_auth(_get_config())
mcp_app = DlightRAGMCPServer(
    "dlightrag",
    version=dlightrag.__version__,
    log_level="INFO",
    warn_on_duplicate_tools=True,
    lifespan=_mcp_lifespan,
    middleware=[DlightRAGRequestScopeMiddleware()],
    auth=_auth,
    token_verifier=_token_verifier,
)


def _normalize_workspace_argument(args: CreateWorkspaceInput) -> tuple[str, str]:
    from dlightrag.application.corpus_admin import validate_workspace_name

    label = validate_workspace_name(args.workspace)
    display_name = validate_workspace_name(args.display_name or label)
    return normalize_workspace(label), display_name


async def _enforce_access(
    action: str,
    workspace: str | None = None,
    *,
    application: Application,
) -> None:
    try:
        await _access_gate(application).check(action, workspace=workspace)
    except AccessDeniedError as exc:
        raise ValueError(str(exc)) from None


def _access_gate(application: Application) -> AccessGate:
    return AccessGate(
        access_control_from_settings(access_settings(application.config)),
        current_request_scope(),
    )


async def _filter_workspace_records(
    records: list[WorkspaceRecord],
    *,
    application: Application,
) -> list[WorkspaceRecord]:
    return await _access_gate(application).filter_workspace_records(
        AccessAction.WORKSPACE_QUERY, records
    )


async def _authorized_workspace_names(
    action: str,
    workspaces: list[str],
    *,
    application: Application,
) -> set[str]:
    return await _access_gate(application).authorized_workspace_ids(action, workspaces)


async def _resolve_authorized_query_workspaces(
    application: Application,
    *,
    workspaces: list[str] | None,
    all_workspaces: bool,
) -> list[str]:
    """Resolve MCP query targets after applying the current request ACL."""
    try:
        return await _access_gate(application).resolve_query_workspaces(
            application.corpora,
            default_workspace=normalize_workspace(application.config.deployment.workspace),
            workspaces=normalize_workspace_ids(workspaces) if workspaces is not None else None,
            all_workspaces=all_workspaces,
        )
    except NoQueryableWorkspacesError:
        raise PermissionError("No workspaces are available for query") from None
    except AccessDeniedError as exc:
        raise ValueError(str(exc)) from None


def _register_tools() -> None:
    from dlightrag.adapters.mcp.tools import register as register_mcp_tools

    register_mcp_tools()


_register_tools()


# ═══════════════════════════════════════════════════════════════════
# Server startup
# ═══════════════════════════════════════════════════════════════════


def create_mcp_http_app() -> ASGIApp:
    """Create the production Streamable HTTP ASGI app."""
    from mcp.server.transport_security import TransportSecuritySettings

    config = _get_config()
    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(config.interfaces.mcp.allowed_hosts),
        allowed_origins=list(config.interfaces.mcp.allowed_origins),
    )
    http_app: ASGIApp = mcp_app.streamable_http_app(
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        transport_security=transport_security,
        host=config.interfaces.mcp.host,
    )
    return CORSMiddleware(
        http_app,
        allow_origins=config.access.cors_allow_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Mcp-Method",
            "Mcp-Name",
            "Mcp-Protocol-Version",
        ],
    )


async def run_stdio() -> None:
    """Run MCP server over stdio transport."""
    await mcp_app.run_stdio_async()


async def run_streamable_http() -> None:
    """Run the MCP 2.0 Streamable HTTP server."""
    import uvicorn

    config = _get_config()
    uvicorn_config = uvicorn.Config(
        create_mcp_http_app(),
        host=config.interfaces.mcp.host,
        port=config.interfaces.mcp.port,
        log_level="info",
    )
    await uvicorn.Server(uvicorn_config).serve()


def run() -> None:
    """Run the configured MCP transport."""
    config = _get_config()
    logging.basicConfig(
        level=getattr(logging, config.observability.log_level.upper(), logging.INFO)
    )

    if config.interfaces.mcp.transport == "streamable-http":
        logger.info(
            "Starting MCP server (streamable-http) on %s:%d",
            config.interfaces.mcp.host,
            config.interfaces.mcp.port,
        )
        asyncio.run(run_streamable_http())
    else:
        logger.info("Starting MCP server (stdio)")
        asyncio.run(run_stdio())


__all__ = ["create_mcp_http_app", "mcp_app", "run"]
