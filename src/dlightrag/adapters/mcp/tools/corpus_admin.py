# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP tools for workspaces, capabilities, ingest, and files."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Annotated, Any, Literal

from mcp.types import ToolAnnotations
from pydantic import Field

from dlightrag.adapters.mcp import server as mcp_server
from dlightrag.adapters.mcp.contracts import (
    CreateWorkspaceInput,
    DeleteFilesInput,
    IngestInput,
    ListFilesInput,
    RetryFilesInput,
)
from dlightrag.adapters.mcp.server import (
    mcp_app,
)
from dlightrag.application.access import AccessAction
from dlightrag.application.answer_runs.capability import answer_image_capability_summary
from dlightrag.application.corpus_admin import (
    FILE_PANEL_PAGE_DEFAULT_LIMIT,
    FILE_PANEL_PAGE_MAX_LIMIT,
    FilePanelCursorError,
    FilePanelPageRequest,
    SourceType,
    WorkspaceCatalogPageRequest,
    ingest_spec_from_payload,
    managed_local_ingest_documents,
    managed_local_ingest_path,
    normalize_workspace,
    validate_workspace_name,
)
from dlightrag.application.runs import RunAdmissionLimitExceededError, RunCreation


async def _accepted_corpus_mutation(operation: Awaitable[RunCreation]) -> dict[str, Any]:
    try:
        creation = await operation
    except RunAdmissionLimitExceededError:
        raise ValueError("Deployment-wide nonterminal admission limit reached") from None
    return mcp_server._run_descriptor(creation.run)


@mcp_app.tool(
    name="list_workspaces",
    description=(
        "List workspaces visible to the current user, ascending by workspace id, "
        "bounded to the first 50. Records contain workspace, display_name, "
        "embedding_model, created_at, and updated_at. `has_more` marks "
        "additional workspaces, available only through the REST workspace "
        "endpoint with cursor paging. Use display_name as the user-facing "
        "workspace label."
    ),
    annotations=ToolAnnotations(read_only_hint=True),
)
async def list_workspaces_tool() -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    page = await application.corpora.list_workspace_records_page(
        page=WorkspaceCatalogPageRequest(),
    )
    records = await mcp_server._filter_workspace_records(list(page.items), application=application)
    return {
        "workspaces": [row["workspace"] for row in records],
        "records": records,
        "has_more": page.next_cursor is not None,
    }


@mcp_app.tool(
    name="get_capabilities",
    description=(
        "Report deployment capabilities agents should honor. Returns "
        "answer_image_capability with status (supported/unsupported/unknown), "
        "effective_max_images (max images the answer model accepts; 0 means send none), "
        "configured_ceiling, and model — query images reach the answer model only when "
        "status is 'supported'."
    ),
    annotations=ToolAnnotations(read_only_hint=True),
)
async def get_capabilities_tool() -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    capabilities = await application.answers.capabilities()
    return {
        "answer_image_capability": answer_image_capability_summary(capabilities.answer),
    }


@mcp_app.tool(
    name="get_workspace_storage_status",
    description=(
        "Operator-facing storage facts for one workspace: storage_tier "
        "(shared or hot), promotion_state, monotonic ingested_docs_total and "
        "ingested_chunks_total, promotion_last_error, promotion_next_retry_at, "
        "and write_fenced with retry_after_seconds. Requires the admin-only "
        "workspace.storage_status action; ordinary users are never granted it."
    ),
    annotations=ToolAnnotations(read_only_hint=True),
)
async def get_workspace_storage_status_tool(
    workspace: Annotated[
        str | None,
        Field(default=None, description="Workspace to inspect. Omit for default."),
    ] = None,
) -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    workspace_name = normalize_workspace(workspace or application.config.deployment.workspace)
    await mcp_server._enforce_access(
        AccessAction.WORKSPACE_STORAGE_STATUS,
        workspace_name,
        application=application,
    )
    status = await application.corpora.get_workspace_storage_status(workspace_name)
    if status is None:
        raise ValueError(f"Workspace not found: {workspace_name}")
    return status


@mcp_app.tool(
    name="create_workspace",
    description=(
        "Create and register an empty DlightRAG workspace. Optional display_name is "
        "the user-facing label; response returns normalized workspace id, display_name, "
        "and created."
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
    ),
)
async def create_workspace_tool(
    workspace: Annotated[str, Field(description="Workspace name to create.")],
    display_name: Annotated[
        str | None,
        Field(default=None, description="Optional user-facing display name."),
    ] = None,
) -> dict[str, Any]:
    args = CreateWorkspaceInput.model_validate(locals())
    application = await mcp_server._ensure_application()
    normalized_workspace, normalized_display_name = mcp_server._normalize_workspace_argument(args)
    await mcp_server._enforce_access(
        AccessAction.WORKSPACE_CREATE,
        normalized_workspace,
        application=application,
    )
    existing = await application.corpora.list_workspaces()
    if normalized_workspace in existing:
        raise ValueError(f"Workspace '{normalized_display_name}' already exists")
    await application.corpora.create_workspace(
        normalized_workspace,
        display_name=normalized_display_name,
    )
    return {
        "workspace": normalized_workspace,
        "display_name": normalized_display_name,
        "created": True,
    }


@mcp_app.tool(
    name="reset_corpus",
    description=(
        "Accept a full Corpus Reset while preserving Workspace identity. "
        "Returns the common durable Run descriptor."
    ),
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True),
)
async def reset_corpus_tool(
    workspace: Annotated[str, Field(description="Workspace whose corpus is reset.")],
    supersedes_run_id: Annotated[
        str | None,
        Field(description="Waiting-for-repair mutation Run explicitly superseded by this Reset."),
    ] = None,
    idempotency_key: Annotated[
        str | None,
        Field(default=None, max_length=255, description="Stable caller replay key."),
    ] = None,
) -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    normalized_workspace = normalize_workspace(validate_workspace_name(workspace))
    await mcp_server._enforce_access(
        AccessAction.WORKSPACE_RESET,
        normalized_workspace,
        application=application,
    )
    return await _accepted_corpus_mutation(
        application.corpus_mutations.create_reset(
            workspace=normalized_workspace,
            submitted_by=mcp_server._owner_id(),
            supersedes_run_id=supersedes_run_id,
            idempotency_key=idempotency_key,
        )
    )


@mcp_app.tool(
    name="ingest",
    description=(
        "Accept durable local, URL, Azure Blob, or S3 ingestion into "
        "a workspace. URL fetch endpoints, stable source identity, and durable download "
        "locators are separate; signed fetches require retention or a queryless locator. "
        "Response is the common Run descriptor; use get_run/cancel_run for lifecycle."
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
    ),
)
async def ingest_tool(
    source_type: Annotated[SourceType, Field(description="Type of data source")],
    path: Annotated[
        str | None,
        Field(default=None, description="File or directory path for local source."),
    ] = None,
    container_name: Annotated[
        str | None,
        Field(default=None, description="Azure Blob container name."),
    ] = None,
    blob_path: Annotated[
        str | None,
        Field(default=None, description="Specific blob path for azure_blob."),
    ] = None,
    bucket: Annotated[
        str | None,
        Field(default=None, description="S3 bucket name."),
    ] = None,
    s3_region: Annotated[
        str | None,
        Field(default=None, description="S3 region name."),
    ] = None,
    s3_key: Annotated[
        str | None,
        Field(default=None, description="S3 object key, single object or prefix."),
    ] = None,
    prefix: Annotated[
        str | None,
        Field(default=None, description="Path/blob/key prefix filter."),
    ] = None,
    url: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Public or signed HTTPS fetch URL. A signed/query-bearing URL requires "
                "retention or a separate queryless download_uri."
            ),
        ),
    ] = None,
    urls: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Public or signed HTTPS fetch URLs. Signed/query-bearing entries require "
                "retention or matching queryless download_uris."
            ),
        ),
    ] = None,
    filename: Annotated[
        str | None,
        Field(default=None, description="Parser filename for a single URL."),
    ] = None,
    source_uri: Annotated[
        str | None,
        Field(
            default=None,
            description="Stable provenance identity for one URL; not a download address.",
        ),
    ] = None,
    source_uris: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Stable provenance identities for a URL batch; not download addresses.",
        ),
    ] = None,
    download_uri: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Durable S3, Azure, or credential-free queryless public HTTPS locator "
                "for one fetched URL."
            ),
        ),
    ] = None,
    download_uris: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=("Durable S3, Azure, or queryless public HTTPS locators for a URL batch."),
        ),
    ] = None,
    documents: Annotated[
        list[dict[str, Any]] | None,
        Field(
            default=None,
            description=(
                "Explicit document manifest. Local documents use path, S3/Azure use key, "
                "URL documents use url. Document metadata overlays request metadata."
            ),
        ),
    ] = None,
    replace: Annotated[
        bool | None,
        Field(default=None, description="Replace existing documents."),
    ] = None,
    workspace: Annotated[
        str | None,
        Field(default=None, description="Target workspace. Omit for default."),
    ] = None,
    title: Annotated[
        str | None,
        Field(default=None, description="Optional document title metadata."),
    ] = None,
    author: Annotated[
        str | None,
        Field(default=None, description="Optional document author metadata."),
    ] = None,
    metadata: Annotated[
        dict[str, Any] | None,
        Field(default=None, description="User metadata to attach to ingested documents."),
    ] = None,
    retain_source_file: Annotated[
        bool | None,
        Field(
            default=None,
            description=(
                "Keep fetched bytes as the download source. Signed URL fetches require this "
                "unless a separate queryless durable locator is supplied."
            ),
        ),
    ] = None,
    idempotency_key: Annotated[
        str | None,
        Field(default=None, max_length=255, description="Stable caller replay key."),
    ] = None,
) -> dict[str, Any]:
    args = IngestInput.model_validate(locals())
    application = await mcp_server._ensure_application()
    workspace_name = args.workspace or application.config.deployment.workspace
    workspace_name = normalize_workspace(workspace_name)
    await mcp_server._enforce_access(
        AccessAction.WORKSPACE_INGEST,
        workspace_name,
        application=application,
    )
    ingest_spec = ingest_spec_from_payload(args)
    if args.source_type == "local":
        path = managed_local_ingest_path(
            source_type=args.source_type,
            path=ingest_spec.path,
            input_dir=application.config.input_dir_path,
            workspace=workspace_name,
        )
        managed_documents = managed_local_ingest_documents(
            source_type=args.source_type,
            documents=ingest_spec.documents,
            input_dir=application.config.input_dir_path,
            workspace=workspace_name,
        )
        ingest_spec = ingest_spec.model_copy(update={"path": path, "documents": managed_documents})
    return await _accepted_corpus_mutation(
        application.corpus_mutations.create_ingest(
            workspace=workspace_name,
            spec=ingest_spec,
            submitted_by=mcp_server._owner_id(),
            idempotency_key=args.idempotency_key,
        )
    )


@mcp_app.tool(
    name="list_files",
    description=(
        "List one bounded page of documents in a workspace. Response returns files, count, "
        "workspace, next_cursor, and fetched_rows."
    ),
    annotations=ToolAnnotations(read_only_hint=True),
)
async def list_files_tool(
    workspace: Annotated[
        str | None,
        Field(default=None, description="Workspace to list files from. Omit for default."),
    ] = None,
    limit: Annotated[
        int,
        Field(
            ge=1,
            le=FILE_PANEL_PAGE_MAX_LIMIT,
            description="Maximum files in this page.",
        ),
    ] = FILE_PANEL_PAGE_DEFAULT_LIMIT,
    cursor: Annotated[
        str | None,
        Field(default=None, min_length=1, max_length=1024, description="Opaque next cursor."),
    ] = None,
) -> dict[str, Any]:
    args = ListFilesInput.model_validate(locals())
    application = await mcp_server._ensure_application()
    workspace_name = normalize_workspace(args.workspace or application.config.deployment.workspace)
    await mcp_server._enforce_access(
        AccessAction.WORKSPACE_LIST_FILES,
        workspace_name,
        application=application,
    )
    try:
        decoded = (
            application.corpora.file_panel_cursor_codec.decode(args.cursor)
            if args.cursor is not None
            else None
        )
        if decoded is not None and decoded.workspace != workspace_name:
            raise FilePanelCursorError("file-panel cursor belongs to another workspace")
        if decoded is not None and decoded.view != "processed":
            raise FilePanelCursorError("file-panel cursor belongs to another view")
        page = FilePanelPageRequest(limit=args.limit, cursor=decoded)
    except (FilePanelCursorError, ValueError) as exc:
        raise ValueError(str(exc)) from None
    snapshot = await application.corpora.file_panel_snapshot(workspace_name, page=page)
    files = snapshot["files"]
    next_cursor = snapshot["next_cursor"]
    return {
        "files": files,
        "count": len(files),
        "workspace": workspace_name,
        "next_cursor": (
            application.corpora.file_panel_cursor_codec.encode(next_cursor)
            if next_cursor is not None
            else None
        ),
        "fetched_rows": snapshot["fetched_rows"],
    }


@mcp_app.tool(
    name="retry_files",
    description=(
        "Accept durable retry of an explicit failed-document cohort, or snapshot all "
        "currently retryable documents. Response is the common Run descriptor."
    ),
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True),
)
async def retry_files_tool(
    document_ids: Annotated[
        list[str] | None,
        Field(default=None, max_length=100, description="Exact document ids to retry."),
    ] = None,
    selector: Annotated[
        Literal["all_retryable"] | None,
        Field(default=None, description="Use all_retryable to snapshot the current cohort."),
    ] = None,
    workspace: Annotated[
        str | None,
        Field(default=None, description="Workspace to retry in. Omit for default."),
    ] = None,
    idempotency_key: Annotated[
        str | None,
        Field(default=None, max_length=255, description="Stable caller replay key."),
    ] = None,
) -> dict[str, Any]:
    args = RetryFilesInput.model_validate(locals())
    application = await mcp_server._ensure_application()
    workspace_name = normalize_workspace(args.workspace or application.config.deployment.workspace)
    await mcp_server._enforce_access(
        AccessAction.WORKSPACE_INGEST,
        workspace_name,
        application=application,
    )
    return await _accepted_corpus_mutation(
        application.corpus_mutations.create_retry(
            workspace=workspace_name,
            submitted_by=mcp_server._owner_id(),
            document_ids=args.document_ids or (),
            selector=args.selector,
            idempotency_key=args.idempotency_key,
        )
    )


@mcp_app.tool(
    name="delete_files",
    description=(
        "Accept durable deletion of exact document ids, filenames, or file paths. "
        "Response is the common Run descriptor."
    ),
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True),
)
async def delete_files_tool(
    filenames: Annotated[
        list[str] | None,
        Field(default=None, max_length=100, description="Exact filenames to delete."),
    ] = None,
    file_paths: Annotated[
        list[str] | None,
        Field(default=None, max_length=100, description="Exact file paths to delete."),
    ] = None,
    document_ids: Annotated[
        list[str] | None,
        Field(default=None, max_length=100, description="Exact document ids to delete."),
    ] = None,
    workspace: Annotated[
        str | None,
        Field(default=None, description="Workspace to delete from. Omit for default."),
    ] = None,
    idempotency_key: Annotated[
        str | None,
        Field(default=None, max_length=255, description="Stable caller replay key."),
    ] = None,
) -> dict[str, Any]:
    args = DeleteFilesInput.model_validate(locals())
    application = await mcp_server._ensure_application()
    workspace_name = normalize_workspace(args.workspace or application.config.deployment.workspace)
    await mcp_server._enforce_access(
        AccessAction.WORKSPACE_DELETE_FILES,
        workspace_name,
        application=application,
    )
    return await _accepted_corpus_mutation(
        application.corpus_mutations.create_delete(
            workspace=workspace_name,
            submitted_by=mcp_server._owner_id(),
            filenames=args.filenames or (),
            file_paths=args.file_paths or (),
            document_ids=args.document_ids or (),
            idempotency_key=args.idempotency_key,
        )
    )
