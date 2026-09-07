# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pydantic input contracts for DlightRAG MCP tools."""

from typing import Any, Literal, Self

from pydantic import Field, model_validator

from dlightrag.application.access import validate_query_workspace_selection
from dlightrag.application.answer_runs.client_contracts import (
    MAX_QUERY_IMAGES,
    AnswerAttachmentLink,
    AnswerRequestContract,
    ClientContractModel,
    ConversationMessage,
    QueryImage,
    RetrieveRequestContract,
)
from dlightrag.application.corpus_admin import FILE_PANEL_PAGE_DEFAULT_LIMIT, IngestSpec


class MCPInput(ClientContractModel):
    pass


class QueryWorkspaceSelection(ClientContractModel):
    workspaces: list[str] | None = None
    all_workspaces: bool = False

    @model_validator(mode="after")
    def _validate_workspace_selection(self) -> Self:
        validate_query_workspace_selection(
            all_workspaces=self.all_workspaces,
            workspaces=self.workspaces,
        )
        return self


class RetrieveInput(QueryWorkspaceSelection, RetrieveRequestContract):
    filters: dict[str, Any] | None = None
    query_images: list[QueryImage] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=list,
        max_length=MAX_QUERY_IMAGES,
    )
    idempotency_key: str | None = Field(default=None, max_length=255)


class AnswerInput(QueryWorkspaceSelection, AnswerRequestContract):
    filters: dict[str, Any] | None = None
    attachments: list[AnswerAttachmentLink] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=list,
    )
    idempotency_key: str | None = Field(default=None, max_length=255)


class AnswerRunInput(MCPInput):
    """One owned answer run addressed by the id the answer tool returned."""

    run_id: str = Field(min_length=1, max_length=64)


class IngestInput(IngestSpec):
    workspace: str | None = None
    idempotency_key: str | None = Field(default=None, max_length=255)


class CreateWorkspaceInput(MCPInput):
    workspace: str
    display_name: str | None = None


class ListFilesInput(MCPInput):
    workspace: str | None = None
    limit: int = FILE_PANEL_PAGE_DEFAULT_LIMIT
    cursor: str | None = None


class RetryFilesInput(MCPInput):
    document_ids: list[str] | None = Field(default=None, max_length=100)
    selector: Literal["all_retryable"] | None = None
    workspace: str | None = None
    idempotency_key: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def _require_selector(self) -> Self:
        if bool(self.document_ids) == bool(self.selector):
            raise ValueError("provide document_ids or selector='all_retryable', but not both")
        return self


class DeleteFilesInput(MCPInput):
    filenames: list[str] | None = Field(default=None, max_length=100)
    file_paths: list[str] | None = Field(default=None, max_length=100)
    document_ids: list[str] | None = Field(default=None, max_length=100)
    workspace: str | None = None
    idempotency_key: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def _require_identifier(self) -> Self:
        if not self.filenames and not self.file_paths and not self.document_ids:
            raise ValueError("at least one exact document identifier is required")
        return self


__all__ = [
    "AnswerInput",
    "AnswerRunInput",
    "ConversationMessage",
    "CreateWorkspaceInput",
    "DeleteFilesInput",
    "IngestInput",
    "ListFilesInput",
    "RetrieveInput",
    "RetryFilesInput",
]
