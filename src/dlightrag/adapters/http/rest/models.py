# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request and response models for the DlightRAG REST API."""

import datetime
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from dlightrag.application.access import validate_query_workspace_selection
from dlightrag.application.answer_runs.citations import SourceReferencePayload
from dlightrag.application.answer_runs.client_contracts import (
    MAX_HISTORY_CONTENT_CHARS,
    MAX_HISTORY_MESSAGES,
    AnswerRequestContract,
    ClientContractModel,
    RetrieveRequestContract,
)
from dlightrag.application.corpus_admin import IngestSpec
from dlightrag.application.runs import RunPhase, RunStatus

# Maximum UTF-8 history payload plus query/workspace/JSON framing. Shared by the
# REST multipart parser and its receive-layer body cap.
ANSWER_REQUEST_PART_MAX_BYTES = MAX_HISTORY_MESSAGES * MAX_HISTORY_CONTENT_CHARS * 4 + 64 * 1024

# ═══════════════════════════════════════════════════════════════════
# Request Models
# ═══════════════════════════════════════════════════════════════════


class QueryWorkspaceSelection(ClientContractModel):
    """REST/MCP query workspace selector."""

    workspaces: list[str] | None = None
    all_workspaces: bool = False

    @model_validator(mode="after")
    def _validate_workspace_selection(self) -> Self:
        validate_query_workspace_selection(
            all_workspaces=self.all_workspaces,
            workspaces=self.workspaces,
        )
        return self


class MetadataFilterRequest(ClientContractModel):
    """Structured metadata filter for retrieval queries."""

    filename: str | None = None
    file_extension: str | None = None
    title: str | None = None
    author: str | None = None
    creation_date_from: datetime.datetime | None = None
    creation_date_to: datetime.datetime | None = None
    custom: dict[str, Any] | None = None


class IngestRequest(IngestSpec):
    workspace: str | None = None


class RetrieveRequest(QueryWorkspaceSelection, RetrieveRequestContract):
    filters: MetadataFilterRequest | None = None


class AnswerRequest(QueryWorkspaceSelection, AnswerRequestContract):
    filters: MetadataFilterRequest | None = None
    """Prior conversation turns supplied by the caller. Stateless: the client
    owns persistence and re-sends history each request; DlightRAG never stores
    it. Feeds the planner's standalone-query rewrite and answer generation."""


class DeleteRequest(ClientContractModel):
    file_paths: list[str] | None = None
    filenames: list[str] | None = None
    document_ids: list[str] | None = None
    workspace: str | None = None


class RetryRequest(ClientContractModel):
    workspace: str | None = None
    document_ids: list[str] | None = None
    selector: Literal["all_retryable"] | None = "all_retryable"


class WorkspaceCreateRequest(ClientContractModel):
    """Request to create an empty workspace."""

    workspace: str
    display_name: str | None = None


class ResetRequest(ClientContractModel):
    """Request to reset a workspace."""

    workspace: str | None = None
    supersedes_run_id: str | None = None


class MetadataUpdateRequest(ClientContractModel):
    metadata: dict[str, Any]


# ═══════════════════════════════════════════════════════════════════
# Response Models
# ═══════════════════════════════════════════════════════════════════


class ReferenceSummary(ClientContractModel):
    id: str
    title: str | None = None


class RetrievalResponse(ClientContractModel):
    contexts: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    sources: list[SourceReferencePayload] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    image_descriptions: list[str] = Field(default_factory=list)


class ArtifactIssueResponse(ClientContractModel):
    kind: str
    description: str
    resource_id: str | None = None


class ArtifactOutcomeResponse(ClientContractModel):
    status: Literal["complete", "partial", "failed"] = "complete"
    issues: list[ArtifactIssueResponse] = Field(default_factory=list)


class AnswerArtifactResponse(ClientContractModel):
    resource_id: str
    media_type: str
    label: str
    filename: str
    byte_size: int
    digest: str
    presentation: Literal["image", "markdown", "html", "pdf", "text", "download"]
    status: Literal["available", "unavailable"]
    uri: str
    width: int | None = None
    height: int | None = None
    data_url: str | None = None
    download_url: str | None = None
    presentation_url: str | None = None
    issue: ArtifactIssueResponse | None = None


class EvidenceImageResponse(ClientContractModel):
    id: str
    chunk_id: str
    source_ref: str
    url: str
    thumbnail_url: str
    label: str
    answer_image_sent: bool = True


class AnswerPartResponse(ClientContractModel):
    type: Literal["markdown", "artifact", "evidence_image"]
    text: str = ""
    artifact: AnswerArtifactResponse | None = None
    evidence_image: EvidenceImageResponse | None = None
    inline: bool = False


class AnswerResponse(RetrievalResponse):
    answer: str | None = None
    parts: list[AnswerPartResponse] = Field(default_factory=list)
    references: list[ReferenceSummary] = Field(default_factory=list)
    evidence_images: list[EvidenceImageResponse] = Field(default_factory=list)
    artifacts: list[AnswerArtifactResponse] = Field(default_factory=list)
    artifact_outcome: ArtifactOutcomeResponse = Field(default_factory=ArtifactOutcomeResponse)
    usage: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)


class RunDescriptor(ClientContractModel):
    """Common durable Run acceptance descriptor."""

    run_id: str
    run_kind: Literal["retrieval", "answer", "corpus_mutation"]
    lane: Literal["query", "corpus_mutation"]
    status: RunStatus
    status_url: str
    events_url: str
    cancel_url: str
    parent_run_id: str | None = None
    continuation_kind: str | None = None


class RunStatusResponse(RunDescriptor):
    """Authoritative lifecycle state, plus the canonical result once it exists."""

    phase: RunPhase | None = None
    durable_progress_version: int = 0
    cancel_requested: bool = False
    result: dict[str, Any] | None = None
    error_kind: str | None = None
    error_message: str | None = None
    repair_reason: str | None = None
    repair_remedy: str | None = None
    created_at: datetime.datetime | None = None
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None


class FileListResponse(ClientContractModel):
    files: list[Any]
    count: int
    workspace: str
    next_cursor: str | None = None
    fetched_rows: int


class FailedFilesResponse(ClientContractModel):
    failed: list[Any]
    count: int
    workspace: str
    next_cursor: str | None = None
    fetched_rows: int


class DeleteFilesResponse(ClientContractModel):
    results: list[dict[str, Any]]
    workspace: str


class WorkspaceRecord(ClientContractModel):
    workspace: str
    display_name: str
    embedding_model: str
    created_at: str | None = None
    updated_at: str | None = None


class WorkspacesResponse(ClientContractModel):
    workspaces: list[str]
    records: list[WorkspaceRecord]
    next_cursor: str | None = None


class WorkspaceCreateResponse(ClientContractModel):
    workspace: str
    display_name: str
    created: bool


class WorkspaceDeleteResponse(ClientContractModel):
    workspace: str
    deleted: bool
    result: dict[str, Any]


class MetadataResponse(ClientContractModel):
    doc_id: str
    metadata: dict[str, Any]


class SearchMetadataResponse(ClientContractModel):
    document_ids: list[str]
    count: int
    workspace: str
    next_cursor: str | None = None


class MetadataUpdateResponse(ClientContractModel):
    status: Literal["success"]
    doc_id: str


class ErrorDetail(ClientContractModel):
    detail: str
    error_type: str  # "unavailable", "validation", "auth", "configuration", "internal"
    error_kind: str | None = None  # stable answer-image error kind, if applicable

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize omitting the optional error_kind unless a classification applies."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)
