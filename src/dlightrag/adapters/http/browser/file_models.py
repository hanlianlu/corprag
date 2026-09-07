# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Typed browser contracts for Files and durable Corpus Mutation acceptance."""

import datetime
from typing import Any

from dlightrag.application.answer_runs.client_contracts import ClientContractModel


class WebFileItem(ClientContractModel):
    file_name: str
    file_path: str


class WebFilePanelSnapshot(ClientContractModel):
    workspace: str
    files: list[WebFileItem]
    next_cursor: str | None = None


class WebCorpusRunReceipt(ClientContractModel):
    run_id: str
    run_kind: str
    lane: str
    status: str
    status_url: str
    events_url: str
    cancel_url: str
    resume_url: str
    workspace: str
    file_count: int | None = None


class WebCorpusRunStatus(WebCorpusRunReceipt):
    phase: str | None = None
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


class WebFailedFileItem(ClientContractModel):
    document_id: str
    file_name: str
    error: str
    updated_at: str


class WebFailedFilesPage(ClientContractModel):
    workspace: str
    failed: list[WebFailedFileItem]
    next_cursor: str | None = None


__all__ = [
    "WebCorpusRunReceipt",
    "WebCorpusRunStatus",
    "WebFailedFileItem",
    "WebFailedFilesPage",
    "WebFileItem",
    "WebFilePanelSnapshot",
]
