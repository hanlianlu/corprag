# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared browser workspace contracts for bootstrap and catalog pages."""

from dlightrag.application.access import WorkspaceRecord
from dlightrag.engine.answer.client_contracts import ClientContractModel


class WebBootstrapWorkspace(ClientContractModel):
    workspace: str
    display_name: str
    embedding_model: str


def project_workspace_record(record: WorkspaceRecord) -> WebBootstrapWorkspace:
    """Project one authorized workspace catalog row into its browser model."""
    workspace = str(record["workspace"])
    return WebBootstrapWorkspace(
        workspace=workspace,
        display_name=str(record.get("display_name") or workspace),
        embedding_model=str(record.get("embedding_model") or ""),
    )


__all__ = [
    "WebBootstrapWorkspace",
    "project_workspace_record",
]
