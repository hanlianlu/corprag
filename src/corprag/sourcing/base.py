# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Base protocol for data sources."""

from __future__ import annotations

from typing import Protocol


class DataSource(Protocol):
    """Protocol for data sources.

    All data sources must implement this interface.
    Responsibilities:
        - List available documents
        - Load document content (bytes)
        - Store document content (optional)

    NOT responsible for:
        - Document parsing
        - Embedding generation
        - Vector storage
        - RAG logic
    """

    def list_documents(self, prefix: str | None = None) -> list[str]:
        """List available document identifiers."""
        ...

    def load_document(self, doc_id: str) -> bytes:
        """Load document content as bytes."""
        ...

    def save_document(self, doc_id: str, content: bytes) -> None:
        """Save document content (optional, not all sources support this)."""
        ...


__all__ = ["DataSource"]
