# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Abstract base classes for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod


class DataSource(ABC):
    """Abstract base class for sync data sources.

    All sync data sources must implement this interface.
    Responsibilities:
        - List available documents
        - Load document content (bytes)

    NOT responsible for:
        - Document parsing
        - Embedding generation
        - Vector storage
        - RAG logic
    """

    @abstractmethod
    def list_documents(self, prefix: str | None = None) -> list[str]:
        """List available document identifiers."""
        ...

    @abstractmethod
    def load_document(self, doc_id: str) -> bytes:
        """Load document content as bytes."""
        ...


class AsyncDataSource(ABC):
    """Abstract base class for async data sources.

    All async data sources must implement this interface.
    """

    @abstractmethod
    async def alist_documents(self, prefix: str | None = None) -> list[str]:
        """List available document identifiers (async)."""
        ...

    @abstractmethod
    async def aload_document(self, doc_id: str) -> bytes:
        """Load document content as bytes (async)."""
        ...


__all__ = ["AsyncDataSource", "DataSource"]
