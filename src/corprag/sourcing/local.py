# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Local filesystem data source adapter."""

from __future__ import annotations

import logging
from pathlib import Path

from corprag.sourcing.base import DataSource

logger = logging.getLogger(__name__)


class LocalDataSource(DataSource):
    """Local filesystem adapter for document loading.

    Wraps local file I/O to match the DataSource protocol.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd()

    def list_documents(self, prefix: str | None = None) -> list[str]:
        """List files in base directory.

        Args:
            prefix: Optional glob pattern (default: all files recursively)
        """
        pattern = prefix or "**/*"
        return [
            str(p.relative_to(self.base_dir)) for p in self.base_dir.glob(pattern) if p.is_file()
        ]

    def load_document(self, doc_id: str) -> bytes:
        """Load file content as bytes."""
        file_path = self.base_dir / doc_id
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path.read_bytes()

    def save_document(self, doc_id: str, content: bytes) -> None:
        """Save content to file."""
        file_path = self.base_dir / doc_id
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)


__all__ = ["LocalDataSource"]
