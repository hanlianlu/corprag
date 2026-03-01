# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Snowflake data source for RAG ingestion.

Exports query results as structured documents for ingestion.
Cortex Analyst is NOT included here (not a RAG concern).

Requires: pip install corprag[snowflake]
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
import uuid

logger = logging.getLogger(__name__)


def _require_snowflake() -> None:
    try:
        import snowflake.connector  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Snowflake support requires: pip install corprag[snowflake]"
        ) from e


class SnowflakeDataSource:
    """Snowflake adapter for document export and ingestion.

    Features:
    - Export query results as structured data for RAG ingestion
    - Automatic session reconnection on expiration
    """

    def __init__(
        self,
        account: str | None = None,
        user: str | None = None,
        password: str | None = None,
        warehouse: str | None = None,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        _require_snowflake()

        import snowflake.connector

        self._snowflake_connector = snowflake.connector

        self._account = account or os.getenv("SNOWFLAKE_ACCOUNT", "")
        self._user = user or os.getenv("SNOWFLAKE_USER", "")
        self._password = password or os.getenv("SNOWFLAKE_PASSWORD", "")
        self._warehouse = warehouse or os.getenv("SNOWFLAKE_WAREHOUSE", "")
        self._database = database or os.getenv("SNOWFLAKE_DATABASE", "")
        self._schema = schema or os.getenv("SNOWFLAKE_SCHEMA", "")

        self.conn = self._create_connection()
        self._cache: dict[str, bytes] = {}

    def _create_connection(self) -> Any:
        """Create a new Snowflake connection with keep-alive enabled."""
        return self._snowflake_connector.connect(
            account=self._account,
            user=self._user,
            password=self._password,
            warehouse=self._warehouse,
            database=self._database,
            schema=self._schema,
            client_session_keep_alive=True,
        )

    def export_table(
        self,
        table: str,
        text_column: str,
        metadata_columns: list[str] | None = None,
        where_clause: str | None = None,
    ) -> list[str]:
        """Export Snowflake table as documents.

        Returns:
            List of document IDs
        """
        metadata_cols = metadata_columns or []
        columns = [text_column] + metadata_cols

        query = f"SELECT {', '.join(columns)} FROM {table}"  # noqa: S608
        if where_clause:
            query += f" WHERE {where_clause}"

        cursor = self.conn.cursor()
        cursor.execute(query)

        doc_ids = []
        for row in cursor:
            doc_id = str(uuid.uuid4())
            doc_data = {
                "text": row[0],
                "metadata": {col: row[i + 1] for i, col in enumerate(metadata_cols)},
                "source": f"snowflake://{table}",
            }
            self._cache[doc_id] = json.dumps(doc_data).encode("utf-8")
            doc_ids.append(doc_id)

        cursor.close()
        return doc_ids

    def list_documents(self, prefix: str | None = None) -> list[str]:  # noqa: ARG002
        """List all cached document IDs (sync)."""
        return list(self._cache.keys())

    async def alist_documents(self, prefix: str | None = None) -> list[str]:  # noqa: ARG002
        """List all cached document IDs (async)."""
        return list(self._cache.keys())

    def load_document(self, doc_id: str) -> bytes:
        """Load cached document content."""
        if doc_id not in self._cache:
            raise KeyError(f"Document not found: {doc_id}")
        return self._cache[doc_id]

    def save_document(self, doc_id: str, content: bytes) -> None:
        """Not supported for Snowflake (read-only)."""
        raise NotImplementedError("Snowflake is read-only")

    def close(self) -> None:
        """Close Snowflake connection."""
        self.conn.close()
        self._cache.clear()


__all__ = ["SnowflakeDataSource"]
