# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Snowflake data source for RAG ingestion.

Exports query results as structured documents for ingestion.
Cortex Analyst is NOT included here (not a RAG concern).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from corprag.sourcing.base import DataSource

logger = logging.getLogger(__name__)


class SnowflakeDataSource(DataSource):
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

    def execute_query(self, query: str, source_label: str = "query") -> list[str]:
        """Execute raw SQL and cache results as documents.

        First column is treated as text content.
        Remaining columns become metadata (keyed by column name).

        Returns:
            List of document IDs
        """
        cursor = self.conn.cursor()
        cursor.execute(query)  # noqa: S608

        col_names = [desc[0] for desc in cursor.description]
        doc_ids = []
        for row in cursor:
            doc_id = str(uuid.uuid4())
            doc_data = {
                "text": str(row[0]),
                "metadata": {col_names[i]: row[i] for i in range(1, len(col_names))},
                "source": f"snowflake://{source_label}",
            }
            self._cache[doc_id] = json.dumps(doc_data).encode("utf-8")
            doc_ids.append(doc_id)

        cursor.close()
        return doc_ids

    def export_table(
        self,
        table: str,
        text_column: str,
        metadata_columns: list[str] | None = None,
        where_clause: str | None = None,
    ) -> list[str]:
        """Export Snowflake table as documents (convenience wrapper).

        Returns:
            List of document IDs
        """
        metadata_cols = metadata_columns or []
        columns = [text_column] + metadata_cols

        query = f"SELECT {', '.join(columns)} FROM {table}"  # noqa: S608
        if where_clause:
            query += f" WHERE {where_clause}"

        return self.execute_query(query, source_label=table)

    def list_documents(self, prefix: str | None = None) -> list[str]:  # noqa: ARG002
        """List all cached document IDs."""
        return list(self._cache.keys())

    def load_document(self, doc_id: str) -> bytes:
        """Load cached document content."""
        if doc_id not in self._cache:
            raise KeyError(f"Document not found: {doc_id}")
        return self._cache[doc_id]

    def close(self) -> None:
        """Close Snowflake connection."""
        self.conn.close()
        self._cache.clear()


__all__ = ["SnowflakeDataSource"]
