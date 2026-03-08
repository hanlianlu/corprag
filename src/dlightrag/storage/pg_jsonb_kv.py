# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Generic JSONB-based KV storage for PostgreSQL.

Implements LightRAG's BaseKVStorage with a single generic table, avoiding
PGKVStorage's hardcoded namespace handling that breaks custom namespaces
like ``visual_chunks``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from lightrag.base import BaseKVStorage

logger = logging.getLogger(__name__)

TABLE = "dlightrag_kv_store"

_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    workspace VARCHAR(255) NOT NULL,
    namespace VARCHAR(255) NOT NULL,
    id        VARCHAR(255) NOT NULL,
    data      JSONB NOT NULL,
    create_time TIMESTAMPTZ DEFAULT NOW(),
    update_time TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (workspace, namespace, id)
)
"""

_UPSERT = f"""
INSERT INTO {TABLE} (workspace, namespace, id, data)
VALUES ($1, $2, $3, $4::jsonb)
ON CONFLICT (workspace, namespace, id)
DO UPDATE SET data = EXCLUDED.data, update_time = NOW()
"""

_GET_BY_ID = f"SELECT data FROM {TABLE} WHERE workspace = $1 AND namespace = $2 AND id = $3"

_GET_BY_IDS = (
    f"SELECT id, data FROM {TABLE} WHERE workspace = $1 AND namespace = $2 AND id = ANY($3)"
)

_DELETE = f"DELETE FROM {TABLE} WHERE workspace = $1 AND namespace = $2 AND id = ANY($3)"

_FILTER_EXISTING = (
    f"SELECT id FROM {TABLE} WHERE workspace = $1 AND namespace = $2 AND id = ANY($3)"
)

_IS_EMPTY = f"SELECT 1 FROM {TABLE} WHERE workspace = $1 AND namespace = $2 LIMIT 1"

_DROP = f"DELETE FROM {TABLE} WHERE workspace = $1 AND namespace = $2"


@dataclass
class PGJsonbKVStorage(BaseKVStorage):
    """Generic JSONB KV storage for PostgreSQL.

    Uses a single ``dlightrag_kv_store`` table with (workspace, namespace, id)
    composite primary key. Reuses LightRAG's shared asyncpg pool via
    ``ClientManager`` (same pattern as ``PGHashIndex``).
    """

    _pool: Any = field(default=None, repr=False)

    def _get_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PGJsonbKVStorage not initialized — call initialize() first")
        return self._pool

    async def initialize(self) -> None:
        if self._pool is None:
            from lightrag.kg.postgres_impl import ClientManager

            db = await ClientManager.get_client()
            self._pool = db.pool
        await self._ensure_table()

    async def _ensure_table(self) -> None:
        async with self._get_pool().acquire() as conn:
            await conn.execute(_CREATE_TABLE)

    async def finalize(self) -> None:
        # Pool is borrowed from ClientManager — don't close it.
        # LightRAG's own storages manage the pool lifecycle.
        self._pool = None

    async def index_done_callback(self) -> None:
        # PostgreSQL commits immediately — no deferred persistence needed.
        pass

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        rows = [(self.workspace, self.namespace, k, json.dumps(v)) for k, v in data.items()]
        async with self._get_pool().acquire() as conn:
            await conn.executemany(_UPSERT, rows)

    @staticmethod
    def _parse_data(raw: Any) -> dict[str, Any]:
        """Parse JSONB data — asyncpg usually returns a dict, but some pool
        configurations may return a raw JSON string instead."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            return json.loads(raw)
        return raw

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._get_pool().acquire() as conn:
            row = await conn.fetchrow(_GET_BY_ID, self.workspace, self.namespace, id)
            if row is None:
                return None
            return self._parse_data(row["data"])

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        if not ids:
            return []
        async with self._get_pool().acquire() as conn:
            rows = await conn.fetch(_GET_BY_IDS, self.workspace, self.namespace, ids)
        lookup = {row["id"]: self._parse_data(row["data"]) for row in rows}
        return [lookup.get(id_) for id_ in ids]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        async with self._get_pool().acquire() as conn:
            rows = await conn.fetch(_FILTER_EXISTING, self.workspace, self.namespace, list(keys))
        existing = {row["id"] for row in rows}
        return keys - existing

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        async with self._get_pool().acquire() as conn:
            await conn.execute(_DELETE, self.workspace, self.namespace, ids)

    async def is_empty(self) -> bool:
        async with self._get_pool().acquire() as conn:
            row = await conn.fetchrow(_IS_EMPTY, self.workspace, self.namespace)
            return row is None

    async def drop(self) -> dict[str, str]:
        async with self._get_pool().acquire() as conn:
            result = await conn.execute(_DROP, self.workspace, self.namespace)
        logger.info("Dropped %s/%s: %s", self.workspace, self.namespace, result)
        return {"status": "success", "message": "data dropped"}
