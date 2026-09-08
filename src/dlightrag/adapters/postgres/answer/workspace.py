# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Claim-bound PostgreSQL workspace epoch, inventory, and spill digests."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from dlightrag.adapters.postgres.core._operations import ConnectionPool
from dlightrag.adapters.postgres.core._pool import pg_pool
from dlightrag.engine.runtime.settlements import InventoryPathRecord
from dlightrag.engine.runtime.workspace import (
    CommittedSpillRecord,
    HandoffCommit,
    HandoffConflict,
    HandoffLeaseLost,
    HandoffResult,
    InventoryReplaceResult,
    _validate_spill_page_limit,
)

_LEASE = """
SELECT 1
FROM dlightrag_runs
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
FOR UPDATE
"""


class PGWorkspaceStore:
    """Fenced workspace metadata for one claimed run."""

    def __init__(
        self,
        *,
        pool: ConnectionPool | None = None,
        owner_id: str,
        run_id: uuid.UUID,
        worker_id: str,
        lease_owner: str,
        fencing_epoch: int,
    ) -> None:
        self._pool = pool
        self._owner_id = owner_id
        self._run_id = run_id
        self._worker_id = worker_id
        self._lease_owner = lease_owner
        self._fencing_epoch = fencing_epoch

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[Any]:
        pool = self._pool if self._pool is not None else await pg_pool.get()
        async with pool.acquire() as conn:
            yield conn

    async def handoff_epoch(
        self,
        *,
        expected_epoch: int | None,
        destination_epoch: int,
        inventory: Sequence[InventoryPathRecord],
    ) -> HandoffResult:
        if destination_epoch < 1:
            raise ValueError("destination epoch must be positive")
        async with self._connection() as conn:
            async with conn.transaction():
                if (
                    await conn.fetchval(
                        _LEASE, self._owner_id, self._run_id, self._lease_owner, self._fencing_epoch
                    )
                    is None
                ):
                    return HandoffLeaseLost()
                current = await conn.fetchval(
                    "SELECT agent_workspace_epoch FROM dlightrag_runs"
                    " WHERE owner_id = $1 AND run_id = $2 FOR UPDATE",
                    self._owner_id,
                    self._run_id,
                )
                current_epoch = int(current) if current is not None else None
                if current_epoch != expected_epoch:
                    return HandoffConflict(
                        expected_epoch=expected_epoch, current_epoch=current_epoch
                    )
                updated = await conn.fetchval(
                    "UPDATE dlightrag_runs SET agent_workspace_epoch = $3, updated_at = NOW()"
                    " WHERE owner_id = $1 AND run_id = $2"
                    " AND agent_workspace_epoch IS NOT DISTINCT FROM $4"
                    " RETURNING agent_workspace_epoch",
                    self._owner_id,
                    self._run_id,
                    destination_epoch,
                    expected_epoch,
                )
                if updated is None:
                    return HandoffLeaseLost()
                await self._replace_inventory_locked(conn, inventory)
                return HandoffCommit(workspace_epoch=int(updated))

    async def load_inventory(self) -> tuple[InventoryPathRecord, ...]:
        async with self._connection() as conn:
            rows = await conn.fetch(
                "SELECT relative_path, entry_type, mode, size_bytes, content_digest"
                " FROM dlightrag_answer_workspace_inventory"
                " WHERE owner_id = $1 AND run_id = $2"
                " ORDER BY relative_path",
                self._owner_id,
                self._run_id,
            )
        return tuple(_inventory_row(row) for row in rows)

    async def replace_inventory(
        self, records: Sequence[InventoryPathRecord]
    ) -> InventoryReplaceResult:
        async with self._connection() as conn:
            async with conn.transaction():
                if (
                    await conn.fetchval(
                        _LEASE, self._owner_id, self._run_id, self._lease_owner, self._fencing_epoch
                    )
                    is None
                ):
                    return "lease_lost"
                await self._replace_inventory_locked(conn, records)
                return "committed"

    async def register_spill(self, spill: CommittedSpillRecord) -> InventoryReplaceResult:
        async with self._connection() as conn:
            async with conn.transaction():
                if (
                    await conn.fetchval(
                        _LEASE, self._owner_id, self._run_id, self._lease_owner, self._fencing_epoch
                    )
                    is None
                ):
                    return "lease_lost"
                await _upsert_spill(conn, self._owner_id, self._run_id, spill)
                return "committed"

    async def load_spills_page(
        self, *, after_resource_id: str | None, limit: int
    ) -> tuple[CommittedSpillRecord, ...]:
        _validate_spill_page_limit(limit)
        async with self._connection() as conn:
            if after_resource_id is None:
                rows = await conn.fetch(
                    "SELECT resource_id, content_digest, size_bytes, session_id::text,"
                    " intent_id::text FROM dlightrag_answer_committed_spills"
                    " WHERE owner_id = $1 AND run_id = $2"
                    " ORDER BY resource_id LIMIT $3",
                    self._owner_id,
                    self._run_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    "SELECT resource_id, content_digest, size_bytes, session_id::text,"
                    " intent_id::text FROM dlightrag_answer_committed_spills"
                    " WHERE owner_id = $1 AND run_id = $2 AND resource_id > $3"
                    " ORDER BY resource_id LIMIT $4",
                    self._owner_id,
                    self._run_id,
                    after_resource_id,
                    limit,
                )
        return tuple(
            CommittedSpillRecord(
                resource_id=str(row["resource_id"]),
                content_digest=str(row["content_digest"]),
                size_bytes=int(row["size_bytes"]),
                session_id=str(row["session_id"]),
                intent_id=str(row["intent_id"]),
            )
            for row in rows
        )

    async def clear_spills(self) -> InventoryReplaceResult:
        async with self._connection() as conn:
            async with conn.transaction():
                if (
                    await conn.fetchval(
                        _LEASE, self._owner_id, self._run_id, self._lease_owner, self._fencing_epoch
                    )
                    is None
                ):
                    return "lease_lost"
                await conn.execute(
                    "DELETE FROM dlightrag_answer_committed_spills"
                    " WHERE owner_id = $1 AND run_id = $2",
                    self._owner_id,
                    self._run_id,
                )
                await conn.execute(
                    "DELETE FROM dlightrag_answer_resources"
                    " WHERE owner_id = $1 AND run_id = $2 AND kind = 'committed_spill'",
                    self._owner_id,
                    self._run_id,
                )
                return "committed"

    async def _replace_inventory_locked(
        self, conn: Any, records: Sequence[InventoryPathRecord]
    ) -> None:
        await conn.execute(
            "DELETE FROM dlightrag_answer_workspace_inventory WHERE owner_id = $1 AND run_id = $2",
            self._owner_id,
            self._run_id,
        )
        for record in records:
            await conn.execute(
                "INSERT INTO dlightrag_answer_workspace_inventory ("
                " owner_id, run_id, relative_path, entry_type, mode, size_bytes, content_digest)"
                " VALUES ($1, $2, $3, $4, $5, $6, $7)",
                self._owner_id,
                self._run_id,
                record.relative_path,
                record.entry_type,
                record.mode,
                record.size_bytes,
                record.content_digest,
            )


def _inventory_row(row: Any) -> InventoryPathRecord:
    return InventoryPathRecord(
        relative_path=str(row["relative_path"]),
        entry_type=str(row["entry_type"]),
        mode=int(row["mode"]) if row["mode"] is not None else None,
        size_bytes=int(row["size_bytes"]),
        content_digest=str(row["content_digest"]) if row["content_digest"] is not None else None,
    )


async def _upsert_spill(
    conn: Any, owner_id: str, run_id: uuid.UUID, spill: CommittedSpillRecord
) -> None:
    await conn.execute(
        "INSERT INTO dlightrag_answer_committed_spills ("
        " owner_id, run_id, resource_id, content_digest, size_bytes, session_id, intent_id)"
        " VALUES ($1, $2, $3, $4, $5, $6::uuid, $7::uuid)"
        " ON CONFLICT (owner_id, run_id, resource_id) DO UPDATE SET"
        " content_digest = EXCLUDED.content_digest, size_bytes = EXCLUDED.size_bytes",
        owner_id,
        run_id,
        spill.resource_id,
        spill.content_digest,
        spill.size_bytes,
        spill.session_id,
        spill.intent_id,
    )
    await conn.execute(
        "INSERT INTO dlightrag_answer_resources ("
        " owner_id, run_id, resource_id, kind, safe_name, media_type, capabilities,"
        " session_id, intent_id)"
        " VALUES ($1, $2, $3, 'committed_spill', $3, 'text/plain', '{}'::jsonb, $4::uuid, $5::uuid)"
        " ON CONFLICT (owner_id, run_id, resource_id) DO NOTHING",
        owner_id,
        run_id,
        spill.resource_id,
        spill.session_id,
        spill.intent_id,
    )


__all__ = ["PGWorkspaceStore"]
