# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Private PostgreSQL persistence for complete owner-scoped blobs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.engine.runtime.blob_chunks import plan_blob

_INSERT_BLOB_METADATA = """
INSERT INTO dlightrag_blobs (owner_id, digest, byte_size)
VALUES ($1, $2, $3)
ON CONFLICT (owner_id, digest) DO NOTHING
"""

_SELECT_BLOB_SIZE = """
SELECT byte_size FROM dlightrag_blobs WHERE owner_id = $1 AND digest = $2
"""

_SELECT_BLOB_CHUNKS = """
SELECT content
FROM dlightrag_blob_chunks
WHERE owner_id = $1 AND digest = $2
ORDER BY chunk_index
"""

_INSERT_BLOB_CHUNKS = """
WITH authoritative AS (
    SELECT byte_size
    FROM dlightrag_blobs
    WHERE owner_id = $1 AND digest = $2
), inserted AS (
    INSERT INTO dlightrag_blob_chunks (owner_id, digest, chunk_index, content)
    SELECT $1, $2, chunks.ordinality - 1, chunks.content
    FROM unnest($3::bytea[]) WITH ORDINALITY AS chunks(content, ordinality)
    CROSS JOIN authoritative
    WHERE authoritative.byte_size = $4
    ON CONFLICT (owner_id, digest, chunk_index) DO NOTHING
    RETURNING 1
)
SELECT byte_size FROM authoritative
"""


class BlobSizeConflict(Exception):
    """The owner already has this digest with a different authoritative size."""


class PGRunBlobStore(PostgresOperationRunner):
    """Owner-scoped RunBlobStore backed by PostgreSQL chunked ``BYTEA`` rows."""

    def __init__(self, *, pool: ConnectionPool | None = None) -> None:
        super().__init__(pool=pool)

    async def _read[T](self, operation: Callable[[Any], Awaitable[T]]) -> T:
        return await self._run(operation)

    async def write_in(self, conn: Any, *, owner_id: str, digest: str, content: bytes) -> None:
        await write_blob_content(conn, owner_id=owner_id, digest=digest, content=content)

    async def stream(
        self,
        *,
        owner_id: str,
        digest: str,
        offset: int = 0,
        length: int | None = None,
    ) -> AsyncIterator[bytes]:
        async def _operation(conn: Any) -> AsyncIterator[bytes]:
            size = await conn.fetchval(_SELECT_BLOB_SIZE, owner_id, digest)
            if size is None:
                return
            skipped = 0
            remaining = length
            async for record in conn.cursor(_SELECT_BLOB_CHUNKS, owner_id, digest):
                content = bytes(record["content"])
                end = skipped + len(content)
                if end <= offset:
                    skipped = end
                    continue
                start = max(0, offset - skipped)
                piece = content[start:]
                if remaining is not None:
                    piece = piece[:remaining]
                    remaining -= len(piece)
                skipped = end
                if piece:
                    yield piece
                if remaining is not None and remaining <= 0:
                    return

        async for piece in self._stream(_operation):
            yield piece

    async def read(self, *, owner_id: str, digest: str) -> bytes | None:
        async def _operation(conn: Any) -> bytes | None:
            if await conn.fetchval(_SELECT_BLOB_SIZE, owner_id, digest) is None:
                return None
            rows = await conn.fetch(_SELECT_BLOB_CHUNKS, owner_id, digest)
            return b"".join(bytes(row["content"]) for row in rows)

        return await self._read(_operation)

    async def size(self, *, owner_id: str, digest: str) -> int | None:
        async def _operation(conn: Any) -> int | None:
            value = await conn.fetchval(_SELECT_BLOB_SIZE, owner_id, digest)
            return None if value is None else int(value)

        return await self._read(_operation)


async def write_blob_content(
    conn: Any,
    *,
    owner_id: str,
    digest: str,
    content: bytes,
) -> None:
    """Persist complete contiguous content without copying every chunk in Python."""
    plan = plan_blob(content)
    view = memoryview(content)
    chunks = tuple(view[start:end] for start, end in plan.chunk_ranges)
    await write_complete_blob(
        conn,
        owner_id=owner_id,
        digest=digest,
        total_bytes=plan.total_bytes,
        chunks=chunks,
    )


async def write_complete_blob(
    conn: Any,
    *,
    owner_id: str,
    digest: str,
    total_bytes: int,
    chunks: Sequence[bytes | memoryview],
) -> None:
    """Persist one complete blob in two set-based statements.

    ``DO NOTHING`` avoids holding update locks on deduplicated metadata rows.
    It also waits for a concurrent insert of the same identity to settle. The
    following statement receives a fresh Read Committed snapshot, gates every
    chunk on the authoritative size, and returns that size for conflict
    detection without a third round trip.
    """
    await conn.execute(
        _INSERT_BLOB_METADATA,
        owner_id,
        digest,
        total_bytes,
    )
    authoritative_size = await conn.fetchval(
        _INSERT_BLOB_CHUNKS,
        owner_id,
        digest,
        chunks,
        total_bytes,
    )
    if authoritative_size != total_bytes:
        raise BlobSizeConflict("blob digest collision with a different byte size")
