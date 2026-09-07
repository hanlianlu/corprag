# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Recording-connection tests for set-based PostgreSQL blob persistence."""

from typing import Any

import pytest

from dlightrag.adapters.postgres.runtime.run_blob_store import BlobSizeConflict, write_complete_blob
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.engine.runtime.records import PendingArtifact


class _RecordingConnection:
    def __init__(self, authoritative_size: int | None) -> None:
        self.authoritative_size = authoritative_size
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []

    async def fetchval(self, sql: str, *args: Any) -> int | None:
        self.calls.append(("fetchval", sql, args))
        return self.authoritative_size

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append(("execute", sql, args))
        return "INSERT 0 1"

    async def executemany(self, sql: str, args: Any) -> None:
        self.calls.append(("executemany", sql, tuple(args)))


@pytest.mark.parametrize(
    "chunks",
    [
        (),
        (b"only",),
        (b"zero", b"one", b"two", b"three"),
    ],
    ids=["empty", "one-chunk", "many-chunks"],
)
async def test_complete_blob_uses_two_set_based_statements_for_every_chunk_count(
    chunks: tuple[bytes, ...],
) -> None:
    total_bytes = sum(len(chunk) for chunk in chunks)
    conn = _RecordingConnection(total_bytes)

    await write_complete_blob(
        conn,
        owner_id="owner-a",
        digest="d" * 64,
        total_bytes=total_bytes,
        chunks=chunks,
    )

    assert [kind for kind, _sql, _args in conn.calls] == ["execute", "fetchval"]
    metadata_sql = conn.calls[0][1]
    assert "ON CONFLICT (owner_id, digest) DO NOTHING" in metadata_sql
    assert "DO UPDATE" not in metadata_sql

    chunk_sql = conn.calls[1][1]
    assert "WITH authoritative AS" in chunk_sql
    assert "WHERE authoritative.byte_size = $4" in chunk_sql
    assert "SELECT byte_size FROM authoritative" in chunk_sql
    assert "unnest($3::bytea[])" in chunk_sql
    assert "WITH ORDINALITY" in chunk_sql
    assert "ordinality - 1" in chunk_sql
    assert conn.calls[1][2] == ("owner-a", "d" * 64, chunks, total_bytes)
    assert all(kind != "executemany" for kind, _sql, _args in conn.calls)


async def test_run_store_acquires_blob_identities_in_digest_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    async def record_blob(
        _self: PGRunStore,
        _conn: Any,
        _owner: str,
        blob: PendingArtifact,
    ) -> None:
        observed.append(blob.digest)

    monkeypatch.setattr(PGRunStore, "_write_blob", record_blob)
    store = object.__new__(PGRunStore)
    blobs = (
        PendingArtifact(content=b"canonical-b"),
        PendingArtifact(content=b"canonical-a"),
    )

    await store._write_blobs(object(), "owner-a", tuple(reversed(blobs)))

    assert observed == sorted(blob.digest for blob in blobs)


async def test_metadata_size_mismatch_raises_the_module_conflict_before_chunks() -> None:
    conn = _RecordingConnection(None)

    with pytest.raises(BlobSizeConflict):
        await write_complete_blob(
            conn,
            owner_id="owner-a",
            digest="d" * 64,
            total_bytes=3,
            chunks=(b"bad",),
        )

    assert [kind for kind, _sql, _args in conn.calls] == ["execute", "fetchval"]
