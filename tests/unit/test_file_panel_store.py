# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded file-panel cursor and PostgreSQL adapter contracts."""

import base64
import datetime
import hashlib
import hmac
import json
from typing import Any

import pytest

from dlightrag.adapters.postgres.corpus import file_panel as pg_file_panel
from dlightrag.adapters.postgres.corpus.file_panel import PGFilePanelStore
from dlightrag.application.corpus_admin import (
    FilePanelCursor,
    FilePanelCursorCodec,
    FilePanelCursorError,
    FilePanelPageRequest,
)


class _Acquire:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _Pool:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


class _Conn:
    def __init__(self, pages: list[list[dict[str, Any]]] | None = None) -> None:
        self.pages = list(pages or [])
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []
        self.executed: list[str] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetches.append((query, args))
        return self.pages.pop(0)

    async def execute(self, query: str) -> None:
        self.executed.append(query)


def _row(doc_id: str, updated_at: datetime.datetime | None) -> dict[str, Any]:
    return {"id": doc_id, "file_path": f"/files/{doc_id}.pdf", "updated_at": updated_at}


def _failed_row(doc_id: str, updated_at: datetime.datetime | None) -> dict[str, Any]:
    return {
        **_row(doc_id, updated_at),
        "error_msg": f"failed {doc_id}",
        "content_summary": "fallback",
    }


def _signed_token(secret: bytes, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    mac = hmac.new(secret, b"file-panel\0" + raw, hashlib.sha256).digest()[:16]

    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode()

    return f"{encode(raw)}.{encode(mac)}"


def test_file_panel_cursor_round_trips_null_and_naive_microseconds() -> None:
    codec = FilePanelCursorCodec(b"cursor-secret")
    timestamp = datetime.datetime(2026, 3, 4, 5, 6, 7, 123456)

    for cursor in (
        FilePanelCursor(workspace="finance", updated_at=None, doc_id="n" * 255),
        FilePanelCursor(workspace="finance", updated_at=timestamp, doc_id="doc-2"),
        FilePanelCursor(
            workspace="finance",
            updated_at=timestamp,
            doc_id="failed-1",
            view="failed",
        ),
    ):
        assert codec.decode(codec.encode(cursor)) == cursor

    legacy = _signed_token(
        b"cursor-secret",
        {
            "doc_id": "legacy-processed",
            "scope": "file-panel",
            "updated_at": None,
            "v": 1,
            "workspace": "finance",
        },
    )
    assert codec.decode(legacy) == FilePanelCursor(
        workspace="finance",
        updated_at=None,
        doc_id="legacy-processed",
    )


def test_file_panel_cursor_rejects_tamper_malformed_scope_version_and_noncanonical() -> None:
    secret = b"cursor-secret"
    codec = FilePanelCursorCodec(secret)
    cursor = FilePanelCursor(workspace="finance", updated_at=None, doc_id="doc-1")
    token = codec.encode(cursor)
    encoded, mac = token.split(".")

    invalid = [
        token + "x",
        "not-a-token",
        f"{encoded}=.{mac}",
        _signed_token(
            secret,
            {
                "doc_id": "doc-1",
                "scope": "conversation-history",
                "updated_at": None,
                "v": 1,
                "workspace": "finance",
            },
        ),
        _signed_token(
            secret,
            {
                "doc_id": "doc-1",
                "scope": "file-panel",
                "updated_at": None,
                "v": 2,
                "workspace": "finance",
            },
        ),
    ]
    for value in invalid:
        with pytest.raises(FilePanelCursorError):
            codec.decode(value)


def test_file_panel_page_validation_and_cursor_invariants() -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        FilePanelPageRequest(limit=0)
    with pytest.raises(ValueError, match="between 1 and 100"):
        FilePanelPageRequest(limit=101)
    with pytest.raises(ValueError, match="integer"):
        FilePanelPageRequest(limit=True)
    with pytest.raises(ValueError, match="timezone"):
        FilePanelCursor(
            workspace="finance",
            updated_at=datetime.datetime.now(datetime.UTC),
            doc_id="doc",
        )
    with pytest.raises(ValueError, match="non-empty"):
        FilePanelCursor(workspace="finance", updated_at=None, doc_id="")


async def test_file_panel_store_fetches_limit_plus_one_and_trims_first_page() -> None:
    timestamp = datetime.datetime(2026, 3, 4, 5, 6, 7)
    conn = _Conn([[_row("a", None), _row("b", None), _row("c", timestamp)]])
    store = PGFilePanelStore(pool=_Pool(conn))

    page = await store.list_processed_files(
        "finance",
        page=FilePanelPageRequest(limit=2),
    )

    assert [item.doc_id for item in page.items] == ["a", "b"]
    assert page.has_more is True
    assert page.fetched_rows == 3
    query, args = conn.fetches[0]
    assert query == pg_file_panel._LIST_FIRST_PAGE
    assert args == ("finance", 3)
    assert "ORDER BY s.updated_at DESC NULLS FIRST, s.id ASC" in query
    assert "INNER JOIN dlightrag_doc_metadata m" in query
    assert "m._dlightrag_finalization_complete IS TRUE" in query
    assert "OFFSET" not in query.upper()


async def test_file_panel_store_traverses_null_then_timestamp_groups() -> None:
    timestamp = datetime.datetime(2026, 3, 4, 5, 6, 7, 123456)
    conn = _Conn(
        [
            [_row("z", timestamp)],
            [_row("same-b", timestamp), _row("older", timestamp - datetime.timedelta(days=1))],
        ]
    )
    store = PGFilePanelStore(pool=_Pool(conn))

    null_page = await store.list_processed_files(
        "finance",
        page=FilePanelPageRequest(
            limit=2,
            cursor=FilePanelCursor(workspace="finance", updated_at=None, doc_id="null-z"),
        ),
    )
    timestamp_page = await store.list_processed_files(
        "finance",
        page=FilePanelPageRequest(
            limit=2,
            cursor=FilePanelCursor(
                workspace="finance",
                updated_at=timestamp,
                doc_id="same-a",
            ),
        ),
    )

    assert [item.doc_id for item in null_page.items] == ["z"]
    assert [item.doc_id for item in timestamp_page.items] == ["same-b", "older"]
    null_query, null_args = conn.fetches[0]
    timestamp_query, timestamp_args = conn.fetches[1]
    assert "(s.updated_at IS NULL AND s.id > $2)" in null_query
    assert "OR s.updated_at IS NOT NULL" in null_query
    assert null_args == ("finance", "null-z", 3)
    assert "s.updated_at < $2::timestamp" in timestamp_query
    assert "s.updated_at = $2::timestamp AND s.id > $3" in timestamp_query
    assert timestamp_args == ("finance", timestamp, "same-a", 3)


async def test_failed_file_store_returns_error_text_in_a_bounded_page() -> None:
    timestamp = datetime.datetime(2026, 3, 4, 5, 6, 7)
    conn = _Conn([[_failed_row("a", timestamp), _failed_row("b", timestamp)]])
    store = PGFilePanelStore(pool=_Pool(conn))

    page = await store.list_failed_files(
        "finance",
        page=FilePanelPageRequest(limit=1),
    )

    assert [(item.doc_id, item.error) for item in page.items] == [("a", "failed a")]
    assert page.has_more is True
    assert page.fetched_rows == 2
    query, args = conn.fetches[0]
    assert query == pg_file_panel._LIST_FAILED_FIRST_PAGE
    assert args == ("finance", 2)
    assert "status = 'failed'" in query
    assert "_dlightrag_finalization_complete" not in query


async def test_file_panel_store_rejects_cross_view_cursor_before_fetch() -> None:
    conn = _Conn([])
    store = PGFilePanelStore(pool=_Pool(conn))
    failed_cursor = FilePanelCursor(
        workspace="finance",
        updated_at=None,
        doc_id="failed-1",
        view="failed",
    )

    with pytest.raises(ValueError, match="another view"):
        await store.list_processed_files(
            "finance",
            page=FilePanelPageRequest(cursor=failed_cursor),
        )
    with pytest.raises(ValueError, match="another view"):
        await store.list_failed_files(
            "finance",
            page=FilePanelPageRequest(
                cursor=FilePanelCursor(
                    workspace="finance",
                    updated_at=None,
                    doc_id="processed-1",
                )
            ),
        )

    assert conn.fetches == []


async def test_file_panel_store_rejects_cross_workspace_before_fetch() -> None:
    conn = _Conn([])
    store = PGFilePanelStore(pool=_Pool(conn))

    with pytest.raises(ValueError, match="another workspace"):
        await store.list_processed_files(
            "finance",
            page=FilePanelPageRequest(
                cursor=FilePanelCursor(workspace="legal", updated_at=None, doc_id="doc")
            ),
        )

    assert conn.fetches == []


async def test_file_panel_writer_creates_exact_partial_page_index() -> None:
    conn = _Conn()
    store = PGFilePanelStore(pool=_Pool(conn))

    await store.ensure_page_index()

    assert conn.executed == [
        pg_file_panel._CREATE_PAGE_INDEX,
        pg_file_panel._CREATE_FAILED_PAGE_INDEX,
    ]
    normalized = " ".join(conn.executed[0].split())
    assert "(workspace, updated_at DESC NULLS FIRST, id ASC)" in normalized
    assert "WHERE status = 'processed'" in normalized
    failed_normalized = " ".join(conn.executed[1].split())
    assert "(workspace, updated_at DESC NULLS FIRST, id ASC)" in failed_normalized
    assert "WHERE status = 'failed'" in failed_normalized
