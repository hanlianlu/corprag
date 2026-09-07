# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded metadata-search cursor, page, and PostgreSQL adapter contracts."""

import base64
import hashlib
import hmac
import json
from typing import Any, cast

import pytest

from dlightrag.adapters.postgres.corpus import pg_metadata_search
from dlightrag.adapters.postgres.corpus.pg_metadata_index import (
    PGMetadataIndex,
    _filename_condition,
    metadata_match_conditions,
)
from dlightrag.adapters.postgres.corpus.pg_metadata_search import PGMetadataSearchStore
from dlightrag.application.corpus_admin import (
    MetadataSearchCursor,
    MetadataSearchCursorCodec,
    MetadataSearchCursorError,
    MetadataSearchPageRequest,
)
from dlightrag.engine.rag.retrieval import MetadataFilter


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

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetches.append((query, args))
        if not self.pages:
            return []
        return self.pages.pop(0)


def _row(doc_id: str) -> dict[str, Any]:
    return {"doc_id": doc_id}


def _signed_token(secret: bytes, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    mac = hmac.new(secret, b"metadata-match\0" + raw, hashlib.sha256).digest()[:16]

    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode()

    return f"{encode(raw)}.{encode(mac)}"


# ---------------------------------------------------------------------------
# Cursor codec
# ---------------------------------------------------------------------------


def test_metadata_search_cursor_round_trips_both_modes_and_max_doc_id() -> None:
    codec = MetadataSearchCursorCodec(b"cursor-secret")

    for cursor in (
        MetadataSearchCursor(workspace="finance", after_doc_id="doc-1", mode="exact"),
        MetadataSearchCursor(workspace="finance", after_doc_id="d" * 255, mode="contains"),
    ):
        assert codec.decode(codec.encode(cursor)) == cursor


def test_metadata_search_cursor_rejects_tamper_malformed_scope_version_and_mode() -> None:
    secret = b"cursor-secret"
    codec = MetadataSearchCursorCodec(secret)
    token = codec.encode(
        MetadataSearchCursor(workspace="finance", after_doc_id="doc-1", mode="exact")
    )
    encoded, mac = token.split(".")

    invalid = [
        token + "x",
        "not-a-token",
        f"{encoded}=.{mac}",
        _signed_token(
            secret,
            {
                "after_doc_id": "doc-1",
                "mode": "exact",
                "scope": "file-panel",
                "v": 1,
                "workspace": "finance",
            },
        ),
        _signed_token(
            secret,
            {
                "after_doc_id": "doc-1",
                "mode": "exact",
                "scope": "metadata-match",
                "v": 2,
                "workspace": "finance",
            },
        ),
        _signed_token(
            secret,
            {
                "after_doc_id": "doc-1",
                "mode": "regex",
                "scope": "metadata-match",
                "v": 1,
                "workspace": "finance",
            },
        ),
        _signed_token(
            secret,
            {
                "after_doc_id": "doc-1",
                "mode": ["exact"],
                "scope": "metadata-match",
                "v": 1,
                "workspace": "finance",
            },
        ),
        _signed_token(
            secret,
            {
                "after_doc_id": "doc-1",
                "mode": "exact",
                "scope": "metadata-match",
                "v": 1,
                "workspace": "finance",
                "extra": True,
            },
        ),
    ]
    for value in invalid:
        with pytest.raises(MetadataSearchCursorError):
            codec.decode(value)


def test_metadata_search_page_validation_and_cursor_invariants() -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        MetadataSearchPageRequest(limit=0)
    with pytest.raises(ValueError, match="between 1 and 100"):
        MetadataSearchPageRequest(limit=101)
    with pytest.raises(ValueError, match="integer"):
        MetadataSearchPageRequest(limit=True)
    with pytest.raises(ValueError, match="non-empty"):
        MetadataSearchCursor(workspace="finance", after_doc_id="", mode="exact")
    with pytest.raises(ValueError, match="exceeds the storage bound"):
        MetadataSearchCursor(workspace="finance", after_doc_id="d" * 256, mode="exact")
    with pytest.raises(ValueError, match="mode"):
        MetadataSearchCursor(workspace="finance", after_doc_id="doc", mode=cast(Any, "regex"))
    with pytest.raises(ValueError, match="canonical"):
        MetadataSearchCursor(workspace="Finance Reports", after_doc_id="doc", mode="exact")


# ---------------------------------------------------------------------------
# Shared condition builder keeps query() and the page path identical
# ---------------------------------------------------------------------------


def test_match_conditions_selects_the_filename_clause_by_mode() -> None:
    filters = MetadataFilter(filename="Quarterly Report")

    exact_conditions, exact_params = metadata_match_conditions(
        "finance", filters, filename_mode="exact"
    )
    contains_conditions, contains_params = metadata_match_conditions(
        "finance", filters, filename_mode="contains"
    )

    assert exact_conditions[:2] == [
        "workspace = $1",
        "_dlightrag_finalization_complete IS TRUE",
    ]
    assert (
        exact_conditions[2]
        == _filename_condition("", "Quarterly Report", filename_mode="exact", idx=2)[0]
    )
    assert (
        contains_conditions[2]
        == _filename_condition("", "Quarterly Report", filename_mode="contains", idx=2)[0]
    )
    assert exact_params == ["finance", "Quarterly Report"]
    assert contains_params == ["finance", "%Quarterly Report%"]
    with pytest.raises(ValueError, match="mode"):
        metadata_match_conditions("finance", filters, filename_mode="regex")


async def test_query_keeps_exact_then_contains_fallback_semantics() -> None:
    conn = _Conn([[], [_row("doc-2")]])
    store = PGMetadataIndex(workspace="finance")
    store._operation_pool = _Pool(conn)

    result = await store.query(MetadataFilter(filename="Quarterly Report"))

    assert result == ["doc-2"]
    assert len(conn.fetches) == 2
    assert (
        _filename_condition("", "Quarterly Report", filename_mode="exact", idx=2)[0]
        in conn.fetches[0][0]
    )
    assert (
        _filename_condition("", "Quarterly Report", filename_mode="contains", idx=2)[0]
        in conn.fetches[1][0]
    )
    assert conn.fetches[0][1] == ("finance", "Quarterly Report")
    assert conn.fetches[1][1] == ("finance", "%Quarterly Report%")


async def test_query_skips_the_widened_retry_when_exact_matches_or_no_filename() -> None:
    conn = _Conn([[_row("doc-1")]])
    store = PGMetadataIndex(workspace="finance")
    store._operation_pool = _Pool(conn)

    matched = await store.query(MetadataFilter(filename="Quarterly Report"))
    assert matched == ["doc-1"]
    assert len(conn.fetches) == 1

    conn = _Conn([[]])
    store = PGMetadataIndex(workspace="finance")
    store._operation_pool = _Pool(conn)
    assert await store.query(MetadataFilter(file_extension=".pdf")) == []
    assert len(conn.fetches) == 1


# ---------------------------------------------------------------------------
# Paged PostgreSQL adapter
# ---------------------------------------------------------------------------


async def test_page_store_fetches_limit_plus_one_and_trims_with_exact_mode() -> None:
    conn = _Conn([[_row("a"), _row("b"), _row("c")]])
    store = PGMetadataSearchStore(pool=_Pool(conn))

    page = await store.search_metadata_page(
        "finance",
        MetadataFilter(file_extension=".pdf"),
        page=MetadataSearchPageRequest(limit=2),
    )

    assert page.document_ids == ("a", "b")
    assert page.has_more is True
    assert page.fetched_rows == 3
    assert page.mode == "exact"
    query, args = conn.fetches[0]
    assert "ORDER BY doc_id ASC LIMIT $3" in query
    assert "OFFSET" not in query.upper()
    assert args == ("finance", "pdf", 3)


async def test_page_store_uses_the_cursor_bound_mode_and_doc_id_keyset() -> None:
    conn = _Conn([[_row("z")]])
    store = PGMetadataSearchStore(pool=_Pool(conn))

    page = await store.search_metadata_page(
        "finance",
        MetadataFilter(filename="Quarterly Report"),
        page=MetadataSearchPageRequest(
            limit=5,
            cursor=MetadataSearchCursor(
                workspace="finance",
                after_doc_id="doc-7",
                mode="contains",
            ),
        ),
    )

    assert page.document_ids == ("z",)
    query, args = conn.fetches[0]
    assert "doc_id > $3" in query
    assert "ORDER BY doc_id ASC LIMIT $4" in query
    assert _filename_condition("", "Quarterly Report", filename_mode="contains", idx=2)[0] in query
    assert args == ("finance", "%Quarterly Report%", "doc-7", 6)


async def test_page_store_falls_back_to_contains_only_on_empty_first_page() -> None:
    conn = _Conn([[], [_row("copy-1")]])
    store = PGMetadataSearchStore(pool=_Pool(conn))

    page = await store.search_metadata_page(
        "finance",
        MetadataFilter(filename="Quarterly Report"),
        page=MetadataSearchPageRequest(limit=10),
    )

    assert page.document_ids == ("copy-1",)
    assert page.mode == "contains"
    assert len(conn.fetches) == 2
    assert (
        _filename_condition("", "Quarterly Report", filename_mode="exact", idx=2)[0]
        in conn.fetches[0][0]
    )
    assert (
        _filename_condition("", "Quarterly Report", filename_mode="contains", idx=2)[0]
        in conn.fetches[1][0]
    )

    # An exact first page is authoritative: no widened retry is issued.
    conn = _Conn([[_row("exact-1")]])
    store = PGMetadataSearchStore(pool=_Pool(conn))
    page = await store.search_metadata_page(
        "finance",
        MetadataFilter(filename="Quarterly Report"),
        page=MetadataSearchPageRequest(limit=10),
    )
    assert page.mode == "exact"
    assert len(conn.fetches) == 1


async def test_page_store_rejects_cross_workspace_before_fetch() -> None:
    conn = _Conn([])
    store = PGMetadataSearchStore(pool=_Pool(conn))

    with pytest.raises(ValueError, match="another workspace"):
        await store.search_metadata_page(
            "finance",
            MetadataFilter(filename="Report"),
            page=MetadataSearchPageRequest(
                cursor=MetadataSearchCursor(
                    workspace="legal",
                    after_doc_id="doc",
                    mode="exact",
                )
            ),
        )

    assert conn.fetches == []


def test_paged_sql_keeps_placeholder_continuity_and_never_uses_offset() -> None:
    conditions, params = metadata_match_conditions(
        "finance",
        MetadataFilter(filename="Report", file_extension=".pdf"),
        filename_mode="exact",
    )

    first = " ".join(pg_metadata_search._paged_sql(conditions, params, after_doc_id=None).split())
    after = " ".join(
        pg_metadata_search._paged_sql(conditions, params, after_doc_id="doc-9").split()
    )

    assert "ORDER BY doc_id ASC LIMIT $4" in first
    assert "doc_id > $4" in after
    assert "ORDER BY doc_id ASC LIMIT $5" in after
    assert "OFFSET" not in first.upper()
    assert "OFFSET" not in after.upper()
