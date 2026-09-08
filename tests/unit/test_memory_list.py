# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded Profile Memory listing: cursor, service gate, REST, MCP, SQL contract."""

import datetime
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from dlightrag_memory import MemoryProvenance, MemoryRecord
from dlightrag_memory.errors import MemoryUnavailableError
from dlightrag_memory.store import InMemoryMemoryStore
from fastapi import HTTPException

from dlightrag.adapters.http.rest.routes.memory import list_memories as rest_list_memories
from dlightrag.adapters.mcp.tools.memory import list_memories_tool
from dlightrag.application.access import RequestScope, UserContext, request_scope_context
from dlightrag.application.memory import (
    MEMORY_LIST_PAGE_DEFAULT_LIMIT,
    MEMORY_LIST_PAGE_MAX_LIMIT,
    InMemoryMemorySettingsStore,
    MemoryDisabledError,
    MemoryListCursor,
    MemoryListCursorCodec,
    MemoryListCursorError,
    MemoryListPage,
    MemoryListPageRequest,
    MemoryService,
)

_UTC = datetime.UTC


def _provenance() -> MemoryProvenance:
    return MemoryProvenance(origin_kind="management", origin_id="request-1")


def _record(
    *,
    owner: str = "alpha",
    body: str = "No email.",
    memory_id: str | None = None,
    updated_at: datetime.datetime | None = None,
) -> MemoryRecord:
    now = updated_at or datetime.datetime.now(_UTC)
    return MemoryRecord(
        owner_id=owner,
        memory_id=memory_id or str(uuid.uuid4()),
        kind="preference",
        body=body,
        provenance=_provenance(),
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Cursor codec and page request contracts
# ---------------------------------------------------------------------------


def _codec() -> MemoryListCursorCodec:
    return MemoryListCursorCodec(b"memory-list-tests")


def test_cursor_roundtrip_is_canonical() -> None:
    codec = _codec()
    cursor = MemoryListCursor(
        updated_at=datetime.datetime(2026, 3, 4, 5, 6, 7, 123456, tzinfo=_UTC),
        memory_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )
    token = codec.encode(cursor)
    assert codec.decode(token) == cursor


@pytest.mark.parametrize(
    "token",
    [
        "",
        "only-one-part",
        "aaa.bbb",
        "x.y.z",
    ],
)
def test_malformed_tokens_are_rejected(token: str) -> None:
    with pytest.raises(MemoryListCursorError):
        _codec().decode(token)


def test_tampered_payload_fails_integrity() -> None:
    codec = _codec()
    cursor = MemoryListCursor(
        updated_at=datetime.datetime(2026, 3, 4, 5, 6, 7, 123456, tzinfo=_UTC),
        memory_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )
    token = codec.encode(cursor)
    encoded, encoded_mac = token.split(".")
    tampered = encoded[:-1] + ("A" if encoded[-1] != "A" else "B")
    with pytest.raises(MemoryListCursorError):
        codec.decode(f"{tampered}.{encoded_mac}")


def test_wrong_scope_and_version_are_rejected() -> None:
    import base64
    import hashlib
    import hmac
    import json

    secret = b"memory-list-tests"

    def make(scope: str, version: int) -> str:
        payload = json.dumps(
            {
                "memory_id": "12345678-1234-5678-1234-567812345678",
                "scope": scope,
                "updated_at": "2026-03-04T05:06:07.123456Z",
                "v": version,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        mac = hmac.new(secret, b"memory-list\0" + payload, hashlib.sha256).digest()[:16]
        encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        encoded_mac = base64.urlsafe_b64encode(mac).rstrip(b"=").decode()
        return f"{encoded}.{encoded_mac}"

    with pytest.raises(MemoryListCursorError):
        _codec().decode(make("other-scope", 1))
    with pytest.raises(MemoryListCursorError):
        _codec().decode(make("memory-list", 2))
    with pytest.raises(MemoryListCursorError):
        _codec().decode(make("memory-list", True))  # type: ignore[arg-type]


def test_noncanonical_uuid_and_timestamp_are_rejected() -> None:
    codec = _codec()

    def token(overrides: dict[str, Any]) -> str:
        import base64
        import hashlib
        import hmac
        import json

        payload = json.dumps(
            {
                "memory_id": "12345678-1234-5678-1234-567812345678",
                "scope": "memory-list",
                "updated_at": "2026-03-04T05:06:07.123456Z",
                "v": 1,
                **overrides,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        mac = hmac.new(b"memory-list-tests", b"memory-list\0" + payload, hashlib.sha256).digest()[
            :16
        ]
        encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        encoded_mac = base64.urlsafe_b64encode(mac).rstrip(b"=").decode()
        return f"{encoded}.{encoded_mac}"

    with pytest.raises(MemoryListCursorError):
        codec.decode(token({"memory_id": "not-a-uuid"}))
    with pytest.raises(MemoryListCursorError):
        codec.decode(token({"memory_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab".upper()}))
    with pytest.raises(MemoryListCursorError):
        codec.decode(token({"updated_at": "2026-03-04 05:06:07"}))


def test_page_request_validation() -> None:
    assert MemoryListPageRequest().limit == MEMORY_LIST_PAGE_DEFAULT_LIMIT
    assert MemoryListPageRequest(limit=1).limit == 1
    assert MemoryListPageRequest(limit=MEMORY_LIST_PAGE_MAX_LIMIT).limit == 100
    for bad in (0, -1, 101, True, "50"):
        with pytest.raises(ValueError):
            MemoryListPageRequest(limit=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MemoryListPageRequest(cursor="not-a-cursor")  # type: ignore[arg-type]


def test_cursor_rejects_bad_field_types() -> None:
    with pytest.raises(ValueError):
        MemoryListCursor(updated_at="now", memory_id=uuid.uuid4())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MemoryListCursor(
            updated_at=datetime.datetime.now(),  # naive
            memory_id=uuid.uuid4(),
        )
    with pytest.raises(ValueError):
        MemoryListCursor(
            updated_at=datetime.datetime.now(_UTC),
            memory_id="12345678-1234-5678-1234-567812345678",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Service gate ordering and paged delegation
# ---------------------------------------------------------------------------


def _service() -> MemoryService:
    return MemoryService(
        InMemoryMemoryStore(),
        settings_store=InMemoryMemorySettingsStore(),
        memory_list_cursor_secret=b"memory-list-tests",
    )


async def test_service_disabled_gate_precedes_any_page_work() -> None:
    service = _service()
    await service.set_enabled(owner_id="alpha", auth_mode="jwt", enabled=False)
    with pytest.raises(MemoryDisabledError):
        await service.list_active_page(owner_id="alpha", auth_mode="jwt")


async def test_service_returns_bounded_pages_with_continuation() -> None:
    service = _service()
    memory = service._memory
    for index in range(7):
        await memory.remember(
            owner_id="alpha",
            kind="preference",
            body=f"Memory {index}.",
            provenance=_provenance(),
            idempotency_key=f"key-{index}",
        )
    seen: list[str] = []
    page_request: MemoryListPageRequest | None = MemoryListPageRequest(limit=3)
    for _ in range(4):
        page = await service.list_active_page(
            owner_id="alpha",
            auth_mode="jwt",
            page=page_request,
        )
        seen.extend(record.body for record in page.records)
        page_request = (
            MemoryListPageRequest(limit=3, cursor=page.next_cursor)
            if page.next_cursor is not None
            else None
        )
        if page_request is None:
            break
    assert len(seen) == 7
    assert len(set(seen)) == 7


async def test_service_maps_after_tuple_from_cursor_and_derives_next_from_last_row() -> None:
    service = _service()
    store = service._memory
    browse_calls: list[dict[str, Any]] = []
    original_browse = service._memory.browse

    async def tracked_browse(**kwargs: Any) -> Any:
        browse_calls.append(kwargs)
        return await original_browse(**kwargs)

    store.browse = tracked_browse  # type: ignore[method-assign]
    await store.remember(
        owner_id="alpha",
        kind="preference",
        body="Only.",
        provenance=_provenance(),
        idempotency_key="key-only",
    )
    page = await service.list_active_page(
        owner_id="alpha",
        auth_mode="jwt",
        page=MemoryListPageRequest(limit=1),
    )
    assert page.next_cursor is None
    assert browse_calls == [{"owner_id": "alpha", "cursor": None, "limit": 1}]

    browse_calls.clear()
    cursor = MemoryListCursor(
        updated_at=datetime.datetime(2026, 1, 1, tzinfo=_UTC),
        memory_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )
    page = await service.list_active_page(
        owner_id="alpha",
        auth_mode="jwt",
        page=MemoryListPageRequest(limit=5, cursor=cursor),
    )
    assert page.next_cursor is None
    assert browse_calls == [
        {
            "owner_id": "alpha",
            "cursor": (cursor.updated_at, str(cursor.memory_id)),
            "limit": 5,
        }
    ]


async def test_service_rejects_continuation_after_empty_page() -> None:
    service = _service()
    service._memory.browse = AsyncMock(  # type: ignore[method-assign]
        return_value=((), (datetime.datetime(2026, 1, 1, tzinfo=_UTC), str(uuid.uuid4())))
    )
    with pytest.raises(RuntimeError, match="after an empty page"):
        await service.list_active_page(owner_id="alpha", auth_mode="jwt")


# ---------------------------------------------------------------------------
# REST route
# ---------------------------------------------------------------------------


class _MemoryFake:
    def __init__(self, *, secret: bytes) -> None:
        self.memory_list_cursor_codec = MemoryListCursorCodec(secret)
        self.list_active_page = AsyncMock()


def _request(application: Any) -> Any:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(application=application)))


def _user() -> UserContext:
    return UserContext(user_id="u-1", auth_mode="jwt")


async def test_rest_returns_page_with_next_cursor() -> None:
    memory = _MemoryFake(secret=b"memory-list-tests")
    record = _record(owner="deployment")
    cursor = MemoryListCursor(
        updated_at=datetime.datetime(2026, 3, 4, tzinfo=_UTC),
        memory_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )
    memory.list_active_page.return_value = MemoryListPage(
        records=(record,),
        next_cursor=cursor,
    )
    application = SimpleNamespace(memory=memory)

    response = await rest_list_memories(_request(application), user=_user())

    assert response["memories"] == [
        {"memory_id": record.memory_id, "kind": record.kind, "body": record.body}
    ]
    assert response["next_cursor"] == memory.memory_list_cursor_codec.encode(cursor)
    memory.list_active_page.assert_awaited_once()
    forwarded = memory.list_active_page.await_args
    assert forwarded is not None
    assert forwarded.kwargs["page"].limit == MEMORY_LIST_PAGE_DEFAULT_LIMIT
    assert forwarded.kwargs["page"].cursor is None


async def test_rest_decodes_cursor_and_passes_it_through() -> None:
    memory = _MemoryFake(secret=b"memory-list-tests")
    cursor = MemoryListCursor(
        updated_at=datetime.datetime(2026, 3, 4, tzinfo=_UTC),
        memory_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )
    memory.list_active_page.return_value = MemoryListPage(records=(), next_cursor=None)
    application = SimpleNamespace(memory=memory)

    response = await rest_list_memories(
        _request(application),
        user=_user(),
        limit=1,
        cursor=memory.memory_list_cursor_codec.encode(cursor),
    )

    assert response["next_cursor"] is None
    forwarded_call = memory.list_active_page.await_args
    assert forwarded_call is not None
    forwarded = forwarded_call.kwargs["page"]
    assert forwarded.limit == 1
    assert forwarded.cursor == cursor


async def test_rest_rejects_tampered_cursor_before_service() -> None:
    memory = _MemoryFake(secret=b"memory-list-tests")
    application = SimpleNamespace(memory=memory)

    with pytest.raises(HTTPException) as exc:
        await rest_list_memories(_request(application), user=_user(), cursor="tampered.token")
    assert exc.value.status_code == 422
    memory.list_active_page.assert_not_awaited()


async def test_rest_maps_disabled_and_unavailable_unchanged() -> None:
    memory = _MemoryFake(secret=b"memory-list-tests")
    application = SimpleNamespace(memory=memory)

    memory.list_active_page.side_effect = MemoryDisabledError()
    with pytest.raises(HTTPException) as exc:
        await rest_list_memories(_request(application), user=_user())
    assert exc.value.status_code == 409

    memory.list_active_page.side_effect = MemoryUnavailableError()
    with pytest.raises(HTTPException) as exc:
        await rest_list_memories(_request(application), user=_user())
    assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------


async def test_mcp_lists_bounded_first_page_with_has_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.adapters.mcp import server as mcp_server

    memory = _MemoryFake(secret=b"memory-list-tests")
    record = _record(owner="deployment")
    memory.list_active_page.return_value = MemoryListPage(records=(record,), next_cursor=None)
    application = SimpleNamespace(memory=memory)
    monkeypatch.setattr(mcp_server, "_ensure_application", AsyncMock(return_value=application))
    monkeypatch.setattr(mcp_server, "_owner_id", lambda: "deployment")

    with request_scope_context(RequestScope(auth_mode="jwt")):
        result = await list_memories_tool()

    assert result["memories"] == [
        {"memory_id": record.memory_id, "kind": record.kind, "body": record.body}
    ]
    assert result["has_more"] is False
    page_call = memory.list_active_page.await_args
    assert page_call is not None
    page_kwargs = page_call.kwargs
    assert page_kwargs["owner_id"] == "deployment"
    assert page_kwargs["auth_mode"] == "jwt"


async def test_mcp_reports_has_more_when_continuation_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.adapters.mcp import server as mcp_server

    memory = _MemoryFake(secret=b"memory-list-tests")
    memory.list_active_page.return_value = MemoryListPage(
        records=(),
        next_cursor=MemoryListCursor(
            updated_at=datetime.datetime(2026, 3, 4, tzinfo=_UTC),
            memory_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        ),
    )
    application = SimpleNamespace(memory=memory)
    monkeypatch.setattr(mcp_server, "_ensure_application", AsyncMock(return_value=application))
    monkeypatch.setattr(mcp_server, "_owner_id", lambda: "deployment")

    with request_scope_context(RequestScope(auth_mode="jwt")):
        result = await list_memories_tool()

    assert result["has_more"] is True


async def test_mcp_disabled_raises_public_message(monkeypatch: pytest.MonkeyPatch) -> None:
    from dlightrag.adapters.mcp import server as mcp_server

    memory = _MemoryFake(secret=b"memory-list-tests")
    memory.list_active_page.side_effect = MemoryDisabledError()
    application = SimpleNamespace(memory=memory)
    monkeypatch.setattr(mcp_server, "_ensure_application", AsyncMock(return_value=application))
    monkeypatch.setattr(mcp_server, "_owner_id", lambda: "deployment")

    with request_scope_context(RequestScope(auth_mode="jwt")):
        with pytest.raises(ValueError, match="not active"):
            await list_memories_tool()


async def test_mcp_tool_declares_the_bound_in_its_description() -> None:
    from dlightrag.adapters.mcp.server import mcp_app

    tools = await mcp_app.list_tools()
    tool = next(tool for tool in tools if tool.name == "list_memories")
    assert tool.description is not None
    assert "first 50" in tool.description
    assert "has_more" in tool.description
    assert "REST" in tool.description
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.idempotent_hint is True


# ---------------------------------------------------------------------------
# PostgreSQL package SQL/index contract
# ---------------------------------------------------------------------------


def test_package_page_sql_contract_and_index_alignment() -> None:
    from dlightrag_memory._storage.pg import (
        _RECORD_INDEXES,
        _SELECT_ACTIVE_PAGE,
        _SELECT_ACTIVE_PAGE_AFTER,
    )

    normalized_first = " ".join(_SELECT_ACTIVE_PAGE.split()).lower()
    assert "order by updated_at desc, memory_id desc" in normalized_first
    assert "limit $2" in normalized_first
    assert "offset" not in normalized_first

    normalized_after = " ".join(_SELECT_ACTIVE_PAGE_AFTER.split()).lower()
    assert "order by updated_at desc, memory_id desc" in normalized_after
    assert "(updated_at, memory_id) < ($2, $3)" in normalized_after
    assert "limit $4" in normalized_after
    assert "offset" not in normalized_after

    list_index = next(
        statement
        for statement in _RECORD_INDEXES
        if "idx_dlightrag_memory_records_list" in statement
    )
    normalized_index = " ".join(list_index.split()).lower()
    assert (
        "on dlightrag_memory_records (owner_id, status, updated_at desc, memory_id desc)"
        in normalized_index
    )
