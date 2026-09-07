# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one async REST helper that owns create-and-wait for durable Answer runs."""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import pytest

import dlightrag.adapters.http.client.client as client_module
from dlightrag.adapters.http.client import (
    EVENT_READ_IDLE_SECONDS,
    MAX_RECONNECT_ATTEMPTS,
    AnswerArtifact,
    AnswerAttachmentUpload,
    AnswerRunClient,
    RunCancelledError,
    RunFailedError,
    parse_sse_frames,
)

_DESCRIPTOR = {
    "run_id": "run-1",
    "run_kind": "answer",
    "lane": "query",
    "status": "queued",
    "status_url": "/runs/run-1",
    "events_url": "/runs/run-1/events",
    "cancel_url": "/runs/run-1",
}
_RESULT = {"answer": "grounded", "contexts": {"chunks": []}}
_RETRIEVAL_DESCRIPTOR = {**_DESCRIPTOR, "run_kind": "retrieval"}
_CORPUS_DESCRIPTOR = {
    **_DESCRIPTOR,
    "run_kind": "corpus_mutation",
    "lane": "corpus_mutation",
}
_CORPUS_RESULT = {"action": "ingest", "document_count": 1, "documents": []}
_RETRIEVAL_RESULT = {
    "contexts": {"chunks": [{"chunk_id": "c1", "content": "evidence"}]},
    "sources": [{"id": "1", "download_url": None}],
    "trace": {"count": 1},
    "image_descriptions": [],
}


@pytest.fixture(autouse=True)
def _no_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff and poll cadence are timing, not behavior; every test runs at once."""
    monkeypatch.setattr(client_module, "RECONNECT_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(client_module, "STATUS_POLL_SECONDS", 0.0)


def _frame(sequence: int, event: str, payload: dict[str, Any]) -> str:
    return f"id: {sequence}\nevent: {event}\ndata: {json.dumps(payload)}\n\n"


def _client(handler) -> tuple[httpx.AsyncClient, AnswerRunClient]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://rag.test")
    return http, AnswerRunClient(http)


def _dropped_stream(chunks: Sequence[str], error: type[httpx.HTTPError]) -> httpx.Response:
    """An events response that delivers ``chunks`` and then loses its connection."""

    async def _body() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk.encode()
        raise error("connection dropped")

    return httpx.Response(200, content=_body(), headers={"content-type": "text/event-stream"})


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


def test_parser_keeps_a_split_frame_until_it_completes() -> None:
    whole = _frame(1, "token", {"text": "hello"})
    head, tail = whole[:20], whole[20:]

    events, buffer = parse_sse_frames(head)
    assert events == []

    events, buffer = parse_sse_frames(tail, buffer=buffer)
    assert [event.sequence for event in events] == [1]
    assert events[0].payload == {"text": "hello"}
    assert buffer == ""


def test_parser_ignores_keepalive_comments() -> None:
    events, buffer = parse_sse_frames(": keepalive\n\n" + _frame(4, "progress", {"phase": "x"}))

    assert [event.sequence for event in events] == [4]
    assert buffer == ""


# ---------------------------------------------------------------------------
# Create and wait
# ---------------------------------------------------------------------------


async def test_answer_creates_then_follows_events_to_the_result() -> None:
    seen: list[str] = []
    tokens: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/events"):
            body = _frame(1, "token", {"text": "grou"}) + _frame(
                2, "done", {"status": "succeeded", "result": _RESULT}
            )
            return httpx.Response(200, text=body)
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        result = await runs.answer({"query": "q"}, on_token=tokens.append)

    assert result.answer == _RESULT["answer"]
    assert tokens == ["grou"]
    assert seen == ["/answer", "/runs/run-1/events"]


async def test_retrieve_exposes_create_handle_and_explicit_wait_path() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                text=_frame(
                    1,
                    "done",
                    {"status": "succeeded", "result": _RETRIEVAL_RESULT},
                ),
            )
        return httpx.Response(202, json=_RETRIEVAL_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        descriptor = await runs.create_retrieve({"query": "q"}, idempotency_key="retrieve-1")
        result = await runs.wait_retrieve(descriptor.run_id)

    assert descriptor.run_kind == "retrieval"
    assert result.contexts["chunks"][0]["content"] == "evidence"
    assert result.trace == {"count": 1}
    assert requests[0].url.path == "/retrieve"
    assert requests[0].headers["Idempotency-Key"] == "retrieve-1"
    assert requests[1].url.path == "/runs/run-1/events"


async def test_retrieve_convenience_creates_once_then_waits() -> None:
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                text=_frame(
                    1,
                    "done",
                    {"status": "succeeded", "result": _RETRIEVAL_RESULT},
                ),
            )
        return httpx.Response(202, json=_RETRIEVAL_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        result = await runs.retrieve({"query": "q"})

    assert result.sources[0]["id"] == "1"
    assert seen == ["/retrieve", "/runs/run-1/events"]


async def test_corpus_ingest_uses_the_run_native_route_and_waits_for_terminal_result() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(202, json=_CORPUS_DESCRIPTOR)
        return httpx.Response(
            200,
            json={**_CORPUS_DESCRIPTOR, "status": "succeeded", "result": _CORPUS_RESULT},
        )

    http, runs = _client(handler)
    async with http:
        result = await runs.ingest(
            {"source_type": "local", "path": "docs", "workspace": "default"},
            idempotency_key="ingest-1",
        )

    assert result == _CORPUS_RESULT
    assert [request.url.path for request in requests] == [
        "/runs/corpus/ingest",
        "/runs/run-1",
    ]
    assert requests[0].headers["Idempotency-Key"] == "ingest-1"


async def test_corpus_wait_returns_waiting_for_repair_status_without_polling_forever() -> None:
    requests: list[httpx.Request] = []
    waiting = {
        **_CORPUS_DESCRIPTOR,
        "status": "running",
        "phase": "waiting_for_repair",
        "repair_reason": "Verify the upstream deletion outcome.",
        "repair_remedy": "Repair it, then resume this Run.",
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=waiting)

    http, runs = _client(handler)
    async with http:
        result = await runs.wait_corpus_mutation("run-1")

    assert result == waiting
    assert [request.url.path for request in requests] == ["/runs/run-1"]


async def test_http_client_exposes_every_corpus_use_case_through_common_observation() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(202, json=_CORPUS_DESCRIPTOR)
        return httpx.Response(
            200,
            json={**_CORPUS_DESCRIPTOR, "status": "succeeded", "result": _CORPUS_RESULT},
        )

    http, runs = _client(handler)
    async with http:
        assert (await runs.replace({"workspace": "default"}))["action"] == "ingest"
        assert (await runs.delete({"workspace": "default", "document_ids": ["doc-1"]}))[
            "action"
        ] == "ingest"
        assert (await runs.retry({"workspace": "default", "selector": "all_retryable"}))[
            "action"
        ] == "ingest"
        assert (await runs.reset({"workspace": "default"}))["action"] == "ingest"
        assert (await runs.resume("run-1"))["action"] == "ingest"

    assert [request.url.path for request in requests] == [
        "/runs/corpus/replace",
        "/runs/run-1",
        "/runs/corpus/delete",
        "/runs/run-1",
        "/runs/corpus/retry",
        "/runs/run-1",
        "/runs/corpus/reset",
        "/runs/run-1",
        "/runs/run-1/resume",
        "/runs/run-1",
    ]


async def test_http_client_uploads_one_source_then_observes_its_run() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            assert b'filename="report.pdf"' in request.content
            assert b'name="content_sha256"' in request.content
            return httpx.Response(202, json=_CORPUS_DESCRIPTOR)
        return httpx.Response(
            200,
            json={**_CORPUS_DESCRIPTOR, "status": "succeeded", "result": _CORPUS_RESULT},
        )

    http, runs = _client(handler)
    async with http:
        result = await runs.upload(
            filename="report.pdf",
            content=b"content",
            workspace="default",
            content_sha256="a" * 64,
        )

    assert result == _CORPUS_RESULT
    assert [request.url.path for request in requests] == [
        "/runs/corpus/ingest/upload",
        "/runs/run-1",
    ]


async def test_multipart_create_sends_the_request_part_and_files() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = request.content
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        descriptor = await runs.create(
            {"query": "q"},
            attachments=[AnswerAttachmentUpload(filename="a.txt", content=b"bytes")],
            idempotency_key="key-1",
        )

    assert descriptor.run_id == "run-1"
    assert captured["content_type"].startswith("multipart/form-data")
    assert b'name="request"' in captured["body"]
    assert b"bytes" in captured["body"]


async def test_reconnect_resumes_after_the_last_sequence_without_gaps() -> None:
    cursors: list[str | None] = []
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if not request.url.path.endswith("/events"):
            return httpx.Response(202, json=_DESCRIPTOR)
        cursors.append(request.headers.get("Last-Event-ID"))
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, text=_frame(1, "token", {"text": "a"}))
        return httpx.Response(
            200,
            text=_frame(2, "token", {"text": "b"})
            + _frame(3, "done", {"status": "succeeded", "result": _RESULT}),
        )

    tokens: list[str] = []
    http, runs = _client(handler)
    async with http:
        result = await runs.answer({"query": "q"}, on_token=tokens.append)

    assert result.answer == _RESULT["answer"]
    assert tokens == ["a", "b"]
    assert cursors == [None, "1"]


@pytest.mark.parametrize("error", [httpx.ReadError, httpx.ReadTimeout])
async def test_a_dropped_stream_resumes_after_the_last_sequence(
    error: type[httpx.HTTPError],
) -> None:
    creates = 0
    cursors: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal creates
        if not request.url.path.endswith("/events"):
            creates += 1
            return httpx.Response(202, json=_DESCRIPTOR)
        cursors.append(request.headers.get("Last-Event-ID"))
        if len(cursors) == 1:
            return _dropped_stream([_frame(1, "token", {"text": "a"})], error)
        return httpx.Response(
            200,
            text=_frame(2, "token", {"text": "b"})
            + _frame(3, "done", {"status": "succeeded", "result": _RESULT}),
        )

    tokens: list[str] = []
    http, runs = _client(handler)
    async with http:
        result = await runs.answer({"query": "q"}, on_token=tokens.append)

    assert result.answer == _RESULT["answer"]
    assert tokens == ["a", "b"]
    assert cursors == [None, "1"]
    assert creates == 1


async def test_reconnect_attempts_are_bounded() -> None:
    creates = 0
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal creates, attempts
        if not request.url.path.endswith("/events"):
            creates += 1
            return httpx.Response(202, json=_DESCRIPTOR)
        attempts += 1
        return _dropped_stream([], httpx.ReadError)

    http, runs = _client(handler)
    async with http:
        with pytest.raises(httpx.ReadError):
            await runs.answer({"query": "q"})

    assert attempts == MAX_RECONNECT_ATTEMPTS
    assert creates == 1


async def test_the_event_stream_reads_with_a_bounded_idle_timeout() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            captured["timeout"] = dict(request.extensions["timeout"])
            return httpx.Response(
                200, text=_frame(1, "done", {"status": "succeeded", "result": _RESULT})
            )
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        assert (await runs.answer({"query": "q"})).answer == _RESULT["answer"]

    assert captured["timeout"]["read"] == EVENT_READ_IDLE_SECONDS
    assert all(value is not None for value in captured["timeout"].values())


async def test_expired_event_log_falls_back_to_the_status_result() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path.endswith("/events"):
            attempts += 1
            return httpx.Response(410, json={"detail": "expired"})
        if request.method == "GET":
            return httpx.Response(200, json={"status": "succeeded", "result": _RESULT})
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        assert (await runs.answer({"query": "q"})).answer == _RESULT["answer"]

    assert attempts == 1


async def test_a_status_error_is_never_retried() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path.endswith("/events"):
            attempts += 1
            return httpx.Response(503, json={"detail": "unavailable"})
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        with pytest.raises(httpx.HTTPStatusError):
            await runs.answer({"query": "q"})

    assert attempts == 1


async def test_cancelling_the_wait_detaches_without_cancelling_the_run() -> None:
    calls: list[str] = []
    streaming = asyncio.Event()
    forever = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if not request.url.path.endswith("/events"):
            return httpx.Response(202, json=_DESCRIPTOR)

        async def _body() -> AsyncIterator[bytes]:
            yield _frame(1, "token", {"text": "a"}).encode()
            streaming.set()
            await forever.wait()

        return httpx.Response(200, content=_body())

    http, runs = _client(handler)
    async with http:
        waiting = asyncio.create_task(runs.answer({"query": "q"}))
        await streaming.wait()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

    assert calls == ["POST /answer", "GET /runs/run-1/events"]


async def test_a_failed_run_raises_its_public_kind() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200, text=_frame(1, "error", {"kind": "run_abandoned", "message": "gone"})
            )
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        with pytest.raises(RunFailedError) as raised:
            await runs.answer({"query": "q"})

    assert raised.value.error_kind == "run_abandoned"
    assert raised.value.public_message == "gone"


async def test_a_cancelled_run_raises_cancellation() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(200, text=_frame(1, "done", {"status": "cancelled"}))
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        with pytest.raises(RunCancelledError):
            await runs.answer({"query": "q"})


async def test_a_stream_that_closes_early_polls_the_run_row() -> None:
    statuses = ["running", "succeeded"]

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(200, text="")
        if request.method == "GET":
            status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
            return httpx.Response(
                200,
                json={"status": status, "result": _RESULT if status == "succeeded" else None},
            )
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        assert (await runs.answer({"query": "q"})).answer == _RESULT["answer"]


async def test_cancel_is_a_plain_delete() -> None:
    seen: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(200, json={"status": "cancelled"})

    http, runs = _client(handler)
    async with http:
        assert (await runs.cancel("run-1"))["status"] == "cancelled"

    assert seen == {"method": "DELETE", "path": "/runs/run-1"}


async def test_agent_control_methods_project_the_shared_rest_contract() -> None:
    seen: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path.endswith(("/follow-up", "/fork")):
            return httpx.Response(202, json=_DESCRIPTOR)
        if request.url.path.endswith("/children"):
            return httpx.Response(200, json={"children": [{"child_session_id": "c"}]})
        if request.url.path.endswith("/transcript"):
            return httpx.Response(200, json={"run_id": "run-1", "messages": []})
        if request.url.path.endswith("/steer"):
            return httpx.Response(202, json={"run_id": "run-1", "control_sequence": 1})
        return httpx.Response(200, json={"run_id": "run-1", "status": "running"})

    http, runs = _client(handler)
    async with http:
        await runs.steer("run-1", "focus")
        await runs.follow_up("run-1", "next")
        await runs.fork("run-1", "branch")
        await runs.transcript("run-1")
        children = await runs.children("run-1")

    assert children == {
        "children": [{"child_session_id": "c"}],
        "next_cursor": None,
    }
    assert seen == [
        ("POST", "/answer/run-1/steer"),
        ("POST", "/answer/run-1/follow-up"),
        ("POST", "/answer/run-1/fork"),
        ("GET", "/answer/run-1/transcript"),
        ("GET", "/answer/run-1/children"),
    ]


async def test_children_page_forwards_cursor_and_limit_and_returns_continuation() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "children": [{"child_session_id": "c1"}],
                "next_cursor": "older-cursor",
            },
        )

    http, runs = _client(handler)
    async with http:
        first = await runs.children("run-1", limit=10)
        older = await runs.children("run-1", cursor="older-cursor", limit=10)

    assert first == {
        "children": [{"child_session_id": "c1"}],
        "next_cursor": "older-cursor",
    }
    assert older["next_cursor"] == "older-cursor"
    assert requests[0].url.params["limit"] == "10"
    assert "cursor" not in requests[0].url.params
    assert requests[1].url.params["cursor"] == "older-cursor"
    assert requests[1].url.params["limit"] == "10"


async def test_children_page_normalizes_an_absent_continuation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"children": [{"child_session_id": "c1"}]})

    http, runs = _client(handler)
    async with http:
        page = await runs.children("run-1")

    assert page == {
        "children": [{"child_session_id": "c1"}],
        "next_cursor": None,
    }


async def test_list_memories_forwards_cursor_and_limit_and_returns_continuation() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "memories": [{"memory_id": "m1", "kind": "preference", "body": "Use Chinese."}],
                "next_cursor": "older-cursor",
            },
        )

    http, client = _client(handler)
    async with http:
        first = await client.list_memories(limit=10)
        older = await client.list_memories(cursor="older-cursor", limit=10)

    assert first == {
        "memories": [{"memory_id": "m1", "kind": "preference", "body": "Use Chinese."}],
        "next_cursor": "older-cursor",
    }
    assert older["next_cursor"] == "older-cursor"
    assert requests[0].url.path == "/memory"
    assert requests[0].url.params["limit"] == "10"
    assert "cursor" not in requests[0].url.params
    assert requests[1].url.params["cursor"] == "older-cursor"


async def test_list_memories_normalizes_an_absent_continuation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"memories": [{"memory_id": "m1", "kind": "preference", "body": "Use Chinese."}]},
        )

    http, client = _client(handler)
    async with http:
        page = await client.list_memories()

    assert page == {
        "memories": [{"memory_id": "m1", "kind": "preference", "body": "Use Chinese."}],
        "next_cursor": None,
    }


async def test_profile_memory_sdk_projects_receipts_settings_and_retry_keys() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/memory/settings":
            return httpx.Response(200, json={"enabled": True, "active_count": 1})
        if request.url.path == "/memory/clear":
            return httpx.Response(204)
        return httpx.Response(
            200,
            json={
                "action": "undo" if request.url.path.endswith("/undo") else "remember",
                "outcome": "changed",
                "change_id": "change-1",
                "memory_ids": ["memory-1"],
                "kind": "preference",
                "body": "Use Chinese.",
            },
        )

    http, client = _client(handler)
    async with http:
        remembered = await client.remember_memory(
            kind="preference",
            body="Use Chinese.",
            idempotency_key="stable-key",
        )
        undone = await client.undo_memory_change(
            remembered.change_id,
            idempotency_key="undo-key",
        )
        settings = await client.memory_settings()
        await client.clear_memory()

    assert remembered.memory_ids == ("memory-1",)
    assert undone.action == "undo"
    assert settings.active_count == 1
    assert requests[0].headers["Idempotency-Key"] == "stable-key"
    assert requests[1].headers["Idempotency-Key"] == "undo-key"


_ARTIFACT_PAYLOAD = {
    "resource_id": "artifact-1",
    "role": "attachment",
    "media_type": "text/plain",
    "label": "Notes",
    "filename": "notes.txt",
    "byte_size": 5,
    "digest": "a" * 64,
    "presentation": "text",
    "status": "available",
    "uri": "dlightrag://answer/run-1/artifacts/artifact-1",
    "data_url": "/answer/run-1/artifacts/artifact-1",
    "download_url": "/answer/run-1/artifacts/artifact-1?download=1",
    "presentation_url": "/answer/run-1/artifacts/artifact-1/presentation",
}


async def test_async_sdk_lists_typed_artifacts_and_reads_bounded_bytes() -> None:
    seen_range = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_range
        if request.url.path.endswith("/artifacts"):
            return httpx.Response(200, json={"artifacts": [_ARTIFACT_PAYLOAD]})
        seen_range = request.headers.get("Range", "")
        return httpx.Response(206, content=b"notes")

    http, runs = _client(handler)
    async with http:
        (artifact,) = await runs.list_artifacts("run-1")
        data = await runs.read_artifact("run-1", artifact.resource_id, offset=2, length=5)

    assert isinstance(artifact, AnswerArtifact)
    assert artifact.uri.startswith("dlightrag://answer/")
    assert data == b"notes"
    assert seen_range == "bytes=2-6"
