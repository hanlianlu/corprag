# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""REST contract for the durable Answer run: create, status, events, cancel."""

import datetime
import json
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.server import create_app
from dlightrag.application.access import UserContext, owner_id_from_user
from dlightrag.application.answer_runs import ChildRosterCursorCodec, ChildRosterPage
from dlightrag.application.config import DlightragConfig
from dlightrag.application.runs import IdempotencyKeyConflict, RunCancellation, RunView
from dlightrag.engine.runtime import (
    RunAccessScope,
    RunCreation,
    RunEvent,
    RunRecord,
)

_ANON = UserContext(user_id="anonymous", auth_mode="none")
_RUN_ID = "0199a0a0-0000-7000-8000-000000000001"
_NOW = datetime.datetime(2026, 8, 13, tzinfo=datetime.UTC)


def _record(**overrides: Any) -> RunRecord:
    fields: dict[str, Any] = {
        "run_id": _RUN_ID,
        "run_kind": "answer",
        "lane": "query",
        "submitted_by": owner_id_from_user(_ANON),
        "access_scope": RunAccessScope(kind="owner", scope_id=owner_id_from_user(_ANON)),
        "submission_key": _RUN_ID,
        "request_fingerprint": "test-fingerprint",
        "prepared_input": {"query": "hi", "workspaces": ["default"]},
        "status": "queued",
        "phase": None,
        "stop_reason": None,
        "cancel_requested_at": None,
        "lease_owner": None,
        "lease_expires_at": None,
        "fencing_epoch": 0,
        "durable_progress_version": 0,
        "last_reclaim_progress_version": 0,
        "reclaims_without_progress": 0,
        "next_event_sequence": 1,
        "events_trimmed_at": None,
        "result": None,
        "error_kind": None,
        "error_message": None,
        "created_at": _NOW,
        "updated_at": _NOW,
        "started_at": None,
        "finished_at": None,
        "purge_after": None,
        "next_attempt_at": None,
        "checkpoint": None,
    }
    fields.update(overrides)
    return RunRecord(**fields)


def _event(sequence: int, event_type: str, payload: dict[str, Any]) -> RunEvent:
    return RunEvent(
        sequence=sequence,
        event_type=event_type,  # pyright: ignore[reportArgumentType]
        payload=payload,
        created_at=_NOW,
    )


class _RunApplication:
    """An Application-shaped fake exposing the AnswerService transport port."""

    def __init__(self, config: DlightragConfig) -> None:
        self.config = config
        self.corpora = SimpleNamespace(
            alist_workspace_records=AsyncMock(return_value=[{"workspace": "default"}])
        )
        self.runs = self
        self.answers = self
        self.child_roster_cursor_codec = ChildRosterCursorCodec(b"run-api-test")
        self.created: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self.subscriptions: list[dict[str, Any]] = []
        self.record: RunRecord | None = _record()
        self.events: list[RunEvent] = []
        self.conflict = False
        self.replayed = False
        self.replay_record: RunRecord | None = None
        self.cancellation = RunCancellation(
            outcome="pending", run=RunView.from_runtime(_record(status="running"))
        )
        self.closed_subscribers = 0
        self.artifact_bytes: bytes | None = None
        self.controls: list[str] = []
        self.continuations: list[dict[str, Any]] = []

    async def list_artifacts(self, *, owner_id: str, run_id: str) -> tuple[Any, ...] | None:
        del owner_id, run_id
        if self.record is None or self.record.run_kind != "answer":
            return None
        return ()

    async def artifact_size(self, *, owner_id: str, run_id: str, resource_id: str) -> int | None:
        del owner_id, run_id, resource_id
        if self.record is None or self.record.run_kind != "answer":
            return None
        return None if self.artifact_bytes is None else len(self.artifact_bytes)

    async def read_artifact(self, *, owner_id: str, run_id: str, resource_id: str) -> bytes | None:
        del owner_id, run_id, resource_id
        if self.record is None or self.record.run_kind != "answer":
            return None
        return self.artifact_bytes

    async def open_artifact(
        self,
        *,
        owner_id: str,
        run_id: str,
        resource_id: str,
        offset: int = 0,
        length: int | None = None,
    ) -> AsyncIterator[bytes] | None:
        del owner_id, run_id, resource_id
        if self.record is None or self.record.run_kind != "answer":
            return None
        if self.artifact_bytes is None:
            return None
        blob = self.artifact_bytes[max(0, offset) :]
        if length is not None:
            blob = blob[: max(0, length)]

        async def _iterate() -> AsyncIterator[bytes]:
            yield blob

        return _iterate()

    async def create(
        self,
        request: Any,
        *,
        owner_id: str,
        idempotency_key: str | None = None,
        auth_mode: str = "none",
    ) -> RunCreation:
        del auth_mode
        if self.conflict:
            raise IdempotencyKeyConflict("reused")
        if self.replay_record is not None:
            return RunCreation(run=self.replay_record, replayed=True)
        self.created.append(
            {
                "owner_id": owner_id,
                "request": request,
                "idempotency_key": idempotency_key,
            }
        )
        record = self.record or _record()
        return RunCreation(run=record, replayed=self.replayed)

    async def get(self, *, owner_id: str, run_id: str) -> RunRecord | None:
        del owner_id, run_id
        return self.record

    async def get_global(self, *, run_id: str) -> RunView | None:
        del run_id
        return RunView.from_runtime(self.record) if self.record is not None else None

    async def steer(self, *, owner_id: str, run_id: str, instruction: str) -> Any:
        del owner_id
        if self.record is None or self.record.run_kind != "answer":
            return None
        self.controls.append(instruction)
        return SimpleNamespace(run_id=run_id, control_sequence=len(self.controls), kind="steer")

    async def continuation_workspaces(
        self, *, owner_id: str, run_id: str
    ) -> tuple[str, ...] | None:
        del owner_id, run_id
        if self.record is None or self.record.run_kind != "answer" or not self.record.terminal:
            return None
        return tuple(str(item) for item in self.record.request_input().get("workspaces") or ())

    async def follow_up(self, **kwargs: Any) -> RunCreation | None:
        if self.record is None or self.record.run_kind != "answer":
            return None
        self.controls.append(f"follow:{kwargs['query']}")
        self.continuations.append(dict(kwargs))
        return RunCreation(run=self.record or _record(), replayed=False)

    async def fork(self, **kwargs: Any) -> RunCreation | None:
        if self.record is None or self.record.run_kind != "answer":
            return None
        self.controls.append(f"fork:{kwargs['query']}")
        self.continuations.append(dict(kwargs))
        return RunCreation(run=self.record or _record(), replayed=False)

    async def transcript_tail(self, *, owner_id: str, run_id: str, limit: int) -> Any:
        del owner_id, limit
        if self.record is None or self.record.run_kind != "answer":
            return None
        return SimpleNamespace(
            run_id=run_id,
            status=(self.record or _record()).status,
            messages=({"role": "user", "content": "hi"},),
        )

    async def children(
        self,
        *,
        owner_id: str,
        run_id: str,
        page: Any = None,
    ) -> Any:
        del owner_id, run_id, page
        if self.record is None or self.record.run_kind != "answer":
            return None
        return ChildRosterPage(
            children=(({"child_session_id": "child-1", "status": "running"},)),
            next_cursor=None,
            fetched_rows=1,
        )

    async def resume(self, *, owner_id: str, run_id: str) -> RunRecord | None:
        del owner_id, run_id
        return self.record

    async def cancel(self, *, owner_id: str, run_id: str) -> RunCancellation:
        del owner_id
        self.cancelled.append(run_id)
        return self.cancellation

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncIterator[RunEvent]:
        self.subscriptions.append(
            {"owner_id": owner_id, "run_id": run_id, "after_sequence": after_sequence}
        )
        events = [event for event in self.events if event.sequence > after_sequence]
        owner = self

        async def _iterate() -> AsyncIterator[RunEvent]:
            try:
                for event in events:
                    yield event
            finally:
                owner.closed_subscribers += 1

        return _iterate()


@pytest.fixture
def _app(test_config: DlightragConfig) -> Iterator[FastAPI]:
    application = create_app(include_web_app=False)
    application.dependency_overrides[get_current_user] = lambda: _ANON
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def run_application(_app: FastAPI, test_config: DlightragConfig) -> _RunApplication:
    application = _RunApplication(test_config)
    _app.state.application = application
    return application


@pytest.fixture
async def client(_app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://test") as api:
        yield api


def _sse_frames(text: str) -> list[dict[str, str]]:
    frames: list[dict[str, str]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        frame: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith(":"):
                frame["comment"] = line[1:].strip()
            elif ": " in line:
                key, value = line.split(": ", 1)
                frame[key] = value
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Request contract
# ---------------------------------------------------------------------------


def test_answer_request_has_no_stream_field() -> None:
    from dlightrag.adapters.http.rest.models import AnswerRequest

    assert "stream" not in AnswerRequest.model_fields


class TestCreate:
    async def test_json_create_returns_202_descriptor(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        response = await client.post("/answer", json={"query": "hello"})

        assert response.status_code == 202
        body = response.json()
        assert body == {
            "run_id": _RUN_ID,
            "run_kind": "answer",
            "lane": "query",
            "status": "queued",
            "status_url": f"/runs/{_RUN_ID}",
            "events_url": f"/runs/{_RUN_ID}/events",
            "cancel_url": f"/runs/{_RUN_ID}",
            "parent_run_id": None,
            "continuation_kind": None,
        }
        assert run_application.created[0]["request"].query == "hello"

    async def test_unknown_stream_field_is_rejected(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        response = await client.post("/answer", json={"query": "hello", "stream": False})

        assert response.status_code == 422
        assert not run_application.created

    async def test_multipart_create_persists_uploaded_bytes_with_the_run(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        response = await client.post(
            "/answer",
            data={"request": json.dumps({"query": "summarize"})},
            files=[("attachments", ("notes.txt", b"payload", "text/plain"))],
        )

        assert response.status_code == 202
        created = run_application.created[0]
        resource = created["request"].resources[0]
        assert resource.content == b"payload"
        assert resource.filename == "notes.txt"

    async def test_octet_stream_image_bytes_reach_the_answer_service_unchanged(
        self,
        client: AsyncClient,
        run_application: _RunApplication,
    ) -> None:
        payload = b"\x89PNG\r\n\x1a\nnot-promoted-by-content-sniffing"

        response = await client.post(
            "/answer",
            data={"request": json.dumps({"query": "summarize"})},
            files=[("attachments", ("chart.png", payload, "application/octet-stream"))],
        )

        assert response.status_code == 202
        (resource,) = run_application.created[0]["request"].resources
        assert resource.declared_mime == "application/octet-stream"
        assert resource.content == payload
        assert resource.loader is None

    async def test_idempotent_replay_returns_the_current_status(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.replay_record = _record(status="running", submission_key="key-1")

        response = await client.post(
            "/answer", json={"query": "hello"}, headers={"Idempotency-Key": "key-1"}
        )

        assert response.status_code == 202
        assert response.json()["status"] == "running"
        assert run_application.created == []

    async def test_idempotency_conflict_is_409(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.conflict = True

        response = await client.post(
            "/answer", json={"query": "hello"}, headers={"Idempotency-Key": "key-1"}
        )

        assert response.status_code == 409

    @pytest.mark.parametrize("key", ["", "   "])
    async def test_a_blank_idempotency_key_is_no_key(
        self, client: AsyncClient, run_application: _RunApplication, key: str
    ) -> None:
        response = await client.post(
            "/answer", json={"query": "hello"}, headers={"Idempotency-Key": key}
        )

        assert response.status_code == 202
        assert run_application.created[0]["idempotency_key"] is None

    async def test_a_meaningful_idempotency_key_is_passed_through_verbatim(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        response = await client.post(
            "/answer", json={"query": "hello"}, headers={"Idempotency-Key": "Key 1"}
        )

        assert response.status_code == 202
        assert run_application.created[0]["idempotency_key"] == "Key 1"

    async def test_authorized_workspaces_are_stored_on_the_run(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        response = await client.post("/answer", json={"query": "hello"})

        assert response.status_code == 202
        assert run_application.created[0]["request"].workspaces == ("default",)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    async def test_unknown_run_is_404(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = None

        response = await client.get(f"/runs/{_RUN_ID}")

        assert response.status_code == 404

    async def test_malformed_run_id_is_404(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = None

        response = await client.get("/answer/not-a-uuid")

        assert response.status_code == 404

    async def test_running_status_reports_phase_and_cancellation(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = _record(
            status="running",
            phase="researching",
            durable_progress_version=2,
            cancel_requested_at=_NOW,
        )

        body = (await client.get(f"/runs/{_RUN_ID}")).json()

        assert body["status"] == "running"
        assert body["phase"] == "researching"
        assert body["durable_progress_version"] == 2
        assert body["cancel_requested"] is True
        assert body["result"] is None

    async def test_succeeded_status_projects_fresh_download_urls(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = _record(status="succeeded", result=_stored_result())

        body = (await client.get(f"/runs/{_RUN_ID}")).json()

        result = body["result"]
        assert result["answer"] == "Answer [1-1]."
        source = result["sources"][0]
        assert source["download_url"] == "/files/raw/doc-report?workspace=default"
        assert "workspace" not in source
        assert result["references"] == [{"id": "1", "title": "report.pdf"}]

    async def test_failed_status_reports_public_error(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = _record(
            status="failed",
            error_kind="answer_stream_failed",
            error_message="Answer run failed.",
            finished_at=_NOW,
        )

        body = (await client.get(f"/runs/{_RUN_ID}")).json()

        assert body["error_kind"] == "answer_stream_failed"
        assert body["error_message"] == "Answer run failed."


def _stored_result() -> dict[str, Any]:
    return {
        "answer": "Answer [1-1].",
        "contexts": {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "file_path": "report.pdf",
                    "content": "Evidence",
                    "_workspace": "default",
                }
            ],
            "entities": [],
            "relationships": [],
        },
        "sources": [
            {
                "id": "1",
                "title": "report.pdf",
                "type": None,
                "source_uri": "s3://bucket/report.pdf",
                "workspace": "default",
                "document_id": "doc-report",
                "cited_chunk_ids": ["c1"],
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "chunk_idx": 1,
                        "page_number": 2,
                        "content": "Evidence",
                        "highlight_phrases": None,
                        "has_visual": True,
                    }
                ],
            }
        ],
        "evidence_images": [
            {
                "id": "c1",
                "chunk_id": "c1",
                "workspace": "default",
                "source_ref": "1-1",
                "label": "report.pdf · Page 2",
                "answer_image_sent": False,
            }
        ],
        "artifacts": [],
        "artifact_outcome": {"status": "complete", "issues": []},
        "trace": {"agent_turns": 1},
        "image_descriptions": [],
    }


class TestResultProjection:
    async def test_evidence_images_keep_stored_transport_state(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = _record(status="succeeded", result=_stored_result())

        result = (await client.get(f"/runs/{_RUN_ID}")).json()["result"]

        image = result["evidence_images"][0]
        assert image["answer_image_sent"] is False
        assert image["url"] == "/images/default/c1?size=full"
        assert "workspace" not in image
        assert result["parts"][0]["type"] == "markdown"
        assert result["parts"][0]["text"] == "Answer [1-1]."
        assert "answer_images" not in result

    async def test_unauthorized_visual_workspace_drops_images(
        self, client: AsyncClient, run_application: _RunApplication, _app: FastAPI
    ) -> None:
        run_application.record = _record(status="succeeded", result=_stored_result())
        _app.state.access_control = _QueryOnlyAccess()

        result = (await client.get(f"/runs/{_RUN_ID}")).json()["result"]

        assert result["evidence_images"] == []
        assert result["sources"][0]["chunks"][0]["image_url"] is None
        assert result["sources"][0]["download_url"] is None


class _QueryOnlyAccess:
    """A caller allowed to query but not to download sources or read visuals."""

    async def check(self, user: Any, action: str, *, workspace: str | None = None) -> None:
        del user, action, workspace

    async def filter_workspaces(self, user: Any, action: str, workspaces: list[str]) -> list[str]:
        del user
        if action in {"workspace.download_source", "workspace.read_visual_asset"}:
            return []
        return list(workspaces)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestEvents:
    async def test_events_replay_from_sequence_one_without_a_cursor(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = _record(status="succeeded", result=_stored_result())
        run_application.events = [
            _event(1, "progress", {"phase": "planning"}),
            _event(2, "token", {"text": "hi"}),
            _event(3, "done", {"status": "succeeded", "result": _stored_result()}),
        ]

        response = await client.get(f"/runs/{_RUN_ID}/events")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = _sse_frames(response.text)
        assert [frame["id"] for frame in frames] == ["1", "2", "3"]
        assert [frame["event"] for frame in frames] == ["progress", "token", "done"]
        assert run_application.subscriptions[0]["after_sequence"] == 0

    async def test_done_event_projects_the_canonical_result(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = _record(status="succeeded", result=_stored_result())
        run_application.events = [
            _event(1, "done", {"status": "succeeded", "result": _stored_result()})
        ]

        response = await client.get(f"/runs/{_RUN_ID}/events")

        payload = json.loads(_sse_frames(response.text)[0]["data"])
        source = payload["result"]["sources"][0]
        assert source["download_url"] == "/files/raw/doc-report?workspace=default"

    async def test_last_event_id_header_resumes_after_that_sequence(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = _record(status="succeeded", result=_stored_result())
        run_application.events = [
            _event(1, "token", {"text": "a"}),
            _event(2, "token", {"text": "b"}),
        ]

        response = await client.get(f"/runs/{_RUN_ID}/events", headers={"Last-Event-ID": "1"})

        assert [frame["id"] for frame in _sse_frames(response.text)] == ["2"]
        assert run_application.subscriptions[0]["after_sequence"] == 1

    async def test_after_query_parameter_resumes(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = _record(status="succeeded", result=_stored_result())
        run_application.events = [
            _event(1, "token", {"text": "a"}),
            _event(2, "token", {"text": "b"}),
        ]

        await client.get(f"/runs/{_RUN_ID}/events?after=1")

        assert run_application.subscriptions[0]["after_sequence"] == 1

    async def test_conflicting_cursors_are_400(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        response = await client.get(
            f"/runs/{_RUN_ID}/events?after=2", headers={"Last-Event-ID": "1"}
        )

        assert response.status_code == 400
        assert not run_application.subscriptions

    async def test_matching_cursors_are_accepted(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = _record(status="succeeded", result=_stored_result())

        response = await client.get(
            f"/runs/{_RUN_ID}/events?after=2", headers={"Last-Event-ID": "2"}
        )

        assert response.status_code == 200
        assert run_application.subscriptions[0]["after_sequence"] == 2

    @pytest.mark.parametrize("cursor", ["-1", "abc", "1.5", ""])
    async def test_malformed_cursor_is_400(
        self, client: AsyncClient, run_application: _RunApplication, cursor: str
    ) -> None:
        response = await client.get(f"/runs/{_RUN_ID}/events?after={cursor}")

        assert response.status_code == 400

    async def test_empty_last_event_id_replays_from_the_beginning(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = _record(status="succeeded", result=_stored_result())
        run_application.events = [_event(1, "token", {"text": "a"})]

        response = await client.get(f"/runs/{_RUN_ID}/events", headers={"Last-Event-ID": ""})

        assert response.status_code == 200
        assert run_application.subscriptions[0]["after_sequence"] == 0

    @pytest.mark.parametrize("cursor", ["abc", "-1", "1.5"])
    async def test_malformed_last_event_id_is_400(
        self, client: AsyncClient, run_application: _RunApplication, cursor: str
    ) -> None:
        response = await client.get(f"/runs/{_RUN_ID}/events", headers={"Last-Event-ID": cursor})

        assert response.status_code == 400
        assert not run_application.subscriptions

    async def test_unknown_run_events_are_404(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = None

        response = await client.get(f"/runs/{_RUN_ID}/events")

        assert response.status_code == 404

    async def test_trimmed_terminal_event_log_is_410(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = _record(
            status="succeeded", result=_stored_result(), finished_at=_NOW, events_trimmed_at=_NOW
        )

        response = await client.get(f"/runs/{_RUN_ID}/events")

        assert response.status_code == 410

    async def test_following_events_never_cancels_the_run(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.record = _record(status="succeeded", result=_stored_result())
        run_application.events = [
            _event(1, "done", {"status": "succeeded", "result": _stored_result()})
        ]

        await client.get(f"/runs/{_RUN_ID}/events")

        assert run_application.cancelled == []
        assert run_application.closed_subscribers == 1


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancel:
    async def test_queued_cancellation_returns_200_terminal_state(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        cancelled = _record(status="cancelled", finished_at=_NOW)
        run_application.cancellation = RunCancellation(
            outcome="cancelled", run=RunView.from_runtime(cancelled)
        )
        run_application.record = cancelled

        response = await client.delete(f"/runs/{_RUN_ID}")

        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    async def test_running_cancellation_is_202(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        running = _record(status="running", cancel_requested_at=_NOW)
        run_application.cancellation = RunCancellation(
            outcome="pending", run=RunView.from_runtime(running)
        )

        response = await client.delete(f"/runs/{_RUN_ID}")

        assert response.status_code == 202
        assert response.json()["cancel_requested"] is True

    async def test_terminal_cancellation_is_idempotent_200(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        finished = _record(status="succeeded", result=_stored_result(), finished_at=_NOW)
        run_application.cancellation = RunCancellation(
            outcome="already_terminal", run=RunView.from_runtime(finished)
        )

        response = await client.delete(f"/runs/{_RUN_ID}")

        assert response.status_code == 200
        assert response.json()["status"] == "succeeded"

    async def test_unknown_run_cancellation_is_404(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        run_application.cancellation = RunCancellation(outcome="unknown", run=None)

        response = await client.delete(f"/runs/{_RUN_ID}")

        assert response.status_code == 404

    async def test_both_cancellation_responses_are_documented(self, client: AsyncClient) -> None:
        responses = (await client.get("/openapi.json")).json()["paths"]["/runs/{run_id}"]["delete"][
            "responses"
        ]

        assert set(responses) >= {"200", "202"}
        assert (
            responses["202"]["content"]["application/json"]["schema"]
            == responses["200"]["content"]["application/json"]["schema"]
        )


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


async def test_schema_validation_error_is_a_safe_503(
    client: AsyncClient, run_application: _RunApplication
) -> None:
    from dlightrag.engine.runtime import RunSchemaError

    run_application.get_global = AsyncMock(  # pyright: ignore[reportAttributeAccessIssue]
        side_effect=RunSchemaError("column dlightrag_runs.secret is missing")
    )

    response = await client.get(f"/runs/{_RUN_ID}")

    assert response.status_code == 503
    assert "dlightrag_runs" not in response.text


# ---------------------------------------------------------------------------
# Artifact reads
# ---------------------------------------------------------------------------

_ARTIFACT_ID = "artifact-report"


def _publish_test_artifact(
    application: _RunApplication,
    *,
    media_type: str = "text/markdown",
    presentation: str = "markdown",
    filename: str = "analysis.md",
    content: bytes = b"0123456789",
) -> None:
    result = _stored_result()
    result["artifacts"] = [
        {
            "resource_id": _ARTIFACT_ID,
            "media_type": media_type,
            "label": "Analysis",
            "filename": filename,
            "byte_size": len(content),
            "digest": "a" * 64,
            "presentation": presentation,
            "status": "available",
        }
    ]
    application.record = _record(status="succeeded", result=result)
    application.artifact_bytes = content


async def test_artifact_list_does_not_invent_an_in_flight_outcome(
    client: AsyncClient, run_application: _RunApplication
) -> None:
    assert run_application.record is not None
    assert run_application.record.result is None
    response = await client.get(f"/answer/{_RUN_ID}/artifacts")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Answer artifacts are not available until the run has a stored result"
    )


async def test_artifact_list_returns_typed_owner_links_without_bytes(
    client: AsyncClient, run_application: _RunApplication
) -> None:
    _publish_test_artifact(run_application)

    response = await client.get(f"/answer/{_RUN_ID}/artifacts")

    assert response.status_code == 200
    artifact = response.json()["artifacts"][0]
    assert "role" not in artifact
    assert artifact["uri"].startswith("dlightrag://answer/")
    assert artifact["data_url"].endswith(f"/{_RUN_ID}/artifacts/{_ARTIFACT_ID}")
    assert "content" not in artifact
    assert response.json()["artifact_outcome"]["status"] == "complete"


async def test_markdown_artifact_presentation_route_matches_its_advertised_url(
    client: AsyncClient, run_application: _RunApplication
) -> None:
    _publish_test_artifact(run_application)
    run_application.artifact_bytes = b"# Report\n\nBody"

    listed = await client.get(f"/answer/{_RUN_ID}/artifacts")
    presentation_url = listed.json()["artifacts"][0]["presentation_url"]
    response = await client.get(presentation_url)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json()["answer"] == "# Report\n\nBody"
    assert response.json()["parts"][0]["text"] == "# Report\n\nBody"
    assert response.json()["artifacts"][0]["uri"].startswith("dlightrag://answer/")


async def test_markdown_presentation_returns_artifact_scoped_sources(
    client: AsyncClient, run_application: _RunApplication
) -> None:
    _publish_test_artifact(run_application, filename="analysis.md")
    assert run_application.record is not None
    assert run_application.record.result is not None
    result = dict(run_application.record.result)
    result["artifact_sources"] = {_ARTIFACT_ID: result["sources"]}
    run_application.record = _record(status="succeeded", result=result)
    run_application.artifact_bytes = b"# Analysis\n\nEvidence [1-1]."

    response = await client.get(f"/answer/{_RUN_ID}/artifacts/{_ARTIFACT_ID}/presentation")

    assert response.status_code == 200
    body = response.json()
    assert body["references"] == [{"id": "1", "title": "report.pdf"}]
    assert body["sources"][0]["chunks"][0]["chunk_idx"] == 1


async def test_full_artifact_read_streams_without_a_range(
    client: AsyncClient, run_application: _RunApplication
) -> None:
    _publish_test_artifact(run_application)

    response = await client.get(f"/answer/{_RUN_ID}/artifacts/{_ARTIFACT_ID}")

    assert response.status_code == 200
    assert response.content == b"0123456789"
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-disposition"].startswith("attachment")
    assert "content-range" not in response.headers


async def test_svg_artifact_is_inline_only_under_an_inert_document_policy(
    client: AsyncClient, run_application: _RunApplication
) -> None:
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
    _publish_test_artifact(
        run_application,
        media_type="image/svg+xml",
        presentation="image",
        filename="chart.svg",
        content=svg,
    )

    response = await client.get(f"/answer/{_RUN_ID}/artifacts/{_ARTIFACT_ID}")

    assert response.status_code == 200
    assert response.content == svg
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.headers["content-disposition"].startswith("inline")
    assert response.headers["content-security-policy"] == (
        "sandbox; default-src 'none'; img-src data:"
    )
    assert response.headers["x-content-type-options"] == "nosniff"


async def test_open_ended_range_returns_206_with_a_content_range(
    client: AsyncClient, run_application: _RunApplication
) -> None:
    _publish_test_artifact(run_application)

    response = await client.get(
        f"/answer/{_RUN_ID}/artifacts/{_ARTIFACT_ID}", headers={"Range": "bytes=5-"}
    )

    assert response.status_code == 206
    assert response.content == b"56789"
    assert response.headers["content-range"] == "bytes 5-9/10"


async def test_suffix_range_returns_the_last_bytes(
    client: AsyncClient, run_application: _RunApplication
) -> None:
    _publish_test_artifact(run_application)

    response = await client.get(
        f"/answer/{_RUN_ID}/artifacts/{_ARTIFACT_ID}", headers={"Range": "bytes=-4"}
    )

    assert response.status_code == 206
    assert response.content == b"6789"
    assert response.headers["content-range"] == "bytes 6-9/10"


async def test_closed_range_returns_exactly_the_window(
    client: AsyncClient, run_application: _RunApplication
) -> None:
    _publish_test_artifact(run_application)

    response = await client.get(
        f"/answer/{_RUN_ID}/artifacts/{_ARTIFACT_ID}", headers={"Range": "bytes=2-5"}
    )

    assert response.status_code == 206
    assert response.content == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"


async def test_unsatisfiable_range_is_416_with_the_total(
    client: AsyncClient, run_application: _RunApplication
) -> None:
    _publish_test_artifact(run_application)

    response = await client.get(
        f"/answer/{_RUN_ID}/artifacts/{_ARTIFACT_ID}", headers={"Range": "bytes=20-"}
    )

    assert response.status_code == 416
    assert response.headers["content-range"] == "bytes */10"


async def test_unknown_artifact_is_404(
    client: AsyncClient, run_application: _RunApplication
) -> None:
    run_application.artifact_bytes = None

    response = await client.get(f"/answer/{_RUN_ID}/artifacts/missing")

    assert response.status_code == 404


class TestAgentControls:
    async def test_rest_projects_shared_agent_controls(
        self, client: AsyncClient, run_application: _RunApplication
    ) -> None:
        steer = await client.post(f"/answer/{_RUN_ID}/steer", json={"content": "focus"})
        follow = await client.post(f"/answer/{_RUN_ID}/follow-up", json={"content": "next"})
        fork = await client.post(f"/answer/{_RUN_ID}/fork", json={"content": "branch"})
        transcript = await client.get(f"/answer/{_RUN_ID}/transcript")
        children = await client.get(f"/answer/{_RUN_ID}/children")

        assert steer.status_code == 202
        assert steer.json()["control_sequence"] == 1
        assert follow.status_code == 202
        assert fork.status_code == 202
        assert transcript.json()["messages"] == [{"role": "user", "content": "hi"}]
        assert children.json()["children"][0]["child_session_id"] == "child-1"
        assert run_application.controls == ["focus", "follow:next", "fork:branch"]

    async def test_continuation_rechecks_current_workspace_authorization(
        self,
        client: AsyncClient,
        run_application: _RunApplication,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.http.rest.routes import answer_runs

        run_application.record = _record(
            status="succeeded",
            accepted_input={"query": "q", "workspaces": ["revoked"]},
        )
        resolver = AsyncMock(side_effect=HTTPException(status_code=403, detail="forbidden"))
        monkeypatch.setattr(answer_runs, "resolve_authorized_query_workspaces", resolver)

        response = await client.post(
            f"/answer/{_RUN_ID}/follow-up",
            json={"content": "next"},
        )

        assert response.status_code == 403
        resolver.assert_awaited_once()
        assert run_application.continuations == []


@pytest.mark.parametrize(
    ("method", "path", "body", "status"),
    [
        ("POST", f"/answer/{_RUN_ID}/steer", {"content": "focus"}, 409),
        ("POST", f"/answer/{_RUN_ID}/follow-up", {"content": "next"}, 409),
        ("POST", f"/answer/{_RUN_ID}/fork", {"content": "branch"}, 409),
        ("GET", f"/answer/{_RUN_ID}/transcript", None, 404),
        ("GET", f"/answer/{_RUN_ID}/children", None, 404),
        ("GET", f"/answer/{_RUN_ID}/artifacts", None, 404),
    ],
)
async def test_retrieval_run_id_is_unknown_to_answer_only_rest_routes(
    client: AsyncClient,
    run_application: _RunApplication,
    method: str,
    path: str,
    body: dict[str, str] | None,
    status: int,
) -> None:
    run_application.record = _record(
        run_kind="retrieval",
        status="succeeded",
        accepted_input={"query": "retrieval", "workspaces": ["default"]},
        result={"answer": "fabricated"},
    )

    response = await client.request(method, path, json=body)

    assert response.status_code == status
    assert run_application.continuations == []


@pytest.mark.parametrize("suffix", ["", "/presentation"])
async def test_retrieval_run_id_cannot_read_published_artifacts_over_rest(
    client: AsyncClient,
    run_application: _RunApplication,
    suffix: str,
) -> None:
    _publish_test_artifact(run_application)
    assert run_application.record is not None
    result = run_application.record.result
    run_application.record = _record(run_kind="retrieval", status="succeeded", result=result)

    response = await client.get(f"/answer/{_RUN_ID}/artifacts/{_ARTIFACT_ID}{suffix}")

    assert response.status_code == 404
