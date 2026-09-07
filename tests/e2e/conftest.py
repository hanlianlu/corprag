# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for E2E Playwright tests.

Starts a real FastAPI server on a random port with a mocked Application so the
browser can exercise the full HTML/JS/CSS pipeline without
needing PostgreSQL, LLM, or embedding backends.

Usage (opt-in, requires Playwright)::

    pytest tests/e2e/ -m e2e
"""

import base64
import hashlib
import os
import re
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Generator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    sync_playwright,
)
from playwright.sync_api import (
    Error as PlaywrightError,
)

from dlightrag.adapters.http.server import create_app
from dlightrag.application.access import WorkspaceRecord
from dlightrag.application.answer_runs import AnswerInputArtifact
from dlightrag.application.answer_runs.capabilities import AnswerCapabilities
from dlightrag.application.answer_runs.capability import AnswerImageCapability
from dlightrag.application.answer_runs.execution import (
    AnswerRunInput,
    AttachmentReference,
    PinnedModelProfile,
)
from dlightrag.application.config import DlightragConfig, set_config
from dlightrag.application.corpus_admin import (
    FilePanelCursorCodec,
    WorkspaceCatalogCursor,
    WorkspaceCatalogCursorCodec,
    WorkspaceCatalogPage,
    WorkspaceCatalogPageRequest,
)
from dlightrag.application.runs import RunEvent, RunStatus, RunView
from dlightrag.application.web_conversations import (
    ConversationCursor,
    ConversationCursorCodec,
    ConversationHead,
    ConversationHistoryCursor,
    ConversationHistoryCursorCodec,
    ConversationHistoryPage,
    ConversationHistoryPageRequest,
    ConversationPage,
    ConversationPageRequest,
    LinkedTurn,
    WebAnswerSubmission,
)
from dlightrag.application.web_conversations import (
    ConversationSummary as ApplicationConversationSummary,
)
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import MODEL_CATALOG_REVISION
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.media import MODEL_IMAGE_MAX_PIXELS
from dlightrag.engine.ai.settings import (
    MODEL_ROLE_NAMES,
    EmbeddingSettings,
    ModelRoleSettings,
    ModelSettings,
)
from tests.config_helpers import mutate_config

MOCK_WORKSPACES = [
    {"workspace": "default", "display_name": "Default", "embedding_model": "voyage-multimodal-3.5"},
    {
        "workspace": "research",
        "display_name": "Research",
        "embedding_model": "voyage-multimodal-3.5",
    },
]

ANSWER_TEXT = "DlightRAG is a multimodal RAG system."
_ANSWER_TOKENS = ("DlightRAG is a ", "multimodal RAG system.")
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)
_E2E_PROFILE = ModelProfile(
    context_window_tokens=1_000_000,
    max_output_tokens=128_000,
    supports_images=True,
)


def _run_request(
    *,
    query: str,
    workspaces: Any,
    attachments: Any,
    idempotency_fingerprint: str,
) -> dict[str, Any]:
    return AnswerRunInput(
        query=query,
        workspaces=tuple(workspaces),
        attachments=tuple(
            AttachmentReference(
                digest=attachment.content_sha256,
                filename=attachment.filename,
                mime_type=attachment.mime_type,
                ordinal=attachment.ordinal,
                byte_size=attachment.byte_size,
            )
            for attachment in attachments
        ),
        pinned_models=tuple(
            PinnedModelProfile(
                role=role,
                fingerprint=ModelFingerprint("openai", f"e2e-{role}", None),
                profile=_E2E_PROFILE,
            )
            for role in MODEL_ROLE_NAMES
        ),
        context_policy_revision=CONTEXT_POLICY_REVISION,
        model_catalog_revision=MODEL_CATALOG_REVISION,
        idempotency_fingerprint=idempotency_fingerprint,
    ).as_request()


def _run_record(
    run_id: str,
    request: dict[str, Any],
    *,
    status: RunStatus,
    result: dict[str, Any] | None = None,
    cancel_requested: bool = False,
) -> RunView:
    now = datetime.now(UTC)
    terminal = status in ("succeeded", "failed", "cancelled")
    return RunView(
        run_id=run_id,
        run_kind="answer",
        lane="query",
        submitted_by="e2e",
        access_scope_kind="owner",
        access_scope_id="e2e",
        status=status,
        phase=None,
        durable_progress_version=0,
        next_event_sequence=1,
        events_trimmed_at=None,
        cancel_requested=cancel_requested,
        result=result,
        error_kind=None,
        error_message=None,
        created_at=now,
        started_at=None,
        finished_at=now if terminal else None,
        request=request,
    )


class E2EConversationService:
    """Resettable in-memory Web conversation service for browser-only tests.

    Mirrors the durable contract: one submission becomes one run plus its linked
    turn, and every read projects that turn from the run's recorded state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cursor_codec = ConversationCursorCodec(b"dlightrag-e2e-conversation-cursor")
        self.history_cursor_codec = ConversationHistoryCursorCodec(b"dlightrag-e2e-history-cursor")
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._conversations: dict[str, dict[str, Any]] = {}
            self._runs: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        return None

    @staticmethod
    def _summary(value: dict[str, Any]) -> ApplicationConversationSummary:
        return ApplicationConversationSummary(
            conversation_id=value["conversation_id"],
            title=value["title"],
            created_at=value["created_at"],
            updated_at=value["updated_at"],
        )

    async def create(self, _user: Any) -> ApplicationConversationSummary:
        now = datetime.now(UTC)
        value = {
            "conversation_id": str(uuid4()),
            "title": None,
            "created_at": now,
            "updated_at": now,
            "turns": [],
        }
        with self._lock:
            self._conversations[value["conversation_id"]] = value
        return self._summary(value)

    async def list(
        self,
        _user: Any,
        *,
        page: ConversationPageRequest,
    ) -> ConversationPage:
        """Return an exact in-memory equivalent of the PostgreSQL keyset page."""
        with self._lock:
            values = sorted(
                self._conversations.values(),
                key=lambda value: (value["updated_at"], value["conversation_id"]),
                reverse=True,
            )
            if page.cursor is not None:
                cursor_key = (
                    page.cursor.updated_at,
                    str(page.cursor.conversation_id),
                )
                values = [
                    value
                    for value in values
                    if (value["updated_at"], value["conversation_id"]) < cursor_key
                ]
            fetched = values[: page.limit + 1]
            returned = fetched[: page.limit]
            next_cursor = None
            if len(fetched) > page.limit:
                last = returned[-1]
                next_cursor = ConversationCursor(
                    updated_at=last["updated_at"],
                    conversation_id=UUID(last["conversation_id"]),
                )
            return ConversationPage(
                items=tuple(self._summary(value) for value in returned),
                next_cursor=next_cursor,
                fetched_rows=len(fetched),
            )

    def seed_conversations(self, *, count: int) -> list[str]:
        """Seed deterministic newest-first rows for HTTP pagination coverage."""
        newest = datetime(2026, 8, 28, tzinfo=UTC)
        values: list[dict[str, Any]] = []
        for index in range(count):
            conversation_id = str(UUID(int=index + 1))
            updated_at = newest - timedelta(microseconds=index)
            values.append(
                {
                    "conversation_id": conversation_id,
                    "title": f"Seeded conversation {index + 1}",
                    "created_at": updated_at,
                    "updated_at": updated_at,
                    "turns": [],
                }
            )
        with self._lock:
            self._conversations.update((value["conversation_id"], value) for value in values)
        return [value["conversation_id"] for value in values]

    async def history(
        self,
        _user: Any,
        conversation_id: str,
        *,
        page: ConversationHistoryPageRequest,
    ) -> ConversationHistoryPage | None:
        with self._lock:
            value = self._conversations.get(conversation_id)
            if value is None:
                return None
            turns = sorted(value["turns"], key=lambda turn: turn.turn_number, reverse=True)
            if page.cursor is not None:
                turns = [
                    turn for turn in turns if turn.turn_number < page.cursor.before_turn_number
                ]
            fetched = turns[: page.limit + 1]
            returned = fetched[: page.limit]
            next_cursor = None
            if len(fetched) > page.limit:
                next_cursor = ConversationHistoryCursor(
                    conversation_id=UUID(conversation_id),
                    before_turn_number=returned[-1].turn_number,
                )
            return ConversationHistoryPage(
                conversation=ConversationHead(
                    principal_id="e2e",
                    conversation_id=conversation_id,
                    content_revision=len(value["turns"]),
                    title=value["title"],
                    created_at=value["created_at"],
                    updated_at=value["updated_at"],
                    agent_session_id=conversation_id,
                    agent_lane_id="main",
                ),
                turns=tuple(reversed(returned)),
                next_cursor=next_cursor,
                fetched_rows=len(fetched),
            )

    async def rename(
        self, _user: Any, conversation_id: str, title: str
    ) -> ApplicationConversationSummary | None:
        with self._lock:
            value = self._conversations.get(conversation_id)
            if value is None:
                return None
            value["title"] = title
            value["updated_at"] = datetime.now(UTC)
            return self._summary(value)

    async def delete(self, _user: Any, conversation_id: str) -> bool:
        with self._lock:
            value = self._conversations.pop(conversation_id, None)
            for turn in (value or {}).get("turns", []):
                self._runs.pop(turn.run.run_id, None)
            return value is not None

    async def delete_all(self, _user: Any) -> int:
        with self._lock:
            count = len(self._conversations)
            self._conversations.clear()
            self._runs.clear()
            return count

    def _project_submission(self, value: dict[str, Any], turn: LinkedTurn) -> WebAnswerSubmission:
        return WebAnswerSubmission(
            run=turn.run,
            turn_id=turn.turn_id,
            turn_number=turn.turn_number,
            conversation=self._summary(value),
            submission_id=turn.submission_id,
            created_at=turn.created_at,
        )

    def _submission_for(self, submission_id: str) -> WebAnswerSubmission | None:
        for candidate in self._conversations.values():
            for turn in candidate["turns"]:
                if turn.submission_id == submission_id:
                    return self._project_submission(candidate, turn)
        return None

    async def submission(self, _user: Any, submission_id: str) -> WebAnswerSubmission | None:
        """Recover one accepted E2E submission through the production seam."""
        with self._lock:
            return self._submission_for(submission_id)

    async def start_answer(
        self,
        _user: Any,
        *,
        conversation_id: str | None,
        submission_id: str,
        query: str,
        workspaces: Any,
        attachments: Any = (),
        mode: str | None = None,
        requested_skill: str | None = None,
    ) -> WebAnswerSubmission | None:
        with self._lock:
            if conversation_id is None:
                replay = self._submission_for(submission_id)
                if replay is not None:
                    return replay
                now = datetime.now(UTC)
                conversation_id = str(uuid4())
                self._conversations[conversation_id] = {
                    "conversation_id": conversation_id,
                    "title": None,
                    "created_at": now,
                    "updated_at": now,
                    "turns": [],
                }
            value = self._conversations.get(conversation_id)
            if value is None:
                return None
            for turn in value["turns"]:
                if turn.submission_id == submission_id:
                    return self._project_submission(value, turn)
            run_id = str(uuid4())
            requested_mode = mode or "auto"
            request = _run_request(
                query=query,
                workspaces=workspaces,
                attachments=attachments,
                idempotency_fingerprint=submission_id,
            )
            turn = LinkedTurn(
                turn_id=str(uuid4()),
                turn_number=len(value["turns"]) + 1,
                submission_id=submission_id,
                created_at=datetime.now(UTC),
                run=_run_record(run_id, request, status="queued"),
            )
            value["turns"].append(turn)
            value["title"] = value["title"] or " ".join(query.split())[:120]
            value["updated_at"] = datetime.now(UTC)
            self._runs[run_id] = {
                "conversation_id": conversation_id,
                "requested_mode": requested_mode,
                "bytes": {
                    attachment.ordinal: (attachment.attachment_bytes, attachment.mime_type)
                    for attachment in attachments
                },
            }
            return self._project_submission(value, turn)

    async def turn_for_run(self, _user: Any, run_id: str) -> LinkedTurn | None:
        with self._lock:
            return self._find_turn(run_id)

    def _find_turn(self, run_id: str) -> LinkedTurn | None:
        entry = self._runs.get(run_id)
        if entry is None:
            return None
        value = self._conversations.get(entry["conversation_id"])
        if value is None:
            return None
        return next((turn for turn in value["turns"] if turn.run.run_id == run_id), None)

    def finish_run(self, run_id: str, *, status: RunStatus = "succeeded") -> None:
        """Record the terminal state a worker would have committed."""
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is None:
                return
            value = self._conversations[entry["conversation_id"]]
            for index, turn in enumerate(value["turns"]):
                if turn.run.run_id != run_id:
                    continue
                result = (
                    {
                        "answer": ANSWER_TEXT,
                        "contexts": {"chunks": []},
                        "sources": [],
                        "evidence_images": [],
                        "artifacts": [],
                        "artifact_outcome": {"status": "complete", "issues": []},
                        "trace": {},
                        "image_descriptions": [],
                    }
                    if status == "succeeded"
                    else None
                )
                value["turns"][index] = LinkedTurn(
                    turn_id=turn.turn_id,
                    turn_number=turn.turn_number,
                    submission_id=turn.submission_id,
                    created_at=turn.created_at,
                    run=_run_record(run_id, dict(turn.run.request), status=status, result=result),
                )

    async def attachment(self, _user: Any, run_id: str, ordinal: int) -> Any:
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is None or ordinal not in entry["bytes"]:
                return None
            turn = self._find_turn(run_id)
            if turn is None:
                return None
            content, _mime = entry["bytes"][ordinal]
        reference = next(
            item for item in _request_attachments(turn.run.request) if item["ordinal"] == ordinal
        )
        return AnswerInputArtifact(
            reference_kind="current_attachment",
            ordinal=ordinal,
            filename=reference["filename"],
            mime_type=reference["mime_type"],
            digest=reference["digest"],
            content=content,
        )

    async def thumbnail(self, user: Any, run_id: str, ordinal: int) -> tuple[bytes, str] | None:
        stored = await self.attachment(user, run_id, ordinal)
        if stored is None or not stored.mime_type.lower().startswith("image/"):
            return None
        return stored.content, stored.mime_type

    def seed_image_history(self, *, turn_count: int) -> str:
        """Create one long image conversation for browser loading probes."""
        conversation_id = str(uuid4())
        now = datetime.now(UTC)
        value: dict[str, Any] = {
            "conversation_id": conversation_id,
            "title": "Image history",
            "created_at": now,
            "updated_at": now,
            "turns": [],
        }
        for index in range(1, turn_count + 1):
            run_id = str(uuid4())
            submission_id = str(uuid4())
            request = _run_request(
                query=f"Image question {index}",
                workspaces=["default"],
                attachments=(
                    SimpleNamespace(
                        content_sha256=f"{index:064d}",
                        filename="chart.png",
                        mime_type="image/png",
                        ordinal=1,
                        byte_size=len(_PNG),
                    ),
                ),
                idempotency_fingerprint=submission_id,
            )
            value["turns"].append(
                LinkedTurn(
                    turn_id=str(uuid4()),
                    turn_number=index,
                    submission_id=submission_id,
                    created_at=now,
                    run=_run_record(
                        run_id,
                        request,
                        status="succeeded",
                        result={
                            "answer": f"Image answer {index}",
                            "contexts": {"chunks": []},
                            "sources": [],
                            "evidence_images": [],
                            "artifacts": [],
                            "artifact_outcome": {"status": "complete", "issues": []},
                            "trace": {},
                            "image_descriptions": [],
                        },
                    ),
                )
            )
            self._runs[run_id] = {
                "conversation_id": conversation_id,
                "bytes": {1: (_PNG, "image/png")},
            }
        with self._lock:
            self._conversations[conversation_id] = value
        return conversation_id


def _request_attachments(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(request.get("attachments") or [])


def _event(sequence: int, event_type: str, payload: dict[str, Any]) -> RunEvent:
    return RunEvent(
        sequence=sequence,
        event_type=event_type,  # type: ignore[arg-type]
        payload=payload,
        created_at=datetime.now(UTC),
    )


def _free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def e2e_conversation_service() -> E2EConversationService:
    return E2EConversationService()


@pytest.fixture(scope="session")
def e2e_base_url(
    e2e_conversation_service: E2EConversationService,
) -> Generator[str, Any]:
    """Start one real FastAPI server for the E2E session on a random port."""
    working_directory = tempfile.TemporaryDirectory(prefix="dlightrag-e2e-")
    application_config = DlightragConfig(
        deployment={"working_dir": working_directory.name},
        models={
            "chat": ModelRoleSettings(
                default=ModelSettings(
                    model="google/gemini-3.7-flash",
                    base_url="https://openrouter.ai/api/v1",
                    api_key="test",
                )
            ),
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="test",
                startup_probe=False,
            ),
        },
    )
    mutate_config(application_config, "answer.generation.image_max_pixels", MODEL_IMAGE_MAX_PIXELS)
    mutate_config(application_config, "answer.generation.max_attachments", 6)
    mutate_config(application_config, "answer.generation.max_attachment_bytes", 100 * 1024 * 1024)
    mutate_config(
        application_config,
        "answer.generation.max_total_attachment_bytes",
        128 * 1024 * 1024,
    )
    set_config(application_config)
    application_double = AsyncMock()
    application_double.config = application_config

    def _events(*, owner_id: str, run_id: str, after_sequence: int = 0) -> Any:
        """Replay this run's durable events from the caller's cursor.

        The browser is the only consumer, so the whole log is short and
        deterministic: two phases, the answer tokens, then the terminal event.
        Resuming after a sequence never repeats an earlier frame.
        """
        del owner_id

        async def _iterate() -> Any:
            log: list[RunEvent] = [
                _event(1, "progress", {"phase": "planning"}),
                _event(2, "progress", {"phase": "generating"}),
                *(
                    _event(3 + index, "token", {"text": token})
                    for index, token in enumerate(_ANSWER_TOKENS)
                ),
            ]
            e2e_conversation_service.finish_run(run_id)
            turn = await e2e_conversation_service.turn_for_run(None, run_id)
            result = dict(turn.run.result or {}) if turn is not None else {}
            log.append(
                _event(
                    3 + len(_ANSWER_TOKENS),
                    "done",
                    {"status": "succeeded", "result": result},
                )
            )
            for event in log:
                if event.sequence > after_sequence:
                    yield event

        return _iterate()

    answer_image_capability = AnswerImageCapability(
        status="supported",
        configured_ceiling=3,
        effective_max_images=3,
        provider="test",
        base_url=None,
        model="test-model",
        failure_kind=None,
    )
    application_double.runs = SimpleNamespace(
        cancel=AsyncMock(),
        get_global=AsyncMock(return_value=None),
        subscribe=MagicMock(side_effect=_events),
    )
    application_double.answers = SimpleNamespace(
        capabilities=AsyncMock(
            return_value=AnswerCapabilities(
                answer=answer_image_capability,
                vlm_status="unknown",
            )
        ),
    )
    workspace_records = [dict(record) for record in MOCK_WORKSPACES]

    async def _list_workspaces() -> list[str]:
        return [str(record["workspace"]) for record in workspace_records]

    async def _list_workspace_records() -> list[dict[str, str]]:
        return [dict(record) for record in workspace_records]

    async def _list_workspace_records_page(
        page: WorkspaceCatalogPageRequest | None = None,
    ) -> WorkspaceCatalogPage:
        requested = page or WorkspaceCatalogPageRequest()
        ordered = sorted(workspace_records, key=lambda record: str(record["workspace"]))
        cursor = requested.cursor
        if cursor is not None:
            ordered = [
                record for record in ordered if str(record["workspace"]) > cursor.after_workspace
            ]
        rows = ordered[: requested.limit]
        next_cursor = None
        if len(ordered) > requested.limit:
            next_cursor = WorkspaceCatalogCursor(after_workspace=str(rows[-1]["workspace"]))

        def _to_record(record: dict[str, str]) -> WorkspaceRecord:
            workspace = str(record["workspace"])
            return {
                "workspace": workspace,
                "display_name": str(record.get("display_name") or workspace),
                "embedding_model": str(record.get("embedding_model") or ""),
                "created_at": None,
                "updated_at": None,
            }

        return WorkspaceCatalogPage(
            items=tuple(_to_record(record) for record in rows),
            next_cursor=next_cursor,
            fetched_rows=min(len(ordered), requested.limit + 1),
        )

    async def _workspace_exists(workspace: str) -> bool:
        return any(str(record["workspace"]) == workspace for record in workspace_records)

    async def _create_workspace(workspace: str, *, display_name: str) -> None:
        workspace_records.append(
            {
                "workspace": workspace,
                "display_name": display_name,
                "embedding_model": "voyage-multimodal-3.5",
            }
        )

    async def _create_reset_run(*, workspace: str, submitted_by: str) -> SimpleNamespace:
        now = datetime.now(UTC)
        run_id = str(uuid4())
        return SimpleNamespace(
            run=RunView(
                run_id=run_id,
                run_kind="corpus_mutation",
                lane="corpus_mutation",
                submitted_by=submitted_by,
                access_scope_kind="workspace",
                access_scope_id=workspace,
                status="succeeded",
                phase="completed",
                durable_progress_version=1,
                next_event_sequence=2,
                events_trimmed_at=None,
                cancel_requested=False,
                result={"action": "reset", "document_count": 0, "documents": []},
                error_kind=None,
                error_message=None,
                created_at=now,
                started_at=now,
                finished_at=now,
                request={"action": "reset", "workspace": workspace},
            )
        )

    application_double.corpus_mutations = SimpleNamespace(
        create_reset=AsyncMock(side_effect=_create_reset_run)
    )
    application_double.corpora.list_workspaces.side_effect = _list_workspaces
    application_double.corpora.alist_workspace_records.side_effect = _list_workspace_records
    application_double.corpora.list_workspace_records_page.side_effect = (
        _list_workspace_records_page
    )
    application_double.corpora.workspace_exists.side_effect = _workspace_exists
    application_double.corpora.file_panel_cursor_codec = FilePanelCursorCodec(b"e2e-files")
    application_double.corpora.workspace_catalog_cursor_codec = WorkspaceCatalogCursorCodec(
        b"e2e-workspaces"
    )
    application_double.corpora.create_workspace.side_effect = _create_workspace
    application_double.corpora.file_panel_snapshot.return_value = {
        "files": [],
        "next_cursor": None,
        "fetched_rows": 0,
    }
    application_double.corpora.failed_file_snapshot.return_value = {
        "failed": [],
        "next_cursor": None,
        "fetched_rows": 0,
    }
    application_double.corpora.get_active_retry_failed_docs.return_value = None
    application_double.corpora.delete_files.return_value = {"deleted_count": 0}
    application_double.web_conversations = e2e_conversation_service

    port = _free_port()
    import uvicorn

    with patch(
        "dlightrag.adapters.http.server.create_application",
        AsyncMock(return_value=application_double),
    ):
        app = create_app(include_web_app=True)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{base_url}/web/", timeout=0.25):
                    break
            except OSError, urllib.error.URLError:
                time.sleep(0.05)
        else:
            server.should_exit = True
            t.join(timeout=3)
            raise RuntimeError("E2E server did not become ready")
        yield base_url
        server.should_exit = True
        t.join(timeout=3)
    working_directory.cleanup()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item) -> Generator[None, Any]:
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)


def _test_failed(item: pytest.Item) -> bool:
    for phase in ("setup", "call"):
        report = getattr(item, f"rep_{phase}", None)
        if report is not None and report.failed:
            return True
    return False


def _failure_artifact_stem(nodeid: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", nodeid).strip("-")[:96] or "e2e"
    digest = hashlib.sha256(nodeid.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


@pytest.fixture(scope="package")
def browser() -> Generator[Browser, Any]:
    """Reuse Chromium within E2E, then release its loop before async suites."""
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        try:
            yield b
        finally:
            b.close()


@pytest.fixture
def e2e_browser_context(
    browser: Browser,
    e2e_base_url: str,
    e2e_conversation_service: E2EConversationService,
    request: pytest.FixtureRequest,
) -> Generator[BrowserContext, Any]:
    """Create a context that retains diagnostics only for failed CI tests."""
    e2e_conversation_service.reset()
    context = browser.new_context(base_url=e2e_base_url)
    artifact_dir_value = os.getenv("DLIGHTRAG_E2E_ARTIFACT_DIR")
    artifact_dir = Path(artifact_dir_value) if artifact_dir_value else None
    tracing_started = False

    if artifact_dir is not None:
        try:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)
            tracing_started = True
        except PlaywrightError:
            pass

    try:
        yield context
    finally:
        failed = _test_failed(request.node)
        stem = _failure_artifact_stem(request.node.nodeid)
        writable_artifact_dir: Path | None = None
        if artifact_dir is not None and failed:
            try:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                writable_artifact_dir = artifact_dir
            except OSError:
                pass

        if writable_artifact_dir is not None:
            for index, candidate in enumerate(context.pages, start=1):
                if candidate.is_closed():
                    continue
                try:
                    candidate.screenshot(
                        path=writable_artifact_dir / f"{stem}-page-{index}.png",
                        full_page=True,
                    )
                except OSError, PlaywrightError:
                    pass

        if tracing_started:
            try:
                if writable_artifact_dir is not None:
                    context.tracing.stop(path=writable_artifact_dir / f"{stem}-trace.zip")
                else:
                    context.tracing.stop()
            except OSError, PlaywrightError:
                pass
        context.close()


@pytest.fixture
def page(e2e_browser_context: BrowserContext) -> Generator[Page, Any]:
    """Fresh page per test, already pointed at the running server."""
    page_obj = e2e_browser_context.new_page()
    page_obj.set_default_timeout(10000)
    yield page_obj
