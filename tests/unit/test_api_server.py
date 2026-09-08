# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for FastAPI REST server endpoints and auth middleware."""

import asyncio
import contextlib
import datetime
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock

import jwt
import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient, Response

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.server import create_app
from dlightrag.application import ApplicationClosedError
from dlightrag.application.access import (
    AuthenticationError,
    UserContext,
    authenticate_bearer_token,
    owner_id_from_principal,
)
from dlightrag.application.access import authentication as authentication_module
from dlightrag.application.answer_runs import (
    AnswerRuntimeUnavailableError,
    ChildRosterCursor,
    ChildRosterCursorCodec,
    ChildRosterPage,
)
from dlightrag.application.config import (
    AccessControlConfig,
    AccessControlRuleConfig,
    DlightragConfig,
    set_config,
)
from dlightrag.application.corpus_admin import (
    FilePanelCursor,
    FilePanelCursorCodec,
    MetadataSearchCursor,
    MetadataSearchCursorCodec,
    MetadataSearchPage,
    MetadataValidationError,
    WorkspaceCatalogCursor,
    WorkspaceCatalogCursorCodec,
    WorkspaceCatalogPage,
)
from dlightrag.application.health import ApplicationHealth
from dlightrag.application.retrieval import (
    CorpusUnavailableError,
    RetrieveProjection,
    RetrieveResponse,
    restore_retrieval_result,
)
from dlightrag.application.retrieval._answer_projection import project_answer_retrieval
from dlightrag.application.runs import (
    IdempotencyKeyConflict,
    RunAdmissionLimitExceededError,
    RunView,
)
from dlightrag.application.settings import authentication_settings
from dlightrag.engine.answer.citations.contracts import SourceReference
from dlightrag.engine.answer.errors import AnswerInputOverflowError
from dlightrag.engine.answer.results import AnswerResult
from dlightrag.engine.rag.retrieval import RetrievalResult
from dlightrag.engine.runtime.records import (
    RunAccessScope,
    RunCreation,
    RunRecord,
)
from tests.config_helpers import clone_config, mutate_config, replace_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ANON = UserContext(user_id="anonymous", auth_mode="none")
app: FastAPI


def _project_stored_retrieval(
    stored: dict[str, Any], projection: RetrieveProjection
) -> RetrieveResponse:
    result = restore_retrieval_result(stored)
    projected = project_answer_retrieval(result, projection=projection)
    return RetrieveResponse(
        contexts=projected.contexts,
        sources=projected.sources,
        trace=result.trace,
        image_descriptions=tuple(result.image_descriptions),
    )


def _finance_source() -> SourceReference:
    return SourceReference(
        id="1",
        title="report.pdf",
        source_uri="s3://bucket/report.pdf",
        workspace="finance",
        document_id="doc-report",
        download_locator="s3://bucket/report.pdf",
    )


def _finance_source_context() -> dict[str, object]:
    return {
        "chunk_id": "c1",
        "reference_id": "1",
        "full_doc_id": "doc-report",
        "file_path": "report.pdf",
        "content": "Evidence",
        "_workspace": "finance",
        "metadata": {
            "source_uri": "s3://bucket/report.pdf",
            "source_download_locator": "s3://bucket/report.pdf",
            "source_file_name": "report.pdf",
        },
    }


def _queued_run_record(
    *,
    run_kind: str = "answer",
    status: str = "queued",
    result: dict[str, Any] | None = None,
    workspaces: tuple[str, ...] = ("default",),
) -> RunRecord:
    now = datetime.datetime(2026, 8, 13, tzinfo=datetime.UTC)
    terminal = status in {"succeeded", "failed", "cancelled"}
    return RunRecord(
        run_id="0199a0a0-0000-7000-8000-0000000000aa",
        run_kind=run_kind,  # type: ignore[arg-type]
        lane="corpus_mutation" if run_kind == "corpus_mutation" else "query",
        submitted_by=owner_id_from_principal(auth_mode="none", user_id="anonymous"),
        access_scope=(
            RunAccessScope(kind="workspace", scope_id="default")
            if run_kind == "corpus_mutation"
            else RunAccessScope(
                kind="owner",
                scope_id=owner_id_from_principal(auth_mode="none", user_id="anonymous"),
            )
        ),
        submission_key=None or "0199a0a0-0000-7000-8000-0000000000aa",
        request_fingerprint="test-fingerprint",
        prepared_input=(
            {"action": "ingest", "workspace": "default"}
            if run_kind == "corpus_mutation"
            else {"query": "hi", "workspaces": ["default"]}
        ),
        status=status,  # type: ignore[arg-type]
        phase=None,
        stop_reason=None,
        cancel_requested_at=None,
        lease_owner=None,
        lease_expires_at=None,
        fencing_epoch=0,
        durable_progress_version=0,
        last_reclaim_progress_version=0,
        reclaims_without_progress=0,
        next_event_sequence=1,
        events_trimmed_at=None,
        result=result,
        error_kind=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        started_at=None,
        finished_at=now if terminal else None,
        accepted_input={"query": "hi", "workspaces": list(workspaces)},
    )


@pytest.fixture
def _api_app(test_config: DlightragConfig) -> Iterator[FastAPI]:
    """Create the API app after test_config has installed the singleton."""
    global app
    app = create_app(include_web_app=False)
    yield app
    app.dependency_overrides.clear()
    if hasattr(app.state, "application"):
        del app.state.application
    if hasattr(app.state, "health"):
        del app.state.health


@pytest.fixture
def mock_config(_api_app: FastAPI, test_config: DlightragConfig) -> Iterator[DlightragConfig]:
    """Override auth dependency to allow all requests (auth_mode=none)."""
    _api_app.dependency_overrides[get_current_user] = lambda: _ANON
    yield test_config
    _api_app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def mock_config_no_auth_override(test_config: DlightragConfig):
    """Provide config WITHOUT overriding auth — real auth logic runs."""
    yield test_config


@pytest.fixture
def mock_service():
    """Create a mock WorkspaceRag."""
    service = AsyncMock()
    service.aingest = AsyncMock(return_value={"status": "success", "processed": 1})
    service.aretrieve = AsyncMock(return_value=RetrievalResult(contexts={"chunks": []}))
    service.aanswer = AsyncMock(
        return_value=AnswerResult(answer="The answer is 42", contexts={"chunks": []})
    )
    service.adelete_files = AsyncMock(return_value=[{"status": "deleted"}])
    return service


@pytest.fixture
def mock_application(_api_app: FastAPI, mock_service, test_config):
    """Create an Application-shaped test double with explicit services."""
    application = AsyncMock()
    application.config = test_config
    corpora = SimpleNamespace()
    application.corpus_mutations = SimpleNamespace(
        create_ingest=AsyncMock(
            return_value=RunCreation(
                run=_queued_run_record(run_kind="corpus_mutation"),
                replayed=False,
            )
        ),
        create_delete=AsyncMock(),
        create_retry=AsyncMock(),
        create_reset=AsyncMock(),
        stage_upload=AsyncMock(),
        discard_staged_run=AsyncMock(),
        create_staged_ingest=AsyncMock(),
        create_staged_batch=AsyncMock(),
    )
    application.retrieval = SimpleNamespace(
        create=AsyncMock(
            return_value=RunCreation(run=_queued_run_record(run_kind="retrieval"), replayed=False)
        ),
        project_stored=MagicMock(side_effect=_project_stored_retrieval),
    )
    corpora.workspace_catalog_cursor_codec = WorkspaceCatalogCursorCodec(b"api-test")
    corpora.list_workspace_records_page = AsyncMock(
        return_value=WorkspaceCatalogPage(
            items=(
                {
                    "workspace": "default",
                    "display_name": "default",
                    "embedding_model": "voyage-multimodal-3.5",
                    "created_at": None,
                    "updated_at": None,
                },
            ),
            next_cursor=None,
            fetched_rows=1,
        )
    )
    run_get = AsyncMock(return_value=_queued_run_record())
    application.runs = SimpleNamespace(
        get=run_get,
        get_global=AsyncMock(
            side_effect=lambda **_kwargs: (
                RunView.from_runtime(run_get.return_value)
                if run_get.return_value is not None
                else None
            )
        ),
        list=AsyncMock(return_value=(_queued_run_record(),)),
        cancel=AsyncMock(),
        subscribe=MagicMock(),
    )
    application.answers = SimpleNamespace(
        create=AsyncMock(return_value=RunCreation(run=_queued_run_record(), replayed=False)),
        list_artifacts=AsyncMock(return_value=()),
        children=AsyncMock(
            return_value=ChildRosterPage(children=(), next_cursor=None, fetched_rows=0)
        ),
        child_roster_cursor_codec=ChildRosterCursorCodec(b"api-server-children"),
    )
    corpora.delete_files = mock_service.adelete_files
    corpora.list_workspaces = AsyncMock(return_value=["default"])
    corpora.alist_workspace_records = AsyncMock(
        return_value=[
            {
                "workspace": "default",
                "display_name": "default",
                "embedding_model": "voyage-multimodal-3.5",
                "created_at": None,
                "updated_at": None,
            }
        ]
    )
    corpora.create_workspace = AsyncMock()
    corpora.reset = AsyncMock(return_value={"workspaces": {"old_ws": {}}, "total_errors": 0})
    corpora.failed_file_snapshot = AsyncMock(
        return_value={"failed": [], "next_cursor": None, "fetched_rows": 0}
    )
    corpora.get_active_retry_failed_docs = AsyncMock(return_value=None)
    corpora.start_retry_failed_docs = AsyncMock(
        return_value={
            "job_id": "retry-1",
            "workspace": "default",
            "source_type": "retry_failed",
            "status": "queued",
            "result": {},
        }
    )
    corpora.retry_failed_docs = AsyncMock(return_value={})
    corpora.prepare_source_download = AsyncMock()
    corpora.get_visual_asset = AsyncMock()
    corpora.get_metadata = AsyncMock(return_value={})
    corpora.update_metadata = AsyncMock()
    corpora.search_metadata = AsyncMock(
        return_value=MetadataSearchPage(document_ids=(), next_cursor=None, fetched_rows=0)
    )
    corpora.file_panel_cursor_codec = FilePanelCursorCodec(b"api-server-test")
    corpora.metadata_search_cursor_codec = MetadataSearchCursorCodec(b"api-server-test")
    corpora.workspace_exists = AsyncMock(return_value=True)
    corpora.file_panel_snapshot = AsyncMock(
        return_value={
            "files": [],
            "next_cursor": None,
            "fetched_rows": 0,
        }
    )
    application.corpora = corpora
    application.get_error_info = lambda: {
        "last_error": None,
        "timestamp": None,
        "retry_after": 30.0,
    }
    from dlightrag.engine.answer.image_capability import AnswerImageCapability

    answer_image_capability = AnswerImageCapability(
        status="supported",
        configured_ceiling=8,
        effective_max_images=8,
        provider="test",
        base_url=None,
        model="test-model",
        failure_kind=None,
    )
    from dlightrag.adapters.postgres.corpus.corpus import PGReadinessProbe
    from dlightrag.engine.answer.image_capability import answer_image_capability_summary

    application.health = ApplicationHealth(
        readiness_probe=PGReadinessProbe(test_config),
    )
    application.health.mark_ready()
    application.health.set_answer_image_capability(
        answer_image_capability_summary(answer_image_capability)
    )
    _api_app.state.health = application.health
    application.close = AsyncMock()
    return application


@pytest.fixture
def _patch_application(_api_app: FastAPI, mock_application):
    """Set the Application-shaped double on app state."""
    _api_app.state.application = mock_application
    yield
    if hasattr(_api_app.state, "application"):
        del _api_app.state.application


@pytest.fixture
async def client(_api_app: FastAPI):
    """Create httpx async client for testing."""
    transport = ASGITransport(app=_api_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestAuthMiddleware:
    """Test pluggable auth (none / simple / jwt)."""

    async def test_no_token_configured_passes(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_application")
    async def test_simple_valid_token_passes(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        mutate_config(cfg, "access.api_token", "secret-token")
        mutate_config(cfg, "access.auth_mode", "simple")
        resp = await client.get(
            "/files",
            headers={"Authorization": "Bearer secret-token"},
        )
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_application")
    async def test_simple_missing_auth_header_401(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        mutate_config(cfg, "access.api_token", "secret-token")
        mutate_config(cfg, "access.auth_mode", "simple")
        resp = await client.get("/files")
        assert resp.status_code == 401


class TestWorkspaceLifecycleAPI:
    """Workspace lifecycle API uses the CorpusAdmin catalog."""

    async def test_routes_use_app_scoped_config_after_singleton_changes(
        self,
        client: AsyncClient,
        _api_app: FastAPI,
        mock_config: DlightragConfig,
        mock_application,
    ) -> None:
        _api_app.state.application = mock_application
        mutate_config(mock_config, "deployment.workspace", "app_ws")
        singleton_config = clone_config(mock_config)
        mutate_config(singleton_config, "deployment.workspace", "singleton_ws")
        set_config(singleton_config)

        resp = await client.get("/files")

        assert resp.status_code == 200
        assert mock_application.corpora.file_panel_snapshot.await_args.args == ("app_ws",)

    async def test_list_workspaces_returns_records(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application

        resp = await client.get("/workspaces")

        assert resp.status_code == 200
        body = resp.json()
        assert body["workspaces"] == ["default"]
        assert body["records"][0]["display_name"] == "default"
        assert body["next_cursor"] is None
        page_call = mock_application.corpora.list_workspace_records_page.await_args
        assert page_call is not None
        page_request = page_call.kwargs["page"]
        assert page_request.limit == 50
        assert page_request.cursor is None

    async def test_list_workspaces_returns_an_opaque_continuation(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        mock_application.corpora.list_workspace_records_page = AsyncMock(
            return_value=WorkspaceCatalogPage(
                items=(
                    {
                        "workspace": "finance",
                        "display_name": "Finance",
                        "embedding_model": "voyage-multimodal-3.5",
                        "created_at": None,
                        "updated_at": None,
                    },
                ),
                next_cursor=WorkspaceCatalogCursor(after_workspace="finance"),
                fetched_rows=2,
            )
        )

        resp = await client.get("/workspaces")

        assert resp.status_code == 200
        body = resp.json()
        assert body["next_cursor"] is not None
        assert "workspace-catalog" not in body["next_cursor"]  # opaque token

        second = await client.get("/workspaces", params={"cursor": body["next_cursor"]})
        assert second.status_code == 200
        assert second.json()["workspaces"] == ["finance"]

    async def test_list_workspaces_rejects_tampered_cursor_before_storage(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        mock_application.corpora.list_workspace_records_page = AsyncMock()

        resp = await client.get("/workspaces", params={"cursor": "AAAA.tampered"})

        assert resp.status_code == 422
        mock_application.corpora.list_workspace_records_page.assert_not_awaited()

    @pytest.mark.parametrize("limit", ["0", "101", "abc"])
    async def test_list_workspaces_rejects_out_of_range_limits_before_storage(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
        limit: str,
    ) -> None:
        app.state.application = mock_application
        mock_application.corpora.list_workspace_records_page = AsyncMock()

        resp = await client.get("/workspaces", params={"limit": limit})

        assert resp.status_code == 422
        mock_application.corpora.list_workspace_records_page.assert_not_awaited()

    async def test_list_workspaces_honors_explicit_limit_and_applies_the_access_gate(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        mock_application.corpora.list_workspace_records_page = AsyncMock(
            return_value=WorkspaceCatalogPage(
                items=(
                    {
                        "workspace": "default",
                        "display_name": "default",
                        "embedding_model": "voyage-multimodal-3.5",
                        "created_at": None,
                        "updated_at": None,
                    },
                    {
                        "workspace": "finance",
                        "display_name": "Finance",
                        "embedding_model": "voyage-multimodal-3.5",
                        "created_at": None,
                        "updated_at": None,
                    },
                ),
                next_cursor=None,
                fetched_rows=2,
            )
        )

        resp = await client.get("/workspaces", params={"limit": "1"})

        assert resp.status_code == 200
        page_call = mock_application.corpora.list_workspace_records_page.await_args
        assert page_call is not None
        assert page_call.kwargs["page"].limit == 1

    async def test_create_workspace_registers_empty_workspace(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        mock_application.corpora.list_workspaces = AsyncMock(return_value=["default"])

        resp = await client.post(
            "/workspaces",
            json={"workspace": "New Workspace", "display_name": "New Workspace"},
        )

        assert resp.status_code == 201
        assert resp.json() == {
            "workspace": "new_workspace",
            "display_name": "New Workspace",
            "created": True,
        }
        mock_application.corpora.create_workspace.assert_awaited_once_with(
            "new_workspace",
            display_name="New Workspace",
        )

    async def test_create_workspace_rejects_duplicate(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        mock_application.corpora.list_workspaces = AsyncMock(return_value=["default"])

        resp = await client.post("/workspaces", json={"workspace": "default"})

        assert resp.status_code == 409
        mock_application.corpora.create_workspace.assert_not_awaited()

    @pytest.mark.usefixtures("_patch_application")
    async def test_simple_wrong_scheme_401(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        mutate_config(cfg, "access.api_token", "secret-token")
        mutate_config(cfg, "access.auth_mode", "simple")
        resp = await client.get(
            "/files",
            headers={"Authorization": "Basic abc123"},
        )
        assert resp.status_code == 401

    @pytest.mark.usefixtures("_patch_application")
    async def test_simple_invalid_token_401(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        mutate_config(cfg, "access.api_token", "secret-token")
        mutate_config(cfg, "access.auth_mode", "simple")
        resp = await client.get(
            "/files",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    @pytest.mark.parametrize(
        "method,path,body",
        [
            (
                "POST",
                "/runs/corpus/ingest",
                {"source_type": "local", "path": "/tmp/f.pdf"},
            ),
            ("POST", "/retrieve", {"query": "hello"}),
            ("POST", "/answer", {"query": "hello"}),
            ("POST", "/runs/corpus/delete", {"filenames": ["f.pdf"]}),
        ],
    )
    @pytest.mark.usefixtures("_patch_application")
    async def test_endpoint_requires_auth(
        self,
        method: str,
        path: str,
        body: dict,
        client: AsyncClient,
        mock_config_no_auth_override: DlightragConfig,
    ) -> None:
        cfg = mock_config_no_auth_override
        mutate_config(cfg, "access.api_token", "secret-token")
        mutate_config(cfg, "access.auth_mode", "simple")
        resp = await client.request(method, path, json=body)
        assert resp.status_code == 401

    @pytest.mark.usefixtures("_patch_application")
    async def test_auth_mode_none_allows_all(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        mutate_config(cfg, "access.auth_mode", "none")
        resp = await client.get("/files")
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_application")
    async def test_token_requires_explicit_simple_auth_mode(
        self, test_config: DlightragConfig
    ) -> None:
        """Setting api_auth_token without auth_mode is a config error."""
        mutate_config(test_config, "access.api_token", "my-token")
        mutate_config(test_config, "access.auth_mode", "none")
        with pytest.raises(ValueError, match="auth_mode='simple'"):
            test_config._validate_auth()


# ---------------------------------------------------------------------------
# TestJWTAuth
# ---------------------------------------------------------------------------

_JWT_VERIFICATION_KEY = "test-jwt-verification-key-for-unit-tests"


class TestJWTAuth:
    """Test JWT authentication strategy."""

    @pytest.mark.usefixtures("_patch_application")
    async def test_jwt_valid_token(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        mutate_config(cfg, "access.auth_mode", "jwt")
        mutate_config(cfg, "access.jwt_verification_key", _JWT_VERIFICATION_KEY)
        mutate_config(cfg, "access.jwt_algorithm", "HS256")

        payload = {
            "sub": "user-42",
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_VERIFICATION_KEY, algorithm="HS256")

        resp = await client.get(
            "/files",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_application")
    async def test_jwt_claims_access_control_denies_unmapped_workspace(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig, mock_application
    ) -> None:
        cfg = mock_config_no_auth_override
        mutate_config(cfg, "access.auth_mode", "jwt")
        mutate_config(cfg, "access.jwt_verification_key", _JWT_VERIFICATION_KEY)
        mutate_config(cfg, "access.jwt_algorithm", "HS256")
        cfg = replace_config(
            cfg,
            "access.control",
            AccessControlConfig(
                mode="jwt_claims",
                rules=[
                    AccessControlRuleConfig(
                        claim="groups",
                        value="finance-rag-readers",
                        workspaces=["finance"],
                        actions=["workspace.query"],
                    )
                ],
            ),
        )
        token = jwt.encode(
            {
                "sub": "user-42",
                "groups": ["legal-rag-readers"],
                "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
            },
            _JWT_VERIFICATION_KEY,
            algorithm="HS256",
        )

        resp = await client.post(
            "/retrieve",
            json={"query": "hello", "workspaces": ["finance"]},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 403
        mock_application.retrieval.create.assert_not_awaited()

    @pytest.mark.usefixtures("_patch_application")
    async def test_jwt_claims_access_control_allows_mapped_workspace(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig, mock_application
    ) -> None:
        cfg = mock_config_no_auth_override
        mutate_config(cfg, "access.auth_mode", "jwt")
        mutate_config(cfg, "access.jwt_verification_key", _JWT_VERIFICATION_KEY)
        mutate_config(cfg, "access.jwt_algorithm", "HS256")
        cfg = replace_config(
            cfg,
            "access.control",
            AccessControlConfig(
                mode="jwt_claims",
                rules=[
                    AccessControlRuleConfig(
                        claim="groups",
                        value="finance-rag-readers",
                        workspaces=["finance"],
                        actions=["workspace.query"],
                    )
                ],
            ),
        )
        token = jwt.encode(
            {
                "sub": "user-42",
                "groups": ["finance-rag-readers"],
                "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
            },
            _JWT_VERIFICATION_KEY,
            algorithm="HS256",
        )

        resp = await client.post(
            "/retrieve",
            json={"query": "hello", "workspaces": ["finance"]},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 202
        mock_application.retrieval.create.assert_awaited_once()

    @pytest.mark.usefixtures("_patch_application")
    @pytest.mark.parametrize(
        ("groups", "expected_status"),
        [
            (["finance-rag-readers"], 202),
            (["legal-rag-readers"], 403),
        ],
    )
    async def test_all_workspaces_is_relative_to_query_authorization(
        self,
        client: AsyncClient,
        _api_app: FastAPI,
        mock_config_no_auth_override: DlightragConfig,
        mock_application,
        groups: list[str],
        expected_status: int,
    ) -> None:
        registered = [f"ws_{index:02d}" for index in range(14)]
        allowed = registered[:10]
        mock_application.corpora.alist_workspace_records.return_value = [
            {"workspace": workspace} for workspace in registered
        ]
        replace_config(
            mock_config_no_auth_override,
            "access.control",
            AccessControlConfig(
                mode="jwt_claims",
                rules=[
                    AccessControlRuleConfig(
                        claim="groups",
                        value="finance-rag-readers",
                        workspaces=allowed,
                        actions=["workspace.query"],
                    )
                ],
            ),
        )
        _api_app.dependency_overrides[get_current_user] = lambda: UserContext(
            user_id="alice",
            auth_mode="jwt",
            claims={"groups": groups},
        )

        response = await client.post(
            "/answer",
            json={"query": "hello", "all_workspaces": True},
        )

        assert response.status_code == expected_status
        if expected_status == 202:
            run_input = mock_application.answers.create.await_args.kwargs["request"]
            assert list(run_input.workspaces) == allowed
        else:
            mock_application.answers.create.assert_not_awaited()

    @pytest.mark.usefixtures("_patch_application")
    async def test_jwt_expired_token(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        mutate_config(cfg, "access.auth_mode", "jwt")
        mutate_config(cfg, "access.jwt_verification_key", _JWT_VERIFICATION_KEY)
        mutate_config(cfg, "access.jwt_algorithm", "HS256")

        payload = {
            "sub": "user-42",
            "exp": datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_VERIFICATION_KEY, algorithm="HS256")

        resp = await client.get(
            "/files",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# TestVerifyBearerToken
# ---------------------------------------------------------------------------


class TestAuthenticateBearerToken:
    """Behavioral tests for the transport-neutral Access authentication seam."""

    @staticmethod
    def authenticate(
        token: str,
        config: DlightragConfig,
        *,
        default_user_id: str = "anonymous",
    ) -> UserContext:
        return authenticate_bearer_token(
            token,
            authentication_settings(config),
            default_user_id=default_user_id,
        )

    def test_simple_valid_token(self, test_config: DlightragConfig) -> None:
        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")
        ctx = self.authenticate("secret-token", test_config)
        assert ctx.user_id == "anonymous"
        assert ctx.auth_mode == "simple"

    def test_simple_invalid_token_raises_403(self, test_config: DlightragConfig) -> None:
        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")
        with pytest.raises(AuthenticationError, match="Invalid token"):
            self.authenticate("wrong-token", test_config)

    def test_simple_empty_token_raises_403(self, test_config: DlightragConfig) -> None:
        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")
        with pytest.raises(AuthenticationError, match="Invalid token"):
            self.authenticate("", test_config)

    def test_simple_default_user_id(self, test_config: DlightragConfig) -> None:
        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")
        ctx = self.authenticate("secret-token", test_config, default_user_id="user-99")
        assert ctx.user_id == "user-99"

    def test_jwt_valid_token(self, test_config: DlightragConfig) -> None:
        mutate_config(test_config, "access.auth_mode", "jwt")
        mutate_config(test_config, "access.jwt_verification_key", _JWT_VERIFICATION_KEY)
        mutate_config(test_config, "access.jwt_algorithm", "HS256")

        payload = {
            "sub": "user-42",
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_VERIFICATION_KEY, algorithm="HS256")
        ctx = self.authenticate(token, test_config)
        assert ctx.user_id == "user-42"
        assert ctx.auth_mode == "jwt"

    def test_jwt_jwks_url_validates_issuer_and_audience(
        self,
        test_config: DlightragConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mutate_config(test_config, "access.auth_mode", "jwt")
        mutate_config(
            test_config, "access.jwt_jwks_url", "https://login.example.com/discovery/keys"
        )
        mutate_config(test_config, "access.jwt_issuer", "https://login.example.com/tenant/v2.0")
        mutate_config(test_config, "access.jwt_audience", "api://dlightrag")
        mutate_config(test_config, "access.jwt_algorithm", "HS256")

        payload = {
            "sub": "user-42",
            "iss": test_config.access.jwt_issuer,
            "aud": test_config.access.jwt_audience,
            "groups": ["finance-rag-readers"],
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        jwks_secret = "jwks-secret-for-unit-tests-32-bytes"
        token = jwt.encode(payload, jwks_secret, algorithm="HS256", headers={"kid": "key-1"})

        class FakeJwksClient:
            def get_signing_key_from_jwt(self, raw_token: str):
                assert raw_token == token
                return SimpleNamespace(key=jwks_secret)

        monkeypatch.setattr(authentication_module, "_jwks_client", lambda _url: FakeJwksClient())

        ctx = self.authenticate(token, test_config)

        assert ctx.user_id == "user-42"
        assert ctx.claims["groups"] == ["finance-rag-readers"]

    def test_jwt_jwks_url_rejects_wrong_audience(
        self,
        test_config: DlightragConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mutate_config(test_config, "access.auth_mode", "jwt")
        mutate_config(
            test_config, "access.jwt_jwks_url", "https://login.example.com/discovery/keys"
        )
        mutate_config(test_config, "access.jwt_issuer", "https://login.example.com/tenant/v2.0")
        mutate_config(test_config, "access.jwt_audience", "api://dlightrag")
        mutate_config(test_config, "access.jwt_algorithm", "HS256")

        payload = {
            "sub": "user-42",
            "iss": test_config.access.jwt_issuer,
            "aud": "api://other",
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        jwks_secret = "jwks-secret-for-unit-tests-32-bytes"
        token = jwt.encode(payload, jwks_secret, algorithm="HS256", headers={"kid": "key-1"})

        class FakeJwksClient:
            def get_signing_key_from_jwt(self, raw_token: str):
                assert raw_token == token
                return SimpleNamespace(key=jwks_secret)

        monkeypatch.setattr(authentication_module, "_jwks_client", lambda _url: FakeJwksClient())

        with pytest.raises(AuthenticationError, match="Invalid token"):
            self.authenticate(token, test_config)

    def test_jwt_jwks_url_accepts_any_of_multiple_audiences(
        self,
        test_config: DlightragConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mutate_config(test_config, "access.auth_mode", "jwt")
        mutate_config(
            test_config, "access.jwt_jwks_url", "https://login.example.com/discovery/keys"
        )
        mutate_config(test_config, "access.jwt_issuer", "https://login.example.com/tenant/v2.0")
        mutate_config(test_config, "access.jwt_audience", ["api://dlightrag", "proxy-client-id"])
        mutate_config(test_config, "access.jwt_algorithm", "HS256")

        payload = {
            "sub": "user-42",
            "iss": test_config.access.jwt_issuer,
            "aud": "proxy-client-id",
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        jwks_secret = "jwks-secret-for-unit-tests-32-bytes"
        token = jwt.encode(payload, jwks_secret, algorithm="HS256", headers={"kid": "key-1"})

        class FakeJwksClient:
            def get_signing_key_from_jwt(self, raw_token: str):
                assert raw_token == token
                return SimpleNamespace(key=jwks_secret)

        monkeypatch.setattr(authentication_module, "_jwks_client", lambda _url: FakeJwksClient())

        ctx = self.authenticate(token, test_config)

        assert ctx.user_id == "user-42"

    def test_jwt_expired_token_raises_401(self, test_config: DlightragConfig) -> None:
        mutate_config(test_config, "access.auth_mode", "jwt")
        mutate_config(test_config, "access.jwt_verification_key", _JWT_VERIFICATION_KEY)
        mutate_config(test_config, "access.jwt_algorithm", "HS256")

        payload = {
            "sub": "user-42",
            "exp": datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_VERIFICATION_KEY, algorithm="HS256")
        with pytest.raises(AuthenticationError, match="Token expired"):
            self.authenticate(token, test_config)

    def test_jwt_missing_sub_claim_raises_401(self, test_config: DlightragConfig) -> None:
        mutate_config(test_config, "access.auth_mode", "jwt")
        mutate_config(test_config, "access.jwt_verification_key", _JWT_VERIFICATION_KEY)
        mutate_config(test_config, "access.jwt_algorithm", "HS256")

        payload = {
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_VERIFICATION_KEY, algorithm="HS256")
        with pytest.raises(AuthenticationError, match="missing 'sub' claim"):
            self.authenticate(token, test_config)

    def test_jwt_wrong_verification_key_raises_401(self, test_config: DlightragConfig) -> None:
        mutate_config(test_config, "access.auth_mode", "jwt")
        mutate_config(test_config, "access.jwt_verification_key", _JWT_VERIFICATION_KEY)
        mutate_config(test_config, "access.jwt_algorithm", "HS256")

        payload = {"sub": "user-42"}
        token = jwt.encode(
            payload,
            "wrong-secret-different-key-for-unit-tests",
            algorithm="HS256",
        )
        with pytest.raises(AuthenticationError, match="Invalid token"):
            self.authenticate(token, test_config)


# ---------------------------------------------------------------------------
# TestIngestEndpoint
# ---------------------------------------------------------------------------


class TestCorpusMutationEndpoints:
    """Run-native Corpus Mutation validation and acceptance."""

    @pytest.mark.parametrize("source_type", ["local", "azure_blob", "s3", "url"])
    @pytest.mark.usefixtures("_patch_application")
    async def test_ingest_source_requires_identity(
        self, client: AsyncClient, source_type: str
    ) -> None:
        response = await client.post(
            "/runs/corpus/ingest",
            headers={"Idempotency-Key": "missing-source"},
            json={"source_type": source_type},
        )
        assert response.status_code == 422

    async def test_ingest_accepts_common_run_descriptor(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        response = await client.post(
            "/runs/corpus/ingest",
            headers={"Idempotency-Key": "ingest-file-1"},
            json={"source_type": "local", "path": "file.pdf"},
        )
        assert response.status_code == 202
        assert response.json()["run_kind"] == "corpus_mutation"
        spec = mock_application.corpus_mutations.create_ingest.await_args.kwargs["spec"]
        assert spec.path == str(mock_config.input_dir_path / "default" / "file.pdf")

    async def test_rest_upload_discards_unaccepted_stage_through_mutation_service(
        self, client: AsyncClient, mock_application, tmp_path: Path
    ) -> None:
        app.state.application = mock_application
        source = tmp_path / "report.pdf"
        source.write_bytes(b"content")
        mock_application.corpus_mutations.stage_upload.return_value = SimpleNamespace(
            path=source,
            filename="report.pdf",
            size_bytes=7,
            content_sha256="a" * 64,
        )
        mock_application.corpus_mutations.create_staged_ingest.side_effect = ValueError(
            "acceptance rejected"
        )

        response = await client.post(
            "/runs/corpus/ingest/upload",
            headers={"Idempotency-Key": "upload-1"},
            files={"file": ("report.pdf", b"content", "application/pdf")},
        )

        assert response.status_code == 400
        cleanup = mock_application.corpus_mutations.discard_staged_run.await_args.kwargs
        assert cleanup["workspace"] == "default"
        assert cleanup["run_id"]

    async def test_ingest_requires_idempotency_key(
        self, client: AsyncClient, mock_application
    ) -> None:
        app.state.application = mock_application
        response = await client.post(
            "/runs/corpus/ingest",
            json={"source_type": "local", "path": "file.pdf"},
        )
        assert response.status_code == 400
        mock_application.corpus_mutations.create_ingest.assert_not_awaited()

    @pytest.mark.usefixtures("_patch_application")
    async def test_legacy_ingest_and_job_routes_are_absent(self, client: AsyncClient) -> None:
        assert (await client.post("/ingest", json={})).status_code == 404
        assert (await client.get("/ingest/jobs/old-job")).status_code == 404

    async def test_list_runs_maps_an_empty_workspace_selector_to_bounded_400(
        self, client: AsyncClient, mock_application
    ) -> None:
        app.state.application = mock_application

        response = await client.get("/runs?workspace=")

        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid workspace"
        mock_application.runs.list.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestRetrieveEndpoint
# ---------------------------------------------------------------------------


class TestRetrieveEndpoint:
    """Test durable ``POST /retrieve`` and common result observation."""

    async def test_retrieve_accepts_with_common_descriptor(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application

        response = await client.post(
            "/retrieve",
            json={"query": "What is RAG?"},
            headers={"Idempotency-Key": "retrieve-1"},
        )

        assert response.status_code == 202
        assert response.json() == {
            "run_id": "0199a0a0-0000-7000-8000-0000000000aa",
            "run_kind": "retrieval",
            "lane": "query",
            "status": "queued",
            "status_url": "/runs/0199a0a0-0000-7000-8000-0000000000aa",
            "events_url": "/runs/0199a0a0-0000-7000-8000-0000000000aa/events",
            "cancel_url": "/runs/0199a0a0-0000-7000-8000-0000000000aa",
            "parent_run_id": None,
            "continuation_kind": None,
        }
        call = mock_application.retrieval.create.await_args
        assert call is not None
        assert call.kwargs["idempotency_key"] == "retrieve-1"
        assert call.kwargs["request"].workspaces == ("default",)
        assert call.kwargs["request"].chunk_top_k is None
        assert "projection" not in call.kwargs["request"].__dataclass_fields__

    async def test_retrieve_returns_before_execution_timeout(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application

        response = await client.post("/retrieve", json={"query": "slow"})

        assert response.status_code == 202
        mock_application.retrieval.create.assert_awaited_once()

    async def test_retrieve_closed_service_is_503(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.retrieval.create.side_effect = CorpusUnavailableError(
            "Retrieval service is closed"
        )
        app.state.application = mock_application

        response = await client.post("/retrieve", json={"query": "during shutdown"})

        assert response.status_code == 503
        assert response.json()["error_type"] == "unavailable"

    async def test_retrieve_changed_idempotent_input_is_409(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.retrieval.create.side_effect = IdempotencyKeyConflict("changed")
        app.state.application = mock_application

        response = await client.post(
            "/retrieve",
            json={"query": "changed"},
            headers={"Idempotency-Key": "retrieve-1"},
        )

        assert response.status_code == 409

    async def test_retrieve_projects_terminal_result_with_current_permissions(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        stored = {
            "contexts": {"chunks": [_finance_source_context()]},
            "trace": {"lightrag_mix_chunk_count": 2},
            "image_descriptions": [],
        }
        mock_application.runs.get.return_value = _queued_run_record(
            run_kind="retrieval", status="succeeded", result=stored, workspaces=("finance",)
        )
        app.state.application = mock_application

        response = await client.get("/runs/0199a0a0-0000-7000-8000-0000000000aa")

        assert response.status_code == 200
        result = response.json()["result"]
        assert result["trace"] == {"lightrag_mix_chunk_count": 2}
        assert result["sources"][0]["download_url"] == ("/files/raw/doc-report?workspace=finance")
        assert {"workspace", "download_locator", "path", "url"}.isdisjoint(result["sources"][0])

    async def test_retrieve_omits_download_and_visual_links_without_permissions(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        class QueryOnlyAccess:
            async def check(self, user, action, *, workspace=None):
                return None

            async def filter_workspaces(self, user, action, workspaces):
                if action in {"workspace.download_source", "workspace.read_visual_asset"}:
                    return []
                return list(workspaces)

        chunk = {**_finance_source_context(), "_has_visual_asset": True}
        mock_application.runs.get.return_value = _queued_run_record(
            run_kind="retrieval",
            status="succeeded",
            result={
                "contexts": {"chunks": [chunk]},
                "trace": {},
                "image_descriptions": [],
            },
            workspaces=("finance",),
        )
        app.state.application = mock_application
        app.state.access_control = QueryOnlyAccess()
        try:
            response = await client.get("/runs/0199a0a0-0000-7000-8000-0000000000aa")
        finally:
            del app.state.access_control

        result = response.json()["result"]
        assert result["sources"][0]["download_url"] is None
        assert "image_url" not in result["contexts"]["chunks"][0]
        assert "_has_visual_asset" not in result["contexts"]["chunks"][0]

    async def test_retrieve_all_workspaces_pins_visible_records(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.alist_workspace_records.return_value = [
            {"workspace": "default"},
            {"workspace": "research_notes"},
        ]
        app.state.application = mock_application

        response = await client.post("/retrieve", json={"query": "hello", "all_workspaces": True})

        assert response.status_code == 202
        request = mock_application.retrieval.create.await_args.kwargs["request"]
        assert request.workspaces == ("default", "research_notes")

    async def test_retrieve_rejects_mode_field(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        response = await client.post("/retrieve", json={"query": "hello", "mode": "local"})

        assert response.status_code == 422
        mock_application.retrieval.create.assert_not_awaited()

    async def test_retrieve_forwards_chunk_top_k_and_query_images(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        response = await client.post(
            "/retrieve",
            json={
                "query": "hello",
                "chunk_top_k": 5,
                "query_images": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    }
                ],
            },
        )

        assert response.status_code == 202
        request = mock_application.retrieval.create.await_args.kwargs["request"]
        assert request.chunk_top_k == 5
        assert len(request.query_images) == 1


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """``/health`` is liveness only: in-process facts, never a database probe."""

    async def test_health_returns_status_without_probing_postgres(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres.core._pool import pg_pool

        probe = AsyncMock(return_value="off")
        monkeypatch.setattr(pg_pool, "run_once", probe)
        mock_application.health.set_search_toolchain(
            {
                "fd": {
                    "path": "/usr/local/bin/fd",
                    "version": "10.5.0",
                    "sha256": "a" * 64,
                },
                "rg": {
                    "path": "/usr/local/bin/rg",
                    "version": "15.2.0",
                    "sha256": "b" * 64,
                },
            }
        )
        app.state.application = mock_application
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "rag_initialized" in body
        assert body["storage"]["doc_status"] == "PGDocStatusStorage"
        assert set(body["components"]) == {
            "process",
            "operational_state",
            "run_coordinator",
            "cancellation_listener",
            "corpus_storage",
            "parser",
            "providers",
        }
        assert "postgres" not in body
        probe.assert_not_awaited()
        cap = body["answer_image_capability"]
        assert cap["status"] == "supported"
        assert cap["effective_max_images"] == 8
        assert cap["configured_ceiling"] == 8
        assert cap["model"] == "test-model"
        assert body["search_toolchain"]["fd"]["version"] == "10.5.0"
        assert body["search_toolchain"]["rg"]["sha256"] == "b" * 64


# ---------------------------------------------------------------------------
# TestHealthEndpointEnhanced
# ---------------------------------------------------------------------------


class TestHealthEndpointEnhanced:
    """Test enhanced /health endpoint with degraded state."""

    async def test_health_shows_degraded(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.health.mark_component_degraded("providers")
        app.state.application = mock_application
        resp = await client.get("/health")
        body = resp.json()
        assert body["status"] == "degraded"
        assert "Model providers unavailable" in body["warnings"]
        assert body["components"]["providers"]["status"] == "degraded"

    async def test_health_healthy_no_warnings(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
    ) -> None:
        app.state.application = mock_application
        resp = await client.get("/health")
        body = resp.json()
        assert body["status"] == "healthy"
        assert "warnings" not in body


# ---------------------------------------------------------------------------
# TestReadinessEndpoint
# ---------------------------------------------------------------------------


class TestReadinessEndpoint:
    """Test strict traffic-readiness semantics independently from /health."""

    async def test_ready_returns_200_without_authentication(
        self,
        client: AsyncClient,
        mock_config_no_auth_override: DlightragConfig,
        mock_application,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres.core._pool import pg_pool

        mutate_config(mock_config_no_auth_override, "access.auth_mode", "simple")
        mutate_config(mock_config_no_auth_override, "access.api_token", "required-elsewhere")
        probe = AsyncMock(return_value="off")
        monkeypatch.setattr(pg_pool, "run_once", probe)
        app.state.application = mock_application

        response = await client.get("/ready")

        assert response.status_code == 200
        assert response.json() == {"status": "ready", "service_role": "writer"}
        probe.assert_awaited_once()

    async def test_not_ready_manager_returns_503_without_probing_postgres(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres.core._pool import pg_pool

        mock_application.health.mark_closed()
        probe = AsyncMock(return_value="off")
        monkeypatch.setattr(pg_pool, "run_once", probe)
        app.state.application = mock_application

        response = await client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {
            "status": "not_ready",
            "service_role": "writer",
            "detail": "RAG service is not ready",
        }
        probe.assert_not_awaited()

    async def test_reader_requires_writable_domain_session(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres.core._pool import pg_pool

        mutate_config(mock_config, "deployment.service_role", "reader")
        monkeypatch.setattr(pg_pool, "run_once", AsyncMock(return_value="on"))
        app.state.application = mock_application

        response = await client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {
            "status": "not_ready",
            "service_role": "reader",
            "detail": "DlightRAG domain database session is not writable",
        }

    async def test_writer_requires_writable_domain_session(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres.core._pool import pg_pool

        monkeypatch.setattr(pg_pool, "run_once", AsyncMock(return_value="on"))
        app.state.application = mock_application

        response = await client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {
            "status": "not_ready",
            "service_role": "writer",
            "detail": "DlightRAG domain database session is not writable",
        }

    async def test_reader_corpus_outage_does_not_change_control_plane_readiness(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlightrag.adapters.postgres.corpus.lightrag_readonly as readonly_module
        from dlightrag.adapters.postgres.core._pool import pg_pool
        from dlightrag.adapters.postgres.corpus.corpus import PGReadinessProbe

        mutate_config(mock_config, "deployment.service_role", "reader")
        mock_application.health = ApplicationHealth(readiness_probe=PGReadinessProbe(mock_config))
        mock_application.health.mark_ready()
        mock_application.health.mark_component_degraded("corpus_storage")
        app.state.health = mock_application.health
        monkeypatch.setattr(pg_pool, "run_once", AsyncMock(return_value="off"))
        corpus_probe = AsyncMock(side_effect=RuntimeError("corpus pool unavailable"))
        monkeypatch.setattr(readonly_module, "verify_reader_corpus_session", corpus_probe)
        app.state.application = mock_application

        response = await client.get("/ready")

        assert response.status_code == 200
        assert response.json() == {"status": "ready", "service_role": "reader"}
        corpus_probe.assert_not_awaited()

    async def test_reader_is_ready_with_writable_operational_state(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres.core._pool import pg_pool
        from dlightrag.adapters.postgres.corpus.corpus import PGReadinessProbe

        mutate_config(mock_config, "deployment.service_role", "reader")
        mock_application.health = ApplicationHealth(readiness_probe=PGReadinessProbe(mock_config))
        mock_application.health.mark_ready()
        app.state.health = mock_application.health
        monkeypatch.setattr(pg_pool, "run_once", AsyncMock(return_value="off"))
        app.state.application = mock_application

        response = await client.get("/ready")

        assert response.status_code == 200
        assert response.json() == {"status": "ready", "service_role": "reader"}

    async def test_repeated_polls_reuse_one_cached_probe(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres.core._pool import pg_pool

        probe = AsyncMock(return_value="off")
        monkeypatch.setattr(pg_pool, "run_once", probe)
        app.state.application = mock_application

        for _ in range(5):
            assert (await client.get("/ready")).status_code == 200

        probe.assert_awaited_once()

    async def test_concurrent_cold_polls_share_one_probe(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A burst against a cold cache costs one round trip, not one per caller."""
        from dlightrag.adapters.postgres.core._pool import pg_pool

        started = asyncio.Event()
        release = asyncio.Event()

        async def _slow_probe(*_args: object, **_kwargs: object) -> str:
            started.set()
            await release.wait()
            return "off"

        probe = AsyncMock(side_effect=_slow_probe)
        monkeypatch.setattr(pg_pool, "run_once", probe)
        app.state.application = mock_application

        polls = [asyncio.create_task(client.get("/ready")) for _ in range(5)]
        await started.wait()
        release.set()
        responses = await asyncio.gather(*polls)

        assert [response.status_code for response in responses] == [200] * 5
        probe.assert_awaited_once()

    async def test_one_abandoned_poller_never_cancels_the_shared_probe(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres.core._pool import pg_pool

        started = asyncio.Event()
        release = asyncio.Event()
        completed = False

        async def _slow_probe(*_args: object, **_kwargs: object) -> str:
            nonlocal completed
            started.set()
            await release.wait()
            completed = True
            return "off"

        monkeypatch.setattr(pg_pool, "run_once", AsyncMock(side_effect=_slow_probe))
        app.state.application = mock_application

        abandoned = asyncio.create_task(client.get("/ready"))
        waiting = asyncio.create_task(client.get("/ready"))
        await started.wait()
        abandoned.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await abandoned
        release.set()

        assert (await waiting).status_code == 200
        assert completed is True

    async def test_the_cached_verdict_expires(
        self,
        client: AsyncClient,
        mock_application,
    ) -> None:
        probe = AsyncMock(return_value=None)
        health = ApplicationHealth(readiness_probe=probe, readiness_cache_seconds=0.0)
        health.mark_ready()
        mock_application.health = health
        app.state.health = health
        app.state.application = mock_application

        await client.get("/ready")
        await client.get("/ready")

        assert probe.await_count == 2

    async def test_only_operational_not_ready_transition_invalidates_cached_verdict(
        self,
        client: AsyncClient,
        mock_application,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Startup and schema transitions must never be served from a stale verdict."""
        from dlightrag.adapters.postgres.core._pool import pg_pool

        probe = AsyncMock(return_value="off")
        monkeypatch.setattr(pg_pool, "run_once", probe)
        app.state.application = mock_application

        assert (await client.get("/ready")).status_code == 200
        mock_application.health.mark_component_degraded("corpus_storage")
        assert (await client.get("/ready")).status_code == 200
        mock_application.health.mark_not_ready()
        assert (await client.get("/ready")).status_code == 503
        mock_application.health.mark_ready()
        assert (await client.get("/ready")).status_code == 200

        assert probe.await_count == 2


# ---------------------------------------------------------------------------
# TestDeleteEndpoint
# ---------------------------------------------------------------------------


class TestDeleteEndpoint:
    async def test_delete_accepts_exact_identifiers_as_a_run(
        self, client: AsyncClient, mock_application
    ) -> None:
        mock_application.corpus_mutations.create_delete.return_value = RunCreation(
            run=_queued_run_record(run_kind="corpus_mutation"), replayed=False
        )
        app.state.application = mock_application
        response = await client.post(
            "/runs/corpus/delete",
            headers={"Idempotency-Key": "delete-report"},
            json={"filenames": ["report.pdf"]},
        )
        assert response.status_code == 202
        assert response.json()["run_kind"] == "corpus_mutation"
        mock_application.corpus_mutations.create_delete.assert_awaited_once_with(
            workspace="default",
            submitted_by=ANY,
            file_paths=(),
            filenames=["report.pdf"],
            document_ids=(),
            idempotency_key="delete-report",
        )

    async def test_delete_requires_an_identifier(
        self, client: AsyncClient, mock_application
    ) -> None:
        app.state.application = mock_application
        response = await client.post(
            "/runs/corpus/delete",
            headers={"Idempotency-Key": "delete-empty"},
            json={},
        )
        assert response.status_code == 400

    @pytest.mark.usefixtures("_patch_application")
    async def test_legacy_delete_route_is_not_a_mutator(self, client: AsyncClient) -> None:
        assert (await client.request("DELETE", "/files", json={})).status_code == 405


# ---------------------------------------------------------------------------
# TestAnswerEndpoint
# ---------------------------------------------------------------------------


class TestAnswerEndpoint:
    """POST /answer admission: what the run's immutable input may carry."""

    async def test_answer_forwards_explicit_filters(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        resp = await client.post(
            "/answer",
            json={
                "query": "What did Ada write?",
                "filters": {"author": "Ada"},
            },
        )
        assert resp.status_code == 202
        answer_request = mock_application.answers.create.await_args.kwargs["request"]
        assert answer_request.filters is not None
        assert answer_request.filters.author == "Ada"

    async def test_answer_forwards_answer_context_limits(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        resp = await client.post(
            "/answer",
            json={
                "query": "What is RAG?",
                "chunk_top_k": 12,
            },
        )
        assert resp.status_code == 202
        answer_request = mock_application.answers.create.await_args.kwargs["request"]
        assert answer_request.retrieval.chunk_top_k == 12
        assert answer_request.semantic_highlights is False

    async def test_answer_forwards_semantic_highlights_opt_in(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        resp = await client.post(
            "/answer",
            json={
                "query": "What is RAG?",
                "semantic_highlights": True,
            },
        )
        assert resp.status_code == 202
        answer_request = mock_application.answers.create.await_args.kwargs["request"]
        assert answer_request.semantic_highlights is True

    async def test_answer_rejects_query_images_and_accepts_attachment_links(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        query_images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]

        rejected = await client.post(
            "/answer",
            json={"query": "What is shown?", "query_images": query_images},
        )
        assert rejected.status_code == 422
        mock_application.answers.create.assert_not_awaited()

        resp = await client.post(
            "/answer",
            json={
                "query": "What is shown?",
                "attachments": [{"url": "https://example.com/report.pdf", "filename": "r.pdf"}],
            },
        )

        assert resp.status_code == 202
        answer_request = mock_application.answers.create.await_args.kwargs["request"]
        assert [resource.url for resource in answer_request.resources] == [
            "https://example.com/report.pdf"
        ]
        assert answer_request.resources[0].filename == "r.pdf"

    async def test_answer_accepts_caller_history(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        history = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ]

        resp = await client.post(
            "/answer",
            json={"query": "And its population?", "history": history},
        )

        assert resp.status_code == 202
        answer_request = mock_application.answers.create.await_args.kwargs["request"]
        assert list(answer_request.history) == history

    async def test_json_enforces_link_count_limit(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        mutate_config(mock_config, "answer.generation.max_attachments", 2)
        app.state.application = mock_application

        resp = await client.post(
            "/answer",
            json={
                "query": "q",
                "attachments": [{"url": f"https://example.com/{index}.pdf"} for index in range(3)],
            },
        )

        assert resp.status_code == 413
        assert "2" in resp.json()["detail"]
        mock_application.answers.create.assert_not_awaited()

    async def test_answer_service_unavailable_503(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.answers.create = AsyncMock(
            side_effect=ApplicationClosedError("RAG not ready")
        )
        app.state.application = mock_application
        resp = await client.post("/answer", json={"query": "hello"})
        assert resp.status_code == 503

    async def test_answer_admission_limit_is_rejected_before_acceptance(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.answers.create = AsyncMock(
            side_effect=RunAdmissionLimitExceededError("limit reached")
        )
        app.state.application = mock_application

        response = await client.post("/answer", json={"query": "hello"})

        assert response.status_code == 503
        assert response.json()["detail"] == "Deployment-wide nonterminal admission limit reached"

    async def test_answer_runtime_unavailable_503(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.answers.create = AsyncMock(
            side_effect=AnswerRuntimeUnavailableError("Answer runtime is unavailable")
        )
        app.state.application = mock_application

        response = await client.post("/answer", json={"query": "hello"})

        assert response.status_code == 503
        assert response.json()["error_type"] == "unavailable"

    async def test_answer_input_rejection_uses_unprocessable_entity(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.answers.create = AsyncMock(
            side_effect=AnswerInputOverflowError("The answer input is too large.")
        )
        app.state.application = mock_application

        response = await client.post("/answer", json={"query": "hello"})

        assert response.status_code == 422
        assert response.json()["error_kind"] == "ANSWER_INPUT_OVERFLOW"


# ---------------------------------------------------------------------------
# TestAnswerMultipart
# ---------------------------------------------------------------------------


class TestAnswerMultipart:
    """POST /answer multipart: one JSON request part plus repeated attachment files."""

    async def test_multipart_mixes_links_and_files(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        import json as json_mod

        app.state.application = mock_application
        request_part = json_mod.dumps(
            {
                "query": "compare",
                "attachments": [{"url": "https://example.com/a.pdf"}],
            }
        )
        resp = await client.post(
            "/answer",
            data={"request": request_part},
            files=[("attachments", ("report.pdf", b"%PDF-body", "application/pdf"))],
        )

        assert resp.status_code == 202
        answer_request = mock_application.answers.create.await_args.kwargs["request"]
        link, upload = answer_request.resources
        assert link.url == "https://example.com/a.pdf"
        assert upload.content == b"%PDF-body"
        assert upload.filename == "report.pdf"
        assert upload.declared_mime == "application/pdf"

    async def test_multipart_accepts_maximum_unicode_history(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        import json as json_mod

        from dlightrag.engine.answer.client_contracts import (
            MAX_HISTORY_CONTENT_CHARS,
            MAX_HISTORY_MESSAGES,
        )

        app.state.application = mock_application
        history = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": "\U0001f642" * MAX_HISTORY_CONTENT_CHARS,
            }
            for index in range(MAX_HISTORY_MESSAGES)
        ]

        response = await client.post(
            "/answer",
            data={
                "request": json_mod.dumps(
                    {"query": "continue", "history": history},
                    ensure_ascii=False,
                )
            },
            files=[("attachments", ("note.txt", b"evidence", "text/plain"))],
        )

        assert response.status_code == 202
        answer_request = mock_application.answers.create.await_args.kwargs["request"]
        assert len(answer_request.history) == MAX_HISTORY_MESSAGES

    async def test_multipart_requires_exactly_one_request_part(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        import json as json_mod

        app.state.application = mock_application
        missing = await client.post(
            "/answer", files=[("attachments", ("a.txt", b"x", "text/plain"))]
        )
        assert missing.status_code == 400

        duplicate = await client.post(
            "/answer",
            data={"request": json_mod.dumps({"query": "q"})},
            files=[
                (
                    "request",
                    (
                        "r.json",
                        json_mod.dumps({"query": "q2"}),
                        "application/json",
                    ),
                )
            ],
        )
        assert duplicate.status_code == 400
        mock_application.answers.create.assert_not_awaited()

    async def test_multipart_rejects_wrong_part_name(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        import json as json_mod

        app.state.application = mock_application
        resp = await client.post(
            "/answer",
            data={"request": json_mod.dumps({"query": "q"})},
            files=[("documents", ("a.txt", b"x", "text/plain"))],
        )

        assert resp.status_code == 400
        mock_application.answers.create.assert_not_awaited()

    async def test_multipart_malformed_request_part_is_422(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        resp = await client.post(
            "/answer",
            data={"request": "{not json"},
            files=[("attachments", ("a.txt", b"x", "text/plain"))],
        )

        assert resp.status_code == 422
        mock_application.answers.create.assert_not_awaited()

    async def test_multipart_enforces_count_limit(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        import json as json_mod

        mutate_config(mock_config, "answer.generation.max_attachments", 2)
        app.state.application = mock_application
        resp = await client.post(
            "/answer",
            data={"request": json_mod.dumps({"query": "q"})},
            files=[("attachments", (f"f{index}.txt", b"x", "text/plain")) for index in range(3)],
        )

        assert resp.status_code == 413
        mock_application.answers.create.assert_not_awaited()

    async def test_multipart_enforces_per_item_limit(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        import json as json_mod

        mutate_config(mock_config, "answer.generation.max_attachment_bytes", 8)
        app.state.application = mock_application
        resp = await client.post(
            "/answer",
            data={"request": json_mod.dumps({"query": "q"})},
            files=[("attachments", ("big.bin", b"x" * 64, "application/octet-stream"))],
        )

        assert resp.status_code == 413
        mock_application.answers.create.assert_not_awaited()

    async def test_multipart_enforces_total_limit(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        import json as json_mod

        mutate_config(mock_config, "answer.generation.max_total_attachment_bytes", 16)
        app.state.application = mock_application
        resp = await client.post(
            "/answer",
            data={"request": json_mod.dumps({"query": "q"})},
            files=[
                ("attachments", ("a.bin", b"x" * 10, "application/octet-stream")),
                ("attachments", ("b.bin", b"y" * 10, "application/octet-stream")),
            ],
        )

        assert resp.status_code == 413
        mock_application.answers.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestFilesEndpoint
# ---------------------------------------------------------------------------


class TestFilesEndpoint:
    """Test GET /files endpoint."""

    @pytest.mark.usefixtures("_patch_application")
    async def test_list_files_success(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.get("/files")
        assert resp.status_code == 200
        body = resp.json()
        assert "files" in body
        assert "count" in body

    async def test_list_files_count_matches(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.file_panel_snapshot.return_value = {
            "files": [{"doc_id": value} for value in ("a", "b", "c")],
            "next_cursor": None,
            "fetched_rows": 3,
        }
        app.state.application = mock_application
        resp = await client.get("/files")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert len(body["files"]) == 3

    async def test_list_files_with_workspace(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application
        resp = await client.get("/files?workspace=project-z&limit=7")
        assert resp.status_code == 200
        call_kwargs = mock_application.corpora.file_panel_snapshot.call_args
        assert call_kwargs.args[0] == "project_z"  # normalized: hyphens → underscores
        assert call_kwargs.kwargs["page"].limit == 7

    async def test_list_files_encodes_next_cursor_and_rejects_tamper(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        next_cursor = FilePanelCursor(
            workspace="default",
            updated_at=datetime.datetime(2026, 8, 27),
            doc_id="doc-1",
        )
        mock_application.corpora.file_panel_snapshot.return_value = {
            "files": [{"doc_id": "doc-1"}],
            "next_cursor": next_cursor,
            "fetched_rows": 2,
        }
        app.state.application = mock_application

        response = await client.get("/files?limit=1")
        tampered = await client.get("/files?cursor=not-a-cursor")
        failed_cursor = mock_application.corpora.file_panel_cursor_codec.encode(
            FilePanelCursor(
                workspace="default",
                updated_at=None,
                doc_id="failed-1",
                view="failed",
            )
        )
        cross_view = await client.get("/files", params={"cursor": failed_cursor})

        assert response.status_code == 200
        assert response.json()["next_cursor"] == (
            mock_application.corpora.file_panel_cursor_codec.encode(next_cursor)
        )
        assert response.json()["fetched_rows"] == 2
        assert tampered.status_code == 422
        assert cross_view.status_code == 422

    async def test_failed_files_use_bounded_snapshot(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.failed_file_snapshot.return_value = {
            "failed": [{"doc_id": "failed-1", "error": "parser failed"}],
            "next_cursor": None,
            "fetched_rows": 1,
        }
        app.state.application = mock_application

        response = await client.get("/files/failed?limit=1")

        assert response.status_code == 200
        assert response.json()["failed"][0]["error"] == "parser failed"
        assert response.json()["fetched_rows"] == 1
        assert mock_application.corpora.failed_file_snapshot.await_args.kwargs["page"].limit == 1


# ---------------------------------------------------------------------------
# TestAnswerStreamMode
# ---------------------------------------------------------------------------


class TestAnswerStreamMode:
    """Runtime failure mappings the app still owns for its non-durable routes."""

    async def test_rejected_metadata_is_a_client_error_not_a_500(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        """Metadata validation happens below the request model, so it needs its own mapping."""
        mock_application.corpora.update_metadata = AsyncMock(
            side_effect=MetadataValidationError("title is a built-in metadata field")
        )
        app.state.application = mock_application
        resp = await client.post("/metadata/doc-1", json={"metadata": {"title": "X"}})
        assert resp.status_code == 400
        assert resp.json()["error_type"] == "validation"


class TestAPIContracts:
    """Request and response contracts are explicit in OpenAPI."""

    async def test_openapi_exposes_pydantic_response_models(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_application
    ) -> None:
        app.state.application = mock_application

        resp = await client.get("/openapi.json")

        assert resp.status_code == 200
        spec = resp.json()
        schemas = spec["components"]["schemas"]
        assert "AnswerResponse" in schemas
        assert "RunDescriptor" in schemas
        assert "RunStatusResponse" in schemas
        assert "RunStatus" not in schemas
        assert "RunPhase" not in schemas
        assert schemas["RunDescriptor"]["properties"]["status"] == {
            "type": "string",
            "enum": ["queued", "running", "succeeded", "failed", "cancelled"],
            "title": "Status",
        }
        phase_schema = schemas["RunStatusResponse"]["properties"]["phase"]
        assert phase_schema["anyOf"][0] == {"type": "string"}
        ingest_properties = schemas["IngestRequest"]["properties"]
        assert "download_uri" in ingest_properties
        assert "download_uris" in ingest_properties
        assert "download_url" not in ingest_properties
        assert "download_urls" not in ingest_properties
        assert (
            spec["paths"]["/retrieve"]["post"]["responses"]["202"]["content"]["application/json"][
                "schema"
            ]["$ref"]
            == "#/components/schemas/RunDescriptor"
        )
        assert (
            spec["paths"]["/workspaces"]["get"]["responses"]["200"]["content"]["application/json"][
                "schema"
            ]["$ref"]
            == "#/components/schemas/WorkspacesResponse"
        )


class TestMetadataAPI:
    @pytest.mark.usefixtures("_patch_application")
    async def test_search_route_is_not_shadowed_by_the_doc_id_route(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
    ) -> None:
        """`/metadata/search` is a literal path, so it must be declared first."""
        mock_application.corpora.search_metadata = AsyncMock(
            return_value=MetadataSearchPage(
                document_ids=("doc-1",),
                next_cursor=None,
                fetched_rows=1,
            )
        )
        app.state.application = mock_application

        resp = await client.post("/metadata/search", json={"custom": {"department": "legal"}})

        assert resp.status_code == 200
        assert resp.json()["document_ids"] == ["doc-1"]

    @pytest.mark.usefixtures("_patch_application")
    async def test_unknown_filter_name_is_rejected_not_ignored(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
    ) -> None:
        """A dropped filter name would match every document instead of failing."""
        app.state.application = mock_application

        resp = await client.post("/metadata/search", json={"nonsense": "x"})

        assert resp.status_code == 422
        mock_application.corpora.search_metadata.assert_not_awaited()

    @pytest.mark.usefixtures("_patch_application")
    async def test_search_pages_by_doc_id_with_an_opaque_round_tripped_cursor(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
    ) -> None:
        codec = MetadataSearchCursorCodec(b"api-server-test")
        continuation = MetadataSearchCursor(
            workspace="default",
            after_doc_id="doc-40",
            mode="exact",
        )

        def search_side_effect(workspace, _filters, *, page):
            captured.append(page)
            return MetadataSearchPage(
                document_ids=("doc-41", "doc-42"),
                next_cursor=continuation,
                fetched_rows=3,
            )

        captured = []
        mock_application.corpora.search_metadata = AsyncMock(side_effect=search_side_effect)
        app.state.application = mock_application

        first = await client.post("/metadata/search", json={"filename": "Report"})
        assert first.status_code == 200
        body = first.json()
        assert body["document_ids"] == ["doc-41", "doc-42"]
        assert body["count"] == 2
        assert body["workspace"] == "default"
        assert body["next_cursor"] == codec.encode(continuation)
        assert captured[-1].limit == 50
        assert captured[-1].cursor is None

        second = await client.post(
            "/metadata/search",
            params={"cursor": body["next_cursor"], "limit": 25},
            json={"filename": "Report"},
        )
        assert second.status_code == 200
        assert captured[-1].limit == 25
        assert captured[-1].cursor == continuation

    @pytest.mark.usefixtures("_patch_application")
    async def test_search_limit_bounds_are_enforced_before_storage(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
    ) -> None:
        app.state.application = mock_application

        for limit in (0, 101):
            resp = await client.post(
                "/metadata/search",
                params={"limit": limit},
                json={"filename": "Report"},
            )
            assert resp.status_code == 422

        resp = await client.post(
            "/metadata/search",
            params={"limit": "abc"},
            json={"filename": "Report"},
        )
        assert resp.status_code == 422
        mock_application.corpora.search_metadata.assert_not_awaited()

    @pytest.mark.usefixtures("_patch_application")
    async def test_search_rejects_tampered_and_cross_workspace_cursors_before_storage(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
    ) -> None:
        codec = MetadataSearchCursorCodec(b"api-server-test")
        foreign = codec.encode(
            MetadataSearchCursor(
                workspace="other_ws",
                after_doc_id="doc-1",
                mode="exact",
            )
        )
        app.state.application = mock_application

        for bad_cursor in ("AAAA.BBBB", foreign):
            resp = await client.post(
                "/metadata/search",
                params={"cursor": bad_cursor},
                json={"filename": "Report"},
            )
            assert resp.status_code == 422
        mock_application.corpora.search_metadata.assert_not_awaited()

    @pytest.mark.usefixtures("_patch_application")
    async def test_search_enforces_authorization_before_filter_validation(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
    ) -> None:
        from dlightrag.application.access import AccessDeniedError

        class DenyAllAccess:
            async def check(self, user, action, *, workspace=None):
                raise AccessDeniedError("denied")

            async def filter_workspaces(self, user, action, workspaces):
                return []

        app.state.application = mock_application
        app.state.access_control = DenyAllAccess()

        try:
            resp = await client.post("/metadata/search", json={"nonsense": "x"})
        finally:
            del app.state.access_control

        assert resp.status_code == 403
        mock_application.corpora.search_metadata.assert_not_awaited()


class TestAnswerRunChildren:
    @pytest.mark.usefixtures("_patch_application")
    async def test_children_default_page_and_additive_shape(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
    ) -> None:
        from uuid import UUID

        run_id = "0199a0a0-0000-7000-8000-000000000077"
        codec = mock_application.answers.child_roster_cursor_codec
        continuation = ChildRosterCursor(
            run_id=UUID(run_id),
            created_at=datetime.datetime(2026, 3, 4, 5, 6, 7, tzinfo=datetime.UTC),
            child_session_id=UUID("0199a0a0-0000-7000-8000-000000000088"),
        )
        captured = []

        def children_side_effect(owner_id, run_id, *, page=None):
            captured.append((owner_id, run_id, page))
            return ChildRosterPage(
                children=({"child_session_id": "0199a0a0-0000-7000-8000-000000000088"},),
                next_cursor=continuation,
                fetched_rows=2,
            )

        mock_application.answers.children = AsyncMock(side_effect=children_side_effect)
        app.state.application = mock_application

        first = await client.get(f"/answer/{run_id}/children")

        assert first.status_code == 200
        body = first.json()
        assert body["run_id"] == run_id
        assert body["children"] == [{"child_session_id": "0199a0a0-0000-7000-8000-000000000088"}]
        assert body["next_cursor"] == codec.encode(continuation)
        assert captured[-1][2].limit == 50
        assert captured[-1][2].cursor is None

        second = await client.get(
            f"/answer/{run_id}/children",
            params={"cursor": body["next_cursor"], "limit": 25},
        )
        assert second.status_code == 200
        assert captured[-1][2].limit == 25
        assert captured[-1][2].cursor == continuation

    @pytest.mark.usefixtures("_patch_application")
    async def test_children_limit_bounds_and_malformed_cursor_are_422(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
    ) -> None:
        run_id = "0199a0a0-0000-7000-8000-000000000077"
        app.state.application = mock_application

        for params in ({"limit": 0}, {"limit": 101}, {"limit": "abc"}):
            resp = await client.get(f"/answer/{run_id}/children", params=params)
            assert resp.status_code == 422
        resp = await client.get(f"/answer/{run_id}/children", params={"cursor": "AAAA.BBBB"})
        assert resp.status_code == 422
        mock_application.answers.children.assert_not_awaited()

    @pytest.mark.usefixtures("_patch_application")
    async def test_children_cross_run_cursor_is_422_before_storage(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
    ) -> None:
        from uuid import UUID

        run_id = "0199a0a0-0000-7000-8000-000000000077"
        codec = mock_application.answers.child_roster_cursor_codec
        foreign = codec.encode(
            ChildRosterCursor(
                run_id=UUID("0199a0a0-0000-7000-8000-000000000080"),
                created_at=datetime.datetime(2026, 3, 4, 5, 6, 7, tzinfo=datetime.UTC),
                child_session_id=UUID("0199a0a0-0000-7000-8000-000000000088"),
            )
        )
        app.state.application = mock_application

        resp = await client.get(f"/answer/{run_id}/children", params={"cursor": foreign})

        assert resp.status_code == 422
        mock_application.answers.children.assert_not_awaited()

    @pytest.mark.usefixtures("_patch_application")
    async def test_children_unknown_run_is_404(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_application,
    ) -> None:
        run_id = "0199a0a0-0000-7000-8000-000000000077"
        mock_application.answers.children = AsyncMock(return_value=None)
        app.state.application = mock_application

        resp = await client.get(f"/answer/{run_id}/children")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Request body limits
# ---------------------------------------------------------------------------


def _echo_length_app(max_bytes: int) -> FastAPI:
    from starlette.requests import Request

    from dlightrag.adapters.http.rest.middleware import RequestBodyLimitMiddleware

    application = FastAPI()

    @application.post("/probe")
    async def probe(request: Request) -> dict[str, int]:
        return {"received": len(await request.body())}

    @application.post("/answer")
    @application.post("/web/api/answer")
    async def answer(request: Request) -> dict[str, int]:
        return {"received": len(await request.body())}

    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=max_bytes,
        multipart_path_max_bytes={"/answer": max_bytes, "/web/api/answer": max_bytes},
    )
    return application


async def _post(app: FastAPI, path: str = "/probe", **kwargs: Any) -> Response:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        return await client.post(path, **kwargs)


@pytest.mark.asyncio
async def test_a_declared_oversize_body_is_refused_before_the_route_runs() -> None:
    response = await _post(
        _echo_length_app(100),
        content=b"x" * 200,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413, response.text
    assert response.json()["error_type"] == "validation"


@pytest.mark.asyncio
async def test_an_undeclared_body_returns_413_at_the_cap() -> None:
    async def chunks():
        for _ in range(10):
            yield b"x" * 50

    response = await _post(
        _echo_length_app(100),
        content=chunks(),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["error_type"] == "validation"


@pytest.mark.asyncio
async def test_an_undeclared_mislabeled_body_still_returns_413_at_the_cap() -> None:
    async def chunks():
        for _ in range(10):
            yield b"x" * 50

    response = await _post(
        _echo_length_app(100),
        content=chunks(),
        headers={"content-type": "text/plain"},
    )

    assert response.status_code == 413
    assert response.json()["error_type"] == "validation"


@pytest.mark.asyncio
async def test_an_unmapped_multipart_upload_uses_the_default_cap() -> None:
    response = await _post(
        _echo_length_app(100),
        files={"f": ("big.bin", b"x" * 400)},
    )

    assert response.status_code == 413
    assert response.json()["error_type"] == "validation"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/answer", "/web/api/answer"])
async def test_a_chunked_answer_multipart_is_refused_at_receive_layer(path: str) -> None:
    async def chunks():
        for _ in range(10):
            yield b"x" * 50

    response = await _post(
        _echo_length_app(100),
        path,
        content=chunks(),
        headers={"content-type": "multipart/form-data; boundary=test"},
    )

    assert response.status_code == 413
    assert response.json()["error_type"] == "validation"


@pytest.mark.asyncio
async def test_real_app_returns_413_for_chunked_answer_multipart_overflow(
    mock_config: DlightragConfig,
) -> None:
    mutate_config(mock_config, "answer.generation.max_total_attachment_bytes", 64)
    set_config(mock_config)

    async def chunks():
        yield (
            b"--test\r\n"
            b'Content-Disposition: form-data; name="attachments"; filename="huge.bin"\r\n'
            b"Content-Type: application/octet-stream\r\n\r\n"
        )
        for _ in range(105):
            yield b"x" * 65_536
        yield b"\r\n--test--\r\n"

    application = create_app(include_web_app=False)
    application_double = AsyncMock()
    application_double.config = mock_config
    application.state.application = application_double
    response = await _post(
        application,
        "/answer",
        content=chunks(),
        headers={
            "content-type": "multipart/form-data; boundary=test",
            "origin": "https://example.test",
            "x-request-id": "body-limit-test",
        },
    )

    assert response.status_code == 413
    assert response.json()["error_type"] == "validation"
    assert response.headers["x-request-id"] == "body-limit-test"
    assert response.headers["access-control-allow-origin"] == "*"


@pytest.mark.asyncio
async def test_real_app_caps_chunked_ingest_multipart_before_parsing(
    mock_config: DlightragConfig,
) -> None:
    mutate_config(mock_config, "interfaces.max_upload_size_mb", 8)
    mutate_config(mock_config, "corpus.ingestion.max_upload_bytes", 1024 * 1024)
    set_config(mock_config)

    async def chunks():
        yield (
            b"--test\r\n"
            b'Content-Disposition: form-data; name="file"; filename="huge.bin"\r\n'
            b"Content-Type: application/octet-stream\r\n\r\n"
        )
        for _ in range(50):
            yield b"x" * 65_536
        yield b"\r\n--test--\r\n"

    application = create_app(include_web_app=False)
    application_double = AsyncMock()
    application_double.config = mock_config
    application.state.application = application_double
    response = await _post(
        application,
        "/runs/corpus/ingest/upload",
        content=chunks(),
        headers={"content-type": "multipart/form-data; boundary=test"},
    )

    assert response.status_code == 413, response.text
    assert response.json()["error_type"] == "validation"
    assert response.json()["detail"] == "Request body is too large"


@pytest.mark.asyncio
async def test_corpus_upload_authenticates_before_parsing_multipart(
    mock_config: DlightragConfig,
) -> None:
    mutate_config(mock_config, "access.auth_mode", "simple")
    mutate_config(mock_config, "access.api_token", "secret-token")
    set_config(mock_config)
    application = create_app(include_web_app=False)
    application.state.application = AsyncMock()

    response = await _post(
        application,
        "/runs/corpus/ingest/upload",
        content=b"malformed multipart body",
        headers={"content-type": "multipart/form-data; boundary=missing"},
    )

    assert response.status_code == 401
    assert "Authorization" in response.json()["detail"]


@pytest.mark.asyncio
async def test_multipart_header_does_not_raise_json_route_body_cap(
    mock_config: DlightragConfig,
) -> None:
    mutate_config(mock_config, "interfaces.max_upload_size_mb", 32)
    set_config(mock_config)

    async def chunks():
        yield b'--test\r\nContent-Disposition: form-data; name="junk"\r\n\r\n'
        for _ in range(200):
            yield b"x" * 65_536
        yield b"\r\n--test--\r\n"

    application = create_app(include_web_app=False)
    application.state.application = AsyncMock()
    response = await _post(
        application,
        "/retrieve",
        content=chunks(),
        headers={"content-type": "multipart/form-data; boundary=test"},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body is too large"


@pytest.mark.asyncio
async def test_the_app_admits_answer_history_with_the_shared_body_cap(
    mock_config: DlightragConfig,
) -> None:
    set_config(mock_config)

    response = await _post(
        create_app(include_web_app=False),
        content=b'{"query":"' + b"x" * (1024 * 1024) + b'"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code != 413


@pytest.mark.asyncio
async def test_the_app_still_refuses_a_body_over_the_shared_json_budget(
    mock_config: DlightragConfig,
) -> None:
    from dlightrag.engine.answer.client_contracts import (
        MAX_HISTORY_CONTENT_CHARS,
        MAX_HISTORY_MESSAGES,
        MAX_QUERY_IMAGES,
    )

    set_config(mock_config)
    history_bytes = MAX_HISTORY_MESSAGES * MAX_HISTORY_CONTENT_CHARS * 4
    image_bytes = MAX_QUERY_IMAGES * (
        ((mock_config.answer.generation.image_max_bytes + 2) // 3) * 4
    )
    over_budget = max(history_bytes, image_bytes) + 2 * 1024 * 1024

    response = await _post(
        create_app(include_web_app=False),
        content=b'{"query":"' + b"x" * over_budget + b'"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413


@pytest.mark.asyncio
async def test_the_app_admits_the_fixed_retrieve_image_contract(
    mock_config: DlightragConfig,
) -> None:
    from dlightrag.engine.answer.client_contracts import MAX_QUERY_IMAGES

    set_config(mock_config)
    image_sized_body = (
        MAX_QUERY_IMAGES * (((mock_config.answer.generation.image_max_bytes + 2) // 3) * 4) - 4096
    )

    response = await _post(
        create_app(include_web_app=False),
        content=b'{"query":"' + b"x" * image_sized_body + b'"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code != 413


@pytest.mark.asyncio
async def test_a_route_that_rejects_an_oversized_upload_is_not_reported_as_an_auth_failure(
    mock_config: DlightragConfig,
) -> None:
    set_config(mock_config)
    application = create_app(include_web_app=False)

    @application.post("/probe")
    async def probe() -> None:
        raise HTTPException(status_code=413, detail="too many documents")

    response = await _post(application)

    assert response.status_code == 413
    assert response.json()["error_type"] == "validation"


def test_body_limit_split_preserves_non_limit_exception_group_members() -> None:
    from dlightrag.adapters.http.rest.middleware import _RequestBodyTooLarge, _split_body_too_large

    matched, remainder = _split_body_too_large(
        ExceptionGroup(
            "mixed",
            [_RequestBodyTooLarge(), ExceptionGroup("server", [RuntimeError("boom")])],
        )
    )

    assert matched is not None
    assert isinstance(remainder, BaseExceptionGroup)
    server_group = remainder.exceptions[0]
    assert isinstance(server_group, BaseExceptionGroup)
    assert isinstance(server_group.exceptions[0], RuntimeError)
    assert str(server_group.exceptions[0]) == "boom"


def test_body_limit_strips_root_path_only_at_segment_boundary() -> None:
    from dlightrag.adapters.http.rest.middleware import _request_path

    assert _request_path({"path": "/answer", "root_path": "/a"}) == "/answer"
    assert _request_path({"path": "/api/answer", "root_path": "/api"}) == "/answer"
