# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for FastAPI REST server endpoints and auth middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.api.server import _get_config, app
from dlightrag.config import DlightragConfig
from dlightrag.core.retrieval.engine import RetrievalResult
from dlightrag.core.servicemanager import RAGServiceUnavailableError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config(test_config: DlightragConfig):
    """Override FastAPI config dependency."""
    app.dependency_overrides[_get_config] = lambda: test_config
    yield test_config
    app.dependency_overrides.pop(_get_config, None)


@pytest.fixture
def mock_service():
    """Create a mock RAGService."""
    service = AsyncMock()
    service.aingest = AsyncMock(return_value={"status": "success", "processed": 1})
    service.aretrieve = AsyncMock(
        return_value=RetrievalResult(answer="42", contexts={"chunks": []}, raw={})
    )
    service.aanswer = AsyncMock(
        return_value=RetrievalResult(answer="The answer is 42", contexts={"chunks": []}, raw={})
    )
    service.alist_ingested_files = AsyncMock(return_value=[])
    service.adelete_files = AsyncMock(return_value=[{"status": "deleted"}])
    return service


@pytest.fixture
def mock_manager(mock_service):
    """Create a mock RAGServiceManager that delegates to mock_service."""
    manager = AsyncMock()
    manager.aingest = mock_service.aingest
    manager.aretrieve = mock_service.aretrieve
    manager.aanswer = mock_service.aanswer
    manager.list_ingested_files = mock_service.alist_ingested_files
    manager.delete_files = mock_service.adelete_files
    manager.list_workspaces = AsyncMock(return_value=["default"])
    manager.is_ready = lambda: True
    manager.get_error_info = lambda: {"last_error": None, "timestamp": None, "retry_after": 30.0}
    manager.close = AsyncMock()
    return manager


@pytest.fixture
def _patch_manager(mock_manager):
    """Set mock manager on app.state."""
    app.state.manager = mock_manager
    yield
    if hasattr(app.state, "manager"):
        del app.state.manager


@pytest.fixture
async def client():
    """Create httpx async client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# TestAuthMiddleware
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    """Test _verify_auth bearer token validation."""

    async def test_no_token_configured_passes(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_manager")
    async def test_valid_token_passes(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        mock_config.api_auth_token = "secret-token"
        resp = await client.get(
            "/files",
            headers={"Authorization": "Bearer secret-token"},
        )
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_manager")
    async def test_missing_auth_header_401(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        mock_config.api_auth_token = "secret-token"
        resp = await client.get("/files")
        assert resp.status_code == 401

    @pytest.mark.usefixtures("_patch_manager")
    async def test_wrong_scheme_401(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        mock_config.api_auth_token = "secret-token"
        resp = await client.get(
            "/files",
            headers={"Authorization": "Basic abc123"},
        )
        assert resp.status_code == 401

    @pytest.mark.usefixtures("_patch_manager")
    async def test_invalid_token_403(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        mock_config.api_auth_token = "secret-token"
        resp = await client.get(
            "/files",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 403

    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("POST", "/ingest", {"source_type": "local", "path": "/tmp/f.pdf"}),
            ("POST", "/retrieve", {"query": "hello"}),
            ("POST", "/answer", {"query": "hello"}),
            ("DELETE", "/files", {"filenames": ["f.pdf"]}),
        ],
    )
    async def test_endpoint_requires_auth(
        self, method: str, path: str, body: dict, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        mock_config.api_auth_token = "secret-token"
        resp = await client.request(method, path, json=body)
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# TestIngestEndpoint
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    """Test /ingest validation and routing."""

    @pytest.mark.usefixtures("_patch_manager")
    async def test_local_requires_path(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.post("/ingest", json={"source_type": "local"})
        assert resp.status_code == 400

    @pytest.mark.usefixtures("_patch_manager")
    async def test_azure_blob_requires_container(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.post("/ingest", json={"source_type": "azure_blob"})
        assert resp.status_code == 400

    @pytest.mark.usefixtures("_patch_manager")
    async def test_snowflake_requires_query(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.post("/ingest", json={"source_type": "snowflake"})
        assert resp.status_code == 400

    async def test_local_success(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={"source_type": "local", "path": "/data/file.pdf"},
        )
        assert resp.status_code == 200
        mock_manager.aingest.assert_awaited_once()

    async def test_azure_blob_success(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={
                "source_type": "azure_blob",
                "container_name": "my-container",
                "blob_path": "docs/file.pdf",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_manager")
    async def test_azure_blob_path_and_prefix_mutually_exclusive(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.post(
            "/ingest",
            json={
                "source_type": "azure_blob",
                "container_name": "c",
                "blob_path": "docs/file.pdf",
                "prefix": "docs/",
            },
        )
        assert resp.status_code == 400

    async def test_ingest_with_workspace(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={"source_type": "local", "path": "/data/file.pdf", "workspace": "project-x"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_manager.aingest.call_args
        assert call_kwargs[0][0] == "project-x"  # first positional arg is workspace

    async def test_ingest_service_unavailable_503(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.aingest = AsyncMock(side_effect=RAGServiceUnavailableError("RAG not ready"))
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={"source_type": "local", "path": "/data/file.pdf"},
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# TestRetrieveEndpoint
# ---------------------------------------------------------------------------


class TestRetrieveEndpoint:
    """Test /retrieve endpoint."""

    async def test_retrieve_success(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post("/retrieve", json={"query": "What is RAG?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "contexts" in body
        assert "raw" in body

    async def test_retrieve_with_custom_mode(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/retrieve",
            json={"query": "hello", "mode": "local"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_manager.aretrieve.call_args
        assert call_kwargs.kwargs["mode"] == "local"


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Test /health endpoint."""

    async def test_health_returns_status(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "rag_initialized" in body
        assert "storage" in body


# ---------------------------------------------------------------------------
# TestDeleteEndpoint
# ---------------------------------------------------------------------------


class TestDeleteEndpoint:
    """Test DELETE /files endpoint."""

    async def test_delete_by_filenames(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.request(
            "DELETE",
            "/files",
            json={"filenames": ["report.pdf"]},
        )
        assert resp.status_code == 200
        mock_manager.delete_files.assert_awaited_once()

    async def test_delete_by_file_paths(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.request(
            "DELETE",
            "/files",
            json={"file_paths": ["/storage/report.pdf"]},
        )
        assert resp.status_code == 200

    async def test_delete_with_workspace(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.request(
            "DELETE",
            "/files",
            json={"filenames": ["report.pdf"], "workspace": "project-y"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_manager.delete_files.call_args
        assert call_kwargs[0][0] == "project-y"  # first positional arg is workspace


# ---------------------------------------------------------------------------
# TestAnswerEndpoint
# ---------------------------------------------------------------------------


class TestAnswerEndpoint:
    """Test /answer endpoint."""

    async def test_answer_success(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "What is RAG?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "contexts" in body
        assert "raw" in body
        assert body["answer"] == "The answer is 42"

    async def test_answer_with_conversation_history(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        resp = await client.post(
            "/answer",
            json={"query": "Follow up", "conversation_history": history},
        )
        assert resp.status_code == 200
        call_kwargs = mock_manager.aanswer.call_args.kwargs
        assert call_kwargs["conversation_history"] == history

    async def test_answer_service_unavailable_503(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.aanswer = AsyncMock(side_effect=RAGServiceUnavailableError("RAG not ready"))
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "hello"})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# TestFilesEndpoint
# ---------------------------------------------------------------------------


class TestFilesEndpoint:
    """Test GET /files endpoint."""

    @pytest.mark.usefixtures("_patch_manager")
    async def test_list_files_success(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.get("/files")
        assert resp.status_code == 200
        body = resp.json()
        assert "files" in body
        assert "count" in body

    async def test_list_files_count_matches(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.list_ingested_files = AsyncMock(return_value=["a.pdf", "b.pdf", "c.pdf"])
        app.state.manager = mock_manager
        resp = await client.get("/files")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert len(body["files"]) == 3

    async def test_list_files_with_workspace(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.get("/files?workspace=project-z")
        assert resp.status_code == 200
        call_kwargs = mock_manager.list_ingested_files.call_args
        assert call_kwargs[0][0] == "project-z"  # first positional arg is workspace


# ---------------------------------------------------------------------------
# TestAnswerStreamMode
# ---------------------------------------------------------------------------


class TestAnswerStreamMode:
    """Test POST /answer with stream=true SSE mode."""

    async def test_stream_returns_sse_content_type(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        async def mock_tokens():
            for t in ["Hello", " world"]:
                yield t

        mock_manager.aanswer_stream = AsyncMock(
            return_value=({"chunks": []}, {}, mock_tokens())
        )
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "test", "stream": True})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    async def test_stream_event_sequence(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """Verify context -> token(s) -> done event order."""
        async def mock_tokens():
            for t in ["Hi", " there"]:
                yield t

        mock_manager.aanswer_stream = AsyncMock(
            return_value=({"chunks": [{"id": "c1"}]}, {"sources": []}, mock_tokens())
        )
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "test", "stream": True})
        lines = [line for line in resp.text.split("\n") if line.startswith("data: ")]

        import json as json_mod

        events = [json_mod.loads(line.removeprefix("data: ")) for line in lines]
        assert events[0]["type"] == "context"
        assert events[0]["data"] == {"chunks": [{"id": "c1"}]}
        assert events[-1]["type"] == "done"
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 2
        assert token_events[0]["content"] == "Hi"
        assert token_events[1]["content"] == " there"

    async def test_stream_error_during_iteration(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """Error mid-stream produces error event."""
        async def mock_tokens():
            yield "start"
            raise RuntimeError("LLM exploded")

        mock_manager.aanswer_stream = AsyncMock(
            return_value=({"chunks": []}, {}, mock_tokens())
        )
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "test", "stream": True})
        lines = [line for line in resp.text.split("\n") if line.startswith("data: ")]

        import json as json_mod

        events = [json_mod.loads(line.removeprefix("data: ")) for line in lines]
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert "LLM exploded" in error_events[0]["message"]

    async def test_stream_service_unavailable_503(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """Pre-stream errors return normal HTTP 503."""
        mock_manager.aanswer_stream = AsyncMock(
            side_effect=RAGServiceUnavailableError("RAG not ready")
        )
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "hello", "stream": True})
        assert resp.status_code == 503

    async def test_stream_false_returns_json(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """stream=false (default) returns normal JSON response."""
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "test", "stream": False})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert body["answer"] == "The answer is 42"

    async def test_default_no_stream_returns_json(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """Omitting stream field returns normal JSON (backwards compatible)."""
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "test"})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
