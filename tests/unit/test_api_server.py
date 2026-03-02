# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for FastAPI REST server endpoints and auth middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from corprag.api.server import _get_config, app
from corprag.config import CorpragConfig
from corprag.pool import RAGServiceUnavailableError
from corprag.retrieval.engine import RetrievalResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config(test_config: CorpragConfig):
    """Override FastAPI config dependency."""
    app.dependency_overrides[_get_config] = lambda: test_config
    yield test_config
    app.dependency_overrides.pop(_get_config, None)


@pytest.fixture
def mock_service():
    """Create a mock RAGService (not injected — use _patch_service)."""
    service = AsyncMock()
    service.aingest = AsyncMock(return_value={"status": "success", "processed": 1})
    service.aretrieve = AsyncMock(
        return_value=RetrievalResult(answer="42", contexts={"chunks": []}, raw={})
    )
    service.list_ingested_files = MagicMock(return_value=[])
    service.adelete_files = AsyncMock(return_value=[{"status": "deleted"}])
    return service


@pytest.fixture
def _patch_service(mock_service):
    """Patch _get_rag_service to return the mock service.

    _get_rag_service is called directly inside endpoint functions (not via
    FastAPI Depends), so dependency_overrides won't intercept it. Patch
    at the module level instead.
    """

    async def _fake_get():
        return mock_service

    with patch("corprag.api.server._get_rag_service", new=_fake_get):
        yield


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
        self, client: AsyncClient, mock_config: CorpragConfig
    ) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_service")
    async def test_valid_token_passes(
        self, client: AsyncClient, mock_config: CorpragConfig
    ) -> None:
        mock_config.api_auth_token = "secret-token"
        resp = await client.get(
            "/files",
            headers={"Authorization": "Bearer secret-token"},
        )
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_service")
    async def test_missing_auth_header_401(
        self, client: AsyncClient, mock_config: CorpragConfig
    ) -> None:
        mock_config.api_auth_token = "secret-token"
        resp = await client.get("/files")
        assert resp.status_code == 401

    @pytest.mark.usefixtures("_patch_service")
    async def test_wrong_scheme_401(self, client: AsyncClient, mock_config: CorpragConfig) -> None:
        mock_config.api_auth_token = "secret-token"
        resp = await client.get(
            "/files",
            headers={"Authorization": "Basic abc123"},
        )
        assert resp.status_code == 401

    @pytest.mark.usefixtures("_patch_service")
    async def test_invalid_token_403(self, client: AsyncClient, mock_config: CorpragConfig) -> None:
        mock_config.api_auth_token = "secret-token"
        resp = await client.get(
            "/files",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# TestIngestEndpoint
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    """Test /ingest validation and routing."""

    @pytest.mark.usefixtures("_patch_service")
    async def test_local_requires_path(
        self, client: AsyncClient, mock_config: CorpragConfig
    ) -> None:
        resp = await client.post("/ingest", json={"source_type": "local"})
        assert resp.status_code == 400

    @pytest.mark.usefixtures("_patch_service")
    async def test_azure_blob_requires_container(
        self, client: AsyncClient, mock_config: CorpragConfig
    ) -> None:
        resp = await client.post("/ingest", json={"source_type": "azure_blob"})
        assert resp.status_code == 400

    @pytest.mark.usefixtures("_patch_service")
    async def test_snowflake_requires_query(
        self, client: AsyncClient, mock_config: CorpragConfig
    ) -> None:
        resp = await client.post("/ingest", json={"source_type": "snowflake"})
        assert resp.status_code == 400

    async def test_local_success(
        self, client: AsyncClient, mock_config: CorpragConfig, mock_service
    ) -> None:
        async def _fake_get():
            return mock_service

        with patch("corprag.api.server._get_rag_service", new=_fake_get):
            resp = await client.post(
                "/ingest",
                json={"source_type": "local", "path": "/data/file.pdf"},
            )

        assert resp.status_code == 200
        mock_service.aingest.assert_awaited_once()

    async def test_azure_blob_success(
        self, client: AsyncClient, mock_config: CorpragConfig, mock_service
    ) -> None:
        async def _fake_get():
            return mock_service

        with patch("corprag.api.server._get_rag_service", new=_fake_get):
            resp = await client.post(
                "/ingest",
                json={
                    "source_type": "azure_blob",
                    "container_name": "my-container",
                    "blob_path": "docs/file.pdf",
                },
            )

        assert resp.status_code == 200

    async def test_rag_unavailable_503(
        self, client: AsyncClient, mock_config: CorpragConfig
    ) -> None:
        async def _fail():
            raise RAGServiceUnavailableError("RAG not ready")

        with patch("corprag.api.server._get_rag_service", new=_fail):
            resp = await client.post(
                "/ingest",
                json={"source_type": "local", "path": "/data/file.pdf"},
            )

        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# TestQueryEndpoint
# ---------------------------------------------------------------------------


class TestRetrieveEndpoint:
    """Test /retrieve endpoint."""

    async def test_retrieve_success(
        self, client: AsyncClient, mock_config: CorpragConfig, mock_service
    ) -> None:
        async def _fake_get():
            return mock_service

        with patch("corprag.api.server._get_rag_service", new=_fake_get):
            resp = await client.post("/retrieve", json={"query": "What is RAG?"})

        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "contexts" in body
        assert "raw" in body

    async def test_retrieve_with_custom_mode(
        self, client: AsyncClient, mock_config: CorpragConfig, mock_service
    ) -> None:
        async def _fake_get():
            return mock_service

        with patch("corprag.api.server._get_rag_service", new=_fake_get):
            resp = await client.post(
                "/retrieve",
                json={"query": "hello", "mode": "local"},
            )

        assert resp.status_code == 200
        call_kwargs = mock_service.aretrieve.call_args
        assert call_kwargs.kwargs["mode"] == "local"


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Test /health endpoint."""

    async def test_health_returns_status(
        self, client: AsyncClient, mock_config: CorpragConfig
    ) -> None:
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
        self, client: AsyncClient, mock_config: CorpragConfig, mock_service
    ) -> None:
        async def _fake_get():
            return mock_service

        with patch("corprag.api.server._get_rag_service", new=_fake_get):
            resp = await client.request(
                "DELETE",
                "/files",
                json={"filenames": ["report.pdf"]},
            )

        assert resp.status_code == 200
        mock_service.adelete_files.assert_awaited_once()

    async def test_delete_by_file_paths(
        self, client: AsyncClient, mock_config: CorpragConfig, mock_service
    ) -> None:
        async def _fake_get():
            return mock_service

        with patch("corprag.api.server._get_rag_service", new=_fake_get):
            resp = await client.request(
                "DELETE",
                "/files",
                json={"file_paths": ["/storage/report.pdf"]},
            )

        assert resp.status_code == 200
