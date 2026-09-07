"""Smoke tests for web routes."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.application.corpus_admin import FilePanelCursorCodec
from tests.unit.conftest import answer_capability_view


@pytest.fixture()
def app(test_config):
    from dlightrag.adapters.http.server import create_app

    assert test_config is not None
    real_app = create_app(include_web_app=True)

    mock_application = MagicMock()
    mock_application.config = test_config
    mock_application.corpora.file_panel_cursor_codec = FilePanelCursorCodec(b"route-test")
    mock_application.corpora.workspace_exists = AsyncMock(return_value=True)
    mock_application.corpora.file_panel_snapshot = AsyncMock(
        return_value={
            "files": [
                {"file_path": "/data/report.pdf", "file_name": "report.pdf"},
                {"file_path": "/data/analysis.xlsx", "file_name": "analysis.xlsx"},
            ],
            "next_cursor": None,
            "fetched_rows": 2,
        }
    )
    mock_application.corpora.list_workspaces = AsyncMock(return_value=["default", "finance"])
    mock_application.corpora.alist_workspace_records = AsyncMock(
        return_value=[
            {
                "workspace": "default",
                "display_name": "Default",
                "embedding_model": "voyage-multimodal-3.5",
            },
            {
                "workspace": "finance",
                "display_name": "Finance",
                "embedding_model": "voyage-multimodal-3.5",
            },
        ]
    )
    real_app.state.application = mock_application

    capability_view = answer_capability_view()
    mock_application.answers.capabilities = capability_view.read

    return real_app


@pytest.fixture()
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        follow_redirects=False,
    ) as c:
        yield c


async def test_index_page(client):
    resp = await client.get("/web/")
    assert resp.status_code == 200
    assert "DlightRAG" in resp.text
    assert "<dl-app>" in resp.text
    assert "/static/app/assets/app-" in resp.text


async def test_file_list(client):
    resp = await client.get("/web/api/files")
    assert resp.status_code == 200
    assert [item["file_name"] for item in resp.json()["files"]] == [
        "report.pdf",
        "analysis.xlsx",
    ]


async def test_web_workspaces_page_roundtrips_an_opaque_cursor(app, client) -> None:
    from dlightrag.application.corpus_admin import (
        WorkspaceCatalogCursor,
        WorkspaceCatalogCursorCodec,
        WorkspaceCatalogPage,
    )

    application = app.state.application
    codec = WorkspaceCatalogCursorCodec(b"web-route-test")
    application.corpora.workspace_catalog_cursor_codec = codec
    application.corpora.list_workspace_records_page = AsyncMock(
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

    resp = await client.get("/web/api/workspaces")

    assert resp.status_code == 200
    body = resp.json()
    assert body["workspaces"] == [
        {
            "workspace": "finance",
            "display_name": "Finance",
            "embedding_model": "voyage-multimodal-3.5",
        }
    ]
    assert body["next_cursor"] is not None

    second = await client.get("/web/api/workspaces", params={"cursor": body["next_cursor"]})
    assert second.status_code == 200
    assert second.json()["workspaces"][0]["workspace"] == "finance"


async def test_web_workspaces_page_rejects_tampered_cursor_before_storage(app, client) -> None:
    from dlightrag.application.corpus_admin import WorkspaceCatalogCursorCodec

    application = app.state.application
    application.corpora.workspace_catalog_cursor_codec = WorkspaceCatalogCursorCodec(
        b"web-route-test"
    )
    application.corpora.list_workspace_records_page = AsyncMock()

    resp = await client.get("/web/api/workspaces", params={"cursor": "AAAA.tampered"})

    assert resp.status_code == 422
    application.corpora.list_workspace_records_page.assert_not_awaited()


async def test_bootstrap_bounds_the_visible_array_but_keeps_full_authorization_inputs(
    app, client
) -> None:
    from dlightrag.application.corpus_admin import (
        WorkspaceCatalogCursor,
        WorkspaceCatalogCursorCodec,
        WorkspaceCatalogPage,
    )

    application = app.state.application
    application.corpora.workspace_catalog_cursor_codec = WorkspaceCatalogCursorCodec(
        b"web-route-test"
    )
    application.corpora.list_workspace_records_page = AsyncMock(
        return_value=WorkspaceCatalogPage(
            items=(
                {
                    "workspace": "default",
                    "display_name": "Default",
                    "embedding_model": "voyage-multimodal-3.5",
                    "created_at": None,
                    "updated_at": None,
                },
            ),
            next_cursor=WorkspaceCatalogCursor(after_workspace="default"),
            fetched_rows=2,
        )
    )
    application.corpora.alist_workspace_records = AsyncMock(
        return_value=[
            {
                "workspace": "default",
                "display_name": "Default",
                "embedding_model": "voyage-multimodal-3.5",
            },
            {
                "workspace": "finance",
                "display_name": "Finance",
                "embedding_model": "voyage-multimodal-3.5",
            },
            {
                "workspace": "research",
                "display_name": "Research",
                "embedding_model": "voyage-multimodal-3.5",
            },
        ]
    )

    resp = await client.get("/web/api/bootstrap")

    assert resp.status_code == 200
    body = resp.json()
    assert [item["workspace"] for item in body["workspaces"]] == ["default"]
    assert body["workspaces_next_cursor"] is not None
    # The full catalog still powers the authorization inputs: without cookies,
    # every visible workspace is active, not just the bounded display page.
    assert body["active_workspaces"] == ["default", "finance", "research"]
    assert body["known_workspaces"] == ["default", "finance", "research"]
    assert body["primary_workspace"] == "default"
