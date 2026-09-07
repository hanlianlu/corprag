# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for corpus administration."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from dlightrag.application.corpus_admin import (
    CorpusAdmin,
    CorpusAdminSettings,
    FailedFileRow,
    FailedFileRowPage,
    FilePanelCursor,
    FilePanelPageRequest,
    FilePanelRowPage,
    IngestSpec,
    MetadataMatchRowPage,
    MetadataSearchCursor,
    MetadataSearchPageRequest,
    ProcessedFileRow,
    RedirectDownloadTarget,
    WorkspaceCatalogCursor,
    WorkspaceCatalogPageRequest,
    public_failure_diagnostic,
)
from dlightrag.engine.rag.retrieval import MetadataFilter


def _settings(
    *,
    read_only: bool = False,
    input_root: str | Path = "/tmp/inputs",
    default_workspace_id: str = "default",
) -> CorpusAdminSettings:
    return CorpusAdminSettings(
        default_workspace_id=default_workspace_id,
        default_display_name="Default",
        default_embedding_model="embedding-model",
        input_root=input_root,
        read_only=read_only,
    )


@asynccontextmanager
async def _noop_write_gate(workspace: str) -> AsyncIterator[None]:
    yield None


def _admin(
    *,
    read_only: bool = False,
    input_root: str | Path = "/tmp/inputs",
    default_workspace_id: str = "default",
    metadata_search: Any | None = None,
) -> tuple[CorpusAdmin, Any, Any, Any, Any, Any]:
    runtime = AsyncMock()
    pool = SimpleNamespace(
        acquire=AsyncMock(return_value=runtime),
        is_loaded=AsyncMock(return_value=False),
        evict=AsyncMock(),
    )
    maintenance = SimpleNamespace(
        initialize=AsyncMock(),
        register_workspace=AsyncMock(),
        list_workspace_records=AsyncMock(return_value=[]),
        list_workspace_records_page=AsyncMock(return_value=([], False)),
        workspace_exists=AsyncMock(return_value=True),
        get_workspace_record=AsyncMock(return_value=None),
        workspace_write_gate=_noop_write_gate,
    )
    jobs = SimpleNamespace(
        start_recovery=AsyncMock(),
        start_job=AsyncMock(return_value={"job_id": "job-1", "status": "queued"}),
        start_retry_failed_job=AsyncMock(return_value={"job_id": "retry-1", "status": "queued"}),
        await_job=AsyncMock(),
        get_job=AsyncMock(),
        get_active_retry_failed_job=AsyncMock(return_value=None),
        cancel_job=AsyncMock(),
        has_active_workspace_job=MagicMock(return_value=False),
        cancel_for_workspace=AsyncMock(return_value=0),
        attach_reset_result=AsyncMock(),
        close=AsyncMock(),
    )
    file_panel = SimpleNamespace(
        list_processed_files=AsyncMock(
            return_value=FilePanelRowPage(items=(), has_more=False, fetched_rows=0)
        ),
        list_failed_files=AsyncMock(
            return_value=FailedFileRowPage(items=(), has_more=False, fetched_rows=0)
        ),
    )
    metadata_store = metadata_search or SimpleNamespace(
        search_metadata_page=AsyncMock(
            return_value=MetadataMatchRowPage(
                document_ids=(),
                has_more=False,
                fetched_rows=0,
                mode="exact",
            )
        )
    )
    download = SimpleNamespace(prepare=AsyncMock())
    admin = CorpusAdmin(
        settings=_settings(
            read_only=read_only,
            input_root=input_root,
            default_workspace_id=default_workspace_id,
        ),
        pool=cast(Any, pool),
        maintenance=cast(Any, maintenance),
        file_panel=cast(Any, file_panel),
        metadata_search=cast(Any, metadata_store),
        source_download_for=MagicMock(return_value=download),
        file_panel_cursor_secret=b"corpus-file-panel-test",
        metadata_search_cursor_secret=b"corpus-metadata-search-test",
        workspace_catalog_cursor_secret=b"corpus-workspace-catalog-test",
    )
    return admin, pool, maintenance, jobs, file_panel, download


async def test_initialize_registers_default_only_for_writer() -> None:
    writer, _, writer_maintenance, _, _, _ = _admin()
    reader, _, reader_maintenance, _, _, _ = _admin(read_only=True)

    await writer.initialize()
    await reader.initialize()

    writer_maintenance.initialize.assert_awaited_once_with(validate_only=False)
    writer_maintenance.register_workspace.assert_awaited_once_with(
        workspace="default",
        display_name="Default",
        embedding_model="embedding-model",
    )
    reader_maintenance.initialize.assert_awaited_once_with(validate_only=True)
    reader_maintenance.register_workspace.assert_not_awaited()


async def test_invalid_default_workspace_fails_on_initialize_not_construction() -> None:
    admin, _, maintenance, _, _, _ = _admin(default_workspace_id="")

    with pytest.raises(ValueError, match="canonical workspace"):
        await admin.initialize()

    maintenance.initialize.assert_not_awaited()


@pytest.mark.parametrize("payload", [{"url": ""}, {"urls": [""]}])
def test_url_ingest_rejects_empty_urls(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        IngestSpec.model_validate({"source_type": "url", **payload})


async def test_workspace_catalog_serializes_rows_and_includes_default() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    maintenance.list_workspace_records.return_value = [
        {
            "workspace": "finance",
            "display_name": "Finance",
            "embedding_model": "embed-v2",
            "created_at": datetime(2026, 8, 17, tzinfo=UTC),
            "updated_at": None,
        }
    ]

    records = await admin.alist_workspace_records()

    assert records[0] == {
        "workspace": "finance",
        "display_name": "Finance",
        "embedding_model": "embed-v2",
        "created_at": "2026-08-17T00:00:00+00:00",
        "updated_at": None,
    }
    assert records[1]["workspace"] == "default"
    assert await admin.list_workspaces() == ["finance", "default"]


async def test_catalog_failure_falls_back_to_default() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    maintenance.list_workspace_records.side_effect = RuntimeError("registry unavailable")

    assert await admin.list_workspaces() == ["default"]


async def test_file_panel_and_source_download_do_not_warm_cold_runtime() -> None:
    from dlightrag.engine.rag.corpus.downloads import (
        RedirectDownloadTarget as RagRedirectDownloadTarget,
    )

    admin, pool, _, _, file_panel, download = _admin()
    file_panel.list_processed_files.return_value = FilePanelRowPage(
        items=(
            ProcessedFileRow(
                doc_id="doc-1",
                file_path="/files/doc-1.pdf",
                updated_at=datetime(2026, 3, 4, 5, 6, 7),
            ),
        ),
        has_more=False,
        fetched_rows=1,
    )
    target = RagRedirectDownloadTarget(url="https://cdn.example.com/report.pdf")
    download.prepare.return_value = target

    snapshot = await admin.file_panel_snapshot("finance")
    prepared = await admin.prepare_source_download("finance", "doc-1")

    assert snapshot == {
        "files": [
            {
                "doc_id": "doc-1",
                "file_path": "/files/doc-1.pdf",
                "status": "processed",
                "updated_at": "2026-03-04T05:06:07.000000",
            }
        ],
        "next_cursor": None,
        "fetched_rows": 1,
    }
    assert isinstance(prepared, RedirectDownloadTarget)
    assert prepared.url == "https://cdn.example.com/report.pdf"
    pool.acquire.assert_not_awaited()


async def test_file_panel_snapshot_derives_bounded_cursor_and_rejects_foreign_cursor() -> None:
    admin, pool, _, _, file_panel, _ = _admin()
    timestamp = datetime(2026, 3, 4, 5, 6, 7, 123456)
    file_panel.list_processed_files.return_value = FilePanelRowPage(
        items=(
            ProcessedFileRow(doc_id="doc-a", file_path="/a", updated_at=timestamp),
            ProcessedFileRow(doc_id="doc-b", file_path="/b", updated_at=timestamp),
        ),
        has_more=True,
        fetched_rows=3,
    )

    snapshot = await admin.file_panel_snapshot(
        "finance",
        page=FilePanelPageRequest(limit=2),
    )

    assert snapshot["next_cursor"] == FilePanelCursor(
        workspace="finance",
        updated_at=timestamp,
        doc_id="doc-b",
    )
    assert snapshot["fetched_rows"] == 3
    file_panel.list_processed_files.assert_awaited_once_with(
        "finance",
        page=FilePanelPageRequest(limit=2),
    )
    pool.acquire.assert_not_awaited()

    file_panel.list_processed_files.reset_mock()
    with pytest.raises(ValueError, match="another workspace"):
        await admin.file_panel_snapshot(
            "finance",
            page=FilePanelPageRequest(
                cursor=FilePanelCursor(
                    workspace="legal",
                    updated_at=None,
                    doc_id="doc-z",
                )
            ),
        )
    file_panel.list_processed_files.assert_not_awaited()


async def test_failed_file_snapshot_is_bounded_and_projects_public_diagnostics() -> None:
    admin, pool, _, _, file_panel, _ = _admin()
    timestamp = datetime(2026, 3, 4, 5, 6, 7, 123456)
    file_panel.list_failed_files.return_value = FailedFileRowPage(
        items=(
            FailedFileRow(
                doc_id="doc-failed",
                file_path="/failed.pdf",
                error=(
                    "parser failed at /srv/private/report.pdf "
                    "https://user:password@example.test/file?token=secret " + "x" * 700
                ),
                updated_at=timestamp,
            ),
        ),
        has_more=True,
        fetched_rows=2,
    )

    snapshot = await admin.failed_file_snapshot(
        "finance",
        page=FilePanelPageRequest(limit=1),
    )

    assert snapshot == {
        "failed": [
            {
                "doc_id": "doc-failed",
                "file_path": "/failed.pdf",
                "error": snapshot["failed"][0]["error"],
                "updated_at": "2026-03-04T05:06:07.123456",
            }
        ],
        "next_cursor": FilePanelCursor(
            workspace="finance",
            updated_at=timestamp,
            doc_id="doc-failed",
            view="failed",
        ),
        "fetched_rows": 2,
    }
    diagnostic = snapshot["failed"][0]["error"]
    assert diagnostic == "Document processing failed."
    assert len(diagnostic) <= 512
    pool.acquire.assert_not_awaited()


@pytest.mark.parametrize(
    "private_value",
    [
        "Authorization: Bearer sk-live-123",
        "Authorization:Bearer private-token",
        r'headers={"Authorization":"Bearer sk-ant-api03-private\\\",continued"}',
        "Proxy-Authorization: Basic dXNlcjpwYXNz",
        "Authorization: Session private-value",
        "Cookie: session=private-value",
        "Cookie: PHPSESSID=private-session; sid=other-private",
        "Cookie: theme=light; session-id=private-value; locale=en",
        "headers={'Authorization': 'Basic dXNlcjpwYXNz'}",
        'headers={"Proxy-Authorization": "Basic cHJveHk6c2VjcmV0"}',
        "headers={'Cookie': 'session=private-value'}",
        'headers={"Set-Cookie": "session=private-value; HttpOnly"}',
        "client_secret=top-secret",
        "access_token=abc",
        "refresh-token='refresh private value'",
        'OPENAI_API_KEY="sk-live-provider"',
        "AWS_SECRET_ACCESS_KEY=provider-private",
        'password="correct horse"',
        "postgresql://user:password@db.internal/app",
        "urn:customer:private-record",
        "archive/private/report.pdf",
        "file:///srv/private/report.pdf",
        "/home dir/alice/report.pdf",
        r"C:\\Users\\Alice Smith\\report.pdf",
        r"客户\报告.pdf",
        r"\\server\share\private report.pdf",
        "../private/report.pdf",
        "pass\u200bword=secret\x00\u202e",
    ],
)
def test_public_failure_diagnostic_redacts_common_private_values(private_value: str) -> None:
    diagnostic = public_failure_diagnostic(f"parser failed: {private_value}")

    assert diagnostic == "Document processing failed."
    assert len(diagnostic) <= 512
    assert public_failure_diagnostic(diagnostic) == diagnostic


def test_public_failure_diagnostic_nfkc_normalizes_before_redaction() -> None:
    diagnostic = public_failure_diagnostic(
        'headers={"Ａｕｔｈｏｒｉｚａｔｉｏｎ": "Ｂａｓｉｃ dXNlcjpwYXNz"}'
    )

    assert diagnostic == "Document processing failed."
    assert public_failure_diagnostic(diagnostic) == diagnostic


def test_public_failure_diagnostic_allows_only_known_application_messages() -> None:
    assert public_failure_diagnostic("source metadata unavailable") == (
        "Source metadata unavailable."
    )
    assert public_failure_diagnostic("retry ingestion failed") == "Retry ingestion failed."
    assert public_failure_diagnostic("") == ""


async def test_workspace_exists_uses_default_fast_path_and_bounded_maintenance_lookup() -> None:
    admin, pool, maintenance, _, _, _ = _admin()
    maintenance.workspace_exists.side_effect = [True, False]

    assert await admin.workspace_exists("default") is True
    assert await admin.workspace_exists("finance") is True
    assert await admin.workspace_exists("legal") is False

    assert [item.args for item in maintenance.workspace_exists.await_args_list] == [
        ("finance",),
        ("legal",),
    ]
    maintenance.workspace_exists.side_effect = RuntimeError("registry unavailable")
    with pytest.raises(RuntimeError, match="registry unavailable"):
        await admin.workspace_exists("research")
    pool.acquire.assert_not_awaited()


# ---------------------------------------------------------------------------
# Metadata search — bounded cold path
# ---------------------------------------------------------------------------


async def test_search_metadata_never_warms_a_runtime_and_derives_next_cursor() -> None:
    admin, pool, _, _, _, _ = _admin(
        metadata_search=SimpleNamespace(
            search_metadata_page=AsyncMock(
                return_value=MetadataMatchRowPage(
                    document_ids=("doc-b", "doc-c"),
                    has_more=True,
                    fetched_rows=3,
                    mode="contains",
                )
            )
        )
    )

    page = await admin.search_metadata("finance", MetadataFilter(filename="Quarterly"))

    assert page.document_ids == ("doc-b", "doc-c")
    assert page.fetched_rows == 3
    assert page.next_cursor == MetadataSearchCursor(
        workspace="finance",
        after_doc_id="doc-c",
        mode="contains",
    )
    pool.acquire.assert_not_awaited()


async def test_search_metadata_has_no_cursor_when_the_page_is_exhausted() -> None:
    store = SimpleNamespace(
        search_metadata_page=AsyncMock(
            return_value=MetadataMatchRowPage(
                document_ids=("doc-z",),
                has_more=False,
                fetched_rows=1,
                mode="exact",
            )
        )
    )
    admin, pool, _, _, _, _ = _admin(metadata_search=store)

    page = await admin.search_metadata(
        "finance",
        MetadataFilter(filename="Report"),
        page=MetadataSearchPageRequest(limit=25),
    )

    assert page.next_cursor is None
    called = store.search_metadata_page.await_args
    assert called.kwargs["page"].limit == 25
    assert called.args[0] == "finance"
    pool.acquire.assert_not_awaited()


async def test_search_metadata_rejects_cross_workspace_cursor_before_storage() -> None:
    store = SimpleNamespace(search_metadata_page=AsyncMock())
    admin, _, _, _, _, _ = _admin(metadata_search=store)

    with pytest.raises(ValueError, match="another workspace"):
        await admin.search_metadata(
            "finance",
            MetadataFilter(filename="Report"),
            page=MetadataSearchPageRequest(
                cursor=MetadataSearchCursor(
                    workspace="legal",
                    after_doc_id="doc-1",
                    mode="exact",
                )
            ),
        )

    store.search_metadata_page.assert_not_awaited()


async def test_workspace_catalog_page_delegates_and_derives_next_cursor() -> None:
    admin, pool, maintenance, _, _, _ = _admin()
    maintenance.list_workspace_records_page = AsyncMock(
        return_value=(
            [
                {
                    "workspace": "finance",
                    "display_name": "Finance",
                    "embedding_model": "voyage-multimodal-3.5",
                    "created_at": None,
                    "updated_at": None,
                },
            ],
            True,
        )
    )

    page = await admin.list_workspace_records_page(
        page=WorkspaceCatalogPageRequest(
            limit=50,
            cursor=WorkspaceCatalogCursor(after_workspace="default"),
        )
    )

    maintenance.list_workspace_records_page.assert_awaited_once_with(
        after_workspace="default",
        limit=50,
    )
    assert [item["workspace"] for item in page.items] == ["finance"]
    assert page.next_cursor == WorkspaceCatalogCursor(after_workspace="finance")
    assert page.fetched_rows == 2
    pool.acquire.assert_not_awaited()


async def test_workspace_catalog_page_rejects_empty_page_with_continuation() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    maintenance.list_workspace_records_page = AsyncMock(return_value=([], True))

    with pytest.raises(RuntimeError, match="empty page"):
        await admin.list_workspace_records_page()


async def test_workspace_catalog_full_reads_remain_full() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    maintenance.list_workspace_records.return_value = [
        {
            "workspace": "default",
            "display_name": "Default",
            "embedding_model": "voyage-multimodal-3.5",
            "created_at": None,
            "updated_at": None,
        }
    ]

    records = await admin.alist_workspace_records()
    workspaces = await admin.list_workspaces()

    assert [record["workspace"] for record in records] == ["default"]
    assert workspaces == ["default"]
    maintenance.list_workspace_records_page.assert_not_awaited()


async def test_workspace_catalog_cursor_codec_is_exposed() -> None:
    from dlightrag.application.corpus_admin import WorkspaceCatalogCursorCodec

    admin, _, _, _, _, _ = _admin()

    assert isinstance(admin.workspace_catalog_cursor_codec, WorkspaceCatalogCursorCodec)


# ---------------------------------------------------------------------------
# Commit 3: promotion fence gates and storage status
# ---------------------------------------------------------------------------


async def test_storage_status_projects_registry_facts_bounded() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    now = datetime.now(UTC)
    maintenance.get_workspace_record.return_value = {
        "workspace": "finance",
        "storage_tier": "hot",
        "promotion_state": "failed",
        "ingested_docs_total": 42,
        "ingested_chunks_total": 900,
        "promotion_retry_count": 3,
        "promotion_last_error": "promotion failed: copy verification failed",
        "promotion_next_retry_at": now,
        "write_fence_owner": None,
        "write_fence_until": None,
    }

    status = await admin.get_workspace_storage_status("finance")
    assert status is not None

    assert status == {
        "workspace": "finance",
        "storage_tier": "hot",
        "promotion_state": "failed",
        "ingested_docs_total": 42,
        "ingested_chunks_total": 900,
        "promotion_retry_count": 3,
        "promotion_last_error": "promotion failed: copy verification failed",
        "promotion_next_retry_at": now.isoformat(),
        "write_fenced": False,
        "retry_after_seconds": None,
    }


async def test_storage_status_reports_active_fence_retry_window() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    maintenance.get_workspace_record.return_value = {
        "workspace": "finance",
        "storage_tier": "shared",
        "promotion_state": "promoting",
        "ingested_docs_total": 0,
        "ingested_chunks_total": 0,
        "promotion_retry_count": 0,
        "promotion_last_error": None,
        "promotion_next_retry_at": None,
        "write_fence_owner": "worker#1",
        "write_fence_until": datetime.now(UTC) + timedelta(seconds=30),
    }

    status = await admin.get_workspace_storage_status("finance")
    assert status is not None

    assert status["write_fenced"] is True
    assert 25.0 <= status["retry_after_seconds"] <= 30.1
    assert status["storage_tier"] == "shared"
    assert status["promotion_state"] == "promoting"


async def test_storage_status_treats_stale_promoting_as_conservatively_fenced() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    maintenance.get_workspace_record.return_value = {
        "workspace": "finance",
        "storage_tier": "shared",
        "promotion_state": "promoting",
        "ingested_docs_total": 0,
        "ingested_chunks_total": 0,
        "promotion_retry_count": 0,
        "promotion_last_error": None,
        "promotion_next_retry_at": None,
        "write_fence_owner": "dead-worker#1",
        "write_fence_until": datetime.now(UTC) - timedelta(seconds=60),  # expired
    }

    status = await admin.get_workspace_storage_status("finance")

    assert status is not None
    # A crashed worker's committed exclusion proofs keep the workspace
    # conservatively write-fenced with a small bounded retry window.
    assert status["write_fenced"] is True
    assert status["retry_after_seconds"] == 5.0
    assert status["promotion_state"] == "promoting"


async def test_start_promotion_worker_starts_only_on_writers() -> None:
    writer, _, _, _, _, _ = _admin()
    reader, _, _, _, _, _ = _admin(read_only=True)

    writer._promotion_worker = cast(Any, SimpleNamespace(start=MagicMock()))
    writer.start_promotion_worker()
    writer._promotion_worker.start.assert_called_once_with()  # type: ignore[union-attr]

    reader._promotion_worker = cast(Any, SimpleNamespace(start=MagicMock()))
    reader.start_promotion_worker()
    reader._promotion_worker.start.assert_not_called()  # type: ignore[union-attr]
