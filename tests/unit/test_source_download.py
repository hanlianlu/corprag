from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from dlightrag.application.config import DlightragConfig
from dlightrag.application.settings import rag_settings
from dlightrag.engine.rag.corpus.downloads import (
    LocalDownloadTarget,
    RedirectDownloadTarget,
    SourceDownloadInvalidError,
    SourceDownloadNotFoundError,
    SourceDownloadService,
    SourceDownloadUnavailableError,
)
from dlightrag.engine.rag.corpus.sources.aws_s3 import S3CredentialsUnavailable
from dlightrag.engine.rag.retrieval.metadata_fields import (
    INGEST_FINALIZATION_COMPLETE_FIELD,
)
from tests.config_helpers import mutate_config


def _service(
    test_config: DlightragConfig,
    metadata_index: AsyncMock,
    *,
    workspace: str = "default",
) -> SourceDownloadService:
    metadata = metadata_index.get.return_value
    if isinstance(metadata, dict):
        metadata.setdefault(INGEST_FINALIZATION_COMPLETE_FIELD, True)
    return SourceDownloadService(
        settings=rag_settings(test_config),
        metadata_index=metadata_index,
        workspace_id=workspace,
    )


async def test_local_download_uses_exact_metadata_document(tmp_path: Path, test_config) -> None:
    source = test_config.input_dir_path / "default" / "notes.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("notes", encoding="utf-8")
    metadata_index = AsyncMock()
    metadata_index.get.return_value = {"download_locator": str(source)}

    target = await _service(test_config, metadata_index).prepare("doc-notes")

    assert target == LocalDownloadTarget(
        path=source.resolve(),
        media_type="text/markdown",
        filename="notes.md",
    )
    metadata_index.get.assert_awaited_once_with("doc-notes")


async def test_local_download_rejects_path_outside_workspace(tmp_path: Path, test_config) -> None:
    outside = tmp_path / "secret.md"
    outside.write_text("secret", encoding="utf-8")
    metadata_index = AsyncMock()
    metadata_index.get.return_value = {"download_locator": str(outside)}

    with pytest.raises(SourceDownloadInvalidError):
        await _service(test_config, metadata_index).prepare("doc-secret")


async def test_missing_contained_local_file_is_not_found(test_config) -> None:
    missing = test_config.input_dir_path / "default" / "missing.md"
    metadata_index = AsyncMock()
    metadata_index.get.return_value = {"download_locator": str(missing)}

    with pytest.raises(SourceDownloadNotFoundError):
        await _service(test_config, metadata_index).prepare("doc-missing-file")


async def test_local_download_repairs_known_lightrag_archive_transition(test_config) -> None:
    original = test_config.input_dir_path / "default" / "notes.md"
    archived = original.parent / "__parsed__" / original.name
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_text("notes", encoding="utf-8")
    metadata_index = AsyncMock()
    metadata_index.get.return_value = {
        "download_locator": str(original),
        "file_path": str(original),
    }

    target = await _service(test_config, metadata_index).prepare("doc-notes")

    assert target == LocalDownloadTarget(
        path=archived.resolve(),
        media_type="text/markdown",
        filename="notes.md",
    )
    metadata_index.upsert.assert_awaited_once_with(
        "doc-notes",
        {
            "download_locator": str(archived.resolve()),
            "file_path": str(archived.resolve()),
        },
    )


@pytest.mark.parametrize("marker", [False, None])
async def test_unpublished_document_is_not_found(test_config, marker: object) -> None:
    metadata_index = AsyncMock()
    metadata_index.get.return_value = {
        "download_locator": "https://cdn.example.com/report.pdf",
        INGEST_FINALIZATION_COMPLETE_FIELD: marker,
    }

    with pytest.raises(SourceDownloadNotFoundError):
        await _service(test_config, metadata_index).prepare("doc-hidden")


async def test_missing_finalization_marker_is_not_found(test_config) -> None:
    metadata_index = AsyncMock()
    metadata_index.get.return_value = {
        "download_locator": "https://cdn.example.com/report.pdf",
    }
    service = SourceDownloadService(
        settings=rag_settings(test_config),
        metadata_index=metadata_index,
        workspace_id="default",
    )

    with pytest.raises(SourceDownloadNotFoundError):
        await service.prepare("doc-hidden")


async def test_missing_document_is_not_found(test_config) -> None:
    metadata_index = AsyncMock()
    metadata_index.get.return_value = None

    with pytest.raises(SourceDownloadNotFoundError):
        await _service(test_config, metadata_index).prepare("doc-missing")


async def test_queryless_https_returns_canonical_redirect(test_config) -> None:
    metadata_index = AsyncMock()
    metadata_index.get.return_value = {"download_locator": "HTTPS://cdn.example.com/report.pdf"}

    target = await _service(test_config, metadata_index).prepare("doc-report")

    assert target == RedirectDownloadTarget(url="https://cdn.example.com/report.pdf")


async def test_signed_https_locator_is_invalid(test_config) -> None:
    metadata_index = AsyncMock()
    metadata_index.get.return_value = {
        "download_locator": "https://cdn.example.com/report.pdf?sig=secret"
    }

    with pytest.raises(SourceDownloadInvalidError):
        await _service(test_config, metadata_index).prepare("doc-report")


async def test_azure_locator_returns_signed_redirect(test_config) -> None:
    mutate_config(
        test_config, "corpus.sources.blob_connection_string", "AccountName=acct;AccountKey=dGVzdA=="
    )
    metadata_index = AsyncMock()
    metadata_index.get.return_value = {"download_locator": "azure://container/report.pdf"}

    with patch(
        "dlightrag.engine.rag.corpus.downloads.generate_azure_sas_url",
        return_value="https://acct.blob.core.windows.net/container/report.pdf?sig=x",
    ) as signer:
        target = await _service(test_config, metadata_index).prepare("doc-report")

    assert target == RedirectDownloadTarget(
        url="https://acct.blob.core.windows.net/container/report.pdf?sig=x"
    )
    signer.assert_called_once_with(
        connection_string=test_config.corpus.sources.blob_connection_string,
        raw_path="azure://container/report.pdf",
        expiry_seconds=test_config.corpus.sources.azure_sas_expiry,
    )


async def test_s3_locator_returns_signed_redirect(test_config) -> None:
    metadata_index = AsyncMock()
    metadata_index.get.return_value = {"download_locator": "s3://bucket/report.pdf"}

    with patch(
        "dlightrag.engine.rag.corpus.downloads.generate_s3_presigned_url",
        new_callable=AsyncMock,
        return_value="https://bucket.s3.example/report.pdf?sig=x",
    ) as signer:
        target = await _service(test_config, metadata_index).prepare("doc-report")

    assert target == RedirectDownloadTarget(url="https://bucket.s3.example/report.pdf?sig=x")
    signer.assert_awaited_once_with(
        raw_path="s3://bucket/report.pdf",
        expiry_seconds=test_config.corpus.sources.s3_presign_expiry,
        region=test_config.corpus.sources.s3_region,
    )


async def test_s3_credentials_failure_is_unavailable(test_config) -> None:
    metadata_index = AsyncMock()
    metadata_index.get.return_value = {"download_locator": "s3://bucket/report.pdf"}

    with (
        patch(
            "dlightrag.engine.rag.corpus.downloads.generate_s3_presigned_url",
            new_callable=AsyncMock,
            side_effect=S3CredentialsUnavailable("secret provider detail"),
        ),
        pytest.raises(SourceDownloadUnavailableError, match="S3 credentials not configured"),
    ):
        await _service(test_config, metadata_index).prepare("doc-report")
