# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Azure Blob Storage data source.

Requires: pip install corprag[azure]
"""

from __future__ import annotations

import os


def _require_azure() -> None:
    try:
        from azure.storage.blob import BlobServiceClient  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Azure support requires: pip install corprag[azure]"
        ) from e


class AzureBlobDataSource:
    """Azure Blob Storage adapter (sync + async)."""

    def __init__(
        self,
        connection_string: str | None = None,
        container_name: str | None = None,
    ) -> None:
        _require_azure()

        from azure.storage.blob import BlobServiceClient, ContainerClient

        self.connection_string = connection_string or os.getenv(
            "BLOB_CONNECTION_STRING", ""
        )
        self.container_name = container_name or os.getenv("BLOB_CONTAINER_NAME", "")

        self.blob_service = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        self.container_client: ContainerClient = self.blob_service.get_container_client(
            self.container_name
        )
        # Lazy async clients
        self._async_blob_service = None
        self._async_container_client = None

    def list_documents(self, prefix: str | None = None) -> list[str]:
        """List all blob names in container (sync)."""
        blobs = self.container_client.list_blobs(name_starts_with=prefix)
        return [blob.name for blob in blobs]

    async def alist_documents(self, prefix: str | None = None) -> list[str]:
        """List all blob names in container (async)."""
        client = await self._get_async_container_client()
        blobs = client.list_blobs(name_starts_with=prefix)
        return [blob.name async for blob in blobs]

    async def _get_async_container_client(self):
        """Lazy init async container client."""
        if self._async_container_client is None:
            from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient

            self._async_blob_service = AsyncBlobServiceClient.from_connection_string(
                self.connection_string
            )
            self._async_container_client = (
                self._async_blob_service.get_container_client(self.container_name)
            )
        return self._async_container_client

    async def aload_document(self, doc_id: str) -> bytes:
        """Async download blob content as bytes."""
        client = await self._get_async_container_client()
        blob_client = client.get_blob_client(doc_id)
        stream = await blob_client.download_blob()
        return await stream.readall()

    async def aupload_document(
        self,
        blob_name: str,
        content: bytes,
        content_type: str | None = None,
    ) -> str:
        """Async upload content to blob storage.

        Returns:
            Full blob URL in format: azure://{container}/{blob_name}
        """
        from azure.storage.blob import ContentSettings

        client = await self._get_async_container_client()
        blob_client = client.get_blob_client(blob_name)

        content_settings = None
        if content_type:
            content_settings = ContentSettings(content_type=content_type)

        await blob_client.upload_blob(
            content,
            overwrite=True,
            content_settings=content_settings,
        )

        return f"azure://{self.container_name}/{blob_name}"

    async def aclose(self) -> None:
        """Close async clients."""
        if self._async_container_client:
            await self._async_container_client.close()
            self._async_container_client = None
        if self._async_blob_service:
            await self._async_blob_service.close()
            self._async_blob_service = None


__all__ = ["AzureBlobDataSource"]
