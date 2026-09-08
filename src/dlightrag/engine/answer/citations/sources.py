# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Project Answer source and context identities for authenticated readers."""

import logging
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from dlightrag.engine.ai.telemetry import safe_log_text
from dlightrag.engine.answer.citations.contracts import SourceReference, SourceReferencePayload
from dlightrag.engine.rag.retrieval import RetrievalContexts

logger = logging.getLogger(__name__)

_PUBLIC_CHUNK_KEYS = (
    "reference_id",
    "file_path",
    "content",
    "page_number",
    "image_url",
    "thumbnail_url",
    "image_mime_type",
    "relevance_score",
    "metadata",
    "_workspace",
)
_REQUEST_OWNED_WORKSPACES = frozenset({"__attachment__", "__web_search__"})


class SourceDownloadInvariantError(RuntimeError):
    """Raised when an Answer source cannot be projected safely."""


class SourceDownloadLinkBuilder:
    """Project a workspace-scoped document ID into one HTTP adapter URL."""

    def __init__(self, base_url: str = "/files/raw") -> None:
        self._base_url = base_url.rstrip("/")

    def resolve(self, document_id: str, *, workspace: str) -> str | None:
        if not document_id or not workspace:
            return None
        return (
            f"{self._base_url}/{quote(document_id, safe='')}?workspace={quote(workspace, safe='')}"
        )


def can_project_workspace_visual(workspace: str | None, allowed: set[str] | None) -> bool:
    """Allow trusted calls and request-owned evidence; otherwise require workspace ACL."""
    return allowed is None or bool(
        workspace and (workspace in _REQUEST_OWNED_WORKSPACES or workspace in allowed)
    )


def project_contexts_for_client(
    contexts: RetrievalContexts,
    *,
    image_url_prefix: str | None = "/images",
    visual_workspaces: set[str] | None = None,
) -> RetrievalContexts:
    """Return client-safe contexts without inline image bytes."""
    chunks = [
        chunk
        for item in contexts.get("chunks", [])
        if (
            chunk := _project_chunk_context(
                item,
                image_url_prefix=image_url_prefix,
                visual_workspaces=visual_workspaces,
            )
        )
        is not None
    ]
    return {
        "chunks": chunks,
        "entities": [dict(item) for item in contexts.get("entities", [])],
        "relationships": [dict(item) for item in contexts.get("relationships", [])],
    }


def _project_chunk_context(
    item: dict[str, Any],
    *,
    image_url_prefix: str | None,
    visual_workspaces: set[str] | None,
) -> dict[str, Any] | None:
    row = dict(item)
    chunk_id = row.get("chunk_id") or row.get("id")
    if chunk_id is None:
        return None

    payload = {key: row[key] for key in _PUBLIC_CHUNK_KEYS if row.get(key) is not None}
    payload = {
        **{
            "chunk_id": str(chunk_id),
            "reference_id": "",
            "file_path": "",
            "content": "",
        },
        **payload,
    }
    payload["file_path"] = _display_file_name(payload["file_path"])
    if "metadata" in payload:
        metadata = payload["metadata"]
        if isinstance(metadata, Mapping):
            public_metadata = dict(metadata)
            public_metadata.pop("source_uri", None)
            public_metadata.pop("source_download_locator", None)
            payload["metadata"] = public_metadata
        else:
            payload.pop("metadata")
    can_read_visual = can_project_workspace_visual(row.get("_workspace"), visual_workspaces)
    if not can_read_visual:
        payload.pop("image_url", None)
        payload.pop("thumbnail_url", None)
        payload.pop("image_mime_type", None)
    if (
        can_read_visual
        and image_url_prefix
        and row.get("_workspace")
        and (row.get("image_data") or _is_visual_chunk(row))
    ):
        base_path = (
            f"{image_url_prefix.rstrip('/')}/"
            f"{quote(str(row['_workspace']), safe='')}/"
            f"{quote(str(chunk_id), safe='')}"
        )
        payload.setdefault("image_url", f"{base_path}?size=full")
        payload.setdefault("thumbnail_url", f"{base_path}?size=thumb")
    return payload


def _is_visual_chunk(row: dict[str, Any]) -> bool:
    sidecar = row.get("sidecar")
    return bool(row.get("_has_visual_asset")) or (
        isinstance(sidecar, dict) and sidecar.get("type") == "drawing"
    )


def _display_file_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    candidate = value
    if "://" in value:
        try:
            candidate = unquote(urlsplit(value).path)
        except ValueError:
            candidate = unquote(value.split("?", 1)[0].split("#", 1)[0])
    if "\\" in candidate:
        return PureWindowsPath(candidate).name
    return Path(candidate.rstrip("/")).name


def project_source_payloads(
    sources: list[SourceReference],
    *,
    resolver: SourceDownloadLinkBuilder | None,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
) -> list[SourceReferencePayload]:
    """Convert internal sources into the strict public source contract."""
    projected: list[SourceReferencePayload] = []
    for source in sources:
        safe_source_id = safe_log_text(source.id)
        download_url = None
        can_download = (
            downloadable_workspaces is None or source.workspace in downloadable_workspaces
        )
        if resolver is not None and can_download:
            if not source.document_id:
                logger.info(
                    "source_download_projection_outcome",
                    extra={"outcome": "invalid", "source_id": safe_source_id},
                )
                raise SourceDownloadInvariantError(
                    f"Could not project source document for source {safe_source_id}"
                ) from None
            try:
                download_url = resolver.resolve(
                    source.document_id,
                    workspace=source.workspace,
                )
            except Exception:
                download_url = None
            if not download_url:
                logger.info(
                    "source_download_projection_outcome",
                    extra={"outcome": "invalid", "source_id": safe_source_id},
                )
                raise SourceDownloadInvariantError(
                    f"Could not project download locator for source {safe_source_id}"
                ) from None
            logger.info(
                "source_download_projection_outcome",
                extra={"outcome": "resolved", "source_id": safe_source_id},
            )
        elif resolver is not None:
            logger.info(
                "source_download_projection_outcome",
                extra={"outcome": "unauthorized", "source_id": safe_source_id},
            )
        chunks = source.chunks
        if chunks and not can_project_workspace_visual(source.workspace, visual_workspaces):
            chunks = [
                chunk.model_copy(update={"image_url": None, "thumbnail_url": None})
                for chunk in chunks
            ]
        projected.append(
            SourceReferencePayload(
                id=source.id,
                title=source.title,
                type=source.type,
                source_uri=source.source_uri,
                download_url=download_url,
                cited_chunk_ids=source.cited_chunk_ids,
                chunks=chunks,
            )
        )
    return projected


__all__ = [
    "SourceDownloadInvariantError",
    "SourceDownloadLinkBuilder",
    "can_project_workspace_visual",
    "project_contexts_for_client",
    "project_source_payloads",
]
