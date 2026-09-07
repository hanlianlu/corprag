"""Shared utility for building document-level SourceReferences from contexts.

Groups chunks by reference_id into SourceReference objects with ChunkSnippets.
Called by both web routes and API server after retrieval.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dlightrag.application.answer_runs.citations import ChunkSnippet, SourceReference
from dlightrag.engine.ai.telemetry import safe_log_text
from dlightrag.engine.rag.retrieval import RetrievalContexts

from .indexer import CitationIndexer
from .utils import filter_content_for_display


class SourceBuildInvariantError(RuntimeError):
    """Raised when retrieval contexts cannot form a durable source reference."""


def build_sources(
    contexts: RetrievalContexts,
    *,
    indexer: CitationIndexer | None = None,
    enriched_chunks: list[dict[str, Any]] | None = None,
    image_url_prefix: str | None = "/images",
    default_workspace: str | None = None,
) -> list[SourceReference]:
    """Group chunks by reference_id into document-level SourceReferences.

    Args:
        contexts: Retrieval contexts dict with "chunks" key.
    Returns:
        List of SourceReference, ordered by first chunk appearance.
        Chunks within each source follow citation-index order.
    """
    return build_sources_from_chunks(
        contexts.get("chunks", []),
        indexer=indexer,
        enriched_chunks=enriched_chunks,
        image_url_prefix=image_url_prefix,
        default_workspace=default_workspace,
    )


def build_sources_from_chunks(
    chunks: list[dict[str, Any]],
    *,
    indexer: CitationIndexer | None = None,
    enriched_chunks: list[dict[str, Any]] | None = None,
    cited_chunks: dict[str, list[str]] | None = None,
    source_catalog: list[SourceReference] | None = None,
    image_url_prefix: str | None = "/images",
    default_workspace: str | None = None,
) -> list[SourceReference]:
    """Project chunk contexts into document sources using citation index order."""
    if not chunks and not enriched_chunks:
        return []

    if indexer is None:
        indexer = CitationIndexer()
        indexer.build_index(chunks or enriched_chunks or [])
    projected_chunks = (
        enriched_chunks if enriched_chunks is not None else indexer.inject_chunk_idx(chunks)
    )

    catalog_by_id = {source.id: source for source in source_catalog or []}
    catalog_chunks: dict[str, dict[str, ChunkSnippet]] = {}
    for source in source_catalog or []:
        if not source.chunks:
            continue
        catalog_chunks[source.id] = {chunk.chunk_id: chunk for chunk in source.chunks}
    allowed_by_ref = {ref_id: set(chunk_ids) for ref_id, chunk_ids in (cited_chunks or {}).items()}

    # Group by reference_id, preserving first-appearance order
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in projected_chunks:
        ref_id = str(chunk.get("reference_id", ""))
        if not ref_id:
            continue
        if cited_chunks is not None:
            allowed = allowed_by_ref.get(ref_id, set())
            if not allowed or chunk.get("chunk_id") not in allowed:
                continue
        groups[ref_id].append(chunk)

    sources: list[SourceReference] = []
    ref_ids = list(cited_chunks) if cited_chunks is not None else list(groups)
    for ref_id in ref_ids:
        group = groups.get(ref_id, [])
        if not group:
            continue
        first = group[0]
        metadata = first.get("metadata") or {}
        catalog = catalog_by_id.get(ref_id)
        source_uri = _required_source_metadata(metadata, "source_uri", ref_id=ref_id)
        download_locator = _required_source_metadata(
            metadata,
            "source_download_locator",
            ref_id=ref_id,
        )
        workspace = catalog.workspace if catalog else first.get("_workspace") or default_workspace
        if not isinstance(workspace, str) or not workspace.strip():
            raise SourceBuildInvariantError(f"Source {safe_log_text(ref_id)} is missing workspace")
        document_id = catalog.document_id if catalog else first.get("full_doc_id")
        if not isinstance(document_id, str) or not document_id.strip():
            document_id = None
        file_path = first.get("file_path", "")

        def _citation_order(chunk: dict[str, Any], ref_id: str = ref_id) -> int:
            chunk_idx = chunk.get("chunk_idx")
            if isinstance(chunk_idx, int):
                return chunk_idx
            resolved = indexer.get_chunk_idx(ref_id, chunk.get("chunk_id", ""))
            return resolved if resolved is not None else 10**9

        ordered_chunks = sorted(group, key=_citation_order)

        snippets: list[ChunkSnippet] = []
        for c in ordered_chunks:
            catalog_chunk = catalog_chunks.get(ref_id, {}).get(c["chunk_id"])
            if catalog_chunk and (catalog_chunk.image_url or catalog_chunk.thumbnail_url):
                image_url = catalog_chunk.image_url
                thumbnail_url = catalog_chunk.thumbnail_url
            else:
                image_url, thumbnail_url = _chunk_image_urls(
                    c,
                    image_url_prefix=image_url_prefix,
                    default_workspace=default_workspace,
                )
            snippets.append(
                ChunkSnippet(
                    chunk_id=c["chunk_id"],
                    chunk_idx=c.get("chunk_idx") or indexer.get_chunk_idx(ref_id, c["chunk_id"]),
                    page_number=c.get("page_number")
                    if c.get("page_number") is not None
                    else (c.get("metadata", {}) or {}).get("page_number"),
                    content=filter_content_for_display(c.get("content", "")),
                    image_url=image_url,
                    thumbnail_url=thumbnail_url,
                )
            )

        cited_chunk_ids = None
        if cited_chunks is not None:
            present = {c.get("chunk_id") for c in ordered_chunks}
            cited_chunk_ids = [cid for cid in cited_chunks.get(ref_id, []) if cid in present]

        sources.append(
            SourceReference(
                id=ref_id,
                title=(catalog.title if catalog else None)
                or metadata.get("source_file_name")
                or metadata.get("file_name")
                or (Path(file_path).name if file_path else None),
                type=(catalog.type if catalog else None) or metadata.get("file_type"),
                source_uri=catalog.source_uri if catalog else source_uri,
                workspace=workspace,
                document_id=document_id,
                download_locator=catalog.download_locator if catalog else download_locator,
                cited_chunk_ids=cited_chunk_ids,
                chunks=snippets,
            )
        )

    return sources


def _required_source_metadata(metadata: Any, key: str, *, ref_id: str) -> str:
    value = metadata.get(key) if isinstance(metadata, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise SourceBuildInvariantError(f"Source {safe_log_text(ref_id)} is missing {key}")
    return value


def _chunk_image_urls(
    chunk: dict[str, Any],
    *,
    image_url_prefix: str | None,
    default_workspace: str | None,
) -> tuple[str | None, str | None]:
    if (chunk.get("metadata") or {}).get("source_type") == "web_attachment":
        # Web attachment documents are delivered inline as ``image_data`` in the
        # answer context; they have no workspace image route, so never build a
        # (non-existent) ``/images/__web_attachment__/{chunk_id}`` URL.
        return None, None
    if chunk.get("image_url") or chunk.get("thumbnail_url"):
        return chunk.get("image_url"), chunk.get("thumbnail_url")
    if not image_url_prefix:
        return None, None
    workspace = chunk.get("_workspace") or default_workspace
    chunk_id = chunk.get("chunk_id")
    if not workspace or not chunk_id:
        return None, None
    if not chunk.get("image_data") and not _is_visual_chunk(chunk):
        return None, None
    base = image_url_prefix.rstrip("/")
    safe_ws = quote(str(workspace), safe="")
    safe_chunk = quote(str(chunk_id), safe="")
    path = f"{base}/{safe_ws}/{safe_chunk}"
    return f"{path}?size=full", f"{path}?size=thumb"


def _is_visual_chunk(chunk: dict[str, Any]) -> bool:
    sidecar = chunk.get("sidecar")
    return bool(chunk.get("_has_visual_asset")) or (
        isinstance(sidecar, dict) and sidecar.get("type") == "drawing"
    )
