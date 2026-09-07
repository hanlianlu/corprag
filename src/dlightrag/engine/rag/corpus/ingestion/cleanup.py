# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG deletion helpers."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from inspect import isawaitable
from pathlib import Path
from typing import Any

from dlightrag.engine.rag.corpus.contracts import DocStatusLookup
from dlightrag.engine.rag.corpus.metadata_index import MetadataIndexProtocol
from dlightrag.engine.rag.retrieval.metadata_fields import (
    INGEST_FINALIZATION_COMPLETE_FIELD,
)

logger = logging.getLogger(__name__)

# _compute_mdhash_id is imported lazily in cascade_delete to avoid a hard
# dependency on lightrag at module import time.


@dataclass
class DeletionContext:
    """Aggregated deletion context from available data sources."""

    identifier: str  # Original filename/path requested for deletion
    doc_ids: set[str] = field(default_factory=set)
    file_paths: set[str] = field(default_factory=set)
    sources_used: list[str] = field(default_factory=list)  # For audit trail


async def collect_deletion_context(
    identifier: str,
    *,
    metadata_index: MetadataIndexProtocol | None,
    doc_status_lookup: DocStatusLookup | None,
) -> DeletionContext:
    """Resolve one deletion identifier through bounded indexed reads only.

    Identifiers are exact source locators, stored filenames, or status
    ``file_path`` values. Bare stems are intentionally not expanded because
    that is ambiguous and previously required corpus-wide status scans.
    """
    ctx = DeletionContext(identifier=identifier)
    normalized = str(identifier).strip()
    if not normalized:
        return ctx
    basename = Path(normalized).name

    metadata_lookup_failed = False
    status_lookup_failed = False

    # A canonical Product Document id is an exact owned identity too.
    if metadata_index is not None:
        try:
            direct_read = metadata_index.get(normalized)
            direct = await direct_read if isawaitable(direct_read) else None
            if isinstance(direct, Mapping):
                ctx.doc_ids.add(normalized)
                ctx.sources_used.append("metadata_index")
        except Exception as exc:
            metadata_lookup_failed = True
            logger.warning("Metadata document lookup failed for %s: %s", identifier, exc)

    # An exact durable locator owns identity. A filename fallback is permitted
    # only when the caller supplied a bare display name and exact resolution
    # completed without error; failures must never widen deletion identity.
    if metadata_index is not None:
        try:
            ctx.doc_ids.update(await metadata_index.find_by_download_locator(normalized))
            if ctx.doc_ids:
                ctx.sources_used.append("metadata_index")
        except Exception as exc:
            metadata_lookup_failed = True
            logger.warning("Metadata index lookup failed for %s: %s", identifier, exc)

    async def merge_status_matches(*, file_paths: tuple[str, ...]) -> None:
        nonlocal status_lookup_failed
        if doc_status_lookup is None:
            return
        try:
            matches = await doc_status_lookup.resolve_deletion_matches(
                file_paths=file_paths,
                doc_ids=tuple(sorted(ctx.doc_ids)),
            )
        except Exception as exc:
            status_lookup_failed = True
            logger.warning("Document status lookup failed for %s: %s", identifier, exc)
            return
        for match in matches:
            ctx.doc_ids.add(match.doc_id)
            if match.file_path:
                ctx.file_paths.add(match.file_path)
        if matches and "doc_status" not in ctx.sources_used:
            ctx.sources_used.append("doc_status")

    await merge_status_matches(file_paths=(normalized,))

    if not ctx.doc_ids and normalized == basename and not metadata_lookup_failed:
        if metadata_index is not None:
            try:
                ctx.doc_ids.update(await metadata_index.find_by_filename(basename))
                if ctx.doc_ids:
                    ctx.sources_used.append("metadata_index")
            except Exception as exc:
                metadata_lookup_failed = True
                logger.warning("Metadata filename lookup failed for %s: %s", identifier, exc)
        if not metadata_lookup_failed and not status_lookup_failed:
            await merge_status_matches(file_paths=(basename,))

    # A metadata locator can differ from the status file_path. Expand once on
    # the exact hydrated paths to include duplicate receipts sharing identity.
    duplicate_paths = tuple(sorted(ctx.file_paths.difference({normalized})))
    if duplicate_paths and not status_lookup_failed:
        await merge_status_matches(file_paths=duplicate_paths)

    logger.info(
        "Deletion context for %s: doc_ids=%d, file_paths=%d, sources=%s",
        identifier,
        len(ctx.doc_ids),
        len(ctx.file_paths),
        ctx.sources_used,
    )
    return ctx


async def cascade_delete(
    ctx: DeletionContext,
    lightrag: Any,
    metadata_index: Any | None = None,
) -> dict[str, Any]:
    """Hide first, inspect the complete public deletion contract, then clean up."""
    stats: dict[str, Any] = {"docs_deleted": 0, "errors": [], "outcomes": []}

    for doc_id in sorted(ctx.doc_ids):
        previous_metadata = None
        if metadata_index is not None:
            try:
                previous_metadata = await metadata_index.get(doc_id)
            except Exception as exc:
                stats["errors"].append(f"Visibility lookup ({doc_id}) failed")
                stats["outcomes"].append({"doc_id": doc_id, "status": "waiting_for_repair"})
                logger.warning("cascade_delete visibility lookup failed for %s: %s", doc_id, exc)
                continue
        # Hide before the first destructive LightRAG effect. If the marker
        # cannot be persisted, do not proceed with a potentially visible
        # partial deletion.
        if metadata_index is not None:
            try:
                await metadata_index.upsert(
                    doc_id,
                    {INGEST_FINALIZATION_COMPLETE_FIELD: False},
                )
            except Exception as exc:
                stats["errors"].append(f"Visibility barrier ({doc_id}) failed")
                stats["outcomes"].append({"doc_id": doc_id, "status": "failed"})
                logger.warning("cascade_delete visibility hide failed for %s: %s", doc_id, exc)
                continue

        try:
            deletion = await lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        except BaseException as exc:
            stats["errors"].append(f"Upstream deletion ({doc_id}) is ambiguous")
            stats["outcomes"].append({"doc_id": doc_id, "status": "waiting_for_repair"})
            logger.warning("cascade_delete upstream call failed for %s: %s", doc_id, exc)
            continue
        raw_status = (
            deletion.get("status")
            if isinstance(deletion, dict)
            else getattr(deletion, "status", None)
        )
        status = str(getattr(raw_status, "value", raw_status) or "").strip().lower()
        if status == "not_allowed":
            # Public LightRAG guarantees this rejection made no write. Restore
            # the exact DlightRAG-owned visibility projection.
            if metadata_index is not None and isinstance(previous_metadata, Mapping):
                try:
                    await metadata_index.upsert(doc_id, previous_metadata)
                except Exception as exc:
                    stats["errors"].append(f"Visibility restore ({doc_id}) failed")
                    stats["outcomes"].append({"doc_id": doc_id, "status": "waiting_for_repair"})
                    logger.warning("cascade_delete visibility restore failed: %s", exc)
                    continue
            stats["outcomes"].append({"doc_id": doc_id, "status": "rejected"})
            continue
        if status not in {"success", "not_found"}:
            stats["errors"].append(f"Upstream deletion ({doc_id}) is ambiguous")
            stats["outcomes"].append({"doc_id": doc_id, "status": "waiting_for_repair"})
            continue

        if metadata_index is not None:
            try:
                await metadata_index.delete(doc_id)
            except Exception as exc:
                stats["errors"].append(f"Projection cleanup ({doc_id}) failed")
                stats["outcomes"].append({"doc_id": doc_id, "status": "waiting_for_repair"})
                logger.warning("cascade_delete projection cleanup failed for %s: %s", doc_id, exc)
                continue
        stats["docs_deleted"] += 1
        stats["outcomes"].append({"doc_id": doc_id, "status": "deleted", "upstream": status})

    return stats


def remove_deleted_files(file_paths: set[str], input_dir: str) -> int:
    """Delete physical files and parsed artifact directories from disk.

    Handles the full LightRAG parser artifact layout:

    - Source files in ``input_dir/``
    - Parsed artifacts under ``input_dir/__parsed__/`` using either the full
      filename or its stem: ``<name>.pdf.parsed/`` or ``<name>.parsed/`` plus
      the corresponding ``.mineru_raw`` / ``.docling_raw`` directories
    - Collision-suffixed variants (``<name>.pdf_001.parsed/``, etc.)

    Missing paths are idempotent. Any observed I/O failure is raised so the
    durable mutation cannot report success with retained requested bytes.

    Args:
        file_paths: Absolute paths to ingested files (from LightRAG doc_status).
        input_dir: The workspace's input directory (parent of the source files).

    Returns:
        Number of files/directories removed.
    """
    import re
    import shutil

    from lightrag.constants import PARSED_ARTIFACT_DIR_SUFFIXES, PARSED_DIR_NAME

    removed = 0
    failures: list[OSError] = []
    input_root = Path(input_dir)
    default_parsed_root = input_root / PARSED_DIR_NAME
    _collision_re = re.compile(r"_\d{3}$")

    for fp in file_paths:
        if _is_remote_source_path(fp):
            continue
        path = Path(fp)
        filename = path.name
        stem = path.stem
        artifact_bases = {filename, stem}
        source_root = _source_root_for_stored_path(path, input_root)
        parsed_roots = [source_root / PARSED_DIR_NAME]
        if default_parsed_root not in parsed_roots:
            parsed_roots.append(default_parsed_root)

        # 1. Remove the source file (may be in input_dir/ or moved into
        #    __parsed__/ by LightRAG after ingest).
        source_candidates = [source_root / filename]
        if path.is_absolute():
            source_candidates.insert(0, path)
        source_candidates.extend(parsed_root / filename for parsed_root in parsed_roots)
        for candidate in dict.fromkeys(source_candidates):
            try:
                if candidate.exists() and candidate.is_file():
                    candidate.unlink()
                    removed += 1
            except OSError as exc:
                failures.append(exc)
                logger.warning("Failed to remove source file: %s", candidate, exc_info=True)

        # 2. Remove parsed artifact directories under __parsed__/. LightRAG
        #    versions have used both the full source filename and its stem as
        #    the artifact base, with optional collision suffixes.
        for parsed_root in parsed_roots:
            try:
                if not parsed_root.exists() or not parsed_root.is_dir():
                    continue
                for entry in sorted(parsed_root.iterdir()):
                    if not entry.is_dir():
                        continue
                    entry_name = entry.name
                    for suffix in PARSED_ARTIFACT_DIR_SUFFIXES:
                        if not entry_name.endswith(suffix):
                            continue
                        artifact_base = _collision_re.sub("", entry_name[: -len(suffix)])
                        if artifact_base in artifact_bases:
                            try:
                                shutil.rmtree(entry)
                            except OSError as exc:
                                failures.append(exc)
                            else:
                                removed += 1
                            break
            except OSError as exc:
                failures.append(exc)
                logger.warning("Failed to scan parsed dir: %s", parsed_root, exc_info=True)

    if failures:
        raise OSError("one or more requested corpus source files could not be removed")
    return removed


def _is_remote_source_path(path: str) -> bool:
    return path.startswith(("azure://", "s3://", "https://"))


def _source_root_for_stored_path(path: Path, input_root: Path) -> Path:
    if path.is_absolute():
        try:
            path.resolve().relative_to(input_root.resolve())
        except ValueError:
            return input_root
        return path.parent
    if len(path.parts) > 1:
        return input_root / Path(*path.parts[:-1])
    return input_root


__all__ = [
    "DeletionContext",
    "cascade_delete",
    "collect_deletion_context",
    "remove_deleted_files",
]
