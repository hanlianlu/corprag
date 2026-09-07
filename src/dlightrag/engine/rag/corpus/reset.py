# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Workspace reset and orphan cleanup for one WorkspaceRag.

5-phase cleanup:
0. Cancel pending tasks
1. Drop LightRAG storages (dynamic discovery)
2. Drop DlightRAG domain stores (registry)
3. Orphan PG table scan (safety net)
4. Remove filesystem artifacts
"""

import logging
import shutil
from pathlib import Path
from typing import Any
from uuid import UUID

from dlightrag.engine.ai.telemetry import safe_log_text
from dlightrag.engine.rag.corpus.metadata_index import MetadataIndexProtocol
from dlightrag.engine.rag.workspace.lifecycle import shutdown_lightrag_worker_pools
from dlightrag.engine.rag.workspace.ports import CorpusMaintenanceStore
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id

logger = logging.getLogger(__name__)


# -- Public entry point --------------------------------------------------------


async def areset(
    *,
    workspace_id: str,
    input_root: Path,
    lightrag: Any,
    metadata_index: MetadataIndexProtocol | None,
    maintenance: CorpusMaintenanceStore,
    keep_files: bool = False,
    dry_run: bool = False,
    preserve_run_sources_after: str | None = None,
) -> dict[str, Any]:
    """Run the module's five-phase cleanup for one workspace.

    Args:
        keep_files: If True, skip Phase 4 (filesystem cleanup).
        dry_run: If True, collect stats without executing any mutations.

    Returns:
        Stats dict with per-phase counts and any errors.
    """
    workspace = require_canonical_workspace_id(workspace_id)
    errors: list[str] = []
    stats: dict[str, Any] = {
        "workspace": workspace,
        "pending_tasks_cancelled": 0,
        "lightrag_storages_dropped": 0,
        "domain_stores_dropped": [],
        "orphan_tables_cleaned": 0,
        "local_files_removed": 0,
        "errors": errors,
    }

    # Phase 0: Cancel pending tasks (worker pools, background tasks)
    try:
        cancelled = await shutdown_lightrag_worker_pools(lightrag, dry_run=dry_run)
        stats["pending_tasks_cancelled"] = cancelled
    except Exception as exc:
        errors.append(f"Phase 0 (cancel tasks): {exc}")
        logger.warning("areset Phase 0 failed: %s", exc)

    # Phase 1: LightRAG storages -- dynamic discovery
    lr = lightrag
    if lr is not None:
        for attr in vars(lr):
            storage = getattr(lr, attr, None)
            if storage is None:
                continue
            if isinstance(storage, type):
                continue
            drop_fn = getattr(storage, "drop", None)
            if drop_fn is None or not callable(drop_fn):
                continue
            try:
                if not dry_run:
                    outcome = drop_fn()
                    if outcome is not None:
                        outcome = await outcome  # type: ignore[misc]
                    # LightRAG PostgreSQL storages swallow failures into
                    # {"status": "error", ...} instead of raising, so an
                    # unchecked drop would be miscounted as a success.
                    if isinstance(outcome, dict) and outcome.get("status") == "error":
                        raise RuntimeError(str(outcome.get("message") or "drop reported an error"))
                stats["lightrag_storages_dropped"] += 1
            except Exception as exc:
                errors.append(f"Phase 1 ({attr}): {exc}")
                logger.warning("areset Phase 1 failed for %s: %s", attr, exc)

    # Phase 2: DlightRAG domain stores
    if metadata_index is not None:
        try:
            if not dry_run:
                await metadata_index.clear()
            stats["domain_stores_dropped"].append("metadata_index")
        except Exception as exc:
            errors.append(f"Phase 2 (metadata_index): {exc}")
            logger.warning("areset Phase 2 failed for metadata_index: %s", exc)

    # Phase 3: Orphan PG table scan (safety net)
    try:
        orphans = await maintenance.clean_orphan_rows(workspace, dry_run=dry_run)
        stats["orphan_tables_cleaned"] = orphans
    except Exception as exc:
        errors.append(f"Phase 3 (orphan tables): {exc}")
        logger.warning("areset Phase 3 failed: %s", exc)

    # Workspace identity, access and operational history are deliberately
    # outside Corpus Reset and remain registered.
    # Phase 4: File system cleanup — workspace-scoped only.
    # Each workspace owns input_dir/<workspace>/; the working_dir root is shared
    # and must never be wiped per-workspace.
    if not keep_files:
        try:
            input_ws_dir = _workspace_input_dir(input_root, workspace)
            if input_ws_dir is not None and input_ws_dir.is_dir():
                stats["local_files_removed"] = _reset_workspace_files(
                    input_ws_dir,
                    dry_run=dry_run,
                    preserve_run_sources_after=preserve_run_sources_after,
                )
        except Exception as exc:
            errors.append(f"Phase 4 (filesystem): {exc}")
            logger.warning("areset Phase 4 failed: %s", safe_log_text(exc))

    logger.info(
        "areset complete for workspace=%s: %s",
        safe_log_text(workspace),
        safe_log_text(stats),
    )
    return stats


# -- Internal helpers ----------------------------------------------------------


def _reset_workspace_files(
    workspace_root: Path,
    *,
    dry_run: bool,
    preserve_run_sources_after: str | None,
) -> int:
    """Remove corpus files while retaining sources accepted after this Reset."""
    removed = 0
    preserve_after = UUID(preserve_run_sources_after) if preserve_run_sources_after else None
    for child in sorted(workspace_root.iterdir()):
        if child.is_symlink():
            raise ValueError("workspace path contains a symlink")
        if child.name == ".runs" and preserve_after is not None:
            for run_root in sorted(child.iterdir()) if child.is_dir() else ():
                if run_root.is_symlink():
                    raise ValueError("run source path contains a symlink")
                try:
                    accepted_after_reset = UUID(run_root.name).int > preserve_after.int
                except ValueError:
                    accepted_after_reset = False
                if accepted_after_reset:
                    continue
                removed += sum(1 for item in run_root.rglob("*") if item.is_file())
                if not dry_run:
                    shutil.rmtree(run_root)
            continue
        removed += sum(1 for item in child.rglob("*") if item.is_file()) if child.is_dir() else 1
        if not dry_run:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    if not dry_run:
        try:
            workspace_root.rmdir()
        except OSError:
            pass
    return removed


def _workspace_input_dir(input_root: Path, workspace: str) -> Path | None:
    """Return the direct input-root child for a canonical workspace."""
    workspace_id = require_canonical_workspace_id(workspace)

    root = input_root.resolve()
    if not root.exists():
        return None
    for child in root.iterdir():
        if child.name != workspace_id:
            continue
        if child.is_symlink():
            raise ValueError("workspace path is a symlink")
        resolved = child.resolve()
        resolved.relative_to(root)
        return resolved
    return None


# -- Orphaned workspace cleanup ------------------------------------------------


async def areset_orphaned_workspace(
    workspace: str,
    *,
    maintenance: CorpusMaintenanceStore,
    keep_files: bool = False,
    dry_run: bool = False,
    input_dir: str | None = None,
) -> dict[str, Any]:
    """Clean up orphaned workspace artifacts without a WorkspaceRag instance.

    For workspaces that no longer exist in ``dlightrag_workspace_meta`` but
    have leftover PG table rows or filesystem artifacts. This is a
    best-effort direct PG cleanup.
    """
    workspace = require_canonical_workspace_id(workspace)
    errors: list[str] = []
    stats: dict[str, Any] = {
        "workspace": workspace,
        "orphan_tables_cleaned": 0,
        "local_files_removed": 0,
        "errors": errors,
    }

    # Clean orphan PG table rows
    try:
        orphans = await maintenance.clean_orphan_rows(workspace, dry_run=dry_run)
        stats["orphan_tables_cleaned"] = orphans
    except Exception as exc:
        errors.append(f"Orphan tables: {exc}")

    # Orphan cleanup is corpus-only. Registry identity is a separate resource.
    # File system cleanup — workspace-scoped only (see areset Phase 4).
    if not keep_files:
        if input_dir:
            try:
                input_ws_dir = _workspace_input_dir(Path(input_dir), workspace)
                if input_ws_dir is not None and input_ws_dir.is_dir():
                    if not dry_run:
                        shutil.rmtree(input_ws_dir, ignore_errors=True)
                    stats["local_files_removed"] += 1
            except Exception as exc:
                errors.append(f"Filesystem (input_dir): {exc}")

    logger.info(
        "areset_orphaned complete for workspace=%s: %s",
        safe_log_text(workspace),
        safe_log_text(stats),
    )
    return stats
