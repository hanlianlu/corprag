#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Reset dlightrag RAG storage — drops all storage backends and local files.

Works with any storage backend configured in DlightragConfig (PostgreSQL,
NanoVector, Neo4J, Milvus, Redis, Mongo, JSON files, etc.).

Usage:
    uv run scripts/reset.py                      # reset default workspace (with confirmation)
    uv run scripts/reset.py --workspace project-a # reset specific workspace
    uv run scripts/reset.py --dry-run             # preview what would be dropped
    uv run scripts/reset.py --keep-files          # drop storages, keep local files
    uv run scripts/reset.py -y                    # skip confirmation prompt
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# LightRAG storage attribute names (in initialization order)
_STORAGE_ATTRS = (
    "full_docs",
    "text_chunks",
    "full_entities",
    "full_relations",
    "entity_chunks",
    "relation_chunks",
    "entities_vdb",
    "relationships_vdb",
    "chunks_vdb",
    "chunk_entity_relation_graph",
    "llm_response_cache",
    "doc_status",
)


def _format_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ── storage drop ─────────────────────────────────────────────────


async def _drop_storages(lightrag: object, *, dry_run: bool) -> int:
    """Call drop() on each LightRAG storage instance.

    Returns the number of storages successfully dropped.
    """
    dropped = 0
    for attr in _STORAGE_ATTRS:
        storage = getattr(lightrag, attr, None)
        if storage is None:
            continue
        class_name = type(storage).__name__
        if dry_run:
            print(f"  [DRY RUN] {attr} ({class_name})")
        else:
            try:
                result = await storage.drop()
                status = result.get("status", "unknown")
                msg = result.get("message", "")
                print(f"  {attr} ({class_name}): {status} — {msg}")
                if status == "success":
                    dropped += 1
            except Exception as exc:
                print(f"  {attr} ({class_name}): ERROR — {exc}")
    return dropped


# ── local files ──────────────────────────────────────────────────


def _reset_local(working_dir: Path, *, dry_run: bool) -> tuple[int, int]:
    """Remove artifacts and caches, but keep sources/. Returns (file_count, bytes)."""
    if not working_dir.exists():
        print(f"  working directory not found: {working_dir}")
        return 0, 0

    total_bytes = 0
    file_count = 0

    for item in sorted(working_dir.iterdir()):
        if item.name == "sources":
            continue
        if item.is_file():
            size = item.stat().st_size
            total_bytes += size
            file_count += 1
            if not dry_run:
                item.unlink()
        elif item.is_dir():
            for f in item.rglob("*"):
                if f.is_file():
                    total_bytes += f.stat().st_size
                    file_count += 1
            if not dry_run:
                shutil.rmtree(item, ignore_errors=True)

    label = "[DRY RUN] " if dry_run else ""
    action = (
        f"{working_dir}: {file_count} files ({_format_size(total_bytes)})"
        if dry_run
        else (f"removed {file_count} files ({_format_size(total_bytes)}) from {working_dir}")
    )
    print(f"  {label}{action} (sources/ preserved)")

    return file_count, total_bytes


# ── orchestrator ─────────────────────────────────────────────────


async def reset_all(
    *, workspace: str | None = None, do_local: bool = True, dry_run: bool = False
) -> dict[str, int]:
    from dlightrag.config import get_config
    from dlightrag.service import RAGService

    config = get_config()
    if workspace:
        config = config.model_copy(update={"workspace": workspace})

    # Show backend info
    print("\nStorage backends (from config):")
    print(f"  KV:         {config.kv_storage}")
    print(f"  Vector:     {config.vector_storage}")
    print(f"  Graph:      {config.graph_storage}")
    print(f"  DocStatus:  {config.doc_status_storage}")
    print(f"  Workspace:  {config.workspace}")

    stats: dict[str, int] = {"storages_dropped": 0, "local_files": 0}

    # Initialize service to get LightRAG with correct backends
    print("\nInitializing RAGService...")
    service = await RAGService.create(config=config)

    try:
        # Get the LightRAG instance
        lightrag = getattr(service.ingestion.rag, "lightrag", None) if service.ingestion else None
        if lightrag is None:
            print("\nERROR: Could not access LightRAG instance from RAGService")
            return stats

        print(f"\nStorages ({len(_STORAGE_ATTRS)}):")
        stats["storages_dropped"] = await _drop_storages(lightrag, dry_run=dry_run)

        # Clean DlightRAG hash_index if present
        hash_index = getattr(service.ingestion, "_hash_index", None)
        if hash_index is not None and hasattr(hash_index, "clear"):
            if dry_run:
                print("\n  [DRY RUN] hash_index: would clear")
            else:
                try:
                    await hash_index.clear()
                    print("\n  hash_index: cleared")
                except Exception as exc:
                    print(f"\n  hash_index: ERROR — {exc}")

        if do_local:
            print("\nLocal files:")
            working_dir = Path(config.working_dir)
            files, _ = _reset_local(working_dir, dry_run=dry_run)
            stats["local_files"] = files

    finally:
        await service.close()

    return stats


# ── CLI ──────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dlightrag-reset",
        description="Reset dlightrag RAG storage (all configured backends)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Drop storages but keep local files",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Target workspace to reset (default: from config)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    do_local = not args.keep_files

    if args.dry_run:
        print("\n(dry run — nothing will be deleted)")

    if not args.dry_run and not args.yes:
        print("\nWARNING: This will permanently delete ALL RAG data in the configured backends.")
        print("Type 'yes' to proceed: ", end="")
        try:
            if input().strip().lower() != "yes":
                print("Cancelled.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 1

    stats = asyncio.run(
        reset_all(workspace=args.workspace, do_local=do_local, dry_run=args.dry_run)
    )

    print("\nDone.")
    print(f"  Storages dropped: {stats['storages_dropped']}")
    if do_local:
        print(f"  Local files removed: {stats['local_files']}")

    if args.dry_run:
        print("\nRun without --dry-run to actually delete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
