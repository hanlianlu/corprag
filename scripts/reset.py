#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Reset dlightrag RAG storage — clears PostgreSQL data, AGE graphs, and local files.

Usage:
    uv run scripts/reset.py                      # reset all (with confirmation)
    uv run scripts/reset.py --dry-run             # preview what would be deleted
    uv run scripts/reset.py --keep-files          # reset DB + graphs, keep local files
    uv run scripts/reset.py -y                    # skip confirmation prompt
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Table prefixes to discover and reset
_TABLE_PREFIXES = ("lightrag_", "dlightrag_")


def _format_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ── PostgreSQL tables ────────────────────────────────────────────


async def _reset_tables(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    workspace: str,
    *,
    dry_run: bool,
) -> dict[str, int]:
    """Delete all rows scoped to workspace from LightRAG + dlightrag tables."""
    import asyncpg  # type: ignore[import-not-found]

    stats: dict[str, int] = {}
    conn = await asyncpg.connect(
        host=host, port=port, user=user, password=password, database=database
    )
    try:
        # Dynamically discover tables matching our prefixes (any schema)
        rows = await conn.fetch("SELECT schemaname, tablename FROM pg_tables")
        all_tables = sorted(
            (r["schemaname"], r["tablename"])
            for r in rows
            if any(r["tablename"].startswith(p) for p in _TABLE_PREFIXES)
        )

        for schema, table in all_tables:
            qualified = f"{schema}.{table}"
            try:
                count = await conn.fetchval(
                    f"SELECT count(*) FROM {qualified} WHERE workspace = $1", workspace
                )
            except Exception:
                continue  # table has no workspace column or other issue

            stats[qualified] = count or 0

            if dry_run:
                print(f"  [DRY RUN] {qualified}: {count} rows")
            else:
                await conn.execute(f"DELETE FROM {qualified} WHERE workspace = $1", workspace)
                print(f"  {qualified}: deleted {count} rows")

    finally:
        await conn.close()
    return stats


# ── AGE graphs ───────────────────────────────────────────────────


async def _reset_graphs(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    *,
    dry_run: bool,
) -> int:
    """Drop all Apache AGE graphs."""
    import asyncpg  # type: ignore[import-not-found]

    conn = await asyncpg.connect(
        host=host, port=port, user=user, password=password, database=database
    )
    total = 0
    try:
        await conn.execute("SET search_path = ag_catalog, '$user', public")
        graphs = await conn.fetch("SELECT name FROM ag_catalog.ag_graph")

        for row in graphs:
            name = row["name"]
            try:
                count = await conn.fetchval(f'SELECT count(*) FROM {name}."_ag_label_vertex"')
            except Exception:
                count = 0
            total += count or 0

            if dry_run:
                print(f"  [DRY RUN] graph '{name}': {count} vertices")
            else:
                await conn.execute(f"SELECT drop_graph('{name}', true)")
                print(f"  dropped graph '{name}' ({count} vertices)")

    finally:
        await conn.close()
    return total


# ── local files ──────────────────────────────────────────────────


def _reset_local(working_dir: Path, *, dry_run: bool) -> tuple[int, int]:
    """Remove artifacts and caches, but keep sources/. Returns (file_count, bytes)."""
    if not working_dir.exists():
        print(f"  working directory not found: {working_dir}")
        return 0, 0

    total_bytes = 0
    file_count = 0

    # Delete everything except sources/ (which holds copied/downloaded originals)
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

    if dry_run:
        print(
            f"  [DRY RUN] {working_dir}: {file_count} files ({_format_size(total_bytes)}) "
            f"(sources/ preserved)"
        )
    else:
        print(
            f"  removed {file_count} files ({_format_size(total_bytes)}) from {working_dir} "
            f"(sources/ preserved)"
        )

    return file_count, total_bytes


# ── orchestrator ─────────────────────────────────────────────────


def _load_env() -> dict[str, str]:
    """Load settings from environment and .env file (no full config validation)."""
    from dotenv import dotenv_values, find_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    env = {**dotenv_values(dotenv_path), **dict(os.environ)}

    # Resolve working_dir relative to the .env file location (project root)
    working_dir = env.get("DLIGHTRAG_WORKING_DIR", "./dlightrag_storage")
    if dotenv_path and not Path(working_dir).is_absolute():
        working_dir = str(Path(dotenv_path).parent / working_dir)

    return {
        "host": env.get("DLIGHTRAG_POSTGRES_HOST", "localhost"),
        "port": env.get("DLIGHTRAG_POSTGRES_PORT", "5432"),
        "user": env.get("DLIGHTRAG_POSTGRES_USER", "dlightrag"),
        "password": env.get("DLIGHTRAG_POSTGRES_PASSWORD", "dlightrag"),
        "database": env.get("DLIGHTRAG_POSTGRES_DATABASE", "dlightrag"),
        "workspace": env.get("DLIGHTRAG_POSTGRES_WORKSPACE", "default"),
        "working_dir": working_dir,
    }


async def reset_all(
    *,
    do_tables: bool = True,
    do_graphs: bool = True,
    do_local: bool = True,
    dry_run: bool = False,
) -> dict[str, int]:
    env = _load_env()
    stats: dict[str, int] = {"table_rows": 0, "graph_vertices": 0, "local_files": 0}

    host = env["host"]
    port = int(env["port"])
    user = env["user"]
    password = env["password"]
    database = env["database"]
    workspace = env["workspace"]

    if do_tables:
        print("\nPostgreSQL tables:")
        try:
            table_stats = await _reset_tables(
                host, port, user, password, database, workspace, dry_run=dry_run
            )
            stats["table_rows"] = sum(table_stats.values())
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if do_graphs:
        print("\nAGE graphs:")
        try:
            stats["graph_vertices"] = await _reset_graphs(
                host, port, user, password, database, dry_run=dry_run
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if do_local:
        print("\nLocal files:")
        try:
            files, _ = _reset_local(Path(env["working_dir"]), dry_run=dry_run)
            stats["local_files"] = files
        except Exception as exc:
            print(f"  ERROR: {exc}")

    return stats


# ── CLI ──────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dlightrag-reset",
        description="Reset dlightrag RAG storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Reset DB + graphs but keep local files (useful for re-ingesting)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    do_tables = True
    do_graphs = True
    do_local = not args.keep_files

    print("\nReset plan:")
    print("  PostgreSQL tables: Y")
    print("  AGE graphs:        Y")
    print(f"  Local files:       {'Y' if do_local else '- (kept)'}")

    if args.dry_run:
        print("\n  (dry run — nothing will be deleted)")

    if not args.dry_run and not args.yes:
        print("\nWARNING: This will permanently delete RAG data.")
        print("Type 'yes' to proceed: ", end="")
        try:
            if input().strip().lower() != "yes":
                print("Cancelled.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 1

    stats = asyncio.run(
        reset_all(do_tables=do_tables, do_graphs=do_graphs, do_local=do_local, dry_run=args.dry_run)
    )

    print("\nDone.")
    if do_tables:
        print(f"  Table rows deleted:  {stats['table_rows']}")
    if do_graphs:
        print(f"  Graph vertices:      {stats['graph_vertices']}")
    if do_local:
        print(f"  Local files removed: {stats['local_files']}")

    if args.dry_run:
        print("\nRun without --dry-run to actually delete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
