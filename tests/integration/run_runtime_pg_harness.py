# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Focused fake-executor and isolated-PostgreSQL support for RunRuntime evidence."""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import asyncpg

from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.engine.runtime import PreparedRunEnvelope, RunAccessScope, Succeeded
from dlightrag.engine.runtime.contracts import RunKind
from tests.integration.pg_conn import PG_CONN_KWARGS


async def require_postgres() -> None:
    """Fail a validation command before skip-capable suites if PostgreSQL is unavailable."""
    connection = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await connection.fetchval("SELECT 1")
    finally:
        await connection.close()


@asynccontextmanager
async def isolated_run_runtime(
    prefix: str,
    *,
    query_max_active_runs: int = 16,
    query_max_nonterminal_runs: int = 30_000,
    corpus_mutation_max_active_runs: int = 2,
    corpus_mutation_max_nonterminal_runs: int = 1_000,
    pool_max_size: int = 20,
) -> AsyncIterator[tuple[PGRunStore, asyncpg.Pool]]:
    """Create, initialize, and force-drop one database owned by a test run."""
    database = f"dlightrag_{prefix}_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(
        **{**PG_CONN_KWARGS, "database": database},
        min_size=1,
        max_size=pool_max_size,
    )
    try:
        store = PGRunStore(
            pool=pool,
            query_max_active_runs=query_max_active_runs,
            query_max_nonterminal_runs=query_max_nonterminal_runs,
            corpus_mutation_max_active_runs=corpus_mutation_max_active_runs,
            corpus_mutation_max_nonterminal_runs=corpus_mutation_max_nonterminal_runs,
        )
        await store.initialize()
        yield store, pool
    finally:
        await pool.close()
        admin = await asyncpg.connect(**PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
        finally:
            await admin.close()


def run_envelope(
    kind: RunKind,
    *,
    key: str,
    owner: str = "load-owner",
    workspace: str = "workspace-000",
    mode: str = "fast",
    workspace_count: int = 1,
    action: str = "delete",
    supersedes_run_id: str | None = None,
) -> PreparedRunEnvelope:
    """Build one bounded prepared envelope without invoking a provider or parser."""
    if kind == "corpus_mutation":
        track_id = f"dlightrag-corpus-{uuid.uuid7()}"
        common: dict[str, Any] = {
            "action": action,
            "workspace": workspace,
            "track_id": track_id,
        }
        if action in {"ingest", "replace"}:
            payload = {
                **common,
                "source": {
                    "source_type": "s3",
                    "bucket": "fake-load-source",
                    "prefix": f"{key}/",
                    "replace": action == "replace",
                },
                "staged_sources": [],
            }
        elif action == "retry":
            payload = {**common, "document_ids": [f"doc-{key}"], "selector": None}
        elif action == "reset":
            payload = {**common, "supersedes_run_id": supersedes_run_id}
        else:
            payload = {
                **common,
                "file_paths": [],
                "filenames": [],
                "document_ids": [f"doc-{key}"],
            }
        return PreparedRunEnvelope(
            run_kind="corpus_mutation",
            lane="corpus_mutation",
            submitted_by=owner,
            access_scope=RunAccessScope(kind="workspace", scope_id=workspace),
            submission_key=key,
            request_fingerprint=f"fingerprint:{kind}:{key}:{action}",
            payload=payload,
            accepted_input={"action": action, "workspace": workspace},
            retention_seconds=7 * 24 * 3600,
            supersedes_run_id=supersedes_run_id,
        )

    workspaces = [f"workspace-{index:03d}" for index in range(workspace_count)]
    payload = {
        "query": f"deterministic query {key}",
        "workspaces": workspaces,
        "mode": mode,
        "agent_session_id": str(uuid.uuid7()),
        "agent_lane_id": "main",
    }
    return PreparedRunEnvelope(
        run_kind=kind,
        lane="query",
        submitted_by=owner,
        access_scope=RunAccessScope(kind="owner", scope_id=owner),
        submission_key=key,
        request_fingerprint=f"fingerprint:{kind}:{key}:{mode}:{workspace_count}",
        payload=payload,
        accepted_input={
            "query": payload["query"],
            "workspaces": workspaces,
            **({"mode": mode} if kind == "answer" else {}),
        },
        retention_seconds=(365 if kind == "answer" else 7) * 24 * 3600,
    )


@dataclass
class TrackingExecutor:
    """Local deterministic executor that records capacity and Workspace ordering."""

    delay_seconds: float = 0.0
    active: int = 0
    max_active: int = 0
    calls: int = 0
    starts_by_workspace: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    retrieval_workspace_counts: set[int] = field(default_factory=set)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def execute(self, session: Any) -> Succeeded:
        prepared: Mapping[str, Any] = session.prepared_input or {}
        workspace = str(prepared.get("workspace") or "")
        async with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls += 1
            if workspace:
                self.starts_by_workspace[workspace].append(session.run_id)
            if "workspaces" in prepared:
                self.retrieval_workspace_counts.add(len(prepared.get("workspaces") or ()))
        try:
            await session.enter_phase("fake_execution")
            await session.emit_tool_event("fake_progress", {"step": 1})
            if prepared.get("mode") == "research":
                await session.emit_tool_event("fake_progress", {"step": 2})
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            return Succeeded(
                {"fake": True, "workspace_count": len(prepared.get("workspaces") or ())}
            )
        finally:
            async with self._lock:
                self.active -= 1
