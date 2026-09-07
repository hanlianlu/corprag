# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for PostgreSQL storage.

Requires a running PostgreSQL instance with pgvector + AGE extensions.
Skipped automatically if PostgreSQL is not available.

Tests:
- CorpusAdmin.list_workspaces() PG workspace discovery
"""

import datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.integration.pg_conn import PG_CONN_KWARGS

# Mark all tests in this module as integration
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


async def _pg_available() -> bool:
    """Check if PostgreSQL is available."""
    try:
        import asyncpg

        conn = await asyncpg.connect(
            host=str(_PG_CONN_KWARGS["host"]),
            port=int(_PG_CONN_KWARGS["port"]),
            user=str(_PG_CONN_KWARGS["user"]),
            password=str(_PG_CONN_KWARGS["password"]),
            database=str(_PG_CONN_KWARGS["database"]),
        )
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def pg_check():
    """Skip test if PostgreSQL is not available."""
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")


_PG_CONN_KWARGS = PG_CONN_KWARGS

_TEST_WORKSPACE_ALPHA = "test_pg_storage_alpha"
_TEST_WORKSPACE_BETA = "test_pg_storage_beta"
_TEST_WORKSPACES = (_TEST_WORKSPACE_ALPHA, _TEST_WORKSPACE_BETA)


async def _open_workspace_registry() -> tuple[Any, Any]:
    import asyncpg

    from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry

    pool = await asyncpg.create_pool(
        host=str(_PG_CONN_KWARGS["host"]),
        port=int(_PG_CONN_KWARGS["port"]),
        user=str(_PG_CONN_KWARGS["user"]),
        password=str(_PG_CONN_KWARGS["password"]),
        database=str(_PG_CONN_KWARGS["database"]),
        min_size=1,
        max_size=1,
    )
    registry = PGWorkspaceRegistry(pool=pool)
    await registry.initialize()
    return pool, registry


async def _delete_test_workspaces(registry: Any, *extra_workspaces: str) -> None:
    """Remove integration-test registry rows from the shared local database."""
    for workspace in (*_TEST_WORKSPACES, *extra_workspaces):
        await registry.delete(workspace)


def _corpus_admin(config: Any) -> Any:
    from dlightrag.adapters.postgres.corpus.corpus import build_pg_corpus_backend
    from dlightrag.application.corpus_admin import CorpusAdmin
    from dlightrag.application.settings import corpus_admin_settings

    backend = build_pg_corpus_backend(config)
    return CorpusAdmin(
        settings=corpus_admin_settings(config),
        pool=cast(Any, SimpleNamespace()),
        maintenance=backend.maintenance,
        file_panel=cast(Any, SimpleNamespace()),
        metadata_search=cast(Any, SimpleNamespace()),
        source_download_for=cast(Any, lambda _workspace: SimpleNamespace()),
        file_panel_cursor_secret=b"pg-storage-file-panel-test",
        metadata_search_cursor_secret=b"pg-storage-metadata-search-test",
        workspace_catalog_cursor_secret=b"pg-storage-workspace-catalog-test",
    )


def _test_config(**overrides: Any) -> Any:
    """Build a DlightragConfig bound to the suite's PostgreSQL instance.

    ``pg_pool.bind`` reads ``storage.postgres`` from the config, so the app
    paths must target the same database the raw test pools use (honoring
    PGHOST/PGPORT overrides).
    """
    from dlightrag.application.config import DlightragConfig

    storage = dict(overrides.pop("storage", {}))
    storage["postgres"] = {
        "host": _PG_CONN_KWARGS["host"],
        "port": _PG_CONN_KWARGS["port"],
        "user": _PG_CONN_KWARGS["user"],
        "password": _PG_CONN_KWARGS["password"],
        "database": _PG_CONN_KWARGS["database"],
    }
    return DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        storage=storage,
        **overrides,
    )


# ---------------------------------------------------------------------------
# File panel - bounded mixed-direction keyset traversal
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("pg_check")
async def test_file_panel_traverses_null_and_timestamp_groups_without_gaps() -> None:
    import asyncpg

    from dlightrag.adapters.postgres.corpus.file_panel import PGFilePanelStore
    from dlightrag.application.corpus_admin import (
        FilePanelCursor,
        FilePanelPageRequest,
    )

    workspace = "test_pg_file_panel"
    other_workspace = "test_pg_file_panel_other"
    pool = await asyncpg.create_pool(
        host=str(_PG_CONN_KWARGS["host"]),
        port=int(_PG_CONN_KWARGS["port"]),
        user=str(_PG_CONN_KWARGS["user"]),
        password=str(_PG_CONN_KWARGS["password"]),
        database=str(_PG_CONN_KWARGS["database"]),
        min_size=1,
        max_size=1,
    )
    try:
        async with pool.acquire() as conn:
            # Temporary tables shadow shared development relations while still
            # exercising the adapter's exact PostgreSQL SQL and indexes.
            await conn.execute(
                """
                CREATE TEMP TABLE LIGHTRAG_DOC_STATUS (
                    workspace varchar(255) NOT NULL,
                    id varchar(255) NOT NULL,
                    status varchar(64),
                    file_path TEXT,
                    content_summary TEXT,
                    error_msg TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                ) ON COMMIT PRESERVE ROWS
                """
            )
            await conn.execute(
                """
                CREATE TEMP TABLE dlightrag_doc_metadata (
                    workspace varchar(255) NOT NULL,
                    doc_id varchar(255) NOT NULL,
                    _dlightrag_finalization_complete BOOLEAN NOT NULL DEFAULT FALSE,
                    PRIMARY KEY (workspace, doc_id)
                ) ON COMMIT PRESERVE ROWS
                """
            )
            await conn.execute(
                "DELETE FROM LIGHTRAG_DOC_STATUS WHERE workspace = ANY($1::varchar[])",
                [workspace, other_workspace],
            )
            timestamp = datetime.datetime(2026, 3, 4, 5, 6, 7, 123456)
            rows = [
                (workspace, "null-a", "processed", "/null-a", None),
                (workspace, "null-b", "processed", "/null-b", None),
                (workspace, "same-a", "processed", "/same-a", timestamp),
                (workspace, "same-b", "processed", "/same-b", timestamp),
                (
                    workspace,
                    "older",
                    "processed",
                    "/older",
                    timestamp - datetime.timedelta(days=1),
                ),
                (workspace, "ignored", "pending", "/ignored", timestamp),
                (workspace, "failed-a", "failed", "/failed-a", timestamp),
                (other_workspace, "foreign", "processed", "/foreign", timestamp),
            ]
            await conn.executemany(
                """
                INSERT INTO LIGHTRAG_DOC_STATUS (
                    workspace, id, status, file_path, updated_at
                ) VALUES ($1, $2, $3, $4, $5)
                """,
                rows,
            )
            await conn.executemany(
                """
                INSERT INTO dlightrag_doc_metadata (
                    workspace, doc_id, _dlightrag_finalization_complete
                ) VALUES ($1, $2, TRUE)
                ON CONFLICT (workspace, doc_id) DO UPDATE
                SET _dlightrag_finalization_complete = TRUE
                """,
                [
                    (row_workspace, doc_id)
                    for row_workspace, doc_id, status, _path, _updated_at in rows
                    if status == "processed"
                ],
            )
            await conn.execute(
                """
                UPDATE LIGHTRAG_DOC_STATUS
                SET error_msg = 'parser failed'
                WHERE workspace = $1 AND id = 'failed-a'
                """,
                workspace,
            )

        store = PGFilePanelStore(pool=pool)
        await store.ensure_page_index()
        cursor: FilePanelCursor | None = None
        observed: list[str] = []
        while True:
            page = await store.list_processed_files(
                workspace,
                page=FilePanelPageRequest(limit=2, cursor=cursor),
            )
            assert len(page.items) <= 2
            assert page.fetched_rows <= 3
            observed.extend(item.doc_id for item in page.items)
            if not page.has_more:
                break
            assert page.items
            last = page.items[-1]
            cursor = FilePanelCursor(
                workspace=workspace,
                updated_at=last.updated_at,
                doc_id=last.doc_id,
            )

        assert observed == ["null-a", "null-b", "same-a", "same-b", "older"]
        assert len(observed) == len(set(observed))
        failed_page = await store.list_failed_files(
            workspace,
            page=FilePanelPageRequest(limit=1),
        )
        assert [(item.doc_id, item.error) for item in failed_page.items] == [
            ("failed-a", "parser failed")
        ]
        assert failed_page.fetched_rows == 1
        async with pool.acquire() as conn:
            indexdef = await conn.fetchval(
                "SELECT indexdef FROM pg_indexes WHERE indexname = $1",
                "idx_dlightrag_file_panel_processed_updated_id",
            )
            assert indexdef is not None
            normalized = " ".join(str(indexdef).split()).lower()
            assert "workspace, updated_at desc, id" in normalized
            assert "where" in normalized and "status" in normalized and "processed" in normalized
            failed_indexdef = await conn.fetchval(
                "SELECT indexdef FROM pg_indexes WHERE indexname = $1",
                "idx_dlightrag_file_panel_failed_updated_id",
            )
            assert failed_indexdef is not None
    finally:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM LIGHTRAG_DOC_STATUS WHERE workspace = ANY($1::varchar[])",
                [workspace, other_workspace],
            )
            await conn.execute(
                "DELETE FROM dlightrag_doc_metadata WHERE workspace = ANY($1::varchar[])",
                [workspace, other_workspace],
            )
        await pool.close()


@pytest.mark.usefixtures("pg_check")
async def test_doc_status_deletion_lookup_returns_exact_duplicates_and_known_ids() -> None:
    import asyncpg

    from dlightrag.adapters.postgres.corpus.doc_status_lookup import PGDocStatusLookup

    workspace = "test_pg_doc_status_lookup"
    other_workspace = "test_pg_doc_status_lookup_other"
    pool = await asyncpg.create_pool(
        host=str(_PG_CONN_KWARGS["host"]),
        port=int(_PG_CONN_KWARGS["port"]),
        user=str(_PG_CONN_KWARGS["user"]),
        password=str(_PG_CONN_KWARGS["password"]),
        database=str(_PG_CONN_KWARGS["database"]),
        min_size=1,
        max_size=1,
    )
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_STATUS (
                    workspace varchar(255) NOT NULL,
                    id varchar(255) NOT NULL,
                    status varchar(64),
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (workspace, id)
                )
                """
            )
            await conn.execute(
                "DELETE FROM LIGHTRAG_DOC_STATUS WHERE workspace = ANY($1::varchar[])",
                [workspace, other_workspace],
            )
            await conn.executemany(
                """
                INSERT INTO LIGHTRAG_DOC_STATUS (workspace, id, status, file_path)
                VALUES ($1, $2, $3, $4)
                """,
                [
                    (workspace, "doc-1", "processed", "report.pdf"),
                    (workspace, "dup-1", "failed", "report.pdf"),
                    (workspace, "doc-2", "processed", "other.pdf"),
                    (other_workspace, "foreign", "processed", "report.pdf"),
                ],
            )

        lookup = PGDocStatusLookup(workspace=workspace, pool=pool)
        matches = await lookup.resolve_deletion_matches(
            file_paths=("report.pdf",),
            doc_ids=("doc-2",),
        )

        assert [(match.doc_id, match.file_path) for match in matches] == [
            ("doc-1", "report.pdf"),
            ("doc-2", "other.pdf"),
            ("dup-1", "report.pdf"),
        ]
    finally:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM LIGHTRAG_DOC_STATUS WHERE workspace = ANY($1::varchar[])",
                [workspace, other_workspace],
            )
        await pool.close()


# ---------------------------------------------------------------------------
# CorpusAdmin.list_workspaces - PG workspace discovery
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("pg_check")
class TestPGWorkspaceDiscovery:
    """Test workspace discovery via SELECT DISTINCT workspace."""

    async def test_discovers_workspaces_from_workspace_meta(self) -> None:
        """list_workspaces() returns workspaces found in dlightrag_workspace_meta."""
        from dlightrag.adapters.postgres.core._pool import pg_pool
        from dlightrag.application.config import set_config
        from dlightrag.engine.ai.settings import EmbeddingSettings

        pool, registry = await _open_workspace_registry()
        try:
            await _delete_test_workspaces(registry)
            await registry.upsert(
                workspace=_TEST_WORKSPACE_ALPHA,
                display_name="Test PG Storage Alpha",
                embedding_model="voyage-multimodal-3.5",
            )
            await registry.upsert(
                workspace=_TEST_WORKSPACE_BETA,
                display_name="Test PG Storage Beta",
                embedding_model="voyage-multimodal-3.5",
            )

            cfg = _test_config(
                models={
                    "embedding": EmbeddingSettings(
                        provider="voyage",
                        model="voyage-multimodal-3.5",
                        api_key="test",
                        startup_probe=False,
                    ),
                },
            )
            set_config(cfg)

            pg_pool.bind(cfg)
            corpora = _corpus_admin(cfg)
            try:
                workspaces = await corpora.list_workspaces()

                assert _TEST_WORKSPACE_ALPHA in workspaces
                assert _TEST_WORKSPACE_BETA in workspaces
            finally:
                await pg_pool.close()
        finally:
            await _delete_test_workspaces(registry)
            await pool.close()

    async def test_empty_table_returns_default_workspace(self) -> None:
        """Empty workspace metadata falls back to config.deployment.workspace."""
        from dlightrag.adapters.postgres.core._pool import pg_pool
        from dlightrag.application.config import set_config
        from dlightrag.engine.ai.settings import EmbeddingSettings

        pool, registry = await _open_workspace_registry()
        try:
            await _delete_test_workspaces(registry, "test-fallback-ws")
            cfg = _test_config(
                deployment={
                    "workspace": "test-fallback-ws",
                },
                models={
                    "embedding": EmbeddingSettings(
                        provider="voyage",
                        model="voyage-multimodal-3.5",
                        api_key="test",
                        startup_probe=False,
                    ),
                },
            )
            set_config(cfg)

            pg_pool.bind(cfg)
            corpora = _corpus_admin(cfg)
            try:
                workspaces = await corpora.list_workspaces()

                # Should at least contain the default workspace
                # (may contain more if table has data from other tests)
                assert isinstance(workspaces, list)
                assert len(workspaces) >= 1
                assert "test_fallback_ws" in workspaces
            finally:
                await pg_pool.close()
        finally:
            await _delete_test_workspaces(registry, "test-fallback-ws")
            await pool.close()


# ---------------------------------------------------------------------------
# Metadata search - bounded doc_id keyset traversal
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("pg_check")
async def test_metadata_search_traverses_contains_fallback_without_gaps() -> None:
    import asyncpg

    from dlightrag.adapters.postgres.corpus.pg_metadata_search import PGMetadataSearchStore
    from dlightrag.application.corpus_admin import (
        MetadataSearchCursor,
        MetadataSearchPageRequest,
    )
    from dlightrag.engine.rag.retrieval import MetadataFilter

    workspace = "test_pg_metadata_search"
    other_workspace = "test_pg_metadata_search_other"
    pool = await asyncpg.create_pool(
        host=str(_PG_CONN_KWARGS["host"]),
        port=int(_PG_CONN_KWARGS["port"]),
        user=str(_PG_CONN_KWARGS["user"]),
        password=str(_PG_CONN_KWARGS["password"]),
        database=str(_PG_CONN_KWARGS["database"]),
        min_size=1,
        max_size=1,
    )
    try:
        async with pool.acquire() as conn:
            # Shadow production without creating an incomplete permanent table;
            # the adapter fixture includes the final fail-closed visibility column.
            await conn.execute(
                """
                CREATE TEMP TABLE dlightrag_doc_metadata (
                    workspace VARCHAR(255) NOT NULL,
                    doc_id VARCHAR(255) NOT NULL,
                    filename VARCHAR(512),
                    filename_stem VARCHAR(512),
                    _dlightrag_finalization_complete BOOLEAN NOT NULL DEFAULT FALSE,
                    PRIMARY KEY (workspace, doc_id)
                ) ON COMMIT PRESERVE ROWS
                """
            )
            await conn.execute(
                "DELETE FROM dlightrag_doc_metadata WHERE workspace = ANY($1::text[])",
                [workspace, other_workspace],
            )
            rows: list[tuple[str, str, str, str]] = []
            # 120 contains-only matches: no filename or stem equals the filter.
            for index in range(120):
                rows.append(
                    (
                        workspace,
                        f"doc-{index:03d}",
                        f"Quarterly Report draft {index}.pdf",
                        f"Quarterly Report draft {index}",
                    )
                )
            # Three exact stem matches; the widened fallback must not run.
            for index in range(3):
                rows.append((workspace, f"exact-{index:03d}", "Exact Doc.pdf", "Exact Doc"))
            # Contains-only matches for the exact filter.
            for index in range(5):
                rows.append(
                    (workspace, f"copy-{index:03d}", "Exact Doc copy.pdf", "Exact Doc copy")
                )
            for index in range(7):
                rows.append(
                    (
                        other_workspace,
                        f"foreign-{index:03d}",
                        "Quarterly Report draft.pdf",
                        "Quarterly Report draft",
                    )
                )
            await conn.executemany(
                """
                INSERT INTO dlightrag_doc_metadata
                    (workspace, doc_id, filename, filename_stem,
                     _dlightrag_finalization_complete)
                VALUES ($1, $2, $3, $4, TRUE)
                """,
                rows,
            )

        store = PGMetadataSearchStore(pool=pool)
        filters = MetadataFilter(filename="Quarterly Report")
        cursor: MetadataSearchCursor | None = None
        observed: list[str] = []
        while True:
            page = await store.search_metadata_page(
                workspace,
                filters,
                page=MetadataSearchPageRequest(limit=40, cursor=cursor),
            )
            assert len(page.document_ids) <= 40
            assert page.fetched_rows <= 41
            observed.extend(page.document_ids)
            if not page.has_more:
                break
            assert page.document_ids
            cursor = MetadataSearchCursor(
                workspace=workspace,
                after_doc_id=page.document_ids[-1],
                mode=page.mode,
            )
        assert len(observed) == 120
        assert len(set(observed)) == 120
        assert page.mode == "contains"

        # A filter with exact matches stays exact: the contains-only rows must
        # never enter the traversal.
        exact_page = await store.search_metadata_page(
            workspace,
            MetadataFilter(filename="Exact Doc"),
            page=MetadataSearchPageRequest(limit=50),
        )
        assert exact_page.mode == "exact"
        assert exact_page.has_more is False
        assert exact_page.fetched_rows == 3
        assert sorted(exact_page.document_ids) == ["exact-000", "exact-001", "exact-002"]
    finally:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM dlightrag_doc_metadata WHERE workspace = ANY($1::text[])",
                [workspace, other_workspace],
            )
        await pool.close()


# ---------------------------------------------------------------------------
# Child roster - bounded newest-first keyset traversal
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("pg_check")
async def test_child_roster_traverses_newest_first_with_timestamp_ties() -> None:
    import uuid

    import asyncpg

    from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
    from dlightrag.application.answer_runs import (
        ChildRosterCursor,
        ChildRosterPageRequest,
    )

    owner = "test_pg_child_roster"
    run_id = str(uuid.uuid4())
    pool = await asyncpg.create_pool(
        host=str(_PG_CONN_KWARGS["host"]),
        port=int(_PG_CONN_KWARGS["port"]),
        user=str(_PG_CONN_KWARGS["user"]),
        password=str(_PG_CONN_KWARGS["password"]),
        database=str(_PG_CONN_KWARGS["database"]),
        min_size=1,
        max_size=1,
    )
    try:
        store = PGRunStore(pool=pool)
        await store.initialize()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM dlightrag_answer_child_sessions WHERE owner_id = $1",
                owner,
            )
            await conn.execute(
                "DELETE FROM dlightrag_runs WHERE owner_id = $1 AND run_id = $2::uuid",
                owner,
                run_id,
            )
            await conn.execute(
                """
                INSERT INTO dlightrag_runs (
                    owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind,
                    submission_key, request_fingerprint, prepared_input_json,
                    accepted_input_json, retention_seconds
                ) VALUES (
                    $1, $2::uuid, 'answer', 'query', $1, 'owner', 'child-roster-test',
                    'child-roster-test', '{}'::jsonb, '{}'::jsonb, 31536000
                )
                """,
                owner,
                run_id,
            )
            # 120 children across three timestamp groups so the newest-first
            # traversal must break ties on child_session_id DESC. The page
            # limit of 30 is deliberately not a divisor of 40, so every
            # continuation lands inside a same-timestamp group and the
            # equality-tie branch (created_at = cursor AND child_session_id <
            # cursor) is exercised on a real page boundary.
            base = datetime.datetime(2026, 3, 4, 5, 6, 7, tzinfo=datetime.UTC)
            rows: list[tuple[Any, ...]] = []
            for index in range(120):
                child_id = uuid.uuid4()
                timestamp = base + datetime.timedelta(days=index // 40)
                rows.append(
                    (
                        owner,
                        run_id,
                        child_id,
                        uuid.uuid4(),
                        f"call-{index}",
                        None,
                        "succeeded",
                        None,
                        f"objective {index}",
                        None,
                        "query",
                        None,
                        None,
                        1,
                        "{}",
                        None,
                        None,
                        "{}",
                        None,
                        None,
                        0,
                        timestamp,
                        timestamp,
                    )
                )
            await conn.executemany(
                """
                INSERT INTO dlightrag_answer_child_sessions (
                    owner_id, run_id, child_session_id, parent_session_id, parent_call_id,
                    parent_intent_id, status, summary, objective, context_mode, model_role,
                    tools_json, usage_json, depth, context_snapshot_json, plan_json,
                    budget_json, host_state_json, lease_owner, lease_expires_at,
                    fencing_epoch, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                          $15, $16, $17, $18, $19, $20, $21, $22, $23)
                """,
                rows,
            )

        cursor: ChildRosterCursor | None = None
        observed: list[str] = []
        tie_continuations = 0
        previous_last_created_at = None
        while True:
            page = await store.list_child_sessions_page(
                owner_id=owner,
                run_id=run_id,
                page=ChildRosterPageRequest(limit=30, cursor=cursor),
            )
            assert len(page.children) <= 30
            assert page.fetched_rows <= 31
            if previous_last_created_at is not None and page.children:
                first_created_at = page.children[0]["created_at"]
                if first_created_at == previous_last_created_at:
                    # The continuation resumed inside the previous page's
                    # timestamp group: the equality-tie branch fired.
                    tie_continuations += 1
            observed.extend(str(row["child_session_id"]) for row in page.children)
            if not page.has_more:
                break
            assert page.children
            last = page.children[-1]
            previous_last_created_at = last["created_at"]
            cursor = ChildRosterCursor(
                run_id=uuid.UUID(run_id),
                created_at=last["created_at"],
                child_session_id=uuid.UUID(str(last["child_session_id"])),
            )

        assert len(observed) == 120
        assert len(set(observed)) == 120
        # Every one of the three continuations resumes inside a same-timestamp
        # group, so the equality-tie predicate must have fired at least once.
        assert tie_continuations == 3

        # The traversal order is newest-first with id ties descending: group 2
        # (latest timestamps) before group 1, then group 0.
        expected = [
            str(child_id)
            for child_id, _timestamp in sorted(
                ((row[2], row[21]) for row in rows),
                key=lambda pair: (pair[1], pair[0]),
                reverse=True,
            )
        ]
        assert observed == expected

        # A foreign owner sees nothing through the same bounded store path.
        foreign = await store.list_child_sessions_page(
            owner_id="someone-else",
            run_id=run_id,
            page=ChildRosterPageRequest(limit=10),
        )
        assert foreign.children == ()
        assert foreign.has_more is False
    finally:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM dlightrag_answer_child_sessions WHERE owner_id = $1",
                owner,
            )
            await conn.execute(
                "DELETE FROM dlightrag_runs WHERE owner_id = $1 AND run_id = $2::uuid",
                owner,
                run_id,
            )
        await pool.close()


# ---------------------------------------------------------------------------
# Workspace catalog - bounded ascending keyset traversal
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("pg_check")
async def test_workspace_catalog_traverses_ascending_pages_without_gaps() -> None:
    import asyncpg

    from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry

    prefix = "test_pg_catalog_"
    # 121 rows with a page limit of 40 always yields at least four pages
    # (40/40/40/1), so the traversal assertion holds even on a clean CI
    # database that carries no other workspaces.
    workspaces = [f"{prefix}{index:03d}" for index in range(1, 122)]
    pool = await asyncpg.create_pool(
        host=str(_PG_CONN_KWARGS["host"]),
        port=int(_PG_CONN_KWARGS["port"]),
        user=str(_PG_CONN_KWARGS["user"]),
        password=str(_PG_CONN_KWARGS["password"]),
        database=str(_PG_CONN_KWARGS["database"]),
        min_size=1,
        max_size=1,
    )
    registry = PGWorkspaceRegistry(pool=pool)
    await registry.initialize()
    try:
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO dlightrag_workspace_meta (workspace, display_name, embedding_model)
                VALUES ($1, $1, 'voyage-multimodal-3.5')
                ON CONFLICT (workspace) DO NOTHING
                """,
                [(workspace,) for workspace in workspaces],
            )

        observed: list[str] = []
        after: str | None = None
        pages = 0
        while True:
            page = await registry.list_page(after_workspace=after, limit=40)
            assert len(page.items) <= 40
            assert page.fetched_rows <= 41
            pages += 1
            observed.extend(str(item["workspace"]) for item in page.items)
            if not page.has_more:
                break
            assert page.items
            after = str(page.items[-1]["workspace"])

        # The registry may also carry other workspaces; assert the traversal
        # only on the rows this test inserted, and that the global ordering
        # never violates ascending workspace order.
        inserted = [workspace for workspace in observed if workspace.startswith(prefix)]
        assert inserted == sorted(workspaces)
        assert len(inserted) == len(set(inserted))
        assert observed == sorted(observed)
        assert pages >= 4
    finally:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM dlightrag_workspace_meta WHERE workspace LIKE $1",
                f"{prefix}%",
            )
        await pool.close()
