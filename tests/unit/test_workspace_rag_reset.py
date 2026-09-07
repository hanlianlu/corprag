# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for workspace reset through the RAG-owned reset module."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import (
    EmbeddingSettings,
    ModelRoleSettings,
    ModelSettings,
    ModelsSettings,
    RerankSettings,
)
from dlightrag.engine.ai.telemetry import NoopTelemetry
from dlightrag.engine.rag.workspace.ports import WorkspaceCorpusBackend
from dlightrag.engine.rag.workspace.settings import (
    CorpusSettings,
    IngestionSettings,
    PipelineSettings,
    RagSettings,
)
from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag

_FAKE_STORAGE_ATTRS = ("full_docs", "chunks_vdb", "doc_status")


class _FakeLightRAG:
    """Lightweight fake LightRAG for reset tests.

    Uses a plain object so ``vars()`` returns exactly the storage attributes
    we set, avoiding MagicMock internals leaking into dynamic discovery.
    """

    pass


def _make_service(*, workspace: str = "test_ws") -> WorkspaceRag:
    """Create a WorkspaceRag through its final constructor."""
    model = ModelSettings(provider="openai", model="gpt-5.4-mini", api_key="test")
    settings = RagSettings(
        models=ModelsSettings(
            chat=ModelRoleSettings(default=model),
            embedding=EmbeddingSettings(
                provider="openai_compatible",
                model="text-embedding-3-small",
                api_key="test",
                startup_probe=False,
                max_concurrency=2,
                batch_size=2,
            ),
            rerank=RerankSettings(enabled=False),
        ),
        corpus=CorpusSettings(
            ingestion=IngestionSettings(
                pipeline=PipelineSettings(
                    max_concurrency=2,
                    max_parallel_insert=1,
                    max_parallel_parse_native=1,
                    max_parallel_parse_mineru=1,
                    max_parallel_parse_docling=1,
                    max_parallel_analyze=1,
                    queue_size_parse=1,
                    queue_size_analyze=1,
                    queue_size_insert=1,
                )
            )
        ),
        input_root=Path("/tmp/dlightrag-test/inputs"),
    )
    maintenance = MagicMock()
    maintenance.clean_orphan_rows = AsyncMock(return_value=0)
    backend = WorkspaceCorpusBackend(
        workspace_id=workspace,
        read_only=False,
        coordination=AsyncMock(),
        maintenance=maintenance,
        runtime=AsyncMock(),
    )
    service = WorkspaceRag(
        workspace_id=workspace,
        settings=settings,
        backend=backend,
        scheduler=ModelScheduler(max_concurrency=1),
        telemetry=NoopTelemetry(),
    )
    service._initialized = True

    # Create fake LightRAG storages (dynamic-discovery friendly)
    lightrag = _FakeLightRAG()
    for attr in _FAKE_STORAGE_ATTRS:
        storage = MagicMock()
        storage.drop = AsyncMock(return_value={"status": "success", "message": "dropped"})
        setattr(lightrag, attr, storage)

    service._lightrag = lightrag

    # Mock DlightRAG-owned domain store.
    service._metadata_index = MagicMock()
    service._metadata_index.clear = AsyncMock()

    return service


class TestAresetPhase0:
    """Phase 0: Cancel pending tasks."""

    async def test_cancels_worker_pools(self) -> None:
        service = _make_service()
        with patch(
            "dlightrag.engine.rag.corpus.reset.shutdown_lightrag_worker_pools",
            new_callable=AsyncMock,
            return_value=2,
        ) as shutdown:
            result = await service.areset()

        assert result["pending_tasks_cancelled"] == 2
        shutdown.assert_awaited_once_with(service.lightrag, dry_run=False)

    async def test_dry_run_counts_worker_pools_without_shutting_down(self) -> None:
        service = _make_service()
        inner_func = SimpleNamespace(shutdown=AsyncMock())
        embedding_func = SimpleNamespace(func=inner_func)
        lr = SimpleNamespace()
        lr.embedding_func = embedding_func
        role_func = SimpleNamespace(shutdown=AsyncMock())
        lr._role_llm_states = {"query": SimpleNamespace(wrapped=role_func)}
        lr.chunks_vdb = SimpleNamespace(drop=AsyncMock())
        service._lightrag = lr

        result = await service.areset(dry_run=True)
        assert result["pending_tasks_cancelled"] == 2
        inner_func.shutdown.assert_not_called()
        role_func.shutdown.assert_not_called()


class TestAresetPhase1:
    """Phase 1: LightRAG stores (dynamic discovery)."""

    async def test_drops_all_lightrag_stores(self) -> None:
        service = _make_service()
        result = await service.areset()

        for attr in _FAKE_STORAGE_ATTRS:
            getattr(service._lightrag, attr).drop.assert_awaited_once()

        assert result["lightrag_storages_dropped"] == len(_FAKE_STORAGE_ATTRS)

    async def test_drop_reporting_error_status_is_recorded(self) -> None:
        service = _make_service()
        service._lightrag.chunks_vdb.drop = AsyncMock(
            return_value={"status": "error", "message": "boom"}
        )

        result = await service.areset()

        # LightRAG PG stores swallow failures into an error dict instead of
        # raising, so the failed store must surface as an error, not a success.
        assert result["lightrag_storages_dropped"] == len(_FAKE_STORAGE_ATTRS) - 1
        assert any("chunks_vdb" in error and "boom" in error for error in result["errors"])

    async def test_skips_lightrag_storage_class_attributes(self) -> None:
        class StorageClass:
            async def drop(self):
                raise AssertionError("class drop must not be called")

        service = _make_service()
        service._lightrag.doc_status_storage_cls = StorageClass

        result = await service.areset()

        assert result["lightrag_storages_dropped"] == len(_FAKE_STORAGE_ATTRS)
        assert not any("doc_status_storage_cls" in error for error in result["errors"])


class TestAresetPhase2:
    """Phase 2: DlightRAG domain stores."""

    async def test_clears_metadata_index(self) -> None:
        service = _make_service()
        await service.areset()
        cast(Any, service._metadata_index).clear.assert_awaited_once()

    async def test_skips_none_metadata_index(self) -> None:
        service = _make_service()
        service._metadata_index = None
        result = await service.areset()
        assert "metadata_index" not in result["domain_stores_dropped"]


class TestAresetPhase3:
    """Phase 3: PG orphan table cleanup."""

    async def test_runs_on_pg_backend(self) -> None:
        service = _make_service()
        maintenance = cast(Any, service._corpus_backend).maintenance
        maintenance.clean_orphan_rows.return_value = 3

        result = await service.areset()

        assert result["orphan_tables_cleaned"] == 3
        maintenance.clean_orphan_rows.assert_awaited_once_with("test_ws", dry_run=False)


class TestAresetPhase4:
    """Phase 4: Local files."""

    async def test_keep_files_skips_cleanup(self) -> None:
        service = _make_service()
        result = await service.areset(keep_files=True)
        assert result["local_files_removed"] == 0

    async def test_removes_local_files(self, tmp_path: Path) -> None:
        service = _make_service()
        service.settings = cast(
            Any,
            SimpleNamespace(input_root=tmp_path / "inputs", read_only=False),
        )
        ws_dir = tmp_path / "inputs" / service.workspace_id

        # Workspace-scoped files under input_dir/<workspace>/
        ws_dir.mkdir(parents=True)
        (ws_dir / "parsed_doc.json").write_text("{}")
        (ws_dir / "subdir").mkdir()
        (ws_dir / "subdir" / "data.bin").write_bytes(b"x" * 100)

        result = await service.areset()

        # Only workspace-scoped files counted and removed
        assert result["local_files_removed"] == 2
        assert not ws_dir.exists()

    async def test_preserves_sources_accepted_after_reset_run(self, tmp_path: Path) -> None:
        service = _make_service()
        service.settings = cast(
            Any,
            SimpleNamespace(input_root=tmp_path / "inputs", read_only=False),
        )
        runs = tmp_path / "inputs" / service.workspace_id / ".runs"
        older = runs / "0199a0a0-0000-7000-8000-000000000001" / "sources"
        reset = runs / "0199a0a0-0000-7000-8000-000000000002" / "sources"
        newer = runs / "0199a0a0-0000-7000-8000-000000000003" / "sources"
        invalid = runs / "stale-partial" / "sources"
        for source in (older, reset, newer, invalid):
            source.mkdir(parents=True)
            (source / "report.pdf").write_bytes(b"pdf")

        result = await service.areset(
            preserve_run_sources_after="0199a0a0-0000-7000-8000-000000000002"
        )

        assert result["local_files_removed"] == 3
        assert not older.exists()
        assert not reset.exists()
        assert not invalid.exists()
        assert (newer / "report.pdf").read_bytes() == b"pdf"

    async def test_root_files_survive_reset(self, tmp_path: Path) -> None:
        """Shared files in working_dir root must NOT be deleted per-workspace."""
        import sqlite3

        service = _make_service()
        service.settings = cast(
            Any,
            SimpleNamespace(input_root=tmp_path / "inputs", read_only=False),
        )

        db_path = tmp_path / "shared_state.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS state (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()
        (tmp_path / "shared_config.json").write_text("{}")

        await service.areset()

        assert db_path.exists()
        assert (tmp_path / "shared_config.json").exists()

    async def test_workspace_path_input_cannot_escape_input_root(self, tmp_path: Path) -> None:
        service = _make_service(workspace="___outside")
        service.settings = cast(
            Any,
            SimpleNamespace(input_root=tmp_path / "inputs", read_only=False),
        )

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("keep")

        normalized_ws_dir = tmp_path / "inputs" / "___outside"
        normalized_ws_dir.mkdir(parents=True)
        (normalized_ws_dir / "staged.txt").write_text("delete")

        result = await service.areset()

        assert result["local_files_removed"] == 1
        assert outside.exists()
        assert not normalized_ws_dir.exists()


class TestAresetDryRun:
    """dry_run=True collects stats without executing."""

    async def test_dry_run_no_drops(self) -> None:
        service = _make_service()
        result = await service.areset(dry_run=True)

        # Stats reported but no actual drops
        assert result["lightrag_storages_dropped"] == len(_FAKE_STORAGE_ATTRS)
        for attr in ("full_docs", "chunks_vdb"):
            getattr(service._lightrag, attr).drop.assert_not_awaited()
        cast(Any, service._metadata_index).clear.assert_not_awaited()
        # A preview must leave the live runtime intact.
        assert service._initialized is True


class TestAresetErrorHandling:
    """Errors in one phase don't block subsequent phases."""

    async def test_phase1_error_continues_to_phase2(self) -> None:
        service = _make_service()
        service._lightrag.full_docs.drop = AsyncMock(side_effect=RuntimeError("boom"))

        result = await service.areset()

        # Phase 2 still ran
        cast(Any, service._metadata_index).clear.assert_awaited_once()
        assert len(result["errors"]) >= 1

    async def test_phase2_error_continues(self) -> None:
        service = _make_service()
        cast(Any, service._metadata_index).clear = AsyncMock(side_effect=RuntimeError("boom"))

        result = await service.areset()

        assert len(result["errors"]) >= 1
        # _initialized still set to False
        assert not service._initialized


class TestAresetState:
    """Service state after reset."""

    async def test_sets_initialized_false(self) -> None:
        service = _make_service()
        assert service._initialized is True
        await service.areset()
        assert service._initialized is False
