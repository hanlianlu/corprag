# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Focused behavior for the PostgreSQL corpus composition adapter."""

import datetime
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from dlightrag.adapters.postgres.corpus import corpus as corpus_module
from dlightrag.adapters.postgres.corpus.corpus import (
    PGCorpusCoordination,
    PGCorpusMaintenanceStore,
    PGCorpusRuntimeBinder,
    build_pg_corpus_backend,
    verify_lightrag_storage_configuration,
)
from dlightrag.application.config import DlightragConfig, LightRAGStorageSettings, StorageSettings
from dlightrag.engine.rag.retrieval.bm25 import BM25Profile
from tests.config_helpers import clone_config, mutate_config


class _Connection:
    def __init__(self, *, max_connections: str) -> None:
        self._max_connections = max_connections

    async def fetchval(self, query: str) -> str:
        assert query == "SHOW max_connections"
        return self._max_connections


async def test_connection_budget_warning_is_owned_by_coordination(
    test_config: DlightragConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    coordination = PGCorpusCoordination(
        connection_kwargs=test_config.pg_connection_kwargs(),
        workspace=test_config.deployment.workspace,
        reader=False,
        require_halfvec=False,
        required_extensions=(),
        lightrag_pool_max_size=16,
        domain_pool_max_size=10,
        acquire_timeout=test_config.storage.postgres.acquire_timeout,
    )

    with caplog.at_level(logging.INFO, logger="dlightrag.adapters.postgres.corpus.corpus"):
        await coordination._log_connection_budget(_Connection(max_connections="50"))

    assert "PostgreSQL connection budget is tight" in caplog.text
    # lightrag(16) + domain(10) + promotion gate(10) per process x 2 processes.
    assert "estimated_pool_connections=72" in caplog.text
    assert "promotion_gate=10" in caplog.text


def test_backend_factory_applies_lightrag_environment_on_create(
    test_config: DlightragConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    apply_backend = MagicMock()
    apply_sidecar = MagicMock()
    apply_runtime = MagicMock()
    monkeypatch.setattr(
        DlightragConfig,
        "apply_lightrag_backend_env",
        lambda _self, *, force=False: apply_backend(force=force),
    )
    monkeypatch.setattr(
        DlightragConfig,
        "apply_lightrag_sidecar_env",
        lambda _self: apply_sidecar(),
    )
    monkeypatch.setattr(
        DlightragConfig,
        "apply_lightrag_runtime_env",
        lambda _self, *, force=False: apply_runtime(force=force),
    )

    backend = build_pg_corpus_backend(test_config)

    assert backend.workspace_id == test_config.deployment.workspace
    apply_backend.assert_called_once_with(force=True)
    apply_sidecar.assert_called_once_with()
    apply_runtime.assert_called_once_with(force=True)


def test_all_four_default_storage_names_use_upstream_public_verification(
    test_config: DlightragConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightrag.kg as kg

    verify = MagicMock()
    monkeypatch.setattr(kg, "verify_storage_implementation", verify)

    verify_lightrag_storage_configuration(test_config)

    assert verify.call_args_list == [
        call("KV_STORAGE", "PGKVStorage"),
        call("VECTOR_STORAGE", "PGVectorStorage"),
        call("GRAPH_STORAGE", "PGTableGraphStorage"),
        call("DOC_STATUS_STORAGE", "PGDocStatusStorage"),
    ]


def test_fake_milvus_contract_requires_client_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightrag.kg as kg

    config = DlightragConfig(
        storage=StorageSettings(
            lightrag=LightRAGStorageSettings(
                vector_storage="MilvusVectorDBStorage",
                milvus_uri="https://milvus.example",
                milvus_db_name="default",
            )
        )
    )
    verify = MagicMock()
    imports: list[str] = []
    monkeypatch.setattr(kg, "verify_storage_implementation", verify)
    monkeypatch.setattr(
        corpus_module.importlib,
        "import_module",
        lambda name: imports.append(name) or object(),
    )

    verify_lightrag_storage_configuration(config)

    assert call("VECTOR_STORAGE", "MilvusVectorDBStorage") in verify.call_args_list
    assert imports == ["grpc", "pymilvus"]


def test_retrieval_partition_specs_cover_vector_filter_and_ann_contract() -> None:
    lightrag = SimpleNamespace(
        chunks_vdb=SimpleNamespace(
            table_name="lightrag_vdb_chunks_8",
            db=SimpleNamespace(vector_index_type="HNSW"),
        )
    )

    chunks, vectors = corpus_module.lightrag_retrieval_table_specs(lightrag)

    assert chunks.name.lower() == "lightrag_doc_chunks"
    assert chunks.primary_key == ("workspace", "id")
    assert "idx_lightrag_doc_chunks_dlightrag_full_doc_id" in chunks.required_indexes
    assert vectors.name == "lightrag_vdb_chunks_8"
    assert "full_doc_id" in vectors.required_columns
    assert vectors.required_index_markers == ("USING hnsw",)


@pytest.mark.parametrize("is_reader", [False, True])
@pytest.mark.parametrize("bm25_enabled", [False, True])
async def test_runtime_binder_composes_workspace_stores(
    test_config: DlightragConfig,
    monkeypatch: pytest.MonkeyPatch,
    is_reader: bool,
    bm25_enabled: bool,
) -> None:
    config = clone_config(test_config)
    mutate_config(config, "deployment.service_role", "reader" if is_reader else "writer")
    mutate_config(config, "corpus.retrieval.bm25_enabled", bm25_enabled)
    metadata = SimpleNamespace(initialize=AsyncMock())
    chunks = SimpleNamespace(ensure_document_scope_index=AsyncMock())
    vectors = SimpleNamespace(ensure_document_scope_index=AsyncMock())
    file_panel = SimpleNamespace(ensure_page_index=AsyncMock())
    bm25 = object()
    profiles = (BM25Profile(name="en", text_config="english", languages=("en",)),)
    metadata_constructor = MagicMock(return_value=metadata)
    chunk_constructor = MagicMock(return_value=chunks)
    vector_constructor = MagicMock(return_value=vectors)
    file_panel_constructor = MagicMock(return_value=file_panel)
    create_bm25 = AsyncMock(return_value=bm25)
    foundation = SimpleNamespace(ensure_tables=AsyncMock(), verify_tables=AsyncMock())
    foundation_constructor = MagicMock(return_value=foundation)
    guard = SimpleNamespace(
        verify_surface=MagicMock(),
        verify_read_only_attach_contract=MagicMock(),
        verify_all=AsyncMock(),
    )
    guard_constructor = MagicMock(return_value=guard)
    attach_read_only = AsyncMock()
    monkeypatch.setattr(corpus_module, "PGMetadataIndex", metadata_constructor)
    monkeypatch.setattr(corpus_module, "PGCorpusChunkStore", chunk_constructor)
    monkeypatch.setattr(corpus_module, "PGFilteredVectorSearch", vector_constructor)
    monkeypatch.setattr(corpus_module, "PGFilePanelStore", file_panel_constructor)
    monkeypatch.setattr(corpus_module, "PGPartitionFoundation", foundation_constructor)
    monkeypatch.setattr(corpus_module, "profiles_from_config", MagicMock(return_value=profiles))
    monkeypatch.setattr(corpus_module, "create_postgres_bm25", create_bm25)
    monkeypatch.setattr(corpus_module, "PGLightRAGContractGuard", guard_constructor)
    monkeypatch.setattr(corpus_module, "attach_lightrag_storages_read_only", attach_read_only)
    chunks_vdb = type("PGVectorStorage", (), {})()
    lightrag = SimpleNamespace(chunks_vdb=chunks_vdb, initialize_storages=AsyncMock())

    stores = await PGCorpusRuntimeBinder(config).attach(lightrag)

    metadata_constructor.assert_called_once_with(workspace=config.deployment.workspace)
    guard_constructor.assert_called_once_with(lightrag)
    guard.verify_surface.assert_called_once_with()
    guard.verify_all.assert_awaited_once_with()
    if is_reader:
        foundation.ensure_tables.assert_not_awaited()
        foundation.verify_tables.assert_awaited_once_with(
            specs=corpus_module.lightrag_retrieval_table_specs(lightrag)
        )
        chunks.ensure_document_scope_index.assert_not_awaited()
        file_panel_constructor.assert_not_called()
        guard.verify_read_only_attach_contract.assert_called_once_with()
        attach_read_only.assert_awaited_once_with(lightrag, config=config)
        lightrag.initialize_storages.assert_not_awaited()
    else:
        foundation.ensure_tables.assert_awaited_once_with(
            specs=corpus_module.lightrag_retrieval_table_specs(
                lightrag,
                require_chunk_scope_index=False,
            )
        )
        chunks.ensure_document_scope_index.assert_awaited_once_with()
        foundation.verify_tables.assert_awaited_once_with(
            specs=corpus_module.lightrag_retrieval_table_specs(lightrag)
        )
        file_panel_constructor.assert_called_once_with()
        file_panel.ensure_page_index.assert_awaited_once_with()
        guard.verify_read_only_attach_contract.assert_not_called()
        attach_read_only.assert_not_awaited()
        lightrag.initialize_storages.assert_awaited_once_with()
    metadata.initialize.assert_awaited_once_with(validate_only=is_reader)
    chunk_constructor.assert_called_once_with(
        lightrag,
        exact_threshold=config.corpus.retrieval.metadata_filter_exact_vector_threshold,
    )
    vector_constructor.assert_called_once_with(
        chunks_vdb,
        exact_threshold=config.corpus.retrieval.metadata_filter_exact_vector_threshold,
    )
    if is_reader:
        vectors.ensure_document_scope_index.assert_not_awaited()
    else:
        vectors.ensure_document_scope_index.assert_awaited_once_with()
    create_bm25.assert_awaited_once_with(
        config,
        profiles=profiles if bm25_enabled else None,
    )
    assert stores.metadata_index is metadata
    assert stores.chunks is chunks
    assert stores.filtered_vectors is vectors
    assert stores.bm25 is bm25
    assert stores.bm25_languages == (("en",) if bm25_enabled else ())
    assert stores.scoped_chunk_reader is chunks


@pytest.mark.parametrize("validate_only", [False, True])
async def test_maintenance_initializes_registry_and_promotion_job_schemas(
    validate_only: bool,
) -> None:
    registry = SimpleNamespace(initialize=AsyncMock())
    promotion_jobs = SimpleNamespace(initialize=AsyncMock())
    store = PGCorpusMaintenanceStore(
        {},
        workspace_registry=registry,  # pyright: ignore[reportArgumentType]
        promotion_jobs=promotion_jobs,  # pyright: ignore[reportArgumentType]
    )

    await store.initialize(validate_only=validate_only)

    registry.initialize.assert_awaited_once_with(validate_only=validate_only)
    promotion_jobs.initialize.assert_awaited_once_with(validate_only=validate_only)


async def test_runtime_binder_rejects_missing_postgres_chunk_backend(
    test_config: DlightragConfig,
) -> None:
    lightrag = SimpleNamespace(
        chunks_vdb=None,
        text_chunks=None,
        full_docs=None,
        doc_status=None,
        initialize_storages=AsyncMock(),
        finalize_storages=AsyncMock(),
        aquery_data=AsyncMock(),
        apipeline_enqueue_documents=AsyncMock(),
        apipeline_process_enqueue_documents=AsyncMock(),
        adelete_by_doc_id=AsyncMock(),
        aget_docs_by_track_id=AsyncMock(),
    )

    with pytest.raises(RuntimeError, match="text_chunks missing"):
        await PGCorpusRuntimeBinder(test_config).attach(lightrag)

    lightrag.initialize_storages.assert_awaited_once_with()


# ---------------------------------------------------------------------------
# Fix round: startup pipeline recovery joins the promotion write gate
# ---------------------------------------------------------------------------


async def test_pipeline_recovery_waits_out_a_fence_then_holds_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from contextlib import asynccontextmanager

    coordination = PGCorpusCoordination(
        connection_kwargs={"host": "localhost", "database": "x"},
        workspace="research",
        reader=False,
        require_halfvec=False,
        required_extensions=(),
        lightrag_pool_max_size=4,
        domain_pool_max_size=2,
        acquire_timeout=5.0,
    )

    fence_values = [30.0, 0.0]  # fenced once, then clear
    states: list[str] = []
    sleepers: list[float] = []

    class _FenceConn:
        def __init__(self) -> None:
            self.calls = 0

        async def fetchrow(self, query: str, *args: Any) -> dict[str, Any]:  # noqa: ANN401
            return {"write_fence_until": datetime.datetime.now(datetime.UTC)}

        async def fetchval(self, query: str, *args: Any) -> float:
            return fence_values.pop(0)

    async def run(op: Any) -> Any:  # noqa: ANN001, ANN401
        return await op(_FenceConn())

    monkeypatch.setattr(corpus_module.pg_pool, "run", run)

    # The gate path also resolves the real domain pool; fake it so the test
    # never needs a live PostgreSQL instance (CI has none on localhost).
    fake_pool = MagicMock()
    fake_conn = AsyncMock()
    fake_pool.acquire.return_value = fake_conn
    fake_conn.__aenter__.return_value = fake_conn
    monkeypatch.setattr(corpus_module.pg_pool, "get", AsyncMock(return_value=fake_pool))

    @asynccontextmanager
    async def fake_gate(workspace: str, *, exclusive: bool = False):  # noqa: ANN001, ANN202
        states.append("gate-open")
        yield None
        states.append("gate-close")

    from dlightrag.adapters.postgres.corpus import workspace_write_gate as gate_module

    monkeypatch.setattr(gate_module, "workspace_write_gate", fake_gate)

    async def fake_sleep(seconds: float) -> None:
        sleepers.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    entered = False
    async with coordination.pipeline_recovery():
        entered = True
        assert states == ["gate-open"]

    assert entered is True
    assert states == ["gate-open", "gate-close"]
    assert sleepers == [5.0]  # polled the remaining fence duration once
    assert [call[0][0] for call in fake_conn.execute.call_args_list] == [
        "SELECT pg_advisory_lock($1)",
        "SELECT pg_advisory_unlock($1)",
    ]


async def test_pipeline_recovery_cancellation_propagates_while_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from contextlib import asynccontextmanager

    coordination = PGCorpusCoordination(
        connection_kwargs={"host": "localhost", "database": "x"},
        workspace="research",
        reader=False,
        require_halfvec=False,
        required_extensions=(),
        lightrag_pool_max_size=4,
        domain_pool_max_size=2,
        acquire_timeout=5.0,
    )

    class _FenceConn:
        async def fetchrow(self, query: str, *args: Any) -> dict[str, Any]:  # noqa: ANN401
            return {"write_fence_until": datetime.datetime.now(datetime.UTC)}

        async def fetchval(self, query: str, *args: Any) -> float:
            return 60.0  # always fenced

    async def run(op: Any) -> Any:  # noqa: ANN001, ANN401
        return await op(_FenceConn())

    monkeypatch.setattr(corpus_module.pg_pool, "run", run)

    @asynccontextmanager
    async def fake_gate(workspace: str, *, exclusive: bool = False):  # noqa: ANN001, ANN202
        pytest.fail("the gate must not open while a fence is active")
        yield None  # pragma: no cover

    from dlightrag.adapters.postgres.corpus import workspace_write_gate as gate_module

    monkeypatch.setattr(gate_module, "workspace_write_gate", fake_gate)

    class _Cancelled(Exception):
        pass

    async def fake_sleep(seconds: float) -> None:
        raise _Cancelled()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(_Cancelled):
        async with coordination.pipeline_recovery():
            pytest.fail("gate must not open while fenced")
