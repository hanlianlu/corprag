# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG contract guard startup checks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from dlightrag.adapters.postgres.corpus.lightrag_contract import PGLightRAGContractGuard


def _fake_lightrag(*, graph_storage: object | None = None) -> SimpleNamespace:
    db = SimpleNamespace(pool=object())
    return SimpleNamespace(
        chunks_vdb=SimpleNamespace(
            db=db,
            table_name="LIGHTRAG_DOC_CHUNKS",
            embedding_func=None,
        ),
        chunk_entity_relation_graph=graph_storage,
        full_docs=None,
        text_chunks=SimpleNamespace(db=db, table_name="LIGHTRAG_DOC_CHUNKS"),
        full_entities=None,
        full_relations=None,
        entity_chunks=None,
        relation_chunks=None,
        entities_vdb=None,
        relationships_vdb=None,
        llm_response_cache=None,
        doc_status=None,
    )


def _stub_runtime_checks(monkeypatch: pytest.MonkeyPatch, guard: PGLightRAGContractGuard) -> None:
    monkeypatch.setattr(guard, "_check_chunks_table_schema", AsyncMock())
    monkeypatch.setattr(guard, "_check_bm25_table", AsyncMock())


async def test_verify_all_excludes_reader_attach_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightrag.kg.postgres_impl as postgres_impl

    guard = PGLightRAGContractGuard(_fake_lightrag())
    _stub_runtime_checks(monkeypatch, guard)

    fake_manager = SimpleNamespace(
        get_config=lambda *, vector_storage=None: {"database": "db"},
        _build_vector_signature=lambda config, vector_storage: {"database": "db"},
        _assert_compatible_vector_signature=lambda signature: None,
        _lock=object(),
    )

    with patch.object(postgres_impl, "ClientManager", fake_manager):
        await guard.verify_all()


async def test_milvus_vector_contract_uses_postgres_text_chunks_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fake_lightrag()
    runtime.chunks_vdb = SimpleNamespace(query=AsyncMock(), upsert=AsyncMock())
    guard = PGLightRAGContractGuard(runtime)
    vector_check = AsyncMock()
    bm25_check = AsyncMock()
    monkeypatch.setattr(guard, "_check_chunks_table_schema", vector_check)
    monkeypatch.setattr(guard, "_check_bm25_table", bm25_check)

    await guard.verify_all(vector_storage="MilvusVectorDBStorage")

    bm25_check.assert_awaited_once_with([])
    vector_check.assert_not_awaited()


def test_verify_read_only_attach_contract_reports_missing_client_manager_attach_surfaces() -> None:
    import lightrag.kg.postgres_impl as postgres_impl

    guard = PGLightRAGContractGuard(_fake_lightrag())

    fake_manager = SimpleNamespace(
        get_config=lambda *, vector_storage=None: {"database": "db"},
        _build_vector_signature=lambda config, vector_storage: {"database": "db"},
        _assert_compatible_vector_signature=lambda signature: None,
        _lock=object(),
    )

    with patch.object(postgres_impl, "ClientManager", fake_manager):
        with pytest.raises(RuntimeError, match=r"ClientManager\._instances"):
            guard.verify_read_only_attach_contract()


def test_verify_read_only_attach_contract_rejects_positional_only_keyword_arg() -> None:
    import lightrag.kg.postgres_impl as postgres_impl

    guard = PGLightRAGContractGuard(_fake_lightrag())

    class FakeClientManager:
        _lock = object()
        _instances = {"db": None, "ref_count": 0, "vector_signature": None}

        @staticmethod
        def get_config(vector_storage, /, optional=None):
            return {"database": "db", "optional": optional}

        @staticmethod
        def _build_vector_signature(config, vector_storage):
            return {"database": config["database"]}

        @staticmethod
        def _assert_compatible_vector_signature(requested_signature):
            return None

    with patch.object(postgres_impl, "ClientManager", FakeClientManager):
        with pytest.raises(RuntimeError, match="ClientManager.get_config signature changed"):
            guard.verify_read_only_attach_contract()


def test_verify_read_only_attach_contract_allows_appended_optional_keyword_params() -> None:
    import lightrag.kg.postgres_impl as postgres_impl

    guard = PGLightRAGContractGuard(_fake_lightrag())

    class FakeClientManager:
        _lock = object()
        _instances = {"db": None, "ref_count": 0, "vector_signature": None}

        @staticmethod
        def get_config(vector_storage, *, optional=None):
            return {"database": "db", "optional": optional}

        @staticmethod
        def _build_vector_signature(config, vector_storage, *, optional=None):
            return {"database": config["database"], "optional": optional}

        @staticmethod
        def _assert_compatible_vector_signature(requested_signature, *, optional=None):
            return None

    def namespace_to_table_name(namespace, *, optional=None):
        return namespace, optional

    with (
        patch.object(postgres_impl, "ClientManager", FakeClientManager),
        patch.object(postgres_impl, "namespace_to_table_name", namespace_to_table_name),
    ):
        guard.verify_read_only_attach_contract()


def test_verify_read_only_attach_contract_rejects_changed_required_signature_prefix() -> None:
    import lightrag.kg.postgres_impl as postgres_impl

    guard = PGLightRAGContractGuard(_fake_lightrag())

    class FakeClientManager:
        _lock = object()
        _instances = {"db": None, "ref_count": 0, "vector_signature": None}

        @staticmethod
        def get_config(config, optional=None):
            return {"database": "db", "optional": optional}

        @staticmethod
        def _build_vector_signature(config, vector_storage):
            return {"database": config["database"]}

        @staticmethod
        def _assert_compatible_vector_signature(requested_signature):
            return None

    with patch.object(postgres_impl, "ClientManager", FakeClientManager):
        with pytest.raises(RuntimeError, match="ClientManager.get_config signature changed"):
            guard.verify_read_only_attach_contract()


def test_verify_read_only_attach_contract_rejects_appended_required_signature_params() -> None:
    import lightrag.kg.postgres_impl as postgres_impl

    guard = PGLightRAGContractGuard(_fake_lightrag())

    class FakeClientManager:
        _lock = object()
        _instances = {"db": None, "ref_count": 0, "vector_signature": None}

        @staticmethod
        def get_config(vector_storage, required_suffix):
            return {"database": "db", "required_suffix": required_suffix}

        @staticmethod
        def _build_vector_signature(config, vector_storage):
            return {"database": config["database"]}

        @staticmethod
        def _assert_compatible_vector_signature(requested_signature):
            return None

    with patch.object(postgres_impl, "ClientManager", FakeClientManager):
        with pytest.raises(RuntimeError, match="ClientManager.get_config signature changed"):
            guard.verify_read_only_attach_contract()


def test_postgres_guard_reports_public_runtime_drift() -> None:
    runtime = _fake_lightrag()
    runtime.initialize_storages = AsyncMock()
    runtime.finalize_storages = AsyncMock()
    runtime.aquery_data = AsyncMock()
    runtime.apipeline_enqueue_documents = AsyncMock()
    runtime.apipeline_process_enqueue_documents = None

    with pytest.raises(RuntimeError, match="apipeline_process_enqueue_documents"):
        PGLightRAGContractGuard(runtime).verify_surface()


def test_postgres_guard_reports_missing_full_status_hydration() -> None:
    runtime = _fake_lightrag()
    runtime.initialize_storages = AsyncMock()
    runtime.finalize_storages = AsyncMock()
    runtime.aquery_data = AsyncMock()
    runtime.apipeline_enqueue_documents = AsyncMock()
    runtime.apipeline_process_enqueue_documents = AsyncMock()
    runtime.doc_status = SimpleNamespace(get_docs_by_statuses_page=AsyncMock())

    with pytest.raises(RuntimeError, match="get_full_docs_by_ids"):
        PGLightRAGContractGuard(runtime).verify_surface()


def test_postgres_guard_reports_missing_bounded_status_pager() -> None:
    runtime = _fake_lightrag()
    runtime.initialize_storages = AsyncMock()
    runtime.finalize_storages = AsyncMock()
    runtime.aquery_data = AsyncMock()
    runtime.apipeline_enqueue_documents = AsyncMock()
    runtime.apipeline_process_enqueue_documents = AsyncMock()
    runtime.doc_status = SimpleNamespace()

    with pytest.raises(RuntimeError, match="get_docs_by_statuses_page"):
        PGLightRAGContractGuard(runtime).verify_surface()
