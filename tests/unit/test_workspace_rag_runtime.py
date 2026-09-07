# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the one-workspace RAG capability."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from dlightrag.application.config import DlightragConfig
from dlightrag.application.settings import rag_settings
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.telemetry import NoopTelemetry
from dlightrag.engine.rag.corpus.ingestion.document_embedding import (
    RobustDocumentEmbedder,
    build_document_embedder,
    resolve_direct_image_embedding_enabled,
)
from dlightrag.engine.rag.corpus.ingestion.engine import PreparedIngestFile, UnifiedIngestionEngine
from dlightrag.engine.rag.corpus.ingestion.paths import (
    iter_ingestable_files,
    lightrag_archived_source_path,
    remote_parser_input_path,
    retained_remote_source_path,
    stage_input_file,
)
from dlightrag.engine.rag.corpus.sources.base import AsyncDataSource, SourceDocument
from dlightrag.engine.rag.workspace.workspace_rag import RemoteIngestWindowProgress, WorkspaceRag
from dlightrag.engine.rag.workspace.workspaces import normalize_workspace
from tests.config_helpers import mutate_config


def _service(
    config: DlightragConfig,
    *,
    backend: Any | None = None,
) -> WorkspaceRag:
    workspace_id = normalize_workspace(config.deployment.workspace)
    return WorkspaceRag(
        workspace_id=workspace_id,
        settings=rag_settings(config),
        backend=cast(
            Any,
            backend if backend is not None else _backend(workspace_id, read_only=config.is_reader),
        ),
        scheduler=ModelScheduler(max_concurrency=1),
        telemetry=NoopTelemetry(),
    )


def _set_failed_docs(service: WorkspaceRag, docs: list[dict[str, Any]]) -> None:
    service._initialized = True

    async def _pages() -> AsyncIterator[list[dict[str, Any]]]:
        yield list(docs)

    service._iter_failed_doc_pages = _pages  # type: ignore[method-assign]


def _runtime_lightrag() -> SimpleNamespace:
    return SimpleNamespace(
        workspace="default",
        chunks_vdb=SimpleNamespace(),
        text_chunks=object(),
        full_docs=object(),
        doc_status=object(),
        initialize_storages=AsyncMock(),
        finalize_storages=AsyncMock(),
        aquery_data=AsyncMock(),
        apipeline_enqueue_documents=AsyncMock(),
        apipeline_process_enqueue_documents=AsyncMock(),
    )


def _runtime_binder() -> SimpleNamespace:
    return SimpleNamespace(
        create=MagicMock(return_value=_runtime_lightrag()),
        attach=AsyncMock(
            return_value=SimpleNamespace(
                metadata_index=object(),
                chunks=object(),
                filtered_vectors=object(),
                bm25=object(),
                bm25_languages=(),
                scoped_chunk_reader=object(),
                doc_status_lookup=object(),
            )
        ),
    )


def _backend(workspace_id: str, *, read_only: bool) -> SimpleNamespace:
    return SimpleNamespace(
        workspace_id=workspace_id,
        read_only=read_only,
        coordination=SimpleNamespace(
            workspace_initialization=lambda: _CoordinationContext(),
            pipeline_recovery=lambda: _CoordinationContext(),
        ),
        maintenance=AsyncMock(),
        runtime=_runtime_binder(),
    )


class _CoordinationContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _FakeChatModels:
    @classmethod
    async def acreate(cls, *_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            default_func=object(),
            role_configs=None,
            aclose=AsyncMock(),
        )


# ---------------------------------------------------------------------------
# TestWorkspaceRagAingest
# ---------------------------------------------------------------------------


class TestWorkspaceRagAingest:
    """Test ingestion logic -- replace defaults, azure lifecycle."""

    def _make_initialized_service(self, config: DlightragConfig) -> tuple[WorkspaceRag, MagicMock]:
        service = _service(config)
        service._initialized = True
        ingestion = MagicMock()
        ingestion.aingest_file = AsyncMock(return_value={"status": "success"})
        ingestion.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": [{"status": "success"}]}
        )
        service._ingestion_engine = ingestion
        return service, ingestion

    async def test_workspace_meta_upsert_uses_canonical_workspace_id(
        self,
        test_config: DlightragConfig,
    ) -> None:
        mutate_config(test_config, "deployment.workspace", "test-fallback-ws")
        maintenance = MagicMock()
        maintenance.register_workspace = AsyncMock()
        backend = _backend("test_fallback_ws", read_only=False)
        backend.maintenance = maintenance
        service = _service(test_config, backend=backend)

        await service._upsert_workspace_meta()

        maintenance.register_workspace.assert_awaited_once_with(
            workspace="test_fallback_ws",
            display_name="test_fallback_ws",
            embedding_model=test_config.models.embedding.model,
        )

    async def test_aingest_not_initialized_raises(self, test_config: DlightragConfig) -> None:
        service = _service(test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.aingest(source_type="local", path="/tmp/f.pdf")

    async def test_acreate_closes_partial_service_when_initialization_fails(
        self,
        test_config: DlightragConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        initialize = AsyncMock(side_effect=RuntimeError("initialization failed"))
        close = AsyncMock()
        monkeypatch.setattr(WorkspaceRag, "initialize", initialize)
        monkeypatch.setattr(WorkspaceRag, "aclose", close)

        with pytest.raises(RuntimeError, match="initialization failed"):
            await WorkspaceRag.acreate(
                workspace_id=normalize_workspace(test_config.deployment.workspace),
                settings=rag_settings(test_config),
                backend=cast(
                    Any,
                    _backend(
                        normalize_workspace(test_config.deployment.workspace),
                        read_only=test_config.is_reader,
                    ),
                ),
                scheduler=ModelScheduler(max_concurrency=1),
                telemetry=NoopTelemetry(),
            )

        close.assert_awaited_once()

    async def test_aingest_replace_default_from_config(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        mutate_config(test_config, "corpus.ingestion.replace_default", True)
        fake_pdf = tmp_path / "file.pdf"
        fake_pdf.write_bytes(b"%PDF-fake")
        service, ingestion = self._make_initialized_service(test_config)
        await service.aingest(source_type="local", path=str(fake_pdf))
        call_kwargs = ingestion.aingest_file.call_args
        assert call_kwargs.kwargs["replace"] is True

    async def test_aingest_replace_explicit_overrides_config(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        mutate_config(test_config, "corpus.ingestion.replace_default", True)
        fake_pdf = tmp_path / "file.pdf"
        fake_pdf.write_bytes(b"%PDF-fake")
        service, ingestion = self._make_initialized_service(test_config)
        await service.aingest(source_type="local", path=str(fake_pdf), replace=False)
        call_kwargs = ingestion.aingest_file.call_args
        assert call_kwargs.kwargs["replace"] is False

    # -- Azure blob lifecycle --

    async def test_aingest_azure_defaults_prefix_when_neither_set(
        self, test_config: DlightragConfig
    ) -> None:
        """When neither blob_path nor prefix provided, prefix defaults to '' (entire container)."""
        service, _ = self._make_initialized_service(test_config)
        mock_source = AsyncMock()
        seen_prefixes: list[str | None] = []

        async def _aiter_documents(prefix: str | None = None):
            seen_prefixes.append(prefix)
            for item in ():
                yield item

        mock_source.aiter_documents = _aiter_documents
        await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        assert seen_prefixes == [""]

    async def test_aingest_azure_calls_aclose(self, test_config: DlightragConfig) -> None:
        """source.aclose() is called after successful ingestion."""
        service, _ = self._make_initialized_service(test_config)
        mock_source = AsyncMock()

        async def _aiter_documents(prefix: str | None = None):
            for item in ():
                yield item

        mock_source.aiter_documents = _aiter_documents
        await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        mock_source.aclose.assert_awaited_once()

    async def test_aingest_azure_calls_aclose_on_error(self, test_config: DlightragConfig) -> None:
        """source.aclose() is called even when ingestion raises."""
        service, ingestion = self._make_initialized_service(test_config)
        ingestion.aingest_files = AsyncMock(side_effect=RuntimeError("ingestion failed"))
        mock_source = AsyncMock()

        async def _aiter_documents(prefix: str | None = None):
            yield SourceDocument(key="f.pdf")

        mock_source.aiter_documents = _aiter_documents
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _document, destination: destination.write_bytes(b"%PDF")
        )
        with pytest.raises(RuntimeError, match="ingestion failed"):
            await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        mock_source.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestWorkspaceRagClose
# ---------------------------------------------------------------------------


class TestWorkspaceRagClose:
    """Test cleanup logic."""

    async def test_close_defers_cancellation_until_all_resources_are_closed(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        service._initialized = True
        service._rerank_func = AsyncMock()
        multimodal_embedder = AsyncMock()
        chat_models = AsyncMock()
        service._multimodal_embedder = multimodal_embedder
        service._chat_models = chat_models
        service._lightrag = MagicMock()
        service._lightrag.finalize_storages = AsyncMock()

        async def cancel_current_task() -> None:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            await asyncio.sleep(0)

        service._rerank_func.aclose.side_effect = cancel_current_task

        with pytest.raises(asyncio.CancelledError):
            await service.aclose()

        multimodal_embedder.aclose.assert_awaited_once()
        chat_models.aclose.assert_awaited_once()
        service._lightrag.finalize_storages.assert_awaited_once()

    async def test_close_handles_errors(self, test_config: DlightragConfig) -> None:
        service = _service(test_config)
        service._initialized = True
        service._lightrag = MagicMock()
        service._lightrag.finalize_storages = AsyncMock(side_effect=RuntimeError("cleanup failed"))

        # Should not raise
        await service.aclose()

    async def test_close_only_closes_underlying_multimodal_embedder(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        service._initialized = True
        service._multimodal_embedder = AsyncMock()
        service._document_embedder = MagicMock(spec=RobustDocumentEmbedder)

        await service.aclose()

        service._multimodal_embedder.aclose.assert_awaited_once()
        assert service._document_embedder.mock_calls == []

    async def test_close_shuts_down_lightrag_role_worker_pools(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        service._initialized = True
        lightrag = MagicMock()
        lightrag.finalize_storages = AsyncMock()
        service._lightrag = lightrag

        with patch(
            "dlightrag.engine.rag.workspace.workspace_rag.shutdown_lightrag_worker_pools",
            new_callable=AsyncMock,
            return_value=3,
        ) as shutdown:
            await service.aclose()

        shutdown.assert_awaited_once_with(lightrag)
        lightrag.finalize_storages.assert_awaited_once()

    async def test_startup_pipeline_recovery_waits_for_current_owner_then_runs(
        self, test_config: DlightragConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        allow_lock = asyncio.Event()

        class FakeCoordination:
            @asynccontextmanager
            async def pipeline_recovery(self):
                await allow_lock.wait()
                yield

        backend = _backend(normalize_workspace(test_config.deployment.workspace), read_only=False)
        backend.coordination = FakeCoordination()
        service = _service(test_config, backend=backend)
        service._lightrag = MagicMock()
        service._lightrag.apipeline_process_enqueue_documents = AsyncMock()

        recovery = asyncio.create_task(service._resume_lightrag_pipeline())
        await asyncio.sleep(0)
        service._lightrag.apipeline_process_enqueue_documents.assert_not_awaited()

        allow_lock.set()
        await recovery

        service._lightrag.apipeline_process_enqueue_documents.assert_awaited_once_with()


# ---------------------------------------------------------------------------
# TestWorkspaceRagRetrieve
# ---------------------------------------------------------------------------


class TestWorkspaceRagRetrieve:
    """Test aretrieve delegation to RetrievalEngine."""

    def _make_retrieval_service(self, config: DlightragConfig) -> tuple[WorkspaceRag, MagicMock]:
        from dlightrag.engine.rag.retrieval import RetrievalResult

        service = _service(config)
        service._initialized = True
        orchestrator = MagicMock()
        orchestrator.aretrieve = AsyncMock(return_value=RetrievalResult(contexts={"chunks": []}))
        service._retrieval_orchestrator = orchestrator
        service._lightrag_stores = MagicMock()
        return service, orchestrator

    async def test_aretrieve_delegates_to_orchestrator(self, test_config):
        service, orchestrator = self._make_retrieval_service(test_config)
        await service.aretrieve("test query")
        orchestrator.aretrieve.assert_awaited_once()

    async def test_aretrieve_tags_all_context_rows_with_workspace(self, test_config) -> None:
        from dlightrag.engine.rag.retrieval import RetrievalResult

        service, orchestrator = self._make_retrieval_service(test_config)
        service._lightrag_stores = None
        orchestrator.aretrieve.return_value = RetrievalResult(
            contexts={
                "chunks": [
                    {
                        "chunk_id": "shared-hash",
                        "reference_id": "1",
                        "file_path": "/docs/report.pdf",
                        "content": "Evidence",
                    }
                ],
                "entities": [
                    {
                        "entity_name": "Revenue",
                        "description": "Metric",
                        "source_id": "shared-hash",
                    }
                ],
                "relationships": [
                    {
                        "src_id": "Acme",
                        "tgt_id": "Revenue",
                        "source_id": "shared-hash",
                    }
                ],
            }
        )

        result = await service.aretrieve("test query")

        assert {
            row["_workspace"]
            for key in ("chunks", "entities", "relationships")
            for row in result.contexts[key]
        } == {test_config.deployment.workspace}

    async def test_aretrieve_forwards_caller_bm25_query(self, test_config):
        service, orchestrator = self._make_retrieval_service(test_config)
        await service.aretrieve("test query", bm25_query="alpha beta")
        call_kwargs = orchestrator.aretrieve.call_args.kwargs
        assert call_kwargs["bm25_query"] == "alpha beta"

    async def test_aretrieve_without_bm25_query_passes_none(self, test_config):
        service, orchestrator = self._make_retrieval_service(test_config)
        await service.aretrieve("test query")
        call_kwargs = orchestrator.aretrieve.call_args.kwargs
        assert call_kwargs["bm25_query"] is None

    async def test_aretrieve_reranks_after_hydrating_fused_chunks(
        self, test_config, monkeypatch: pytest.MonkeyPatch
    ):
        from dlightrag.engine.rag.retrieval import RetrievalResult

        service, orchestrator = self._make_retrieval_service(test_config)
        orchestrator.aretrieve.return_value = RetrievalResult(
            contexts={
                "chunks": [
                    {"chunk_id": "semantic-a", "content": "semantic"},
                    {"chunk_id": "bm25-visual", "content": "visual text"},
                ],
                "entities": [],
                "relationships": [],
            }
        )

        async def hydrate(_stores, chunks, *, include_image_data=True, cache=None):
            assert include_image_data is True
            chunks[1]["image_data"] = "hydrated-image"

        async def rerank_func(query: str, chunks: list[dict], top_k: int) -> list[dict]:
            assert query == "test query"
            assert top_k == 3
            assert chunks[1]["image_data"] == "hydrated-image"
            return [chunks[1], chunks[0]]

        service._rerank_func = rerank_func
        monkeypatch.setattr(
            "dlightrag.engine.rag.retrieval.provenance.hydrate_lightrag_chunk_provenance",
            hydrate,
        )

        result = await service.aretrieve("test query", chunk_top_k=3)

        assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["bm25-visual", "semantic-a"]
        assert result.trace["reranked_chunk_count"] == 2

    async def test_aretrieve_retains_mix_graph_independently_of_reranked_chunks(
        self, test_config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dlightrag.engine.rag.retrieval import RetrievalResult

        service, orchestrator = self._make_retrieval_service(test_config)
        orchestrator.aretrieve.return_value = RetrievalResult(
            contexts={
                "chunks": [
                    {"chunk_id": "first", "content": "first"},
                    {"chunk_id": "winner", "content": "winner"},
                ],
                "entities": [
                    {"entity_name": "No source"},
                    {"entity_name": "Other source", "source_id": "not-retained"},
                    {"entity_name": "Multi source", "source_id": "first,not-retained"},
                ],
                "relationships": [
                    {"src_id": "A", "tgt_id": "B"},
                    {"src_id": "C", "tgt_id": "D", "source_id": "not-retained"},
                    {
                        "src_id": "E",
                        "tgt_id": "F",
                        "source_id": "winner,not-retained",
                    },
                ],
            }
        )

        async def hydrate(_stores, chunks, *, include_image_data=True, cache=None):
            return None

        async def rerank_func(query: str, chunks: list[dict], top_k: int) -> list[dict]:
            assert query == "test query"
            assert top_k == 1
            return [chunks[1]]

        service._rerank_func = rerank_func
        monkeypatch.setattr(
            "dlightrag.engine.rag.retrieval.provenance.hydrate_lightrag_chunk_provenance",
            hydrate,
        )

        result = await service.aretrieve("test query", chunk_top_k=1)

        assert [chunk["chunk_id"] for chunk in result.contexts["chunks"]] == ["winner"]
        assert [entity["entity_name"] for entity in result.contexts["entities"]] == [
            "No source",
            "Other source",
            "Multi source",
        ]
        assert [(row["src_id"], row["tgt_id"]) for row in result.contexts["relationships"]] == [
            ("A", "B"),
            ("C", "D"),
            ("E", "F"),
        ]
        assert result.trace["reranked_chunk_count"] == 1

    async def test_aretrieve_caps_fused_chunks_when_rerank_disabled(
        self, test_config, monkeypatch: pytest.MonkeyPatch
    ):
        from dlightrag.engine.rag.retrieval import RetrievalResult

        service, orchestrator = self._make_retrieval_service(test_config)
        orchestrator.aretrieve.return_value = RetrievalResult(
            contexts={
                "chunks": [{"chunk_id": f"c{i}", "content": f"c{i}"} for i in range(5)],
                "entities": [],
                "relationships": [],
            }
        )

        async def hydrate(_stores, chunks, *, include_image_data=True, cache=None):
            return None

        service._rerank_func = None  # reranker disabled
        monkeypatch.setattr(
            "dlightrag.engine.rag.retrieval.provenance.hydrate_lightrag_chunk_provenance",
            hydrate,
        )

        result = await service.aretrieve("test query", chunk_top_k=3)

        # Budget is honored even without a reranker: only the top-3 fused chunks
        # survive instead of leaking the full semantic∪BM25 union.
        assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["c0", "c1", "c2"]
        assert result.trace["reranked_chunk_count"] == 3

    async def test_aretrieve_rerank_failure_keeps_hydrated_rrf_top_k(
        self, test_config, monkeypatch: pytest.MonkeyPatch
    ):
        from dlightrag.engine.rag.retrieval import RetrievalResult

        service, orchestrator = self._make_retrieval_service(test_config)
        orchestrator.aretrieve.return_value = RetrievalResult(
            contexts={
                "chunks": [{"chunk_id": f"c{i}", "content": f"c{i}"} for i in range(5)],
                "entities": [],
                "relationships": [],
            }
        )

        async def hydrate(_stores, chunks, *, include_image_data=True, cache=None):
            for chunk in chunks:
                chunk["page_number"] = 7

        async def fail_rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
            from dlightrag.engine.rag.retrieval.rerank import RerankBatchError

            assert query == "test query"
            assert top_k == 3
            assert all(chunk["page_number"] == 7 for chunk in chunks)
            error = RerankBatchError(
                batch_ordinal=2,
                batch_start=7,
                error_type="RuntimeError",
            )
            raise error from RuntimeError("provider unavailable")

        service._rerank_func = fail_rerank
        monkeypatch.setattr(
            "dlightrag.engine.rag.retrieval.provenance.hydrate_lightrag_chunk_provenance",
            hydrate,
        )

        result = await service.aretrieve("test query", chunk_top_k=3)

        assert [chunk["chunk_id"] for chunk in result.contexts["chunks"]] == [
            "c0",
            "c1",
            "c2",
        ]
        assert result.trace["reranked_chunk_count"] == 3
        assert result.trace["rerank_error_type"] == "RuntimeError"
        assert result.trace["rerank_failed_batch"] == 2

    async def test_aretrieve_defers_image_hydration_for_text_reranker(
        self, test_config, monkeypatch: pytest.MonkeyPatch
    ):
        """Text reranker: image bytes are read only for chunks that survive rerank."""
        from dlightrag.engine.rag.retrieval import RetrievalResult

        service, orchestrator = self._make_retrieval_service(test_config)
        service._rerank_consumes_images = False  # text-only reranker
        orchestrator.aretrieve.return_value = RetrievalResult(
            contexts={
                "chunks": [
                    {"chunk_id": "keep", "content": "keep"},
                    {"chunk_id": "drop", "content": "drop"},
                ],
                "entities": [],
                "relationships": [],
            }
        )

        calls: list[tuple[list[str], bool]] = []

        async def hydrate(_stores, chunks, *, include_image_data=True, cache=None):
            calls.append(([c["chunk_id"] for c in chunks], include_image_data))
            if include_image_data:
                for c in chunks:
                    c["image_data"] = f"img-{c['chunk_id']}"

        async def rerank_func(query: str, chunks: list[dict], top_k: int) -> list[dict]:
            # Image bytes are still deferred at rerank time for a text reranker.
            assert "image_data" not in chunks[0]
            return [c for c in chunks if c["chunk_id"] == "keep"]

        service._rerank_func = rerank_func
        monkeypatch.setattr(
            "dlightrag.engine.rag.retrieval.provenance.hydrate_lightrag_chunk_provenance",
            hydrate,
        )

        result = await service.aretrieve("test query", chunk_top_k=3)

        # Pass 1 hydrates metadata only (no image bytes) for the union; pass 2
        # reads image bytes for the single surviving chunk.
        assert calls[0] == (["keep", "drop"], False)
        assert calls[1] == (["keep"], True)
        survivors = result.contexts["chunks"]
        assert [c["chunk_id"] for c in survivors] == ["keep"]
        assert survivors[0]["image_data"] == "img-keep"

    async def test_aretrieve_not_initialized_raises(self, test_config):
        service = _service(test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.aretrieve("query")


# ---------------------------------------------------------------------------
# TestWorkspaceRagFileManagement
# ---------------------------------------------------------------------------


class TestWorkspaceRagFileManagement:
    """Test file deletion delegation."""

    async def test_adelete_not_initialized_raises(self, test_config):
        service = _service(test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.adelete_files(filenames=["a.pdf"])

    async def test_adelete_uses_cascade_pipeline(self, test_config):
        service = _service(test_config)
        service._initialized = True
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock(return_value={"status": "success"})
        service._lightrag.doc_status = MagicMock()
        service._lightrag.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
        service._lightrag.doc_status.get_docs_by_status = AsyncMock(
            return_value={"d1": MagicMock(file_path="/tmp/a.pdf")}
        )
        service._doc_status_lookup = MagicMock()
        service._doc_status_lookup.resolve_deletion_matches = AsyncMock(
            return_value=(SimpleNamespace(doc_id="d1", file_path="/tmp/a.pdf"),)
        )
        result = await service.adelete_files(filenames=["a.pdf"])
        assert result[0]["status"] == "deleted"
        assert result[0]["docs_deleted"] == 1
        service._lightrag.adelete_by_doc_id.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestBuildAddonParams
# ---------------------------------------------------------------------------


class TestBuildAddonParams:
    """Test LightRAG 1.5 addon_params contract."""

    def test_uses_lightrag_15_entity_guidance(self, test_config: DlightragConfig) -> None:
        mutate_config(
            test_config,
            "corpus.retrieval.kg_entity_types",
            ["Product", "Technology", "Organization"],
        )

        result = rag_settings(test_config).addon_params()

        assert result["language"] == "English"
        assert "entity_types" not in result
        assert result["entity_types_guidance"] == (
            "Prioritize domain entities in these categories: Product, Technology, Organization."
        )

    def test_addon_params_include_extraction_prompt_profile(
        self, test_config: DlightragConfig
    ) -> None:
        mutate_config(test_config, "corpus.extraction.language", "Chinese")
        mutate_config(
            test_config, "corpus.extraction.entity_type_prompt_file", "domain-entities.yaml"
        )
        mutate_config(test_config, "corpus.retrieval.kg_entity_types", [])

        result = rag_settings(test_config).addon_params()

        assert result == {
            "language": "Chinese",
            "entity_type_prompt_file": "domain-entities.yaml",
            "chunker": {"paragraph_semantic": {"chunk_token_size": 2000}},
        }


class TestBuildRetrievalBackend:
    """Test DlightRAG retrieval backend configuration wiring."""

    def test_passes_query_budget_config(self, test_config: DlightragConfig) -> None:
        mutate_config(test_config, "corpus.retrieval.max_entity_tokens", 111)
        mutate_config(test_config, "corpus.retrieval.max_relation_tokens", 222)
        mutate_config(test_config, "corpus.retrieval.max_total_tokens", 333)

        backend = WorkspaceRag._build_retrieval_backend(
            rag_settings(test_config),
            lightrag=MagicMock(),
            stores=MagicMock(),
        )

        assert backend._max_entity_tokens == 111
        assert backend._max_relation_tokens == 222
        assert backend._max_total_tokens == 333


class TestDirectImageEmbeddingCapability:
    """Test image embedding capability resolution for the direct-visual leg."""

    async def test_text_only_embedder_disables_direct_image_embedding_without_probe(self) -> None:
        embedder = MagicMock()
        embedder.supports_images = False
        embedder.probe_image_embedding = AsyncMock()

        enabled = await resolve_direct_image_embedding_enabled(
            embedder,
            startup_probe=True,
            require_image_support=False,
        )

        assert enabled is False
        embedder.probe_image_embedding.assert_not_awaited()

    async def test_failed_startup_probe_disables_direct_image_embedding(self) -> None:
        embedder = MagicMock()
        embedder.supports_images = True
        embedder.probe_image_embedding = AsyncMock(side_effect=RuntimeError("image rejected"))

        enabled = await resolve_direct_image_embedding_enabled(
            embedder,
            startup_probe=True,
            require_image_support=False,
        )

        assert enabled is False
        embedder.probe_image_embedding.assert_awaited_once()

    async def test_required_multimodal_rejects_text_only_embedder(self) -> None:
        embedder = MagicMock()
        embedder.supports_images = False
        embedder.probe_image_embedding = AsyncMock()

        with pytest.raises(
            ValueError,
            match="input_modality='multimodal'.*does not support image inputs",
        ):
            await resolve_direct_image_embedding_enabled(
                embedder,
                startup_probe=True,
                require_image_support=True,
            )

        embedder.probe_image_embedding.assert_not_awaited()

    async def test_required_multimodal_probe_failure_is_fatal(self) -> None:
        embedder = MagicMock()
        embedder.supports_images = True
        embedder.probe_image_embedding = AsyncMock(side_effect=RuntimeError("image rejected"))

        with pytest.raises(
            ValueError,
            match="input_modality='multimodal'.*startup probe failed",
        ):
            await resolve_direct_image_embedding_enabled(
                embedder,
                startup_probe=True,
                require_image_support=True,
            )

    async def test_disabled_probe_trusts_resolved_multimodal_capability(self) -> None:
        embedder = MagicMock()
        embedder.supports_images = True
        embedder.probe_image_embedding = AsyncMock()

        enabled = await resolve_direct_image_embedding_enabled(
            embedder,
            startup_probe=False,
            require_image_support=True,
        )

        assert enabled is True
        embedder.probe_image_embedding.assert_not_awaited()

    def test_document_embedder_factory_uses_shared_runtime_limits(
        self,
        test_config: DlightragConfig,
    ) -> None:
        mutate_config(test_config, "models.embedding.dim", 7)
        mutate_config(test_config, "models.embedding.max_concurrency", 5)
        mutate_config(test_config, "corpus.sidecars.vlm.min_image_pixel", 80)
        embedder = MagicMock()
        expected = MagicMock()

        with patch(
            "dlightrag.engine.rag.corpus.ingestion.document_embedding.RobustDocumentEmbedder",
            return_value=expected,
        ) as constructor:
            result = build_document_embedder(
                rag_settings(test_config),
                embedder,
                image_enabled=True,
            )

        assert result is expected
        constructor.assert_called_once_with(
            embedder=embedder,
            image_enabled=True,
            dimension=7,
            min_image_pixel=80,
            batch_size=8,
            max_concurrency=5,
        )


# ---------------------------------------------------------------------------
# TestWorkspaceRagLightRAGMainPath
# ---------------------------------------------------------------------------


class TestWorkspaceRagLightRAGMainPath:
    """Test LightRAG-main path behavior in WorkspaceRag."""

    async def test_initialize_syncs_and_validates_parser_config(
        self, test_config: DlightragConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []
        monkeypatch.setattr(
            DlightragConfig,
            "apply_lightrag_backend_env",
            lambda self, *, force=False: events.append("backend"),
        )
        monkeypatch.setattr(
            DlightragConfig,
            "apply_lightrag_sidecar_env",
            lambda self: events.append("sidecar"),
        )
        monkeypatch.setattr(
            DlightragConfig,
            "apply_lightrag_runtime_env",
            lambda self, *, force=False: events.append("runtime"),
        )
        monkeypatch.setattr(
            "dlightrag.engine.rag.lightrag.patches.apply",
            lambda **_kwargs: events.append("patches"),
        )
        monkeypatch.setattr(
            "lightrag.parser.routing.validate_parser_routing_config",
            lambda rules: events.append(f"validate:{rules}"),
        )
        from dlightrag.adapters.postgres.corpus.corpus import build_pg_corpus_backend

        backend = build_pg_corpus_backend(test_config)
        service = _service(test_config, backend=backend)
        monkeypatch.setattr(service, "_do_initialize_unified", AsyncMock())

        await service._do_initialize()

        assert events == [
            "backend",
            "sidecar",
            "runtime",
            f"validate:{test_config.parser_rules}",
            "patches",
        ]

    async def test_writer_initialization_uses_backend_attach(
        self, test_config: DlightragConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resume_pipeline = AsyncMock()

        class FakeCoordination:
            @asynccontextmanager
            async def pipeline_recovery(self):
                yield

        runtime = _runtime_binder()
        runtime.create.return_value.apipeline_process_enqueue_documents = resume_pipeline
        backend = _backend(normalize_workspace(test_config.deployment.workspace), read_only=False)
        backend.coordination = FakeCoordination()
        backend.runtime = runtime
        service = _service(test_config, backend=backend)

        monkeypatch.setattr(
            "dlightrag.engine.rag.workspace.workspace_rag.LightRagChatModels",
            _FakeChatModels,
        )
        monkeypatch.setattr(
            "dlightrag.engine.rag.workspace.workspace_rag.build_product_reranker",
            lambda *_a, **_k: None,
        )
        monkeypatch.setattr(
            "dlightrag.engine.rag.workspace.workspace_rag.create_embedding_model",
            lambda *_args, **_kwargs: object(),
        )
        monkeypatch.setattr(
            "dlightrag.engine.rag.workspace.workspace_rag.build_lightrag_embedding",
            lambda *_args: object(),
        )
        monkeypatch.setattr(
            "dlightrag.engine.rag.workspace.workspace_rag.resolve_direct_image_embedding_enabled",
            AsyncMock(return_value=False),
        )
        monkeypatch.setattr(
            "dlightrag.engine.rag.workspace.workspace_rag.build_document_embedder",
            lambda *args, **kwargs: object(),
        )
        monkeypatch.setattr(
            WorkspaceRag, "_build_retrieval_backend", lambda self, *args, **kwargs: object()
        )
        with (
            patch(
                "dlightrag.engine.rag.workspace.workspace_rag.rerank_consumes_images",
                return_value=False,
            ),
            patch(
                "dlightrag.engine.rag.retrieval.filtering.FilteredVectorStorage",
                side_effect=lambda **kwargs: kwargs["original"],
            ),
            patch("dlightrag.engine.rag.lightrag.stores.LightRAGStores", return_value=object()),
            patch(
                "dlightrag.engine.rag.corpus.visual_assets.ThumbnailCache", return_value=object()
            ),
            patch(
                "dlightrag.engine.rag.corpus.visual_assets.VisualAssetResolver",
                return_value=object(),
            ),
            patch(
                "dlightrag.engine.rag.retrieval.retriever.UnifiedRetriever", return_value=object()
            ),
        ):
            await service._do_initialize_unified()
            await asyncio.sleep(0)

        # The KG legs resolve chunks through text_chunks, so a document scope
        # only reaches them while this wrapper is installed.
        assert type(service._lightrag.text_chunks).__name__ == "FilteredChunkStore"
        resume_pipeline.assert_awaited_once_with()

    async def test_backend_attach_failure_aborts_initialization(
        self, test_config: DlightragConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate_config(test_config, "deployment.service_role", "reader")
        service = _service(test_config)
        cast(Any, service.backend.runtime.attach).side_effect = RuntimeError("reader attach drift")

        monkeypatch.setattr(
            "dlightrag.engine.rag.workspace.workspace_rag.LightRagChatModels",
            _FakeChatModels,
        )
        monkeypatch.setattr(
            "dlightrag.engine.rag.workspace.workspace_rag.build_product_reranker",
            lambda *_a, **_k: None,
        )
        monkeypatch.setattr(
            "dlightrag.engine.rag.workspace.workspace_rag.create_embedding_model",
            lambda *_args, **_kwargs: object(),
        )
        monkeypatch.setattr(
            "dlightrag.engine.rag.workspace.workspace_rag.build_lightrag_embedding",
            lambda *_args: object(),
        )
        monkeypatch.setattr(
            "dlightrag.engine.rag.workspace.workspace_rag.resolve_direct_image_embedding_enabled",
            AsyncMock(return_value=False),
        )
        monkeypatch.setattr(
            "dlightrag.engine.rag.workspace.workspace_rag.build_document_embedder",
            lambda *args, **kwargs: object(),
        )
        monkeypatch.setattr(
            WorkspaceRag, "_build_retrieval_backend", lambda self, *args, **kwargs: object()
        )
        with (
            patch(
                "dlightrag.engine.rag.workspace.workspace_rag.rerank_consumes_images",
                return_value=False,
            ),
            patch("dlightrag.engine.rag.lightrag.stores.LightRAGStores", return_value=object()),
            patch(
                "dlightrag.engine.rag.corpus.visual_assets.ThumbnailCache", return_value=object()
            ),
            patch(
                "dlightrag.engine.rag.corpus.visual_assets.VisualAssetResolver",
                return_value=object(),
            ),
            patch(
                "dlightrag.engine.rag.retrieval.retriever.UnifiedRetriever", return_value=object()
            ),
        ):
            with pytest.raises(RuntimeError, match="reader attach drift"):
                await service._do_initialize_unified()

    async def test_aingest_azure_blob_single(self, test_config: DlightragConfig) -> None:
        """Downloads one blob into an ephemeral parser item and stores remote metadata."""
        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            assert len(items) == 1
            assert items[0].parser_path.exists()
            return {
                "processed": 1,
                "errors": [],
                "results": [{"doc_id": "d1", "page_count": 2}],
            }

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        mock_source = AsyncMock()
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _doc_id, destination: destination.write_bytes(b"%PDF-fake-content")
        )

        result = await service.aingest(
            source_type="azure_blob",
            container_name="test-container",
            blob_path="docs/report.pdf",
            source=mock_source,
        )

        service._ingestion_engine.aingest_file.assert_not_awaited()
        service._ingestion_engine.aingest_files.assert_awaited_once()
        assert result["doc_id"] == "d1"
        item = seen_items[0]
        assert item.source_uri == "azure://test-container/docs/report.pdf"
        assert item.download_locator == "azure://test-container/docs/report.pdf"
        assert item.display_filename == "report.pdf"
        assert item.parser_path.suffix == ".pdf"
        assert "report" in item.parser_path.name
        assert not item.parser_path.exists()
        assert not (test_config.deployment.working_dir_path / "sources").exists()

    async def test_aingest_unified_azure_blob_batch(self, test_config: DlightragConfig) -> None:
        """Prefix ingest calls the engine once with a prepared batch."""
        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            assert all(item.parser_path.exists() for item in items)
            return {
                "processed": len(items),
                "errors": [],
                "results": [{"doc_id": item.display_filename} for item in items],
            }

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        mock_source = AsyncMock()

        async def _aiter_documents(prefix: str | None = None):
            assert prefix == "docs/"
            yield SourceDocument(key="team-a/report.pdf")
            yield SourceDocument(key="team-b/report.pdf")

        mock_source.aiter_documents = _aiter_documents
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _document, destination: destination.write_bytes(b"%PDF-fake")
        )

        result = await service.aingest(
            source_type="azure_blob",
            container_name="c",
            prefix="docs/",
            source=mock_source,
        )

        service._ingestion_engine.aingest_file.assert_not_awaited()
        service._ingestion_engine.aingest_files.assert_awaited_once()
        assert result["processed"] == 2
        assert [item.source_uri for item in seen_items] == [
            "azure://c/team-a/report.pdf",
            "azure://c/team-b/report.pdf",
        ]
        assert [item.download_locator for item in seen_items] == [
            "azure://c/team-a/report.pdf",
            "azure://c/team-b/report.pdf",
        ]
        assert [item.display_filename for item in seen_items] == ["report.pdf", "report.pdf"]
        assert len({item.parser_path.name for item in seen_items}) == 2
        assert all(item.parser_path.suffix == ".pdf" for item in seen_items)
        assert all(not item.parser_path.exists() for item in seen_items)

    async def test_aingest_unified_s3(self, test_config: DlightragConfig) -> None:
        """S3 single-object ingest uses the same remote batch path."""
        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={
                "processed": 1,
                "errors": [],
                "results": [{"doc_id": "d1", "status": "success"}],
            }
        )

        mock_source = AsyncMock()
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _doc_id, destination: destination.write_bytes(b"%PDF-fake")
        )
        mock_source.aclose = AsyncMock()

        with patch(
            "dlightrag.engine.rag.corpus.sources.aws_s3.S3DataSource", return_value=mock_source
        ):
            result = await service.aingest(
                source_type="s3",
                bucket="my-bucket",
                s3_key="docs/report.pdf",
                replace=True,
            )

        service._ingestion_engine.aingest_file.assert_not_awaited()
        service._ingestion_engine.aingest_files.assert_awaited_once()
        assert result["status"] == "success"

    async def test_aingest_s3_prefix_streams_keys_without_materializing_list(
        self, test_config: DlightragConfig
    ) -> None:
        """S3 prefix ingest consumes keys as they stream."""

        class StreamingS3Source:
            def __init__(self) -> None:
                self.loaded: list[str] = []

            async def aiter_documents(self, prefix: str | None = None):
                assert prefix == "docs/"
                yield SourceDocument(key="docs/a.pdf")
                yield SourceDocument(key="docs/b.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                self.loaded.append(document.key)
                destination.write_bytes(b"%PDF-fake")

            async def aclose(self) -> None:
                return None

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={
                "processed": 2,
                "errors": [],
                "results": [{"doc_id": "d1"}, {"doc_id": "d2"}],
            }
        )
        source = StreamingS3Source()

        result = await service.aingest(
            source_type="s3",
            bucket="my-bucket",
            prefix="docs/",
            source=source,
        )

        assert result["processed"] == 2
        assert source.loaded == ["docs/a.pdf", "docs/b.pdf"]

    async def test_aingest_s3_prefix_resume_skips_completed_windows(
        self, test_config: DlightragConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recovered remote jobs resume from the next unfinished source window."""
        monkeypatch.setattr(
            "dlightrag.engine.rag.workspace.workspace_rag._REMOTE_INGEST_BATCH_SIZE", 2
        )

        class StreamingS3Source:
            def __init__(self) -> None:
                self.loaded: list[str] = []

            async def aiter_documents(self, prefix: str | None = None):
                assert prefix == "docs/"
                for name in ("a", "b", "c", "d", "e"):
                    yield SourceDocument(key=f"docs/{name}.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                self.loaded.append(document.key)
                destination.write_bytes(b"%PDF-fake")

            async def aclose(self) -> None:
                return None

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            side_effect=lambda items, **_: {
                "processed": len(items),
                "errors": [],
                "results": [{"doc_id": item.display_filename} for item in items],
            }
        )
        source = StreamingS3Source()

        result = await service.aingest(
            source_type="s3",
            bucket="my-bucket",
            prefix="docs/",
            source=source,
            _resume_from_window=2,
        )

        assert result["processed"] == 1
        assert source.loaded == ["docs/e.pdf"]
        service._ingestion_engine.aingest_files.assert_awaited_once()

    async def test_aingest_s3_prefix_progress_includes_download_errors(
        self, test_config: DlightragConfig
    ) -> None:
        class PartiallyFailingS3Source:
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(key="docs/a.pdf")
                yield SourceDocument(key="docs/b.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                if document.key == "docs/b.pdf":
                    raise RuntimeError("download failed")
                destination.write_bytes(b"%PDF-fake")

            async def aclose(self) -> None:
                return None

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": [{"doc_id": "d1"}]}
        )
        progress_events: list[RemoteIngestWindowProgress] = []

        async def _record_progress(progress: RemoteIngestWindowProgress) -> None:
            progress_events.append(progress)

        result = await service.aingest(
            source_type="s3",
            bucket="my-bucket",
            prefix="docs/",
            source=PartiallyFailingS3Source(),
            _progress_callback=_record_progress,
        )

        assert result["processed"] == 1
        assert result["errors"] == ["b.pdf: remote materialization failed"]
        assert len(progress_events) == 1
        progress = progress_events[0]
        assert progress.total_delta == 2
        assert progress.processed_delta == 1
        assert progress.failed_delta == 1
        assert progress.errors == ("b.pdf: remote materialization failed",)

    @pytest.mark.parametrize(
        ("source_type", "source_uri"),
        [
            ("s3", "s3://my-bucket/docs/report.pdf"),
            ("azure_blob", "azure://my-container/docs/report.pdf"),
            ("url", "https://example.com/docs/report.pdf?token=secret"),
        ],
    )
    async def test_remote_source_retention_keeps_workspace_file_and_explicit_contract(
        self,
        test_config: DlightragConfig,
        source_type: str,
        source_uri: str,
    ) -> None:
        class RemoteSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(key="asset-123", display_filename="report.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                destination.write_bytes(b"%PDF-retained")

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            assert items[0].parser_path.exists()
            return {"processed": 1, "errors": [], "results": [{"doc_id": "d1"}]}

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        result = await service.aingest_source(
            RemoteSource(),
            source_type=source_type,
            source_uri_for_key=lambda _key: source_uri,
            retain_source_file=True,
        )

        assert result["processed"] == 1
        item = seen_items[0]
        assert item.parser_path.exists()
        assert item.parser_path.read_bytes() == b"%PDF-retained"
        assert item.source_uri == source_uri
        assert item.download_locator == str(item.parser_path)
        assert item.display_filename == "report.pdf"
        assert item.parser_path.suffix == ".pdf"
        assert item.source_uri != item.download_locator
        assert item.parser_path.is_relative_to(
            test_config.input_dir_path
            / test_config.deployment.workspace
            / "__remote_sources__"
            / source_type
        )
        assert "__remote_ingest__" not in str(item.parser_path)

    async def test_remote_source_retention_call_override_can_disable_config(
        self, test_config: DlightragConfig
    ) -> None:
        mutate_config(test_config, "corpus.ingestion.retain_remote_source_files", True)

        class RemoteSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(key="docs/report.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                destination.write_bytes(b"%PDF-transient")

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            assert items[0].parser_path.exists()
            return {"processed": 1, "errors": [], "results": [{"doc_id": "d1"}]}

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        result = await service.aingest_source(
            RemoteSource(),
            source_type="s3",
            source_uri_for_key=lambda _key: "s3://my-bucket/docs/report.pdf",
            download_uri_for_key=lambda _key: "s3://my-bucket/docs/report.pdf",
            retain_source_file=False,
        )

        assert result["processed"] == 1
        item = seen_items[0]
        assert item.source_uri == "s3://my-bucket/docs/report.pdf"
        assert item.download_locator == "s3://my-bucket/docs/report.pdf"
        assert not item.parser_path.exists()

    async def test_aingest_source_accepts_sdk_async_data_source(
        self, test_config: DlightragConfig
    ) -> None:
        """SDK callers can ingest custom connector output without pretending it is S3."""

        class BynderSource(AsyncDataSource):
            def __init__(self) -> None:
                self.loaded: list[str] = []
                self.closed = False

            async def aiter_documents(
                self, prefix: str | None = None
            ) -> AsyncIterator[SourceDocument]:
                assert prefix == "approved/"
                yield SourceDocument(key="asset-123/report.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                self.loaded.append(document.key)
                destination.write_bytes(b"%PDF-fake")

            async def aclose(self) -> None:
                self.closed = True

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            return {
                "processed": len(items),
                "errors": [],
                "results": [{"doc_id": item.display_filename} for item in items],
            }

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)
        source = BynderSource()

        result = await service.aingest_source(
            source,
            source_type="bynder",
            prefix="approved/",
            source_uri_for_key=lambda key: f"bynder://assets/{key}",
            download_uri_for_key=lambda key: f"https://cdn.example.com/assets/{Path(key).name}",
            title="Approved asset",
            metadata={"collection": "marketing"},
        )

        assert result["processed"] == 1
        assert source.loaded == ["asset-123/report.pdf"]
        assert source.closed is True
        assert seen_items[0].source_uri == "bynder://assets/asset-123/report.pdf"
        assert seen_items[0].download_locator == ("https://cdn.example.com/assets/report.pdf")
        assert seen_items[0].display_filename == "report.pdf"
        assert seen_items[0].parser_path.suffix == ".pdf"

    async def test_custom_non_retained_source_without_download_uri_fails_before_materialize(
        self, test_config: DlightragConfig
    ) -> None:
        materialized = False

        class BynderSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(key="asset/report.pdf", source_uri="bynder://asset/1")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                nonlocal materialized
                materialized = True

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": []}
        )

        result = await service.aingest_source(
            BynderSource(),
            source_type="bynder",
            retain_source_file=False,
        )

        assert result["processed"] == 0
        assert result["errors"] == [
            "retain_source_file=false requires a durable download_uri for source report.pdf; "
            "provide download_uri/download_uri_for_key or enable retain_source_file"
        ]
        assert materialized is False

    async def test_signed_url_requires_retention_or_explicit_download_uri(
        self, test_config: DlightragConfig
    ) -> None:
        class SignedUrlSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(
                    key="report.pdf",
                    source_uri="https://fetch.example.com/report.pdf?token=secret",
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                raise AssertionError("materialization must not run")

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": []}
        )

        result = await service.aingest_source(
            SignedUrlSource(),
            source_type="url",
            retain_source_file=False,
        )

        assert result["processed"] == 0
        assert "durable download_uri" in result["errors"][0]
        assert "token=secret" not in result["errors"][0]

    async def test_custom_non_retained_source_accepts_separate_download_uri(
        self, test_config: DlightragConfig
    ) -> None:
        class BynderSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(
                    key="asset/report.pdf",
                    source_uri="bynder://asset/1",
                    download_uri="https://cdn.example.com/assets/1.pdf",
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"%PDF-fake")

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": []}
        )

        result = await service.aingest_source(
            BynderSource(),
            source_type="bynder",
            download_uri_for_key=lambda _key: "s3://fallback/report.pdf",
            retain_source_file=False,
        )

        assert result["processed"] == 1
        await_args = service._ingestion_engine.aingest_files.await_args
        assert await_args is not None
        prepared = await_args.args[0][0]
        assert prepared.source_uri == "bynder://asset/1"
        assert prepared.download_locator == "https://cdn.example.com/assets/1.pdf"

    async def test_invalid_download_uri_fails_before_materialize_and_logs_safely(
        self,
        test_config: DlightragConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        materialized = False

        class BynderSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(
                    key="asset/report.pdf",
                    source_uri="bynder://asset/1",
                    download_uri="https://cdn.example.com/assets/1.pdf?token=secret",
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                nonlocal materialized
                materialized = True

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()

        with caplog.at_level(logging.INFO, logger="dlightrag.engine.rag.workspace.workspace_rag"):
            result = await service.aingest_source(
                BynderSource(),
                source_type="bynder",
                retain_source_file=False,
            )

        assert result["processed"] == 0
        assert result["errors"] == [
            "invalid durable download_uri for source report.pdf; "
            "provide a supported durable URI or enable retain_source_file"
        ]
        assert materialized is False
        outcome = next(
            record
            for record in caplog.records
            if record.message == "source_download_locator_outcome"
        )
        outcome_fields = vars(outcome)
        assert outcome_fields["outcome"] == "unsupported"
        assert outcome_fields["locator_kind"] == "https"
        assert outcome_fields["source_filename"] == "report.pdf"
        assert "token=secret" not in caplog.text

    async def test_materialization_failure_does_not_expose_remote_exception(
        self,
        test_config: DlightragConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        signed_url = "https://fetch.example.com/report.pdf?token=secret"

        class BynderSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(
                    key=signed_url,
                    source_uri="bynder://asset/1",
                    download_uri="https://cdn.example.com/assets/1.pdf",
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                raise ValueError(f"download failed from {signed_url}")

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()

        with caplog.at_level(logging.WARNING):
            result = await service.aingest_source(
                BynderSource(), source_type="bynder", retain_source_file=False
            )

        assert result["errors"] == ["report.pdf: remote materialization failed"]
        assert "token=secret" not in caplog.text
        assert signed_url not in caplog.text

    async def test_source_uri_callback_failure_is_sanitized_before_bounded_map(
        self,
        test_config: DlightragConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        materialized = False
        secret = "resolver_token=secret"

        class BynderSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(key="asset/report.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                nonlocal materialized
                materialized = True

        def source_uri_for_key(_key: str) -> str:
            raise RuntimeError(f"resolver failed: https://fetch.example.com/?{secret}")

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()

        with caplog.at_level(logging.WARNING):
            result = await service.aingest_source(
                BynderSource(),
                source_type="bynder",
                source_uri_for_key=source_uri_for_key,
                retain_source_file=False,
            )

        assert result["errors"] == ["report.pdf: source_uri resolution failed"]
        assert materialized is False
        assert secret not in caplog.text
        assert all(
            secret not in str(value) for record in caplog.records for value in vars(record).values()
        )

    async def test_remote_replace_uses_only_exact_download_locator_matches(
        self, test_config: DlightragConfig
    ) -> None:
        class BynderSource(AsyncDataSource):
            def __init__(self, asset_id: str) -> None:
                self.asset_id = asset_id

            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(
                    key="asset/report.pdf",
                    source_uri=f"bynder://asset/{self.asset_id}",
                    download_uri=(f"https://cdn.example.com/assets/{self.asset_id}/report.pdf"),
                    display_filename="report.pdf",
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"%PDF-fake")

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": []}
        )
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock(
            return_value=SimpleNamespace(status="success")
        )
        service._lightrag.doc_status = MagicMock()
        service._lightrag.doc_status.get_doc_by_file_path = AsyncMock(
            side_effect=AssertionError("remote replace must not use fuzzy file lookup")
        )
        service._lightrag.doc_status.get_docs_by_status = AsyncMock(
            side_effect=AssertionError("remote replace must not scan same-basename docs")
        )
        service._metadata_index = AsyncMock()

        retained_locators = {
            asset_id: str(
                retained_remote_source_path(
                    input_root=service._workspace_input_root(),
                    source_type="bynder",
                    source_uri=f"bynder://asset/{asset_id}",
                    key="report.pdf",
                )
            )
            for asset_id in ("1", "2")
        }
        archived_locators = {
            asset_id: str(lightrag_archived_source_path(Path(retained_locator)))
            for asset_id, retained_locator in retained_locators.items()
        }

        async def find_exact(locator: str) -> list[str]:
            if locator == "https://cdn.example.com/assets/1/report.pdf":
                return ["doc-exact"]
            if locator == archived_locators["1"]:
                return ["doc-archived-1"]
            if locator == archived_locators["2"]:
                return ["doc-archived-2"]
            return []

        service._metadata_index.find_by_download_locator.side_effect = find_exact

        await service.aingest_source(
            BynderSource("1"),
            source_type="bynder",
            replace=True,
            retain_source_file=False,
        )
        await service.aingest_source(
            BynderSource("2"),
            source_type="bynder",
            replace=True,
            retain_source_file=True,
        )

        locator_calls = service._metadata_index.find_by_download_locator.await_args_list
        assert locator_calls == [
            call("https://cdn.example.com/assets/1/report.pdf"),
            call(retained_locators["1"]),
            call(archived_locators["1"]),
            call("https://cdn.example.com/assets/2/report.pdf"),
            call(retained_locators["2"]),
            call(archived_locators["2"]),
        ]
        first_items = service._ingestion_engine.aingest_files.await_args_list[0].args[0]
        second_items = service._ingestion_engine.aingest_files.await_args_list[1].args[0]
        assert first_items[0].replacement_doc_ids == ("doc-exact", "doc-archived-1")
        assert first_items[0].replacement_ownership == (
            (
                "doc-exact",
                "https://cdn.example.com/assets/1/report.pdf",
                "bynder://asset/1",
            ),
            ("doc-archived-1", archived_locators["1"], "bynder://asset/1"),
        )
        assert second_items[0].replacement_doc_ids == ("doc-archived-2",)
        assert second_items[0].replacement_ownership == (
            ("doc-archived-2", archived_locators["2"], "bynder://asset/2"),
        )
        assert all(
            awaited.kwargs["replace"] is True
            for awaited in service._ingestion_engine.aingest_files.await_args_list
        )
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()
        service._lightrag.doc_status.get_doc_by_file_path.assert_not_awaited()
        service._lightrag.doc_status.get_docs_by_status.assert_not_awaited()

    async def test_aingest_source_accepts_per_document_metadata(
        self, test_config: DlightragConfig
    ) -> None:
        class BynderSource(AsyncDataSource):
            async def aiter_documents(
                self, prefix: str | None = None
            ) -> AsyncIterator[SourceDocument]:
                assert prefix == "approved/"
                yield SourceDocument(
                    key="asset-123/report.pdf",
                    source_uri="bynder://assets/asset-123",
                    download_uri="https://cdn.example.com/assets/asset-123.pdf",
                    display_filename="report.pdf",
                    title="Asset report",
                    metadata={"department": "Legal", "asset_id": "asset-123"},
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                assert document.key == "asset-123/report.pdf"
                destination.write_bytes(b"%PDF-fake")

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **kwargs: object) -> dict[str, object]:
            seen_items.extend(items)
            assert kwargs["metadata"] == {"source_system": "bynder", "department": "Marketing"}
            return {"processed": len(items), "errors": [], "results": []}

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        result = await service.aingest_source(
            BynderSource(),
            source_type="bynder",
            prefix="approved/",
            metadata={"source_system": "bynder", "department": "Marketing"},
        )

        assert result["processed"] == 1
        assert seen_items[0].source_uri == "bynder://assets/asset-123"
        assert seen_items[0].download_locator == ("https://cdn.example.com/assets/asset-123.pdf")
        assert seen_items[0].display_filename == "report.pdf"
        assert seen_items[0].title == "Asset report"
        assert seen_items[0].metadata == {"department": "Legal", "asset_id": "asset-123"}

    async def test_remote_parser_path_uses_display_filename_extension(
        self, test_config: DlightragConfig
    ) -> None:
        class BynderSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(
                    key="asset-123",
                    source_uri="bynder://assets/asset-123",
                    download_uri="https://cdn.example.com/assets/asset-123",
                    display_filename="report.pdf",
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                destination.write_bytes(b"%PDF-fake")

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            return {"processed": 1, "errors": [], "results": [{"doc_id": "doc-new"}]}

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        result = await service.aingest_source(
            BynderSource(), source_type="bynder", retain_source_file=False
        )

        assert result["processed"] == 1
        assert seen_items[0].parser_path.suffix == ".pdf"
        assert seen_items[0].display_filename == "report.pdf"

    async def test_aingest_local_manifest_preserves_workspace_relative_path(
        self, test_config: DlightragConfig
    ) -> None:
        input_root = test_config.input_dir_path / test_config.deployment.workspace
        source = input_root / "docs" / "report.pdf"
        explicit = input_root / "docs" / "custom.pdf"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"%PDF-fake")
        explicit.write_bytes(b"%PDF-explicit")
        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            return {"processed": len(items), "errors": [], "results": []}

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        result = await service.aingest(
            source_type="local",
            documents=[
                {"path": str(source), "metadata": {"asset_id": "local-a"}},
                {
                    "path": str(explicit),
                    "source_uri": "local://custom/docs/custom.pdf",
                    "filename": "renamed.pdf",
                },
            ],
        )

        assert result["processed"] == 2
        assert seen_items[0].parser_path == source
        assert (
            seen_items[0].source_uri
            == f"local://{test_config.deployment.workspace}/docs/report.pdf"
        )
        assert seen_items[0].download_locator == str(source)
        assert seen_items[0].metadata == {"asset_id": "local-a"}
        assert seen_items[0].source_uri_explicit is False
        assert seen_items[0].download_locator_explicit is False
        assert seen_items[0].display_filename_explicit is False
        assert seen_items[1].parser_path == explicit
        assert seen_items[1].source_uri == "local://custom/docs/custom.pdf"
        assert seen_items[1].download_locator == str(explicit)
        assert seen_items[1].display_filename == "renamed.pdf"
        assert seen_items[1].source_uri_explicit is True
        assert seen_items[1].download_locator_explicit is False
        assert seen_items[1].display_filename_explicit is True

    async def test_aingest_source_accepts_sync_close(self, test_config: DlightragConfig) -> None:
        class SyncCloseSource(AsyncDataSource):
            def __init__(self) -> None:
                self.closed = False

            async def aiter_documents(
                self, prefix: str | None = None
            ) -> AsyncIterator[SourceDocument]:
                yield SourceDocument(key="asset-123/report.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                destination.write_bytes(b"%PDF-fake")

            def aclose(self) -> None:
                self.closed = True

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": [{"doc_id": "d1"}]}
        )
        source = SyncCloseSource()

        result = await service.aingest_source(
            source,
            source_type="bynder",
            source_uri_for_key=lambda key: f"bynder://assets/{key}",
            download_uri_for_key=lambda _key: "https://cdn.example.com/assets/report.pdf",
        )

        assert result["processed"] == 1
        assert source.closed is True

    async def test_aingest_url_uses_url_data_source(self, test_config: DlightragConfig) -> None:
        """REST/MCP URL jobs enter the same remote ingest pipeline."""
        mutate_config(
            test_config, "corpus.ingestion.url_private_host_allowlist", ("*.corp.example",)
        )
        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={
                "processed": 1,
                "errors": [],
                "results": [{"doc_id": "d1", "status": "success"}],
            }
        )

        mock_source = MagicMock()

        async def _aiter_documents(prefix: str | None = None):
            assert prefix is None
            yield SourceDocument(
                key="getting-started.html",
                source_uri="https://api.bynder.com/docs/getting-started",
            )

        mock_source.aiter_documents = _aiter_documents
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _document, destination: destination.write_bytes(b"<html></html>")
        )
        mock_source.source_uri_for_key = lambda key: "https://api.bynder.com/docs/getting-started"
        mock_source.download_uri_for_key = lambda key: "https://api.bynder.com/docs/getting-started"
        mock_source.aclose = AsyncMock()

        with patch(
            "dlightrag.engine.rag.corpus.sources.url.URLDataSource", return_value=mock_source
        ) as cls:
            result = await service.aingest(
                source_type="url",
                url="https://api.bynder.com/docs/getting-started",
                filename="getting-started.html",
            )

        cls.assert_called_once_with(
            urls=["https://api.bynder.com/docs/getting-started"],
            filename="getting-started.html",
            max_download_bytes=test_config.corpus.ingestion.url_max_bytes,
            allow_private_hosts=("*.corp.example",),
        )
        assert result["status"] == "success"
        await_args = service._ingestion_engine.aingest_files.await_args
        assert await_args is not None
        call_items = await_args.args[0]
        assert call_items[0].source_uri == "https://api.bynder.com/docs/getting-started"
        assert call_items[0].download_locator == ("https://api.bynder.com/docs/getting-started")
        assert call_items[0].display_filename == "getting-started.html"
        mock_source.aclose.assert_awaited_once()

    @pytest.mark.parametrize(
        ("request_fields", "constructor_fields"),
        [
            (
                {
                    "url": "https://fetch.example.com/report.pdf?token=secret",
                    "download_uri": "https://cdn.example.com/report.pdf",
                },
                {"download_uri": "https://cdn.example.com/report.pdf"},
            ),
            (
                {
                    "urls": [
                        "https://fetch.example.com/a.pdf?token=one",
                        "https://fetch.example.com/b.pdf?token=two",
                    ],
                    "download_uris": [
                        "https://cdn.example.com/a.pdf",
                        "https://cdn.example.com/b.pdf",
                    ],
                },
                {
                    "download_uris": [
                        "https://cdn.example.com/a.pdf",
                        "https://cdn.example.com/b.pdf",
                    ]
                },
            ),
        ],
    )
    async def test_aingest_url_forwards_download_shortcuts_and_callbacks(
        self,
        test_config: DlightragConfig,
        request_fields: dict[str, object],
        constructor_fields: dict[str, object],
    ) -> None:
        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service.aingest_source = AsyncMock(  # type: ignore[method-assign]
            return_value={"processed": 1, "errors": [], "results": []}
        )
        mock_source = MagicMock()
        mock_source.source_uri_for_key = MagicMock()
        mock_source.download_uri_for_key = MagicMock()

        with patch(
            "dlightrag.engine.rag.corpus.sources.url.URLDataSource", return_value=mock_source
        ) as cls:
            await service.aingest(source_type="url", **request_fields)

        constructor_call = cls.call_args
        assert constructor_call is not None
        for field, value in constructor_fields.items():
            assert constructor_call.kwargs[field] == value
        delegated_call = service.aingest_source.await_args
        assert delegated_call is not None
        delegated = delegated_call.kwargs
        assert delegated["source_uri_for_key"] is mock_source.source_uri_for_key
        assert delegated["download_uri_for_key"] is mock_source.download_uri_for_key

    async def test_aingest_unified_blob_batch_failure_leaves_no_temp_dirs(
        self, test_config: DlightragConfig
    ) -> None:
        """Remote batch failures do not create obsolete temp directories."""
        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            side_effect=RuntimeError("render failed")
        )

        mock_source = AsyncMock()
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _doc_id, destination: destination.write_bytes(b"%PDF-fake")
        )

        with pytest.raises(RuntimeError, match="render failed"):
            await service.aingest(
                source_type="azure_blob",
                container_name="c",
                blob_path="f.pdf",
                source=mock_source,
            )

        # Verify no temp dirs remain
        import os

        temp_base = test_config.temp_dir
        if temp_base.exists():
            assert len(os.listdir(temp_base)) == 0
        assert not (test_config.deployment.working_dir_path / "sources").exists()

    async def test_aingest_unified_delegates_to_engine(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """Local ingestion stages parser sources before delegating to the unified engine."""
        fake_pdf = tmp_path / "f.pdf"
        fake_pdf.write_bytes(b"%PDF-fake")

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock(
            return_value={"doc_id": "d1", "page_count": 3, "file_path": str(fake_pdf)}
        )

        result = await service.aingest(source_type="local", path=str(fake_pdf))
        service._ingestion_engine.aingest_file.assert_awaited_once()
        staged = test_config.input_dir_path / test_config.deployment.workspace / "f.pdf"
        assert service._ingestion_engine.aingest_file.call_args.args[0] == staged
        assert service._ingestion_engine.aingest_file.call_args.kwargs["source_uri"] == (
            f"local://{test_config.deployment.workspace}/f.pdf"
        )
        assert service._ingestion_engine.aingest_file.call_args.kwargs["download_locator"] == str(
            staged
        )
        assert staged.read_bytes() == b"%PDF-fake"
        assert result["doc_id"] == "d1"
        assert result["page_count"] == 3
        assert (
            service._ingestion_engine.aingest_file.call_args.kwargs["source_uri_explicit"] is False
        )
        assert (
            service._ingestion_engine.aingest_file.call_args.kwargs["download_locator_explicit"]
            is False
        )

    async def test_aingest_local_directory_uses_batch_pipeline(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """Local directory ingestion batches contained files into LightRAG's staged pipeline."""
        docs_dir = tmp_path / "docs"
        nested_dir = docs_dir / "nested"
        upload_tmp_dir = docs_dir / "__uploads__" / "batch"
        nested_dir.mkdir(parents=True)
        upload_tmp_dir.mkdir(parents=True)
        pdf = docs_dir / "b.pdf"
        docx = docs_dir / "a.docx"
        pptx = nested_dir / "c.pptx"
        for path in (pdf, docx, pptx):
            path.write_bytes(b"fake")
        (upload_tmp_dir / "stale.pdf").write_bytes(b"stale")

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            side_effect=lambda paths, **kwargs: {
                "processed": len(paths),
                "errors": [],
                "results": [
                    {"doc_id": item.parser_path.name, "file_path": str(item.parser_path)}
                    for item in paths
                ],
            }
        )

        result = await service.aingest(source_type="local", path=str(docs_dir))

        assert result["processed"] == 3
        assert [item["doc_id"] for item in result["results"]] == ["a.docx", "b.pdf", "c.pptx"]
        staged_root = test_config.input_dir_path / test_config.deployment.workspace
        service._ingestion_engine.aingest_file.assert_not_awaited()
        service._ingestion_engine.aingest_files.assert_awaited_once()
        await_args = service._ingestion_engine.aingest_files.await_args
        assert await_args is not None
        items = list(await_args.args[0])
        assert [item.parser_path for item in items] == [
            staged_root / "a.docx",
            staged_root / "b.pdf",
            staged_root / "nested" / "c.pptx",
        ]
        assert [item.source_uri for item in items] == [
            f"local://{test_config.deployment.workspace}/a.docx",
            f"local://{test_config.deployment.workspace}/b.pdf",
            f"local://{test_config.deployment.workspace}/nested/c.pptx",
        ]
        assert [item.download_locator for item in items] == [
            str(staged_root / "a.docx"),
            str(staged_root / "b.pdf"),
            str(staged_root / "nested" / "c.pptx"),
        ]
        assert all(item.source_uri_explicit is False for item in items)
        assert all(item.download_locator_explicit is False for item in items)
        assert all(item.display_filename_explicit is False for item in items)
        assert (staged_root / "a.docx").read_bytes() == b"fake"
        assert (staged_root / "b.pdf").read_bytes() == b"fake"
        assert (staged_root / "nested" / "c.pptx").read_bytes() == b"fake"

    async def test_aingest_local_directory_offloads_scan_and_staging(
        self,
        test_config: DlightragConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.pdf").write_bytes(b"a")
        (docs_dir / "b.pdf").write_bytes(b"b")
        calls = []

        async def fake_to_thread(func, *args, **kwargs):
            calls.append(func)
            return func(*args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 2, "errors": [], "results": []}
        )

        await service.aingest(source_type="local", path=str(docs_dir))

        assert iter_ingestable_files in calls
        assert calls.count(stage_input_file) == 2

    async def test_aingest_explicit_upload_batch_directory_is_ingestable(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """Web upload batches live under __uploads__ and must still be ingestable."""
        upload_dir = tmp_path / "docs" / "__uploads__" / "batch"
        upload_dir.mkdir(parents=True)
        pdf = upload_dir / "uploaded.pdf"
        pdf.write_bytes(b"%PDF-fake")

        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": []}
        )

        result = await service.aingest(source_type="local", path=str(upload_dir))

        assert result["processed"] == 1
        staged_root = test_config.input_dir_path / test_config.deployment.workspace
        service._ingestion_engine.aingest_files.assert_awaited_once()
        await_args = service._ingestion_engine.aingest_files.await_args
        assert await_args is not None
        item = await_args.args[0][0]
        assert item.parser_path == staged_root / "uploaded.pdf"
        assert item.source_uri == f"local://{test_config.deployment.workspace}/uploaded.pdf"
        assert item.download_locator == str(staged_root / "uploaded.pdf")
        assert (staged_root / "uploaded.pdf").read_bytes() == b"%PDF-fake"

    async def test_aingest_replace_delegates_cleanup_to_ingestion_engine(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """Workspace never deletes ahead of the engine's compensation seam."""
        fake_pdf = tmp_path / "f.pdf"
        fake_pdf.write_bytes(b"%PDF-fake")
        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock(
            return_value={"doc_id": "new-doc", "page_count": 1, "file_path": str(fake_pdf)}
        )
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()

        result = await service.aingest(source_type="local", path=str(fake_pdf), replace=True)

        assert result["doc_id"] == "new-doc"
        assert service._ingestion_engine.aingest_file.await_args.kwargs["replace"] is True
        service._lightrag.adelete_by_doc_id.assert_not_awaited()

    async def test_aretrieve_unified_delegates(self, test_config: DlightragConfig) -> None:
        """aretrieve delegates directly to the retrieval orchestrator."""
        from dlightrag.engine.rag.retrieval import RetrievalResult

        expected = RetrievalResult(
            contexts={"chunks": []},
        )

        service = _service(test_config)
        service._initialized = True
        service._retrieval_orchestrator = MagicMock()
        service._retrieval_orchestrator.aretrieve = AsyncMock(return_value=expected)

        result = await service.aretrieve("test query")
        service._retrieval_orchestrator.aretrieve.assert_awaited_once()
        assert isinstance(result, RetrievalResult)
        assert result.contexts == {"chunks": []}

    async def test_metadata_search_passes_custom_filters_through_verbatim(
        self, test_config: DlightragConfig
    ) -> None:
        """Case folding belongs to the SQL comparison, not to the value in flight."""
        from dlightrag.engine.rag.retrieval import MetadataFilter

        service = _service(test_config)
        service._metadata_index = AsyncMock()
        service._metadata_index.query = AsyncMock(return_value=["doc-1"])

        result = await service.asearch_metadata(MetadataFilter(custom={"department": " Finance "}))

        assert result == ["doc-1"]
        sent_filter = service._metadata_index.query.await_args.args[0]
        assert sent_filter.custom == {"department": " Finance "}

    async def test_metadata_enrichment_uses_full_doc_id_without_path_fallback(
        self, test_config: DlightragConfig
    ) -> None:
        from dlightrag.engine.rag.retrieval import RetrievalResult

        service = _service(test_config)
        service._metadata_index = AsyncMock()
        service._metadata_index.find_by_file_path = AsyncMock()
        service._metadata_index.find_by_filename = AsyncMock()
        service._metadata_index.get = AsyncMock()
        service._metadata_index.get_many = AsyncMock(
            return_value={
                "doc-right": {
                    "filename": "report.pdf",
                    "custom_metadata": {"department": "finance"},
                }
            }
        )
        result = RetrievalResult(
            contexts={
                "chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "full_doc_id": "doc-right",
                        "file_path": "/inputs/default/finance/report.pdf",
                    },
                    {
                        "chunk_id": "chunk-2",
                        "file_path": "/inputs/default/finance/report.pdf",
                    },
                ]
            }
        )

        await service._enrich_chunks_with_metadata(result)

        # Custom metadata is why the column exists: the answer context must see it.
        assert result.contexts["chunks"][0]["metadata"] == {
            "department": "finance",
            "source_file_name": "report.pdf",
        }
        assert "metadata" not in result.contexts["chunks"][1]
        service._metadata_index.get_many.assert_awaited_once_with(["doc-right"])
        service._metadata_index.get.assert_not_awaited()
        service._metadata_index.find_by_file_path.assert_not_awaited()
        service._metadata_index.find_by_filename.assert_not_awaited()

    async def test_metadata_enrichment_surfaces_distinct_source_contract(
        self, test_config: DlightragConfig
    ) -> None:
        from dlightrag.engine.rag.retrieval import RetrievalResult

        service = _service(test_config)
        service._metadata_index = AsyncMock()
        service._metadata_index.get_many = AsyncMock(
            return_value={
                "doc-remote": {
                    "filename": "report.pdf",
                    "source_uri": "bynder://asset/1",
                    "download_locator": "https://cdn.example.com/assets/1.pdf",
                }
            }
        )
        result = RetrievalResult(
            contexts={
                "chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "full_doc_id": "doc-remote",
                        "file_path": "report__a1b2c3d4e5f6.pdf",
                    }
                ]
            }
        )

        await service._enrich_chunks_with_metadata(result)

        meta = result.contexts["chunks"][0]["metadata"]
        assert meta == {
            "source_uri": "bynder://asset/1",
            "source_download_locator": "https://cdn.example.com/assets/1.pdf",
            "source_file_name": "report.pdf",
        }
        assert "download_locator" not in meta

    async def test_get_metadata_hides_internal_paths_and_locator(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "workspace": "finance",
            "doc_id": "doc-1",
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "https://cdn.example.com/assets/1.pdf",
            "_dlightrag_finalization_complete": True,
        }

        result = await service.aget_metadata("doc-1")

        assert result == {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
        }

    async def test_get_metadata_hides_unpublished_rows(self, test_config: DlightragConfig) -> None:
        service = _service(test_config)
        service._metadata_index = AsyncMock()
        for metadata in (
            {"filename": "pending.pdf", "_dlightrag_finalization_complete": False},
            {"filename": "legacy.pdf"},
            None,
        ):
            service._metadata_index.get.return_value = metadata
            assert await service.aget_metadata("doc-hidden") == {}

    async def test_metadata_updates_reject_unpublished_and_concurrently_hidden_rows(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        service._metadata_index = AsyncMock()
        service._metadata_index.is_visible.return_value = False

        with pytest.raises(KeyError):
            await service.aupdate_metadata("doc-hidden", {"team": "core"})
        service._metadata_index.merge_custom_metadata.assert_not_awaited()

        service._metadata_index.is_visible.return_value = True
        service._metadata_index.merge_custom_metadata.return_value = False
        with pytest.raises(KeyError):
            await service.aupdate_metadata("doc-raced", {"team": "core"})

    async def test_failed_doc_pages_hydrate_full_error_rows(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        stores = MagicMock()

        async def _pages(*_args, **_kwargs):
            yield {"doc-failed": object()}

        stores.iter_doc_status_pages = _pages
        stores.get_full_doc_statuses = AsyncMock(
            return_value={
                "doc-failed": SimpleNamespace(
                    file_path="report.pdf",
                    error_msg="parser failed",
                    content_summary="fallback",
                    updated_at="2026-08-27T00:00:00",
                )
            }
        )
        service._lightrag_stores = stores

        pages = [page async for page in service._iter_failed_doc_pages()]

        assert pages == [
            [
                {
                    "doc_id": "doc-failed",
                    "file_path": "report.pdf",
                    "error": "parser failed",
                    "updated_at": "2026-08-27T00:00:00",
                }
            ]
        ]
        stores.get_full_doc_statuses.assert_awaited_once_with(["doc-failed"])

    async def test_retry_failed_docs_requires_initialized_runtime(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await service.aretry_failed_docs()

    async def test_retry_returns_exact_outcomes_within_mutation_bound(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        _set_failed_docs(
            service,
            [{"doc_id": f"doc-{index}", "error": "failed"} for index in range(101)],
        )
        service._metadata_index = None
        service._lightrag = MagicMock()

        result = await service.aretry_failed_docs()

        assert result["retried"] == 101
        assert result["failed"] == 101
        assert len(result["failed_docs"]) == 101
        assert result["details_truncated"] is False

    async def test_recovered_processed_item_reenters_same_id_finalization_seam(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        service._initialized = True
        service._metadata_index = AsyncMock()
        # The durable completion marker is authoritative even if the original
        # download locator is no longer available after restart.
        service._metadata_index.get.return_value = {
            "_dlightrag_finalization_complete": True,
        }
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            return_value={"processed": 1, "doc_id": "doc-committed"}
        )
        service._lightrag_stores = AsyncMock()
        service._lightrag_stores.get_full_doc_statuses.return_value = {
            "doc-committed": SimpleNamespace(
                status="processed",
                file_path="report.pdf",
            )
        }
        outcomes: list[tuple[str, str]] = []

        async def outcome(doc_id: str, state: str, _summary: dict[str, Any]) -> None:
            outcomes.append((doc_id, state))

        result = await service.aretry_failed_docs(
            cohort_doc_ids=("doc-committed",),
            outcome_callback=outcome,
        )

        assert result["retried"] == 1
        assert result["succeeded"] == 1
        assert outcomes == [("doc-committed", "succeeded")]
        service._metadata_index.get.assert_awaited_once_with("doc-committed")
        service._aingest_download_locator.assert_not_awaited()

    async def test_recovered_processed_replay_retires_remaining_locator_owner(
        self, test_config: DlightragConfig
    ) -> None:
        import hashlib

        from lightrag.utils import compute_mdhash_id
        from lightrag.utils_pipeline import normalize_document_file_path

        service = _service(test_config)
        source = service._workspace_input_root() / "report.pdf"
        source.parent.mkdir(parents=True, exist_ok=True)
        content = b"%PDF-1.4 recovered"
        source.write_bytes(content)
        candidate_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
        old_id = "doc-old-owner"
        source_uri = "local://default/report.pdf"
        locator = str(source)
        statuses: dict[str, dict[str, object]] = {
            candidate_id: {
                "status": "processed",
                "file_path": locator,
                "chunks_list": ["chunk-new"],
                "content_hash": f"sha256:{hashlib.sha256(content).hexdigest()}",
            }
        }
        metadata: dict[str, dict[str, object]] = {
            candidate_id: {
                "filename": "report.pdf",
                "source_uri": source_uri,
                "download_locator": locator,
                "_dlightrag_finalization_complete": False,
            },
            old_id: {
                "filename": "old-report.pdf",
                "source_uri": source_uri,
                "download_locator": locator,
            },
        }
        events: list[tuple[str, str, object]] = []

        stores = AsyncMock()
        stores.get_full_doc_statuses.return_value = {
            candidate_id: SimpleNamespace(status="processed", file_path=locator)
        }
        stores.get_doc_status.side_effect = lambda doc_id: statuses.get(doc_id)
        stores.get_full_doc.return_value = {"sidecar_location": None}

        metadata_index = AsyncMock()
        metadata_index.get.side_effect = lambda doc_id: metadata.get(doc_id)
        metadata_index.find_by_download_locator.return_value = [candidate_id, old_id]

        async def upsert_metadata(doc_id: str, row: dict[str, object]) -> None:
            metadata[doc_id] = dict(row)
            events.append(("upsert", doc_id, row.get("_dlightrag_finalization_complete")))

        async def delete_metadata(doc_id: str) -> None:
            metadata.pop(doc_id, None)
            events.append(("delete", doc_id, None))

        metadata_index.upsert.side_effect = upsert_metadata
        metadata_index.delete.side_effect = delete_metadata
        lightrag = MagicMock()
        lightrag.apipeline_enqueue_documents = AsyncMock()
        lightrag.apipeline_process_enqueue_documents = AsyncMock()
        document_embedder = AsyncMock()
        document_embedder.image_enabled = False
        document_embedder.dimension = test_config.models.embedding.dim

        service._initialized = True
        service._lightrag = lightrag
        service._lightrag_stores = stores
        service._metadata_index = metadata_index
        service._ingestion_engine = UnifiedIngestionEngine(
            lightrag=lightrag,
            stores=stores,
            metadata_index=metadata_index,
            document_embedder=document_embedder,
            workspace=test_config.deployment.workspace,
            parser_rules=test_config.corpus.parser_rules,
            chunk_options=dict(test_config.corpus.parser.chunk_options),
        )

        result = await service.aretry_failed_docs(cohort_doc_ids=(candidate_id,))

        assert result["succeeded"] == 1
        assert set(metadata) == {candidate_id}
        retirement = events.index(("delete", old_id, None))
        completion = events.index(("upsert", candidate_id, True))
        assert retirement < completion
        assert metadata[candidate_id]["_dlightrag_finalization_complete"] is True
        assert call(old_id) in metadata_index.get.await_args_list
        lightrag.apipeline_enqueue_documents.assert_not_awaited()
        lightrag.apipeline_process_enqueue_documents.assert_not_awaited()

    async def test_remote_recovered_processed_replay_retires_archived_retained_owner(
        self, test_config: DlightragConfig
    ) -> None:
        import hashlib

        from lightrag.utils import compute_mdhash_id
        from lightrag.utils_pipeline import normalize_document_file_path

        service = _service(test_config)
        source_uri = "bynder://asset/1"
        primary_locator = "https://cdn.example.com/assets/1/report.pdf"
        display_filename = "report.pdf"
        parser_filename = remote_parser_input_path(
            batch_root=Path(), source_uri=source_uri, key=display_filename
        ).name
        parser_path = service._workspace_input_root() / parser_filename
        content = b"%PDF-1.4 remote recovered"
        candidate_id = compute_mdhash_id(normalize_document_file_path(parser_path), prefix="doc-")
        old_id = "doc-archived-owner"
        retained_locator = retained_remote_source_path(
            input_root=service._workspace_input_root(),
            source_type="url",
            source_uri=source_uri,
            key=display_filename,
        )
        archived_locator = str(lightrag_archived_source_path(retained_locator))
        statuses: dict[str, dict[str, object]] = {
            candidate_id: {
                "status": "processed",
                "file_path": str(parser_path),
                "chunks_list": ["chunk-new"],
                "content_hash": f"sha256:{hashlib.sha256(content).hexdigest()}",
            }
        }
        metadata: dict[str, dict[str, object]] = {
            candidate_id: {
                "filename": display_filename,
                "source_uri": source_uri,
                "download_locator": primary_locator,
                "_dlightrag_finalization_complete": False,
            },
            old_id: {
                "filename": "old-report.pdf",
                "source_uri": source_uri,
                "download_locator": archived_locator,
            },
        }
        events: list[tuple[str, str, object]] = []

        stores = AsyncMock()
        stores.get_full_doc_statuses.return_value = {
            candidate_id: SimpleNamespace(status="processed", file_path=str(parser_path))
        }
        stores.get_doc_status.side_effect = lambda doc_id: statuses.get(doc_id)
        stores.get_full_doc.return_value = {"sidecar_location": None}

        metadata_index = AsyncMock()
        metadata_index.get.side_effect = lambda doc_id: metadata.get(doc_id)

        async def find_owners(locator: str) -> list[str]:
            if locator == primary_locator:
                return [candidate_id]
            if locator == archived_locator:
                return [old_id]
            return []

        metadata_index.find_by_download_locator.side_effect = find_owners

        async def upsert_metadata(doc_id: str, row: dict[str, object]) -> None:
            metadata[doc_id] = dict(row)
            events.append(("upsert", doc_id, row.get("_dlightrag_finalization_complete")))

        async def delete_metadata(doc_id: str) -> None:
            metadata.pop(doc_id, None)
            events.append(("delete", doc_id, None))

        metadata_index.upsert.side_effect = upsert_metadata
        metadata_index.delete.side_effect = delete_metadata
        lightrag = MagicMock()
        lightrag.apipeline_enqueue_documents = AsyncMock()
        lightrag.apipeline_process_enqueue_documents = AsyncMock()
        document_embedder = AsyncMock()
        document_embedder.image_enabled = False
        document_embedder.dimension = test_config.models.embedding.dim

        service._initialized = True
        service._lightrag = lightrag
        service._lightrag_stores = stores
        service._metadata_index = metadata_index
        service._ingestion_engine = UnifiedIngestionEngine(
            lightrag=lightrag,
            stores=stores,
            metadata_index=metadata_index,
            document_embedder=document_embedder,
            workspace=test_config.deployment.workspace,
            parser_rules=test_config.corpus.parser_rules,
            chunk_options=dict(test_config.corpus.parser.chunk_options),
        )
        source = MagicMock()

        async def iter_documents(prefix: str | None = None):
            assert prefix is None
            yield SourceDocument(
                key=primary_locator,
                source_uri=source_uri,
                download_uri=primary_locator,
                display_filename=display_filename,
            )

        source.aiter_documents = iter_documents
        source.source_uri_for_key = lambda _key: source_uri
        source.download_uri_for_key = lambda _key: primary_locator
        source.amaterialize_document = AsyncMock(
            side_effect=lambda _document, destination: destination.write_bytes(content)
        )
        source.aclose = AsyncMock()

        with patch("dlightrag.engine.rag.corpus.sources.url.URLDataSource", return_value=source):
            result = await service.aretry_failed_docs(cohort_doc_ids=(candidate_id,))

        assert result["succeeded"] == 1, result
        assert set(metadata) == {candidate_id}
        retirement = events.index(("delete", old_id, None))
        completion = events.index(("upsert", candidate_id, True))
        assert retirement < completion
        assert metadata[candidate_id]["_dlightrag_finalization_complete"] is True
        assert metadata_index.find_by_download_locator.await_args_list == [
            call(primary_locator),
            call(str(retained_locator)),
            call(archived_locator),
        ]
        assert call(old_id) in metadata_index.get.await_args_list
        lightrag.apipeline_enqueue_documents.assert_not_awaited()
        lightrag.apipeline_process_enqueue_documents.assert_not_awaited()

    async def test_recovered_processed_incomplete_metadata_failure_stays_uncertain(
        self, test_config: DlightragConfig
    ) -> None:
        from dlightrag.engine.rag.corpus.ingestion.errors import RetryOutcomeUncertainError

        service = _service(test_config)
        service._initialized = True
        service._metadata_index = AsyncMock()
        service._metadata_index.get.side_effect = RuntimeError("metadata temporarily down")
        service._lightrag_stores = AsyncMock()
        service._lightrag_stores.get_full_doc_statuses.return_value = {
            "doc-committed": SimpleNamespace(status="processed", file_path="report.pdf")
        }
        outcomes: list[tuple[str, str]] = []

        async def outcome(doc_id: str, state: str, _summary: dict[str, Any]) -> None:
            outcomes.append((doc_id, state))

        with pytest.raises(RetryOutcomeUncertainError, match="metadata read failed"):
            await service.aretry_failed_docs(
                cohort_doc_ids=("doc-committed",), outcome_callback=outcome
            )

        assert outcomes == []

    async def test_recovered_processed_invalid_source_preflight_stays_uncertain(
        self, test_config: DlightragConfig
    ) -> None:
        from dlightrag.engine.rag.corpus.ingestion.errors import RetryOutcomeUncertainError

        service = _service(test_config)
        service._initialized = True
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/committed",
            "download_locator": "file:///outside/workspace/report.pdf",
            "_dlightrag_finalization_complete": False,
        }
        service._lightrag_stores = AsyncMock()
        service._lightrag_stores.get_full_doc_statuses.return_value = {
            "doc-committed": SimpleNamespace(status="processed", file_path="report.pdf")
        }
        outcomes: list[tuple[str, str]] = []

        async def outcome(doc_id: str, state: str, _summary: dict[str, Any]) -> None:
            outcomes.append((doc_id, state))

        with pytest.raises(RetryOutcomeUncertainError, match="source preflight failed"):
            await service.aretry_failed_docs(
                cohort_doc_ids=("doc-committed",),
                outcome_callback=outcome,
            )

        assert outcomes == []

    async def test_recovered_cohort_status_read_failure_is_typed_uncertainty(
        self, test_config: DlightragConfig
    ) -> None:
        from dlightrag.engine.rag.corpus.ingestion.errors import RetryOutcomeUncertainError

        service = _service(test_config)
        service._initialized = True
        service._lightrag_stores = AsyncMock()
        service._lightrag_stores.get_full_doc_statuses.side_effect = RuntimeError(
            "status temporarily down"
        )

        with pytest.raises(RetryOutcomeUncertainError, match="cohort status read failed"):
            await service.aretry_failed_docs(cohort_doc_ids=("doc-a",))

    async def test_recovered_cohort_partial_status_read_is_typed_uncertainty(
        self, test_config: DlightragConfig
    ) -> None:
        from dlightrag.engine.rag.corpus.ingestion.errors import RetryOutcomeUncertainError

        service = _service(test_config)
        service._initialized = True
        service._lightrag_stores = AsyncMock()
        service._lightrag_stores.get_full_doc_statuses.return_value = {}
        outcomes: list[tuple[str, str]] = []

        async def outcome(doc_id: str, state: str, _summary: dict[str, Any]) -> None:
            outcomes.append((doc_id, state))

        with pytest.raises(RetryOutcomeUncertainError, match="rows are incomplete"):
            await service.aretry_failed_docs(cohort_doc_ids=("doc-a",), outcome_callback=outcome)

        assert outcomes == []

    async def test_retry_freezes_initial_failed_cohort_before_rows_are_recreated(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        service._initialized = True
        scheduled = [
            {"doc_id": f"doc-{index}", "file_path": f"{index}.pdf", "error": "failed"}
            for index in range(200)
        ]

        async def dynamic_failed_pages() -> AsyncIterator[list[dict[str, Any]]]:
            index = 0
            while index < len(scheduled) and index < 400:
                yield [scheduled[index]]
                index += 1

        service._iter_failed_doc_pages = dynamic_failed_pages  # type: ignore[method-assign]
        service._metadata_index = AsyncMock()
        current_doc_id = ""

        async def metadata(doc_id: str) -> dict[str, str]:
            nonlocal current_doc_id
            current_doc_id = doc_id
            return {
                "filename": f"{doc_id}.pdf",
                "source_uri": f"bynder://asset/{doc_id}",
                "download_locator": f"https://cdn.example.com/{doc_id}.pdf",
            }

        async def retry(*_args: object, **_kwargs: object) -> dict[str, object]:
            scheduled.append(
                {"doc_id": current_doc_id, "file_path": "recreated.pdf", "error": "again"}
            )
            return {"doc_id": current_doc_id}

        service._metadata_index.get.side_effect = metadata
        service._lightrag = MagicMock()
        service._aingest_download_locator = AsyncMock(side_effect=retry)  # type: ignore[attr-defined]

        result = await service.aretry_failed_docs()

        assert result["retried"] == 200
        assert service._aingest_download_locator.await_count == 200  # type: ignore[attr-defined]

    async def test_retry_failed_doc_uses_metadata_locator_not_deleted_parser_path(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        deleted_parser_path = "/srv/dlightrag/inputs/finance/__remote_ingest__/url/batch/report.pdf"
        events: list[str] = []
        _set_failed_docs(
            service,
            [
                {
                    "doc_id": "doc-failed",
                    "file_path": deleted_parser_path,
                    "error": "parser failed",
                }
            ],
        )
        service._metadata_index = AsyncMock()

        async def get_metadata(doc_id: str) -> dict[str, str]:
            assert doc_id == "doc-failed"
            events.append("metadata")
            return {
                "filename": "/private/reports/report.pdf",
                "source_uri": "bynder://asset/1",
                "download_locator": "https://cdn.example.com/assets/1.pdf",
            }

        service._metadata_index.get = AsyncMock(side_effect=get_metadata)
        service._lightrag = MagicMock()

        service._lightrag.adelete_by_doc_id = AsyncMock()

        async def retry_locator(
            source_uri: str,
            download_locator: str,
            filename: str,
            retry_metadata: dict[str, object],
            *,
            track_id: str | None = None,
        ) -> dict[str, str]:
            assert track_id is None
            assert events == ["metadata"]
            assert source_uri == "bynder://asset/1"
            assert download_locator == "https://cdn.example.com/assets/1.pdf"
            assert filename == "report.pdf"
            assert deleted_parser_path not in (source_uri, download_locator)
            assert retry_metadata == {}
            events.append("ingest")
            return {"doc_id": "doc-failed", "status": "success"}

        service._aingest_download_locator = AsyncMock(side_effect=retry_locator)  # type: ignore[attr-defined]
        service.aingest = AsyncMock(side_effect=AssertionError("must not parse doc_status path"))

        result = await service.aretry_failed_docs()

        assert events == ["metadata", "ingest"]
        assert result["retried"] == 1
        assert result["succeeded"] == 1
        assert result["failed"] == 0

    async def test_retry_failed_doc_preserves_allowed_metadata_and_exact_identity(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        _set_failed_docs(
            service,
            [{"doc_id": "doc-same", "file_path": "report.pdf", "error": "parser failed"}],
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "https://cdn.example.com/assets/1.pdf",
            "title": "Annual report",
            "author": "Finance team",
            "creation_date": "2026-01-02",
            "custom_metadata": {"department": "finance"},
            "workspace": "must-not-be-forwarded",
        }
        service._lightrag = MagicMock()

        async def retry_locator(
            source_uri: str,
            download_locator: str,
            filename: str,
            metadata: dict[str, object],
            *,
            track_id: str | None = None,
        ) -> dict[str, object]:
            assert track_id is None
            assert metadata == {
                "title": "Annual report",
                "author": "Finance team",
                "creation_date": "2026-01-02",
                "custom_metadata": {"department": "finance"},
            }
            return {"processed": 1, "errors": [], "results": [{"doc_id": "doc-same"}]}

        service._aingest_download_locator = AsyncMock(side_effect=retry_locator)  # type: ignore[attr-defined]

        result = await service.aretry_failed_docs()

        assert service._aingest_download_locator.await_count == 1, result  # type: ignore[attr-defined]
        assert result["succeeded"] == 1
        assert result["succeeded_docs"] == [
            {"doc_id": "doc-same", "file_path": "report.pdf", "replacement_count": 1}
        ]

    async def test_retry_enqueue_failure_leaves_hidden_finalization_tombstone(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        from lightrag.utils import compute_mdhash_id
        from lightrag.utils_pipeline import normalize_document_file_path

        from dlightrag.engine.rag.corpus.ingestion.errors import RetryOutcomeUncertainError

        service = _service(test_config)
        source = service._workspace_input_root() / "report.pdf"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"%PDF-1.4")
        original_doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
        original_status = {
            "status": "failed",
            "chunks_list": [],
            "content_hash": None,
            "content_summary": "parser failed",
            "error_msg": "parser failed",
        }
        original_metadata = {
            "filename": "report.pdf",
            "source_uri": "local://default/report.pdf",
            "download_locator": str(source),
        }
        statuses = {original_doc_id: dict(original_status)}
        full_docs = {original_doc_id: {"sidecar_location": None}}
        metadata_records: dict[str, dict[str, object]] = {original_doc_id: dict(original_metadata)}

        stores = AsyncMock()
        stores.get_doc_status.side_effect = lambda doc_id: statuses.get(doc_id)
        stores.get_full_doc.side_effect = lambda doc_id: full_docs.get(doc_id)

        async def upsert_status(rows: dict[str, dict[str, object]]) -> None:
            statuses.update({doc_id: dict(row) for doc_id, row in rows.items()})

        stores.doc_status = MagicMock()
        stores.doc_status.upsert = AsyncMock(side_effect=upsert_status)

        metadata_index = AsyncMock()
        metadata_index.get.side_effect = lambda doc_id: metadata_records.get(doc_id)

        async def upsert_metadata(doc_id: str, record: dict[str, object]) -> None:
            metadata_records[doc_id] = dict(record)

        async def delete_metadata(doc_id: str) -> None:
            metadata_records.pop(doc_id, None)

        metadata_index.upsert.side_effect = upsert_metadata
        metadata_index.delete.side_effect = delete_metadata

        lightrag = MagicMock()

        async def delete_document(doc_id: str, *, delete_llm_cache: bool) -> SimpleNamespace:
            assert delete_llm_cache is True
            statuses.pop(doc_id, None)
            full_docs.pop(doc_id, None)
            return SimpleNamespace(status="success")

        lightrag.adelete_by_doc_id = AsyncMock(side_effect=delete_document)
        lightrag.apipeline_enqueue_documents = AsyncMock(side_effect=RuntimeError("enqueue failed"))
        lightrag.apipeline_process_enqueue_documents = AsyncMock()

        service._initialized = True
        service._lightrag = lightrag
        service._lightrag_stores = stores
        service._metadata_index = metadata_index
        document_embedder = AsyncMock()
        document_embedder.image_enabled = False
        document_embedder.dimension = test_config.models.embedding.dim
        service._ingestion_engine = UnifiedIngestionEngine(
            lightrag=lightrag,
            stores=stores,
            metadata_index=metadata_index,
            document_embedder=document_embedder,
            workspace=test_config.deployment.workspace,
            parser_rules=test_config.corpus.parser_rules,
            chunk_options=dict(test_config.corpus.parser.chunk_options),
        )
        _set_failed_docs(
            service,
            [
                {
                    "doc_id": original_doc_id,
                    "file_path": "report.pdf",
                    "error": "parser failed",
                }
            ],
        )

        with pytest.raises(RetryOutcomeUncertainError):
            await service.aretry_failed_docs()

        assert original_doc_id not in statuses
        assert {
            key: metadata_records[original_doc_id][key] for key in original_metadata
        } == original_metadata
        assert metadata_records[original_doc_id]["_dlightrag_finalization_complete"] is False
        lightrag.adelete_by_doc_id.assert_awaited_once_with(original_doc_id, delete_llm_cache=True)

    async def test_remote_retry_reuses_original_parser_identity_and_metadata(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._metadata_index = AsyncMock()
        primary_locator = "https://cdn.example.com/assets/1"
        retained_locator = str(
            retained_remote_source_path(
                input_root=service._workspace_input_root(),
                source_type="url",
                source_uri="bynder://asset/1",
                key="report.pdf",
            )
        )
        service._metadata_index.find_by_download_locator.side_effect = lambda locator: (
            ["doc-new"]
            if locator == primary_locator
            else ["doc-old"]
            if locator == retained_locator
            else []
        )
        seen_items: list[PreparedIngestFile] = []

        async def ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            return {"processed": 1, "errors": [], "results": [{"doc_id": "doc-new"}]}

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=ingest)
        source = MagicMock()

        async def iter_documents(prefix: str | None = None):
            assert prefix is None
            yield SourceDocument(
                key="report.pdf",
                source_uri="bynder://asset/1",
                download_uri="https://cdn.example.com/assets/1",
                display_filename="report.pdf",
            )

        source.aiter_documents = iter_documents
        source.source_uri_for_key = lambda _key: "bynder://asset/1"
        source.download_uri_for_key = lambda _key: "https://cdn.example.com/assets/1"
        source.amaterialize_document = AsyncMock(
            side_effect=lambda _document, destination: destination.write_bytes(b"%PDF-1.4")
        )
        source.aclose = AsyncMock()

        with patch("dlightrag.engine.rag.corpus.sources.url.URLDataSource", return_value=source):
            result = await service._aingest_download_locator(  # type: ignore[attr-defined]
                "bynder://asset/1",
                "https://cdn.example.com/assets/1",
                "report.pdf",
            )

        assert result["doc_id"] == "doc-new"
        assert len(seen_items) == 1
        item = seen_items[0]
        expected_parser_path = remote_parser_input_path(
            batch_root=service._workspace_input_root(),  # type: ignore[attr-defined]
            source_uri="bynder://asset/1",
            key="report.pdf",
        )
        assert item.parser_path == expected_parser_path
        assert item.display_filename == "report.pdf"
        assert item.source_uri == "bynder://asset/1"
        assert item.download_locator == "https://cdn.example.com/assets/1"
        assert item.replacement_doc_ids == ("doc-new", "doc-old")
        assert item.replacement_ownership == (
            ("doc-new", primary_locator, "bynder://asset/1"),
            ("doc-old", retained_locator, "bynder://asset/1"),
        )

    @pytest.mark.parametrize(
        "metadata",
        [
            None,
            {"download_locator": "https://cdn.example.com/assets/1.pdf"},
            {"source_uri": "bynder://asset/1"},
            {
                "source_uri": "bynder://asset/1",
                "download_locator": "https://cdn.example.com/assets/1.pdf",
            },
        ],
    )
    async def test_retry_failed_doc_requires_complete_metadata_contract(
        self,
        test_config: DlightragConfig,
        metadata: dict[str, str] | None,
    ) -> None:
        service = _service(test_config)
        _set_failed_docs(
            service,
            [
                {
                    "doc_id": "doc-failed",
                    "file_path": "https://legacy.example.com/report.pdf",
                    "error": "parser failed",
                }
            ],
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = metadata
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._aingest_download_locator = AsyncMock()  # type: ignore[attr-defined]
        service.aingest = AsyncMock()

        result = await service.aretry_failed_docs()

        assert result["failed"] == 1
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._aingest_download_locator.assert_not_awaited()  # type: ignore[attr-defined]
        service.aingest.assert_not_awaited()

    async def test_retry_failed_doc_rejects_non_raising_ingest_failure(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        _set_failed_docs(
            service,
            [
                {
                    "doc_id": "doc-failed",
                    "error": "parser failed",
                }
            ],
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "https://cdn.example.com/assets/1.pdf",
        }
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                "processed": 0,
                "errors": ["report.pdf: remote materialization failed"],
                "results": [],
            }
        )

        result = await service.aretry_failed_docs()

        assert result["retried"] == 1
        assert result["succeeded"] == 0
        assert result["failed"] == 1
        assert result["failed_docs"] == [
            {"doc_id": "doc-failed", "reason": "retry ingestion failed"}
        ]
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()

    async def test_retry_failed_doc_hides_metadata_load_exception(
        self, test_config: DlightragConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = _service(test_config)
        _set_failed_docs(
            service,
            [
                {
                    "doc_id": "doc-failed",
                    "error": "parser failed",
                }
            ],
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.side_effect = RuntimeError(
            "database failed for https://private?token=secret"
        )
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()

        result = await service.aretry_failed_docs()

        assert result["failed_docs"] == [
            {"doc_id": "doc-failed", "reason": "source metadata unavailable"}
        ]
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        assert "token=secret" not in caplog.text

    async def test_retry_failed_doc_preserves_original_when_ingest_raises(
        self, test_config: DlightragConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = _service(test_config)
        _set_failed_docs(
            service,
            [
                {
                    "doc_id": "doc-failed",
                    "file_path": "/deleted/__remote_ingest__/report.pdf",
                    "error": "parser failed",
                }
            ],
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "https://cdn.example.com/assets/1.pdf",
        }
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._lightrag_stores = AsyncMock()
        service._lightrag_stores.get_doc_status.return_value = {"status": "failed"}
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError(
                "download failed for https://signed.example.com/report?token=secret"
            )
        )

        result = await service.aretry_failed_docs()

        assert result["failed_docs"] == [
            {
                "doc_id": "doc-failed",
                "file_path": "/deleted/__remote_ingest__/report.pdf",
                "reason": "retry ingestion failed",
            }
        ]
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()
        assert "token=secret" not in caplog.text

    async def test_retry_failed_doc_requires_returned_document_id(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        _set_failed_docs(
            service,
            [
                {
                    "doc_id": "doc-failed",
                    "error": "parser failed",
                }
            ],
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "https://cdn.example.com/assets/1.pdf",
        }
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._lightrag_stores = AsyncMock()
        service._lightrag_stores.get_doc_status.return_value = {"status": "failed"}
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            return_value={"processed": 1, "errors": [], "results": []}
        )

        result = await service.aretry_failed_docs()

        assert result["succeeded"] == 0
        assert result["failed"] == 1
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()

    async def test_retry_failed_doc_never_deletes_a_reported_mismatched_document_id(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        _set_failed_docs(service, [{"doc_id": "doc-old", "error": "parser failed"}])
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "s3://documents/assets/1.pdf",
        }
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock(
            return_value=SimpleNamespace(status="success")
        )
        service._lightrag_stores = AsyncMock()
        service._lightrag_stores.get_doc_status.return_value = {"status": "failed"}
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                "processed": 1,
                "errors": [],
                "results": [{"doc_id": "doc-new", "chunks": ["not-persisted"]}],
            }
        )

        result = await service.aretry_failed_docs()

        assert result["succeeded"] == 0
        assert result["failed"] == 1
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()

    async def test_retry_identity_mismatch_overrides_processed_reconciliation(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        _set_failed_docs(
            service,
            [{"doc_id": "doc-expected", "file_path": "report.pdf", "error": "parser failed"}],
        )
        source_contract = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/expected",
            "download_locator": "s3://documents/assets/expected.pdf",
        }
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = source_contract
        processed_row = {
            "status": "processed",
            "file_path": "/inputs/default/report.pdf",
            "chunks_list": ["chunk-expected"],
            "chunks_count": 1,
            "content_hash": "sha256:expected",
            "content_summary": "committed",
            "error_msg": None,
        }
        service._lightrag_stores = AsyncMock()
        service._lightrag_stores.get_doc_status.return_value = processed_row
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                "processed": 1,
                "errors": [],
                "results": [{"doc_id": "doc-unrelated"}],
            }
        )
        outcomes: list[tuple[str, str]] = []

        async def outcome(doc_id: str, state: str, _summary: dict[str, Any]) -> None:
            outcomes.append((doc_id, state))

        result = await service.aretry_failed_docs(outcome_callback=outcome)

        assert result["succeeded"] == 0
        assert result["failed"] == 1
        assert outcomes == [("doc-expected", "failed")]
        service._lightrag_stores.doc_status.upsert.assert_not_awaited()
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()
        service._metadata_index.upsert.assert_not_awaited()

    async def test_retry_rejects_extra_ids_without_deleting_unrelated_documents(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        _set_failed_docs(service, [{"doc_id": "doc-same", "error": "parser failed"}])
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "s3://documents/assets/1.pdf",
        }
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock(
            return_value=SimpleNamespace(status="success")
        )
        service._lightrag_stores = AsyncMock()
        service._lightrag_stores.get_doc_status.return_value = {"status": "failed"}
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                "doc_id": "doc-same",
                "processed": 1,
                "results": [{"doc_id": "doc-extra"}],
            }
        )

        result = await service.aretry_failed_docs()

        assert result["succeeded"] == 0
        assert result["failed"] == 1
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()

    async def test_retry_failed_doc_keeps_old_metadata_for_same_single_doc_id(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        _set_failed_docs(
            service,
            [
                {
                    "doc_id": "doc-same",
                    "file_path": "/inputs/default/report.pdf",
                    "error": "parser failed",
                }
            ],
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "local://default/report.pdf",
            "download_locator": "/inputs/default/report.pdf",
        }
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._validate_retry_source_contract = MagicMock(  # type: ignore[method-assign]
            return_value=("local", {"path": "/inputs/default/report.pdf"})
        )
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            return_value={"doc_id": "doc-same", "source_kind": "document"}
        )

        result = await service.aretry_failed_docs()

        assert result["succeeded"] == 1
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()

    @pytest.mark.parametrize(
        ("download_locator", "source_type", "source_class_path"),
        [
            (
                "https://cdn.example.com/assets/1.pdf",
                "url",
                "dlightrag.engine.rag.corpus.sources.url.URLDataSource",
            ),
            (
                "s3://documents/assets/1.pdf",
                "s3",
                "dlightrag.engine.rag.corpus.sources.aws_s3.S3DataSource",
            ),
            (
                "azure://documents/assets/1.pdf",
                "azure_blob",
                "dlightrag.engine.rag.corpus.sources.azure_blob.AzureBlobDataSource",
            ),
        ],
    )
    async def test_download_locator_dispatch_preserves_remote_source_identity(
        self,
        test_config: DlightragConfig,
        download_locator: str,
        source_type: str,
        source_class_path: str,
    ) -> None:
        service = _service(test_config)
        service._initialized = True
        service._metadata_index = AsyncMock()
        retained_locator = str(
            retained_remote_source_path(
                input_root=service._workspace_input_root(),
                source_type=source_type,
                source_uri="bynder://asset/1",
                key="report.pdf",
            )
        )
        service._metadata_index.find_by_download_locator.side_effect = lambda locator: (
            ["doc-candidate"]
            if locator == download_locator
            else ["doc-old"]
            if locator == retained_locator
            else []
        )
        service._ingestion_engine = AsyncMock()
        service._ingestion_engine.aingest_files.return_value = {
            "processed": 1,
            "errors": [],
            "results": [{"status": "success"}],
        }
        content = b"%PDF-1.4 remote dispatch"
        source = MagicMock()
        source.amaterialize_document = AsyncMock(
            side_effect=lambda _document, destination: destination.write_bytes(content)
        )
        source.aclose = AsyncMock()

        with patch(source_class_path, return_value=source):
            result = await service._aingest_download_locator(  # type: ignore[attr-defined]
                "bynder://asset/1",
                download_locator,
                "report.pdf",
            )

        assert result == {"status": "success"}
        source.amaterialize_document.assert_awaited_once()
        source.aclose.assert_awaited_once()
        assert service._ingestion_engine.aingest_files.await_count == 1
        call = service._ingestion_engine.aingest_files.await_args
        assert call.kwargs["replace"] is False
        assert len(call.args[0]) == 1
        item = call.args[0][0]
        assert isinstance(item, PreparedIngestFile)
        assert item.download_locator == download_locator
        assert item.source_uri == "bynder://asset/1"
        assert item.display_filename == "report.pdf"
        assert item.replacement_doc_ids == ("doc-candidate", "doc-old")
        assert item.replacement_ownership == (
            ("doc-candidate", download_locator, "bynder://asset/1"),
            ("doc-old", retained_locator, "bynder://asset/1"),
        )
        expected_parser_path = (
            service._workspace_input_root()
            / remote_parser_input_path(
                batch_root=Path(), source_uri="bynder://asset/1", key="report.pdf"
            ).name
        )
        assert item.parser_path == expected_parser_path
        # The transient parser source is removed after the engine settles it.
        assert not expected_parser_path.exists()

    async def test_remote_retry_materialization_failure_removes_partial_parser_source(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        service._initialized = True
        service._metadata_index = AsyncMock()
        service._metadata_index.find_by_download_locator.return_value = []
        service._ingestion_engine = AsyncMock()
        parser_path = (
            service._workspace_input_root()
            / remote_parser_input_path(
                batch_root=Path(), source_uri="bynder://asset/1", key="report.pdf"
            ).name
        )
        source = MagicMock()

        async def fail_after_partial_write(_document: object, destination: Path) -> None:
            destination.write_bytes(b"partial")
            raise RuntimeError("download interrupted")

        source.amaterialize_document = AsyncMock(side_effect=fail_after_partial_write)
        source.aclose = AsyncMock()

        with (
            patch(
                "dlightrag.engine.rag.corpus.sources.url.URLDataSource",
                return_value=source,
            ),
            pytest.raises(RuntimeError, match="download interrupted"),
        ):
            await service._aingest_download_locator(  # type: ignore[attr-defined]
                "bynder://asset/1",
                "https://cdn.example.com/assets/1.pdf",
                "report.pdf",
            )

        assert not parser_path.exists()
        source.aclose.assert_awaited_once()
        service._ingestion_engine.aingest_files.assert_not_awaited()

    async def test_download_locator_dispatch_preserves_local_source_identity(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        service = _service(test_config)
        source = service._workspace_input_root() / "report.pdf"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"%PDF-1.4")
        service._metadata_index = AsyncMock()
        service._metadata_index.find_by_download_locator.return_value = [
            "doc-candidate",
            "doc-old",
        ]
        service._ingestion_engine = AsyncMock()
        service._ingestion_engine.aingest_files.return_value = {
            "processed": 1,
            "errors": [],
            "results": [{"status": "success"}],
        }

        result = await service._aingest_download_locator(  # type: ignore[attr-defined]
            "local://legacy/abcdef/report.pdf",
            str(source),
            "report.pdf",
        )

        assert result == {"status": "success"}
        service._ingestion_engine.aingest_file.assert_not_awaited()
        service._ingestion_engine.aingest_files.assert_awaited_once()
        items = service._ingestion_engine.aingest_files.await_args.args[0]
        assert len(items) == 1
        item = items[0]
        assert item.source_uri == "local://legacy/abcdef/report.pdf"
        assert item.download_locator == str(source)
        assert item.display_filename == "report.pdf"
        assert item.parser_path == source
        assert item.parser_path.exists()
        assert item.replacement_doc_ids == ("doc-candidate", "doc-old")
        assert item.replacement_ownership == (
            ("doc-candidate", str(source), "local://legacy/abcdef/report.pdf"),
            ("doc-old", str(source), "local://legacy/abcdef/report.pdf"),
        )

    async def test_download_locator_dispatch_rejects_existing_file_outside_workspace(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        service = _service(test_config)
        outside = tmp_path / "other-workspace" / "report.pdf"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(b"%PDF-private")
        service._ingestion_engine = AsyncMock()

        with pytest.raises(FileNotFoundError, match="download locator is unavailable"):
            await service._aingest_download_locator(  # type: ignore[attr-defined]
                "local://other/report.pdf",
                str(outside),
                "report.pdf",
            )

        service._ingestion_engine.aingest_files.assert_not_awaited()

    async def test_download_locator_dispatch_recovers_lightrag_moved_local_source(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        input_root = service._workspace_input_root()  # type: ignore[attr-defined]
        original = input_root / "report.pdf"
        moved = input_root / "__parsed__" / original.name
        moved.parent.mkdir(parents=True)
        moved.write_bytes(b"%PDF-1.4 moved by parser")
        service._metadata_index = AsyncMock()
        service._metadata_index.find_by_download_locator.return_value = []
        service._ingestion_engine = AsyncMock()
        service._ingestion_engine.aingest_files.return_value = {
            "processed": 1,
            "errors": [],
            "results": [{"status": "success"}],
        }

        result = await service._aingest_download_locator(  # type: ignore[attr-defined]
            "local://default/report.pdf",
            str(original),
            "report.pdf",
        )

        assert result == {"status": "success"}
        items = service._ingestion_engine.aingest_files.await_args.args[0]
        assert len(items) == 1
        assert items[0].download_locator == str(original)
        assert items[0].parser_path == moved
        assert moved.is_file()

    async def test_download_locator_dispatch_rejects_invalid_remote_locator(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        service.aingest = AsyncMock()

        with pytest.raises(ValueError, match="durable download_uri"):
            await service._aingest_download_locator(  # type: ignore[attr-defined]
                "bynder://asset/1",
                "https://cdn.example.com/assets/1.pdf?token=secret",
                "report.pdf",
            )

        service.aingest.assert_not_awaited()

    async def test_download_locator_dispatch_fails_closed_on_owner_lookup_error(
        self, test_config: DlightragConfig
    ) -> None:
        from dlightrag.engine.rag.corpus.ingestion.errors import RetryOutcomeUncertainError

        service = _service(test_config)
        service._metadata_index = AsyncMock()
        service._metadata_index.find_by_download_locator.side_effect = RuntimeError(
            "metadata unavailable"
        )
        service.aingest = AsyncMock()

        with pytest.raises(RetryOutcomeUncertainError, match="ownership lookup failed"):
            await service._aingest_download_locator(  # type: ignore[attr-defined]
                "bynder://asset/1",
                "https://cdn.example.com/assets/1.pdf",
                "report.pdf",
            )

        service.aingest.assert_not_awaited()

    async def test_close_lightrag_main_cleanup(self, test_config: DlightragConfig) -> None:
        """close() finalizes LightRAG storages."""
        service = _service(test_config)
        service._initialized = True
        service._lightrag = AsyncMock()

        await service.aclose()

        service._lightrag.finalize_storages.assert_awaited_once()

    async def test_adelete_files_dry_run_reports_matches_without_mutating(
        self, test_config: DlightragConfig
    ) -> None:
        service = _service(test_config)
        service._initialized = True
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._lightrag.doc_status = MagicMock()
        service._lightrag.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
        service._lightrag.doc_status.get_docs_by_status = AsyncMock(
            return_value={"d1": MagicMock(file_path="/tmp/a.pdf")}
        )
        service._metadata_index = MagicMock()
        service._metadata_index.find_by_download_locator = AsyncMock(return_value=[])
        service._metadata_index.find_by_filename = AsyncMock(return_value=[])
        service._metadata_index.delete = AsyncMock()
        service._doc_status_lookup = MagicMock()
        service._doc_status_lookup.resolve_deletion_matches = AsyncMock(
            return_value=(SimpleNamespace(doc_id="d1", file_path="/tmp/a.pdf"),)
        )

        results = await service.adelete_files(filenames=["a.pdf"], dry_run=True)

        assert results == [
            {
                "identifier": "a.pdf",
                "status": "would_delete",
                "dry_run": True,
                "docs_deleted": 0,
                "errors": [],
                "matched_doc_ids": ["d1"],
                "matched_file_paths": ["/tmp/a.pdf"],
                "sources_used": ["doc_status"],
            }
        ]
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()

    async def test_adelete_files_unified_not_found(self, test_config: DlightragConfig) -> None:
        """Deletion returns not_found when doc is not in doc_status or metadata."""
        service = _service(test_config)
        service._initialized = True
        service._lightrag = MagicMock()
        service._lightrag.doc_status = MagicMock()
        service._lightrag.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
        service._lightrag.doc_status.get_docs_by_status = AsyncMock(return_value={})

        results = await service.adelete_files(filenames=["nonexistent.pdf"])

        assert results[0]["status"] == "not_found"


async def test_retry_status_read_error_stays_uncertain_then_recovers_processed(
    test_config: DlightragConfig,
) -> None:
    from dlightrag.engine.rag.corpus.ingestion.errors import RetryOutcomeUncertainError

    service = _service(test_config)
    _set_failed_docs(service, [{"doc_id": "doc-a", "file_path": "a.pdf"}])
    service._metadata_index = AsyncMock()
    service._metadata_index.get.return_value = {
        "filename": "a.pdf",
        "source_uri": "bynder://asset/a",
        "download_locator": "https://cdn.example.com/a.pdf",
    }
    service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
        side_effect=[
            RuntimeError("late finalization"),
            {"processed": 1, "doc_id": "doc-a"},
        ]
    )
    stores = AsyncMock()
    stores.get_doc_status.side_effect = RuntimeError("status database unavailable")
    stores.get_full_doc_statuses.return_value = {
        "doc-a": SimpleNamespace(status="processed", file_path="a.pdf")
    }
    service._lightrag_stores = stores
    outcomes: list[tuple[str, str]] = []

    async def outcome(doc_id: str, state: str, _summary: dict[str, Any]) -> None:
        outcomes.append((doc_id, state))

    with pytest.raises(RetryOutcomeUncertainError):
        await service.aretry_failed_docs(outcome_callback=outcome)
    assert outcomes == []

    result = await service.aretry_failed_docs(cohort_doc_ids=("doc-a",), outcome_callback=outcome)
    assert result["succeeded"] == 1
    assert result["failed"] == 0
    assert outcomes == [("doc-a", "succeeded")]


async def test_retry_identity_mismatch_records_failed_without_status_write(
    test_config: DlightragConfig,
) -> None:
    service = _service(test_config)
    _set_failed_docs(service, [{"doc_id": "doc-a", "file_path": "a.pdf"}])
    service._metadata_index = AsyncMock()
    service._metadata_index.get.return_value = {
        "filename": "a.pdf",
        "source_uri": "bynder://asset/a",
        "download_locator": "https://cdn.example.com/a.pdf",
    }
    service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
        return_value={"processed": 1, "doc_id": "doc-other"}
    )
    service._lightrag_stores = AsyncMock()
    service._lightrag_stores.get_doc_status.return_value = {
        "status": "processed",
        "chunks_list": ["chunk"],
    }
    service._lightrag_stores.doc_status.upsert.side_effect = RuntimeError("store down")
    outcomes: list[tuple[str, str]] = []

    async def outcome(doc_id: str, state: str, _summary: dict[str, Any]) -> None:
        outcomes.append((doc_id, state))

    result = await service.aretry_failed_docs(outcome_callback=outcome)

    assert result["failed"] == 1
    assert outcomes == [("doc-a", "failed")]
    service._lightrag_stores.doc_status.upsert.assert_not_awaited()
