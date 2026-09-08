# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Opt-in PostgreSQL 18 + LightRAG main-path smoke tests.

Run with:
    DLIGHTRAG_RUN_E2E_PG18=1 uv run pytest tests/e2e -m e2e_pg18 -q
"""

import asyncio
from pathlib import Path

import pytest

from dlightrag.application.config import set_config
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope
from dlightrag.engine.rag.retrieval.filtering import metadata_filter_scope
from tests.config_helpers import clone_config, mutate_config
from tests.e2e.pg18_harness import (
    RUN_E2E_ENV,
    e2e_enabled,
    fetch_pg_prereq_report,
    install_fake_model_functions,
    isolated_pg18_database,
    make_e2e_config,
    make_workspace_name,
    pg_conn_kwargs_from_env,
    stable_vector,
)

pytestmark = [
    pytest.mark.e2e_pg18,
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not e2e_enabled(),
        reason=f"set {RUN_E2E_ENV}=1 to run PG18 E2E smoke tests",
    ),
]


@pytest.fixture(scope="module", autouse=True)
async def _isolated_pg18_database():
    async with isolated_pg18_database():
        yield


@pytest.fixture
async def pg_conn():
    import asyncpg

    conn = await asyncpg.connect(**pg_conn_kwargs_from_env())
    try:
        yield conn
    finally:
        await conn.close()


async def test_pg18_extensions_and_preload_are_ready(pg_conn) -> None:
    report = await fetch_pg_prereq_report(pg_conn)

    assert report.server_major == 18, report.server_version
    assert report.missing_extensions == []
    assert report.missing_preload_libraries == []


async def test_unified_text_ingest_replace_and_filtered_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.adapters.observability import LangfuseTelemetry
    from dlightrag.adapters.postgres.core._pool import pg_pool
    from dlightrag.adapters.postgres.corpus.corpus import build_pg_corpus_backend
    from dlightrag.adapters.postgres.corpus.file_panel import PGFilePanelStore
    from dlightrag.adapters.postgres.corpus.pg_metadata_index import PGMetadataIndex
    from dlightrag.application.corpus_admin import FilePanelPageRequest
    from dlightrag.application.settings import rag_settings
    from dlightrag.engine.rag.corpus.downloads import (
        LocalDownloadTarget,
        SourceDownloadNotFoundError,
        SourceDownloadService,
    )
    from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag

    conn_kwargs = pg_conn_kwargs_from_env()
    workspace = make_workspace_name()
    cfg = make_e2e_config(
        working_dir=tmp_path / "storage",
        workspace=workspace,
        conn_kwargs=conn_kwargs,
    )
    set_config(cfg)
    install_fake_model_functions(monkeypatch, dim=cfg.models.embedding.dim)

    service = await WorkspaceRag.acreate(
        workspace_id=workspace,
        settings=rag_settings(cfg),
        backend=build_pg_corpus_backend(cfg),
        scheduler=ModelScheduler(max_concurrency=cfg.models.max_concurrency),
        telemetry=LangfuseTelemetry(),
    )
    doc_path = tmp_path / "pg18-native-smoke.md"
    doc_text = (
        "# PG18 native smoke document\n\n"
        "This document proves metadata filtering, BM25 retrieval, vector storage, "
        "and replace semantics against the PostgreSQL 18 LightRAG path.\n"
    )
    doc_path.write_text(doc_text, encoding="utf-8")

    try:
        first = await service.aingest(
            source_type="local",
            path=str(doc_path),
            replace=True,
            title="PG18 E2E Document",
            metadata={"e2e_case": " pg18 "},
        )
        second = await service.aingest(
            source_type="local",
            path=str(doc_path),
            replace=True,
            title="PG18 E2E Document",
            metadata={"e2e_case": " pg18 "},
        )

        doc_id = second["doc_id"]
        chunk_id = second["chunks"][0]
        assert first["doc_id"] == doc_id
        assert first["chunks"] == second["chunks"]

        metadata = await service.aget_metadata(doc_id)
        assert metadata["filename"] == doc_path.name
        assert metadata["title"] == "PG18 E2E Document"

        doc_ids = await service.asearch_metadata(MetadataFilter(custom={"e2e_case": "pg18"}))
        assert doc_ids == [doc_id]

        assert service._metadata_index is not None
        assert service._lightrag_stores is not None
        scope = await service._lightrag_stores.resolve_scope(
            MetadataFilter(custom={"e2e_case": "pg18"})
        )
        assert scope.doc_exists is True
        assert scope.candidate_count >= 1
        assert scope.candidate_count_exact is True

        assert service._bm25 is not None
        bm25_rows = await service._bm25.search(
            "PG18 native smoke document",
            scope=scope,
            top_k=5,
        )
        assert any(row["chunk_id"] == chunk_id for row in bm25_rows)

        assert service._lightrag_stores is not None
        raw_chunks = await service._lightrag_stores.get_text_chunks([chunk_id])
        indexed_text = str(raw_chunks[0]["content"])
        query_embedding = stable_vector(f"document:{indexed_text}", dim=cfg.models.embedding.dim)
        missing_scope = MetadataScope(
            filters=MetadataFilter(filename="missing-doc.pdf"),
            filename_mode="exact",
            doc_exists=True,
            candidate_count=1,
            candidate_count_exact=True,
        )
        async with metadata_filter_scope(missing_scope):
            assert (
                await service._lightrag.chunks_vdb.query(
                    "",
                    top_k=5,
                    query_embedding=query_embedding,
                )
            ) == []

        async with metadata_filter_scope(scope):
            vector_rows = await service._lightrag.chunks_vdb.query(
                "",
                top_k=5,
                query_embedding=query_embedding,
            )
        assert any(row["id"] == chunk_id for row in vector_rows)

        assert (await service.aget_metadata(doc_id))["filename"] == doc_path.name
        downloads = SourceDownloadService(
            settings=service.settings,
            metadata_index=service._metadata_index,
            workspace_id=workspace,
        )
        ready_download = await downloads.prepare(doc_id)
        assert isinstance(ready_download, LocalDownloadTarget)
        assert ready_download.path.is_file()
        ready_files = await PGFilePanelStore().list_processed_files(
            workspace,
            page=FilePanelPageRequest(limit=10),
        )
        assert doc_id in {item.doc_id for item in ready_files.items}

        # Model a hard exit after LightRAG reaches PROCESSED but before the
        # application commit marker. Every direct surface must immediately
        # fail closed while the same durable artifacts remain in PostgreSQL.
        import asyncpg

        conn = await asyncpg.connect(**conn_kwargs)
        try:
            await conn.execute(
                "UPDATE dlightrag_doc_metadata "
                "SET _dlightrag_finalization_complete = FALSE "
                "WHERE workspace = $1 AND doc_id = $2",
                workspace,
                doc_id,
            )
        finally:
            await conn.close()

        assert await service.aget_metadata(doc_id) == {}
        assert await service.asearch_metadata(MetadataFilter()) == []
        assert (await PGMetadataIndex(workspace).get_field_schema())["filters"] == []
        with pytest.raises(SourceDownloadNotFoundError):
            await downloads.prepare(doc_id)
        with pytest.raises(KeyError):
            await service.aupdate_metadata(doc_id, {"team": "hidden"})
        assert service._visual_asset_resolver is not None
        assert await service._visual_asset_resolver.resolve(chunk_id) is None

        hidden_files = await PGFilePanelStore().list_processed_files(
            workspace,
            page=FilePanelPageRequest(limit=10),
        )
        assert doc_id not in {item.doc_id for item in hidden_files.items}
        assert not any(
            row.get("chunk_id") == chunk_id
            for row in await service._bm25.search(
                "PG18 native smoke document",
                scope=None,
                top_k=5,
            )
        )
        assert not any(
            row.get("id") == chunk_id
            for row in await service._lightrag.chunks_vdb.query(
                "",
                top_k=5,
                query_embedding=query_embedding,
            )
        )
        hidden_retrieval = await service.aretrieve(
            "PG18 native smoke document",
            top_k=5,
            chunk_top_k=5,
        )
        assert chunk_id not in {
            row.get("chunk_id") for row in hidden_retrieval.contexts.get("chunks", [])
        }

        # Restore the simulated marker, then prove observable deletion through
        # every leg of the supported default LightRAG PostgreSQL composition.
        conn = await asyncpg.connect(**conn_kwargs)
        try:
            await conn.execute(
                "UPDATE dlightrag_doc_metadata "
                "SET _dlightrag_finalization_complete = TRUE "
                "WHERE workspace = $1 AND doc_id = $2",
                workspace,
                doc_id,
            )
        finally:
            await conn.close()
        assert type(service._lightrag.full_docs).__name__ == "PGKVStorage"
        assert type(service._lightrag.chunks_vdb).__name__ == "FilteredVectorStorage"
        assert type(service._lightrag.chunks_vdb._original).__name__ == "PGVectorStorage"
        assert type(service._lightrag.chunk_entity_relation_graph).__name__ == "PGTableGraphStorage"
        assert type(service._lightrag.doc_status).__name__ == "PGDocStatusStorage"

        graph_entity_ids = ("LightRAG", "PostgreSQL")
        graph_src_id, graph_tgt_id = sorted(graph_entity_ids)
        conn = await asyncpg.connect(**conn_kwargs)
        try:
            graph_nodes_before = await conn.fetch(
                "SELECT namespace, id FROM lightrag_graph_nodes "
                "WHERE workspace = $1 AND id = ANY($2::text[]) "
                "AND properties->>'source_id' = $3",
                workspace,
                list(graph_entity_ids),
                chunk_id,
            )
            graph_edges_before = await conn.fetch(
                "SELECT namespace, src_id, tgt_id FROM lightrag_graph_edges "
                "WHERE workspace = $1 AND src_id = $2 AND tgt_id = $3 "
                "AND properties->>'source_id' = $4",
                workspace,
                graph_src_id,
                graph_tgt_id,
                chunk_id,
            )
        finally:
            await conn.close()
        assert {str(row["id"]) for row in graph_nodes_before} == set(graph_entity_ids)
        assert len(graph_edges_before) == 1
        graph_node_keys = tuple(
            (str(row["namespace"]), str(row["id"])) for row in graph_nodes_before
        )
        graph_edge_keys = tuple(
            (str(row["namespace"]), str(row["src_id"]), str(row["tgt_id"]))
            for row in graph_edges_before
        )

        deleted = await service.adelete_files(file_paths=[doc_id], dry_run=False)
        assert deleted[0]["status"] == "deleted"
        assert deleted[0]["errors"] == []

        async def delete_converged() -> bool:
            conn = await asyncpg.connect(**conn_kwargs)
            try:
                vector_tables = [
                    str(row["tablename"])
                    for row in await conn.fetch(
                        "SELECT tablename FROM pg_tables "
                        "WHERE schemaname = 'public' "
                        "AND tablename LIKE 'lightrag_vdb_chunks_%'"
                    )
                ]
                vector_count = 0
                for table in vector_tables:
                    assert table.replace("_", "").isalnum()
                    vector_count += int(
                        await conn.fetchval(
                            f'SELECT COUNT(*) FROM "{table}" '
                            "WHERE workspace = $1 AND (id = $2 OR full_doc_id = $3)",
                            workspace,
                            chunk_id,
                            doc_id,
                        )
                        or 0
                    )
                graph_count = 0
                for namespace, entity_id in graph_node_keys:
                    graph_count += int(
                        await conn.fetchval(
                            "SELECT COUNT(*) FROM lightrag_graph_nodes "
                            "WHERE workspace = $1 AND namespace = $2 AND id = $3",
                            workspace,
                            namespace,
                            entity_id,
                        )
                        or 0
                    )
                for namespace, src_id, tgt_id in graph_edge_keys:
                    graph_count += int(
                        await conn.fetchval(
                            "SELECT COUNT(*) FROM lightrag_graph_edges "
                            "WHERE workspace = $1 AND namespace = $2 "
                            "AND src_id = $3 AND tgt_id = $4",
                            workspace,
                            namespace,
                            src_id,
                            tgt_id,
                        )
                        or 0
                    )
                counts = {
                    "kv": int(
                        await conn.fetchval(
                            "SELECT COUNT(*) FROM lightrag_doc_full "
                            "WHERE workspace = $1 AND id = $2",
                            workspace,
                            doc_id,
                        )
                        or 0
                    )
                    + int(
                        await conn.fetchval(
                            "SELECT COUNT(*) FROM lightrag_doc_chunks "
                            "WHERE workspace = $1 AND (id = $2 OR full_doc_id = $3)",
                            workspace,
                            chunk_id,
                            doc_id,
                        )
                        or 0
                    ),
                    "vector": vector_count,
                    "graph": graph_count,
                    "doc_status": int(
                        await conn.fetchval(
                            "SELECT COUNT(*) FROM lightrag_doc_status "
                            "WHERE workspace = $1 AND id = $2",
                            workspace,
                            doc_id,
                        )
                        or 0
                    ),
                    "product_metadata": int(
                        await conn.fetchval(
                            "SELECT COUNT(*) FROM dlightrag_doc_metadata "
                            "WHERE workspace = $1 AND doc_id = $2",
                            workspace,
                            doc_id,
                        )
                        or 0
                    ),
                }
                return all(value == 0 for value in counts.values())
            finally:
                await conn.close()

        async with asyncio.timeout(10):
            while not await delete_converged():
                await asyncio.sleep(0.05)
        assert await service.aget_metadata(doc_id) == {}
        assert await service._lightrag_stores.get_text_chunks([chunk_id]) == [None]
        after_delete = await service.aretrieve(
            "native image",
            top_k=5,
            chunk_top_k=5,
        )
        assert chunk_id not in {
            row.get("chunk_id") for row in after_delete.contexts.get("chunks", [])
        }
    finally:
        if service._initialized:
            await service.areset(keep_files=False)
        await service.aclose()
        await pg_pool.close()


async def test_reader_role_attaches_read_only_and_rejects_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader attaches read-only to the corpus, serves reads, and owns run state.

    Writer and reader run sequentially in one process (fully closing between
    phases resets the process-wide LightRAG client and domain pool). The reader
    reads the corpus through read-only sessions while its DlightRAG domain pool
    stays writable for durable Answer run state.
    """
    import uuid

    from dlightrag.adapters.observability import LangfuseTelemetry
    from dlightrag.adapters.postgres.core._pool import pg_pool
    from dlightrag.adapters.postgres.corpus.corpus import build_pg_corpus_backend
    from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
    from dlightrag.application.config import reset_config, set_config
    from dlightrag.application.settings import rag_settings
    from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag
    from dlightrag.engine.runtime import (
        PreparedRunEnvelope,
        RunAccessScope,
        run_request_fingerprint,
    )

    conn_kwargs = pg_conn_kwargs_from_env()
    workspace = make_workspace_name("reader")
    writer_cfg = make_e2e_config(
        working_dir=tmp_path / "storage",
        workspace=workspace,
        conn_kwargs=conn_kwargs,
    )
    set_config(writer_cfg)
    install_fake_model_functions(monkeypatch, dim=writer_cfg.models.embedding.dim)
    model_scheduler = ModelScheduler(max_concurrency=writer_cfg.models.max_concurrency)

    # ── Writer: provision schema + ingest ──────────────────────────────
    writer = await WorkspaceRag.acreate(
        workspace_id=workspace,
        settings=rag_settings(writer_cfg),
        backend=build_pg_corpus_backend(writer_cfg),
        scheduler=model_scheduler,
        telemetry=LangfuseTelemetry(),
    )
    try:
        doc_path = tmp_path / "reader-smoke.md"
        doc_path.write_text(
            "# Reader smoke\n\nA replica reader attaches to the existing schema "
            "and serves stateless reads.\n",
            encoding="utf-8",
        )
        result = await writer.aingest(
            source_type="local",
            path=str(doc_path),
            replace=True,
            title="Reader Smoke",
            metadata={"e2e_case": "reader"},
        )
        doc_id = result["doc_id"]
        chunk_id = result["chunks"][0]
        await PGRunStore().initialize()
    finally:
        await writer.aclose()
        await pg_pool.close()
        reset_config()

    # ── Reader: read-only corpus, writable operational state ───────────
    reader_cfg = clone_config(writer_cfg)
    mutate_config(reader_cfg, "deployment.service_role", "reader")
    set_config(reader_cfg)
    pg_pool.bind(reader_cfg)
    reader = await WorkspaceRag.acreate(
        workspace_id=workspace,
        settings=rag_settings(reader_cfg),
        backend=build_pg_corpus_backend(reader_cfg),
        scheduler=model_scheduler,
        telemetry=LangfuseTelemetry(),
    )
    store = PGRunStore()
    run_owner = "reader-owner"
    created_run_id: str | None = None
    try:
        assert reader.settings.read_only

        retrieval = await reader.aretrieve(
            "reader attaches to the existing schema", top_k=5, chunk_top_k=5
        )
        chunk_ids = {c.get("chunk_id") for c in retrieval.contexts.get("chunks", [])}
        assert chunk_id in chunk_ids

        metadata = await reader.aget_metadata(doc_id)
        assert metadata["title"] == "Reader Smoke"

        # Corpus reads run read-only; the domain pool still accepts run state.
        await store.initialize(validate_only=True)
        run_request = {
            "query": "reader operational write",
            "workspaces": [workspace],
            "agent_session_id": "00000000-0000-7000-8000-000000000001",
            "agent_lane_id": "main",
        }
        run_id = str(uuid.uuid7())
        creation = await store.create_run(
            envelope=PreparedRunEnvelope(
                run_kind="answer",
                lane="query",
                submitted_by=run_owner,
                access_scope=RunAccessScope(kind="owner", scope_id=run_owner),
                submission_key=run_id,
                request_fingerprint=run_request_fingerprint(run_request),
                payload=run_request,
                accepted_input=run_request,
                retention_seconds=365 * 24 * 60 * 60,
            ),
            run_id=run_id,
        )
        created_run_id = creation.run.run_id
        assert await store.get_run(owner_id=run_owner, run_id=created_run_id) is not None

        with pytest.raises(PermissionError):
            await reader.areset()
        with pytest.raises(PermissionError):
            await reader.aupdate_metadata(doc_id, {"note": "nope"})
        with pytest.raises(PermissionError):
            await reader.aingest(source_type="local", path=str(doc_path))
    finally:
        try:
            if created_run_id is not None:
                deletion = await store.delete_runs(
                    owner_id=run_owner,
                    run_ids=[created_run_id],
                )
                assert deletion.runs == 1
        finally:
            await reader.aclose()
            await pg_pool.close()
            reset_config()

    # ── Cleanup: remove the workspace via a writer ─────────────────────
    set_config(writer_cfg)
    pg_pool.bind(writer_cfg)
    cleanup = await WorkspaceRag.acreate(
        workspace_id=workspace,
        settings=rag_settings(writer_cfg),
        backend=build_pg_corpus_backend(writer_cfg),
        scheduler=model_scheduler,
        telemetry=LangfuseTelemetry(),
    )
    try:
        await cleanup.areset(keep_files=False)
    finally:
        await cleanup.aclose()
        await pg_pool.close()
        reset_config()
