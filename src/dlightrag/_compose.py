# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Private process composition for a started Application."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dlightrag.application.application import Application, _ApplicationComponents
from dlightrag.application.config import DlightragConfig, get_config
from dlightrag.application.opaque_cursor import CursorSecretBox
from dlightrag.application.skills import skills_bundle_factory
from dlightrag.engine.agent.environment.toolchain import SearchToolchain
from dlightrag.engine.ai.embedding import MultimodalEmbedder
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.telemetry import Telemetry

logger = logging.getLogger(__name__)


def _operational_pool_factory() -> Any:
    """Lazily resolve DlightRAG's process-wide operational pool."""
    from dlightrag.adapters.postgres.core._pool import pg_pool

    return pg_pool.get


def _memory_embedder(
    config: DlightragConfig, *, scheduler: ModelScheduler, telemetry: Telemetry
) -> MultimodalEmbedder:
    """Build the Memory dense leg from DlightRAG's embedding endpoint."""
    from dlightrag.engine.ai.embedding import create_embedding_model

    return create_embedding_model(config.models.embedding, scheduler=scheduler, telemetry=telemetry)


def _actionable_error(exc: Exception) -> str:
    """Return fixed operator guidance without reflecting endpoints or secrets."""
    text = str(exc).lower()
    if "authentication" in text or "password" in text or "denied" in text:
        return "Dependency authentication failed; check deployment credentials."
    if "connection" in text or "timeout" in text or "timed out" in text:
        return "A configured dependency is unreachable; check deployment endpoints."
    return f"Dependency initialization failed ({type(exc).__name__})."


def _initialize_process(config: DlightragConfig) -> None:
    """Initialize tracing and bind the process-wide operational pool."""
    from dlightrag.adapters.observability import init_tracing
    from dlightrag.adapters.postgres.core._pool import pg_pool

    init_tracing(config.observability)
    pg_pool.bind(config)


async def _close_process() -> None:
    """Close process-wide resources while always flushing tracing."""
    from dlightrag.adapters.observability import shutdown_tracing
    from dlightrag.adapters.postgres.core._pool import pg_pool

    try:
        await pg_pool.close()
    finally:
        shutdown_tracing()


def _compose(config: DlightragConfig) -> _ApplicationComponents:
    """Construct this process's collaborators from one resolved configuration."""
    from dlightrag_memory.postgres import PostgresMemoryStore
    from PIL import Image

    from dlightrag.adapters.observability import LangfuseTelemetry
    from dlightrag.adapters.postgres.answer.memory_settings import PGMemorySettingsStore
    from dlightrag.adapters.postgres.corpus.corpus import PGReadinessProbe, build_pg_corpus_backend
    from dlightrag.adapters.postgres.corpus.file_panel import PGFilePanelStore
    from dlightrag.adapters.postgres.corpus.pg_metadata_index import PGMetadataIndex
    from dlightrag.adapters.postgres.corpus.pg_metadata_search import PGMetadataSearchStore
    from dlightrag.adapters.postgres.model_catalogue import PGModelCatalogueStore
    from dlightrag.adapters.postgres.runtime import PGRunBlobStore, PGRunStore
    from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore
    from dlightrag.application.answer_runs import AnswerService
    from dlightrag.application.answer_runs.capabilities import (
        AnswerCapabilityCoordinator,
        AnswerCapabilityView,
    )
    from dlightrag.application.corpus_admin import (
        CorpusAdmin,
        CorpusMutationExecutor,
        CorpusMutationService,
    )
    from dlightrag.application.health import ApplicationHealth
    from dlightrag.application.memory import MemoryService
    from dlightrag.application.model_catalogue import ModelCatalogueAdmin
    from dlightrag.application.retrieval import RetrievalExecutor, RetrievalService
    from dlightrag.application.retrieval._answer_projection import (
        AnswerQueryImagePreparer,
        project_answer_retrieval,
    )
    from dlightrag.application.runs import RunService
    from dlightrag.application.settings import (
        answer_capability_settings,
        answer_executor_settings,
        answer_model_runtime_settings,
        answer_resource_settings,
        corpus_admin_settings,
        model_profile_for_role,
        model_settings_for_role,
        rag_settings,
        rerank_scoring_model_settings,
        retrieval_settings,
    )
    from dlightrag.application.web_conversations import WebConversationService
    from dlightrag.engine.ai.catalog import (
        MODEL_CATALOGUE,
        catalogue_overlay_revision,
        parse_catalogue_overlay,
    )
    from dlightrag.engine.ai.fingerprints import model_fingerprint
    from dlightrag.engine.ai.media import MAX_DECODE_IMAGE_PIXELS
    from dlightrag.engine.ai.scheduler import ModelScheduler
    from dlightrag.engine.ai.settings import MODEL_ROLE_NAMES, ModelRole
    from dlightrag.engine.ai.telemetry import safe_log_text
    from dlightrag.engine.ai.vision import ModelImageCapabilities
    from dlightrag.engine.answer.execution import AnswerExecutor, AnswerResourceResolver
    from dlightrag.engine.answer.model_runtime import AnswerModelRuntime
    from dlightrag.engine.rag.corpus.downloads import SourceDownloadService
    from dlightrag.engine.rag.retrieval.federation import FederatedReranker
    from dlightrag.engine.rag.retrieval.rerank import build_product_reranker
    from dlightrag.engine.rag.retrieval.runtime import RetrievalPlannerRuntime
    from dlightrag.engine.rag.workspace.pool import WorkspacePool
    from dlightrag.engine.rag.workspace.ports import CorpusSchemaError
    from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag
    from dlightrag.engine.rag.workspace.workspaces import normalize_workspace
    from dlightrag.engine.runtime import RunCoordinator, RunExecutor, RunKind

    # Large document scans are DlightRAG product policy, not an AI package import side effect.
    Image.MAX_IMAGE_PIXELS = MAX_DECODE_IMAGE_PIXELS
    startup_catalogue = parse_catalogue_overlay(
        config.models.catalogue_data(),
        source="startup model catalogue",
        path="models.catalogue",
    )
    health = ApplicationHealth(readiness_probe=PGReadinessProbe(config))
    scheduler = ModelScheduler(max_concurrency=config.models.max_concurrency)
    telemetry = LangfuseTelemetry()
    corpus_backend = build_pg_corpus_backend(config)

    # Image capability is role-specific but cached per resolved model config,
    # so roles that share one model share one probe.
    capabilities = AnswerCapabilityCoordinator(
        settings=answer_capability_settings(config),
        profile_for_role=lambda role: model_profile_for_role(config, role),
        model_settings_for_role=lambda role: model_settings_for_role(config, role),
        rerank_model_settings=lambda: rerank_scoring_model_settings(config),
        image_capabilities=ModelImageCapabilities(scheduler=scheduler, telemetry=telemetry),
        on_answer_capability=health.set_answer_image_capability,
    )

    def workspace_config(workspace_id: str) -> DlightragConfig:
        deployment = config.deployment.model_copy(update={"workspace": workspace_id})
        return config.model_copy(update={"deployment": deployment})

    async def build_workspace(workspace_id: str) -> WorkspaceRag:
        resolved = workspace_config(workspace_id)
        settings = rag_settings(resolved)
        backend = build_pg_corpus_backend(resolved)
        try:
            runtime = await WorkspaceRag.acreate(
                workspace_id=workspace_id,
                settings=settings,
                backend=backend,
                scheduler=scheduler,
                telemetry=LangfuseTelemetry(),
                rerank_supports_vision=capabilities.rerank_supports_vision,
            )
        except CorpusSchemaError:
            raise
        except Exception as exc:
            raise RuntimeError(_actionable_error(exc)) from exc
        logger.info("Created WorkspaceRag for workspace '%s'", safe_log_text(workspace_id))
        return runtime

    default_workspace = normalize_workspace(config.deployment.workspace)

    def workspace_unavailable(workspace_id: str) -> None:
        if workspace_id == default_workspace:
            health.mark_component_degraded("corpus_storage")

    def workspace_available(workspace_id: str) -> None:
        if workspace_id == default_workspace:
            health.mark_component_healthy("corpus_storage")

    pool = WorkspacePool(
        build=build_workspace,
        on_workspace_unavailable=workspace_unavailable,
        on_workspace_available=workspace_available,
    )
    cursor_secrets = CursorSecretBox(
        (
            f"{config.storage.postgres.host}\0"
            f"{config.storage.postgres.database}\0"
            f"{config.storage.postgres.password}"
        ).encode()
    )

    source_download_settings = rag_settings(config)
    corpora = CorpusAdmin(
        settings=corpus_admin_settings(config),
        pool=pool,
        maintenance=corpus_backend.maintenance,
        file_panel=PGFilePanelStore(),
        metadata_search=PGMetadataSearchStore(),
        promotion_worker=corpus_backend.promotion,
        source_download_for=lambda workspace: SourceDownloadService(
            settings=source_download_settings,
            metadata_index=PGMetadataIndex(workspace=workspace),
            workspace_id=workspace,
        ),
        # Stable across workers sharing the operational database. The cursor
        # carries no authorization state and expires on credential rotation.
        file_panel_cursor_secret=cursor_secrets.derive("dlightrag-file-panel-cursor"),
        metadata_search_cursor_secret=cursor_secrets.derive("dlightrag-metadata-search-cursor"),
        workspace_catalog_cursor_secret=cursor_secrets.derive("dlightrag-workspace-catalog-cursor"),
    )

    model_catalogue = ModelCatalogueAdmin(
        store=PGModelCatalogueStore(
            initial_revision=catalogue_overlay_revision(()),
            connection_kwargs=config.pg_connection_kwargs(),
        ),
        configured_models=lambda: tuple(
            config.models.chat.resolve(role) for role in MODEL_ROLE_NAMES
        ),
        on_publish=lambda _snapshot: capabilities.invalidate_model_catalogue(),
        read_only=config.is_reader,
    )

    models = AnswerModelRuntime(
        settings=answer_model_runtime_settings(config),
        scheduler=scheduler,
        telemetry=telemetry,
        answer_image_policy=capabilities.answer_image_policy,
        vlm_image_policy=capabilities.vlm_image_policy,
        vlm_profile=lambda: capabilities.model_profile("vlm"),
    )
    resources = AnswerResourceResolver(
        settings=answer_resource_settings(config),
        models=models,
        capabilities=capabilities,
        resource_identity_secret=cursor_secrets.derive("answer-resource-identity"),
        resource_cursor_secret=cursor_secrets.derive("answer-resource-cursor"),
    )
    schema_index = PGMetadataIndex(workspace=normalize_workspace(config.deployment.workspace))

    async def schema_lookup(workspaces: Sequence[str]) -> dict[str, Any]:
        return await schema_index.get_field_schema(workspaces=tuple(workspaces))

    def federated_reranker_factory() -> FederatedReranker | None:
        """Build the shared federation reranker lazily, on first flagged request.

        Deferred past startup so the capabilities vision probe has settled;
        built from the same product reranker settings the workspaces use.
        """
        resolved_rerank = rag_settings(config).rerank
        return build_product_reranker(
            resolved_rerank,
            scoring_settings=rerank_scoring_model_settings(config),
            scheduler=scheduler,
            supports_vision=capabilities.rerank_supports_vision,
            telemetry=telemetry,
        )

    retrieval = RetrievalService(
        pool=pool,
        planners=RetrievalPlannerRuntime(
            model_settings=model_settings_for_role(config, "extract"),
            default_profile=lambda: capabilities.model_profile("extract"),
            scheduler=scheduler,
            telemetry=telemetry,
        ),
        schema_lookup=schema_lookup,
        image_preparer=AnswerQueryImagePreparer(capabilities=capabilities, models=models),
        projector=project_answer_retrieval,
        settings=retrieval_settings(config),
        telemetry=telemetry,
        model_profile_for_role=lambda role: capabilities.model_profile(role),
        model_fingerprint_for_role=lambda role: model_fingerprint(
            model_settings_for_role(config, role)
        ),
        federated_reranker_factory=federated_reranker_factory,
    )

    run_blob_store = PGRunBlobStore()
    run_store = PGRunStore(
        retention_seconds=config.runtime.run_retention_days * 24 * 3600,
        query_max_nonterminal_runs=config.runtime.query.max_nonterminal_runs,
        corpus_mutation_max_nonterminal_runs=(config.runtime.corpus_mutation.max_nonterminal_runs),
        promotion_doc_threshold=config.corpus.promotion.doc_threshold,
        promotion_chunk_threshold=config.corpus.promotion.chunk_threshold,
    )
    memory_embedder = _memory_embedder(config, scheduler=scheduler, telemetry=telemetry)
    memory_store = PostgresMemoryStore(
        pool_factory=_operational_pool_factory(),
        embedder=memory_embedder,
    )
    memory_settings = PGMemorySettingsStore()
    memory = MemoryService(
        memory_store,
        settings_store=memory_settings,
        superseded_retention_days=config.runtime.run_retention_days,
        # Stable across workers sharing the operational database. Cursors
        # carry no authorization state and expire on credential rotation.
        memory_list_cursor_secret=cursor_secrets.derive("dlightrag-memory-list-cursor"),
    )
    from dlightrag.adapters.mcp.outbound import OutboundMcpServer, outbound_mcp_tools

    outbound_tools = outbound_mcp_tools(
        tuple(
            OutboundMcpServer(
                name=server.name,
                transport=server.transport,
                tools=server.tools,
                command=server.command,
                args=server.args,
                url=server.url,
            )
            for server in config.answer.agent.outbound_mcp
        )
    )
    agent_config = config.answer.agent
    search_toolchain = SearchToolchain(
        fd=agent_config.fd_path,
        ripgrep=agent_config.ripgrep_path,
        cache_root=(
            Path(agent_config.search_tool_cache_root)
            if agent_config.search_tool_cache_root is not None
            else None
        ),
        auto_install=agent_config.search_tool_auto_install,
    )
    answer_executor = AnswerExecutor(
        store=run_store,
        blob_store=run_blob_store,
        pool=pool,
        warm=retrieval.warm,
        retrieve=retrieval.retrieve_result,
        planner_history_input_measure=retrieval.planner_history_input_measure,
        models=models,
        capabilities=capabilities,
        resources=resources,
        settings=answer_executor_settings(config),
        telemetry=telemetry,
        model_fingerprint_for_role=lambda role: model_fingerprint(
            model_settings_for_role(config, role)
        ),
        execution_environment=config.answer.agent.execution_environment,
        workspace_root=config.answer.agent.workspace_root,
        search_toolchain=search_toolchain,
        working_dir=config.deployment.working_dir,
        memory_store=memory_store,
        memory_recall_enabled=memory.recall_enabled,
        memory_capability_current=memory.capability_current,
        external_tools=outbound_tools,
        skills_bundle_factory=skills_bundle_factory(config, ensure_dirs=True),
        on_dependency_unavailable=health.mark_component_degraded,
        on_dependency_recovered=health.mark_component_healthy,
    )

    model_roles: dict[str, ModelRole] = {name: name for name in MODEL_ROLE_NAMES}

    def retrieval_model_fingerprint(role: str):
        selected_role = model_roles.get(role)
        if selected_role is None:
            raise ValueError(f"unknown pinned Retrieval model role: {role}")
        return model_fingerprint(model_settings_for_role(config, selected_role))

    retrieval_executor = RetrievalExecutor(
        operation=retrieval,
        timeout_seconds=config.corpus.retrieval.timeout,
        model_fingerprint_for_role=retrieval_model_fingerprint,
        on_dependency_unavailable=health.mark_component_degraded,
        on_dependency_recovered=health.mark_component_healthy,
    )
    executors: dict[RunKind, RunExecutor] = {
        "answer": answer_executor,
        "retrieval": retrieval_executor,
    }
    if not config.is_reader:
        executors["corpus_mutation"] = CorpusMutationExecutor(
            pool=pool,
            maintenance=corpus_backend.maintenance,
            store=run_store,
        )
    coordinator = RunCoordinator(
        store=run_store,
        executors=executors,
        query_worker_concurrency=config.runtime.query.worker_concurrency,
        corpus_mutation_worker_concurrency=(config.runtime.corpus_mutation.worker_concurrency),
    )
    retrieval.bind_runtime(store=run_store, coordinator=coordinator)

    runs = RunService(store=run_store, scheduler=coordinator)
    corpus_mutations = CorpusMutationService(
        input_root=config.input_dir_path,
        store=run_store,
        coordinator=coordinator,
    )

    async def _cancel_local(owner: str, run_id: str) -> None:
        coordinator.cancel_local(owner, run_id)

    cancellation_listener = run_store.build_cancellation_listener(
        worker_id=coordinator.worker_id,
        on_cancel=_cancel_local,
    )

    answers = AnswerService(
        store=run_store,
        blob_store=run_blob_store,
        coordinator=coordinator,
        retrieval=retrieval,
        capabilities=capabilities,
        capability_view=AnswerCapabilityView(capabilities),
        models=models,
        resources=resources,
        model_fingerprint_for_role=lambda role: model_fingerprint(
            model_settings_for_role(config, role)
        ),
        research_tool_supplements=answer_executor.acceptance_research_tools,
        memory_capability=memory.execution_capability,
        # Stable across workers sharing the operational database. Cursors
        # carry no authorization state and expire on credential rotation.
        child_roster_cursor_secret=cursor_secrets.derive("dlightrag-child-roster-cursor"),
        run_retention_seconds=config.runtime.run_retention_days * 24 * 3600,
    )
    web_store = PGWebConversationStore(run_store=run_store)
    web_conversations = WebConversationService(
        store=web_store,
        answers=answers,
        max_attachments=config.answer.generation.max_attachments,
        # Stable across workers sharing the operational database. Cursors
        # carry no authorization state and naturally expire on credential rotation.
        cursor_secret=cursor_secrets.derive("dlightrag-web-conversation-cursor"),
    )
    MODEL_CATALOGUE.replace_startup(startup_catalogue)
    return _ApplicationComponents(
        health=health,
        capabilities=capabilities,
        model_catalogue=model_catalogue,
        pool=pool,
        models=models,
        run_store=run_store,
        web_store=web_store,
        coordinator=coordinator,
        cancellation_listener=cancellation_listener,
        corpora=corpora,
        corpus_mutations=corpus_mutations,
        retrieval=retrieval,
        runs=runs,
        answers=answers,
        memory=memory,
        memory_store=memory_store,
        memory_embedder=memory_embedder,
        web_conversations=web_conversations,
        search_toolchain=search_toolchain,
        initialize_process=_initialize_process,
        close_agent_execution=answer_executor.aclose,
        close_process=_close_process,
    )


async def create_application(
    config: DlightragConfig | None = None,
    *,
    web_enabled: bool = False,
) -> Application:
    """Compose, start, and return one Application."""
    resolved = config or get_config()
    application = Application(resolved, _compose(resolved), web_enabled=web_enabled)
    await application.astart()
    return application


__all__ = ["create_application"]
