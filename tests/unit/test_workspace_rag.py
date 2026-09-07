# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Interface contract for one storage-neutral workspace RAG capability."""

from contextlib import asynccontextmanager
from types import MappingProxyType
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

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
    ParserSettings,
    PipelineSettings,
    RagSettings,
)
from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag
from dlightrag.engine.runtime import RunCoordinator
from tests.config_helpers import mutate_config


class _Coordination:
    @asynccontextmanager
    async def workspace_initialization(self):
        yield

    @asynccontextmanager
    async def pipeline_recovery(self):
        yield


def _backend() -> WorkspaceCorpusBackend:
    return WorkspaceCorpusBackend(
        workspace_id="research_team",
        read_only=False,
        coordination=_Coordination(),
        maintenance=AsyncMock(),
        runtime=AsyncMock(),
    )


def _settings() -> RagSettings:
    model = ModelSettings(provider="openai", model="gpt-5.4-mini", api_key="test")
    return RagSettings(
        models=ModelsSettings(
            chat=ModelRoleSettings(default=model),
            embedding=EmbeddingSettings(
                provider="openai_compatible",
                model="text-embedding-3-small",
                api_key="test",
                startup_probe=False,
                max_concurrency=7,
                batch_size=11,
            ),
            rerank=RerankSettings(enabled=False),
        ),
        corpus=CorpusSettings(
            parser=ParserSettings(chunk_options={"paragraph": {"max_tokens": 512}}),
            ingestion=IngestionSettings(
                pipeline=PipelineSettings(
                    max_concurrency=9,
                    max_parallel_insert=2,
                    max_parallel_parse_native=3,
                    max_parallel_parse_mineru=4,
                    max_parallel_parse_docling=5,
                    max_parallel_analyze=6,
                    queue_size_parse=20,
                    queue_size_analyze=30,
                    queue_size_insert=40,
                )
            ),
        ),
    )


def test_rag_settings_are_deeply_immutable() -> None:
    settings = _settings()

    with pytest.raises(ValidationError):
        settings.rag_pipeline_max_async = 10  # type: ignore[misc]
    assert isinstance(settings.chunk_options, MappingProxyType)
    with pytest.raises(TypeError):
        settings.chunk_options["paragraph"] = {}  # type: ignore[index]
    nested = settings.chunk_options["paragraph"]
    assert isinstance(nested, MappingProxyType)
    with pytest.raises(TypeError):
        nested["max_tokens"] = 256  # type: ignore[index]


def test_workspace_rag_constructor_accepts_only_final_collaborators() -> None:
    settings = _settings()
    backend = _backend()

    rag = WorkspaceRag(
        workspace_id="research_team",
        settings=settings,
        backend=backend,
        scheduler=ModelScheduler(max_concurrency=1),
        telemetry=NoopTelemetry(),
        rerank_supports_vision=True,
    )

    assert rag.workspace_id == "research_team"
    assert rag.settings is settings
    assert rag.backend is backend


def test_workspace_rag_rejects_backend_workspace_drift() -> None:
    backend = _backend()
    object.__setattr__(backend, "workspace_id", "other")

    with pytest.raises(ValueError, match="backend workspace"):
        WorkspaceRag(
            workspace_id="research_team",
            settings=_settings(),
            backend=backend,
            scheduler=ModelScheduler(max_concurrency=1),
            telemetry=NoopTelemetry(),
        )


def test_workspace_rag_rejects_backend_role_drift() -> None:
    backend = _backend()
    object.__setattr__(backend, "read_only", True)

    with pytest.raises(ValueError, match="reader role"):
        WorkspaceRag(
            workspace_id="research_team",
            settings=_settings(),
            backend=backend,
            scheduler=ModelScheduler(max_concurrency=1),
            telemetry=NoopTelemetry(),
        )


def test_root_config_maps_independent_rag_pipeline_settings(test_config) -> None:
    from dlightrag.application.settings import rag_settings

    mutate_config(test_config, "models.max_concurrency", 3)
    mutate_config(test_config, "corpus.ingestion.pipeline.max_concurrency", 13)
    mutate_config(test_config, "models.embedding.max_concurrency", 7)
    mutate_config(test_config, "corpus.parser.chunk_options", {"paragraph": {"max_tokens": 384}})

    settings = rag_settings(test_config)

    assert settings.rag_pipeline_max_async == 13
    assert settings.embedding_func_max_async == 7
    assert settings.model_roles.default.model == test_config.models.chat.default.model
    assert settings.embedding.model == test_config.models.embedding.model
    assert settings.chunk_options["paragraph"]["max_tokens"] == 384
    assert test_config.models.max_concurrency == 3


def test_ai_runtime_and_rag_concurrency_owners_vary_independently(test_config) -> None:
    from dlightrag.application.settings import rag_settings

    mutate_config(test_config, "models.max_concurrency", 3)
    mutate_config(test_config, "runtime.query.worker_concurrency", 5)
    mutate_config(test_config, "corpus.ingestion.pipeline.max_concurrency", 13)

    scheduler = ModelScheduler(max_concurrency=test_config.models.max_concurrency)
    coordinator = RunCoordinator(
        store=AsyncMock(),
        executors={"answer": AsyncMock()},
        query_worker_concurrency=test_config.runtime.query.worker_concurrency,
    )

    assert scheduler.max_concurrency == 3
    assert coordinator.query_worker_concurrency == 5
    assert rag_settings(test_config).rag_pipeline_max_async == 13


@pytest.mark.parametrize("workspace_id", ["Research Team", "research-team", "", "../research"])
def test_workspace_rag_rejects_noncanonical_workspace_ids(workspace_id: str) -> None:
    with pytest.raises(ValueError, match="canonical workspace id"):
        WorkspaceRag(
            workspace_id=workspace_id,
            settings=_settings(),
            backend=_backend(),
            scheduler=ModelScheduler(max_concurrency=1),
            telemetry=NoopTelemetry(),
        )
