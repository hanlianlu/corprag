# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Root adapters from Answer projection policy into inline Retrieval."""

from collections.abc import Mapping, Sequence
from typing import Any

from dlightrag.application.retrieval import (
    ProjectedRetrieval,
    RetrieveProjection,
)
from dlightrag.engine.answer.capabilities import AnswerCapabilityCoordinator
from dlightrag.engine.answer.citations.source_builder import build_sources
from dlightrag.engine.answer.citations.sources import (
    SourceDownloadLinkBuilder,
    project_contexts_for_client,
    project_source_payloads,
)
from dlightrag.engine.answer.model_runtime import AnswerModelRuntime
from dlightrag.engine.answer.resources.images import prepare_query_images
from dlightrag.engine.rag.retrieval import RetrievalResult


class AnswerQueryImagePreparer:
    """Reuse Answer's VLM capability and image policy for raw retrieval."""

    def __init__(
        self,
        *,
        capabilities: AnswerCapabilityCoordinator,
        models: AnswerModelRuntime,
    ) -> None:
        self._capabilities = capabilities
        self._models = models

    async def __call__(self, images: Sequence[Mapping[str, Any]]) -> list[str]:
        if not images:
            return []
        await self._capabilities.refresh_vlm()
        return await prepare_query_images(
            query_images=[dict(image) for image in images],
            describer=self._models.query_image_describer(),
        )


def project_answer_retrieval(
    result: RetrievalResult,
    projection: RetrieveProjection,
) -> ProjectedRetrieval:
    """Reuse Answer's canonical source/context projection for retrieval readers."""
    sources = build_sources(
        result.contexts,
        image_url_prefix=projection.image_url_prefix,
    )
    source_payloads = project_source_payloads(
        sources,
        resolver=(SourceDownloadLinkBuilder() if projection.include_download_links else None),
        downloadable_workspaces=(
            set(projection.downloadable_workspaces)
            if projection.downloadable_workspaces is not None
            else None
        ),
        visual_workspaces=(
            set(projection.visual_workspaces) if projection.visual_workspaces is not None else None
        ),
    )
    contexts = project_contexts_for_client(
        result.contexts,
        image_url_prefix=projection.image_url_prefix,
        visual_workspaces=(
            set(projection.visual_workspaces) if projection.visual_workspaces is not None else None
        ),
    )
    return ProjectedRetrieval(
        contexts=contexts,
        sources=tuple(source.model_dump() for source in source_payloads),
    )


__all__ = [
    "AnswerQueryImagePreparer",
    "project_answer_retrieval",
]
