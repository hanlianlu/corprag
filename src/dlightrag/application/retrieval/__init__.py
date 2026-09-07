# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Retrieval use case, execution, and reader contracts."""

from dlightrag.engine.rag.retrieval import MetadataFilter, RetrievalOptions

from .execution import (
    PinnedRetrievalModel,
    RetrievalExecutor,
    RetrievalRunInput,
    canonical_retrieval_result,
    restore_retrieval_result,
)
from .service import (
    CorpusUnavailableError,
    ProjectedRetrieval,
    QueryImagePreparer,
    RetrievalInputError,
    RetrievalService,
    RetrievalSettings,
    RetrievalTimeoutError,
    RetrieveProjection,
    RetrieveRequest,
    RetrieveResponse,
    SchemaLookup,
    retrieval_response_payload,
)

__all__ = [
    "CorpusUnavailableError",
    "MetadataFilter",
    "PinnedRetrievalModel",
    "ProjectedRetrieval",
    "QueryImagePreparer",
    "RetrievalExecutor",
    "RetrievalInputError",
    "RetrievalOptions",
    "RetrievalRunInput",
    "RetrievalService",
    "RetrievalSettings",
    "RetrievalTimeoutError",
    "RetrieveProjection",
    "RetrieveRequest",
    "RetrieveResponse",
    "SchemaLookup",
    "canonical_retrieval_result",
    "restore_retrieval_result",
    "retrieval_response_payload",
]
