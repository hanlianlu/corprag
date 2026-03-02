# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for reranking results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RankedChunk(BaseModel):
    """Single reranked chunk result from LLM reranker."""

    index: int = Field(description="0-based chunk index from original list")
    relevance_score: float = Field(ge=0, le=1, description="Relevance score 0-1")

    model_config = {"extra": "forbid"}


class RerankResult(BaseModel):
    """LLM reranker output - list of chunks sorted by relevance."""

    ranked_chunks: list[RankedChunk] = Field(description="Chunks sorted by relevance descending")

    model_config = {"extra": "forbid"}


__all__ = ["RankedChunk", "RerankResult"]
