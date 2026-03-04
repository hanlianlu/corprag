# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Pydantic reranking schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlightrag.models.schemas import RankedChunk, RerankResult

# ---------------------------------------------------------------------------
# RankedChunk
# ---------------------------------------------------------------------------


class TestRankedChunk:
    def test_valid(self) -> None:
        chunk = RankedChunk(index=0, relevance_score=0.5)
        assert chunk.index == 0
        assert chunk.relevance_score == 0.5

    def test_boundary_scores(self) -> None:
        assert RankedChunk(index=0, relevance_score=0).relevance_score == 0
        assert RankedChunk(index=0, relevance_score=1).relevance_score == 1

    def test_score_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankedChunk(index=0, relevance_score=-0.1)

    def test_score_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankedChunk(index=0, relevance_score=1.5)

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            RankedChunk(index=0, relevance_score=0.5, foo="bar")


# ---------------------------------------------------------------------------
# RerankResult
# ---------------------------------------------------------------------------


class TestRerankResult:
    def test_valid_with_chunks(self) -> None:
        result = RerankResult(
            ranked_chunks=[
                RankedChunk(index=1, relevance_score=0.9),
                RankedChunk(index=0, relevance_score=0.3),
            ]
        )
        assert len(result.ranked_chunks) == 2

    def test_empty_chunks_allowed(self) -> None:
        result = RerankResult(ranked_chunks=[])
        assert result.ranked_chunks == []
