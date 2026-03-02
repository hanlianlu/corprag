# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for IngestionPolicy."""

from __future__ import annotations

from corprag.ingestion.policy import IngestionPolicy, IngestionPolicyConfig


class TestIngestionPolicy:
    """Test content filtering policy."""

    def test_filters_discarded_blocks(self) -> None:
        """Test that discarded blocks are filtered out."""
        content_list = [
            {"type": "text", "text": "Important content"},
            {"type": "discarded", "text": "Noise"},
            {"type": "text", "text": "More content"},
        ]

        policy = IngestionPolicy()
        result = policy.apply(content_list)

        assert len(result.index_stream) == 2
        assert result.stats.total == 3
        assert result.stats.indexed == 2
        assert result.stats.dropped_by_type == 1

    def test_filters_headers_footers(self) -> None:
        """Test filtering of headers, footers, page numbers."""
        content_list = [
            {"type": "text", "text": "Body text"},
            {"type": "header", "text": "Page Header"},
            {"type": "footer", "text": "Page Footer"},
            {"type": "page_number", "text": "42"},
            {"type": "page_footnote", "text": "Footnote text"},
        ]

        policy = IngestionPolicy()
        result = policy.apply(content_list)

        assert len(result.index_stream) == 1
        assert result.index_stream[0]["text"] == "Body text"
        assert result.stats.dropped_by_type == 4

    def test_empty_content_list(self) -> None:
        """Test empty input."""
        policy = IngestionPolicy()
        result = policy.apply([])

        assert len(result.index_stream) == 0
        assert result.stats.total == 0
        assert result.stats.drop_rate == 0.0

    def test_custom_drop_types(self) -> None:
        """Test custom drop types configuration."""
        config = IngestionPolicyConfig(drop_types=frozenset({"custom_noise"}))
        policy = IngestionPolicy(config)

        content_list = [
            {"type": "text", "text": "Keep this"},
            {"type": "custom_noise", "text": "Drop this"},
            {"type": "discarded", "text": "Keep this too (custom config)"},
        ]

        result = policy.apply(content_list)
        assert len(result.index_stream) == 2
        assert result.stats.dropped_by_type == 1

    def test_case_insensitive_type(self) -> None:
        """Test that type matching is case-insensitive."""
        content_list = [
            {"type": "DISCARDED", "text": "Should be dropped"},
            {"type": "Discarded", "text": "Also dropped"},
            {"type": "text", "text": "Kept"},
        ]

        policy = IngestionPolicy()
        result = policy.apply(content_list)
        assert len(result.index_stream) == 1

    def test_drop_rate_calculation(self) -> None:
        """Test drop rate is calculated correctly."""
        content_list = [
            {"type": "text", "text": "Keep"},
            {"type": "discarded", "text": "Drop"},
            {"type": "text", "text": "Keep"},
            {"type": "discarded", "text": "Drop"},
        ]

        policy = IngestionPolicy()
        result = policy.apply(content_list)
        assert result.stats.drop_rate == 50.0
