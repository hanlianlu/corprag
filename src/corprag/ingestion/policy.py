# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Ingestion policy for filtering MinerU content_list before RAGAnything processing.

Filters out noise content (discarded blocks) from MinerU's parse output
before it reaches RAGAnything's insert_content_list().
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestionPolicyConfig(BaseModel):
    """Configuration for content filtering policy."""

    model_config = {"frozen": True}

    drop_types: frozenset[str] = Field(
        default_factory=lambda: frozenset(
            {
                "discarded",
                "header",
                "footer",
                "page_number",
                "page_footnote",
            }
        )
    )


class PolicyStats(BaseModel):
    """Statistics for policy application."""

    total: int = Field(ge=0)
    indexed: int = Field(ge=0)
    dropped_by_type: int = Field(ge=0)

    @property
    def drop_rate(self) -> float:
        return (self.dropped_by_type / self.total * 100) if self.total > 0 else 0.0


class PolicyResult(BaseModel):
    """Result of applying ingestion policy."""

    index_stream: list[dict[str, Any]]
    stats: PolicyStats


class IngestionPolicy:
    """Filter MinerU content_list to remove noise before RAGAnything processing.

    Usage:
        policy = IngestionPolicy()
        result = policy.apply(content_list)
        # result.index_stream → rag.insert_content_list()
    """

    def __init__(self, config: IngestionPolicyConfig | None = None) -> None:
        self.config = config or IngestionPolicyConfig()

    def apply(self, content_list: list[dict[str, Any]]) -> PolicyResult:
        """Filter content_list by dropping noise types."""
        total = len(content_list)
        dropped_by_type = 0
        index_stream: list[dict[str, Any]] = []

        for item in content_list:
            item_type = str(item.get("type", "text")).lower()

            if item_type in self.config.drop_types:
                dropped_by_type += 1
                continue

            index_stream.append(item)

        indexed = len(index_stream)
        stats = PolicyStats(
            total=total,
            indexed=indexed,
            dropped_by_type=dropped_by_type,
        )

        return PolicyResult(index_stream=index_stream, stats=stats)


__all__ = [
    "IngestionPolicy",
    "IngestionPolicyConfig",
    "PolicyResult",
    "PolicyStats",
]
