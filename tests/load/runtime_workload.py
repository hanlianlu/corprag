# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Deterministic workload shape and metric helpers for Slice 6."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class QuerySubmission:
    index: int
    kind: Literal["retrieval", "answer"]
    mode: Literal["fast", "research"] | None
    workspace_count: int


type MutationAction = Literal["ingest", "replace", "delete", "retry", "reset"]


def mutation_action(index: int) -> MutationAction:
    """Return the exact repeating five-action Corpus Mutation mix."""
    actions: tuple[MutationAction, ...] = ("ingest", "replace", "delete", "retry", "reset")
    return actions[index % len(actions)]


def query_submission(index: int) -> QuerySubmission:
    """Return the exact repeating 10% Retrieval / 40% Fast / 50% Research mix."""
    bucket = index % 10
    retrieval_ordinal = index // 10
    if bucket == 0:
        workspace_counts = (1, 10, 50, 100)
        return QuerySubmission(
            index=index,
            kind="retrieval",
            mode=None,
            workspace_count=workspace_counts[retrieval_ordinal % len(workspace_counts)],
        )
    if bucket <= 4:
        return QuerySubmission(index=index, kind="answer", mode="fast", workspace_count=1)
    return QuerySubmission(index=index, kind="answer", mode="research", workspace_count=10)


def percentile(values: list[float], quantile: float) -> float:
    """Nearest-rank percentile, stable for a bounded captured latency sample."""
    if not values:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50_ms": round(percentile(values, 0.50) * 1000, 3),
        "p95_ms": round(percentile(values, 0.95) * 1000, 3),
        "p99_ms": round(percentile(values, 0.99) * 1000, 3),
        "max_ms": round(max(values, default=0.0) * 1000, 3),
    }
