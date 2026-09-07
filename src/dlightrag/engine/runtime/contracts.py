# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Operation-neutral public contracts for durable runs."""

from typing import Literal, TypeAlias

RunStatus: TypeAlias = Literal[  # noqa: UP040 - preserve the inline OpenAPI enum
    "queued", "running", "succeeded", "failed", "cancelled"
]
RunKind: TypeAlias = Literal[  # noqa: UP040 - preserve runtime introspection
    "retrieval", "answer", "corpus_mutation"
]
RunLane: TypeAlias = Literal["query", "corpus_mutation"]  # noqa: UP040
# Phases are executor-owned durable labels. The runtime persists them without
# importing an operation-specific phase vocabulary.
RunPhase: TypeAlias = str  # noqa: UP040 - keep OpenAPI schema inline

__all__ = ["RunKind", "RunLane", "RunPhase", "RunStatus"]
