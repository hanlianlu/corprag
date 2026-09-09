# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Narrow host interface for independent Profile Memory."""

__version__ = "2.0.7"

from dlightrag_memory.errors import MemoryUnavailableError, MemoryWriteRejectedError
from dlightrag_memory.memory import Memory, RecallResult
from dlightrag_memory.models import (
    MemoryKind,
    MemoryOperation,
    MemoryOperationAction,
    MemoryOperationOutcome,
    MemoryOperationReceipt,
    MemoryOriginKind,
    MemoryProvenance,
    MemoryRecord,
    MemoryStatus,
)
from dlightrag_memory.store import MemoryStore

__all__ = [
    "Memory",
    "MemoryKind",
    "MemoryOperation",
    "MemoryOperationAction",
    "MemoryOperationOutcome",
    "MemoryOperationReceipt",
    "MemoryOriginKind",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryStatus",
    "MemoryStore",
    "MemoryUnavailableError",
    "MemoryWriteRejectedError",
    "RecallResult",
    "__version__",
]
