# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Profile Memory product gate."""

from .errors import MemoryDisabledError
from .memory_list import (
    MEMORY_LIST_PAGE_DEFAULT_LIMIT,
    MEMORY_LIST_PAGE_MAX_LIMIT,
    MemoryListCursor,
    MemoryListCursorCodec,
    MemoryListCursorError,
    MemoryListPage,
    MemoryListPageRequest,
)
from .service import (
    InMemoryMemorySettingsStore,
    MemoryCapability,
    MemoryService,
    MemorySettings,
    MemorySettingsStore,
    NoopMemorySettingsStore,
)

__all__ = [
    "InMemoryMemorySettingsStore",
    "MEMORY_LIST_PAGE_DEFAULT_LIMIT",
    "MEMORY_LIST_PAGE_MAX_LIMIT",
    "MemoryCapability",
    "MemoryDisabledError",
    "MemoryListCursor",
    "MemoryListCursorCodec",
    "MemoryListCursorError",
    "MemoryListPage",
    "MemoryListPageRequest",
    "MemoryService",
    "MemorySettings",
    "MemorySettingsStore",
    "NoopMemorySettingsStore",
]
