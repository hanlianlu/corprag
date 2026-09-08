# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Conflict-aware access scheduler for one host's concurrent tool batch."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class PathAccess:
    """A rooted path read, write, or recursive search."""

    path: str
    kind: Literal["read", "write", "readwrite", "search"]


@dataclass(frozen=True, slots=True)
class ExternalAccess:
    """A non-filesystem tool access that may run beside path reads."""


@dataclass(frozen=True, slots=True)
class WorkspaceAccess:
    """An operation, such as Bash, that may touch any workspace path.

    Workspace-wide access conflicts with path reads/writes/searches but not
    independent external retrieval. This is concurrency scheduling, not an
    authorization or shell-policy decision.
    """


type ToolAccess = PathAccess | ExternalAccess | WorkspaceAccess


class AccessScheduler:
    """Grant non-overlapping accesses without a waiter-order guarantee."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._changed = asyncio.Condition(self._lock)
        self._workspace = False
        self._writers: set[str] = set()
        self._readers: dict[str, int] = {}
        self._searches = 0

    @asynccontextmanager
    async def hold(self, access: ToolAccess) -> AsyncIterator[None]:
        """Wait until ``access`` is compatible, then occupy it until exit."""
        async with self._changed:
            while self._conflicts(access):
                await self._changed.wait()
            self._acquire(access)
        try:
            yield
        finally:
            async with self._changed:
                self._release(access)
                self._changed.notify_all()

    def _conflicts(self, access: ToolAccess) -> bool:
        if isinstance(access, WorkspaceAccess):
            return (
                self._workspace or bool(self._writers) or bool(self._readers) or self._searches > 0
            )
        if isinstance(access, ExternalAccess):
            return False
        if self._workspace:
            return True
        if access.kind in {"write", "readwrite"}:
            return (
                access.path in self._writers
                or self._readers.get(access.path, 0) > 0
                or self._searches > 0
            )
        if access.kind == "search":
            return bool(self._writers)
        return access.path in self._writers

    def _acquire(self, access: ToolAccess) -> None:
        if isinstance(access, WorkspaceAccess):
            self._workspace = True
            return
        if isinstance(access, ExternalAccess):
            return
        if access.kind in {"write", "readwrite"}:
            self._writers.add(access.path)
            return
        if access.kind == "search":
            self._searches += 1
            return
        self._readers[access.path] = self._readers.get(access.path, 0) + 1

    def _release(self, access: ToolAccess) -> None:
        if isinstance(access, WorkspaceAccess):
            self._workspace = False
            return
        if isinstance(access, ExternalAccess):
            return
        if access.kind in {"write", "readwrite"}:
            self._writers.discard(access.path)
            return
        if access.kind == "search":
            self._searches = max(0, self._searches - 1)
            return
        count = self._readers.get(access.path, 0) - 1
        if count <= 0:
            self._readers.pop(access.path, None)
        else:
            self._readers[access.path] = count


__all__ = [
    "AccessScheduler",
    "WorkspaceAccess",
    "ExternalAccess",
    "PathAccess",
    "ToolAccess",
]
