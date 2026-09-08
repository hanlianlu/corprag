# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Execution modes and the trusted adapter boundary.

The kernel defines the seam but provides only the explicitly trusted local
adapter. ``sandbox`` never falls back to host execution: deployments must
supply a sandbox adapter or startup fails.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

from dlightrag.engine.agent.environment.local import (
    CompletedProcess,
    DirectoryEntry,
    LocalExecutionEnvironment,
    ProcessOutputSink,
)

type ExecutionMode = Literal["disabled", "trust", "sandbox"]


class ExecutionEnvironment(Protocol):
    """The filesystem/process operations base tools may request."""

    @property
    def root(self) -> Path: ...

    @property
    def integrity_violations(self) -> tuple[str, ...]: ...

    @property
    def quota_violation(self) -> str | None: ...

    def refresh_integrity(self) -> tuple[str, ...]: ...

    def prepare_process_directories(self) -> tuple[Path, Path]: ...

    def resolve(self, relative: str) -> Path: ...

    def stat_kind(self, path: Path) -> str: ...

    def list_directory(self, path: Path) -> tuple[DirectoryEntry, ...]: ...

    def read_bytes(self, path: Path) -> bytes: ...

    def write_bytes(self, path: Path, data: bytes) -> None: ...

    async def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
        on_output: ProcessOutputSink | None = None,
    ) -> CompletedProcess: ...


class ExecutionEnvironmentAdapter(Protocol):
    """Trusted host seam that binds one already-admitted workspace."""

    def create(self, workspace: Path) -> ExecutionEnvironment: ...


class TrustExecutionAdapter:
    """Bind DlightRAG's rooted host environment. This is not a sandbox."""

    def create(self, workspace: Path) -> LocalExecutionEnvironment:
        return LocalExecutionEnvironment(workspace)


class SandboxUnavailableError(RuntimeError):
    """The operator selected sandbox mode without installing a backend."""


def resolve_execution_adapter(
    mode: ExecutionMode,
    *,
    trust: ExecutionEnvironmentAdapter | None = None,
    sandbox: ExecutionEnvironmentAdapter | None = None,
) -> ExecutionEnvironmentAdapter | None:
    """Resolve one mode without implicit downgrade or backend discovery."""
    if mode == "disabled":
        return None
    if mode == "trust":
        return trust or TrustExecutionAdapter()
    if sandbox is None:
        raise SandboxUnavailableError(
            "agent execution mode 'sandbox' requires a configured sandbox adapter"
        )
    return sandbox


__all__ = [
    "ExecutionEnvironment",
    "ExecutionEnvironmentAdapter",
    "ExecutionMode",
    "SandboxUnavailableError",
    "TrustExecutionAdapter",
    "resolve_execution_adapter",
]
