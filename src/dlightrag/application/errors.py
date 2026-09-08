# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Errors owned by the Application lifecycle."""

import math

from dlightrag.engine.dependencies import TransientDependencyError
from dlightrag.engine.runtime.errors import RunSchemaError


class ApplicationClosedError(RuntimeError):
    """Raised when a closed Application is asked for one of its services."""

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or "Application is shutting down"
        super().__init__(self.detail)


class CorpusUnavailableError(TransientDependencyError):
    """An Application use case cannot currently reach corpus state."""

    def __init__(self, detail: str | None = None) -> None:
        super().__init__("corpus_storage", detail or "Corpus storage is temporarily unavailable")


class StorageSchemaError(RuntimeError):
    """Durable storage schema is incompatible with this revision."""


class WorkspaceWriteFencedError(RuntimeError):
    """A workspace write was refused while its promotion fence is active.

    Retryable: transports surface HTTP 409 with a ``Retry-After`` header.
    """

    def __init__(self, *, workspace: str, retry_after_seconds: float) -> None:
        self.workspace = workspace
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Workspace '{workspace}' is being promoted to dedicated storage; "
            f"retry after {int(math.ceil(retry_after_seconds))} seconds"
        )


__all__ = [
    "ApplicationClosedError",
    "CorpusUnavailableError",
    "RunSchemaError",
    "StorageSchemaError",
    "WorkspaceWriteFencedError",
]
