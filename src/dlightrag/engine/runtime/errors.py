# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage- and policy-neutral durable runtime failures.

Checkpoint-era errors, codecs, and constants are gone: Session settlements,
durable progress, and effect outcomes carry closed conflict values, and run
failures surface through :class:`RunExecutionError`.
"""


class RunSchemaError(RuntimeError):
    """The durable run schema is incompatible with this Runtime revision."""


class IncompatibleActiveRunError(RuntimeError):
    """An accepted active Run cannot execute under this Runtime revision."""


class RunCancelledError(RuntimeError):
    """The run this caller waited on was cancelled by its owner."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run {run_id} was cancelled")
        self.run_id = run_id


class RunFailedError(RuntimeError):
    """The run this caller waited on failed with one public error."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.error_kind = kind
        self.public_message = message


class RunExecutionError(RuntimeError):
    """An owning executor's already-classified public run failure."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.public_message = message


__all__ = [
    "IncompatibleActiveRunError",
    "RunCancelledError",
    "RunFailedError",
    "RunExecutionError",
    "RunSchemaError",
]
