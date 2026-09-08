# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Errors owned by the Profile Memory application capability."""


class MemoryDisabledError(Exception):
    """The owner explicitly deactivated Profile Memory."""

    error_kind = "memory_disabled"
    public_message = "Profile Memory is not active for this owner."

    def __init__(self) -> None:
        super().__init__(self.public_message)


__all__ = ["MemoryDisabledError"]
