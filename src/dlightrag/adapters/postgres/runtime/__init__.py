# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL adapters for the durable RunRuntime.

Exports are resolved lazily so importing an Answer-owned repository can load
private runtime helpers without eagerly importing the Run store that composes
that repository.
"""

from importlib import import_module
from typing import Any

PGRunBlobStore: Any
PGRunStore: Any

_EXPORTS = {
    "PGRunBlobStore": (".run_blob_store", "PGRunBlobStore"),
    "PGRunStore": (".run_store", "PGRunStore"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = ["PGRunBlobStore", "PGRunStore"]
