# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Explicit, dependency-neutral classification for retryable interruptions.

Only failures that cross a typed dependency boundary, or a small set of known
client-library transport/status surfaces, are retryable.  Configuration,
authentication, schema, input, context-capacity, and unknown failures remain
non-retryable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import httpx

type DependencyComponent = Literal["corpus_storage", "parser", "providers"]


class TransientDependencyError(RuntimeError):
    """A named external dependency is temporarily unavailable."""

    def __init__(self, component: DependencyComponent, message: str) -> None:
        self.component: DependencyComponent = component
        super().__init__(message)


class ProviderUnavailableError(TransientDependencyError):
    """A model provider request failed for an explicitly transient reason."""

    def __init__(self) -> None:
        super().__init__("providers", "Model provider is temporarily unavailable")


class ParserUnavailableError(TransientDependencyError):
    """The configured document parser is temporarily unavailable."""

    def __init__(self) -> None:
        super().__init__("parser", "Document parser is temporarily unavailable")


_AUTH_STATUS_CODES = frozenset({401, 403})
_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_PROVIDER_MODULE_PREFIXES = ("openai", "anthropic", "google.genai", "google.api_core")
_STORAGE_MODULE_PREFIXES = ("asyncpg", "pymilvus", "grpc")
_AUTH_NAME_MARKERS = ("authentication", "unauthorized", "permissiondenied", "forbidden")
_TRANSIENT_PROVIDER_CLASS_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "DeadlineExceeded",
        "InternalServerError",
        "RateLimitError",
        "ServerError",
        "ServiceUnavailable",
        "ServiceUnavailableError",
        "TooManyRequests",
    }
)
_TRANSIENT_STORAGE_TEXT = (
    "broken pipe",
    "closed channel",
    "connection refused",
    "connection reset",
    "deadline exceeded",
    "fail connecting to server",
    "failed to connect",
    "ping timeout",
    "server unavailable",
    "temporarily unavailable",
)
_NON_RETRYABLE_TEXT = (
    "authentication",
    "context length",
    "context window",
    "credential",
    "forbidden",
    "invalid api key",
    "invalid input",
    "password authentication failed",
    "permission denied",
    "prompt is too long",
    "schema",
    "too many tokens",
    "unauthorized",
    "unsupported",
)


def classify_transient_dependency(
    exc: BaseException,
    *,
    component_hint: DependencyComponent | None = None,
) -> DependencyComponent | None:
    """Return the interrupted component only for an explicit transient failure.

    The whole cause chain is considered because adapters add useful domain
    context while retaining the original client exception.  Non-retryable
    markers win across the chain so an authentication or deterministic request
    rejection can never become retryable merely because a wrapper is present.
    """

    chain = tuple(_exception_chain(exc))
    if any(_is_non_retryable(item) for item in chain):
        return None
    for item in chain:
        if isinstance(item, TransientDependencyError):
            return item.component
        if isinstance(item, TimeoutError | ConnectionError) and component_hint is not None:
            return component_hint
        if isinstance(item, httpx.TransportError):
            return component_hint or "providers"
        if isinstance(item, httpx.HTTPStatusError):
            if _status_code(item) in _RETRYABLE_STATUS_CODES:
                return component_hint or "providers"
            continue
        module = type(item).__module__
        name = type(item).__name__
        status = _status_code(item)
        if module.startswith(_PROVIDER_MODULE_PREFIXES):
            if status in _RETRYABLE_STATUS_CODES or name in _TRANSIENT_PROVIDER_CLASS_NAMES:
                return "providers"
        if module.startswith(_STORAGE_MODULE_PREFIXES):
            if status in _RETRYABLE_STATUS_CODES or any(
                marker in str(item).lower() for marker in _TRANSIENT_STORAGE_TEXT
            ):
                return "corpus_storage"
    return None


def next_dependency_retry(
    checkpoint: Mapping[str, Any] | None,
    component: DependencyComponent,
    *,
    base_seconds: int = 5,
    max_seconds: int = 60,
) -> tuple[dict[str, Any], int]:
    """Build one bounded, secret-free durable retry checkpoint and delay."""

    key = (
        "corpus_unavailable_attempt"
        if component == "corpus_storage"
        else f"{component}_unavailable_attempt"
    )
    previous: Any = checkpoint.get(key) if isinstance(checkpoint, Mapping) else None
    try:
        attempt = max(1, int(previous or 0) + 1)
    except TypeError, ValueError:
        attempt = 1
    # Bound the persisted counter as well as exponentiation.  Once the delay is
    # capped, a larger counter carries no scheduling information.
    capped_attempt = min(attempt, 32)
    delay = min(max_seconds, base_seconds * (2 ** (min(capped_attempt, 8) - 1)))
    return ({key: capped_attempt}, delay)


def dependency_component_from_checkpoint(
    checkpoint: Mapping[str, Any] | None,
) -> DependencyComponent | None:
    """Read only this module's closed component vocabulary from a checkpoint."""

    if not isinstance(checkpoint, Mapping):
        return None
    for component, key in (
        ("corpus_storage", "corpus_unavailable_attempt"),
        ("parser", "parser_unavailable_attempt"),
        ("providers", "providers_unavailable_attempt"),
    ):
        if key in checkpoint:
            return component  # type: ignore[return-value]
    return None


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_non_retryable(exc: BaseException) -> bool:
    status = _status_code(exc)
    if status in _AUTH_STATUS_CODES or (
        status is not None and 400 <= status < 500 and status not in {408, 425, 429}
    ):
        return True
    name = type(exc).__name__.lower()
    if any(marker in name for marker in _AUTH_NAME_MARKERS):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _NON_RETRYABLE_TEXT)


def _status_code(exc: BaseException) -> int | None:
    for value in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raw_value = getattr(value, "value", None)
        if isinstance(raw_value, int) and not isinstance(raw_value, bool):
            return raw_value
    return None


__all__ = [
    "DependencyComponent",
    "ParserUnavailableError",
    "ProviderUnavailableError",
    "TransientDependencyError",
    "classify_transient_dependency",
    "dependency_component_from_checkpoint",
    "next_dependency_retry",
]
