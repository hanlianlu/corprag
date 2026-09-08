# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared answer error taxonomy (design §14.2).

One error-kind vocabulary and classifier consumed by every surface (Web SSE,
REST, MCP) so callers can branch on a stable, machine-readable ``error_kind``
instead of parsing free-form messages. Caller input rejections derive from
``AnswerInputError``; server-side failures such as
``InvalidToolConfigurationError`` stand outside it so no surface reports them as
validation.
"""

from __future__ import annotations

CURRENT_IMAGES_UNSUPPORTED = "CURRENT_IMAGES_UNSUPPORTED"
CURRENT_IMAGE_LIMIT_EXCEEDED = "CURRENT_IMAGE_LIMIT_EXCEEDED"
CURRENT_DOCUMENT_PARSE_FAILED = "CURRENT_DOCUMENT_PARSE_FAILED"
ANSWER_IMAGE_CAPABILITY_UNKNOWN = "ANSWER_IMAGE_CAPABILITY_UNKNOWN"
ANSWER_INPUT_OVERFLOW = "ANSWER_INPUT_OVERFLOW"
ANSWER_STREAM_FAILED = "ANSWER_STREAM_FAILED"
INVALID_TOOL_CONFIGURATION = "invalid_tool_configuration"
MODEL_CAPABILITY_UNAVAILABLE = "MODEL_CAPABILITY_UNAVAILABLE"
ANSWER_RESOURCE_INVALID = "ANSWER_RESOURCE_INVALID"
UNSUPPORTED_ANSWER_MODE = "unsupported_answer_mode"
UNSUPPORTED_RESOURCE_CAPABILITY = "unsupported_resource_capability"
ROUTING_FAILED = "routing_failed"

_IMAGES_NOT_SUPPORTED_MARKER = "[IMAGES_NOT_SUPPORTED_BY_MODEL]"


class AnswerInputError(ValueError):
    """Answer input rejected with a client-safe message and stable kind.

    Subclasses must construct ``public_message`` only from sanitized content.
    """

    def __init__(self, public_message: str, *, error_kind: str) -> None:
        super().__init__(public_message)
        self.public_message = public_message
        self.error_kind = error_kind


class AnswerImageError(AnswerInputError):
    """Answer-image request rejected at the capability/transport boundary.

    Carries a stable ``error_kind`` from the answer-image taxonomy so every
    surface surfaces the same machine-readable classification.
    """

    def __init__(self, message: str, *, error_kind: str) -> None:
        super().__init__(public_message=message, error_kind=error_kind)


class CurrentImagePayloadError(AnswerImageError):
    """Explicit user images cannot fit the configured answer transport."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_kind=CURRENT_IMAGE_LIMIT_EXCEEDED)


class CurrentDocumentParseError(AnswerInputError):
    """A current attachment document could not be parsed safely."""

    def __init__(self, safe_filename: str) -> None:
        super().__init__(
            public_message=(
                f"Could not read {safe_filename}. Check that the document is valid and "
                "the document parser is available."
            ),
            error_kind=CURRENT_DOCUMENT_PARSE_FAILED,
        )


class AnswerInputOverflowError(AnswerInputError):
    """Answer inputs exceed the packable context capacity."""

    def __init__(self, public_message: str) -> None:
        super().__init__(public_message, error_kind=ANSWER_INPUT_OVERFLOW)


class AnswerModelCapabilityError(AnswerInputError):
    """The request requires a capability absent from its resolved model profile."""

    def __init__(self) -> None:
        super().__init__(
            "The configured query model cannot use the tools required for this answer request.",
            error_kind=MODEL_CAPABILITY_UNAVAILABLE,
        )


class UnsupportedAnswerModeError(AnswerInputError):
    """Requested Answer Mode is outside the Valid Mode Set, or auto has none."""

    def __init__(self, requested: str) -> None:
        super().__init__(
            f"Answer mode {requested!r} is not valid for this request.",
            error_kind=UNSUPPORTED_ANSWER_MODE,
        )


class UnsupportedResourceCapabilityError(AnswerInputError):
    """No mode remains because a prepared resource has no registered branch."""

    def __init__(self) -> None:
        super().__init__(
            "This request needs a resource capability that no answer mode can provide.",
            error_kind=UNSUPPORTED_RESOURCE_CAPABILITY,
        )


class AnswerResourceAdmissionError(AnswerInputError):
    """A caller resource violates the safe Answer admission contract."""

    def __init__(self) -> None:
        super().__init__(
            "An answer attachment or link could not be admitted safely.",
            error_kind=ANSWER_RESOURCE_INVALID,
        )


class InvalidToolConfigurationError(RuntimeError):
    """A run composed two peer tools that share one model-visible name.

    Tool names are server-defined, so a collision is a server composition
    failure, not caller input: it never becomes a validation rejection. The
    exception string names the colliding tools for operators, while
    ``public_message`` stays generic for every client surface.
    """

    def __init__(self, duplicate_names: tuple[str, ...]) -> None:
        super().__init__(f"Duplicate answer tool names: {', '.join(duplicate_names)}")
        self.error_kind = INVALID_TOOL_CONFIGURATION
        self.public_message = "Answer tooling is misconfigured."


def classify_answer_error(exc: BaseException) -> str:
    """Map an answer-stream failure to a stable answer error kind."""
    if isinstance(exc, AnswerInputError | InvalidToolConfigurationError):
        return exc.error_kind
    if _IMAGES_NOT_SUPPORTED_MARKER in str(exc):
        return CURRENT_IMAGES_UNSUPPORTED
    return ANSWER_STREAM_FAILED


__all__ = [
    "ANSWER_IMAGE_CAPABILITY_UNKNOWN",
    "ANSWER_INPUT_OVERFLOW",
    "ANSWER_STREAM_FAILED",
    "ANSWER_RESOURCE_INVALID",
    "CURRENT_DOCUMENT_PARSE_FAILED",
    "CURRENT_IMAGES_UNSUPPORTED",
    "CURRENT_IMAGE_LIMIT_EXCEEDED",
    "INVALID_TOOL_CONFIGURATION",
    "MODEL_CAPABILITY_UNAVAILABLE",
    "UNSUPPORTED_ANSWER_MODE",
    "UNSUPPORTED_RESOURCE_CAPABILITY",
    "ROUTING_FAILED",
    "AnswerInputError",
    "AnswerImageError",
    "AnswerInputOverflowError",
    "AnswerModelCapabilityError",
    "AnswerResourceAdmissionError",
    "CurrentDocumentParseError",
    "CurrentImagePayloadError",
    "InvalidToolConfigurationError",
    "UnsupportedAnswerModeError",
    "UnsupportedResourceCapabilityError",
    "classify_answer_error",
]
