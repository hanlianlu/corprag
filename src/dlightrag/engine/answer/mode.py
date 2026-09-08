# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Public Answer Mode and the capability-derived Valid Mode Set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from dlightrag.engine.answer.errors import (
    UnsupportedAnswerModeError,
    UnsupportedResourceCapabilityError,
)

AnswerMode = Literal["auto", "fast", "research"]
ResolvedMode = Literal["fast", "research"]
ResourceRole = Literal["image", "document", "other"]

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
_IMAGE_PREFIX = "image/"


@dataclass(frozen=True, slots=True)
class ModeResource:
    """One prepared resource as seen by mode validation."""

    role: ResourceRole


@dataclass(frozen=True, slots=True)
class ModeCapability:
    """Pinned model/tool facts that decide which modes can run."""

    query_supports_images: bool
    inspect_available: bool = False
    web_search_available: bool = False


def canonical_answer_mode(mode: str | None) -> AnswerMode:
    """Omitted public mode is auto."""
    if mode is None or mode == "":
        return "auto"
    if mode == "auto" or mode == "fast" or mode == "research":
        return mode
    raise UnsupportedAnswerModeError(mode)


def resource_role(*, filename: str | None, mime_type: str | None) -> ResourceRole:
    """Classify one attachment or link for Fast representability."""
    mime = (mime_type or "").lower()
    name = urlsplit(filename or "").path.lower()
    if mime.startswith(_IMAGE_PREFIX) or name.endswith(_IMAGE_SUFFIXES):
        return "image"
    if filename or mime:
        return "document"
    return "other"


def valid_modes(
    *,
    resources: tuple[ModeResource, ...] = (),
    capability: ModeCapability,
) -> frozenset[ResolvedMode]:
    """Return the capability-derived subset of {fast, research}.

    Web Search being configured does not remove Fast. A filesystem is not
    required for Research.
    """
    allowed: set[ResolvedMode] = set()
    if _fast_can_represent(resources, capability):
        allowed.add("fast")
    if _research_can_represent(resources, capability):
        allowed.add("research")
    return frozenset(allowed)


def require_supported_mode(
    *,
    requested: str | None,
    valid: frozenset[ResolvedMode],
) -> AnswerMode:
    """Reject a request that cannot legally resolve, without creating a run."""
    requested_mode = canonical_answer_mode(requested)
    if not valid:
        raise UnsupportedResourceCapabilityError()
    if requested_mode in {"fast", "research"} and requested_mode not in valid:
        raise UnsupportedAnswerModeError(requested_mode)
    return requested_mode


def _fast_can_represent(resources: tuple[ModeResource, ...], capability: ModeCapability) -> bool:
    for resource in resources:
        if resource.role == "image":
            if not capability.query_supports_images:
                return False
            continue
        if resource.role == "document":
            return False
    return True


def _research_can_represent(
    resources: tuple[ModeResource, ...], capability: ModeCapability
) -> bool:
    for resource in resources:
        if resource.role == "image" and not (
            capability.query_supports_images or capability.inspect_available
        ):
            return False
    return True


__all__ = [
    "AnswerMode",
    "ModeCapability",
    "ModeResource",
    "ResolvedMode",
    "canonical_answer_mode",
    "require_supported_mode",
    "resource_role",
    "valid_modes",
]
