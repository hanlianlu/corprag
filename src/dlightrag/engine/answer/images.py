# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer LLM image budgeting."""

import base64
import ipaddress
import logging
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from dlightrag.engine.ai.media import ImagePayloadBudget, image_url_block
from dlightrag.engine.ai.telemetry import safe_log_text

logger = logging.getLogger(__name__)

# Schemes that answer_image_budget allows as image sources.
# - data: → decoded, resized, re-encoded through the byte budget
# - https: → provider's server fetches; only blocked when the host is an
#   IP literal in a private / loopback / link-local range
# Everything else (http://, file://, ftp://, custom:) is rejected.
_ALLOWED_SCHEMES = frozenset({"data", "https"})


@dataclass(frozen=True, slots=True)
class AnswerImagePolicy:
    """Immutable Answer transport policy shared by image-bearing consumers."""

    max_images: int
    max_total_bytes: int
    max_bytes_per_image: int
    max_pixels: int
    max_px: int
    min_px: int
    quality: int
    min_quality: int

    def new_budget(self, *, max_px: int | None = None) -> AnswerImageBudget:
        """Create a fresh mutable budget for one run or provider call.

        ``max_px`` lowers the per-image edge for a specific call (a multi-page
        overview packs more pages by shrinking each one). The policy is frozen and
        every budget starts with zeroed counters.
        """
        edge = self.max_px if max_px is None else max(1, max_px)
        return AnswerImageBudget(
            max_images=self.max_images,
            max_total_bytes=self.max_total_bytes,
            max_bytes_per_image=self.max_bytes_per_image,
            max_pixels=self.max_pixels,
            max_px=edge,
            min_px=min(self.min_px, edge),
            quality=self.quality,
            min_quality=self.min_quality,
        )


def _is_unsafe_host(host: str | None) -> bool:
    """Return True if *host* is an IP literal in a dangerous range.

    Only called when ``urlparse`` succeeds — no DNS resolution.
    A trailing dot (DNS FQDN marker) is stripped so that ``127.0.0.1.``
    is treated the same as ``127.0.0.1``.
    """
    if not host:
        return False
    host = host.rstrip(".")

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        try:
            # inet_aton performs numeric-only legacy IPv4 parsing without DNS.
            # It canonicalizes one-to-four component decimal, octal, and hex
            # forms before ipaddress applies the actual address classification.
            addr = ipaddress.ip_address(socket.inet_aton(host))
        except OSError:
            return False  # not a numeric IP literal — safe (hostname)
    return (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_unspecified  # 0.0.0.0 / ::
    )


def _validate_image_url(text: str, *, label: str) -> str | None:
    """Return *text* if it is safe to pass to a model provider, else None.

    Whitelist approach: only ``data:`` and ``https:`` are permitted. For
    ``https:`` URLs whose host is an IP literal we additionally reject private,
    loopback, and link-local addresses (SSRF guard).
    """
    # Fast path — no parse needed for these
    if text.startswith("data:"):
        return text
    colon = text.find(":")
    if colon < 0:
        logger.warning(
            "Rejected image source without scheme · %s · %s",
            safe_log_text(label),
            safe_log_text(text),
        )
        return None
    scheme = text[:colon].lower().rstrip("/")
    if scheme not in _ALLOWED_SCHEMES:
        logger.warning(
            "Rejected image source with unsafe scheme '%s' · %s · %.80s",
            safe_log_text(scheme),
            safe_log_text(label),
            safe_log_text(text),
        )
        return None

    # https: — parse and validate host
    parsed = urlparse(text)
    if _is_unsafe_host(parsed.hostname):
        logger.warning(
            "Rejected image URL with unsafe host '%s' · %s",
            safe_log_text(parsed.hostname),
            safe_log_text(label),
        )
        return None
    return text


@dataclass
class AnswerImageBudget:
    """Bound image payloads sent to an answer model."""

    max_images: int
    max_total_bytes: int
    max_bytes_per_image: int
    max_pixels: int
    max_px: int
    min_px: int
    quality: int
    min_quality: int
    count: int = 0
    used_bytes: int = 0

    def add_base64(self, value: str, *, label: str) -> dict[str, Any] | None:
        """Add a raw base64/data URI image if it fits the remaining budget."""
        bounded = self._bound_base64(value, label=label)
        if bounded is None:
            return None
        uri, _ = bounded
        return {"type": "image_url", "image_url": {"url": uri}}

    def _bound_base64(self, value: str, *, label: str) -> tuple[str, int] | None:
        """Bound a raw base64/data URI image and record consumed bytes."""
        budget = ImagePayloadBudget(
            max_images=self.max_images,
            max_total_bytes=self.max_total_bytes,
            max_bytes_per_image=self.max_bytes_per_image,
            max_pixels=self.max_pixels,
            max_px=self.max_px,
            min_px=self.min_px,
            quality=self.quality,
            min_quality=self.min_quality,
            count=self.count,
            used_bytes=self.used_bytes,
        )
        bounded = budget.add_base64(value, label=label)
        if bounded is None:
            return None
        self.count = budget.count
        self.used_bytes = budget.used_bytes
        return bounded

    def reserve_prepared(self, data: bytes, *, label: str) -> bool:
        """Reserve an already-persisted derivative without changing its bytes.

        Recovery may run under a stricter policy than the execution that wrote
        the attachment. In that case the attachment is omitted rather than
        silently re-encoded behind its durable digest metadata.
        """
        candidate = AnswerImageBudget(
            max_images=self.max_images,
            max_total_bytes=self.max_total_bytes,
            max_bytes_per_image=self.max_bytes_per_image,
            max_pixels=self.max_pixels,
            max_px=self.max_px,
            min_px=self.min_px,
            quality=self.quality,
            min_quality=self.min_quality,
            count=self.count,
            used_bytes=self.used_bytes,
        )
        bounded = candidate._bound_base64(base64.b64encode(data).decode("ascii"), label=label)
        if bounded is None:
            return False
        uri, _size = bounded
        try:
            restored = base64.b64decode(uri.partition(",")[2], validate=True)
        except ValueError:
            return False
        if restored != data:
            return False
        self.count = candidate.count
        self.used_bytes = candidate.used_bytes
        return True

    def add_user_image(self, value: str | dict[str, Any], *, label: str) -> dict[str, Any] | None:
        """Add a user image after validating its source URL.

        ``data:`` URIs and bare base64 strings are decoded, resized, and
        re-encoded through the byte budget.  ``https:`` URLs are passed
        through (the provider's server fetches them) — private / loopback /
        link-local IP literals are rejected. All other URI schemes (including
        ``http:``) are rejected.
        """
        if self.count >= self.max_images:
            return None
        if isinstance(value, dict):
            return self._add_image_url_block(value, label=label)
        text = value.strip()

        # data: URIs → base64 budget pipeline (resize + compress)
        if text.startswith("data:"):
            return self.add_base64(text, label=label)
        # Only validate strings that look like URLs (have a scheme://).
        # Bare base64, raw bytes, or unrecognized formats fall through
        # to the byte-budget pipeline.
        if text.find(":") < 0 or "//" not in text:
            return self.add_base64(text, label=label)

        # Any scheme:// source goes through the single whitelist / SSRF gate.
        safe = _validate_image_url(text, label=label)
        if safe is None:
            return None
        self.count += 1
        return {"type": "image_url", "image_url": {"url": safe}}

    def _add_image_url_block(self, value: dict[str, Any], *, label: str) -> dict[str, Any] | None:
        """Add an OpenAI-style image block after validating the inner URL.

        ``image_url.url`` is validated through the same whitelist as
        ``add_user_image``.  ``data:`` URIs inside blocks go through
        the byte budget; non-``data:`` safe schemes pass through.
        """
        block = image_url_block(value)
        if block is None:
            return None
        image_url = block.get("image_url")
        if not isinstance(image_url, dict):
            return None
        url = image_url.get("url")
        if not isinstance(url, str) or not url.strip():
            return None
        text = url.strip()

        # data: inside a dict block → still goes through budget
        if text.startswith("data:"):
            bounded = self._bound_base64(text, label=label)
            if bounded is None:
                return None
            bounded_url, _ = bounded
            bounded_block = dict(block)
            bounded_image_url = dict(image_url)
            bounded_image_url["url"] = bounded_url
            bounded_block["image_url"] = bounded_image_url
            return bounded_block

        # All other schemes → validate through whitelist gate
        safe = _validate_image_url(text, label=label)
        if safe is None:
            return None
        self.count += 1
        return block


__all__ = ["AnswerImageBudget", "AnswerImagePolicy"]
