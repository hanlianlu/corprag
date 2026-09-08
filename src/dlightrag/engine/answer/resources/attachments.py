# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Caller-facing answer-attachment conveniences and their ResourceInput adapter.

``AnswerAttachment`` is a convenience for Python SDK callers. Its adapter opens
local paths, bounds inline bytes, and validates HTTP(S) links before producing the
transport-neutral :class:`ResourceInput`. A caller path is read immediately and
never travels into a ``ResourceInput`` (or the model): only the basename
survives as a display filename.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from dlightrag.engine.answer.client_contracts import AnswerAttachmentLink
from dlightrag.engine.answer.resources.models import ResourceInput

_DEFAULT_MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AnswerAttachment:
    """One SDK answer attachment: a local path, inline bytes, or an HTTP(S) link.

    Constructed only through :meth:`from_path`, :meth:`from_bytes`, or
    :meth:`from_url`; the adapter normalizes it into a :class:`ResourceInput`.
    """

    filename: str | None = None
    content: bytes | None = None
    url: str | None = None
    declared_mime: str | None = None
    _path: str | None = None

    @classmethod
    def from_path(cls, path: str | Path, *, declared_mime: str | None = None) -> AnswerAttachment:
        """Reference a local file; its bytes are read only by the adapter."""
        resolved = Path(path)
        return cls(filename=resolved.name, declared_mime=declared_mime, _path=str(resolved))

    @classmethod
    def from_bytes(
        cls,
        content: bytes,
        *,
        filename: str | None = None,
        declared_mime: str | None = None,
    ) -> AnswerAttachment:
        """Attach inline bytes already held by the caller."""
        return cls(filename=filename, content=bytes(content), declared_mime=declared_mime)

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        filename: str | None = None,
        declared_mime: str | None = None,
    ) -> AnswerAttachment:
        """Attach an inert HTTP(S) link, validated for scheme and credentials now."""
        AnswerAttachmentLink(url=url, filename=filename)
        return cls(filename=filename, url=url, declared_mime=declared_mime)


def _bounded_file_bytes(path: str, *, max_attachment_bytes: int) -> bytes:
    with open(path, "rb") as handle:
        data = handle.read(max_attachment_bytes + 1)
    if len(data) > max_attachment_bytes:
        raise ValueError("attachment exceeds the per-attachment byte limit")
    return data


def resource_inputs_from_attachments(
    attachments: Iterable[AnswerAttachment],
    *,
    max_attachment_bytes: int = _DEFAULT_MAX_ATTACHMENT_BYTES,
) -> list[ResourceInput]:
    """Normalize SDK attachments into run-scoped :class:`ResourceInput` objects."""
    resources: list[ResourceInput] = []
    for attachment in attachments:
        if attachment._path is not None:
            content = _bounded_file_bytes(
                attachment._path, max_attachment_bytes=max_attachment_bytes
            )
            resources.append(
                ResourceInput(
                    filename=attachment.filename,
                    content=content,
                    declared_mime=attachment.declared_mime,
                )
            )
        elif attachment.content is not None:
            if len(attachment.content) > max_attachment_bytes:
                raise ValueError("attachment exceeds the per-attachment byte limit")
            resources.append(
                ResourceInput(
                    filename=attachment.filename,
                    content=attachment.content,
                    declared_mime=attachment.declared_mime,
                )
            )
        elif attachment.url is not None:
            resources.append(
                ResourceInput(
                    filename=attachment.filename,
                    url=attachment.url,
                    declared_mime=attachment.declared_mime,
                )
            )
        else:  # pragma: no cover - factories always set one source
            raise ValueError("attachment requires a path, bytes, or url")
    return resources


__all__ = [
    "AnswerAttachment",
    "resource_inputs_from_attachments",
]
