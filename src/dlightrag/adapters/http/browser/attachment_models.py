# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web current-turn attachment admission (one ordered images+documents collection)."""

import hashlib
import mimetypes
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from dlightrag.application.corpus_admin import safe_upload_basename
from dlightrag.engine.ai.media import MODEL_IMAGE_MAX_PIXELS, verify_web_image_bytes

# One ordered attachment collection per message. Images and documents mix; the
# Answer preparation extracts verified images into current-image blocks and registers the
# rest as request-local resources. The admitted count is owned at runtime by
# ``config.answer.generation.max_attachments`` and threaded in by callers.
SUPPORTED_DOCUMENT_EXTENSIONS = frozenset(
    {
        "pdf",
        "docx",
        "pptx",
        "xlsx",
        "md",
        "textpack",
        "txt",
        "csv",
        "json",
        "html",
        "htm",
        "xml",
        "yaml",
        "yml",
        "rtf",
        "odt",
        "epub",
        "tex",
        "log",
        "py",
        "js",
        "ts",
        "css",
        "scss",
        "sql",
        "sh",
        "conf",
        "ini",
        "properties",
    }
)

AttachmentKind = Literal["image", "document", "unsupported"]


def _suffix(filename: str) -> str:
    safe = safe_upload_basename(filename)
    dot = safe.rfind(".")
    return safe[dot:].lower() if dot >= 0 else ""


def classify_web_attachment(filename: str, mime_type: str | None) -> AttachmentKind:
    mime = (mime_type or mimetypes.guess_type(filename)[0] or "").lower()
    if mime.startswith("image/"):
        return "image"
    extension = _suffix(filename).lstrip(".")
    return "document" if extension in SUPPORTED_DOCUMENT_EXTENSIONS else "unsupported"


@dataclass(frozen=True, slots=True)
class ValidatedWebAttachment:
    """One admitted current-turn Web attachment (image or document)."""

    attachment_id: str
    ordinal: int
    filename: str
    mime_type: str
    suffix: str
    attachment_bytes: bytes
    content_sha256: str
    kind: Literal["image", "document"]

    @property
    def byte_size(self) -> int:
        return len(self.attachment_bytes)


def validate_web_attachments(
    items: Sequence[tuple[str, str | None, bytes]],
    *,
    max_attachments: int,
    max_attachment_bytes: int,
    max_total_attachment_bytes: int,
    image_max_pixels: int = MODEL_IMAGE_MAX_PIXELS,
) -> tuple[ValidatedWebAttachment, ...]:
    """Admit one ordered mixed collection of current-turn image/document uploads."""
    if len(items) > max_attachments:
        raise ValueError(f"Web answer accepts at most {max_attachments} attachments per message")
    total_bytes = sum(len(payload) for _, _, payload in items)
    if total_bytes > max_total_attachment_bytes:
        raise ValueError(f"total attachment bytes exceed {max_total_attachment_bytes}")
    validated: list[ValidatedWebAttachment] = []
    for ordinal, (filename, mime_type, payload) in enumerate(items, start=1):
        safe_name = safe_upload_basename(filename)
        suffix = _suffix(safe_name)
        if not payload:
            raise ValueError(f"Attachment is empty: {safe_name}")
        kind = classify_web_attachment(safe_name, mime_type)
        if kind == "image":
            if len(payload) > max_attachment_bytes:
                raise ValueError(f"image {safe_name} exceeds the {max_attachment_bytes}-byte limit")
            try:
                detected_mime = verify_web_image_bytes(payload, max_pixels=image_max_pixels)
            except ValueError as exc:
                raise ValueError(f"image {safe_name} {exc}") from exc
        elif kind == "document":
            if len(payload) > max_attachment_bytes:
                raise ValueError(f"document {safe_name} exceeds the size limit")
            detected_mime = (
                mime_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
            )
        else:
            raise ValueError(f"Unsupported attachment: {safe_name}")
        validated.append(
            ValidatedWebAttachment(
                attachment_id=str(uuid4()),
                ordinal=ordinal,
                filename=safe_name,
                mime_type=detected_mime,
                suffix=suffix,
                attachment_bytes=payload,
                content_sha256=hashlib.sha256(payload).hexdigest(),
                kind="image" if kind == "image" else "document",
            )
        )
    return tuple(validated)


__all__ = [
    "SUPPORTED_DOCUMENT_EXTENSIONS",
    "ValidatedWebAttachment",
    "classify_web_attachment",
    "validate_web_attachments",
]
