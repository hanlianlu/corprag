# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Adapters from client attachment-link descriptors to request-local resources."""

from collections.abc import Sequence

from dlightrag.engine.answer.client_contracts import AnswerAttachmentLink
from dlightrag.engine.answer.resources.models import ResourceInput


def answer_link_resources(
    attachments: Sequence[AnswerAttachmentLink] | None,
) -> list[ResourceInput]:
    """Map transport link descriptors to inert HTTPS :class:`ResourceInput` objects."""
    if not attachments:
        return []
    return [ResourceInput(filename=link.filename, url=link.url) for link in attachments]


__all__ = [
    "answer_link_resources",
]
