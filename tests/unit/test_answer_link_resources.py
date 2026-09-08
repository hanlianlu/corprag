# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the client attachment-link descriptor adapter."""

from dlightrag.engine.answer.client_contracts import AnswerAttachmentLink
from dlightrag.engine.answer.resources.links import answer_link_resources


def test_answer_link_resources_maps_descriptors() -> None:
    links = [AnswerAttachmentLink(url="https://example.com/a.pdf", filename="a.pdf")]

    [resource] = answer_link_resources(links)

    assert resource.url == "https://example.com/a.pdf"
    assert resource.filename == "a.pdf"
    assert resource.content is None
    assert answer_link_resources(None) == []
    assert answer_link_resources([]) == []
