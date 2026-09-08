# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Pydantic schemas."""

from typing import Any, cast

from dlightrag.engine.answer.results import Reference

# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


class TestReference:
    def test_valid_reference(self) -> None:
        ref = Reference(id=cast(Any, 1), title="report.pdf")
        assert ref.id == "1"
        assert ref.title == "report.pdf"

    def test_json_roundtrip(self) -> None:
        ref = Reference(id=cast(Any, 2), title="spec.pdf")
        data = ref.model_dump()
        assert data == {"id": "2", "title": "spec.pdf"}
