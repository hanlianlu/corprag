# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""End-to-end integration test: ingest -> query.

Requires:
- PostgreSQL with pgvector + AGE
- Valid LLM API key
- LibreOffice (for Excel conversion tests)

Skipped automatically if dependencies are not available.
"""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


class TestIngestQueryE2E:
    """End-to-end ingest and query flow."""

    async def test_placeholder(self) -> None:
        """Placeholder for E2E tests.

        Full E2E tests require:
        1. Running PostgreSQL instance
        2. Valid LLM API credentials
        3. Test document fixtures

        These tests should be run in CI with proper fixtures.
        """
        pytest.skip("E2E tests require full infrastructure - run in CI")
