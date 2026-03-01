# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGService facade."""

from __future__ import annotations

import pytest

from corprag.config import CorpragConfig
from corprag.service import RAGService


class TestRAGServiceInit:
    """Test RAGService construction and configuration."""

    def test_init_stores_config(self, test_config: CorpragConfig) -> None:
        """Test that constructor stores config without initializing."""
        service = RAGService(config=test_config)

        assert service.config is test_config
        assert not service._initialized
        assert service.ingestion is None
        assert service.rag_text is None

    def test_ensure_initialized_raises(self, test_config: CorpragConfig) -> None:
        """Test that API methods raise if not initialized."""
        service = RAGService(config=test_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            service._ensure_initialized()

    def test_callbacks_stored(self, test_config: CorpragConfig) -> None:
        """Test that callbacks are stored."""

        async def cancel() -> bool:
            return False

        def transform(url: str) -> str:
            return f"signed:{url}"

        service = RAGService(
            config=test_config,
            cancel_checker=cancel,
            url_transformer=transform,
        )

        assert service._cancel_checker is cancel
        assert service._url_transformer is transform

    def test_vlm_disabled(self, test_config: CorpragConfig) -> None:
        """Test VLM can be disabled."""
        service = RAGService(config=test_config, enable_vlm=False)
        assert not service.enable_vlm
