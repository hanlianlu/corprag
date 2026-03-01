# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for CorpragConfig."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from corprag.config import CorpragConfig, get_config, reset_config


class TestCorpragConfig:
    """Test CorpragConfig initialization and validation."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default values are sensible."""
        monkeypatch.setenv("CORPRAG_OPENAI_API_KEY", "test-key")
        config = CorpragConfig()  # type: ignore[call-arg]

        assert config.postgres_host == "localhost"
        assert config.postgres_port == 5432
        assert config.postgres_user == "rag"
        assert config.postgres_database == "corprag"
        assert config.postgres_workspace == "default"
        assert config.vector_storage == "PGVectorStorage"
        assert config.graph_storage == "PGGraphStorage"
        assert config.kv_storage == "PGKVStorage"
        assert config.doc_status_storage == "PGDocStatusStorage"
        assert config.pg_hnsw_m == 32
        assert config.pg_hnsw_ef == 300
        assert config.parser == "mineru"
        assert config.default_mode == "mix"

    def test_env_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test CORPRAG_ env prefix works."""
        monkeypatch.setenv("CORPRAG_OPENAI_API_KEY", "my-test-key")
        monkeypatch.setenv("CORPRAG_POSTGRES_HOST", "my-pg-host")
        monkeypatch.setenv("CORPRAG_EMBEDDING_DIM", "1536")

        config = CorpragConfig()  # type: ignore[call-arg]
        assert config.openai_api_key == "my-test-key"
        assert config.postgres_host == "my-pg-host"
        assert config.embedding_dim == 1536

    def test_computed_properties(self, test_config: CorpragConfig) -> None:
        """Test computed properties."""
        assert isinstance(test_config.working_dir_path, Path)
        assert test_config.sources_dir == test_config.working_dir_path / "sources"
        assert test_config.artifacts_dir == test_config.working_dir_path / "artifacts"

    def test_model_names_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test model name resolution for OpenAI provider."""
        monkeypatch.setenv("CORPRAG_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CORPRAG_OPENAI_CHAT_MODEL", "gpt-4o")
        monkeypatch.setenv("CORPRAG_OPENAI_VISION_MODEL", "gpt-4o-vision")

        config = CorpragConfig()  # type: ignore[call-arg]
        assert config.chat_model_name == "gpt-4o"
        assert config.vision_model_name == "gpt-4o-vision"
        assert config.unified_api_key == "test-key"

    def test_azure_provider_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Azure provider requires endpoint and key."""
        monkeypatch.setenv("CORPRAG_LLM_PROVIDER", "azure_openai")

        with pytest.raises(ValueError, match="azure_openai_endpoint"):
            CorpragConfig()  # type: ignore[call-arg]

    def test_pg_env_bridge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that model_post_init bridges POSTGRES_* env vars."""
        # Clear any existing POSTGRES_ vars
        for key in list(os.environ):
            if key.startswith("POSTGRES_"):
                monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("CORPRAG_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CORPRAG_POSTGRES_HOST", "custom-host")

        CorpragConfig()  # type: ignore[call-arg]

        assert os.environ.get("POSTGRES_HOST") == "custom-host"

    def test_singleton_get_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_config returns same instance."""
        monkeypatch.setenv("CORPRAG_OPENAI_API_KEY", "test-key")
        reset_config()

        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test reset_config creates new instance."""
        monkeypatch.setenv("CORPRAG_OPENAI_API_KEY", "test-key")
        reset_config()

        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2
