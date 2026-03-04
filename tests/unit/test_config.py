# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for DlightragConfig."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dlightrag.config import DlightragConfig, get_config, reset_config


class TestDlightragConfig:
    """Test DlightragConfig initialization and validation."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default values are sensible."""
        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "test-key")
        config = DlightragConfig()  # type: ignore[call-arg]

        assert config.postgres_host == "localhost"
        assert config.postgres_port == 5432
        assert config.postgres_user == "dlightrag"
        assert config.postgres_database == "dlightrag"
        assert config.workspace == "default"
        assert config.vector_storage == "PGVectorStorage"
        assert config.graph_storage == "PGGraphStorage"
        assert config.kv_storage == "PGKVStorage"
        assert config.doc_status_storage == "PGDocStatusStorage"
        assert config.pg_hnsw_m == 32
        assert config.pg_hnsw_ef_construction == 300
        assert config.pg_hnsw_ef_search == 300
        assert config.parser == "mineru"
        assert config.default_mode == "mix"

    def test_env_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test DLIGHTRAG_ env prefix works."""
        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "my-test-key")
        monkeypatch.setenv("DLIGHTRAG_POSTGRES_HOST", "my-pg-host")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING_DIM", "1536")

        config = DlightragConfig()  # type: ignore[call-arg]
        assert config.openai_api_key == "my-test-key"
        assert config.postgres_host == "my-pg-host"
        assert config.embedding_dim == 1536

    def test_computed_properties(self, test_config: DlightragConfig) -> None:
        """Test computed properties."""
        assert isinstance(test_config.working_dir_path, Path)
        assert test_config.sources_dir == test_config.working_dir_path / "sources"
        assert test_config.artifacts_dir == test_config.working_dir_path / "artifacts"

    def test_model_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test unified model name resolution."""
        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("DLIGHTRAG_CHAT_MODEL", "gpt-4o")
        monkeypatch.setenv("DLIGHTRAG_VISION_MODEL", "gpt-4o-vision")

        config = DlightragConfig()  # type: ignore[call-arg]
        assert config.chat_model_name == "gpt-4o"
        assert config.vision_model_name == "gpt-4o-vision"
        assert config._get_provider_api_key("openai") == "test-key"

    def test_azure_provider_requires_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Azure provider requires API key."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "azure_openai")
        monkeypatch.setenv("DLIGHTRAG_AZURE_OPENAI_API_KEY", "")

        with pytest.raises(ValueError, match="azure_openai_api_key"):
            DlightragConfig()  # type: ignore[call-arg]

    def test_azure_provider_requires_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Azure provider requires endpoint."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "azure_openai")
        monkeypatch.setenv("DLIGHTRAG_AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.setenv("DLIGHTRAG_AZURE_OPENAI_BASE_URL", "")

        with pytest.raises(ValueError, match="azure_openai_base_url"):
            DlightragConfig()  # type: ignore[call-arg]

    def test_anthropic_provider_requires_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Anthropic provider requires API key."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "anthropic")

        with pytest.raises(ValueError, match="anthropic_api_key"):
            DlightragConfig()  # type: ignore[call-arg]

    def test_google_provider_requires_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Google Gemini provider requires API key."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "google_gemini")

        with pytest.raises(ValueError, match="google_gemini_api_key"):
            DlightragConfig()  # type: ignore[call-arg]

    def test_qwen_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Qwen provider config."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "qwen")
        monkeypatch.setenv("DLIGHTRAG_QWEN_API_KEY", "qwen-key")

        config = DlightragConfig()  # type: ignore[call-arg]
        assert config._get_provider_api_key("qwen") == "qwen-key"
        assert "dashscope" in config._get_url("qwen_base_url")

    def test_minimax_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test MiniMax provider config."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "minimax")
        monkeypatch.setenv("DLIGHTRAG_MINIMAX_API_KEY", "mm-key")

        config = DlightragConfig()  # type: ignore[call-arg]
        assert config._get_provider_api_key("minimax") == "mm-key"

    def test_ollama_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Ollama provider config (no API key needed)."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "ollama")

        config = DlightragConfig()  # type: ignore[call-arg]
        assert config._get_provider_api_key("ollama") == "ollama"
        assert "11434" in config._get_url("ollama_base_url")

    def test_xinference_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Xinference provider config (no API key needed)."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "xinference")

        config = DlightragConfig()  # type: ignore[call-arg]
        assert config._get_provider_api_key("xinference") == "xinference"
        assert "9997" in config._get_url("xinference_base_url")

    def test_openrouter_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test OpenRouter provider config."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("DLIGHTRAG_OPENROUTER_API_KEY", "sk-or-key")

        config = DlightragConfig()  # type: ignore[call-arg]
        assert config._get_provider_api_key("openrouter") == "sk-or-key"
        assert "openrouter" in config._get_url("openrouter_base_url")

    def test_openrouter_requires_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test OpenRouter provider requires API key."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "openrouter")

        with pytest.raises(ValueError, match="openrouter_api_key"):
            DlightragConfig()  # type: ignore[call-arg]

    def test_embedding_provider_defaults_to_llm_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test embedding provider defaults to llm_provider (no magic fallback)."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("DLIGHTRAG_ANTHROPIC_API_KEY", "ant-key")

        config = DlightragConfig()  # type: ignore[call-arg]
        assert config.effective_embedding_provider == "anthropic"

    def test_explicit_embedding_provider_requires_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that explicit embedding_provider validates its own API key."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "openai")
        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "oai-key")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING_PROVIDER", "google_gemini")

        with pytest.raises(ValueError, match="google_gemini_api_key"):
            DlightragConfig()  # type: ignore[call-arg]

    def test_explicit_vision_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test explicit vision provider override."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("DLIGHTRAG_ANTHROPIC_API_KEY", "ant-key")
        monkeypatch.setenv("DLIGHTRAG_VISION_PROVIDER", "qwen")
        monkeypatch.setenv("DLIGHTRAG_QWEN_API_KEY", "qwen-key")

        config = DlightragConfig()  # type: ignore[call-arg]
        assert config.effective_vision_provider == "qwen"

    def test_explicit_embedding_provider_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test explicit embedding provider override."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("DLIGHTRAG_ANTHROPIC_API_KEY", "ant-key")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "oai-key")

        config = DlightragConfig()  # type: ignore[call-arg]
        assert config.effective_embedding_provider == "openai"

    def test_rerank_llm_provider_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test rerank LLM provider defaults to llm_provider."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "openai")
        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "test-key")

        config = DlightragConfig()  # type: ignore[call-arg]
        assert config.effective_rerank_llm_provider == "openai"

    def test_rerank_llm_provider_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test explicit rerank LLM provider override."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("DLIGHTRAG_ANTHROPIC_API_KEY", "ant-key")
        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "oai-key")
        monkeypatch.setenv("DLIGHTRAG_RERANK_LLM_PROVIDER", "openai")

        config = DlightragConfig()  # type: ignore[call-arg]
        assert config.effective_rerank_llm_provider == "openai"

    def test_rerank_llm_provider_requires_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that rerank_llm_provider validates API key."""
        monkeypatch.setenv("DLIGHTRAG_LLM_PROVIDER", "openai")
        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "oai-key")
        monkeypatch.setenv("DLIGHTRAG_RERANK_BACKEND", "llm")
        monkeypatch.setenv("DLIGHTRAG_RERANK_LLM_PROVIDER", "anthropic")

        with pytest.raises(ValueError, match="anthropic_api_key"):
            DlightragConfig()  # type: ignore[call-arg]

    def test_pg_env_bridge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that model_post_init bridges POSTGRES_* env vars."""
        # Clear any existing POSTGRES_ vars
        for key in list(os.environ):
            if key.startswith("POSTGRES_"):
                monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("DLIGHTRAG_POSTGRES_HOST", "custom-host")

        DlightragConfig()  # type: ignore[call-arg]

        assert os.environ.get("POSTGRES_HOST") == "custom-host"

    def test_workspace_env_bridge_pg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test workspace bridges to POSTGRES_WORKSPACE when PG backends are used."""
        for key in list(os.environ):
            if key.startswith("POSTGRES_"):
                monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("DLIGHTRAG_WORKSPACE", "my-project")

        DlightragConfig()  # type: ignore[call-arg]

        assert os.environ.get("POSTGRES_WORKSPACE") == "my-project"

    def test_workspace_env_bridge_neo4j(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test workspace bridges to NEO4J_WORKSPACE when Neo4j is used."""
        monkeypatch.delenv("NEO4J_WORKSPACE", raising=False)
        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("DLIGHTRAG_WORKSPACE", "ws-neo")
        monkeypatch.setenv("DLIGHTRAG_GRAPH_STORAGE", "Neo4JStorage")

        DlightragConfig()  # type: ignore[call-arg]

        assert os.environ.get("NEO4J_WORKSPACE") == "ws-neo"

    def test_workspace_env_bridge_no_pg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test POSTGRES_WORKSPACE is NOT set when no PG backends are used."""
        for key in list(os.environ):
            if key.startswith("POSTGRES_"):
                monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("DLIGHTRAG_WORKSPACE", "local-ws")
        monkeypatch.setenv("DLIGHTRAG_KV_STORAGE", "JsonKVStorage")
        monkeypatch.setenv("DLIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage")
        monkeypatch.setenv("DLIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
        monkeypatch.setenv("DLIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")

        DlightragConfig()  # type: ignore[call-arg]

        assert "POSTGRES_WORKSPACE" not in os.environ

    def test_singleton_get_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_config returns same instance."""
        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "test-key")
        reset_config()

        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test reset_config creates new instance."""
        monkeypatch.setenv("DLIGHTRAG_OPENAI_API_KEY", "test-key")
        reset_config()

        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2
