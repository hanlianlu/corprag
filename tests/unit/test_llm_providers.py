# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for multi-provider LLM factory functions."""

from __future__ import annotations

import pytest

from corprag.config import CorpragConfig
from corprag.models.llm import _build_chat_model, _ensure_bytes


class TestEnsureBytes:
    """Test _ensure_bytes normalization."""

    def test_base64_string(self) -> None:
        import base64

        raw = b"test image data"
        b64 = base64.b64encode(raw).decode()
        assert _ensure_bytes(b64) == raw

    def test_data_uri(self) -> None:
        import base64

        raw = b"png data"
        b64 = base64.b64encode(raw).decode()
        uri = f"data:image/png;base64,{b64}"
        assert _ensure_bytes(uri) == raw

    def test_invalid_string(self) -> None:
        assert _ensure_bytes("not base64!!!") is None


class TestBuildChatModel:
    """Test _build_chat_model dispatching."""

    def _make_config(self, **overrides) -> CorpragConfig:
        """Create a config with minimal required fields.

        Always includes openai_api_key since many providers fall back to
        OpenAI for vision/embeddings.
        """
        defaults = {
            "openai_api_key": "test-key",
            "llm_provider": "openai",
        }
        defaults.update(overrides)
        return CorpragConfig(**defaults)  # type: ignore[call-arg]

    def test_openai_returns_chat_openai(self) -> None:
        config = self._make_config(llm_provider="openai")
        model = _build_chat_model(config, "gpt-4.1-mini")

        from langchain_openai import ChatOpenAI

        assert isinstance(model, ChatOpenAI)

    def test_qwen_returns_chat_openai_with_base_url(self) -> None:
        config = self._make_config(
            llm_provider="qwen",
            qwen_api_key="qwen-key",
        )
        model = _build_chat_model(config, "qwen3.5-plus")

        from langchain_openai import ChatOpenAI

        assert isinstance(model, ChatOpenAI)

    def test_minimax_returns_chat_openai_with_base_url(self) -> None:
        config = self._make_config(
            llm_provider="minimax",
            minimax_api_key="mm-key",
        )
        model = _build_chat_model(config, "MiniMax-M2.5")

        from langchain_openai import ChatOpenAI

        assert isinstance(model, ChatOpenAI)

    def test_anthropic_returns_chat_anthropic(self) -> None:
        config = self._make_config(
            llm_provider="anthropic",
            anthropic_api_key="ant-key",
        )

        try:
            model = _build_chat_model(config, "claude-sonnet-4-6")
            # If langchain-anthropic is installed, verify type
            from langchain_anthropic import ChatAnthropic

            assert isinstance(model, ChatAnthropic)
        except ImportError:
            pytest.skip("langchain-anthropic not installed")

    def test_google_returns_chat_google(self) -> None:
        config = self._make_config(
            llm_provider="google_gemini",
            google_gemini_api_key="google-key",
        )

        try:
            model = _build_chat_model(config, "gemini-2.5-flash")
            from langchain_google_genai import ChatGoogleGenerativeAI

            assert isinstance(model, ChatGoogleGenerativeAI)
        except ImportError:
            pytest.skip("langchain-google-genai not installed")

    def test_explicit_provider_overrides_config(self) -> None:
        """When provider is passed, it overrides config.llm_provider."""
        config = self._make_config(
            llm_provider="openai",
            qwen_api_key="qwen-key",
        )
        model = _build_chat_model(config, "qwen3.5-plus", provider="qwen")

        from langchain_openai import ChatOpenAI

        assert isinstance(model, ChatOpenAI)

    def test_unsupported_provider_raises(self) -> None:
        config = self._make_config()

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            _build_chat_model(config, "some-model", provider="unsupported")

    def test_temperature_override(self) -> None:
        config = self._make_config(llm_temperature=0.8)
        model = _build_chat_model(config, "gpt-4.1-mini", temperature=0.1)

        # Temperature should be the override, not config default
        assert model.temperature == 0.1

    def test_default_temperature_from_config(self) -> None:
        config = self._make_config(llm_temperature=0.7)
        model = _build_chat_model(config, "gpt-4.1-mini")

        assert model.temperature == 0.7


class TestGetVisionModelFunc:
    """Test vision model factory dispatching."""

    def test_no_vision_model_returns_none(self) -> None:
        from corprag.models.llm import get_vision_model_func

        config = CorpragConfig(  # type: ignore[call-arg]
            llm_provider="openai",
            openai_api_key="test-key",
            vision_model=None,
        )
        # vision_model_name will be None, so should return None
        func = get_vision_model_func(config)
        assert func is None


class TestGetEmbeddingFunc:
    """Test embedding factory dispatching."""

    def test_openai_embedding(self) -> None:
        from corprag.models.llm import get_embedding_func

        config = CorpragConfig(  # type: ignore[call-arg]
            llm_provider="openai",
            openai_api_key="test-key",
        )
        func = get_embedding_func(config)
        assert func is not None
        assert func.embedding_dim == 1024

    def test_explicit_embedding_provider(self) -> None:
        """Test explicit embedding_provider uses its own credentials."""
        from corprag.models.llm import get_embedding_func

        config = CorpragConfig(  # type: ignore[call-arg]
            llm_provider="anthropic",
            anthropic_api_key="ant-key",
            embedding_provider="openai",
            openai_api_key="oai-key",
        )
        func = get_embedding_func(config)
        assert func is not None
        assert func.embedding_dim == 1024


class TestConvertOpenaiToAnthropicMessages:
    """Test message format conversion."""

    def test_system_message_extracted(self) -> None:
        from corprag.models.llm import _convert_openai_to_anthropic_messages

        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result, system = _convert_openai_to_anthropic_messages(msgs)
        assert system == "You are helpful."
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_image_url_converted_to_base64_source(self) -> None:
        import base64

        from corprag.models.llm import _convert_openai_to_anthropic_messages

        b64 = base64.b64encode(b"fake png").decode()
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ]
        result, system = _convert_openai_to_anthropic_messages(msgs)
        assert system == ""
        assert len(result) == 1
        blocks = result[0]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image"
        assert blocks[1]["source"]["type"] == "base64"
        assert blocks[1]["source"]["media_type"] == "image/png"
