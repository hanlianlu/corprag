# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for multi-provider LLM factory functions."""

from __future__ import annotations

from functools import partial

import pytest

from dlightrag.config import DlightragConfig
from dlightrag.models.llm import _ensure_bytes, get_ingestion_llm_model_func, get_llm_model_func


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


class TestGetLlmModelFunc:
    """Test get_llm_model_func returns correct partial for each provider."""

    def _make_config(self, **overrides) -> DlightragConfig:
        defaults = {
            "openai_api_key": "test-key",
            "llm_provider": "openai",
        }
        defaults.update(overrides)
        return DlightragConfig(**defaults)  # type: ignore[call-arg]

    def test_openai_returns_partial(self) -> None:
        config = self._make_config(llm_provider="openai", chat_model="gpt-4.1-mini")
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.openai"
        assert func.keywords["model"] == "gpt-4.1-mini"

    def test_qwen_returns_openai_partial(self) -> None:
        config = self._make_config(llm_provider="qwen", qwen_api_key="qwen-key")
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.openai"

    def test_minimax_returns_openai_partial(self) -> None:
        config = self._make_config(llm_provider="minimax", minimax_api_key="mm-key")
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.openai"

    def test_ollama_returns_ollama_partial(self) -> None:
        config = self._make_config(llm_provider="ollama")
        try:
            func = get_llm_model_func(config)
        except ModuleNotFoundError:
            pytest.skip("ollama package not installed")
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.ollama"
        # Host should have /v1 stripped
        assert func.keywords.get("host", "").endswith("/v1") is False

    def test_openrouter_returns_openai_partial(self) -> None:
        config = self._make_config(llm_provider="openrouter", openrouter_api_key="sk-or-key")
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.openai"

    def test_xinference_returns_openai_partial(self) -> None:
        config = self._make_config(llm_provider="xinference")
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.openai"

    def test_anthropic_returns_anthropic_partial(self) -> None:
        config = self._make_config(llm_provider="anthropic", anthropic_api_key="ant-key")
        try:
            func = get_llm_model_func(config)
        except ModuleNotFoundError:
            pytest.skip("anthropic/voyageai package not installed")
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.anthropic"

    def test_google_gemini_returns_gemini_partial(self) -> None:
        config = self._make_config(llm_provider="google_gemini", google_gemini_api_key="google-key")
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.gemini"

    def test_unsupported_provider_raises(self) -> None:
        config = self._make_config()
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm_model_func(config, provider="unsupported")

    def test_ingestion_uses_ingestion_model(self) -> None:
        config = self._make_config(ingestion_model="gpt-4.1-nano")
        func = get_ingestion_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.keywords["model"] == "gpt-4.1-nano"

    def test_model_name_override(self) -> None:
        config = self._make_config()
        func = get_llm_model_func(config, model_name="gpt-4.1")
        assert func.keywords["model"] == "gpt-4.1"


class TestGetVisionModelFunc:
    """Test vision model factory dispatching."""

    def test_no_vision_model_returns_none(self) -> None:
        from dlightrag.models.llm import get_vision_model_func

        config = DlightragConfig(  # type: ignore[call-arg]
            llm_provider="openai",
            openai_api_key="test-key",
            vision_model=None,
        )
        # vision_model_name will be None, so should return None
        func = get_vision_model_func(config)
        assert func is None


class TestGetEmbeddingFunc:
    """Test embedding factory dispatching."""

    def _make_config(self, **overrides) -> DlightragConfig:
        defaults = {
            "openai_api_key": "test-key",
            "llm_provider": "openai",
        }
        defaults.update(overrides)
        return DlightragConfig(**defaults)  # type: ignore[call-arg]

    def test_openai_embedding(self) -> None:
        from dlightrag.models.llm import get_embedding_func

        config = self._make_config()
        func = get_embedding_func(config)
        assert func is not None
        assert func.embedding_dim == 1024
        assert isinstance(func.func, partial)

    def test_explicit_embedding_provider(self) -> None:
        from dlightrag.models.llm import get_embedding_func

        config = self._make_config(
            llm_provider="anthropic",
            anthropic_api_key="ant-key",
            embedding_provider="openai",
            openai_api_key="oai-key",
        )
        func = get_embedding_func(config)
        assert func is not None
        assert func.embedding_dim == 1024
        assert isinstance(func.func, partial)


class TestJsonKwargsForProvider:
    """Test _json_kwargs_for_provider returns correct JSON mode params."""

    def test_openai(self) -> None:
        from dlightrag.models.llm import _json_kwargs_for_provider

        assert _json_kwargs_for_provider("openai") == {"response_format": {"type": "json_object"}}

    def test_ollama(self) -> None:
        from dlightrag.models.llm import _json_kwargs_for_provider

        assert _json_kwargs_for_provider("ollama") == {"format": "json"}

    def test_google_gemini(self) -> None:
        from dlightrag.models.llm import _json_kwargs_for_provider

        result = _json_kwargs_for_provider("google_gemini")
        assert result == {"generation_config": {"response_mime_type": "application/json"}}

    def test_anthropic(self) -> None:
        from dlightrag.models.llm import _json_kwargs_for_provider

        assert _json_kwargs_for_provider("anthropic") == {}

    def test_openai_compatible_providers(self) -> None:
        from dlightrag.models.llm import _json_kwargs_for_provider

        for provider in ("azure_openai", "qwen", "minimax", "openrouter", "xinference"):
            result = _json_kwargs_for_provider(provider)
            assert result == {"response_format": {"type": "json_object"}}, f"Failed for {provider}"


class TestExtractJson:
    """Test _extract_json handles various LLM response formats."""

    def test_raw_json(self) -> None:
        from dlightrag.models.llm import _extract_json

        assert _extract_json('{"key": "value"}') == '{"key": "value"}'

    def test_markdown_fenced_json(self) -> None:
        from dlightrag.models.llm import _extract_json

        text = 'Here is the result:\n```json\n{"key": "value"}\n```'
        assert _extract_json(text) == '{"key": "value"}'

    def test_text_with_embedded_json(self) -> None:
        from dlightrag.models.llm import _extract_json

        text = 'Some preamble text {"key": "value"}'
        assert _extract_json(text) == '{"key": "value"}'

    def test_json_with_trailing_text(self) -> None:
        from dlightrag.models.llm import _extract_json

        text = '{"key": "value"} Here is my reasoning...'
        assert _extract_json(text) == '{"key": "value"}'


class TestConvertOpenaiToAnthropicMessages:
    """Test message format conversion."""

    def test_system_message_extracted(self) -> None:
        from dlightrag.models.llm import _convert_openai_to_anthropic_messages

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

        from dlightrag.models.llm import _convert_openai_to_anthropic_messages

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
