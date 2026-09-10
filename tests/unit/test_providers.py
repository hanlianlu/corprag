# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for provider ABC, registry, and concrete implementations."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from dlightrag.engine.ai.messages import ToolDefinition
from dlightrag.engine.ai.providers import get_provider
from dlightrag.engine.ai.providers.base import CompletionOutput, CompletionProvider
from dlightrag.engine.ai.providers.openai_compatible import (
    OpenAICompatibleProvider,
    _openai_tool_messages,
)


def _openai_error_response(status_code: int) -> httpx2.Response:
    """Build the HTTP response type used by the OpenAI SDK transport."""
    return httpx2.Response(
        status_code,
        request=httpx2.Request("POST", "https://t/v1"),
    )


class TestCompletionProviderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            cast(Any, CompletionProvider)(api_key="k", timeout=10.0, max_retries=1)


class TestProviderRegistry:
    @pytest.mark.parametrize("provider_name", ["openai", "anthropic", "gemini"])
    def test_get_provider_returns_completion_provider(self, provider_name: str):
        p = get_provider(provider_name, api_key="test-key")
        assert isinstance(p, CompletionProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown_provider")

    def test_error_message_lists_available(self):
        with pytest.raises(ValueError, match="openai"):
            get_provider("bad")


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_complete_extracts_system_message(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="reply")]
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            result = await p.complete(
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hi"},
                ],
                "claude-sonnet-4-20250514",
            )
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["system"] == "You are helpful."
            assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert result == "reply"

    @pytest.mark.asyncio
    async def test_complete_preserves_token_limit_stop_reason(self):
        p = get_provider("anthropic", api_key="test-key")
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="partial")],
            stop_reason="max_tokens",
            usage=None,
        )
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as sdk:
            sdk.return_value.messages.create = AsyncMock(return_value=response)
            cast(Any, p)._client = None
            result = await p.complete(
                [{"role": "user", "content": "hi"}],
                "claude-sonnet-4-20250514",
            )

        assert result == "partial"
        assert result.stop_reason == "length"

    @pytest.mark.asyncio
    async def test_complete_defaults_max_tokens(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="ok")]
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete([{"role": "user", "content": "hi"}], "claude-sonnet-4-20250514")
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["max_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_complete_tool_turn_converts_tools_history_and_response(self):
        p = get_provider("anthropic", api_key="test-key")
        tool = ToolDefinition(
            name="search_web",
            description="Search the open web.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )
        response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="thinking",
                    thinking="Need another source.",
                    signature="anthropic-signature",
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="call-2",
                    name="search_web",
                    input={"query": "inflation"},
                ),
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=8, output_tokens=3),
        )
        messages = [
            {
                "role": "assistant",
                "content": "",
                "provider_state": {
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "Previous thought.",
                            "signature": "previous-signature",
                        }
                    ]
                },
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"prices"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search_web",
                "content": "price evidence",
                "is_error": False,
            },
        ]

        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as sdk:
            create = AsyncMock(return_value=response)
            sdk.return_value.messages.create = create
            cast(Any, p)._client = None
            turn = await p.complete_tool_turn(
                messages,
                "claude-sonnet-4-20250514",
                tools=[tool],
                tool_choice="required",
            )

        await_args = create.await_args
        assert await_args is not None
        request = await_args.kwargs
        assert request["tools"] == [
            {
                "name": "search_web",
                "description": "Search the open web.",
                "input_schema": tool.parameters,
            }
        ]
        assert request["tool_choice"] == {"type": "any"}
        assert request["messages"] == [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Previous thought.",
                        "signature": "previous-signature",
                    },
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "search_web",
                        "input": {"query": "prices"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": "price evidence",
                        "is_error": False,
                    }
                ],
            },
        ]
        assert turn.stop_reason == "tool_use"
        assert turn.reasoning == "Need another source."
        assert turn.provider_state == {
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Need another source.",
                    "signature": "anthropic-signature",
                }
            ]
        }
        assert turn.tool_calls[0].arguments == {"query": "inflation"}
        assert turn.usage_details == {"input_tokens": 8, "output_tokens": 3}

    @pytest.mark.asyncio
    async def test_complete_tool_turn_streaming_preserves_text_thinking_and_tool_calls(self):
        p = get_provider("anthropic", api_key="test-key")

        async def fake_stream():
            yield SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=5)),
            )
            yield SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="thinking", thinking="", signature=""),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="thinking_delta", thinking="Think."),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="signature_delta", signature="signed"),
            )
            yield SimpleNamespace(
                type="content_block_start",
                index=1,
                content_block=SimpleNamespace(type="text", text="Draft "),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(type="text_delta", text="answer"),
            )
            yield SimpleNamespace(
                type="content_block_start",
                index=2,
                content_block=SimpleNamespace(
                    type="tool_use", id="call-1", name="search_web", input={}
                ),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                index=2,
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"query":'),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                index=2,
                delta=SimpleNamespace(type="input_json_delta", partial_json='"inflation"}'),
            )
            yield SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
                usage=SimpleNamespace(output_tokens=7),
            )

        emitted: list[str] = []

        async def emit_text(text: str) -> None:
            emitted.append(text)

        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as sdk:
            sdk.return_value.messages.create = AsyncMock(return_value=fake_stream())
            cast(Any, p)._client = None
            turn = await p.complete_tool_turn_streaming(
                [{"role": "user", "content": "latest inflation"}],
                "claude-sonnet-4-20250514",
                tools=[],
                emit_text=emit_text,
            )

        assert emitted == ["Draft ", "answer"]
        assert turn.text == "Draft answer"
        assert turn.reasoning == "Think."
        assert turn.tool_calls[0].arguments == {"query": "inflation"}
        assert turn.provider_state == {
            "thinking_blocks": [{"type": "thinking", "thinking": "Think.", "signature": "signed"}]
        }
        assert turn.usage_details == {"input_tokens": 5, "output_tokens": 7}

    @pytest.mark.asyncio
    async def test_complete_routes_thinking_to_top_level(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="thought")]
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete(
                [{"role": "user", "content": "hi"}],
                "claude-sonnet-4-20250514",
                model_kwargs={"thinking": {"type": "enabled", "budget_tokens": 1024}},
            )
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024}
            assert "extra_body" not in call_kwargs

    @pytest.mark.asyncio
    async def test_json_object_response_format_is_rejected(self):
        p = get_provider("anthropic", api_key="test-key")
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            cast(Any, p)._client = None
            with pytest.raises(ValueError, match="json_schema"):
                await p.complete(
                    [{"role": "user", "content": "hi"}],
                    "claude-sonnet-4-20250514",
                    response_format={"type": "json_object"},
                )
            MockSDK.return_value.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_json_schema_response_format_uses_output_config(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text='{"answer": "ok"}')]
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete(
                [{"role": "user", "content": "hi"}],
                "claude-sonnet-4-20250514",
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "demo_plan", "schema": schema, "strict": True},
                },
            )
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["output_config"] == {
                "format": {"type": "json_schema", "schema": schema}
            }
            assert "system" not in call_kwargs

    @pytest.mark.asyncio
    async def test_json_schema_and_reasoning_effort_share_output_config(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text='{"answer": "ok"}')]
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete(
                [{"role": "user", "content": "hi"}],
                "claude-opus-5",
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "demo_plan", "schema": schema, "strict": True},
                },
                model_kwargs={
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                },
            )
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["thinking"] == {"type": "adaptive"}
            assert call_kwargs["output_config"] == {
                "format": {"type": "json_schema", "schema": schema},
                "effort": "high",
            }

    @pytest.mark.asyncio
    async def test_complete_converts_https_image_url(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="ok")]
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/chart.png"},
                            },
                            {"type": "text", "text": "describe"},
                        ],
                    }
                ],
                "claude-sonnet-4-20250514",
            )
            call_kwargs = MockSDK.return_value.messages.create.call_args[1]
            assert call_kwargs["messages"][0]["content"][0] == {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/chart.png"},
            }

    @pytest.mark.asyncio
    async def test_complete_handles_thinking_blocks_and_usage(self):
        p = get_provider("anthropic", api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="thinking", thinking="let me think"),
            MagicMock(type="text", text="answer"),
        ]
        mock_response.usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=3,
            cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=7),
        )
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            result = await p.complete(
                [{"role": "user", "content": "hi"}],
                "claude-sonnet-4-20250514",
                model_kwargs={"thinking": {"type": "enabled", "budget_tokens": 1024}},
            )
        assert result == "answer"
        assert cast(Any, p).last_reasoning == "let me think"
        assert result.usage_details == {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation.ephemeral_5m_input_tokens": 7,
        }

    @pytest.mark.asyncio
    async def test_stream_merges_message_start_and_delta_usage(self):
        p = get_provider("anthropic", api_key="test-key")
        holder: dict[str, Any] = {}

        async def fake_stream():
            yield SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=0)),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="hi"),
            )
            yield SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=6))

        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as MockSDK:
            MockSDK.return_value.messages.create = AsyncMock(return_value=fake_stream())
            cast(Any, p)._client = None
            tokens = [
                t
                async for t in cast(Any, p).stream(
                    [{"role": "user", "content": "hi"}],
                    "claude-sonnet-4-20250514",
                    usage_holder=holder,
                )
            ]

        assert tokens == ["hi"]
        assert holder == {"usage_details": {"input_tokens": 10, "output_tokens": 6}}

    async def test_stream_tool_text_replays_native_tool_history(self):

        p = get_provider("anthropic", api_key="test-key")

        async def fake_stream():
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="final "),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="answer"),
            )

        messages = [
            {
                "role": "assistant",
                "content": "",
                "provider_state": {
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "thought",
                            "signature": "signature",
                        }
                    ]
                },
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search",
                "content": "evidence",
                "is_error": False,
            },
        ]
        with patch("dlightrag.engine.ai.providers.anthropic_native.AsyncAnthropic") as sdk:
            create = AsyncMock(return_value=fake_stream())
            sdk.return_value.messages.create = create
            cast(Any, p)._client = None
            tokens = [
                token
                async for token in p.stream_tool_text(
                    messages,
                    "claude-sonnet-4-20250514",
                )
            ]

        assert tokens == ["final ", "answer"]
        await_args = create.await_args
        assert await_args is not None
        assert await_args.kwargs["messages"][0]["content"][0]["signature"] == "signature"
        assert await_args.kwargs["messages"][1]["content"][0]["type"] == "tool_result"


class TestOpenAICompatibleProvider:
    async def test_complete_returns_content(self):
        p = get_provider("openai", api_key="test-key")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await p.complete([{"role": "user", "content": "hi"}], "gpt-5.4-mini")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_complete_preserves_token_limit_stop_reason(self):
        p = get_provider("openai", api_key="test-key")
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="partial", model_extra=None),
                    finish_reason="length",
                )
            ],
            usage=None,
        )
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=response)
            result = await p.complete([{"role": "user", "content": "hi"}], "gpt-5.4-mini")

        assert result == "partial"
        assert result.stop_reason == "length"

    @pytest.mark.asyncio
    async def test_complete_tool_turn_sends_tools_and_normalizes_calls(self):
        p = get_provider("openai", api_key="test-key")
        function = SimpleNamespace(name="search_web", arguments='{"query":"inflation"}')
        tool_call = SimpleNamespace(id="call-1", type="function", function=function)
        message = SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
            model_extra={
                "reasoning_content": "Need current evidence.",
                "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
            },
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
        )
        tool = ToolDefinition(
            name="search_web",
            description="Search the open web.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        )

        with patch.object(p, "_get_client") as mock_client:
            create = AsyncMock(return_value=response)
            mock_client.return_value.chat.completions.create = create
            turn = await p.complete_tool_turn(
                [{"role": "user", "content": "latest inflation"}],
                "mimo-v2.5",
                tools=[tool],
                tool_choice="required",
            )

        await_args = create.await_args
        assert await_args is not None
        request = await_args.kwargs
        assert request["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the open web.",
                    "parameters": tool.parameters,
                },
            }
        ]
        assert request["tool_choice"] == "required"
        assert turn.stop_reason == "tool_use"
        assert turn.text == ""
        assert turn.reasoning == "Need current evidence."
        assert turn.provider_state == {
            "reasoning_content": "Need current evidence.",
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
        }
        assert turn.tool_calls[0].id == "call-1"
        assert turn.tool_calls[0].name == "search_web"
        assert turn.tool_calls[0].arguments == {"query": "inflation"}
        assert turn.tool_calls[0].argument_error is None
        assert turn.usage_details == {"prompt_tokens": 4, "completion_tokens": 2}

        replay = {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
            "provider_state": turn.provider_state,
        }
        with patch.object(p, "_get_client") as replay_client:
            replay_client.return_value.chat.completions.create = AsyncMock(
                return_value=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="done", tool_calls=None, model_extra=None
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )
            )
            await p.complete_tool_turn([replay], "mimo-v2.5", tools=[])
        replay_args = replay_client.return_value.chat.completions.create.await_args
        assert replay_args is not None
        replay_message = replay_args.kwargs["messages"][0]
        assert "provider_state" not in replay_message
        assert replay_message["reasoning_content"] == "Need current evidence."
        assert replay_message["reasoning_details"] == [
            {"type": "reasoning.encrypted", "data": "opaque"}
        ]

    @pytest.mark.asyncio
    async def test_complete_tool_turn_preserves_bad_arguments_for_the_loop_to_reject(self):
        p = get_provider("openai", api_key="test-key")
        function = SimpleNamespace(name="search_web", arguments='{"query":')
        message = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(id="call-1", type="function", function=function)],
            model_extra=None,
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=None,
        )

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=response)
            turn = await p.complete_tool_turn(
                [{"role": "user", "content": "q"}],
                "mimo-v2.5",
                tools=[],
            )

        assert turn.tool_calls[0].arguments == {}
        assert turn.tool_calls[0].argument_error is not None

    @pytest.mark.asyncio
    async def test_complete_tool_turn_maps_plain_text_stop(self):
        p = get_provider("openai", api_key="test-key")
        message = SimpleNamespace(content="final answer", tool_calls=None, model_extra=None)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
        )

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=response)
            turn = await p.complete_tool_turn(
                [{"role": "user", "content": "q"}],
                "mimo-v2.5",
                tools=[],
            )

        assert turn.stop_reason == "stop"
        assert turn.text == "final answer"
        assert turn.tool_calls == ()

    @pytest.mark.asyncio
    async def test_complete_tool_turn_streaming_accumulates_fragmented_calls(self):
        p = get_provider("openai", api_key="test-key")

        async def fake_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="Draft ",
                            model_extra={"reasoning_content": "Think."},
                            tool_calls=None,
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            model_extra=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call-1",
                                    function=SimpleNamespace(
                                        name="search_web", arguments='{"query":'
                                    ),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            model_extra=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    function=SimpleNamespace(name=None, arguments='"inflation"}'),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[], usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3)
            )

        emitted: list[str] = []

        async def emit_text(text: str) -> None:
            emitted.append(text)

        with patch.object(p, "_open_stream", AsyncMock(return_value=fake_stream())):
            turn = await p.complete_tool_turn_streaming(
                [{"role": "user", "content": "latest inflation"}],
                "mimo-v2.5",
                tools=[],
                emit_text=emit_text,
            )

        assert emitted == ["Draft "]
        assert turn.text == "Draft "
        assert turn.reasoning == "Think."
        assert turn.stop_reason == "tool_use"
        assert turn.tool_calls[0].id == "call-1"
        assert turn.tool_calls[0].arguments == {"query": "inflation"}
        assert turn.usage_details == {"prompt_tokens": 4, "completion_tokens": 3}

    @pytest.mark.asyncio
    async def test_stream_captures_usage_and_cost_from_final_chunk(self):
        p = get_provider("openai", api_key="test-key")
        holder: dict[str, Any] = {}

        async def _fake_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="he", model_extra=None))],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="llo", model_extra=None))],
                usage=None,
            )
            # Final usage-only chunk (empty choices), as sent with include_usage.
            yield SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=5, completion_tokens=2, total_tokens=7, cost=0.0012
                ),
            )

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=_fake_stream()
            )
            stream = cast(Any, p).stream(
                [{"role": "user", "content": "hi"}], "gpt", usage_holder=holder
            )
            chunks = [c async for c in stream]

        assert chunks == ["he", "llo"]
        assert holder == {
            "usage_details": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            "cost_details": {"total": 0.0012},
        }

    @pytest.mark.asyncio
    async def test_stream_falls_back_when_stream_options_unsupported(self):
        from openai import BadRequestError

        p = get_provider("openai", api_key="test-key")
        holder: dict[str, Any] = {}
        calls: list[dict[str, Any]] = []

        async def _fake_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hi", model_extra=None))],
                usage=None,
            )

        async def _create(**kwargs: Any):
            calls.append(kwargs)
            if "stream_options" in kwargs:
                raise BadRequestError(
                    "stream_options unsupported",
                    response=_openai_error_response(400),
                    body=None,
                )
            return _fake_stream()

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = _create
            stream = cast(Any, p).stream(
                [{"role": "user", "content": "hi"}], "gpt", usage_holder=holder
            )
            chunks = [c async for c in stream]

        assert chunks == ["hi"]
        assert len(calls) == 2
        assert "stream_options" in calls[0]
        assert "stream_options" not in calls[1]
        assert holder == {}  # the fallback stream carries no usage

    @pytest.mark.parametrize(
        ("message", "body"),
        (
            ("invalid parameter: stream_options", None),
            ("stream_options is not permitted", None),
            (
                "Request validation failed",
                {
                    "detail": [
                        {
                            "loc": ["body", "stream_options"],
                            "msg": "extra inputs are not permitted",
                        }
                    ]
                },
            ),
        ),
    )
    @pytest.mark.asyncio
    async def test_stream_falls_back_for_explicit_stream_options_rejections(
        self,
        message: str,
        body: dict[str, Any] | None,
    ):
        from openai import BadRequestError

        p = get_provider("openai", api_key="test-key")
        calls: list[dict[str, Any]] = []

        async def _fake_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hi", model_extra=None))],
                usage=None,
            )

        async def _create(**kwargs: Any):
            calls.append(kwargs)
            if "stream_options" in kwargs:
                raise BadRequestError(
                    message,
                    response=_openai_error_response(400),
                    body=body,
                )
            return _fake_stream()

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = _create
            chunks = [
                chunk
                async for chunk in cast(Any, p).stream(
                    [{"role": "user", "content": "hi"}],
                    "gpt",
                )
            ]

        assert chunks == ["hi"]
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_stream_falls_back_for_422_stream_options_validation(self):
        from openai import UnprocessableEntityError

        p = get_provider("openai", api_key="test-key")
        calls: list[dict[str, Any]] = []

        async def _fake_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hi", model_extra=None))],
                usage=None,
            )

        async def _create(**kwargs: Any):
            calls.append(kwargs)
            if "stream_options" in kwargs:
                raise UnprocessableEntityError(
                    "Request validation failed",
                    response=_openai_error_response(422),
                    body={
                        "detail": [
                            {
                                "loc": ["body", "stream_options"],
                                "msg": "extra inputs are not permitted",
                            }
                        ]
                    },
                )
            return _fake_stream()

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = _create
            chunks = [
                chunk
                async for chunk in cast(Any, p).stream(
                    [{"role": "user", "content": "hi"}],
                    "gpt",
                )
            ]

        assert chunks == ["hi"]
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_stream_does_not_retry_provider_content_inspection_error(self):
        from openai import BadRequestError

        p = get_provider("openai", api_key="test-key")
        calls: list[dict[str, Any]] = []
        error_body = {
            "error": {
                "message": "Provider returned error",
                "code": 400,
                "metadata": {
                    "raw": (
                        'data: {"error":{"code":"data_inspection_failed",'
                        '"message":"Input text data may contain inappropriate content."}}'
                    ),
                    "provider_name": "Alibaba",
                },
            }
        }

        async def _create(**kwargs: Any):
            calls.append(kwargs)
            raise BadRequestError(
                "Provider returned error",
                response=_openai_error_response(400),
                body=error_body,
            )

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = _create
            stream = cast(Any, p).stream([{"role": "user", "content": "hi"}], "qwen")
            with pytest.raises(BadRequestError, match="Provider returned error"):
                _ = [chunk async for chunk in stream]

        assert len(calls) == 1
        assert "stream_options" in calls[0]

    @pytest.mark.asyncio
    async def test_stream_does_not_retry_on_non_badrequest_error(self):
        p = get_provider("openai", api_key="test-key")
        calls: list[dict[str, Any]] = []

        async def _create(**kwargs: Any):
            calls.append(kwargs)
            raise RuntimeError("network down")

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = _create
            stream = cast(Any, p).stream([{"role": "user", "content": "hi"}], "gpt")
            with pytest.raises(RuntimeError, match="network down"):
                _ = [c async for c in stream]

        assert len(calls) == 1  # genuine errors are not retried

    @pytest.mark.asyncio
    async def test_complete_returns_usage_and_cost_metadata(self):
        p = get_provider("openai", api_key="test-key")
        usage = SimpleNamespace(
            prompt_tokens=4,
            completion_tokens=3,
            total_tokens=7,
            cost=0.002,
        )
        mock_response = SimpleNamespace(usage=usage)
        mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await p.complete([{"role": "user", "content": "hi"}], "gpt-5.4-mini")

        assert isinstance(result, CompletionOutput)
        assert result == "hello"
        assert result.usage_details == {
            "prompt_tokens": 4,
            "completion_tokens": 3,
            "total_tokens": 7,
        }
        assert result.cost_details == {"total": 0.002}

    @pytest.mark.asyncio
    async def test_complete_captures_provider_extra_token_counters(self):
        # DeepSeek-style flat counters arrive as SDK ``model_extra`` fields.
        class _Usage:
            model_extra = {"prompt_cache_hit_tokens": 8, "prompt_cache_miss_tokens": 2}

            def __init__(self) -> None:
                self.prompt_tokens = 10
                self.completion_tokens = 5
                self.total_tokens = 15

        p = get_provider("openai", api_key="test-key")
        mock_response = SimpleNamespace(usage=_Usage())
        mock_response.choices = [MagicMock(message=MagicMock(content="hi"))]
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await p.complete([{"role": "user", "content": "hi"}], "deepseek-flash")
        assert result.usage_details == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_cache_hit_tokens": 8,
            "prompt_cache_miss_tokens": 2,
        }

    @pytest.mark.asyncio
    async def test_complete_flattens_nested_token_details(self):
        # OpenAI/Azure/Zhipu-style nested detail objects are flattened.
        p = get_provider("openai", api_key="test-key")
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=SimpleNamespace(cached_tokens=80),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=20),
        )
        mock_response = SimpleNamespace(usage=usage)
        mock_response.choices = [MagicMock(message=MagicMock(content="hi"))]
        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await p.complete([{"role": "user", "content": "hi"}], "gpt-5.4-mini")
        assert result.usage_details == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details.cached_tokens": 80,
            "completion_tokens_details.reasoning_tokens": 20,
        }

    @pytest.mark.asyncio
    async def test_complete_routes_model_kwargs_to_extra_body(self):
        p = get_provider("openai", api_key="test-key")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        with patch.object(p, "_get_client") as mock_client:
            create_mock = AsyncMock(return_value=mock_response)
            mock_client.return_value.chat.completions.create = create_mock
            await p.complete(
                [{"role": "user", "content": "hi"}],
                "gpt-5.4-mini",
                model_kwargs={"enable_thinking": True},
            )
            call_kwargs = create_mock.call_args[1]
            assert call_kwargs["extra_body"] == {"enable_thinking": True}

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        p = get_provider("openai", api_key="test-key")

        async def fake_stream():
            for text in ["hel", "lo"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content=text))]
                yield chunk

        with patch.object(p, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=fake_stream())
            tokens = []
            async for t in cast(Any, p).stream([{"role": "user", "content": "hi"}], "gpt-5.4-mini"):
                tokens.append(t)
        assert tokens == ["hel", "lo"]

    @pytest.mark.asyncio
    async def test_stream_tool_text_replays_reasoning_state(self):
        p = get_provider("openai", api_key="test-key")

        async def fake_stream():
            for text in ("final ", "answer"):
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(delta=SimpleNamespace(content=text, model_extra=None))
                    ],
                    usage=None,
                )

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [],
                "provider_state": {
                    "reasoning_content": "thought",
                    "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
                },
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search",
                "content": "evidence",
                "is_error": False,
            },
        ]
        with patch.object(p, "_get_client") as client:
            create = AsyncMock(return_value=fake_stream())
            client.return_value.chat.completions.create = create
            tokens = [
                token
                async for token in p.stream_tool_text(
                    messages,
                    "mimo-v2.5",
                )
            ]

        assert tokens == ["final ", "answer"]
        first_request = create.await_args_list[0].kwargs
        replayed = first_request["messages"][0]
        assert "provider_state" not in replayed
        assert replayed["reasoning_content"] == "thought"
        assert replayed["reasoning_details"][0]["data"] == "opaque"


class TestGeminiProvider:
    @pytest.mark.asyncio
    async def test_complete_extracts_system_instruction(self):
        p = get_provider("gemini", api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = "reply"
        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            result = await p.complete(
                [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "hi"},
                ],
                "gemini-2.0-flash",
            )
            call_kwargs = mock_client.aio.models.generate_content.call_args[1]
            assert "Be concise." in str(call_kwargs.get("config", {}).get("system_instruction", ""))
        assert result == "reply"

    @pytest.mark.asyncio
    async def test_complete_preserves_token_limit_stop_reason(self):
        p = get_provider("gemini", api_key="test-key")
        response = SimpleNamespace(
            text="partial",
            candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
            usage_metadata=None,
        )
        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as genai:
            client = MagicMock()
            genai.Client.return_value = client
            client.aio.models.generate_content = AsyncMock(return_value=response)
            cast(Any, p)._client = None
            result = await p.complete(
                [{"role": "user", "content": "hi"}],
                "gemini-2.0-flash",
            )

        assert result == "partial"
        assert result.stop_reason == "length"

    @pytest.mark.asyncio
    async def test_role_mapping_assistant_to_model(self):
        p = get_provider("gemini", api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = "ok"
        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete(
                [
                    {"role": "assistant", "content": "I said hi"},
                    {"role": "user", "content": "continue"},
                ],
                "gemini-2.0-flash",
            )
            call_args = mock_client.aio.models.generate_content.call_args
            contents = call_args[1].get(
                "contents", call_args[0][1] if len(call_args[0]) > 1 else None
            )
            # Verify assistant → model role mapping
            assert any(c.get("role") == "model" for c in contents if isinstance(c, dict))

    @pytest.mark.asyncio
    async def test_complete_tool_turn_converts_tools_history_and_response(self):
        p = get_provider("gemini", api_key="test-key")
        tool = ToolDefinition(
            name="search_web",
            description="Search the open web.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )
        function_call = SimpleNamespace(
            id="call-2",
            name="search_web",
            args={"query": "inflation"},
        )
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    finish_reason="STOP",
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text=None,
                                thought=False,
                                function_call=function_call,
                                thought_signature="gemini-signature",
                            )
                        ]
                    ),
                )
            ],
            usage_metadata=SimpleNamespace(prompt_token_count=8, candidates_token_count=3),
        )
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"prices"}',
                        },
                        "thought_signature": "previous-gemini-signature",
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search_web",
                "content": "price evidence",
                "is_error": False,
            },
        ]

        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as sdk:
            client = MagicMock()
            sdk.Client.return_value = client
            create = AsyncMock(return_value=response)
            client.aio.models.generate_content = create
            cast(Any, p)._client = None
            turn = await p.complete_tool_turn(
                messages,
                "gemini-2.5-flash",
                tools=[tool],
                tool_choice="required",
            )

        await_args = create.await_args
        assert await_args is not None
        request = await_args.kwargs
        assert request["config"]["tools"] == [
            {
                "function_declarations": [
                    {
                        "name": "search_web",
                        "description": "Search the open web.",
                        "parameters": tool.parameters,
                    }
                ]
            }
        ]
        assert request["config"]["tool_config"] == {"function_calling_config": {"mode": "ANY"}}
        assert request["contents"] == [
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "id": "call-1",
                            "name": "search_web",
                            "args": {"query": "prices"},
                        },
                        "thought_signature": "previous-gemini-signature",
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "call-1",
                            "name": "search_web",
                            "response": {"output": "price evidence", "is_error": False},
                        }
                    }
                ],
            },
        ]
        assert turn.stop_reason == "tool_use"
        assert turn.tool_calls[0].id == "call-2"
        assert turn.tool_calls[0].arguments == {"query": "inflation"}
        assert turn.tool_calls[0].thought_signature == "gemini-signature"
        assert turn.usage_details == {"prompt_tokens": 8, "candidates_tokens": 3}

    @pytest.mark.asyncio
    async def test_complete_tool_turn_streaming_preserves_text_thoughts_and_calls(self):
        p = get_provider("gemini", api_key="test-key")
        function_call = SimpleNamespace(id="call-1", name="search_web", args={"query": "inflation"})

        async def fake_stream():
            yield SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        finish_reason=None,
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(text="Think.", thought=True, function_call=None),
                                SimpleNamespace(text="Draft ", thought=False, function_call=None),
                            ]
                        ),
                    ),
                    SimpleNamespace(
                        finish_reason=None,
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(text="alternate", thought=False, function_call=None)
                            ]
                        ),
                    ),
                ],
                usage_metadata=None,
            )
            yield SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        finish_reason="STOP",
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    text=None,
                                    thought=False,
                                    function_call=function_call,
                                    thought_signature="signed",
                                )
                            ]
                        ),
                    )
                ],
                usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=3),
            )

        emitted: list[str] = []

        async def emit_text(text: str) -> None:
            emitted.append(text)

        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as sdk:
            client = MagicMock()
            sdk.Client.return_value = client
            client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
            cast(Any, p)._client = None
            turn = await p.complete_tool_turn_streaming(
                [{"role": "user", "content": "latest inflation"}],
                "gemini-2.5-flash",
                tools=[],
                emit_text=emit_text,
            )

        assert emitted == ["Draft "]
        assert turn.text == "Draft "
        assert turn.reasoning == "Think."
        assert turn.stop_reason == "tool_use"
        assert turn.tool_calls[0].arguments == {"query": "inflation"}
        assert turn.tool_calls[0].thought_signature == "signed"
        assert turn.usage_details == {"prompt_tokens": 5, "candidates_tokens": 3}

    @pytest.mark.asyncio
    async def test_stream_uses_gemini_async_stream_api(self):
        p = get_provider("gemini", api_key="test-key")

        async def fake_stream():
            for text in ("hel", "lo"):
                yield SimpleNamespace(text=text)

        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
            cast(Any, p)._client = None
            tokens = [
                token
                async for token in cast(Any, p).stream(
                    [{"role": "user", "content": "hi"}],
                    "gemini-2.0-flash",
                )
            ]

        assert tokens == ["hel", "lo"]
        mock_client.aio.models.generate_content_stream.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_captures_usage_metadata(self):
        p = get_provider("gemini", api_key="test-key")
        holder: dict[str, Any] = {}

        async def fake_stream():
            yield SimpleNamespace(text="hel", usage_metadata=None)
            yield SimpleNamespace(
                text="lo",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12, candidates_token_count=4, total_token_count=16
                ),
            )

        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
            cast(Any, p)._client = None
            tokens = [
                t
                async for t in cast(Any, p).stream(
                    [{"role": "user", "content": "hi"}],
                    "gemini-2.0-flash",
                    usage_holder=holder,
                )
            ]

        assert tokens == ["hel", "lo"]
        assert holder == {
            "usage_details": {
                "prompt_token_count": 12,
                "candidates_token_count": 4,
                "total_token_count": 16,
            }
        }

    @pytest.mark.asyncio
    async def test_stream_tool_text_replays_thought_signature(self):
        p = get_provider("gemini", api_key="test-key")

        async def fake_stream():
            yield SimpleNamespace(text="final ", usage_metadata=None)
            yield SimpleNamespace(text="answer", usage_metadata=None)

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                        "thought_signature": "signature",
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search",
                "content": "evidence",
                "is_error": False,
            },
        ]
        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as sdk:
            client = MagicMock()
            sdk.Client.return_value = client
            stream = AsyncMock(return_value=fake_stream())
            client.aio.models.generate_content_stream = stream
            cast(Any, p)._client = None
            tokens = [
                token
                async for token in p.stream_tool_text(
                    messages,
                    "gemini-2.5-flash",
                )
            ]

        assert tokens == ["final ", "answer"]
        await_args = stream.await_args
        assert await_args is not None
        first_part = await_args.kwargs["contents"][0]["parts"][0]
        assert first_part["thought_signature"] == "signature"

    @pytest.mark.asyncio
    async def test_json_schema_response_format_uses_response_schema(self):
        p = get_provider("gemini", api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = '{"answer": "ok"}'
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            await p.complete(
                [{"role": "user", "content": "hi"}],
                "gemini-2.5-flash",
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "demo_plan", "schema": schema, "strict": True},
                },
            )
            call_kwargs = mock_client.aio.models.generate_content.call_args[1]
            assert call_kwargs["config"]["response_mime_type"] == "application/json"
            assert call_kwargs["config"]["response_schema"] == schema

    @pytest.mark.asyncio
    async def test_aclose_closes_async_client(self):
        p = get_provider("gemini", api_key="test-key")
        mock_client = MagicMock()
        mock_client.aio.aclose = AsyncMock()
        cast(Any, p)._client = mock_client
        await p.aclose()
        mock_client.aio.aclose.assert_awaited_once()
        assert cast(Any, p)._client is None

    @pytest.mark.asyncio
    async def test_complete_captures_cache_and_thought_tokens(self):
        p = get_provider("gemini", api_key="test-key")
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_response.usage_metadata = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
            cached_content_token_count=80,
            thoughts_token_count=20,
        )
        with patch("dlightrag.engine.ai.providers.gemini_native.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            cast(Any, p)._client = None
            result = await p.complete(
                [{"role": "user", "content": "hi"}],
                "gemini-2.5-flash",
            )
        assert result.usage_details == {
            "prompt_tokens": 100,
            "candidates_tokens": 50,
            "total_tokens": 150,
            "cached_content_tokens": 80,
            "thoughts_tokens": 20,
        }


async def test_empty_tool_calls_arrays_are_stripped_for_strict_endpoints():
    OpenAICompatibleProvider(
        api_key="test-key",
        base_url="http://localhost:8888/v1",
        timeout=10.0,
        max_retries=1,
    )
    messages = [
        {"role": "assistant", "content": "text", "tool_calls": []},
        {
            "role": "assistant",
            "content": "tools",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
    ]

    converted = _openai_tool_messages(messages)

    assert "tool_calls" not in converted[0]
    assert converted[1]["tool_calls"] == messages[1]["tool_calls"]
