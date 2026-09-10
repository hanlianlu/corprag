# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for root settings mapping and AI model factories."""

import asyncio
import operator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from dlightrag.application.config import DlightragConfig
from dlightrag.application.settings import (
    model_profile_for_role,
    model_settings_for_role,
    rerank_scoring_model_settings,
)
from dlightrag.engine.ai.completion import CompletionModel
from dlightrag.engine.ai.providers.base import CompletionOutput
from dlightrag.engine.ai.scheduler import ModelScheduler, model_call_scope
from dlightrag.engine.ai.settings import (
    EmbeddingSettings,
    ModelRoleOverrides,
    ModelRoleSettings,
    ModelSettings,
    RerankSettings,
)
from dlightrag.engine.ai.structured import StructuredOutput


class DemoPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


DEMO_STRUCTURED_OUTPUT = StructuredOutput(name="demo_plan", schema=DemoPlan)


class CapturingProvider:
    supports_native_json_schema: bool = False

    def __init__(self, seen: dict[str, Any], *, supports_native_json_schema: bool = False) -> None:
        self.seen = seen
        self.supports_native_json_schema = supports_native_json_schema

    async def complete(self, **kwargs: Any) -> str:
        self.seen.update(kwargs)
        return '{"answer": "ok"}'

    def stream(self, **kwargs: Any):  # pragma: no cover - not used
        raise AssertionError("stream should not be called")

    async def aclose(self) -> None:
        return None


def _capture_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    supports_native_json_schema: bool = False,
) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: CapturingProvider(
            seen,
            supports_native_json_schema=supports_native_json_schema,
        ),
    )
    return seen


def _embedding_config() -> EmbeddingSettings:
    return EmbeddingSettings(
        provider="voyage",
        model="voyage-multimodal-3.5",
        api_key="sk-test",
        startup_probe=False,
    )


def test_root_maps_explicit_keyless_role_to_immutable_ai_settings() -> None:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        models={
            "chat": ModelRoleSettings(
                default=ModelSettings(model="default-model", api_key="default-key"),
                roles=ModelRoleOverrides(
                    query=ModelSettings(
                        model="local-query",
                        api_key=None,
                        base_url="http://host.docker.internal:8888/v1",
                        model_kwargs={"reasoning": {"enabled": False}},
                    )
                ),
            ),
            "embedding": _embedding_config(),
        },
    )

    settings = model_settings_for_role(config, "query")

    assert settings.model == "local-query"
    assert settings.api_key is None
    assert settings.base_url == "http://host.docker.internal:8888/v1"
    with pytest.raises(TypeError):
        operator.setitem(settings.model_kwargs["reasoning"], "enabled", True)


def test_root_resolves_model_profiles_independently_per_role() -> None:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        models={
            "chat": ModelRoleSettings(
                default=ModelSettings(
                    model="google/gemini-3.8-flash",
                    base_url="https://openrouter.ai/api/v1",
                ),
                roles=ModelRoleOverrides(
                    extract=ModelSettings(
                        model="deepseek-flash",
                        base_url="https://api.deepseek.com",
                        api_key=None,
                    ),
                    query=ModelSettings(
                        model="private-query",
                        base_url="http://localhost:8888/v1",
                        api_key=None,
                    ),
                ),
            ),
            "embedding": _embedding_config(),
        },
    )

    assert model_profile_for_role(config, "keyword").max_output_tokens == 65_536
    assert model_profile_for_role(config, "extract").max_output_tokens == 384_000
    query = model_profile_for_role(config, "query")
    assert query.context_window_tokens == 1_048_576
    assert query.max_input_tokens is None


def test_model_fingerprint_canonicalizes_endpoint_without_retaining_url() -> None:
    from dlightrag.engine.ai.fingerprints import model_fingerprint

    first = model_fingerprint(
        ModelSettings(
            provider="openai",
            model="model-a",
            base_url="HTTPS://API.EXAMPLE.COM:443/v1/../v1/?token=secret",
        )
    )
    second = model_fingerprint(
        ModelSettings(
            provider="openai",
            model="model-a",
            base_url="https://api.example.com/v1",
        )
    )

    assert first == second
    assert first.provider == "openai"
    assert first.model == "model-a"
    assert first.endpoint_fingerprint is not None
    assert "example.com" not in first.endpoint_fingerprint


async def test_ai_completion_model_owns_provider_telemetry_and_lifecycle(monkeypatch) -> None:
    class Provider:
        closed = False

        async def complete(self, **kwargs: Any) -> CompletionOutput:
            assert kwargs["model"] == "model-a"
            return CompletionOutput("answer", usage_details={"input_tokens": 3})

        async def aclose(self) -> None:
            self.closed = True

    class Observation:
        updates: list[dict[str, Any]] = []

        def update(self, **kwargs: Any) -> None:
            self.updates.append(kwargs)

    class Telemetry:
        observation = Observation()
        capture_sensitive_data = False

        @asynccontextmanager
        async def observe(self, name: str, **kwargs: Any):
            assert name == "llm_model-a"
            assert kwargs["as_type"] == "generation"
            yield self.observation

    provider = Provider()
    monkeypatch.setattr(
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    model = CompletionModel(
        ModelSettings(provider="openai", model="model-a", api_key="key"),
        scheduler=ModelScheduler(max_concurrency=1),
        telemetry=Telemetry(),
    )

    result = await model(messages=[{"role": "user", "content": "question"}])
    await model.aclose()

    assert result == "answer"
    assert provider.closed is True
    assert Telemetry.observation.updates == [
        {
            "output": {"text_length": 6},
            "usage_details": {"input_tokens": 3},
            "cost_details": None,
        }
    ]


async def test_root_maps_embedding_settings_into_ai_factory() -> None:
    from dlightrag.engine.ai import embedding
    from dlightrag.engine.ai.providers.embed_providers import VoyageEmbedProvider

    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        models={
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="embed-key",
                dim=1024,
                input_modality="multimodal",
            ).model_copy(
                update={
                    "timeout": 45,
                }
            ),
        },
    )

    settings = config.models.embedding
    model = embedding.create_embedding_model(
        settings,
        scheduler=ModelScheduler(max_concurrency=1),
    )

    assert settings.timeout == 45
    assert isinstance(model.provider, VoyageEmbedProvider)
    assert model.base_url == "https://api.voyageai.com/v1"
    assert model.fingerprint.endpoint_fingerprint is not None
    assert model.dim == 1024
    await model.aclose()


def test_root_maps_rerank_settings_to_immutable_ai_value() -> None:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        models={
            "rerank": RerankSettings(
                strategy="voyage_reranker",
                model="rerank-2.5",
                api_key="rerank-key",
                input_modality="multimodal",
                score_threshold=0.42,
                max_concurrency=3,
                batch_size=5,
                model_kwargs={"truncation": {"enabled": False}},
            ),
            "embedding": _embedding_config(),
        },
    )

    settings = config.models.rerank

    assert settings.strategy == "voyage_reranker"
    assert settings.score_threshold == 0.42
    assert settings.max_concurrency == 3
    assert settings.batch_size == 5
    assert settings.model_kwargs == {"truncation": {"enabled": False}}


def test_chat_rerank_scoring_settings_preserve_model_kwargs_and_temperature() -> None:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        models={
            "rerank": RerankSettings(
                strategy="chat_llm_reranker",
                provider="openai",
                model="scoring-model",
                api_key="key",
                temperature=None,
                model_kwargs={"reasoning": {"enabled": False}},
            ),
            "embedding": _embedding_config(),
        },
    )

    scoring = rerank_scoring_model_settings(config)

    assert scoring.model_kwargs == {"reasoning": {"enabled": False}}
    assert scoring.temperature == 0.0


def test_rerank_scoring_settings_ignore_unrelated_role_overrides() -> None:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        models={
            "chat": ModelRoleSettings(
                default=ModelSettings(model="default-model", api_key="default-key"),
                roles=ModelRoleOverrides(
                    query=ModelSettings(model="query-model", api_key="query-key"),
                    vlm=ModelSettings(model="vlm-model", api_key="vlm-key"),
                ),
            ),
            "rerank": RerankSettings(strategy="chat_llm_reranker"),
            "embedding": _embedding_config(),
        },
    )

    assert rerank_scoring_model_settings(config).model == "default-model"


@pytest.mark.parametrize("api_key", ["", "   "])
def test_incomplete_independent_reranker_falls_back_to_default(api_key: str) -> None:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        models={
            "chat": ModelRoleSettings(
                default=ModelSettings(model="default-model", api_key="default-key")
            ),
            "rerank": RerankSettings(
                strategy="chat_llm_reranker",
                provider="openai",
                model="incomplete-reranker",
                api_key=api_key,
            ),
            "embedding": _embedding_config(),
        },
    )

    assert rerank_scoring_model_settings(config).model == "default-model"


async def test_structured_output_uses_openai_json_schema(monkeypatch) -> None:
    seen = _capture_provider(monkeypatch)
    model = CompletionModel(
        ModelSettings(provider="openai", model="gpt-5.4-mini", api_key="sk-test"),
        scheduler=ModelScheduler(max_concurrency=1),
    )

    await model(
        messages=[{"role": "user", "content": "hi"}],
        structured_output=DEMO_STRUCTURED_OUTPUT,
    )

    response_format = seen["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "demo_plan"
    assert response_format["json_schema"]["strict"] is True


async def test_structured_output_auto_prefers_json_schema_for_compatible_endpoint(
    monkeypatch,
) -> None:
    seen = _capture_provider(monkeypatch)
    model = CompletionModel(
        ModelSettings(
            provider="openai",
            model="deepseek-flash",
            api_key="sk-test",
            base_url="https://api.deepseek.com",
        ),
        scheduler=ModelScheduler(max_concurrency=1),
    )

    await model(
        messages=[{"role": "user", "content": "hi"}],
        structured_output=DEMO_STRUCTURED_OUTPUT,
    )

    response_format = seen["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "demo_plan"
    assert response_format["json_schema"]["strict"] is True


async def test_explicit_json_schema_overrides_custom_endpoint(monkeypatch) -> None:
    seen = _capture_provider(monkeypatch)
    model = CompletionModel(
        ModelSettings(
            provider="openai",
            model="schema-model",
            api_key="sk-test",
            base_url="https://llm.example.test/v1",
            structured_output="json_schema",
        ),
        scheduler=ModelScheduler(max_concurrency=1),
    )

    await model(
        messages=[{"role": "user", "content": "hi"}],
        structured_output=DEMO_STRUCTURED_OUTPUT,
    )

    assert seen["response_format"]["type"] == "json_schema"
    assert seen["response_format"]["json_schema"]["name"] == "demo_plan"


@pytest.mark.parametrize(
    ("provider", "model_name"),
    [("anthropic", "claude-sonnet-4"), ("gemini", "gemini-2.5-flash")],
)
async def test_native_provider_auto_uses_json_schema(
    monkeypatch,
    provider,
    model_name,
) -> None:
    seen = _capture_provider(monkeypatch, supports_native_json_schema=True)
    model = CompletionModel(
        ModelSettings(provider=provider, model=model_name, api_key="sk-test"),
        scheduler=ModelScheduler(max_concurrency=1),
    )

    await model(
        messages=[{"role": "user", "content": "hi"}],
        structured_output=DEMO_STRUCTURED_OUTPUT,
    )

    assert seen["response_format"]["type"] == "json_schema"
    assert seen["response_format"]["json_schema"]["name"] == "demo_plan"


async def test_openai_strict_schema_failure_retries_json_object(monkeypatch) -> None:
    seen: list[dict[str, Any]] = []
    fallback_started = asyncio.Event()
    release_fallback = asyncio.Event()
    ordinary_started = asyncio.Event()

    class Provider:
        supports_native_json_schema = False

        async def complete(self, **kwargs: Any) -> str:
            seen.append(kwargs)
            response_format = kwargs["response_format"]
            if response_format is None:
                ordinary_started.set()
                return "ordinary"
            if response_format["type"] == "json_schema":
                raise RuntimeError("strict schemas unsupported")
            fallback_started.set()
            await release_fallback.wait()
            return '{"answer": "ok"}'

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    scheduler = ModelScheduler(max_concurrency=1)
    model = CompletionModel(
        ModelSettings(provider="openai", model="local-model", api_key="sk-test"),
        scheduler=scheduler,
    )

    async def structured_request() -> str:
        with model_call_scope("run-a"):
            return await model(
                messages=[{"role": "user", "content": "hi"}],
                structured_output=DEMO_STRUCTURED_OUTPUT,
            )

    async def ordinary_request() -> str:
        with model_call_scope("run-b"):
            return await model(messages=[{"role": "user", "content": "other"}])

    structured = asyncio.create_task(structured_request())
    await fallback_started.wait()
    ordinary = asyncio.create_task(ordinary_request())
    await asyncio.sleep(0)
    assert not ordinary_started.is_set()

    release_fallback.set()
    assert await structured == '{"answer": "ok"}'
    assert await ordinary == "ordinary"

    assert [call["response_format"]["type"] for call in seen[:2]] == [
        "json_schema",
        "json_object",
    ]
    assert seen[2]["response_format"] is None
