# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for answer-model image capability derivation."""

import asyncio
import base64
import dataclasses
import io
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from dlightrag.application.config import DlightragConfig
from dlightrag.application.settings import (
    answer_capability_settings,
    answer_resource_settings,
    model_profile_for_role,
    model_settings_for_role,
    rerank_scoring_model_settings,
)
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import (
    EmbeddingSettings,
    ModelRoleOverrides,
    ModelRoleSettings,
    ModelSettings,
)
from dlightrag.engine.ai.vision import (
    ImageCapabilityStatus,
    ImageProbeOutcome,
    ModelImageCapabilities,
)
from dlightrag.engine.answer.capabilities import (
    AnswerCapabilities,
    AnswerCapabilityCoordinator,
    AnswerCapabilityView,
)
from dlightrag.engine.answer.execution import AnswerResourceResolver
from dlightrag.engine.answer.image_capability import (
    AnswerImageCapability,
    derive_effective_max_images,
)
from tests.config_helpers import mutate_config, replace_config


@pytest.mark.parametrize(
    ("status", "configured_ceiling", "expected"),
    [
        pytest.param("unsupported", 0, 0, id="unsupported_zero_ceiling_forces_zero"),
        pytest.param("supported", 6, 6, id="supported_uses_configured_ceiling"),
        pytest.param("unknown", 6, 0, id="unknown_is_zero"),
        pytest.param("unsupported", 6, 0, id="unsupported_is_zero"),
    ],
)
def test_derive_effective_max_images(
    status: ImageCapabilityStatus, configured_ceiling: int, expected: int
) -> None:
    assert derive_effective_max_images(status, configured_ceiling) == expected


def test_capability_snapshot_is_frozen() -> None:
    cap = AnswerImageCapability(
        status="supported",
        configured_ceiling=6,
        effective_max_images=6,
        provider="openai",
        base_url=None,
        model="gpt-4o",
        failure_kind=None,
    )
    assert cap.effective_max_images == 6

    with pytest.raises(dataclasses.FrozenInstanceError):
        cap.effective_max_images = 0  # type: ignore[misc]


def _coordinator(
    config: DlightragConfig,
    *,
    image_capabilities: ModelImageCapabilities | None = None,
    profile_for_role: Any | None = None,
) -> tuple[AnswerCapabilityCoordinator, list[dict[str, object]]]:
    health_updates: list[dict[str, object]] = []
    coordinator = AnswerCapabilityCoordinator(
        settings=answer_capability_settings(config),
        profile_for_role=profile_for_role or (lambda role: model_profile_for_role(config, role)),
        model_settings_for_role=lambda role: model_settings_for_role(config, role),
        rerank_model_settings=lambda: rerank_scoring_model_settings(config),
        image_capabilities=image_capabilities
        or ModelImageCapabilities(scheduler=ModelScheduler(max_concurrency=1)),
        on_answer_capability=health_updates.append,
    )
    return coordinator, health_updates


def _stub_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    *statuses: ImageCapabilityStatus,
) -> tuple[ModelImageCapabilities, AsyncMock]:
    capabilities = ModelImageCapabilities(scheduler=ModelScheduler(max_concurrency=1))
    resolve = AsyncMock(side_effect=[ImageProbeOutcome(status=status) for status in statuses])
    monkeypatch.setattr(capabilities, "resolve", resolve)
    return capabilities, resolve


def test_public_capability_snapshot_is_frozen() -> None:
    coordinator, _health_updates = _coordinator(_reprobe_config())
    snapshot = coordinator.snapshot

    assert isinstance(snapshot, AnswerCapabilities)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.vlm_status = "supported"  # type: ignore[misc]


async def test_catalogue_invalidation_during_answer_probe_cannot_recache_old_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    profile = ModelProfile(context_window_tokens=100_000, supports_images=True)
    profiles = {role: profile for role in ("extract", "query", "vlm")}
    image_capabilities = ModelImageCapabilities(scheduler=ModelScheduler(max_concurrency=1))

    async def resolve(_settings: ModelSettings) -> ImageProbeOutcome:
        started.set()
        await release.wait()
        return ImageProbeOutcome(status="supported")

    resolve_mock = AsyncMock(side_effect=resolve)
    monkeypatch.setattr(image_capabilities, "resolve", resolve_mock)
    coordinator, _updates = _coordinator(
        _reprobe_config(),
        image_capabilities=image_capabilities,
        profile_for_role=lambda role: profiles[role],
    )

    pending = asyncio.create_task(coordinator.probe_answer())
    await started.wait()
    profiles["query"] = dataclasses.replace(profile, supports_images=False)
    coordinator.invalidate_model_catalogue()
    release.set()
    await pending

    assert coordinator.answer_image_capability is not None
    assert coordinator.answer_image_capability.status == "unsupported"
    assert coordinator.answer_image_capability.failure_kind == "profile_declared_unsupported"
    resolve_mock.assert_awaited_once()


async def test_capability_probe_targets_resolved_query_role_without_borrowing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        models={
            "chat": ModelRoleSettings(
                default=ModelSettings(model="default-model", api_key="default-key"),
                roles=ModelRoleOverrides(
                    query=ModelSettings(
                        model="local-query",
                        api_key=None,
                        base_url="http://host.docker.internal:8888/v1",
                    )
                ),
            ),
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        },
    )
    coordinator, health_updates = _coordinator(config)
    probed: dict[str, object] = {}

    async def fake_probe(provider, *, model, model_kwargs=None):
        probed["model"] = model
        return ImageProbeOutcome(status="supported")

    class _StubProvider:
        async def aclose(self) -> None:
            pass

    def fake_get_provider(*_args, **kwargs):
        probed["api_key"] = kwargs["api_key"]
        return _StubProvider()

    monkeypatch.setattr("dlightrag.engine.ai.vision.get_provider", fake_get_provider)
    monkeypatch.setattr("dlightrag.engine.ai.vision.probe_image_capability", fake_probe)

    await coordinator.probe_answer()

    cap = coordinator.answer_image_capability
    ceiling = int(config.answer.generation.max_images)
    query_cfg = model_settings_for_role(config, "query")
    assert isinstance(cap, AnswerImageCapability)
    assert cap.status == "supported"
    assert cap.effective_max_images == ceiling
    assert probed["model"] == query_cfg.model
    assert probed["api_key"] is None
    assert health_updates[-1]["status"] == "supported"


def _reprobe_config() -> DlightragConfig:
    return DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        models={
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        },
    )


async def test_unknown_capability_lazily_reprobes_to_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_capabilities, resolve = _stub_capabilities(
        monkeypatch,
        "unknown",
        "supported",
    )
    coordinator, _health_updates = _coordinator(
        _reprobe_config(), image_capabilities=image_capabilities
    )
    view = AnswerCapabilityView(coordinator)

    await coordinator.probe_answer()
    snapshot = await view.read()

    cap = snapshot.answer
    assert resolve.await_count == 2
    assert cap is not None and cap.status == "supported"
    assert cap.effective_max_images == _reprobe_config().answer.generation.max_images


async def test_confirmed_image_context_refreshes_the_query_profile_after_reprobe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_capabilities, _resolve = _stub_capabilities(
        monkeypatch,
        "unknown",
        "supported",
    )
    coordinator, _health_updates = _coordinator(
        _reprobe_config(), image_capabilities=image_capabilities
    )
    await coordinator.probe_answer()
    before = coordinator.request_model_context(None)

    after, capability = await coordinator.confirmed_live_answer_context(before)

    assert before.query.supports_images is False
    assert after.query.supports_images is True
    assert capability is not None and capability.status == "supported"


async def test_reprobe_updates_synthesizer_image_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.engine.answer.synthesizer import AnswerSynthesizer

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    contexts = {
        "chunks": [
            {
                "chunk_id": "visual-1",
                "file_path": "chart.png",
                "content": "chart",
                "image_data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
        ],
        "entities": [],
        "relationships": [],
    }
    image_capabilities, _resolve = _stub_capabilities(
        monkeypatch,
        "unknown",
        "supported",
    )
    coordinator, _health_updates = _coordinator(
        _reprobe_config(), image_capabilities=image_capabilities
    )
    await coordinator.probe_answer()
    old_profile = coordinator.model_profile("query")
    synthesizer = AnswerSynthesizer(
        image_policy=coordinator.answer_image_policy(old_profile),
        model_profile=old_profile,
    )

    before = synthesizer._prepare_prompt_context("question", contexts)
    await coordinator.refresh_answer()
    refreshed_profile = coordinator.model_profile("query")
    refreshed = AnswerSynthesizer(
        image_policy=coordinator.answer_image_policy(refreshed_profile),
        model_profile=refreshed_profile,
    )
    after = refreshed._prepare_prompt_context("question", contexts)

    assert before.trace["answer_context_images_sent"] == 0
    assert refreshed is not synthesizer
    assert after.trace["answer_context_images_sent"] == 1


@pytest.mark.parametrize(
    "status",
    [
        pytest.param("supported", id="supported_is_terminal_no_reprobe"),
        pytest.param("unsupported", id="unsupported_is_terminal_no_reprobe"),
    ],
)
async def test_terminal_status_is_terminal_no_reprobe(
    monkeypatch: pytest.MonkeyPatch,
    status: ImageCapabilityStatus,
) -> None:
    image_capabilities, resolve = _stub_capabilities(
        monkeypatch,
        status,
        "unknown" if status == "supported" else "supported",
    )
    coordinator, _health_updates = _coordinator(
        _reprobe_config(), image_capabilities=image_capabilities
    )
    await coordinator.probe_answer()
    await coordinator.refresh_answer()

    cap = coordinator.answer_image_capability
    assert resolve.await_count == 1
    assert cap is not None and cap.status == status


async def test_reprobe_respects_cooldown_when_still_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The cooldown lives in the shared probe cache, so a capability re-probe only
    # reaches a model call when that model's own cooldown has elapsed.
    probed = _probed_models(monkeypatch, "unknown")
    coordinator, _health_updates = _coordinator(_reprobe_config())

    await coordinator.probe_answer()
    await coordinator.refresh_answer()

    assert len(probed) == 1


# --- Role-specific capability resolved from each role's own model config ------


def _probed_models(monkeypatch: pytest.MonkeyPatch, *statuses: ImageCapabilityStatus) -> list[str]:
    """Record every probed model, replying with *statuses* in order (last repeats)."""
    probed: list[str] = []
    replies = list(statuses) or ["supported"]

    async def fake_probe(_provider, *, model, model_kwargs=None):
        probed.append(model)
        return ImageProbeOutcome(status=replies[min(len(probed) - 1, len(replies) - 1)])

    class _StubProvider:
        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(
        "dlightrag.engine.ai.vision.get_provider", lambda *_a, **_k: _StubProvider()
    )
    monkeypatch.setattr("dlightrag.engine.ai.vision.probe_image_capability", fake_probe)
    return probed


async def test_identical_resolved_configurations_share_one_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed = _probed_models(monkeypatch, "supported")
    capabilities = ModelImageCapabilities(scheduler=ModelScheduler(max_concurrency=1))
    first = ModelSettings(
        provider="openai", model="shared", api_key="k", base_url="https://api.example/v1"
    )
    second = ModelSettings(
        provider="openai", model="shared", api_key="k", base_url="https://api.example/v1"
    )

    assert (await capabilities.resolve(first)).status == "supported"
    assert (await capabilities.resolve(second)).status == "supported"
    assert probed == ["shared"]


async def test_distinct_resolved_configurations_are_probed_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed = _probed_models(monkeypatch, "supported", "unsupported")
    capabilities = ModelImageCapabilities(scheduler=ModelScheduler(max_concurrency=1))

    answer = await capabilities.resolve(
        ModelSettings(provider="openai", model="answer-model", api_key="k")
    )
    vlm = await capabilities.resolve(
        ModelSettings(provider="openai", model="vlm-model", api_key="k")
    )

    assert probed == ["answer-model", "vlm-model"]
    assert (answer.status, vlm.status) == ("supported", "unsupported")


async def test_distinct_capability_probes_share_scheduler_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = ModelScheduler(max_concurrency=1)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    calls = 0

    async def probe(_provider, *, model, model_kwargs=None):
        nonlocal calls
        del model, model_kwargs
        calls += 1
        if calls == 1:
            first_started.set()
            await release_first.wait()
        return ImageProbeOutcome(status="supported")

    class Provider:
        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("dlightrag.engine.ai.vision.get_provider", lambda *_a, **_k: Provider())
    monkeypatch.setattr("dlightrag.engine.ai.vision.probe_image_capability", probe)
    capabilities = ModelImageCapabilities(scheduler=scheduler)
    first = asyncio.create_task(
        capabilities.resolve(ModelSettings(provider="openai", model="first", api_key="k"))
    )
    await first_started.wait()
    second = asyncio.create_task(
        capabilities.resolve(ModelSettings(provider="openai", model="second", api_key="k"))
    )
    await asyncio.sleep(0)
    assert calls == 1

    release_first.set()
    assert [outcome.status for outcome in await asyncio.gather(first, second)] == [
        "supported",
        "supported",
    ]
    assert calls == 2


async def test_clear_during_inflight_probe_discards_the_stale_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    calls = 0

    async def probe(_provider, *, model, model_kwargs=None):
        nonlocal calls
        del model, model_kwargs
        calls += 1
        if calls == 1:
            first_started.set()
            await release_first.wait()
            return ImageProbeOutcome(status="supported")
        return ImageProbeOutcome(status="unsupported")

    class Provider:
        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("dlightrag.engine.ai.vision.get_provider", lambda *_a, **_k: Provider())
    monkeypatch.setattr("dlightrag.engine.ai.vision.probe_image_capability", probe)
    capabilities = ModelImageCapabilities(scheduler=ModelScheduler(max_concurrency=1))
    settings = ModelSettings(provider="openai", model="changed", api_key="k")

    pending = asyncio.create_task(capabilities.resolve(settings))
    await first_started.wait()
    capabilities.clear()
    release_first.set()

    assert (await pending).status == "unsupported"
    assert calls == 2
    assert (await capabilities.resolve(settings)).status == "unsupported"
    assert calls == 2


async def test_same_endpoint_with_a_different_key_is_not_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed = _probed_models(monkeypatch, "supported")
    capabilities = ModelImageCapabilities(scheduler=ModelScheduler(max_concurrency=1))

    await capabilities.resolve(ModelSettings(provider="openai", model="m", api_key="key-one"))
    await capabilities.resolve(ModelSettings(provider="openai", model="m", api_key="key-two"))

    assert probed == ["m", "m"]


async def test_only_unknown_reprobes_and_only_once_per_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed = _probed_models(monkeypatch, "unknown")
    capabilities = ModelImageCapabilities(
        scheduler=ModelScheduler(max_concurrency=1),
        reprobe_cooldown_seconds=3600.0,
    )
    unknown = ModelSettings(provider="openai", model="flaky", api_key="k")
    terminal = ModelSettings(provider="openai", model="steady", api_key="k")

    await capabilities.resolve(unknown)
    await capabilities.resolve(unknown)  # inside the cooldown -> no second probe
    assert probed == ["flaky"]

    monkeypatch.setattr(capabilities, "_cooldown_seconds", 0.0)
    await capabilities.resolve(unknown)
    assert probed == ["flaky", "flaky"]

    probed.clear()
    monkeypatch.setattr(
        "dlightrag.engine.ai.vision.probe_image_capability",
        _recording_probe(probed, "supported"),
    )
    await capabilities.resolve(terminal)
    await capabilities.resolve(terminal)
    assert probed == ["steady"]


def _recording_probe(sink: list[str], status: ImageCapabilityStatus):
    async def probe(_provider, *, model, model_kwargs=None):
        sink.append(model)
        return ImageProbeOutcome(status=status)

    return probe


async def test_a_slow_probe_does_not_spend_its_own_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable model must not be re-probed the moment its timeout returns."""
    import asyncio

    probed: list[str] = []

    async def slow_probe(_provider, *, model, model_kwargs=None):
        probed.append(model)
        await asyncio.sleep(0.05)
        return ImageProbeOutcome(status="unknown", failure_kind="TimeoutError")

    class _StubProvider:
        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(
        "dlightrag.engine.ai.vision.get_provider", lambda *_a, **_k: _StubProvider()
    )
    monkeypatch.setattr("dlightrag.engine.ai.vision.probe_image_capability", slow_probe)
    capabilities = ModelImageCapabilities(
        scheduler=ModelScheduler(max_concurrency=1),
        reprobe_cooldown_seconds=0.04,
    )
    cfg = ModelSettings(provider="openai", model="unreachable", api_key="k")

    await capabilities.resolve(cfg)
    await capabilities.resolve(cfg)

    assert probed == ["unreachable"]


async def test_concurrent_resolution_of_one_configuration_probes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    probed = _probed_models(monkeypatch, "supported")
    capabilities = ModelImageCapabilities(scheduler=ModelScheduler(max_concurrency=1))
    cfg = ModelSettings(provider="openai", model="single-flight", api_key="k")

    await asyncio.gather(*(capabilities.resolve(cfg) for _ in range(4)))

    assert probed == ["single-flight"]


async def test_cancelled_probe_finishes_provider_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    class Provider:
        async def aclose(self) -> None:
            close_started.set()
            await release_close.wait()

    async def cancelled_probe(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr("dlightrag.engine.ai.vision.get_provider", lambda *_a, **_k: Provider())
    monkeypatch.setattr("dlightrag.engine.ai.vision.probe_image_capability", cancelled_probe)
    capabilities = ModelImageCapabilities(scheduler=ModelScheduler(max_concurrency=1))
    task = asyncio.create_task(
        capabilities.resolve(ModelSettings(provider="openai", model="cancelled", api_key="k"))
    )
    await close_started.wait()
    release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await task


def _role_config(**roles: ModelSettings) -> DlightragConfig:
    default = ModelSettings(model="default-model", api_key="default-key")
    return DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        models={
            "chat": ModelRoleSettings(
                default=default,
                roles=ModelRoleOverrides(**roles),
            ),
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        },
    )


async def test_inspect_follows_vlm_capability_not_answer_capability() -> None:
    from dlightrag.engine.answer.resources import ResourceInput
    from dlightrag.engine.answer.resources.models import TextWindowBudget

    config = _role_config()
    capabilities, _ = _coordinator(config)
    capabilities.resolve_profiles()
    capabilities.narrow_role_image_profile("vlm", "supported")
    resolver = AnswerResourceResolver(
        settings=answer_resource_settings(config),
        models=cast(Any, SimpleNamespace(vlm_func=MagicMock(return_value=AsyncMock()))),
        capabilities=capabilities,
    )

    _registry, tools = resolver.build_resource_context(
        [ResourceInput(filename="chart.png", content=b"\x89PNG", declared_mime="image/png")],
        text_window_budget=TextWindowBudget(tokens=1_000),
        vlm_profile=capabilities.model_profile("vlm"),
    )

    assert [tool.name for tool in tools] == ["inspect"]


async def test_inspect_is_withheld_when_only_the_answer_model_sees_images() -> None:
    from dlightrag.engine.answer.resources import ResourceInput
    from dlightrag.engine.answer.resources.models import TextWindowBudget

    config = _role_config()
    capabilities, _ = _coordinator(config)
    capabilities.resolve_profiles()
    capabilities.narrow_role_image_profile("vlm", "unsupported")
    resolver = AnswerResourceResolver(
        settings=answer_resource_settings(config),
        models=cast(Any, SimpleNamespace(vlm_func=MagicMock(return_value=AsyncMock()))),
        capabilities=capabilities,
    )

    _registry, tools = resolver.build_resource_context(
        [ResourceInput(filename="chart.png", content=b"\x89PNG", declared_mime="image/png")],
        text_window_budget=TextWindowBudget(tokens=1_000),
        vlm_profile=dataclasses.replace(
            capabilities.model_profile("vlm"),
            supports_images=False,
        ),
    )

    assert [tool.name for tool in tools] == []


async def test_zero_configured_ceiling_disables_answer_images_without_a_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed = _probed_models(monkeypatch, "supported")
    config = _role_config()
    mutate_config(config, "answer.generation.max_images", 0)
    coordinator, _health_updates = _coordinator(config)

    await coordinator.probe_answer()
    capability = coordinator.answer_image_capability

    assert probed == []
    assert capability is not None
    assert capability.status == "unsupported"
    assert capability.failure_kind == "config_disabled"
    assert capability.effective_max_images == 0


async def test_live_probe_cannot_widen_profile_declared_image_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        models={
            "chat": ModelRoleSettings(
                roles=ModelRoleOverrides(
                    query=ModelSettings(
                        provider="openai",
                        model="declared-text-only",
                        base_url="https://example.invalid/v1",
                        api_key=None,
                    )
                )
            ),
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        },
    )
    image_capabilities, resolve = _stub_capabilities(monkeypatch, "supported")
    coordinator, _health_updates = _coordinator(
        config,
        image_capabilities=image_capabilities,
        profile_for_role=lambda role: (
            ModelProfile(
                context_window_tokens=1_048_576,
                max_output_tokens=262_144,
                supports_images=False,
            )
            if role == "query"
            else model_profile_for_role(config, role)
        ),
    )

    await coordinator.probe_answer()

    capability = coordinator.answer_image_capability
    assert capability is not None
    assert capability.status == "unsupported"
    assert capability.failure_kind == "profile_declared_unsupported"
    assert coordinator.model_profile("query").supports_images is False
    resolve.assert_not_awaited()


async def test_zero_configured_ceiling_settles_the_vlm_role_without_a_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No role has an image slot under a zero ceiling, so the probe buys nothing."""
    probed = _probed_models(monkeypatch, "supported")
    config = _role_config()
    mutate_config(config, "answer.generation.max_images", 0)
    coordinator, _health_updates = _coordinator(config)

    await coordinator.probe_vlm()

    assert probed == []
    assert coordinator.snapshot.vlm_status == "unsupported"


async def test_rerank_capability_is_probed_from_the_rerank_scoring_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.engine.ai.settings import RerankSettings

    config = _role_config()
    config = replace_config(
        config,
        "models.rerank",
        RerankSettings(
            enabled=True,
            strategy="chat_llm_reranker",
            provider="openai",
            model="rerank-scorer",
            api_key=None,
            base_url="http://host.docker.internal:9999/v1",
        ),
    )
    provider = type("Provider", (), {"aclose": AsyncMock()})()
    provider_factory = MagicMock(return_value=provider)
    probe = AsyncMock(return_value=ImageProbeOutcome(status="unsupported"))
    monkeypatch.setattr("dlightrag.engine.ai.vision.get_provider", provider_factory)
    monkeypatch.setattr("dlightrag.engine.ai.vision.probe_image_capability", probe)
    coordinator, _health_updates = _coordinator(config)

    await coordinator.probe_rerank()

    provider_factory.assert_called_once_with(
        "openai",
        api_key=None,
        base_url="http://host.docker.internal:9999/v1",
        timeout=240.0,
        max_retries=3,
    )
    probe.assert_awaited_once_with(provider, model="rerank-scorer", model_kwargs=None)
    assert coordinator.rerank_supports_vision is False
