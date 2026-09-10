# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Strict parsing and content-addressing tests for the packaged model catalog."""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from dlightrag.engine.ai import catalog
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.catalog import CatalogueEntry
from dlightrag.engine.ai.fingerprints import ModelFingerprint, normalized_endpoint_fingerprint
from dlightrag.engine.ai.reasoning import ReasoningLevels, ReasoningProfile

_OPENAI_LEVELS = {
    "off": "none",
    "minimal": None,
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}
_ANTHROPIC_LEVELS = {
    "off": "disabled",
    "minimal": None,
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}
_GEMINI_LEVELS = {
    "off": None,
    "minimal": None,
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": None,
    "max": None,
}
_KIMI_LEVELS = {
    "off": None,
    "minimal": None,
    "low": "low",
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": "max",
}
_DEEPSEEK_ROUTER_LEVELS = {
    "off": "disabled",
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": "xhigh",
    "max": None,
}
_DEEPSEEK_CURRENT_LEVELS = {
    "off": "disabled",
    "minimal": None,
    "low": "low",
    "medium": None,
    "high": "high",
    "xhigh": "high",
    "max": "max",
}
_MIMO_LEVELS = {
    "off": "disabled",
    "minimal": None,
    "low": "enabled",
    "medium": None,
    "high": None,
    "xhigh": None,
    "max": None,
}
_GROK_LEVELS = {
    "off": None,
    "minimal": None,
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": None,
}
_QWEN_ROUTER_LEVELS = {
    "off": "disabled",
    "minimal": None,
    "low": "low",
    "medium": "medium",
    "high": None,
    "xhigh": "xhigh",
    "max": None,
}


def _valid_model(
    *,
    provider: object = "openai",
    model: object = "test-model",
    base_url: object = "https://api.example.test/v1",
) -> dict[str, object]:
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "profile": {
            "context_window_tokens": 1024,
            "max_input_tokens": None,
            "max_output_tokens": 128,
            "supports_images": False,
            "reasoning": {
                "format": "openai",
                "levels": {
                    "off": "none",
                    "minimal": "minimal",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": None,
                    "max": None,
                },
            },
        },
    }


def _payload(models: Sequence[object] | None = None) -> dict[str, object]:
    return {"models": list(models) if models is not None else [_valid_model()]}


def _load_text(
    text: str,
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[str, tuple[CatalogueEntry, ...]]:
    (tmp_path / "model_catalog.json").write_text(text, encoding="utf-8")
    monkeypatch.setattr(catalog, "files", lambda _package: tmp_path)
    return catalog._load_catalog()


def _load_payload(
    payload: object,
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[str, tuple[CatalogueEntry, ...]]:
    return _load_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False),
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )


def test_packaged_catalog_revision_is_derived_from_models() -> None:
    path = Path(catalog.__file__).with_name("model_catalog.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert set(payload) == {"models"}
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", catalog.MODEL_CATALOG_REVISION)


@pytest.mark.parametrize(
    (
        "provider",
        "model",
        "endpoint",
        "context_window",
        "max_input",
        "max_output",
        "reasoning_format",
        "reasoning_levels",
    ),
    [
        (
            "openai",
            "gpt-5.6-sol",
            None,
            1_050_000,
            922_000,
            128_000,
            "openai",
            _OPENAI_LEVELS,
        ),
        (
            "openai",
            "gpt-5.6-terra",
            None,
            1_050_000,
            922_000,
            128_000,
            "openai",
            _OPENAI_LEVELS,
        ),
        (
            "openai",
            "gpt-5.6-luna",
            None,
            1_050_000,
            922_000,
            128_000,
            "openai",
            _OPENAI_LEVELS,
        ),
        (
            "anthropic",
            "claude-fable-5",
            None,
            1_000_000,
            None,
            128_000,
            "anthropic",
            {**_ANTHROPIC_LEVELS, "off": None},
        ),
        (
            "anthropic",
            "claude-opus-5",
            None,
            1_000_000,
            None,
            128_000,
            "anthropic",
            _ANTHROPIC_LEVELS,
        ),
        (
            "gemini",
            "gemini-3.8-flash",
            None,
            1_048_576,
            None,
            65_536,
            "gemini",
            _GEMINI_LEVELS,
        ),
        (
            "openai",
            "kimi-k3",
            "https://api.moonshot.ai/v1",
            1_048_576,
            None,
            1_048_576,
            "openai",
            _KIMI_LEVELS,
        ),
        (
            "openai",
            "grok-4.6",
            "https://api.x.ai/v1",
            500_000,
            None,
            None,
            "openai",
            _GROK_LEVELS,
        ),
        (
            "openai",
            "glm-5.3-flash",
            "https://api.z.ai/api/paas/v4",
            1_000_000,
            None,
            131_072,
            "openai",
            _KIMI_LEVELS,
        ),
    ],
)
def test_packaged_catalogue_contains_requested_endpoint_profiles(
    provider: str,
    model: str,
    endpoint: str | None,
    context_window: int,
    max_input: int | None,
    max_output: int | None,
    reasoning_format: str,
    reasoning_levels: dict[str, str | None],
) -> None:
    matches = [
        entry
        for entry in catalog._BUILTIN_MODEL_ENTRIES
        if (entry.provider, entry.model, entry.base_url) == (provider, model, endpoint)
    ]

    assert len(matches) == 1
    profile = matches[0].profile
    assert profile.context_window_tokens == context_window
    assert profile.max_input_tokens == max_input
    assert profile.max_output_tokens == max_output
    assert profile.supports_images is True
    assert profile.reasoning is not None
    assert profile.reasoning.format == reasoning_format
    assert profile.reasoning.levels.as_dict() == reasoning_levels


def test_packaged_catalogue_contains_native_multimodal_deepseek_profile() -> None:
    matches = [
        entry
        for entry in catalog._BUILTIN_MODEL_ENTRIES
        if (entry.provider, entry.model, entry.base_url)
        == ("openai", "deepseek-flash", "https://api.deepseek.com")
    ]

    assert len(matches) == 1
    profile = matches[0].profile
    assert profile.context_window_tokens == 1_048_576
    assert profile.max_input_tokens is None
    assert profile.max_output_tokens == 384_000
    assert profile.supports_images is True
    assert profile.reasoning is not None
    assert profile.reasoning.format == "deepseek"
    assert profile.reasoning.levels.as_dict() == _DEEPSEEK_CURRENT_LEVELS


@pytest.mark.parametrize(
    ("model", "context_window", "max_input", "max_output", "images", "levels"),
    [
        (
            "anthropic/claude-fable-5",
            1_000_000,
            None,
            128_000,
            True,
            {**_ANTHROPIC_LEVELS, "off": None},
        ),
        (
            "anthropic/claude-opus-5",
            1_000_000,
            None,
            128_000,
            True,
            _ANTHROPIC_LEVELS,
        ),
        (
            "deepseek/deepseek-v4.1-flash",
            1_048_576,
            None,
            384_000,
            True,
            _DEEPSEEK_ROUTER_LEVELS,
        ),
        (
            "google/gemini-3.8-flash",
            1_048_576,
            None,
            65_536,
            True,
            _GEMINI_LEVELS,
        ),
        (
            "moonshotai/kimi-k3",
            1_048_576,
            None,
            943_718,
            True,
            _KIMI_LEVELS,
        ),
        (
            "openai/gpt-5.6-sol",
            1_050_000,
            922_000,
            128_000,
            True,
            _OPENAI_LEVELS,
        ),
        (
            "openai/gpt-5.6-terra",
            1_050_000,
            922_000,
            128_000,
            True,
            _OPENAI_LEVELS,
        ),
        (
            "openai/gpt-5.6-luna",
            1_050_000,
            922_000,
            128_000,
            True,
            _OPENAI_LEVELS,
        ),
        (
            "qwen/qwen3.8-flash",
            1_000_000,
            983_616,
            131_072,
            True,
            _QWEN_ROUTER_LEVELS,
        ),
        (
            "x-ai/grok-4.6",
            500_000,
            None,
            450_000,
            True,
            _GROK_LEVELS,
        ),
        (
            "xiaomi/mimo-v2.5",
            1_048_576,
            None,
            131_072,
            True,
            _MIMO_LEVELS,
        ),
        (
            "z-ai/glm-5.3-flash",
            1_000_000,
            None,
            131_072,
            True,
            _KIMI_LEVELS,
        ),
    ],
)
def test_packaged_catalogue_contains_requested_openrouter_profiles(
    model: str,
    context_window: int,
    max_input: int | None,
    max_output: int,
    images: bool,
    levels: dict[str, str | None],
) -> None:
    matches = [
        entry
        for entry in catalog._BUILTIN_MODEL_ENTRIES
        if (
            entry.provider,
            entry.model,
            entry.base_url,
        )
        == ("openai", model, "https://openrouter.ai/api/v1")
    ]

    assert len(matches) == 1
    profile = matches[0].profile
    assert profile.context_window_tokens == context_window
    assert profile.max_input_tokens == max_input
    assert profile.max_output_tokens == max_output
    assert profile.supports_images is images
    assert profile.reasoning is not None
    assert profile.reasoning.format == "openrouter"
    assert profile.reasoning.levels.as_dict() == levels


def test_catalog_rejects_malformed_json_without_leaking_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="model catalog is not valid JSON") as exc_info:
        _load_text(
            '{"models":["should-not-leak",',
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
        )

    assert "should-not-leak" not in str(exc_info.value)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_catalog_rejects_nonstandard_json_constants(
    constant: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    text = f'{{"models":[{constant}]}}'

    with pytest.raises(RuntimeError, match="model catalog is not valid JSON"):
        _load_text(text, monkeypatch=monkeypatch, tmp_path=tmp_path)


def test_catalog_revision_changes_with_semantic_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _payload()
    original_revision, _ = _load_payload(payload, monkeypatch=monkeypatch, tmp_path=tmp_path)
    profile = payload["models"][0]["profile"]  # type: ignore[index]
    profile["max_output_tokens"] = 64  # type: ignore[index]

    changed_revision, _ = _load_payload(payload, monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert changed_revision != original_revision


@pytest.mark.parametrize(
    "endpoint",
    [
        "not-a-url",
        "ftp://should-not-leak.example/v1",
        "https://should-not-leak.example:not-a-port/v1?token=secret",
    ],
)
def test_catalog_rejects_malformed_nonempty_base_url_without_leaking_it(
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _payload([_valid_model(base_url=endpoint)])

    with pytest.raises(RuntimeError, match=r"models\[0\]\.base_url") as exc_info:
        _load_payload(payload, monkeypatch=monkeypatch, tmp_path=tmp_path)

    message = str(exc_info.value)
    assert "should-not-leak" not in message
    assert "secret" not in message


def test_catalog_rejects_string_false_boolean(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _valid_model()
    model["profile"]["supports_images"] = "false"  # type: ignore[index]

    with pytest.raises(RuntimeError, match=r"models\[0\]\.profile\.supports_images"):
        _load_payload(_payload([model]), monkeypatch=monkeypatch, tmp_path=tmp_path)


def test_catalog_rejects_string_integer_coercion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _valid_model()
    model["profile"]["max_output_tokens"] = "128"  # type: ignore[index]

    with pytest.raises(RuntimeError, match=r"models\[0\]\.profile\.max_output_tokens"):
        _load_payload(_payload([model]), monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_window_tokens", True),
        ("context_window_tokens", 1024.0),
        ("max_input_tokens", False),
        ("max_output_tokens", 128.0),
        ("supports_images", 0),
        ("reasoning", True),
    ],
)
def test_catalog_rejects_non_exact_profile_json_types(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _valid_model()
    model["profile"][field] = value  # type: ignore[index]

    with pytest.raises(RuntimeError, match=rf"models\[0\]\.profile\.{field}"):
        _load_payload(_payload([model]), monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        ([], "root"),
        ({"models": {}}, "root.models"),
        (_payload([None]), "models[0]"),
        (_payload([{"provider": "openai"}]), "models[0]"),
        (_payload([{**_valid_model(), "profile": []}]), "models[0].profile"),
    ],
)
def test_catalog_rejects_invalid_root_models_item_and_profile_shapes(
    payload: object,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match=re.escape(path)):
        _load_payload(payload, monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("level", "key", "change"),
    [
        ("root", "revision", "extra"),
        ("root", "models", "missing"),
        ("root", "unexpected", "extra"),
        ("model", "base_url", "missing"),
        ("model", "api_key", "extra"),
        ("profile", "max_input_tokens", "missing"),
        ("profile", "supports_images", "missing"),
        ("profile", "support_images", "extra"),
    ],
)
def test_catalog_rejects_missing_and_extra_keys(
    level: str,
    key: str,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _payload()
    model = payload["models"][0]  # type: ignore[index]
    containers: dict[str, dict[str, Any]] = {
        "root": payload,
        "model": model,
        "profile": model["profile"],
    }
    container = containers[level]
    if change == "missing":
        container.pop(key)
    else:
        container[key] = "unexpected"
    path = {"root": "root", "model": "models[0]", "profile": "models[0].profile"}[level]

    with pytest.raises(RuntimeError, match=re.escape(f"{path}.{key}")):
        _load_payload(payload, monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", 1),
        ("provider", ""),
        ("provider", " OpenAI "),
        ("provider", "OpenAI"),
        ("model", 1),
        ("model", ""),
        ("model", " test-model "),
        ("base_url", 443),
        ("base_url", ""),
    ],
)
def test_catalog_rejects_noncanonical_identity_fields(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _valid_model()
    model[field] = value

    with pytest.raises(RuntimeError, match=rf"models\[0\]\.{field}"):
        _load_payload(_payload([model]), monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("first_url", "second_url"),
    [
        ("HTTPS://API.EXAMPLE.TEST:443/v1/", "https://api.example.test/v1"),
        (None, None),
    ],
)
def test_catalog_rejects_duplicate_normalized_fingerprints_with_both_indices(
    first_url: str | None,
    second_url: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = [_valid_model(base_url=first_url), _valid_model(base_url=second_url)]

    with pytest.raises(RuntimeError) as exc_info:
        _load_payload(_payload(models), monkeypatch=monkeypatch, tmp_path=tmp_path)

    message = str(exc_info.value)
    assert "models[0]" in message
    assert "models[1]" in message


@pytest.mark.parametrize(
    ("level", "needle", "replacement", "path"),
    [
        ("root", '"models":', '"models":[],"models":', "root.models"),
        (
            "model",
            '"provider":"openai"',
            '"provider":"openai","provider":"openai"',
            "models[0].provider",
        ),
        (
            "profile",
            '"supports_images":false',
            '"supports_images":false,"supports_images":false',
            "models[0].profile.supports_images",
        ),
    ],
)
def test_catalog_rejects_duplicate_json_members(
    level: str,
    needle: str,
    replacement: str,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del level
    text = json.dumps(_payload(), separators=(",", ":"))
    duplicated = text.replace(needle, replacement, 1)
    assert duplicated != text

    with pytest.raises(RuntimeError, match=re.escape(path)):
        _load_text(duplicated, monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_window_tokens", 0),
        ("max_input_tokens", 2048),
        ("max_output_tokens", 0),
    ],
)
def test_catalog_reports_model_profile_numeric_invariants_at_the_field(
    field: str,
    value: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _valid_model()
    model["profile"][field] = value  # type: ignore[index]

    with pytest.raises(RuntimeError, match=rf"models\[0\]\.profile\.{field}"):
        _load_payload(_payload([model]), monkeypatch=monkeypatch, tmp_path=tmp_path)


def test_catalog_accepts_complete_profiles_with_null_and_valid_http_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = [
        _valid_model(model="default-endpoint", base_url=None),
        _valid_model(model="routed-endpoint", base_url="http://api.example.test:8080/v1"),
    ]
    payload = _payload(models)

    revision, parsed = _load_payload(payload, monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert re.fullmatch(r"sha256:[0-9a-f]{64}", revision)
    assert isinstance(parsed, tuple)
    assert parsed[0].fingerprint == ModelFingerprint(
        provider="openai", model="default-endpoint", endpoint_fingerprint=None
    )
    assert parsed[0].profile == ModelProfile(
        context_window_tokens=1024,
        max_output_tokens=128,
        supports_images=False,
        reasoning=ReasoningProfile(
            format="openai",
            levels=ReasoningLevels(
                off="none",
                minimal="minimal",
                low="low",
                medium="medium",
                high="high",
                xhigh=None,
                max=None,
            ),
        ),
    )
    assert parsed[1].fingerprint == ModelFingerprint(
        provider="openai",
        model="routed-endpoint",
        endpoint_fingerprint=normalized_endpoint_fingerprint("http://api.example.test:8080/v1"),
    )
