# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Strict built-in model catalogue plus an atomically replaceable runtime overlay."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from importlib.resources import files
from types import MappingProxyType
from typing import Never, cast

from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.fingerprints import ModelFingerprint, normalized_endpoint_fingerprint
from dlightrag.engine.ai.reasoning import (
    REASONING_LEVELS,
    ReasoningLevels,
    ReasoningProfile,
    best_effort_reasoning_profile,
)

_logger = logging.getLogger(__name__)

_ROOT_KEYS = frozenset({"models"})
_MODEL_KEYS = frozenset({"provider", "model", "base_url", "profile"})
_PROFILE_KEYS = frozenset(
    {
        "context_window_tokens",
        "max_input_tokens",
        "max_output_tokens",
        "supports_images",
        "reasoning",
    }
)
_REASONING_KEYS = frozenset({"format", "levels"})
_REASONING_LEVEL_KEYS = frozenset(REASONING_LEVELS)


class _JSONObject:
    """Object pairs retained until duplicate-member validation has paths."""

    def __init__(self, pairs: list[tuple[str, object]]) -> None:
        self.pairs = pairs


class UnknownModelProfileError(ValueError):
    """Raised by callers that require catalogue facts instead of fallback facts."""

    def __init__(self, fingerprint: ModelFingerprint) -> None:
        self.fingerprint = fingerprint
        endpoint = fingerprint.endpoint_fingerprint or "default"
        super().__init__(
            "No trusted model profile for "
            f"provider={fingerprint.provider!r}, model={fingerprint.model!r}, "
            f"endpoint={endpoint[:12]!r}; configure or publish an explicit catalogue entry"
        )


@dataclass(frozen=True, slots=True)
class CatalogueEntry:
    """One complete endpoint profile; runtime overlays never patch fields."""

    provider: str
    model: str
    base_url: str | None
    profile: ModelProfile
    fingerprint: ModelFingerprint

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "profile": model_profile_data(self.profile),
        }


@dataclass(frozen=True, slots=True)
class CatalogueSnapshot:
    """One immutable effective catalogue and its single content revision."""

    revision: str
    entries: tuple[CatalogueEntry, ...]
    profiles: Mapping[ModelFingerprint, ModelProfile]
    startup_fingerprints: frozenset[ModelFingerprint]
    overlay_fingerprints: frozenset[ModelFingerprint]

    def resolve(self, fingerprint: ModelFingerprint) -> ModelProfile | None:
        return self.profiles.get(fingerprint)


def _entry_order(entry: CatalogueEntry) -> tuple[str, str, str]:
    return (
        entry.provider,
        entry.model,
        entry.fingerprint.endpoint_fingerprint or "",
    )


def _unique_entries(
    entries: Sequence[CatalogueEntry],
    *,
    source: str,
) -> dict[ModelFingerprint, CatalogueEntry]:
    unique: dict[ModelFingerprint, CatalogueEntry] = {}
    for entry in entries:
        if entry.fingerprint in unique:
            raise ValueError(f"{source} contains a duplicate endpoint")
        unique[entry.fingerprint] = entry
    return unique


class ModelCatalogue:
    """Own the process's built-in, startup, and runtime catalogue snapshot."""

    def __init__(
        self,
        *,
        builtin_revision: str,
        builtin_entries: Sequence[CatalogueEntry],
    ) -> None:
        self._builtin_revision = builtin_revision
        self._builtin_entries = tuple(builtin_entries)
        self._builtin = MappingProxyType(
            {entry.fingerprint: entry for entry in self._builtin_entries}
        )
        self._startup: tuple[CatalogueEntry, ...] = ()
        self._overlay: tuple[CatalogueEntry, ...] = ()
        self._snapshot = self._merge(())
        if self._snapshot.revision != builtin_revision:
            raise RuntimeError("built-in model catalogue revision changed during loading")

    @property
    def builtin_revision(self) -> str:
        return self._builtin_revision

    @property
    def revision(self) -> str:
        return self._snapshot.revision

    @property
    def snapshot(self) -> CatalogueSnapshot:
        return self._snapshot

    @property
    def startup(self) -> tuple[CatalogueEntry, ...]:
        return self._startup

    @property
    def overlay(self) -> tuple[CatalogueEntry, ...]:
        return self._overlay

    def is_builtin(self, fingerprint: ModelFingerprint) -> bool:
        return fingerprint in self._builtin

    def preview(self, overlay: Sequence[CatalogueEntry]) -> CatalogueSnapshot:
        return self._merge(tuple(overlay))

    def replace_startup(self, startup: Sequence[CatalogueEntry]) -> CatalogueSnapshot:
        candidate = tuple(startup)
        previous = self._startup
        self._startup = candidate
        try:
            snapshot = self._merge(self._overlay)
        except Exception:
            self._startup = previous
            raise
        self._snapshot = snapshot
        return snapshot

    def replace_overlay(self, overlay: Sequence[CatalogueEntry]) -> CatalogueSnapshot:
        candidate = tuple(overlay)
        snapshot = self._merge(candidate)
        # One assignment publishes all effective facts and their revision.
        self._overlay = candidate
        self._snapshot = snapshot
        return snapshot

    def _merge(self, overlay: tuple[CatalogueEntry, ...]) -> CatalogueSnapshot:
        startup = _unique_entries(self._startup, source="startup model catalogue")
        overrides = _unique_entries(overlay, source="runtime model catalogue overlay")

        effective: list[CatalogueEntry] = []
        baseline_fingerprints: set[ModelFingerprint] = set()
        for builtin in self._builtin_entries:
            baseline_fingerprints.add(builtin.fingerprint)
            configured = startup.get(builtin.fingerprint, builtin)
            effective.append(overrides.get(builtin.fingerprint, configured))

        custom_startup = sorted(
            (
                entry
                for fingerprint, entry in startup.items()
                if fingerprint not in baseline_fingerprints
            ),
            key=_entry_order,
        )
        for entry in custom_startup:
            baseline_fingerprints.add(entry.fingerprint)
            effective.append(overrides.get(entry.fingerprint, entry))

        custom_overlay = sorted(
            (
                entry
                for fingerprint, entry in overrides.items()
                if fingerprint not in baseline_fingerprints
            ),
            key=_entry_order,
        )
        effective.extend(custom_overlay)
        models = [entry.as_dict() for entry in effective]
        profiles = MappingProxyType({entry.fingerprint: entry.profile for entry in effective})
        effective_revision = _model_catalog_revision(cast(list[object], models))
        revision = (
            effective_revision
            if not startup and not overlay
            else _model_catalog_revision(
                cast(
                    list[object],
                    [
                        {"effective_revision": effective_revision},
                        {"startup_revision": catalogue_overlay_revision(self._startup)},
                        {"overlay_revision": catalogue_overlay_revision(overlay)},
                    ],
                )
            )
        )
        return CatalogueSnapshot(
            revision=revision,
            entries=tuple(effective),
            profiles=profiles,
            startup_fingerprints=frozenset(startup),
            overlay_fingerprints=frozenset(overrides),
        )


def _reject_json_constant(_value: str) -> Never:
    raise ValueError("non-standard JSON constants are not allowed")


def _materialize_json(value: object, path: str) -> object:
    if isinstance(value, _JSONObject):
        materialized: dict[str, object] = {}
        for key, item in value.pairs:
            member_path = f"{path}.{key}"
            if key in materialized:
                raise RuntimeError(f"{member_path} is duplicated")
            materialized[key] = _materialize_json(item, member_path)
        return materialized
    if type(value) is list:
        return [
            _materialize_json(item, f"{path}[{index}]")
            for index, item in enumerate(cast(list[object], value))
        ]
    return value


def _decode_catalog_json(text: str) -> object:
    """Decode strict JSON without exposing source text in failures."""
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_JSONObject,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError, ValueError:
        raise RuntimeError("model catalog is not valid JSON") from None
    return _materialize_json(decoded, "root")


def _require_object(
    value: object,
    *,
    path: str,
    keys: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict:
        raise RuntimeError(f"{path} must be an object")
    obj = cast(dict[str, object], value)
    missing = sorted(keys - obj.keys())
    if missing:
        raise RuntimeError(f"{path}.{missing[0]} is missing")
    unknown = sorted(obj.keys() - keys)
    if unknown:
        raise RuntimeError(f"{path}.{unknown[0]} is not allowed")
    return obj


def _canonical_identity(value: object, *, path: str, lowercase: bool) -> str:
    if type(value) is not str:
        raise RuntimeError(f"{path} must be a string")
    identity = cast(str, value)
    if not identity or identity != identity.strip():
        raise RuntimeError(f"{path} must be a non-empty canonical string")
    if lowercase and identity != identity.lower():
        raise RuntimeError(f"{path} must be lowercase")
    return identity


def _profile_integer(
    profile: dict[str, object],
    field: str,
    *,
    path: str,
    optional: bool,
) -> int | None:
    value = profile[field]
    if optional and value is None:
        return None
    if type(value) is not int:
        suffix = " or null" if optional else ""
        raise RuntimeError(f"{path}.{field} must be an integer{suffix}")
    return cast(int, value)


def _profile_boolean(profile: dict[str, object], field: str, *, path: str) -> bool:
    value = profile[field]
    if type(value) is not bool:
        raise RuntimeError(f"{path}.{field} must be a boolean")
    return cast(bool, value)


def _profile_reasoning(value: object, *, path: str) -> ReasoningProfile | None:
    if value is None:
        return None
    reasoning = _require_object(value, path=path, keys=_REASONING_KEYS)
    format_name = _canonical_identity(
        reasoning["format"],
        path=f"{path}.format",
        lowercase=True,
    )
    levels = _require_object(
        reasoning["levels"],
        path=f"{path}.levels",
        keys=_REASONING_LEVEL_KEYS,
    )
    parsed: dict[str, str | None] = {}
    for level in REASONING_LEVELS:
        raw = levels[level]
        if raw is None:
            parsed[level] = None
        else:
            parsed[level] = _canonical_identity(
                raw,
                path=f"{path}.levels.{level}",
                lowercase=False,
            )
    try:
        return ReasoningProfile(
            format=format_name,
            levels=ReasoningLevels(**parsed),  # type: ignore[arg-type]
        )
    except ValueError as exc:
        raise RuntimeError(f"{path}: {exc}") from None


def _validated_profile(value: object, *, path: str) -> ModelProfile:
    facts = _require_object(value, path=path, keys=_PROFILE_KEYS)
    context_window_tokens = cast(
        int,
        _profile_integer(
            facts,
            "context_window_tokens",
            path=path,
            optional=False,
        ),
    )
    max_input_tokens = _profile_integer(
        facts,
        "max_input_tokens",
        path=path,
        optional=True,
    )
    max_output_tokens = _profile_integer(
        facts,
        "max_output_tokens",
        path=path,
        optional=True,
    )
    supports_images = _profile_boolean(facts, "supports_images", path=path)
    reasoning = _profile_reasoning(facts["reasoning"], path=f"{path}.reasoning")
    try:
        return ModelProfile(
            context_window_tokens=context_window_tokens,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            supports_images=supports_images,
            reasoning=reasoning,
        )
    except ValueError as exc:
        field = next(
            (
                candidate
                for candidate in (
                    "context_window_tokens",
                    "max_input_tokens",
                    "max_output_tokens",
                )
                if str(exc).startswith(candidate)
            ),
            "context_window_tokens",
        )
        raise RuntimeError(f"{path}.{field}: {exc}") from None


def model_profile_data(profile: ModelProfile) -> dict[str, object]:
    return {
        "context_window_tokens": profile.context_window_tokens,
        "max_input_tokens": profile.max_input_tokens,
        "max_output_tokens": profile.max_output_tokens,
        "supports_images": profile.supports_images,
        "reasoning": profile.reasoning.as_dict() if profile.reasoning is not None else None,
    }


def parse_catalogue_entry(value: object, *, path: str = "entry") -> CatalogueEntry:
    item = _require_object(value, path=path, keys=_MODEL_KEYS)
    provider = _canonical_identity(item["provider"], path=f"{path}.provider", lowercase=True)
    model = _canonical_identity(item["model"], path=f"{path}.model", lowercase=False)
    base_url = item["base_url"]
    if base_url is None:
        canonical_base_url = None
        endpoint_fingerprint = None
    else:
        if type(base_url) is not str or not base_url:
            raise RuntimeError(f"{path}.base_url must be null or a valid HTTP(S) URL")
        endpoint_fingerprint = normalized_endpoint_fingerprint(base_url)
        if endpoint_fingerprint is None:
            raise RuntimeError(f"{path}.base_url must be null or a valid HTTP(S) URL")
        canonical_base_url = cast(str, base_url)
    profile = _validated_profile(item["profile"], path=f"{path}.profile")
    fingerprint = ModelFingerprint(
        provider=provider,
        model=model,
        endpoint_fingerprint=endpoint_fingerprint,
    )
    return CatalogueEntry(
        provider=provider,
        model=model,
        base_url=canonical_base_url,
        profile=profile,
        fingerprint=fingerprint,
    )


def parse_catalogue_overlay(
    value: object,
    *,
    source: str = "runtime model catalogue overlay",
    path: str = "overlay",
) -> tuple[CatalogueEntry, ...]:
    if type(value) is not list:
        raise RuntimeError(f"{source} must be an array")
    entries: list[CatalogueEntry] = []
    seen: dict[ModelFingerprint, int] = {}
    for index, raw in enumerate(cast(list[object], value)):
        entry = parse_catalogue_entry(raw, path=f"{path}[{index}]")
        if entry.fingerprint in seen:
            raise RuntimeError(
                f"{path}[{index}] duplicates the normalized fingerprint from "
                f"{path}[{seen[entry.fingerprint]}]"
            )
        seen[entry.fingerprint] = index
        entries.append(entry)
    return tuple(entries)


def catalogue_overlay_data(entries: Sequence[CatalogueEntry]) -> list[dict[str, object]]:
    return [entry.as_dict() for entry in entries]


def catalogue_overlay_revision(entries: Sequence[CatalogueEntry]) -> str:
    """Return the built-in-independent CAS revision for one complete overlay."""
    return _model_catalog_revision(cast(list[object], catalogue_overlay_data(entries)))


def _model_catalog_revision(models: list[object]) -> str:
    """Hash canonical parsed model content, excluding top-level revision metadata."""
    canonical = json.dumps(
        models,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _parse_catalog(text: str) -> tuple[str, tuple[CatalogueEntry, ...]]:
    """Parse the built-in catalogue and derive its canonical content revision."""
    root = _require_object(_decode_catalog_json(text), path="root", keys=_ROOT_KEYS)
    models_value = root["models"]
    if type(models_value) is not list:
        raise RuntimeError("root.models must be an array")
    models = cast(list[object], models_value)

    entries: list[CatalogueEntry] = []
    fingerprint_indices: dict[ModelFingerprint, int] = {}
    for index, value in enumerate(models):
        entry = parse_catalogue_entry(value, path=f"models[{index}]")
        if entry.fingerprint in fingerprint_indices:
            first_index = fingerprint_indices[entry.fingerprint]
            raise RuntimeError(
                f"models[{index}] duplicates the normalized fingerprint from models[{first_index}]"
            )
        entries.append(entry)
        fingerprint_indices[entry.fingerprint] = index

    return _model_catalog_revision(models), tuple(entries)


def _load_catalog() -> tuple[str, tuple[CatalogueEntry, ...]]:
    text = files("dlightrag.engine.ai").joinpath("model_catalog.json").read_text("utf-8")
    return _parse_catalog(text)


MODEL_CATALOG_REVISION, _BUILTIN_MODEL_ENTRIES = _load_catalog()
MODEL_CATALOGUE = ModelCatalogue(
    builtin_revision=MODEL_CATALOG_REVISION,
    builtin_entries=_BUILTIN_MODEL_ENTRIES,
)

#: The unconditional capacity guess for endpoints the catalogue does not know.
#: It is deliberately generous: provider rejection remains explicit rather than
#: triggering endpoint probes or persistent adaptation. Reasoning is attached
#: separately as an unverified, protocol-derived request mapping.
FALLBACK_MODEL_PROFILE = ModelProfile(
    context_window_tokens=1_048_576,
    max_input_tokens=None,
    max_output_tokens=262_144,
    supports_images=True,
    reasoning=None,
)
_OPENROUTER_ENDPOINT = normalized_endpoint_fingerprint("https://openrouter.ai/api/v1")
_DEEPSEEK_ENDPOINTS = frozenset(
    {
        normalized_endpoint_fingerprint("https://api.deepseek.com"),
        normalized_endpoint_fingerprint("https://api.deepseek.com/v1"),
    }
)


def _fallback_reasoning_format(fingerprint: ModelFingerprint) -> str:
    if fingerprint.provider == "anthropic":
        return "anthropic"
    if fingerprint.provider == "gemini":
        return "gemini"
    if fingerprint.endpoint_fingerprint == _OPENROUTER_ENDPOINT:
        return "openrouter"
    if fingerprint.endpoint_fingerprint in _DEEPSEEK_ENDPOINTS:
        return "deepseek"
    return "openai"


def fallback_model_profile(fingerprint: ModelFingerprint) -> ModelProfile:
    """Attach protocol-derived, unverified reasoning controls to fallback capacity."""
    return replace(
        FALLBACK_MODEL_PROFILE,
        reasoning=best_effort_reasoning_profile(_fallback_reasoning_format(fingerprint)),
    )


def current_model_catalog_revision() -> str:
    return MODEL_CATALOGUE.revision


def resolve_model_profile(fingerprint: ModelFingerprint) -> ModelProfile:
    """Resolve runtime overlay, built-in catalogue, then permissive fallback."""
    profile = MODEL_CATALOGUE.snapshot.resolve(fingerprint)
    if profile is not None:
        return profile
    _logger.warning(
        "uncatalogued model resolved to the fallback profile: provider=%r model=%r endpoint=%r",
        fingerprint.provider,
        fingerprint.model,
        fingerprint.endpoint_fingerprint,
    )
    return fallback_model_profile(fingerprint)


__all__ = [
    "CatalogueEntry",
    "CatalogueSnapshot",
    "FALLBACK_MODEL_PROFILE",
    "MODEL_CATALOGUE",
    "MODEL_CATALOG_REVISION",
    "ModelCatalogue",
    "UnknownModelProfileError",
    "catalogue_overlay_data",
    "catalogue_overlay_revision",
    "current_model_catalog_revision",
    "fallback_model_profile",
    "model_profile_data",
    "parse_catalogue_entry",
    "parse_catalogue_overlay",
    "resolve_model_profile",
]
