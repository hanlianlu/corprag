# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical immutable settings consumed by AI model factories."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.contracts import ChatProvider, InputModality
from dlightrag.engine.ai.reasoning import (
    ReasoningLevel,
    ReasoningLevels,
    ReasoningProfile,
    conflicting_reasoning_keys,
)

type ModelRole = Literal["extract", "keyword", "query", "vlm"]
MODEL_ROLE_NAMES: tuple[ModelRole, ...] = ("extract", "keyword", "query", "vlm")
type RerankStrategy = Literal[
    "chat_llm_reranker",
    "jina_reranker",
    "aliyun_reranker",
    "local_reranker",
    "voyage_reranker",
    "cohere_reranker",
    "azure_cohere",
]


def freeze_settings_value(value: Any) -> Any:
    """Recursively freeze collection-valued settings."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): freeze_settings_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(freeze_settings_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(freeze_settings_value(item) for item in value)
    return value


def thaw_settings_value(value: Any) -> Any:
    """Return ordinary provider/serializer-safe containers."""
    if isinstance(value, Mapping):
        return {str(key): thaw_settings_value(item) for key, item in value.items()}
    if isinstance(value, tuple | frozenset):
        return [thaw_settings_value(item) for item in value]
    return value


def _canonical_provider(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


class FrozenSettings(BaseModel):
    """Strict, frozen base for all canonical settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class ModelSettings(FrozenSettings):
    """One chat-model endpoint and its provider options."""

    provider: ChatProvider = "openai"
    model: str
    api_key: str | None = None
    base_url: str | None = None
    structured_output: Literal["auto", "json_schema", "json_object"] = "auto"
    temperature: float | None = Field(default=None, ge=0)
    timeout: float = Field(default=240.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    reasoning: ReasoningLevel | None = None
    agentic_reasoning: ReasoningLevel | None = None
    model_kwargs: Mapping[str, Any] = Field(default_factory=dict)
    agentic_model_kwargs: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def _fold_provider(cls, value: Any) -> Any:
        return _canonical_provider(value)

    @field_validator("model_kwargs", "agentic_model_kwargs", mode="after")
    @classmethod
    def _freeze_mapping(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_settings_value(value)

    @model_validator(mode="after")
    def _validate_model_options(self) -> Self:
        if self.provider == "anthropic" and self.structured_output == "json_object":
            raise ValueError("Anthropic native structured output requires json_schema")
        if self.reasoning is not None:
            conflicts = conflicting_reasoning_keys(self.model_kwargs)
            if conflicts:
                raise ValueError(
                    "reasoning conflicts with model_kwargs fields: " + ", ".join(conflicts)
                )
        if self.effective_agentic_reasoning is not None:
            conflicts = conflicting_reasoning_keys(
                {**self.model_kwargs, **self.agentic_model_kwargs}
            )
            if conflicts:
                raise ValueError(
                    "agentic_reasoning conflicts with agentic model kwargs fields: "
                    + ", ".join(conflicts)
                )
        return self

    @property
    def effective_agentic_reasoning(self) -> ReasoningLevel | None:
        """Inherit ordinary reasoning only when the agentic field was omitted."""
        if "agentic_reasoning" in self.model_fields_set:
            return self.agentic_reasoning
        return self.reasoning

    @property
    def has_explicit_auth(self) -> bool:
        if "api_key" not in self.model_fields_set:
            return False
        return self.api_key is None or bool(self.api_key.strip())

    @field_serializer("model_kwargs", "agentic_model_kwargs")
    def _serialize_mappings(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_settings_value(value)

    def model_kwargs_copy(self) -> dict[str, Any]:
        return thaw_settings_value(self.model_kwargs)

    def agentic_model_kwargs_copy(self) -> dict[str, Any]:
        return {
            **self.model_kwargs_copy(),
            **thaw_settings_value(self.agentic_model_kwargs),
        }


class ModelCatalogueReasoningSettings(FrozenSettings):
    """Verified reasoning facts for one startup-configured endpoint."""

    format: str
    levels: ReasoningLevels

    @model_validator(mode="after")
    def _validate_format(self) -> Self:
        self.as_reasoning_profile()
        return self

    def as_reasoning_profile(self) -> ReasoningProfile:
        return ReasoningProfile(format=self.format, levels=self.levels)


class ModelCatalogueProfileSettings(FrozenSettings):
    """One complete startup-configured endpoint capability profile."""

    context_window_tokens: int = Field(gt=0)
    max_input_tokens: int | None
    max_output_tokens: int | None
    supports_images: bool
    reasoning: ModelCatalogueReasoningSettings | None

    @model_validator(mode="after")
    def _validate_capacity(self) -> Self:
        self.as_model_profile()
        return self

    def as_model_profile(self) -> ModelProfile:
        return ModelProfile(
            context_window_tokens=self.context_window_tokens,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            supports_images=self.supports_images,
            reasoning=(
                self.reasoning.as_reasoning_profile() if self.reasoning is not None else None
            ),
        )


class ModelCatalogueEntrySettings(FrozenSettings):
    """One complete static model catalogue entry from startup configuration."""

    provider: ChatProvider
    model: str
    base_url: str | None
    profile: ModelCatalogueProfileSettings

    @field_validator("provider", mode="before")
    @classmethod
    def _fold_provider(cls, value: Any) -> Any:
        return _canonical_provider(value)

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("model must be a non-empty canonical string")
        return value

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value != value.strip():
            raise ValueError("base_url must be null or a valid HTTP(S) URL")
        try:
            parsed = urlsplit(value)
            _ = parsed.port
        except ValueError:
            raise ValueError("base_url must be null or a valid HTTP(S) URL") from None
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be null or a valid HTTP(S) URL")
        return value

    def as_catalogue_data(self) -> dict[str, object]:
        profile = self.profile.as_model_profile()
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "profile": {
                "context_window_tokens": profile.context_window_tokens,
                "max_input_tokens": profile.max_input_tokens,
                "max_output_tokens": profile.max_output_tokens,
                "supports_images": profile.supports_images,
                "reasoning": (
                    profile.reasoning.as_dict() if profile.reasoning is not None else None
                ),
            },
        }


class ModelRoleOverrides(FrozenSettings):
    extract: ModelSettings | None = None
    keyword: ModelSettings | None = None
    query: ModelSettings | None = None
    vlm: ModelSettings | None = None


class ModelRoleSettings(FrozenSettings):
    """Default chat model plus complete role overrides."""

    default: ModelSettings = Field(
        default_factory=lambda: ModelSettings(
            provider="openai",
            model="google/gemini-3.8-flash",
            base_url="https://openrouter.ai/api/v1",
            temperature=1.0,
        )
    )
    roles: ModelRoleOverrides = Field(default_factory=ModelRoleOverrides)

    @model_validator(mode="before")
    @classmethod
    def _merge_partial_default(cls, value: Any) -> Any:
        """Let secret-only env input overlay the shipped default endpoint."""
        if not isinstance(value, Mapping) or not isinstance(value.get("default"), Mapping):
            return value
        payload = dict(value)
        shipped = cls.model_fields["default"].get_default(call_default_factory=True)
        if not isinstance(shipped, ModelSettings):
            raise TypeError("default model factory did not return ModelSettings")
        supplied = dict(value["default"])
        merged = {name: getattr(shipped, name) for name in ModelSettings.model_fields}
        merged.update(supplied)
        settings = ModelSettings.model_validate(merged)
        object.__setattr__(settings, "__pydantic_fields_set__", set(supplied))
        payload["default"] = settings
        return payload

    @property
    def overrides(self) -> Mapping[ModelRole, ModelSettings]:
        return MappingProxyType(
            {
                role: candidate
                for role in MODEL_ROLE_NAMES
                if (candidate := getattr(self.roles, role)) is not None
                and candidate.has_explicit_auth
            }
        )

    def resolve(self, role: ModelRole) -> ModelSettings:
        return self.overrides.get(role, self.default)


class EmbeddingSettings(FrozenSettings):
    provider: Literal[
        "azure_cohere",
        "cohere",
        "gemini",
        "jina",
        "openai",
        "openai_compatible",
        "voyage",
    ] = "voyage"
    model: str = "voyage-multimodal-3.5"
    api_key: str | None = None
    base_url: str | None = None
    dim: int = Field(default=1024, ge=1)
    max_token_size: int = Field(default=8192, ge=1)
    input_modality: InputModality = "auto"
    startup_probe: bool = True
    timeout: float = Field(default=120.0, gt=0)
    max_concurrency: int = Field(default=16, ge=1)
    batch_size: int = Field(default=64, ge=1)


class RerankSettings(FrozenSettings):
    enabled: bool = True
    strategy: RerankStrategy = "chat_llm_reranker"
    provider: ChatProvider | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    input_modality: InputModality = "auto"
    score_threshold: float | None = Field(default=None, ge=0)
    max_concurrency: int = Field(default=8, ge=1)
    batch_size: int = Field(default=8, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    model_kwargs: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def _fold_provider(cls, value: Any) -> Any:
        return _canonical_provider(value)

    @field_validator("model_kwargs", mode="after")
    @classmethod
    def _freeze_mapping(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_settings_value(value)

    @field_serializer("model_kwargs")
    def _serialize_model_kwargs(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_settings_value(value)

    @property
    def has_explicit_auth(self) -> bool:
        if "api_key" not in self.model_fields_set:
            return False
        return self.api_key is None or bool(self.api_key.strip())

    def scoring_model(self, default: ModelSettings) -> ModelSettings:
        """Resolve the independent chat reranker or reuse the default model."""
        if not (self.provider and self.model and self.has_explicit_auth):
            return default
        return ModelSettings(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature or 0.0,
            model_kwargs=self.model_kwargs,
        )


class ModelsSettings(FrozenSettings):
    catalogue: tuple[ModelCatalogueEntrySettings, ...] = ()
    chat: ModelRoleSettings = Field(default_factory=ModelRoleSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    max_concurrency: int = Field(default=16, ge=1)

    def catalogue_data(self) -> list[dict[str, object]]:
        return [entry.as_catalogue_data() for entry in self.catalogue]


__all__ = [
    "EmbeddingSettings",
    "FrozenSettings",
    "MODEL_ROLE_NAMES",
    "ModelCatalogueEntrySettings",
    "ModelCatalogueProfileSettings",
    "ModelCatalogueReasoningSettings",
    "ModelRole",
    "ModelRoleOverrides",
    "ModelRoleSettings",
    "ModelSettings",
    "ModelsSettings",
    "RerankSettings",
    "RerankStrategy",
    "freeze_settings_value",
    "thaw_settings_value",
]
