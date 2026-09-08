# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-owned model capability coordination and image policy."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.settings import MODEL_ROLE_NAMES, ModelRole, ModelSettings
from dlightrag.engine.ai.vision import (
    ImageCapabilityStatus,
    ImageProbeOutcome,
    ModelImageCapabilities,
)
from dlightrag.engine.answer.image_capability import (
    AnswerImageCapability,
    answer_image_capability_summary,
    derive_effective_max_images,
)
from dlightrag.engine.answer.images import AnswerImagePolicy


@dataclass(frozen=True, slots=True)
class AnswerImagePolicySettings:
    max_images: int
    max_total_bytes: int
    max_bytes_per_image: int
    max_pixels: int
    max_px: int
    min_px: int
    quality: int
    min_quality: int


@dataclass(frozen=True, slots=True)
class AnswerCapabilitySettings:
    images: AnswerImagePolicySettings
    web_search_enabled: bool
    rerank_enabled: bool
    rerank_strategy: str


@dataclass(frozen=True, slots=True)
class RequestModelContext:
    extract: ModelProfile
    query: ModelProfile
    vlm: ModelProfile


@dataclass(frozen=True, slots=True)
class AnswerCapabilities:
    """Immutable public snapshot of Answer and VLM image capability."""

    answer: AnswerImageCapability | None
    vlm_status: ImageCapabilityStatus


class AnswerCapabilityView:
    """Read-only capability view for callers outside Answer composition."""

    def __init__(self, coordinator: AnswerCapabilityCoordinator) -> None:
        self._coordinator = coordinator

    async def read(self) -> AnswerCapabilities:
        """Return an immutable snapshot after a cooldown-governed refresh."""
        return await self._coordinator.refresh_answer()


class AnswerCapabilityCoordinator:
    """Own role profiles, image probes, and image transport policy."""

    def __init__(
        self,
        *,
        settings: AnswerCapabilitySettings,
        profile_for_role: Callable[[ModelRole], ModelProfile],
        model_settings_for_role: Callable[[ModelRole], ModelSettings],
        rerank_model_settings: Callable[[], ModelSettings],
        image_capabilities: ModelImageCapabilities,
        on_answer_capability: Callable[[dict[str, object]], None],
    ) -> None:
        self._settings = settings
        self._profile_for_role = profile_for_role
        self._model_settings_for_role = model_settings_for_role
        self._rerank_model_settings = rerank_model_settings
        self._image_capabilities = image_capabilities
        self._on_answer_capability = on_answer_capability
        self._catalogue_profiles: dict[ModelRole, ModelProfile] = {}
        self._profiles: dict[ModelRole, ModelProfile] = {}
        self._answer_image_capability: AnswerImageCapability | None = None
        self._vlm_image_status: ImageCapabilityStatus = "unknown"
        self._rerank_supports_vision: bool | None = None
        self._catalogue_generation = 0

    @property
    def snapshot(self) -> AnswerCapabilities:
        return AnswerCapabilities(
            answer=self._answer_image_capability,
            vlm_status=self._vlm_image_status,
        )

    @property
    def answer_image_capability(self) -> AnswerImageCapability | None:
        return self._answer_image_capability

    @property
    def rerank_supports_vision(self) -> bool | None:
        return self._rerank_supports_vision

    def resolve_profiles(self) -> None:
        self._catalogue_profiles = {role: self._profile_for_role(role) for role in MODEL_ROLE_NAMES}
        self._profiles = dict(self._catalogue_profiles)

    def invalidate_model_catalogue(self) -> None:
        """Drop profile and probe caches after one atomic catalogue publication."""
        self._catalogue_generation += 1
        self._catalogue_profiles.clear()
        self._profiles.clear()
        self._answer_image_capability = None
        self._vlm_image_status = "unknown"
        self._rerank_supports_vision = None
        self._image_capabilities.clear()

    def catalogue_profile(self, role: ModelRole) -> ModelProfile:
        profile = self._catalogue_profiles.get(role)
        if profile is None:
            profile = self._profile_for_role(role)
            self._catalogue_profiles[role] = profile
            self._profiles.setdefault(role, profile)
        return profile

    def model_profile(self, role: ModelRole) -> ModelProfile:
        profile = self._profiles.get(role)
        if profile is None:
            profile = self.catalogue_profile(role)
            self._profiles[role] = profile
        return profile

    def current_profiles(self) -> dict[ModelRole, ModelProfile]:
        return {role: self.model_profile(role) for role in MODEL_ROLE_NAMES}

    def request_model_context(
        self,
        pinned: Mapping[ModelRole, ModelProfile] | None,
    ) -> RequestModelContext:
        if pinned is not None:
            return RequestModelContext(
                extract=pinned["extract"],
                query=pinned["query"],
                vlm=pinned["vlm"],
            )
        return RequestModelContext(
            extract=self.model_profile("extract"),
            query=self.model_profile("query"),
            vlm=self.model_profile("vlm"),
        )

    def narrow_role_image_profile(
        self,
        role: ModelRole,
        status: ImageCapabilityStatus,
    ) -> None:
        declared = self.catalogue_profile(role)
        self._profiles[role] = replace(
            declared,
            supports_images=declared.supports_images and status == "supported",
        )

    async def probe_all(self) -> None:
        await self.probe_answer()
        await self.probe_vlm()
        await self.probe_rerank()

    async def probe_answer(self) -> None:
        if self._answer_image_capability is not None:
            return
        self._cache_answer_capability(await self._discover_current_answer_capability())

    async def refresh_answer(self) -> AnswerCapabilities:
        capability = self._answer_image_capability
        if capability is None or capability.status == "unknown":
            self._cache_answer_capability(await self._discover_current_answer_capability())
        return self.snapshot

    async def _discover_current_answer_capability(self) -> AnswerImageCapability:
        while True:
            generation = self._catalogue_generation
            capability = await self._discover_answer_capability()
            if generation == self._catalogue_generation:
                return capability

    def _cache_answer_capability(self, capability: AnswerImageCapability) -> None:
        self._answer_image_capability = capability
        self._on_answer_capability(answer_image_capability_summary(capability))
        self.narrow_role_image_profile("query", capability.status)

    async def confirmed_live_answer_context(
        self,
        _models: RequestModelContext,
    ) -> tuple[RequestModelContext, AnswerImageCapability | None]:
        await self.refresh_answer()
        return self.request_model_context(None), self._answer_image_capability

    async def pinned_answer_context(
        self,
        models: RequestModelContext,
    ) -> tuple[RequestModelContext, AnswerImageCapability]:
        return models, self.answer_capability_from_profile(models.query)

    async def _discover_answer_capability(self) -> AnswerImageCapability:
        ceiling = self._settings.images.max_images
        model_settings = self._model_settings_for_role("query")
        if ceiling <= 0:
            outcome = ImageProbeOutcome(status="unsupported", failure_kind="config_disabled")
        elif not self.catalogue_profile("query").supports_images:
            outcome = ImageProbeOutcome(
                status="unsupported",
                failure_kind="profile_declared_unsupported",
            )
        else:
            outcome = await self._image_capabilities.resolve(model_settings)
        return AnswerImageCapability(
            status=outcome.status,
            configured_ceiling=ceiling,
            effective_max_images=derive_effective_max_images(outcome.status, ceiling),
            provider=model_settings.provider,
            base_url=model_settings.base_url,
            model=model_settings.model,
            failure_kind=outcome.failure_kind,
        )

    async def probe_vlm(self) -> None:
        while True:
            generation = self._catalogue_generation
            if self._settings.images.max_images <= 0:
                status: ImageCapabilityStatus = "unsupported"
            elif not self.catalogue_profile("vlm").supports_images:
                status = "unsupported"
            else:
                outcome = await self._image_capabilities.resolve(
                    self._model_settings_for_role("vlm")
                )
                status = outcome.status
            if generation == self._catalogue_generation:
                self._vlm_image_status = status
                self.narrow_role_image_profile("vlm", status)
                return

    async def refresh_vlm(self) -> AnswerCapabilities:
        if self._vlm_image_status == "unknown":
            await self.probe_vlm()
        return self.snapshot

    async def probe_rerank(self) -> None:
        if self._rerank_supports_vision is not None:
            return
        if not (
            self._settings.rerank_enabled and self._settings.rerank_strategy == "chat_llm_reranker"
        ):
            return
        while True:
            generation = self._catalogue_generation
            outcome = await self._image_capabilities.resolve(self._rerank_model_settings())
            if generation == self._catalogue_generation:
                self._rerank_supports_vision = {
                    "supported": True,
                    "unsupported": False,
                }.get(outcome.status)
                return

    def answer_image_policy(self, profile: ModelProfile) -> AnswerImagePolicy:
        return self._image_policy(
            self._settings.images.max_images if profile.supports_images else 0
        )

    def answer_capability_from_profile(self, profile: ModelProfile) -> AnswerImageCapability:
        model_settings = self._model_settings_for_role("query")
        ceiling = self._settings.images.max_images
        status: ImageCapabilityStatus = "supported" if profile.supports_images else "unsupported"
        return AnswerImageCapability(
            status=status,
            configured_ceiling=ceiling,
            effective_max_images=derive_effective_max_images(status, ceiling),
            provider=model_settings.provider,
            base_url=model_settings.base_url,
            model=model_settings.model,
            failure_kind=None if profile.supports_images else "pinned_profile_unsupported",
        )

    def vlm_image_policy(self, profile: ModelProfile) -> AnswerImagePolicy:
        return self._image_policy(
            self._settings.images.max_images if profile.supports_images else 0
        )

    def _image_policy(self, max_images: int) -> AnswerImagePolicy:
        images = self._settings.images
        return AnswerImagePolicy(
            max_images=max_images,
            max_total_bytes=images.max_total_bytes,
            max_bytes_per_image=images.max_bytes_per_image,
            max_pixels=images.max_pixels,
            max_px=images.max_px,
            min_px=images.min_px,
            quality=images.quality,
            min_quality=images.min_quality,
        )


__all__ = [
    "AnswerCapabilities",
    "AnswerCapabilityCoordinator",
    "AnswerCapabilitySettings",
    "AnswerCapabilityView",
    "AnswerImagePolicySettings",
    "RequestModelContext",
]
