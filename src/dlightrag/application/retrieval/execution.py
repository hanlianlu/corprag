# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable top-level Retrieval execution behind the common Run executor seam."""

from __future__ import annotations

import asyncio
import datetime
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.reasoning import REASONING_LEVELS, ReasoningLevels, ReasoningProfile
from dlightrag.engine.dependencies import (
    DependencyComponent,
    classify_transient_dependency,
    dependency_component_from_checkpoint,
    next_dependency_retry,
)
from dlightrag.engine.rag.retrieval import MetadataFilter, RetrievalOptions, RetrievalResult
from dlightrag.engine.runtime.coordinator import (
    LeaseLostError,
    RunCancellationObserved,
    RunSession,
)
from dlightrag.engine.runtime.errors import IncompatibleActiveRunError, RunExecutionError
from dlightrag.engine.runtime.records import (
    Deferred,
    Failed,
    Succeeded,
)

_RETRIEVAL_TIMEOUT_KIND = "retrieval_timeout"
_RETRIEVAL_TIMEOUT_MESSAGE = "Retrieval timed out before it completed."
_RETRIEVAL_FAILURE_KIND = "retrieval_failed"
_RETRIEVAL_FAILURE_MESSAGE = "Retrieval failed."
_RETRIEVAL_MODEL_CHANGED_KIND = "retrieval_model_changed"
_RETRIEVAL_MODEL_CHANGED_MESSAGE = (
    "The model configuration needed by this accepted retrieval changed before execution."
)
_DEFER_BASE_SECONDS = 5
_DEFER_MAX_SECONDS = 60


type DependencyStateCallback = Callable[[DependencyComponent], None]


@dataclass(frozen=True, slots=True)
class PinnedRetrievalModel:
    """One model identity and capacity snapshot required to recover Retrieval."""

    role: str
    fingerprint: ModelFingerprint
    profile: ModelProfile

    def as_json(self) -> dict[str, Any]:
        reasoning = self.profile.reasoning
        return {
            "role": self.role,
            "fingerprint": {
                "provider": self.fingerprint.provider,
                "model": self.fingerprint.model,
                "endpoint_fingerprint": self.fingerprint.endpoint_fingerprint,
            },
            "profile": {
                "context_window_tokens": self.profile.context_window_tokens,
                "max_input_tokens": self.profile.max_input_tokens,
                "max_output_tokens": self.profile.max_output_tokens,
                "supports_images": self.profile.supports_images,
                "reasoning": (
                    {**reasoning.as_dict(), "best_effort": reasoning.best_effort}
                    if reasoning is not None
                    else None
                ),
            },
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> PinnedRetrievalModel:
        fingerprint = value.get("fingerprint")
        profile = value.get("profile")
        role = str(value.get("role") or "")
        if not isinstance(fingerprint, Mapping) or not isinstance(profile, Mapping):
            raise ValueError("pinned retrieval model requires fingerprint and profile objects")
        if (
            not role
            or not str(fingerprint.get("provider") or "")
            or not str(fingerprint.get("model") or "")
        ):
            raise ValueError("pinned retrieval model requires role, provider, and model")
        if "reasoning" not in profile:
            raise ValueError("pinned retrieval model requires explicit reasoning facts")
        return cls(
            role=role,
            fingerprint=ModelFingerprint(
                provider=str(fingerprint.get("provider") or ""),
                model=str(fingerprint.get("model") or ""),
                endpoint_fingerprint=(
                    str(fingerprint["endpoint_fingerprint"])
                    if fingerprint.get("endpoint_fingerprint") is not None
                    else None
                ),
            ),
            profile=ModelProfile(
                context_window_tokens=int(profile["context_window_tokens"]),
                max_input_tokens=_optional_int(profile.get("max_input_tokens")),
                max_output_tokens=_optional_int(profile.get("max_output_tokens")),
                supports_images=bool(profile.get("supports_images")),
                reasoning=_reasoning_profile(profile.get("reasoning")),
            ),
        )


@dataclass(frozen=True, slots=True)
class RetrievalRunInput:
    """Normalized immutable input that every claim and reclaim executes."""

    query: str
    workspaces: tuple[str, ...]
    retrieval: RetrievalOptions
    bm25_query: str | None
    filters: Mapping[str, Any] | None
    query_images: tuple[Mapping[str, Any], ...]
    pinned_models: tuple[PinnedRetrievalModel, ...]
    context_policy_revision: str
    model_catalog_revision: str
    idempotency_fingerprint: str

    def as_request(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "workspaces": list(self.workspaces),
            "top_k": self.retrieval.top_k,
            "chunk_top_k": self.retrieval.chunk_top_k,
            "federated_rerank": self.retrieval.federated_rerank,
            "bm25_query": self.bm25_query,
            "filters": dict(self.filters) if self.filters is not None else None,
            "query_images": [dict(image) for image in self.query_images],
            "pinned_models": [model.as_json() for model in self.pinned_models],
            "context_policy_revision": self.context_policy_revision,
            "model_catalog_revision": self.model_catalog_revision,
            "idempotency_fingerprint": self.idempotency_fingerprint,
        }

    @classmethod
    def from_prepared_input(cls, prepared: Mapping[str, Any] | None) -> RetrievalRunInput:
        if prepared is None:
            raise RunExecutionError(
                "retrieval_input_missing",
                "Retrieval has no accepted input to execute.",
            )
        filters = prepared.get("filters")
        raw_pins = prepared.get("pinned_models")
        raw_workspaces = prepared.get("workspaces")
        raw_images = prepared.get("query_images")
        try:
            if not isinstance(raw_pins, list) or not all(
                isinstance(item, Mapping) for item in raw_pins
            ):
                raise ValueError("pinned_models must be an array of objects")
            if (
                not isinstance(raw_workspaces, list)
                or not raw_workspaces
                or not all(isinstance(value, str) and value for value in raw_workspaces)
            ):
                raise ValueError("workspaces must be a non-empty string array")
            if not isinstance(raw_images, list) or not all(
                isinstance(image, Mapping) for image in raw_images
            ):
                raise ValueError("query_images must be an array of objects")
            if not isinstance(prepared.get("query"), str) or not str(prepared["query"]).strip():
                raise ValueError("query must be non-empty")
            pins = tuple(PinnedRetrievalModel.from_json(item) for item in raw_pins)
            fingerprint = str(prepared.get("idempotency_fingerprint") or "")
            revision = str(prepared.get("context_policy_revision") or "")
            catalog_revision = str(prepared.get("model_catalog_revision") or "")
            if not pins or not fingerprint or not revision or not catalog_revision:
                raise ValueError("required recovery facts are missing")
        except (KeyError, TypeError, ValueError) as exc:
            raise RunExecutionError(
                "retrieval_input_incompatible",
                "Retrieval accepted input is incompatible with this deployment.",
            ) from exc
        return cls(
            query=str(prepared["query"]),
            workspaces=tuple(raw_workspaces),
            retrieval=RetrievalOptions(
                top_k=_optional_int(prepared.get("top_k")),
                chunk_top_k=_optional_int(prepared.get("chunk_top_k")),
                federated_rerank=bool(prepared.get("federated_rerank")),
            ),
            bm25_query=(
                str(prepared["bm25_query"]) if prepared.get("bm25_query") is not None else None
            ),
            filters=dict(filters) if isinstance(filters, Mapping) else None,
            query_images=tuple(dict(image) for image in raw_images),
            pinned_models=pins,
            context_policy_revision=revision,
            model_catalog_revision=catalog_revision,
            idempotency_fingerprint=fingerprint,
        )

    def model_profile(self, role: str) -> ModelProfile | None:
        return next((model.profile for model in self.pinned_models if model.role == role), None)


def validate_active_retrieval_input(
    prepared: Mapping[str, Any],
    *,
    model_fingerprint_for_role: Callable[[str], ModelFingerprint],
) -> None:
    """Require one active Retrieval input to remain executable by this deployment."""
    try:
        run_input = RetrievalRunInput.from_prepared_input(prepared)
    except (AttributeError, KeyError, TypeError, ValueError, RunExecutionError) as exc:
        raise IncompatibleActiveRunError(
            "active retrieval runs use an incompatible durable input schema; "
            "drain or owner-cancel them before deployment"
        ) from exc
    expected_roles = {"extract"}
    if run_input.query_images:
        expected_roles.add("vlm")
    pinned = {item.role: item for item in run_input.pinned_models}
    if len(run_input.pinned_models) != len(expected_roles) or set(pinned) != expected_roles:
        raise IncompatibleActiveRunError(
            "active retrieval runs do not contain the required model role set; "
            "drain or owner-cancel them before deployment"
        )
    if run_input.context_policy_revision != CONTEXT_POLICY_REVISION:
        raise IncompatibleActiveRunError(
            "active retrieval runs use another context policy revision; "
            "drain or owner-cancel them before deployment"
        )
    if run_input.model_catalog_revision != current_model_catalog_revision():
        raise IncompatibleActiveRunError(
            "active retrieval runs use another model catalog revision; "
            "drain or owner-cancel them before deployment"
        )
    if any(pinned[role].fingerprint != model_fingerprint_for_role(role) for role in expected_roles):
        raise IncompatibleActiveRunError(
            "active retrieval runs target another model endpoint configuration; "
            "drain or owner-cancel them before deployment"
        )


class RetrievalOperation(Protocol):
    """The one deep Retrieval service instance used by top-level and Answer work."""

    def warm(self, workspaces: Sequence[str]) -> None: ...

    async def prepare_query_images(self, images: Sequence[Mapping[str, Any]]) -> list[str]: ...

    async def retrieve_result(
        self,
        query: str,
        *,
        workspaces: Sequence[str],
        retrieval: RetrievalOptions,
        bm25_query: str | None = None,
        filters: MetadataFilter | None = None,
        query_images: Sequence[Mapping[str, Any]] = (),
        image_descriptions: Sequence[str] = (),
        model_profile: ModelProfile | None = None,
    ) -> RetrievalResult: ...


class RetrievalExecutor:
    """Execute accepted Retrieval input on the shared Query-lane runtime."""

    def __init__(
        self,
        *,
        operation: RetrievalOperation,
        timeout_seconds: float,
        model_fingerprint_for_role: Callable[[str], ModelFingerprint],
        now: Callable[[], datetime.datetime] | None = None,
        on_dependency_unavailable: DependencyStateCallback | None = None,
        on_dependency_recovered: DependencyStateCallback | None = None,
    ) -> None:
        self._operation = operation
        self._timeout_seconds = float(timeout_seconds)
        self._model_fingerprint_for_role = model_fingerprint_for_role
        self._now = now or (lambda: datetime.datetime.now(datetime.UTC))
        self._on_dependency_unavailable = on_dependency_unavailable
        self._on_dependency_recovered = on_dependency_recovered

    def validate_active_prepared_input(self, prepared: Mapping[str, Any]) -> None:
        """Validate active durable Retrieval input using this executor's model bindings."""
        validate_active_retrieval_input(
            prepared,
            model_fingerprint_for_role=self._model_fingerprint_for_role,
        )

    async def execute(self, session: RunSession) -> Succeeded | Failed | Deferred:
        run_input = RetrievalRunInput.from_prepared_input(session.prepared_input)
        self._require_compatible_models(run_input)
        if run_input.context_policy_revision != CONTEXT_POLICY_REVISION:
            raise RunExecutionError(
                "retrieval_context_policy_changed",
                "The context policy needed by this accepted retrieval changed before execution.",
            )
        if run_input.model_catalog_revision != current_model_catalog_revision():
            raise RunExecutionError(
                "retrieval_model_catalog_changed",
                "The model catalog needed by this accepted retrieval changed before execution.",
            )

        self._operation.warm(run_input.workspaces)
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await session.enter_phase("planning")
                images = tuple(dict(image) for image in run_input.query_images)
                descriptions = await self._operation.prepare_query_images(images) if images else []
                await session.check_cancelled()
                await session.enter_phase("searching")
                result = await self._operation.retrieve_result(
                    run_input.query,
                    workspaces=run_input.workspaces,
                    retrieval=run_input.retrieval,
                    bm25_query=run_input.bm25_query,
                    filters=(
                        MetadataFilter.model_validate(run_input.filters)
                        if run_input.filters is not None
                        else None
                    ),
                    query_images=images,
                    image_descriptions=descriptions,
                    model_profile=run_input.model_profile("extract"),
                )
        except TimeoutError as exc:
            raise RunExecutionError(
                _RETRIEVAL_TIMEOUT_KIND,
                _RETRIEVAL_TIMEOUT_MESSAGE,
            ) from exc
        except RunExecutionError, RunCancellationObserved, LeaseLostError:
            raise
        except Exception as exc:
            component = classify_transient_dependency(exc, component_hint="providers")
            if component is not None:
                await session.check_cancelled()
                checkpoint, delay = next_dependency_retry(
                    session.checkpoint,
                    component,
                    base_seconds=_DEFER_BASE_SECONDS,
                    max_seconds=_DEFER_MAX_SECONDS,
                )
                self._notify_dependency(self._on_dependency_unavailable, component)
                return Deferred(
                    checkpoint=checkpoint,
                    next_attempt_at=self._now() + datetime.timedelta(seconds=delay),
                )
            return Failed(
                error_kind=_RETRIEVAL_FAILURE_KIND,
                error_message=_RETRIEVAL_FAILURE_MESSAGE,
            )
        recovered = dependency_component_from_checkpoint(session.checkpoint)
        if recovered is not None:
            self._notify_dependency(self._on_dependency_recovered, recovered)
        return Succeeded(result=canonical_retrieval_result(result))

    @staticmethod
    def _notify_dependency(
        callback: DependencyStateCallback | None,
        component: DependencyComponent,
    ) -> None:
        if callback is not None:
            callback(component)

    def _require_compatible_models(self, run_input: RetrievalRunInput) -> None:
        for model in run_input.pinned_models:
            try:
                current = self._model_fingerprint_for_role(model.role)
            except Exception as exc:
                raise RunExecutionError(
                    _RETRIEVAL_MODEL_CHANGED_KIND,
                    _RETRIEVAL_MODEL_CHANGED_MESSAGE,
                ) from exc
            if current != model.fingerprint:
                raise RunExecutionError(
                    _RETRIEVAL_MODEL_CHANGED_KIND,
                    _RETRIEVAL_MODEL_CHANGED_MESSAGE,
                )


def canonical_retrieval_result(result: RetrievalResult) -> dict[str, Any]:
    """Return transport-neutral Retrieval output safe for durable storage."""
    return {
        "contexts": _canonical_contexts(result.contexts),
        "trace": _strip_projection_values(result.trace),
        "image_descriptions": [str(item) for item in result.image_descriptions],
    }


def restore_retrieval_result(value: Mapping[str, Any]) -> RetrievalResult:
    """Decode canonical durable output for a fresh reader projection."""
    contexts = value.get("contexts")
    return RetrievalResult(
        contexts=(
            {
                "chunks": [dict(item) for item in contexts.get("chunks") or ()],
                "entities": [dict(item) for item in contexts.get("entities") or ()],
                "relationships": [dict(item) for item in contexts.get("relationships") or ()],
            }
            if isinstance(contexts, Mapping)
            else {"chunks": [], "entities": [], "relationships": []}
        ),
        trace=dict(value.get("trace") or {}),
        image_descriptions=[str(item) for item in value.get("image_descriptions") or ()],
    )


def _canonical_contexts(contexts: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        key: [
            _canonical_context_row(item)
            for item in contexts.get(key, ())
            if isinstance(item, Mapping)
        ]
        for key in ("chunks", "entities", "relationships")
    }


def _canonical_context_row(item: Mapping[str, Any]) -> dict[str, Any]:
    had_visual = bool(item.get("image_data") or item.get("image_url") or item.get("thumbnail_url"))
    row = _strip_projection_values(item)
    if had_visual:
        row["_has_visual_asset"] = True
    return row


def _strip_projection_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_projection_values(item)
            for key, item in value.items()
            if key not in {"image_data", "image_url", "thumbnail_url", "download_url"}
        }
    if isinstance(value, list | tuple):
        return [_strip_projection_values(item) for item in value]
    return value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _reasoning_profile(value: Any) -> ReasoningProfile | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("pinned retrieval reasoning profile has an invalid shape")
    raw_levels = value.get("levels")
    if not isinstance(raw_levels, Mapping) or set(raw_levels) != set(REASONING_LEVELS):
        raise ValueError("pinned retrieval reasoning profile requires every explicit level")
    levels: dict[str, str | None] = {}
    for level in REASONING_LEVELS:
        raw = raw_levels[level]
        if raw is not None and not isinstance(raw, str):
            raise ValueError("pinned retrieval reasoning level must be a string or null")
        levels[level] = raw
    parsed = ReasoningLevels(**levels)  # type: ignore[arg-type]
    best_effort = value.get("best_effort", False)
    if type(best_effort) is not bool:
        raise ValueError("pinned retrieval reasoning best_effort must be a boolean")
    if best_effort:
        return ReasoningProfile.unverified(format=str(value.get("format") or ""), levels=parsed)
    return ReasoningProfile(format=str(value.get("format") or ""), levels=parsed)


__all__ = [
    "PinnedRetrievalModel",
    "RetrievalExecutor",
    "RetrievalOperation",
    "RetrievalRunInput",
    "canonical_retrieval_result",
    "restore_retrieval_result",
    "validate_active_retrieval_input",
]
