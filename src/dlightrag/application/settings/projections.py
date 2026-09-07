# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Map root Pydantic configuration into provider-neutral AI settings."""

from pathlib import Path

from dlightrag.application.access import (
    AccessRule,
    AccessSettings,
    AuthenticationSettings,
)
from dlightrag.application.answer_runs.capabilities import (
    AnswerCapabilitySettings,
    AnswerImagePolicySettings,
)
from dlightrag.application.config import DlightragConfig
from dlightrag.application.corpus_admin import CorpusAdminSettings
from dlightrag.application.retrieval import RetrievalSettings
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.catalog import resolve_model_profile
from dlightrag.engine.ai.fingerprints import model_fingerprint
from dlightrag.engine.ai.reasoning import resolve_reasoning
from dlightrag.engine.ai.settings import ModelRole, ModelSettings
from dlightrag.engine.answer.execution import (
    AnswerExecutorSettings,
    AnswerResourceSettings,
)
from dlightrag.engine.answer.highlights import SemanticHighlightSettings
from dlightrag.engine.answer.model_runtime import (
    AnswerModelRuntimeSettings,
    WebSourceRuntimeSettings,
)
from dlightrag.engine.answer.resources.images import MAX_QUERY_IMAGES
from dlightrag.engine.rag.workspace.settings import RagSettings
from dlightrag.engine.rag.workspace.workspaces import normalize_workspace


def access_settings(config: DlightragConfig) -> AccessSettings:
    """Snapshot root authorization configuration into immutable Access settings."""
    return AccessSettings(
        mode=config.access.control.mode,
        rules=tuple(
            AccessRule(
                claim=rule.claim,
                value=rule.value,
                workspaces=tuple(
                    workspace if workspace == "*" else normalize_workspace(workspace)
                    for workspace in rule.workspaces
                ),
                actions=tuple(rule.actions),
            )
            for rule in config.access.control.rules
        ),
    )


def authentication_settings(
    config: DlightragConfig,
    *,
    audience: str | None = None,
) -> AuthenticationSettings:
    """Snapshot root bearer configuration into immutable Access settings."""
    configured_audience = audience if audience is not None else config.access.jwt_audience
    if isinstance(configured_audience, list):
        resolved_audience: str | tuple[str, ...] | None = tuple(configured_audience)
    else:
        resolved_audience = configured_audience
    return AuthenticationSettings(
        mode=config.access.auth_mode,
        api_token=config.access.api_token,
        jwt_verification_key=config.access.jwt_verification_key,
        jwt_jwks_url=config.access.jwt_jwks_url,
        jwt_issuer=config.access.jwt_issuer,
        jwt_audience=resolved_audience,
        jwt_algorithm=config.access.jwt_algorithm,
    )


def answer_capability_settings(config: DlightragConfig) -> AnswerCapabilitySettings:
    """Snapshot root Answer capability and image policy configuration."""
    answer = config.answer.generation
    return AnswerCapabilitySettings(
        images=AnswerImagePolicySettings(
            max_images=int(answer.max_images),
            max_total_bytes=answer.image_max_total_bytes,
            max_bytes_per_image=answer.image_max_bytes,
            max_pixels=answer.image_max_pixels,
            max_px=answer.image_max_px,
            min_px=answer.image_min_px,
            quality=answer.image_quality,
            min_quality=answer.image_min_quality,
        ),
        web_search_enabled=bool(config.answer.web_sources.search_order()),
        rerank_enabled=config.models.rerank.enabled,
        rerank_strategy=config.models.rerank.strategy,
    )


def semantic_highlight_settings(config: DlightragConfig) -> SemanticHighlightSettings:
    """Snapshot root semantic highlight configuration into Answer settings."""
    highlights = config.answer.citations.highlights
    return SemanticHighlightSettings(
        enabled=highlights.enabled,
        timeout=highlights.timeout,
        max_concurrency=highlights.max_concurrency,
        batch_size=highlights.batch_size,
        max_input_chars=highlights.max_input_chars,
        cache_size=highlights.cache_size,
    )


def answer_model_runtime_settings(config: DlightragConfig) -> AnswerModelRuntimeSettings:
    """Snapshot Answer model roles and Web-search configuration."""
    web = config.answer.web_sources
    return AnswerModelRuntimeSettings(
        model_roles=config.models.chat,
        web_sources=WebSourceRuntimeSettings(
            exa_api_key=web.exa.api_key,
            tavily_api_key=web.tavily.api_key,
            search_providers=web.search_order(),
            extract_providers=web.extract_order(),
        ),
        query_image_limit=MAX_QUERY_IMAGES,
    )


def answer_resource_settings(config: DlightragConfig) -> AnswerResourceSettings:
    """Snapshot Answer attachment and current-image resource limits."""
    answer = config.answer.generation
    return AnswerResourceSettings(
        max_attachments=answer.max_attachments,
        max_attachment_bytes=answer.max_attachment_bytes,
        max_total_attachment_bytes=answer.max_total_attachment_bytes,
        image_max_bytes=answer.image_max_bytes,
        image_max_pixels=answer.image_max_pixels,
    )


def agent_skills_root(config: DlightragConfig) -> Path:
    """Resolve the global Agent Skills root; unset resolves to ~/.dlightrag/skills.

    Mirrors the Agent Workspace default convention. The directory is ensured
    eagerly by composition so operators can drop skills in without a restart;
    discovery itself tolerates an absent root.
    """
    configured = config.answer.agent.skills_root
    return Path(configured).expanduser() if configured else Path.home() / ".dlightrag" / "skills"


def owner_skills_root(config: DlightragConfig) -> Path:
    """Resolve the per-owner Agent Skills root; unset resolves to ~/.dlightrag/owner_skills."""
    configured = config.answer.agent.owner_skills_root
    return (
        Path(configured).expanduser() if configured else Path.home() / ".dlightrag" / "owner_skills"
    )


def answer_executor_settings(config: DlightragConfig) -> AnswerExecutorSettings:
    """Snapshot durable Answer execution and Artifact publication policy."""
    from dlightrag.engine.answer.publication import PublicationLimits

    publication = config.answer.agent.publication
    return AnswerExecutorSettings(
        default_top_k=config.corpus.retrieval.top_k,
        default_chunk_top_k=config.corpus.retrieval.chunk_top_k,
        semantic_highlights=semantic_highlight_settings(config),
        publication=PublicationLimits(**publication.model_dump()),
    )


def retrieval_settings(config: DlightragConfig) -> RetrievalSettings:
    """Snapshot caller-awaited Retrieval policy."""
    return RetrievalSettings(
        default_top_k=config.corpus.retrieval.top_k,
        default_chunk_top_k=config.corpus.retrieval.chunk_top_k,
        federation_min_chunks_per_workspace=(
            config.corpus.retrieval.federation_min_chunks_per_workspace
        ),
        timeout_seconds=config.corpus.retrieval.timeout,
        query_image_limit=MAX_QUERY_IMAGES,
    )


def corpus_admin_settings(config: DlightragConfig) -> CorpusAdminSettings:
    """Snapshot root corpus administration policy into immutable settings."""
    return CorpusAdminSettings(
        default_workspace_id=normalize_workspace(config.deployment.workspace),
        default_display_name=config.deployment.workspace,
        default_embedding_model=config.models.embedding.model,
        input_root=config.input_dir_path,
        read_only=config.is_reader,
    )


def model_settings_for_role(config: DlightragConfig, role: ModelRole) -> ModelSettings:
    """Resolve one complete role override, otherwise snapshot the default model."""
    return config.models.chat.resolve(role)


def model_profile_for_settings(
    config: DlightragConfig,
    settings: ModelSettings,
) -> ModelProfile:
    """Resolve endpoint facts and validate its configured semantic reasoning."""
    del config
    profile = resolve_model_profile(model_fingerprint(settings))
    resolve_reasoning(profile.reasoning, settings.reasoning)
    resolve_reasoning(profile.reasoning, settings.effective_agentic_reasoning)
    return profile


def model_profile_for_role(config: DlightragConfig, role: ModelRole) -> ModelProfile:
    """Resolve one role's model settings and independent capacity profile."""
    return model_profile_for_settings(config, model_settings_for_role(config, role))


def rerank_scoring_model_settings(config: DlightragConfig) -> ModelSettings:
    """Resolve the independent chat reranker or default chat model."""
    return config.models.rerank.scoring_model(config.models.chat.default)


def rag_settings(config: DlightragConfig) -> RagSettings:
    """Compose runtime RAG ownership from canonical settings by reference."""
    return RagSettings(
        models=config.models,
        corpus=config.corpus,
        input_root=config.input_dir_path,
        read_only=config.is_reader,
    )


__all__ = [
    "model_profile_for_role",
    "model_profile_for_settings",
    "model_settings_for_role",
    "rag_settings",
    "rerank_scoring_model_settings",
]
