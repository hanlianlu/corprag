# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral invariants retained by the canonical 3.0 configuration."""

from __future__ import annotations

import os
import ssl
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dlightrag.application.config import (
    AccessSectionSettings,
    AnswerConfig,
    AnswerSectionSettings,
    ApiInterfaceSettings,
    CitationHighlightConfig,
    DeploymentSettings,
    DlightragConfig,
    InterfacesSettings,
    LightRAGStorageSettings,
    McpInterfaceSettings,
    PostgresSettings,
    StorageSettings,
    WebSourceProviderConfig,
    WebSourcesConfig,
    load_config,
)
from dlightrag.engine.ai.settings import (
    EmbeddingSettings,
    ModelSettings,
    ModelsSettings,
    RerankSettings,
)
from dlightrag.engine.rag.workspace.settings import (
    BM25ProfileSettings,
    CorpusSettings,
    DoclingSidecarSettings,
    ExtractionSettings,
    MinerUSidecarSettings,
    ParserSidecarsSettings,
    RetrievalSettings,
    VisualAssetSettings,
    VLMSidecarSettings,
)


@pytest.fixture(autouse=True)
def _clean_config_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("DLIGHTRAG_") or key in {
            "LIGHTRAG_PARSER",
            "POSTGRES_SERVER_SETTINGS",
            "POSTGRES_WORKSPACE",
        }:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("api://dlightrag", "api://dlightrag"),
        (["api://dlightrag", "proxy-id"], ("api://dlightrag", "proxy-id")),
        ('["a", "b"]', ("a", "b")),
        ("   ", None),
    ],
)
def test_jwt_audience_normalization(raw: Any, expected: Any) -> None:
    assert AccessSectionSettings(jwt_audience=raw).jwt_audience == expected


def test_model_defaults_and_case_folding() -> None:
    settings = ModelSettings(provider=cast(Any, " GEMINI "), model="gemini-model")

    assert settings.provider == "gemini"
    assert settings.temperature is None
    assert settings.timeout == 240.0
    assert settings.max_retries == 3
    assert settings.structured_output == "auto"


@pytest.mark.parametrize(
    "values",
    [
        {"model": "x", "temperature": -0.1},
        {"model": "x", "timeout": 0},
        {"model": "x", "max_retries": -1},
        {"model": "x", "structured_output": "json_yaml"},
        {"provider": "anthropic", "model": "x", "structured_output": "json_object"},
        {"provider": "invalid", "model": "x"},
    ],
)
def test_invalid_model_settings_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelSettings(**values)  # type: ignore[arg-type]


def test_startup_catalogue_requires_complete_profile_facts() -> None:
    with pytest.raises(ValidationError, match="max_input_tokens"):
        ModelsSettings.model_validate(
            {
                "catalogue": [
                    {
                        "provider": "openai",
                        "model": "new-model",
                        "base_url": None,
                        "profile": {
                            "context_window_tokens": 100_000,
                            "max_output_tokens": 10_000,
                            "supports_images": False,
                            "reasoning": None,
                        },
                    }
                ]
            }
        )


@pytest.mark.parametrize("values", [{"dim": 0}, {"max_token_size": 0}, {"batch_size": 0}])
def test_invalid_embedding_bounds_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EmbeddingSettings(**values)


@pytest.mark.parametrize(
    "provider",
    [
        "azure_cohere",
        "cohere",
        "gemini",
        "jina",
        "openai",
        "openai_compatible",
        "voyage",
    ],
)
def test_embedding_wire_protocols_are_accepted(provider: str) -> None:
    assert EmbeddingSettings(provider=provider).provider == provider  # type: ignore[arg-type]


def test_retired_or_unknown_embedding_configuration_is_rejected() -> None:
    with pytest.raises(ValidationError, match="provider"):
        EmbeddingSettings(provider="ollama")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="asymmetric"):
        EmbeddingSettings(asymmetric="disable")  # type: ignore[call-arg]


def test_embedding_defaults_preserve_shipped_pipeline_contract() -> None:
    settings = EmbeddingSettings()
    assert settings.provider == "voyage"
    assert settings.model == "voyage-multimodal-3.5"
    assert settings.base_url is None  # the selected adapter owns its native default URL
    assert settings.dim == 1024
    assert settings.max_token_size == 8192
    assert settings.batch_size == 64
    assert settings.max_concurrency == 16
    assert settings.timeout == 120


@pytest.mark.parametrize(
    "values",
    [
        {"max_concurrency": 0},
        {"batch_size": 0},
        {"score_threshold": -0.1},
        {"temperature": -0.1},
    ],
)
def test_invalid_rerank_bounds_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RerankSettings(**values)


def test_rerank_defaults_and_provider_case_folding() -> None:
    settings = RerankSettings(provider=cast(Any, " OpenAI "))
    assert settings.enabled is True
    assert settings.strategy == "chat_llm_reranker"
    assert settings.provider == "openai"
    assert settings.max_concurrency == 8
    assert settings.batch_size == 8


def test_answer_and_citation_defaults_are_preserved() -> None:
    answer = AnswerConfig()
    highlights = CitationHighlightConfig()
    assert answer.max_attachments == 6
    assert answer.max_attachment_bytes == 100 * 1024 * 1024
    assert answer.max_total_attachment_bytes == 128 * 1024 * 1024
    assert answer.max_images == 12
    assert answer.image_max_pixels == 40_000_000
    assert highlights.enabled is True
    assert highlights.timeout == 10.0


@pytest.mark.parametrize(
    "values",
    [
        {"max_images": -1},
        {"image_max_total_bytes": 0},
        {"image_min_quality": 96},
        {"max_attachments": -1},
        {"max_attachment_bytes": 0},
    ],
)
def test_invalid_answer_bounds_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AnswerConfig(**values)


@pytest.mark.parametrize("values", [{"thumb_max_px": 0}, {"thumb_cache_size": 0}])
def test_invalid_visual_asset_bounds_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        VisualAssetSettings(**values)


def test_storage_defaults_and_only_explicit_milvus_vector_alternative() -> None:
    storage = LightRAGStorageSettings(vector_index_type="HNSW_HALFVEC")
    assert storage.vector_storage == "PGVectorStorage"
    assert storage.graph_storage == "PGTableGraphStorage"
    assert storage.kv_storage == "PGKVStorage"
    assert storage.doc_status_storage == "PGDocStatusStorage"
    assert (
        LightRAGStorageSettings(vector_storage="MilvusVectorDBStorage").vector_storage
        == "MilvusVectorDBStorage"
    )
    for values in (
        {"vector_storage": "QdrantStorage"},
        {"graph_storage": "PGGraphStorage"},
        {"graph_storage": "AGEStorage"},
        {"kv_storage": "RedisKVStorage"},
        {"doc_status_storage": "JsonDocStatusStorage"},
    ):
        with pytest.raises(ValidationError):
            LightRAGStorageSettings(**values)  # type: ignore[arg-type]


def test_vector_and_pool_defaults_export_lightrag_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DlightragConfig()
    config.apply_lightrag_backend_env(force=True)

    assert config.storage.lightrag.hnsw_ef_construction == 256
    assert config.storage.lightrag.hnsw_ef_search == 256
    assert config.domain_pool_server_settings()["hnsw.ef_search"] == "256"
    assert os.environ["POSTGRES_HNSW_EF"] == "256"
    assert os.environ["POSTGRES_VECTOR_INDEX_TYPE"] == "HNSW_HALFVEC"
    assert os.environ["POSTGRES_MAX_CONNECTIONS"] == "16"


def test_backend_env_disables_global_lightrag_workspace_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_WORKSPACE", "inherited-workspace")
    config = DlightragConfig(deployment=DeploymentSettings(workspace="resolved-workspace"))

    config.apply_lightrag_backend_env(force=True)

    assert os.environ["POSTGRES_WORKSPACE"] == ""


def test_milvus_environment_overrides_only_resolved_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MILVUS_URI", "inherited-uri")
    monkeypatch.setenv("MILVUS_TOKEN", "inherited-token")
    monkeypatch.setenv("MILVUS_DB_NAME", "inherited-db")
    config = DlightragConfig(
        storage=StorageSettings(
            lightrag=LightRAGStorageSettings(
                vector_storage="MilvusVectorDBStorage",
                milvus_uri="resolved-uri",
                milvus_db_name="resolved-db",
            )
        )
    )

    config.apply_lightrag_backend_env(force=True)

    assert os.environ["MILVUS_URI"] == "resolved-uri"
    assert os.environ["MILVUS_DB_NAME"] == "resolved-db"
    assert os.environ["MILVUS_TOKEN"] == "inherited-token"


def test_milvus_reader_and_pg_only_options_are_rejected_without_secret_echo() -> None:
    secret = "never-echo-this-milvus-token"
    storage = StorageSettings(
        lightrag=LightRAGStorageSettings(
            vector_storage="MilvusVectorDBStorage",
            milvus_uri="https://milvus.example",
            milvus_token=secret,
            milvus_db_name="default",
        )
    )
    with pytest.raises(ValidationError) as raised:
        DlightragConfig(
            deployment=DeploymentSettings(service_role="reader"),
            storage=storage,
        )
    assert secret not in str(raised.value)

    with pytest.raises(ValidationError, match="require"):
        LightRAGStorageSettings(milvus_token=secret)


def test_milvus_vector_kwargs_are_bounded_and_exclude_credentials() -> None:
    with pytest.raises(ValidationError, match="unsupported keys"):
        LightRAGStorageSettings(
            vector_storage="MilvusVectorDBStorage",
            vector_db_kwargs={"unknown_index_knob": 1},
        )
    with pytest.raises(ValidationError, match="must not contain credentials"):
        LightRAGStorageSettings(vector_db_kwargs={"token": "secret"})
    with pytest.raises(ValidationError, match="at most 16"):
        LightRAGStorageSettings(vector_db_kwargs={f"key_{index}": index for index in range(17)})


def test_milvus_rejects_postgres_only_promotion_configuration() -> None:
    with pytest.raises(ValidationError, match="promotion thresholds require PGVectorStorage"):
        DlightragConfig(
            storage=StorageSettings(
                lightrag=LightRAGStorageSettings(
                    vector_storage="MilvusVectorDBStorage",
                    milvus_uri="https://milvus.example",
                    milvus_db_name="default",
                )
            ),
            corpus=CorpusSettings(
                promotion={"chunk_threshold": 100},  # type: ignore[arg-type]
            ),
        )


def test_postgres_ssl_modes_project_to_asyncpg() -> None:
    required = DlightragConfig(
        storage=StorageSettings(postgres=PostgresSettings(ssl_mode="require"))
    )
    disabled = DlightragConfig(
        storage=StorageSettings(postgres=PostgresSettings(ssl_mode="disable"))
    )
    verified = DlightragConfig(
        storage=StorageSettings(postgres=PostgresSettings(ssl_mode="verify-full"))
    )
    assert required.pg_connection_kwargs()["ssl"] is True
    assert disabled.pg_connection_kwargs()["ssl"] is False
    context = verified.pg_connection_kwargs()["ssl"]
    assert isinstance(context, ssl.SSLContext)
    assert context.check_hostname is True


def test_postgres_session_settings_merge_hnsw_and_reader_policy() -> None:
    config = DlightragConfig(
        deployment=DeploymentSettings(service_role="reader"),
        storage=StorageSettings(
            postgres=PostgresSettings(
                session_settings={"application_name": "test", "hnsw.ef_search": 999}
            ),
            lightrag=LightRAGStorageSettings(hnsw_ef_search=256),
        ),
    )
    assert config.domain_pool_server_settings() == {
        "hnsw.ef_search": "999",
        "application_name": "test",
    }
    assert config.lightrag_pool_server_settings()["default_transaction_read_only"] == "on"


def test_bm25_defaults_cover_languages_and_one_fallback() -> None:
    profiles = RetrievalSettings().bm25_profiles
    assert {profile.languages[0] for profile in profiles if profile.languages} >= {"zh", "en"}
    assert sum(profile.fallback for profile in profiles) == 1


@pytest.mark.parametrize(
    "profile",
    [
        {"name": "bad-name", "text_config": "english", "languages": ("en",)},
        {"name": "x", "text_config": "unsafe;drop", "languages": ("en",)},
        {"name": "x", "text_config": "english", "languages": ("en", "de")},
        {"name": "x", "text_config": "simple", "languages": ("en",), "fallback": True},
    ],
)
def test_invalid_bm25_profiles_are_rejected(profile: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BM25ProfileSettings(**profile)


def test_parser_defaults_to_self_hosted_mineru() -> None:
    defaults = ParserSidecarsSettings()
    assert defaults.active_parser == "mineru"
    assert defaults.mineru == MinerUSidecarSettings(
        api_mode="local",
        local_endpoint="http://127.0.0.1:8210",
        language="ch",
        backend="hybrid-engine",
    )
    assert defaults.docling is None

    mineru_only = ParserSidecarsSettings(mineru=MinerUSidecarSettings())
    assert mineru_only.active_parser == "mineru"
    assert mineru_only.docling is None

    both = ParserSidecarsSettings(
        mineru=MinerUSidecarSettings(),
        docling=DoclingSidecarSettings(),
    )
    assert both.active_parser == "mineru"


def test_docling_only_selection_and_sidecar_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "stale")
    config = DlightragConfig(
        corpus=CorpusSettings(
            sidecars=ParserSidecarsSettings(
                docling=DoclingSidecarSettings(endpoint="http://docling:5001")
            )
        )
    )
    config.apply_lightrag_sidecar_env()
    assert config.parser_rules == "*:docling-iteP"
    assert os.environ["DOCLING_ENDPOINT"] == "http://docling:5001"
    # Unset optional DlightRAG bindings leave upstream environment behavior untouched.
    assert os.environ["MINERU_LOCAL_ENDPOINT"] == "stale"


def test_mineru_backend_and_vlm_environment_are_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    config = DlightragConfig(
        corpus=CorpusSettings(
            sidecars=ParserSidecarsSettings(
                vlm=VLMSidecarSettings(min_image_pixel=80),
                mineru=MinerUSidecarSettings(backend="hybrid-engine"),
            )
        )
    )
    config.apply_lightrag_sidecar_env()
    assert os.environ["MINERU_LOCAL_BACKEND"] == "hybrid-engine"
    assert os.environ["VLM_MIN_IMAGE_PIXEL"] == "80"
    assert "LIGHTRAG_PARSER" not in os.environ
    config.apply_lightrag_runtime_env()
    assert os.environ["LIGHTRAG_PARSER"] == "*:mineru-iteP"


def test_entity_type_prompt_file_is_one_yaml_filename() -> None:
    assert ExtractionSettings(entity_type_prompt_file="finance.yaml").entity_type_prompt_file == (
        "finance.yaml"
    )
    for value in ("../finance.yaml", "/tmp/finance.yaml", "finance.txt"):
        with pytest.raises(ValidationError):
            ExtractionSettings(entity_type_prompt_file=value)


def test_public_listener_without_auth_is_refused_and_override_is_explicit() -> None:
    public = InterfacesSettings(api=ApiInterfaceSettings(host="0.0.0.0"))
    with pytest.raises(ValidationError, match="non-loopback"):
        DlightragConfig(interfaces=public)
    config = DlightragConfig(
        interfaces=public,
        access=AccessSectionSettings(allow_insecure_no_auth=True),
    )
    assert config.interfaces.api.host == "0.0.0.0"


def test_jwt_jwks_and_mcp_oauth_validation() -> None:
    access = AccessSectionSettings(
        auth_mode="jwt",
        jwt_jwks_url="https://issuer.example/jwks.json",
        jwt_issuer="https://issuer.example",
        jwt_audience="api://dlightrag",
        jwt_algorithm="RS256",
    )
    interfaces = InterfacesSettings(
        mcp=McpInterfaceSettings(
            transport="streamable-http",
            resource_server_url="https://rag.example/mcp",
        )
    )
    assert DlightragConfig(access=access, interfaces=interfaces).access.jwt_algorithm == "RS256"
    with pytest.raises(ValidationError, match="requires jwt_issuer and jwt_audience"):
        DlightragConfig(
            access=AccessSectionSettings(auth_mode="jwt", jwt_jwks_url="https://x/jwks")
        )


def test_explicit_env_file_and_error_redaction(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=super-secret-value\n",
        encoding="utf-8",
    )
    config = load_config(env_file)
    assert config.models.chat.default.api_key == "super-secret-value"
    assert "super-secret-value" not in repr(config)

    bad = tmp_path / "bad.env"
    bad.write_text(
        "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=never-echo-me\n"
        "DLIGHTRAG_MODELS__CHAT__DEFAULT__TIMEOUT=0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as caught:
        load_config(bad)
    assert "never-echo-me" not in str(caught.value)


def test_incomplete_secret_only_role_error_never_echoes_key(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY=never-echo-role-secret\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models.chat.roles.extract.model") as caught:
        load_config(env_file)
    assert "never-echo-role-secret" not in str(caught.value)


def test_legacy_dotenv_key_is_rejected(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("DLIGHTRAG_POSTGRES_HOST=legacy-db\n", encoding="utf-8")

    with pytest.raises(ValueError, match="postgres_host: Extra inputs"):
        load_config(env_file)


def test_web_source_orders_derive_from_available_credentials() -> None:
    config = WebSourcesConfig(
        exa=WebSourceProviderConfig(api_key="exa-key"),
        tavily=WebSourceProviderConfig(api_key="tavily-key"),
    )

    assert config.search_order() == ("exa", "tavily")
    assert config.extract_order() == ("exa", "tavily")


def test_web_source_orders_are_independent_and_empty_disables_operation() -> None:
    config = WebSourcesConfig(
        exa=WebSourceProviderConfig(api_key="exa-key"),
        tavily=WebSourceProviderConfig(api_key="tavily-key"),
        search_providers=("tavily", "exa"),
        extract_providers=(),
    )

    assert config.search_order() == ("tavily", "exa")
    assert config.extract_order() == ()


def test_web_source_order_rejects_provider_without_credential() -> None:
    with pytest.raises(ValidationError, match="lack api_key"):
        WebSourcesConfig(search_providers=("tavily",))


def test_web_source_order_rejects_duplicate_provider() -> None:
    with pytest.raises(ValidationError, match="contain duplicates"):
        WebSourcesConfig(
            exa=WebSourceProviderConfig(api_key="exa-key"),
            search_providers=("exa", "exa"),
        )


def test_config_composes_canonical_models_without_snapshot_copy() -> None:
    from dlightrag.application.settings import rag_settings

    models = ModelsSettings(embedding=EmbeddingSettings(startup_probe=False))
    answer = AnswerSectionSettings()
    config = DlightragConfig(models=models, answer=answer)
    runtime = rag_settings(config)
    assert config.models is models
    assert config.answer is answer
    assert runtime.models is config.models
    assert runtime.corpus is config.corpus
